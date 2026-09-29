package library

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"time"

	"goodkind.io/lm-semantic-search/internal/clock"
)

// rankConfig is the ranking formula that one result snapshot fixes. A cursor
// fails with [ErrCursorMismatch] when the open library ranks with another
// configuration.
type rankConfig struct {
	Mode                   SearchMode `json:"mode"`
	RRFK                   int        `json:"rrf_k"`
	BM25K1                 float64    `json:"bm25_k1"`
	BM25B                  float64    `json:"bm25_b"`
	Analyzer               string     `json:"analyzer"`
	EmbeddingModel         string     `json:"embedding_model"`
	EmbeddingRevision      string     `json:"embedding_revision"`
	Dimension              int        `json:"dimension"`
	Normalization          string     `json:"normalization"`
	QueryInstructionPrefix string     `json:"query_instruction_prefix"`
}

// requestIdentity is the part of a request that one result snapshot fixes.
// PageSize and Cursor can change from page to page.
type requestIdentity struct {
	Namespace     string  `json:"namespace"`
	Query         string  `json:"query"`
	Filter        *Filter `json:"filter"`
	GroupBy       string  `json:"group_by"`
	PerGroupLimit int     `json:"per_group_limit"`
	MinScore      float64 `json:"min_score"`
}

// searchPlan is one validated request with its hashes.
type searchPlan struct {
	spec        NamespaceSpec
	request     SearchRequest
	requestHash string
	rank        rankConfig
	rankHash    string
}

// Search returns one page of the complete ranking of every occurrence in the
// namespace that matches the filter. Page one ranks every eligible occurrence
// from one catalog read transaction, scores every eligible vector exactly, and
// persists the full ordered result when more pages follow. A cursor page reads
// that persisted result. A later write does not change the page sequence of a
// cursor. Search acquires no kernel writer lock; page one writes its snapshot
// in one SQLite write transaction.
//
// A request that fails [Config.ValidateSearchRequest] returns an error that
// wraps [ErrInvalidRequest] before any embedding or scoring. An expired or
// missing snapshot returns [ErrCursorExpired], and a cursor for another
// request or rank configuration returns [ErrCursorMismatch]. A search that
// cannot finish before QueryTimeout returns [ErrDeadline], and a search over
// MaxTemporaryBytes or MaxSnapshotBytes returns [ErrResourceLimit]. A missing
// or corrupt vector returns [ErrVectorMissing] or [ErrVectorCorrupt]. Every
// failure returns no page.
func (library *Library) Search(ctx context.Context, request SearchRequest) (SearchPage, error) {
	ctx, cancel := context.WithTimeout(ctx, library.config.QueryTimeout)
	defer cancel()
	started := clock.Now()
	page, err := library.search(ctx, request)
	if err != nil {
		return SearchPage{}, classifySearchError(ctx, request.Namespace, err)
	}
	slog.DebugContext(ctx, "library search finished",
		"namespace", request.Namespace,
		"hits", len(page.Hits),
		"has_more", page.HasMore,
		"elapsed", clock.Now().Sub(started),
	)
	return page, nil
}

// classifySearchError logs a failed search and wraps a failure caused by the
// search deadline with [ErrDeadline]. An error that already wraps a library
// sentinel keeps that classification alone.
func classifySearchError(ctx context.Context, namespace string, err error) error {
	if !wrapsLibraryError(err) && (errors.Is(err, context.DeadlineExceeded) || errors.Is(ctx.Err(), context.DeadlineExceeded)) {
		err = fmt.Errorf("%w: search did not finish before its deadline: %w", ErrDeadline, err)
	}
	slog.WarnContext(ctx, "library search failed", "namespace", namespace, "err", err)
	return err
}

// libraryErrors are the sentinels that a search error keeps without an added
// ErrDeadline.
var libraryErrors = []error{
	ErrStoreMismatch, ErrInvalidRequest, ErrVectorCorrupt, ErrVectorMissing,
	ErrCursorExpired, ErrCursorMismatch, ErrDeadline, ErrResourceLimit,
}

// wrapsLibraryError reports whether err already wraps a library sentinel.
func wrapsLibraryError(err error) bool {
	for _, sentinel := range libraryErrors {
		if errors.Is(err, sentinel) {
			return true
		}
	}
	return false
}

func (library *Library) search(ctx context.Context, request SearchRequest) (SearchPage, error) {
	started := clock.Now()
	plan, err := library.planSearch(ctx, request)
	if err != nil {
		return SearchPage{}, err
	}
	if request.Cursor != "" {
		return library.readCursorPage(ctx, plan)
	}
	phases := &searchPhases{plan: clock.Now().Sub(started)}
	page, err := library.searchFirstPage(ctx, plan, phases)
	if err != nil {
		return SearchPage{}, err
	}
	phases.log(ctx, request.Namespace)
	return page, nil
}

// searchPhases are the wall-clock durations of one page-one search. verify
// and score sum the VerifyStrong and ScoreExact durations of every block,
// which run concurrently inside dense. writeWait is the time until the
// snapshot write transaction starts, which includes waiting for another
// SQLite writer; write is the rest of that transaction.
type searchPhases struct {
	plan      time.Duration
	read      time.Duration
	embed     time.Duration
	dense     time.Duration
	verify    time.Duration
	score     time.Duration
	lexical   time.Duration
	rank      time.Duration
	writeWait time.Duration
	write     time.Duration
	hits      time.Duration
}

// log writes the phase durations in milliseconds at debug level.
func (phases *searchPhases) log(ctx context.Context, namespace string) {
	milliseconds := func(duration time.Duration) float64 {
		return float64(duration.Microseconds()) / 1000
	}
	slog.DebugContext(ctx, "library search phases",
		"namespace", namespace,
		"plan_ms", milliseconds(phases.plan),
		"read_ms", milliseconds(phases.read),
		"embed_ms", milliseconds(phases.embed),
		"dense_ms", milliseconds(phases.dense),
		"verify_ms", milliseconds(phases.verify),
		"score_ms", milliseconds(phases.score),
		"lexical_ms", milliseconds(phases.lexical),
		"rank_ms", milliseconds(phases.rank),
		"write_wait_ms", milliseconds(phases.writeWait),
		"write_ms", milliseconds(phases.write),
		"hits_ms", milliseconds(phases.hits),
	)
}

// planSearch loads the registered namespace declaration, validates the
// request against it, and computes the request and rank configuration hashes.
func (library *Library) planSearch(ctx context.Context, request SearchRequest) (searchPlan, error) {
	var spec NamespaceSpec
	if err := library.read(ctx, func(tx *sql.Tx) error {
		var loadErr error
		spec, loadErr = loadNamespace(ctx, tx, request.Namespace)
		return loadErr
	}); err != nil {
		return searchPlan{}, err
	}
	if err := library.config.ValidateSearchRequest(spec, request); err != nil {
		return searchPlan{}, err
	}
	requestHash, err := hashJSON(requestIdentity{
		Namespace:     request.Namespace,
		Query:         request.Query,
		Filter:        request.Filter,
		GroupBy:       request.GroupBy,
		PerGroupLimit: request.PerGroupLimit,
		MinScore:      request.MinScore,
	})
	if err != nil {
		return searchPlan{}, err
	}
	descriptor := library.config.Store
	rank := rankConfig{
		Mode:                   library.config.SearchMode,
		RRFK:                   library.config.RRFK,
		BM25K1:                 library.config.BM25K1,
		BM25B:                  *library.config.BM25B,
		Analyzer:               library.config.AnalyzerIdentity,
		EmbeddingModel:         descriptor.EmbeddingModel,
		EmbeddingRevision:      descriptor.EmbeddingRevision,
		Dimension:              descriptor.Dimension,
		Normalization:          descriptor.Normalization,
		QueryInstructionPrefix: library.config.QueryInstructionPrefix,
	}
	rankHash, err := hashJSON(rank)
	if err != nil {
		return searchPlan{}, err
	}
	return searchPlan{spec: spec, request: request, requestHash: requestHash, rank: rank, rankHash: rankHash}, nil
}

// hashJSON returns the hex SHA-256 of the JSON encoding of a request identity
// or a rank configuration.
func hashJSON[Identity requestIdentity | rankConfig](value Identity) (string, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		slog.Error("encode search identity failed", "err", err)
		return "", fmt.Errorf("encode search identity: %w", err)
	}
	return hexSHA256(encoded), nil
}

// searchFirstPage copies the eligible occurrences under one read transaction,
// scores and ranks them in a query database, and returns the first page.
func (library *Library) searchFirstPage(ctx context.Context, plan searchPlan, phases *searchPhases) (_ SearchPage, err error) {
	started := clock.Now()
	query, err := openQueryDatabase(ctx, library.config.Store.CatalogPath, library.config.MaxTemporaryBytes, 2*library.config.QueryTimeout)
	if err != nil {
		return SearchPage{}, err
	}
	defer func() {
		err = errors.Join(err, query.close())
	}()
	var revisions snapshotRevisions
	var leg lexicalLeg
	if err := library.read(ctx, func(tx *sql.Tx) error {
		var copyErr error
		revisions, leg, copyErr = library.copySnapshot(ctx, tx, query, plan)
		return copyErr
	}); err != nil {
		return SearchPage{}, err
	}
	started = phases.mark(&phases.read, started)
	queryVector, err := library.embedQuery(ctx, plan.request.Query)
	if err != nil {
		return SearchPage{}, err
	}
	started = phases.mark(&phases.embed, started)
	if err := library.scoreDense(ctx, query, queryVector, phases); err != nil {
		return SearchPage{}, err
	}
	started = phases.mark(&phases.dense, started)
	if err := scoreLexical(ctx, query, leg); err != nil {
		return SearchPage{}, err
	}
	started = phases.mark(&phases.lexical, started)
	total, err := rankCandidates(ctx, query, plan)
	if err != nil {
		return SearchPage{}, err
	}
	phases.mark(&phases.rank, started)
	return library.firstPage(ctx, query, plan, revisions, total, phases)
}

// mark stores the time since started in phase and returns the current time.
func (phases *searchPhases) mark(phase *time.Duration, started time.Time) time.Time {
	now := clock.Now()
	*phase = now.Sub(started)
	return now
}

// copySnapshot reads the catalog revisions, the lexical corpus in Hybrid
// mode, and every eligible occurrence inside one read transaction.
func (library *Library) copySnapshot(
	ctx context.Context,
	tx *sql.Tx,
	query *queryDatabase,
	plan searchPlan,
) (snapshotRevisions, lexicalLeg, error) {
	revisions, err := readRevisions(ctx, tx)
	if err != nil {
		return snapshotRevisions{}, lexicalLeg{}, err
	}
	var leg lexicalLeg
	if plan.rank.Mode == Hybrid {
		leg, err = library.copyLexical(ctx, tx, query, plan.request.Namespace, plan.request.Query)
		if err != nil {
			return snapshotRevisions{}, lexicalLeg{}, err
		}
		revisions.Statistics = leg.generation
	}
	if _, err := copyCandidates(ctx, tx, query, plan); err != nil {
		return snapshotRevisions{}, lexicalLeg{}, err
	}
	return revisions, leg, nil
}

// firstPage returns page one from the query database ranking. When more rows
// follow, it persists the full ranking and returns a cursor for the next
// ordinal.
func (library *Library) firstPage(
	ctx context.Context,
	query *queryDatabase,
	plan searchPlan,
	revisions snapshotRevisions,
	total int64,
	phases *searchPhases,
) (SearchPage, error) {
	pageSize := plan.request.PageSize
	rows, err := readRankedRows(ctx, query, 0, pageSize)
	if err != nil {
		return SearchPage{}, err
	}
	hasMore := total > int64(pageSize)
	nextCursor := ""
	if hasMore {
		snapshotID, err := library.persistSnapshot(ctx, query, plan, revisions, phases)
		if err != nil {
			return SearchPage{}, err
		}
		nextCursor, err = encodeCursor(searchCursor{
			Version:     cursorVersion,
			SnapshotID:  snapshotID,
			NextOrdinal: int64(pageSize),
			RequestHash: plan.requestHash,
			RankHash:    plan.rankHash,
			Revisions:   revisions,
		})
		if err != nil {
			return SearchPage{}, err
		}
	}
	started := clock.Now()
	var hits []SearchHit
	if err := library.read(ctx, func(tx *sql.Tx) error {
		var buildErr error
		hits, buildErr = buildHits(ctx, tx, plan.request.Namespace, rows)
		return buildErr
	}); err != nil {
		return SearchPage{}, err
	}
	phases.mark(&phases.hits, started)
	return SearchPage{Hits: hits, HasMore: hasMore, NextCursor: nextCursor}, nil
}

// readCursorPage reads the next page of a persisted snapshot.
func (library *Library) readCursorPage(ctx context.Context, plan searchPlan) (SearchPage, error) {
	cursor, err := decodeCursor(plan.request.Cursor)
	if err != nil {
		return SearchPage{}, err
	}
	if cursor.RequestHash != plan.requestHash || cursor.RankHash != plan.rankHash {
		mismatch := fmt.Errorf("%w: the cursor was issued for another request or rank configuration", ErrCursorMismatch)
		slog.WarnContext(ctx, "search cursor rejected", "err", mismatch)
		return SearchPage{}, mismatch
	}
	pageSize := plan.request.PageSize
	var page SearchPage
	err = library.read(ctx, func(tx *sql.Tx) error {
		if err := loadSnapshotForCursor(ctx, tx, cursor, plan, clock.Now().UnixMilli()); err != nil {
			return err
		}
		rows, err := readSnapshotRows(ctx, tx, cursor.SnapshotID, cursor.NextOrdinal, min(pageSize, math.MaxInt-1)+1)
		if err != nil {
			return err
		}
		page.HasMore = len(rows) > pageSize
		if page.HasMore {
			rows = rows[:pageSize]
			next := cursor
			next.NextOrdinal = cursor.NextOrdinal + int64(pageSize)
			page.NextCursor, err = encodeCursor(next)
			if err != nil {
				return err
			}
		}
		page.Hits, err = buildHits(ctx, tx, plan.request.Namespace, rows)
		return err
	})
	if err != nil {
		return SearchPage{}, err
	}
	return page, nil
}

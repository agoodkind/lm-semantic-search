package library

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"time"

	"goodkind.io/lm-semantic-search/library/observation"

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
// cursor. Search acquires no kernel writer lock. Each successful cursor page
// renews SnapshotTTL after building its hits. Its SQLite write transaction
// serializes expiration validation and renewal with snapshot cleanup.
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
	ctx = observation.WithPurpose(ctx, observation.Query)
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
	phases.log(ctx, request.Namespace, err)
	if err != nil {
		return SearchPage{}, err
	}
	return page, nil
}

// searchPhases are the wall-clock durations of one page-one search. init,
// copyLexical, and copyCandidates are parts of read; scoreSave is part of dense. verify
// and score sum separate VerifyStrong and ScoreExact calls; combined sums
// ScoreExactVerified calls. These durations include joined failed blocks,
// which run concurrently inside dense. writeWait is the time until the
// snapshot write transaction starts, which includes waiting for another
// SQLite writer; write is the rest of that transaction. verified and scored
// count identities from successful block calls, including a joined wave with
// another failed block. Earlier verification at the same revision is excluded.
type searchPhases struct {
	stage              string
	filterNodes        [5]filterNodeTiming
	filterNodeCount    int
	filterEval         time.Duration
	selectedCopy       time.Duration
	candidateInsert    time.Duration
	candidateCommit    time.Duration
	acceptedCandidates int64
	queryBytes         int64
	cacheSize          int64
	pageSize           int64
	mmapSize           int64
	maxPageCount       int64
	init               time.Duration
	copyLexical        time.Duration
	copyCandidates     time.Duration
	scoreSave          time.Duration
	plan               time.Duration
	read               time.Duration
	embed              time.Duration
	dense              time.Duration
	combined           time.Duration
	verify             time.Duration
	score              time.Duration
	lexical            time.Duration
	rank               time.Duration
	writeWait          time.Duration
	write              time.Duration
	hits               time.Duration
	verified           int
	scored             int
}

// acceptedCandidates counts rows accepted by both insertion buffers, including
// rows pending a flush. selectedCopy minus candidateInsert includes catalog row
// reads, scalar JSON evaluation, and validation; it is not SQL execution alone.
// log uses warning level for failures and debug level for completed searches.
func (phases *searchPhases) log(ctx context.Context, namespace string, err error) {
	milliseconds := func(duration time.Duration) float64 {
		return float64(duration.Microseconds()) / 1000
	}
	level, outcome, failedStage := slog.LevelDebug, "success", ""
	if err != nil {
		level, outcome, failedStage = slog.LevelWarn, "failure", phases.stage
	}
	for _, node := range phases.filterNodes[:min(phases.filterNodeCount, len(phases.filterNodes))] {
		slog.Log(ctx, level, "library filter materialization",
			"namespace", namespace, "node", node.node, "operation", node.operation, "column", node.column,
			"scanned_rows", node.scanned, "accepted_rows", node.accepted,
			"wall_ms", milliseconds(node.elapsed), "complete", node.complete)
	}
	slog.Log(ctx, level, "library search phases",
		"namespace", namespace,
		"outcome", outcome,
		"failed_stage", failedStage,
		"init_ms", milliseconds(phases.init),
		"copy_lexical_ms", milliseconds(phases.copyLexical),
		"copy_candidates_ms", milliseconds(phases.copyCandidates),
		"filter_eval_ms", milliseconds(phases.filterEval),
		"filter_materializations", phases.filterNodeCount,
		"omitted_filter_materializations", max(phases.filterNodeCount-len(phases.filterNodes), 0),
		"selected_copy_ms", milliseconds(phases.selectedCopy),
		"candidate_insert_ms", milliseconds(phases.candidateInsert),
		"selected_read_validation_ms", milliseconds(phases.selectedCopy-phases.candidateInsert),
		"candidate_commit_ms", milliseconds(phases.candidateCommit),
		"accepted_candidate_rows", phases.acceptedCandidates,
		"query_file_bytes_before_close", phases.queryBytes,
		"query_cache_size", phases.cacheSize,
		"query_page_size", phases.pageSize,
		"query_mmap_size", phases.mmapSize,
		"query_max_page_count", phases.maxPageCount,
		"score_save_ms", milliseconds(phases.scoreSave),
		"native_duration_kind", "sum_concurrent_blocks",
		"phase_duration_kind", "wall_elapsed",
		"plan_ms", milliseconds(phases.plan),
		"read_ms", milliseconds(phases.read),
		"embed_ms", milliseconds(phases.embed),
		"dense_ms", milliseconds(phases.dense),
		"combined_verify_score_ms", milliseconds(phases.combined),
		"verify_ms", milliseconds(phases.verify),
		"score_ms", milliseconds(phases.score),
		"lexical_ms", milliseconds(phases.lexical),
		"rank_ms", milliseconds(phases.rank),
		"write_wait_ms", milliseconds(phases.writeWait),
		"write_ms", milliseconds(phases.write),
		"hits_ms", milliseconds(phases.hits),
		"verified_vectors", phases.verified,
		"scored_vectors", phases.scored,
		"complete", err == nil,
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
	phases.stage = "initialize"
	query, err := openQueryDatabase(ctx, library.config.Store.CatalogPath, library.config.MaxTemporaryBytes, 2*library.config.QueryTimeout)
	phases.mark(&phases.init, started)
	if err != nil {
		phases.mark(&phases.read, started)
		return SearchPage{}, err
	}
	defer func() {
		stat, statErr := os.Stat(query.path)
		if statErr == nil {
			phases.queryBytes = stat.Size()
		}
		if err == nil && statErr != nil {
			phases.stage = "query_file_stat"
		}
		if statErr != nil {
			err = errors.Join(err, fmt.Errorf("stat query database before close: %w", statErr))
		}
		closeErr := query.close()
		if err == nil && closeErr != nil {
			phases.stage = "query_close"
		}
		err = errors.Join(err, closeErr)
	}()
	phases.stage = "query_settings"
	if err := phases.readQuerySettings(ctx, query); err != nil {
		phases.mark(&phases.read, started)
		return SearchPage{}, err
	}
	var revisions snapshotRevisions
	var leg lexicalLeg
	phases.stage = "catalog_read"
	readErr := library.read(ctx, func(tx *sql.Tx) error {
		var copyErr error
		revisions, leg, copyErr = library.copySnapshot(ctx, tx, query, plan, phases)
		if copyErr == nil {
			phases.stage = "catalog_read"
		}
		return copyErr
	})
	started = phases.mark(&phases.read, started)
	if readErr != nil {
		return SearchPage{}, readErr
	}
	phases.stage = "embed"
	queryVector, err := library.embedQuery(ctx, plan.request.Query)
	started = phases.mark(&phases.embed, started)
	if err != nil {
		return SearchPage{}, err
	}
	phases.stage = "dense"
	denseErr := library.scoreDense(ctx, query, queryVector, revisions.Visibility, phases)
	started = phases.mark(&phases.dense, started)
	if denseErr != nil {
		return SearchPage{}, denseErr
	}
	phases.stage = "lexical"
	lexicalErr := scoreLexical(ctx, query, leg)
	started = phases.mark(&phases.lexical, started)
	if lexicalErr != nil {
		return SearchPage{}, lexicalErr
	}
	phases.stage = "rank"
	total, err := rankCandidates(ctx, query, plan)
	phases.mark(&phases.rank, started)
	if err != nil {
		return SearchPage{}, err
	}
	phases.stage = "first_page"
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
	phases *searchPhases,
) (snapshotRevisions, lexicalLeg, error) {
	revisions, err := readRevisions(ctx, tx)
	if err != nil {
		return snapshotRevisions{}, lexicalLeg{}, err
	}
	var leg lexicalLeg
	if plan.rank.Mode == Hybrid {
		phases.stage = "copy_lexical"
		started := clock.Now()
		leg, err = library.copyLexical(ctx, tx, query, plan.request.Namespace, plan.request.Query)
		phases.mark(&phases.copyLexical, started)
		if err != nil {
			return snapshotRevisions{}, lexicalLeg{}, err
		}
		revisions.Statistics = leg.generation
	}
	phases.stage = "copy_candidates"
	started := clock.Now()
	_, copyErr := copyCandidates(ctx, tx, query, plan, phases)
	phases.mark(&phases.copyCandidates, started)
	if copyErr != nil {
		return snapshotRevisions{}, lexicalLeg{}, copyErr
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
	phases.stage = "first_page_rows"
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
		phases.stage = "encode_cursor"
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
	phases.stage = "hits"
	var hits []SearchHit
	readErr := library.read(ctx, func(tx *sql.Tx) error {
		var buildErr error
		hits, buildErr = buildHits(ctx, tx, plan.request.Namespace, rows)
		return buildErr
	})
	phases.mark(&phases.hits, started)
	if readErr != nil {
		return SearchPage{}, readErr
	}
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
	err = library.write(ctx, func(tx *sql.Tx) error {
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
		if err != nil {
			return err
		}
		expiresAt := clock.Now().Add(library.config.SnapshotTTL).UnixMilli()
		if _, err := tx.ExecContext(ctx, renewSnapshotStatement, expiresAt, cursor.SnapshotID); err != nil {
			slog.ErrorContext(ctx, "renew search snapshot failed", "snapshot", cursor.SnapshotID, "err", err)
			return fmt.Errorf("renew search snapshot %s: %w", cursor.SnapshotID, err)
		}
		return nil
	})
	if err != nil {
		return SearchPage{}, err
	}
	return page, nil
}

func (phases *searchPhases) readQuerySettings(ctx context.Context, query *queryDatabase) error {
	for _, setting := range []struct {
		statement string
		value     *int64
	}{
		{statement: "PRAGMA cache_size", value: &phases.cacheSize},
		{statement: "PRAGMA page_size", value: &phases.pageSize},
		{statement: "PRAGMA mmap_size", value: &phases.mmapSize},
		{statement: "PRAGMA max_page_count", value: &phases.maxPageCount},
	} {
		if err := query.conn.QueryRowContext(ctx, setting.statement).Scan(setting.value); err != nil {
			return queryDatabaseError(ctx, "read query database setting", err)
		}
	}
	return nil
}

type filterNodeTiming struct {
	node              int
	operation, column string
	scanned, accepted int64
	elapsed           time.Duration
	complete          bool
}

func (phases *searchPhases) recordFilterNode(node filterNodeTiming) {
	if phases == nil {
		return
	}
	if phases.filterNodeCount < len(phases.filterNodes) {
		phases.filterNodes[phases.filterNodeCount] = node
	}
	phases.filterNodeCount++
}

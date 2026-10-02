package library

import (
	"context"
	"database/sql"
	_ "embed"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"sync"
	"time"

	"goodkind.io/lm-semantic-search/internal/clock"
)

//go:embed search_maximum_pages.sql
var searchMaximumPagesStatement string

type scoringTotals struct {
	verify, score, combined time.Duration
	verified, scored        int
}

type searchScoring struct {
	library               *Library
	query                 *queryDatabase
	phases                *searchPhases
	queryText             string
	revision              int64
	cancel                context.CancelCauseFunc
	jobs                  chan []VectorIdentity
	pending               []VectorIdentity
	workers               sync.WaitGroup
	closeJobs             sync.Once
	mutex                 sync.Mutex
	totals                scoringTotals
	stream                *scoreStream
	queryVector           []float32
	started, copyFinished time.Time
	reserved              int64
	recorded              bool
}

func newSearchScoring(cancel context.CancelCauseFunc, library *Library, query *queryDatabase, queryText string, phases *searchPhases) *searchScoring {
	return &searchScoring{
		library: library, query: query, queryText: queryText, phases: phases, cancel: cancel,
		jobs: make(chan []VectorIdentity, library.config.QueryWorkers),
	}
}

func (scoring *searchScoring) append(ctx context.Context, writer *sql.Tx, identities []VectorIdentity) error {
	for _, identity := range identities {
		scoring.pending = append(scoring.pending, identity)
		if len(scoring.pending) == scoring.library.config.QueryBlockSize {
			if err := scoring.submit(ctx, writer); err != nil {
				return err
			}
		}
	}
	return nil
}

func (scoring *searchScoring) start(ctx context.Context) error {
	if scoring.stream != nil {
		return nil
	}
	embedding := clock.Now()
	vector, err := scoring.library.embedQuery(ctx, scoring.queryText)
	scoring.phases.embed += clock.Now().Sub(embedding)
	if err != nil {
		return err
	}
	stream, err := newScoreStream(ctx, scoring.query.path)
	if err != nil {
		return err
	}
	scoring.stream, scoring.queryVector = stream, vector
	scoring.started = clock.Now()
	for range scoring.library.config.QueryWorkers {
		scoring.workers.Go(func() { scoring.work(ctx) })
	}
	return nil
}

func (scoring *searchScoring) submit(ctx context.Context, writer *sql.Tx) error {
	if len(scoring.pending) == 0 {
		return nil
	}
	if err := scoring.start(ctx); err != nil {
		return err
	}
	additional := int64(0)
	for _, identity := range scoring.pending {
		additional += int64(len(identity.ID)) + scoreFrameOverhead
	}
	remaining := scoring.library.config.MaxTemporaryBytes - scoring.reserved - additional
	pages := remaining / queryDatabasePageBytes
	if pages < 1 {
		slog.ErrorContext(ctx, "verified scores exceed the temporary budget", "err", ErrResourceLimit)
		return fmt.Errorf("%w: verified scores exceed the aggregate temporary budget", ErrResourceLimit)
	}
	statement := strings.ReplaceAll(searchMaximumPagesStatement, "{{pages}}", strconv.FormatInt(pages, 10))
	var actual int64
	if err := writer.QueryRowContext(ctx, statement).Scan(&actual); err != nil {
		return queryDatabaseError(ctx, "reserve score stream bytes", err)
	}
	if actual > pages {
		return fmt.Errorf("%w: query database and verified scores exceed MaxTemporaryBytes", ErrResourceLimit)
	}
	scoring.reserved += additional
	scoring.stream.reserve(additional)
	waiting := clock.Now()
	select {
	case scoring.jobs <- scoring.pending:
		scoring.pending = nil
		scoring.phases.copyScoreWait += clock.Now().Sub(waiting)
		return nil
	case <-ctx.Done():
		scoring.phases.copyScoreWait += clock.Now().Sub(waiting)
		return fmt.Errorf("submit verified score block: %w", context.Cause(ctx))
	}
}

func (scoring *searchScoring) work(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case identities, open := <-scoring.jobs:
			if !open {
				return
			}
			block := &scoreBlock{identities: identities}
			block.scores, block.err = scoring.library.scoreBlock(ctx, scoring.queryVector, scoring.revision, block)
			scoring.mutex.Lock()
			scoring.totals.verify += block.verifyTime
			scoring.totals.score += block.scoreTime
			scoring.totals.combined += block.combinedTime
			scoring.totals.verified += block.verified
			scoring.totals.scored += len(block.scores)
			scoring.mutex.Unlock()
			if block.err == nil {
				block.err = scoring.stream.write(ctx, block.scores)
			}
			if block.err != nil {
				scoring.cancel(block.err)
				return
			}
		}
	}
}

func (scoring *searchScoring) join() {
	scoring.closeJobs.Do(func() { close(scoring.jobs) })
	scoring.workers.Wait()
}

func (scoring *searchScoring) finish(ctx context.Context) error {
	scoring.join()
	scoring.recordPhases()
	if err := context.Cause(ctx); err != nil {
		slog.ErrorContext(ctx, "native score blocks failed", "err", err)
		return fmt.Errorf("finish native score blocks: %w", err)
	}
	if scoring.stream == nil {
		started := clock.Now()
		_, err := scoring.library.embedQuery(ctx, scoring.queryText)
		scoring.phases.embed += clock.Now().Sub(started)
		if err != nil {
			return err
		}
		return checkUnscoredVectors(ctx, scoring.query)
	}
	if err := scoring.save(ctx); err != nil {
		return err
	}
	return checkUnscoredVectors(ctx, scoring.query)
}

func (scoring *searchScoring) save(ctx context.Context) error {
	started := clock.Now()
	if err := scoring.stream.importScores(ctx, scoring.query); err != nil {
		return err
	}
	scoring.phases.scoreSave += clock.Now().Sub(started)
	if err := scoring.stream.close(ctx); err != nil {
		return err
	}
	statement := strings.ReplaceAll(searchMaximumPagesStatement, "{{pages}}", strconv.FormatInt(max(scoring.library.config.MaxTemporaryBytes/queryDatabasePageBytes, 1), 10))
	if _, err := scoring.query.conn.ExecContext(ctx, statement); err != nil {
		return queryDatabaseError(ctx, "release score stream reservation", err)
	}
	return nil
}

func (scoring *searchScoring) recordPhases() {
	if scoring.recorded {
		return
	}
	scoring.recorded = true
	scoring.phases.verify += scoring.totals.verify
	scoring.phases.score += scoring.totals.score
	scoring.phases.combined += scoring.totals.combined
	scoring.phases.verified += scoring.totals.verified
	scoring.phases.scored += scoring.totals.scored
	if !scoring.started.IsZero() {
		now := clock.Now()
		scoring.phases.dense = now.Sub(scoring.started)
		if scoring.copyFinished.After(scoring.started) {
			scoring.phases.copyScoreOverlap = min(now.Sub(scoring.started), scoring.copyFinished.Sub(scoring.started))
		}
	}
	if scoring.stream != nil {
		scoring.phases.scoreStreamBytes = scoring.stream.bytes
	}
}

func (scoring *searchScoring) close(ctx context.Context, cause error) error {
	scoring.cancel(cause)
	scoring.join()
	scoring.recordPhases()
	if scoring.stream == nil {
		return nil
	}
	return scoring.stream.close(ctx)
}

func checkUnscoredVectors(ctx context.Context, query *queryDatabase) error {
	var count int64
	if err := query.conn.QueryRowContext(ctx, unscoredVectorsStatement).Scan(&count); err != nil {
		return queryDatabaseError(ctx, "count unscored vectors", err)
	}
	if count != 0 {
		slog.ErrorContext(ctx, "dense scoring left vectors unscored", "vectors", count, "err", ErrVectorMissing)
		return fmt.Errorf("%w: %d eligible vectors have no exact score", ErrVectorMissing, count)
	}
	return nil
}

func scoringFailure(ctx context.Context, copyErr error) error {
	if copyErr == nil {
		return nil
	}
	slog.ErrorContext(ctx, "copy candidates while scoring failed", "err", copyErr)
	return errors.Join(copyErr, context.Cause(ctx))
}

package library

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"sync"
	"time"

	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/library/internal/vectorcodec"
)

// embedQuery embeds QueryInstructionPrefix followed by the query text and
// validates the vector. A nonfinite, zero-norm, or wrong-length vector fails.
func (library *Library) embedQuery(ctx context.Context, query string) ([]float32, error) {
	vectors, err := library.config.Embedder.EmbedBatch(ctx, []string{library.config.QueryInstructionPrefix + query})
	if err != nil {
		slog.ErrorContext(ctx, "embed search query failed", "err", err)
		return nil, fmt.Errorf("embed search query: %w", err)
	}
	if len(vectors) != 1 {
		err := fmt.Errorf("embedder returned %d vectors for one search query", len(vectors))
		slog.ErrorContext(ctx, "embed search query failed", "err", err)
		return nil, err
	}
	if err := vectorcodec.Validate(vectors[0], library.config.Store.Dimension); err != nil {
		slog.ErrorContext(ctx, "embedder returned an invalid query vector", "err", err)
		return nil, fmt.Errorf("embedder query vector: %w", err)
	}
	return vectors[0], nil
}

// scoreBlock is one bounded request of distinct vector identities with its
// result and the durations of its VerifyStrong and ScoreExact calls.
type scoreBlock struct {
	identities []VectorIdentity
	scores     []VectorScore
	err        error
	verifyTime time.Duration
	scoreTime  time.Duration
	// verified is the count of identities that VerifyStrong checked for this
	// block; identities verified earlier at the same revision are skipped.
	verified int
}

// scoreDense verifies and scores every distinct eligible vector in the query
// database. It reads QueryWorkers blocks of QueryBlockSize identities at a
// time in vector ID order, runs VerifyStrong and then ScoreExact for each
// block concurrently, and saves each block's scores. Any block failure fails
// the search. The last check requires a score for every vector. revision is
// the catalog visibility revision of the snapshot, which keys the
// verification cache.
func (library *Library) scoreDense(
	ctx context.Context,
	query *queryDatabase,
	queryVector []float32,
	revision int64,
	phases *searchPhases,
) error {
	after := ""
	blockSize := library.config.QueryBlockSize
	for {
		phases.stage = "dense_read_blocks"
		blocks, last, err := readScoreBlocks(ctx, query, after, blockSize, library.config.QueryWorkers)
		if err != nil {
			return err
		}
		if len(blocks) == 0 {
			break
		}
		phases.stage = "dense_native"
		scoreErr := library.runScoreBlocks(ctx, queryVector, revision, blocks)
		for _, block := range blocks {
			phases.verify += block.verifyTime
			phases.verified += block.verified
			phases.score += block.scoreTime
			phases.scored += len(block.scores)
		}
		if scoreErr != nil {
			return scoreErr
		}
		phases.stage = "dense_save_scores"
		started := clock.Now()
		saveErr := saveScores(ctx, query, blocks)
		phases.scoreSave += clock.Now().Sub(started)
		if saveErr != nil {
			return saveErr
		}
		after = last
	}
	phases.stage = "dense_check_scores"
	var unscored int64
	if err := query.conn.QueryRowContext(ctx, unscoredVectorsStatement).Scan(&unscored); err != nil {
		return queryDatabaseError(ctx, "count unscored vectors", err)
	}
	if unscored != 0 {
		missing := fmt.Errorf("%w: %d eligible vectors have no exact score", ErrVectorMissing, unscored)
		slog.ErrorContext(ctx, "dense scoring left vectors unscored", "err", missing)
		return missing
	}
	return nil
}

// readScoreBlocks reads up to workers blocks of blockSize identities with a
// vector ID above after. It returns the last vector ID it read.
func readScoreBlocks(
	ctx context.Context,
	query *queryDatabase,
	after string,
	blockSize int,
	workers int,
) (_ []*scoreBlock, _ string, err error) {
	rows, err := query.conn.QueryContext(ctx, vectorBlockStatement, after, blockSize*workers)
	if err != nil {
		return nil, "", queryDatabaseError(ctx, "read vector identities", err)
	}
	defer func() {
		err = errors.Join(err, closeRows(ctx, rows))
	}()
	var blocks []*scoreBlock
	last := after
	for rows.Next() {
		var identity VectorIdentity
		if err := rows.Scan(&identity.ID, &identity.IdentityDigest, &identity.Checksum); err != nil {
			return nil, "", queryDatabaseError(ctx, "scan vector identity", err)
		}
		if len(blocks) == 0 || len(blocks[len(blocks)-1].identities) == blockSize {
			blocks = append(blocks, &scoreBlock{identities: make([]VectorIdentity, 0, blockSize), scores: nil, err: nil})
		}
		current := blocks[len(blocks)-1]
		current.identities = append(current.identities, identity)
		last = identity.ID
	}
	if err := rows.Err(); err != nil {
		return nil, "", queryDatabaseError(ctx, "read vector identities", err)
	}
	return blocks, last, nil
}

// runScoreBlocks scores every block on its own goroutine and records each
// block's scores in the block. The first failure cancels the other blocks,
// and runScoreBlocks returns that failure.
func (library *Library) runScoreBlocks(ctx context.Context, queryVector []float32, revision int64, blocks []*scoreBlock) error {
	blockContext, cancel := context.WithCancel(ctx)
	defer cancel()
	var group sync.WaitGroup
	var failure error
	var failed sync.Once
	for _, block := range blocks {
		group.Go(func() {
			block.scores, block.err = library.scoreBlock(blockContext, queryVector, revision, block)
			if block.err != nil {
				failed.Do(func() {
					failure = block.err
					cancel()
				})
			}
		})
	}
	group.Wait()
	return failure
}

// scoreBlock verifies the identity digest and checksum of each vector of block
// with a strong read, except the identities that an earlier search verified
// at revision, and then asks for one exact score per ID. It records the
// duration of each call in block. It rejects a result with a missing, extra,
// reordered, duplicate, or nonfinite score.
func (library *Library) scoreBlock(ctx context.Context, queryVector []float32, revision int64, block *scoreBlock) ([]VectorScore, error) {
	identities := block.identities
	started := clock.Now()
	pending := library.verified.unverified(revision, identities)
	if len(pending) > 0 {
		if err := library.config.Vectors.VerifyStrong(ctx, pending); err != nil {
			block.verifyTime = clock.Now().Sub(started)
			slog.ErrorContext(ctx, "verify eligible vectors failed", "vectors", len(pending), "err", err)
			return nil, fmt.Errorf("verify %d eligible vectors: %w", len(pending), err)
		}
		library.verified.record(ctx, revision, pending)
	}
	block.verified = len(pending)
	verified := clock.Now()
	block.verifyTime = verified.Sub(started)
	ids := make([]string, 0, len(identities))
	for _, identity := range identities {
		ids = append(ids, identity.ID)
	}
	scores, err := library.config.Vectors.ScoreExact(ctx, queryVector, ids)
	block.scoreTime = clock.Now().Sub(verified)
	if err != nil {
		slog.ErrorContext(ctx, "exact scoring failed", "vectors", len(ids), "err", err)
		return nil, fmt.Errorf("score %d eligible vectors: %w", len(ids), err)
	}
	if len(scores) < len(ids) {
		missing := fmt.Errorf("%w: the vector store returned %d scores for %d IDs", ErrVectorMissing, len(scores), len(ids))
		slog.ErrorContext(ctx, "exact scoring returned too few scores", "err", missing)
		return nil, missing
	}
	if len(scores) > len(ids) {
		extra := fmt.Errorf("%w: the vector store returned %d scores for %d IDs", ErrVectorCorrupt, len(scores), len(ids))
		slog.ErrorContext(ctx, "exact scoring returned too many scores", "err", extra)
		return nil, extra
	}
	for index, score := range scores {
		if score.ID != ids[index] || math.IsNaN(score.Score) || math.IsInf(score.Score, 0) {
			corrupt := fmt.Errorf("%w: score %d is %q %v for requested ID %q", ErrVectorCorrupt, index, score.ID, score.Score, ids[index])
			slog.ErrorContext(ctx, "exact scoring returned an invalid score", "err", corrupt)
			return nil, corrupt
		}
	}
	return scores, nil
}

func saveScores(ctx context.Context, query *queryDatabase, blocks []*scoreBlock) (err error) {
	writer, err := query.conn.BeginTx(ctx, nil)
	if err != nil {
		return queryDatabaseError(ctx, "begin score save", err)
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, writer.Rollback())
		}
	}()
	for _, block := range blocks {
		for _, score := range block.scores {
			if _, err := writer.ExecContext(ctx, saveScoreStatement, score.Score, score.ID); err != nil {
				return queryDatabaseError(ctx, "save vector score", err)
			}
		}
	}
	if err := writer.Commit(); err != nil {
		return queryDatabaseError(ctx, "commit vector scores", err)
	}
	return nil
}

// Query database statements of the dense leg.
const (
	vectorBlockStatement     = `SELECT vector_id, identity_digest, vector_checksum FROM query_vectors WHERE vector_id > ? ORDER BY vector_id LIMIT ?`
	saveScoreStatement       = `UPDATE query_vectors SET score = ? WHERE vector_id = ?`
	unscoredVectorsStatement = `SELECT COUNT(*) FROM query_vectors WHERE score IS NULL`
)

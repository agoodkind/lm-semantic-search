package library

import (
	"context"
	"fmt"
	"log/slog"
	"math"
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
// result and separate or combined native call durations.
type scoreBlock struct {
	identities   []VectorIdentity
	scores       []VectorScore
	err          error
	verifyTime   time.Duration
	scoreTime    time.Duration
	combinedTime time.Duration
	// verified counts identities checked by separate or combined verification in this
	// block; identities verified earlier at the same revision are skipped.
	verified int
}

// scoreBlock uses optional combined verification and scoring for an entirely
// unverified block. Mixed and warm blocks retain separate verification and
// scoring. It caches a combined result only after validating every ordered,
// finite score.
func (library *Library) scoreBlock(ctx context.Context, queryVector []float32, revision int64, block *scoreBlock) ([]VectorScore, error) {
	identities := block.identities
	started := clock.Now()
	pending := library.verified.unverified(revision, identities)
	if scorer, ok := library.config.Vectors.(VerifiedExactScorer); ok && len(pending) == len(identities) {
		return library.scoreVerifiedBlock(ctx, queryVector, revision, block, scorer)
	}
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
	return validateBlockScores(ctx, ids, scores)
}

func (library *Library) scoreVerifiedBlock(ctx context.Context, queryVector []float32, revision int64, block *scoreBlock, scorer VerifiedExactScorer) ([]VectorScore, error) {
	started := clock.Now()
	scores, err := scorer.ScoreExactVerified(ctx, queryVector, block.identities)
	block.combinedTime = clock.Now().Sub(started)
	if err != nil {
		slog.ErrorContext(ctx, "verified exact scoring failed", "vectors", len(block.identities), "err", err)
		return nil, fmt.Errorf("verify and score %d eligible vectors: %w", len(block.identities), err)
	}
	ids := make([]string, len(block.identities))
	for index, identity := range block.identities {
		ids[index] = identity.ID
	}
	validated, err := validateBlockScores(ctx, ids, scores)
	if err != nil {
		return nil, err
	}
	library.verified.record(ctx, revision, block.identities)
	block.verified = len(block.identities)
	return validated, nil
}

func validateBlockScores(ctx context.Context, ids []string, scores []VectorScore) ([]VectorScore, error) {
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

// Query database statements of the dense leg.
const (
	unscoredVectorsStatement = `SELECT COUNT(*) FROM query_vectors WHERE score IS NULL`
)

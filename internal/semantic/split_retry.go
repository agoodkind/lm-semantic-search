package semantic

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"

	"goodkind.io/lm-semantic-search/internal/embedding"
	"goodkind.io/lm-semantic-search/internal/model"
)

const contextLengthExceededReason = "context_length_exceeded"

// ErrNilEmbeddingVector reports a provider contract violation where an input
// has neither a vector nor a matching SkippedInput.
var ErrNilEmbeddingVector = errors.New("embedding returned nil vector without a skip")

// EmbedBatchFunc embeds a batch of texts, returning one vector per input (nil for
// an input the endpoint rejected as un-embeddable) plus the endpoint's per-input
// rejections. A provider's EmbedBatch satisfies it directly.
type EmbedBatchFunc func(ctx context.Context, texts []string) (embedding.BatchResult, error)

// ChunkPackFunc groups chunks into embedding requests.
type ChunkPackFunc func(chunks []model.StoredChunk) [][]model.StoredChunk

type splitRetryPackResult struct {
	keptChunks   []model.StoredChunk
	keptVectors  [][]float32
	retryChunks  []model.StoredChunk
	droppedInput int
}

// EmbedChunksSplittingOversize embeds chunks and, for any chunk the endpoint
// rejects as context_length_exceeded, re-splits it into smaller sub-chunks and
// retries. It applies pack to every retry round. Only a context-length rejection
// may split, and splitting stops at the endpoint's token limit expressed as a
// conservative byte floor. A refusal at or below that floor is dropped as one
// input and logged once, which prevents an endpoint fault from shredding content
// codepoint by codepoint. Each split strictly shortens a splittable piece, so the
// byte floor also bounds the number of rounds. keptChunks and keptVectors remain
// index-aligned.
func EmbedChunksSplittingOversize(ctx context.Context, chunks []model.StoredChunk, pack ChunkPackFunc, embed EmbedBatchFunc) (keptChunks []model.StoredChunk, keptVectors [][]float32, droppedInputs int, err error) {
	keptChunks = make([]model.StoredChunk, 0, len(chunks))
	keptVectors = make([][]float32, 0, len(chunks))
	queue := slices.Clone(chunks)
	retryRound := 0
	for len(queue) > 0 {
		var nextQueue []model.StoredChunk
		for _, chunkPack := range pack(queue) {
			packResult, packErr := embedSplitRetryPack(ctx, chunkPack, retryRound, embed)
			if packErr != nil {
				return nil, nil, 0, packErr
			}
			keptChunks = append(keptChunks, packResult.keptChunks...)
			keptVectors = append(keptVectors, packResult.keptVectors...)
			nextQueue = append(nextQueue, packResult.retryChunks...)
			droppedInputs += packResult.droppedInput
		}
		queue = nextQueue
		retryRound++
	}
	return keptChunks, keptVectors, droppedInputs, nil
}

func embedSplitRetryPack(ctx context.Context, chunkPack []model.StoredChunk, retryRound int, embed EmbedBatchFunc) (splitRetryPackResult, error) {
	texts := make([]string, len(chunkPack))
	for index := range chunkPack {
		texts[index] = chunkPack[index].Content
	}
	result, err := embed(ctx, texts)
	if err != nil {
		slog.ErrorContext(ctx, "split-retry embed batch failed", "chunks", len(chunkPack), "err", err)
		return splitRetryPackResult{}, fmt.Errorf("split-retry embed batch: %w", err)
	}
	if len(result.Vectors) != len(chunkPack) {
		countErr := errors.New("vector count mismatch")
		slog.ErrorContext(ctx, "split-retry embed returned unexpected vector count", "want", len(chunkPack), "got", len(result.Vectors), "err", countErr)
		return splitRetryPackResult{}, fmt.Errorf("split-retry embed returned %d vectors for %d chunks: %w", len(result.Vectors), len(chunkPack), countErr)
	}
	skippedByIndex := make(map[int]embedding.SkippedInput, len(result.Skipped))
	for _, skip := range result.Skipped {
		skippedByIndex[skip.Index] = skip
	}
	packResult := splitRetryPackResult{
		keptChunks:   make([]model.StoredChunk, 0, len(chunkPack)),
		keptVectors:  make([][]float32, 0, len(chunkPack)),
		retryChunks:  nil,
		droppedInput: 0,
	}
	for index := range chunkPack {
		skip, wasSkipped := skippedByIndex[index]
		if !wasSkipped {
			vector := result.Vectors[index]
			if vector == nil {
				slog.ErrorContext(ctx, "split-retry embed returned nil vector without skip", "chunk_index", index, "relative_path", chunkPack[index].RelativePath, "err", ErrNilEmbeddingVector)
				return splitRetryPackResult{}, fmt.Errorf("split-retry chunk %d: %w", index, ErrNilEmbeddingVector)
			}
			packResult.keptChunks = append(packResult.keptChunks, chunkPack[index])
			packResult.keptVectors = append(packResult.keptVectors, vector)
			continue
		}
		if shouldSplitRejectedChunk(chunkPack[index], skip) {
			packResult.retryChunks = append(packResult.retryChunks, splitChunkInHalf(chunkPack[index])...)
			continue
		}
		logRejectedDrop(ctx, chunkPack[index], skip, retryRound)
		packResult.droppedInput++
	}
	return packResult, nil
}

func shouldSplitRejectedChunk(chunk model.StoredChunk, skip embedding.SkippedInput) bool {
	if skip.Reason != contextLengthExceededReason {
		return false
	}
	if isIndivisibleContent(chunk.Content) {
		return false
	}
	if len(chunk.Content) <= max(skip.MaxTokens, 1) {
		return false
	}
	return true
}

func rejectedDropKind(chunk model.StoredChunk, skip embedding.SkippedInput) string {
	if skip.Reason != contextLengthExceededReason {
		return "unexpected_reason"
	}
	if isIndivisibleContent(chunk.Content) {
		return "indivisible"
	}
	if len(chunk.Content) <= max(skip.MaxTokens, 1) {
		return "below_token_floor"
	}
	return "unknown"
}

func logRejectedDrop(ctx context.Context, chunk model.StoredChunk, skip embedding.SkippedInput, retryRound int) {
	slog.WarnContext(
		ctx,
		"semantic.embed_input_dropped",
		"drop_kind", rejectedDropKind(chunk, skip),
		"reason", skip.Reason,
		"conversation_id", chunk.ConversationID,
		"relative_path", chunk.RelativePath,
		"estimated_tokens", estimatedTokenCount(chunk.Content),
		"content_bytes", len(chunk.Content),
		"model_max_tokens", skip.MaxTokens,
		"reported_tokens", skip.ReportedTokens,
		"retry_round", retryRound,
	)
}

package semantic

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/embedding"
	"goodkind.io/lm-semantic-search/internal/model"
)

const (
	contextLengthExceededReason = "context_length_exceeded"
	// splitRetrySpecialTokenMargin reserves room for the start and end tokens
	// a tokenizer may add without consuming content bytes.
	splitRetrySpecialTokenMargin = 2
)

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
// may split, and splitting stops at a conservative content-byte floor derived
// from the token limit the endpoint reported, or from the active model limit when
// the endpoint reported none. The floor reserves two tokens for tokenizer-added
// start and end markers. A limit with no remaining content capacity drops the
// rejected input whole because splitting cannot help. Otherwise, every split strictly shortens a piece until
// it reaches the positive floor or one codepoint, so the loop stays bounded
// without a round cap. keptChunks and keptVectors remain index-aligned.
func EmbedChunksSplittingOversize(ctx context.Context, chunks []model.StoredChunk, activeModelMaxTokens int, pack ChunkPackFunc, embed EmbedBatchFunc) (keptChunks []model.StoredChunk, keptVectors [][]float32, droppedInputs int, err error) {
	keptChunks = make([]model.StoredChunk, 0, len(chunks))
	keptVectors = make([][]float32, 0, len(chunks))
	queue := slices.Clone(chunks)
	retryRound := 0
	for len(queue) > 0 {
		var nextQueue []model.StoredChunk
		for _, chunkPack := range pack(queue) {
			packResult, packErr := embedSplitRetryPack(
				ctx,
				chunkPack,
				activeModelMaxTokens,
				retryRound,
				embed,
			)
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

func embedSplitRetryPack(ctx context.Context, chunkPack []model.StoredChunk, activeModelMaxTokens int, retryRound int, embed EmbedBatchFunc) (splitRetryPackResult, error) {
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
		if shouldSplitRejectedChunk(chunkPack[index], skip, activeModelMaxTokens) {
			packResult.retryChunks = append(packResult.retryChunks, splitChunkInHalf(chunkPack[index])...)
			continue
		}
		logRejectedDrop(ctx, chunkPack[index], skip, activeModelMaxTokens, retryRound)
		packResult.droppedInput++
	}
	return packResult, nil
}

func shouldSplitRejectedChunk(chunk model.StoredChunk, skip embedding.SkippedInput, activeModelMaxTokens int) bool {
	if skip.Reason != contextLengthExceededReason {
		return false
	}
	if isIndivisibleContent(chunk.Content) {
		return false
	}
	byteFloor, hasContentCapacity := splitRetryContentByteFloor(
		skip.MaxTokens,
		activeModelMaxTokens,
	)
	if !hasContentCapacity {
		return false
	}
	if len(chunk.Content) <= byteFloor {
		return false
	}
	return true
}

// splitRetryContentByteFloor converts a token limit to the smallest content-byte
// size where another split cannot be assumed to help. A limit no larger than the
// special-token margin has no content capacity, so the caller drops the rejected
// input whole.
func splitRetryContentByteFloor(reportedMaxTokens adapterr.EmbedFigure, activeModelMaxTokens int) (int, bool) {
	modelMaxTokens := splitRetryModelMaxTokens(reportedMaxTokens, activeModelMaxTokens)
	if modelMaxTokens <= splitRetrySpecialTokenMargin {
		return 0, false
	}
	return modelMaxTokens - splitRetrySpecialTokenMargin, true
}

// splitRetryModelMaxTokens picks the token limit the next split is measured
// against. The loop cannot make progress without a number, so it takes the
// endpoint's figure only when the endpoint reported one that a limit could
// actually be, which is any count from zero up: a maximum genuinely reported as
// zero says the model has no room at all, and the caller drops the input rather
// than retrying against a limit the endpoint never claimed. A figure the endpoint
// never reported, and a negative one no limit could take, both fall back to the
// active model's limit, and then to the smallest limit any known embedding model
// accepts when the caller passes no usable limit either.
func splitRetryModelMaxTokens(reportedMaxTokens adapterr.EmbedFigure, activeModelMaxTokens int) int {
	if reportedMaxTokens.Reported && reportedMaxTokens.Value >= 0 {
		return reportedMaxTokens.Value
	}
	if activeModelMaxTokens > 0 {
		return activeModelMaxTokens
	}
	return config.MinimumKnownEmbedModelInputTokenLimit
}

func rejectedDropKind(chunk model.StoredChunk, skip embedding.SkippedInput, activeModelMaxTokens int) string {
	if skip.Reason != contextLengthExceededReason {
		return "unexpected_reason"
	}
	if isIndivisibleContent(chunk.Content) {
		return "indivisible"
	}
	byteFloor, hasContentCapacity := splitRetryContentByteFloor(
		skip.MaxTokens,
		activeModelMaxTokens,
	)
	if !hasContentCapacity {
		return "no_content_capacity"
	}
	if len(chunk.Content) <= byteFloor {
		return "below_token_floor"
	}
	return "unknown"
}

func logRejectedDrop(ctx context.Context, chunk model.StoredChunk, skip embedding.SkippedInput, activeModelMaxTokens int, retryRound int) {
	byteFloor, _ := splitRetryContentByteFloor(skip.MaxTokens, activeModelMaxTokens)
	slog.WarnContext(
		ctx,
		"semantic.embed_input_dropped",
		"drop_kind", rejectedDropKind(chunk, skip, activeModelMaxTokens),
		"reason", skip.Reason,
		"conversation_id", chunk.ConversationID,
		"relative_path", chunk.RelativePath,
		"estimated_tokens", estimatedTokenCount(chunk.Content),
		"content_bytes", len(chunk.Content),
		"model_max_tokens", skip.MaxTokens,
		"active_model_max_tokens", activeModelMaxTokens,
		"content_byte_floor", byteFloor,
		"reported_tokens", skip.ReportedTokens,
		"retry_round", retryRound,
	)
}

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

// EmbedBatchFunc embeds a batch of texts, returning one vector per input (nil for
// an input the endpoint rejected as un-embeddable) plus the endpoint's per-input
// rejections. A provider's EmbedBatch satisfies it directly, so the
// split-and-retry loop is identical for the Milvus and local backends.
type EmbedBatchFunc func(ctx context.Context, texts []string) (embedding.BatchResult, error)

// EmbedChunksSplittingOversize embeds chunks and, for any chunk the endpoint
// rejects as context_length_exceeded, re-splits it into smaller sub-chunks and
// retries, looping until every piece embeds or a piece is a single indivisible
// UTF-8 codepoint the endpoint still rejects. An indivisible piece is dropped,
// logged loudly at WARN with its identity and the endpoint's token figures, and
// counted in droppedIndivisible; every other piece is embedded and returned with
// its vector. The returned chunks carry distinct SplitParts so identical pieces
// never collide on the primary key, and all pieces keep the parent relativePath
// so a message delete-by-prefix still removes them. keptChunks and keptVectors
// are index-aligned and may outnumber the input when chunks were re-split.
func EmbedChunksSplittingOversize(ctx context.Context, chunks []model.StoredChunk, embed EmbedBatchFunc) (keptChunks []model.StoredChunk, keptVectors [][]float32, droppedIndivisible int, err error) {
	keptChunks = make([]model.StoredChunk, 0, len(chunks))
	keptVectors = make([][]float32, 0, len(chunks))
	queue := slices.Clone(chunks)
	for len(queue) > 0 {
		texts := make([]string, len(queue))
		for index := range queue {
			texts[index] = queue[index].Content
		}
		result, embedErr := embed(ctx, texts)
		if embedErr != nil {
			slog.ErrorContext(ctx, "split-retry embed batch failed", "chunks", len(queue), "err", embedErr)
			return nil, nil, 0, fmt.Errorf("split-retry embed batch: %w", embedErr)
		}
		if len(result.Vectors) != len(queue) {
			countErr := errors.New("vector count mismatch")
			slog.ErrorContext(ctx, "split-retry embed returned unexpected vector count", "want", len(queue), "got", len(result.Vectors), "err", countErr)
			return nil, nil, 0, fmt.Errorf("split-retry embed returned %d vectors for %d chunks: %w", len(result.Vectors), len(queue), countErr)
		}
		skippedByIndex := make(map[int]embedding.SkippedInput, len(result.Skipped))
		for _, skip := range result.Skipped {
			skippedByIndex[skip.Index] = skip
		}
		var nextQueue []model.StoredChunk
		for index := range queue {
			skip, wasSkipped := skippedByIndex[index]
			if !wasSkipped {
				keptChunks = append(keptChunks, queue[index])
				keptVectors = append(keptVectors, result.Vectors[index])
				continue
			}
			if isIndivisibleContent(queue[index].Content) {
				logIndivisibleDrop(ctx, queue[index], skip)
				droppedIndivisible++
				continue
			}
			nextQueue = append(nextQueue, splitChunkInHalf(queue[index])...)
		}
		queue = nextQueue
	}
	return keptChunks, keptVectors, droppedIndivisible, nil
}

// logIndivisibleDrop records at WARN that one input was dropped because it is a
// single indivisible UTF-8 codepoint the embedding endpoint still rejects as too
// large. It names the input by conversation id or relative path and reports both
// the local size estimate and the endpoint's own token figures, so the drop is
// diagnosable and never silent.
func logIndivisibleDrop(ctx context.Context, chunk model.StoredChunk, skip embedding.SkippedInput) {
	slog.WarnContext(
		ctx,
		"semantic.embed_input_dropped_indivisible",
		"reason", skip.Reason,
		"conversation_id", chunk.ConversationID,
		"relative_path", chunk.RelativePath,
		"estimated_tokens", estimatedTokenCount(chunk.Content),
		"content_bytes", len(chunk.Content),
		"model_max_tokens", skip.MaxTokens,
		"reported_tokens", skip.ReportedTokens,
	)
}

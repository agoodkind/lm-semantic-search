package semantic

import (
	"context"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/embedding"
	"goodkind.io/lm-semantic-search/internal/model"
)

type rejectingAllEmbedder struct {
	requestCount int
}

func (embedder *rejectingAllEmbedder) Embed(
	_ context.Context,
	_ string,
) ([]float32, error) {
	return []float32{0}, nil
}

func (embedder *rejectingAllEmbedder) EmbedBatch(
	_ context.Context,
	texts []string,
) (embedding.BatchResult, error) {
	embedder.requestCount++
	vectors := make([][]float32, len(texts))
	skipped := make([]embedding.SkippedInput, 0, len(texts))
	for index := range texts {
		skipped = append(skipped, embedding.SkippedInput{
			Index:          index,
			Reason:         contextLengthExceededReason,
			ReportedTokens: len(texts[index]),
			MaxTokens:      512,
		})
	}
	return embedding.BatchResult{Vectors: vectors, Skipped: skipped}, nil
}

func (embedder *rejectingAllEmbedder) ProviderName() string {
	return "rejecting-all"
}

func (embedder *rejectingAllEmbedder) Health(_ context.Context) error {
	return nil
}

func TestInsertChunksBatchedReportsSplitRetryDrops(t *testing.T) {
	t.Parallel()
	embedder := &rejectingAllEmbedder{}
	service := &Service{
		cfg: config.Config{
			EmbeddingBatchSize:        32,
			EmbeddingBatchTokenBudget: 6000,
		},
		embedder: embedder,
	}
	chunks := []model.StoredChunk{{
		Content:      strings.Repeat("x", 9215),
		RelativePath: "conv/x/drop-progress",
	}}
	var reports []Progress

	err := service.insertChunksBatched(
		context.Background(),
		"test_collection",
		chunks,
		true,
		"test",
		func(progress Progress) {
			reports = append(reports, progress)
		},
		nil,
		StoreColumnSetCode,
	)
	if err != nil {
		t.Fatalf("insertChunksBatched returned error: %v", err)
	}
	if len(reports) != 1 {
		t.Fatalf("progress reports = %d, want 1", len(reports))
	}
	if reports[0].ChunksDropped != 63 {
		t.Fatalf("ChunksDropped = %d, want 63", reports[0].ChunksDropped)
	}
	if embedder.requestCount != 8 {
		t.Fatalf("embedding requests = %d, want 8 including the initial rejected batch", embedder.requestCount)
	}
}

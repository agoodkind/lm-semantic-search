package semantic

import (
	"context"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
)

func TestExpandOverTokenBudgetSplitsOversizeChunk(t *testing.T) {
	t.Parallel()
	service := &Service{cfg: config.Config{EmbeddingMaxTokens: 4096}}
	tokenCap := config.EffectiveEmbedTokenCap(4096)
	byteBudget := config.EmbedChunkByteBudget(4096)

	content := strings.Repeat("a", byteBudget*3+7)
	chunks := []model.StoredChunk{{Content: content, RelativePath: "conv/x/1"}}

	out := service.expandOverTokenBudget(context.Background(), "cb", chunks, "test")
	if len(out) < 3 {
		t.Fatalf("expected at least 3 sub-chunks, got %d", len(out))
	}
	var joined strings.Builder
	for _, chunk := range out {
		if estimatedTokenCount(chunk.Content) > tokenCap {
			t.Fatalf("sub-chunk over cap: est %d > %d", estimatedTokenCount(chunk.Content), tokenCap)
		}
		if chunk.RelativePath != "conv/x/1" {
			t.Fatalf("sub-chunk lost RelativePath: %q", chunk.RelativePath)
		}
		joined.WriteString(chunk.Content)
	}
	if joined.String() != content {
		t.Fatal("sub-chunks did not round-trip to the original content")
	}
}

func TestExpandOverTokenBudgetPassesThroughWhenDisabled(t *testing.T) {
	t.Parallel()
	service := &Service{cfg: config.Config{EmbeddingMaxTokens: 0}}
	content := strings.Repeat("a", 100000)
	chunks := []model.StoredChunk{{Content: content, RelativePath: "x"}}

	out := service.expandOverTokenBudget(context.Background(), "cb", chunks, "test")
	if len(out) != 1 || out[0].Content != content {
		t.Fatalf("expected passthrough when the cap is disabled, got %d chunks", len(out))
	}
}

func TestExpandOverTokenBudgetLeavesUnderBudgetUnchanged(t *testing.T) {
	t.Parallel()
	service := &Service{cfg: config.Config{EmbeddingMaxTokens: 4096}}
	chunks := []model.StoredChunk{
		{Content: "small one", RelativePath: "a"},
		{Content: "small two", RelativePath: "b"},
	}
	out := service.expandOverTokenBudget(context.Background(), "cb", chunks, "test")
	if len(out) != 2 {
		t.Fatalf("under-budget chunks should be unchanged, got %d", len(out))
	}
}

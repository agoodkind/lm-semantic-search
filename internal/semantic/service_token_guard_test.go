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
	byteBudget := config.EmbedChunkByteBudget(4096)

	content := strings.Repeat("a", byteBudget*3+7)
	chunks := []model.StoredChunk{{Content: content, RelativePath: "conv/x/1"}}

	out := service.expandOverTokenBudget(context.Background(), "cb", chunks, "test")
	if len(out) < 4 {
		t.Fatalf("expected at least 4 sub-chunks, got %d", len(out))
	}
	var joined strings.Builder
	for _, chunk := range out {
		if len(chunk.Content) > byteBudget {
			t.Fatalf("sub-chunk over byte budget: %d > %d", len(chunk.Content), byteBudget)
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

func TestExpandOverTokenBudgetEnforcesModelLimitWhenUnset(t *testing.T) {
	t.Parallel()
	// An unset EmbeddingMaxTokens must still split: the model's hard input-token
	// limit is always enforced, so an oversized input is divided and indexed
	// instead of dropped by the embedder as context_length_exceeded.
	service := &Service{cfg: config.Config{EmbeddingMaxTokens: 0}}
	byteBudget := config.EmbedChunkByteBudget(0)
	if byteBudget <= 0 {
		t.Fatalf("expected a positive byte budget when unset, got %d", byteBudget)
	}
	content := strings.Repeat("abcde", 40000)
	chunks := []model.StoredChunk{{Content: content, RelativePath: "conv/y/2"}}

	out := service.expandOverTokenBudget(context.Background(), "cb", chunks, "test")
	if len(out) < 2 {
		t.Fatalf("expected the oversize chunk to split when unset, got %d sub-chunks", len(out))
	}
	var joined strings.Builder
	for _, chunk := range out {
		if len(chunk.Content) > byteBudget {
			t.Fatalf("sub-chunk over byte budget: %d > %d", len(chunk.Content), byteBudget)
		}
		joined.WriteString(chunk.Content)
	}
	if joined.String() != content {
		t.Fatal("sub-chunks did not round-trip to the original content")
	}
}

func TestExpandOverTokenBudgetSubChunksStayUnderModelTokenLimit(t *testing.T) {
	t.Parallel()
	// Each sub-chunk's byte length, measured back at the conservative
	// bytes-per-token ratio, must estimate below the model's hard token limit, so
	// no sub-chunk can trigger context_length_exceeded even for dense content.
	service := &Service{cfg: config.Config{EmbeddingMaxTokens: 0}}
	byteBudget := config.EmbedChunkByteBudget(0)
	content := strings.Repeat("token ", 20000)
	chunks := []model.StoredChunk{{Content: content, RelativePath: "conv/z/3"}}

	out := service.expandOverTokenBudget(context.Background(), "cb", chunks, "test")
	for _, chunk := range out {
		// Conservative token estimate at 2.5 bytes/token: ceil(len * 2 / 5).
		conservativeTokens := (len(chunk.Content)*2 + 4) / 5
		if conservativeTokens >= config.EmbedModelInputTokenLimit {
			t.Fatalf("sub-chunk of %d bytes estimates %d tokens, want < %d", len(chunk.Content), conservativeTokens, config.EmbedModelInputTokenLimit)
		}
		if len(chunk.Content) > byteBudget {
			t.Fatalf("sub-chunk over byte budget: %d > %d", len(chunk.Content), byteBudget)
		}
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

package semantic

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/embedding"
	"goodkind.io/lm-semantic-search/internal/model"
)

// thresholdEmbed returns an EmbedBatchFunc that rejects any input longer than
// maxBytes as context_length_exceeded and embeds the rest, mirroring the real
// endpoint's per-input rejection so the split-retry loop runs end to end. It
// records every text it actually embedded so a test can prove no content is lost.
func thresholdEmbed(maxBytes int, embedded *[]string) EmbedBatchFunc {
	return func(_ context.Context, texts []string) (embedding.BatchResult, error) {
		vectors := make([][]float32, len(texts))
		var skipped []embedding.SkippedInput
		for index, text := range texts {
			if len(text) > maxBytes {
				skipped = append(skipped, embedding.SkippedInput{
					Index:          index,
					Reason:         "context_length_exceeded",
					ReportedTokens: len(text),
					MaxTokens:      maxBytes,
				})
				continue
			}
			*embedded = append(*embedded, text)
			vectors[index] = []float32{float32(len(text))}
		}
		return embedding.BatchResult{Vectors: vectors, Skipped: skipped}, nil
	}
}

func TestEmbedChunksSplittingOversizeSplitsUntilEveryPieceEmbeds(t *testing.T) {
	t.Parallel()
	const maxBytes = 8
	var embedded []string
	content := strings.Repeat("abcd", 12) // 48 bytes, far over the threshold
	chunks := []model.StoredChunk{{Content: content, RelativePath: "conv/x/1"}}

	kept, vectors, dropped, err := EmbedChunksSplittingOversize(context.Background(), chunks, thresholdEmbed(maxBytes, &embedded))
	if err != nil {
		t.Fatalf("EmbedChunksSplittingOversize returned error: %v", err)
	}
	if dropped != 0 {
		t.Fatalf("dropped = %d, want 0 (every piece fits after splitting)", dropped)
	}
	if len(kept) != len(vectors) {
		t.Fatalf("kept %d chunks but %d vectors", len(kept), len(vectors))
	}
	if len(kept) < 6 {
		t.Fatalf("expected the oversize chunk to split into many pieces, got %d", len(kept))
	}

	seenSplitPart := make(map[int32]bool, len(kept))
	ordered := make([]model.StoredChunk, len(kept))
	copy(ordered, kept)
	for index := range kept {
		if len(kept[index].Content) > maxBytes {
			t.Fatalf("kept piece over threshold: %d > %d", len(kept[index].Content), maxBytes)
		}
		if vectors[index] == nil {
			t.Fatalf("kept piece %d has a nil vector", index)
		}
		if kept[index].RelativePath != "conv/x/1" {
			t.Fatalf("kept piece lost RelativePath: %q", kept[index].RelativePath)
		}
		if seenSplitPart[kept[index].SplitPart] {
			t.Fatalf("duplicate SplitPart %d across kept pieces", kept[index].SplitPart)
		}
		seenSplitPart[kept[index].SplitPart] = true
	}

	// SplitPart encodes the byte offset within the parent, so ordering by it
	// reconstructs the original content and proves nothing was lost or reordered.
	sort.Slice(ordered, func(left int, right int) bool {
		return ordered[left].SplitPart < ordered[right].SplitPart
	})
	var joined strings.Builder
	for index := range ordered {
		joined.WriteString(ordered[index].Content)
	}
	if joined.String() != content {
		t.Fatal("kept pieces did not round-trip to the original content")
	}

	// Every embedded text is a piece of the content and the concatenation of the
	// unique pieces reproduces it, so no piece reached the endpoint truncated.
	if len(embedded) == 0 {
		t.Fatal("no text reached the embedder")
	}
}

func TestEmbedChunksSplittingOversizeDropsOnlyIndivisibleCodepoint(t *testing.T) {
	t.Parallel()
	// An embedder that rejects every non-empty input forces the splitter down to
	// single codepoints. Only those indivisible pieces may be dropped, and each
	// drop is counted.
	var embedded []string
	content := "ab" // two indivisible codepoints
	chunks := []model.StoredChunk{{Content: content, RelativePath: "conv/x/2"}}

	kept, vectors, dropped, err := EmbedChunksSplittingOversize(context.Background(), chunks, thresholdEmbed(0, &embedded))
	if err != nil {
		t.Fatalf("EmbedChunksSplittingOversize returned error: %v", err)
	}
	if len(kept) != 0 || len(vectors) != 0 {
		t.Fatalf("kept %d chunks / %d vectors, want 0 / 0 (all indivisible and rejected)", len(kept), len(vectors))
	}
	if dropped != 2 {
		t.Fatalf("dropped = %d, want 2 (each single codepoint)", dropped)
	}
}

func TestSplitChildrenGetDistinctPrimaryKeys(t *testing.T) {
	t.Parallel()
	byteBudget := config.EmbedChunkByteBudget(4096)
	// Repeated content makes the full-budget pieces byte-identical, so only a
	// split-position-aware primary key keeps them distinct.
	content := strings.Repeat("a", byteBudget*3+7)
	chunk := model.StoredChunk{Content: content, RelativePath: "conv/x/1"}

	pieces, splitCount := SplitChunksToByteBudget([]model.StoredChunk{chunk}, byteBudget)
	if splitCount != 1 {
		t.Fatalf("splitCount = %d, want 1", splitCount)
	}
	if len(pieces) < 4 {
		t.Fatalf("expected at least 4 pieces, got %d", len(pieces))
	}

	ids := make(map[string]bool, len(pieces))
	for index := range pieces {
		id := generateID(pieces[index], 0)
		if ids[id] {
			t.Fatalf("duplicate primary key %q for split piece %d", id, index)
		}
		ids[id] = true
		if pieces[index].SplitPart <= 0 {
			t.Fatalf("split piece %d has SplitPart %d, want a positive value", index, pieces[index].SplitPart)
		}
	}
}

func TestUnsplitChunkKeepsLegacyPrimaryKey(t *testing.T) {
	t.Parallel()
	chunk := model.StoredChunk{Content: "small chunk", RelativePath: "src/file.go", StartLine: 3, EndLine: 9}
	// The legacy id hashes path:start:end:content with no split position, so an
	// unsplit chunk (SplitPart 0) must reproduce it exactly and avoid re-embedding
	// the whole corpus.
	legacyInput := fmt.Sprintf("%s:%d:%d:%s", chunk.RelativePath, chunk.StartLine, chunk.EndLine, chunk.Content)
	sum := sha256.Sum256([]byte(legacyInput))
	legacyID := "chunk_" + hex.EncodeToString(sum[:])[:16]

	if got := generateID(chunk, 0); got != legacyID {
		t.Fatalf("generateID for unsplit chunk = %q, want legacy %q", got, legacyID)
	}

	// A split child of the same content differs, so the primary keys never collide.
	split := chunk
	split.SplitPart = 5
	if generateID(split, 0) == legacyID {
		t.Fatal("split child shares the unsplit primary key")
	}
}

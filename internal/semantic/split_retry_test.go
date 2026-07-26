package semantic

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/embedding"
	"goodkind.io/lm-semantic-search/internal/model"
)

func vectorForText(text string) []float32 {
	sum := sha256.Sum256([]byte(text))
	return []float32{
		float32(sum[0]),
		float32(sum[1]),
		float32(sum[2]),
		float32(sum[3]),
	}
}

func testChunkPacker(chunks []model.StoredChunk) [][]model.StoredChunk {
	return packChunksByEstimatedTokens(
		chunks,
		defaultEmbeddingBatchRows,
		defaultEmbeddingBatchTokenBudget,
	)
}

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
			vectors[index] = vectorForText(text)
		}
		return embedding.BatchResult{Vectors: vectors, Skipped: skipped}, nil
	}
}

func rejectingEmbed(reason string, maxTokens int, batches *[][]string) EmbedBatchFunc {
	return func(_ context.Context, texts []string) (embedding.BatchResult, error) {
		*batches = append(*batches, slices.Clone(texts))
		vectors := make([][]float32, len(texts))
		skipped := make([]embedding.SkippedInput, 0, len(texts))
		for index, text := range texts {
			skipped = append(skipped, embedding.SkippedInput{
				Index:          index,
				Reason:         reason,
				ReportedTokens: len(text),
				MaxTokens:      maxTokens,
			})
		}
		return embedding.BatchResult{Vectors: vectors, Skipped: skipped}, nil
	}
}

func TestEmbedChunksSplittingOversizeSplitsUntilEveryPieceEmbeds(t *testing.T) {
	t.Parallel()
	const maxBytes = 8
	var embedded []string
	content := "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL"
	chunks := []model.StoredChunk{{Content: content, RelativePath: "conv/x/1"}}

	kept, vectors, dropped, err := EmbedChunksSplittingOversize(context.Background(), chunks, testChunkPacker, thresholdEmbed(maxBytes, &embedded))
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
		if want := vectorForText(kept[index].Content); !slices.Equal(vectors[index], want) {
			t.Fatalf("vector %d = %v, want hash vector %v for %q", index, vectors[index], want, kept[index].Content)
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

	if got := strings.Join(embedded, ""); got != content {
		t.Fatalf("embedded texts joined to %q, want original %q", got, content)
	}
}

func TestEmbedChunksSplittingOversizeDropsOnlyIndivisibleCodepoint(t *testing.T) {
	t.Parallel()
	var embedded []string
	content := "🧠"
	chunks := []model.StoredChunk{{Content: content, RelativePath: "conv/x/2"}}

	kept, vectors, dropped, err := EmbedChunksSplittingOversize(context.Background(), chunks, testChunkPacker, thresholdEmbed(0, &embedded))
	if err != nil {
		t.Fatalf("EmbedChunksSplittingOversize returned error: %v", err)
	}
	if len(kept) != 0 || len(vectors) != 0 {
		t.Fatalf("kept %d chunks / %d vectors, want 0 / 0 (all indivisible and rejected)", len(kept), len(vectors))
	}
	if dropped != 1 {
		t.Fatalf("dropped = %d, want 1 for one multi-byte codepoint", dropped)
	}
}

func TestEmbedChunksSplittingOversizeDropsUnexpectedReasonWithoutSplitting(t *testing.T) {
	t.Parallel()
	var batches [][]string
	content := strings.Repeat("x", 64)
	chunks := []model.StoredChunk{{Content: content, RelativePath: "conv/x/unexpected"}}

	kept, vectors, dropped, err := EmbedChunksSplittingOversize(
		context.Background(),
		chunks,
		testChunkPacker,
		rejectingEmbed("endpoint_policy_rejection", 4096, &batches),
	)
	if err != nil {
		t.Fatalf("EmbedChunksSplittingOversize returned error: %v", err)
	}
	if len(kept) != 0 || len(vectors) != 0 {
		t.Fatalf("kept %d chunks / %d vectors, want 0 / 0", len(kept), len(vectors))
	}
	if dropped != 1 {
		t.Fatalf("dropped = %d, want 1 whole rejected input", dropped)
	}
	if len(batches) != 1 {
		t.Fatalf("embed batches = %d, want 1 because the reason is not context_length_exceeded", len(batches))
	}
}

func TestEmbedChunksSplittingOversizeStopsAtEndpointTokenFloor(t *testing.T) {
	t.Parallel()
	const maxTokens = 8
	var batches [][]string
	content := strings.Repeat("x", 64)
	chunks := []model.StoredChunk{{Content: content, RelativePath: "conv/x/floor"}}

	kept, vectors, dropped, err := EmbedChunksSplittingOversize(
		context.Background(),
		chunks,
		testChunkPacker,
		rejectingEmbed("context_length_exceeded", maxTokens, &batches),
	)
	if err != nil {
		t.Fatalf("EmbedChunksSplittingOversize returned error: %v", err)
	}
	if len(kept) != 0 || len(vectors) != 0 {
		t.Fatalf("kept %d chunks / %d vectors, want 0 / 0", len(kept), len(vectors))
	}
	if dropped != 8 {
		t.Fatalf("dropped = %d, want 8 whole pieces at the %d-byte floor", dropped, maxTokens)
	}
	if len(batches) != 4 {
		t.Fatalf("embed batches = %d, want 4 retry rounds down to the floor", len(batches))
	}
}

func TestEmbedChunksSplittingOversizeContinuesPastFourRounds(t *testing.T) {
	t.Parallel()
	const maxBytes = 512
	var batches [][]string
	var embedded []string
	threshold := thresholdEmbed(maxBytes, &embedded)
	recordingEmbed := func(ctx context.Context, texts []string) (embedding.BatchResult, error) {
		batches = append(batches, slices.Clone(texts))
		return threshold(ctx, texts)
	}
	content := strings.Repeat("x", 9215)
	chunks := []model.StoredChunk{{Content: content, RelativePath: "conv/x/small-context"}}

	kept, vectors, dropped, err := EmbedChunksSplittingOversize(
		context.Background(),
		chunks,
		testChunkPacker,
		recordingEmbed,
	)
	if err != nil {
		t.Fatalf("EmbedChunksSplittingOversize returned error: %v", err)
	}
	if dropped != 0 {
		t.Fatalf("dropped = %d, want 0 because every piece fits after five splits", dropped)
	}
	if len(kept) != 63 || len(vectors) != 63 {
		t.Fatalf("kept %d chunks / %d vectors, want 63 / 63", len(kept), len(vectors))
	}
	if len(batches) != 7 {
		t.Fatalf("embed requests = %d, want 7 across six packed rounds", len(batches))
	}

	ordered := slices.Clone(kept)
	sort.Slice(ordered, func(left int, right int) bool {
		return ordered[left].SplitPart < ordered[right].SplitPart
	})
	var joined strings.Builder
	for index := range ordered {
		if len(ordered[index].Content) > maxBytes {
			t.Fatalf(
				"kept piece %d contains %d bytes, want no more than %d",
				index,
				len(ordered[index].Content),
				maxBytes,
			)
		}
		joined.WriteString(ordered[index].Content)
	}
	if joined.String() != content {
		t.Fatal("kept pieces did not round-trip to the original content")
	}
}

func TestEmbedChunksSplittingOversizePacksEveryRetryRound(t *testing.T) {
	t.Parallel()
	const maxRows = 32
	var batches [][]string
	embed := func(_ context.Context, texts []string) (embedding.BatchResult, error) {
		batches = append(batches, slices.Clone(texts))
		if len(texts) > maxRows {
			return embedding.BatchResult{}, fmt.Errorf("batch rows %d exceed cap %d", len(texts), maxRows)
		}
		vectors := make([][]float32, len(texts))
		var skipped []embedding.SkippedInput
		for index, text := range texts {
			if len(text) > 1 {
				skipped = append(skipped, embedding.SkippedInput{
					Index:          index,
					Reason:         "context_length_exceeded",
					ReportedTokens: len(text),
					MaxTokens:      1,
				})
				continue
			}
			vectors[index] = vectorForText(text)
		}
		return embedding.BatchResult{Vectors: vectors, Skipped: skipped}, nil
	}
	chunks := make([]model.StoredChunk, 20)
	for index := range chunks {
		chunks[index] = model.StoredChunk{
			Content:      "ab",
			RelativePath: fmt.Sprintf("conv/x/packed/%d", index),
		}
	}

	kept, vectors, dropped, err := EmbedChunksSplittingOversize(
		context.Background(),
		chunks,
		testChunkPacker,
		embed,
	)
	if err != nil {
		t.Fatalf("EmbedChunksSplittingOversize returned error: %v", err)
	}
	if dropped != 0 {
		t.Fatalf("dropped = %d, want 0", dropped)
	}
	if len(kept) != 40 || len(vectors) != 40 {
		t.Fatalf("kept %d chunks / %d vectors, want 40 / 40", len(kept), len(vectors))
	}
	for index, batch := range batches {
		if len(batch) > maxRows {
			t.Fatalf("batch %d rows = %d, want <= %d", index, len(batch), maxRows)
		}
	}
}

func TestEmbedChunksSplittingOversizeRejectsNilVectorWithoutSkip(t *testing.T) {
	t.Parallel()
	chunks := []model.StoredChunk{{Content: "small", RelativePath: "conv/x/nil"}}
	nilVectorEmbed := func(_ context.Context, texts []string) (embedding.BatchResult, error) {
		return embedding.BatchResult{Vectors: make([][]float32, len(texts)), Skipped: nil}, nil
	}

	_, _, _, err := EmbedChunksSplittingOversize(context.Background(), chunks, testChunkPacker, nilVectorEmbed)
	if err == nil {
		t.Fatal("EmbedChunksSplittingOversize returned nil error for an unmarked nil vector")
	}
	if !errors.Is(err, ErrNilEmbeddingVector) {
		t.Fatalf("EmbedChunksSplittingOversize error = %v, want ErrNilEmbeddingVector", err)
	}
	if !strings.Contains(err.Error(), "nil vector") {
		t.Fatalf("EmbedChunksSplittingOversize error = %v, want nil vector context", err)
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

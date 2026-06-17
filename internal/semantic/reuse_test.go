package semantic

import (
	"context"
	"slices"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

// countingEmbedder records every EmbedBatch call so a test can prove that a
// reused vector never reaches the embedder. Each returned vector is a single
// element holding the input length, which is deterministic and lets the test
// tell an embedded vector apart from a reused one.
type countingEmbedder struct {
	batches [][]string
}

func (embedder *countingEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return []float32{0}, nil
}

func (embedder *countingEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	embedder.batches = append(embedder.batches, slices.Clone(texts))
	vectors := make([][]float32, len(texts))
	for index, text := range texts {
		vectors[index] = []float32{float32(len(text))}
	}
	return vectors, nil
}

func (embedder *countingEmbedder) ProviderName() string { return "counting" }

func (embedder *countingEmbedder) Health(_ context.Context) error { return nil }

func TestEmbedChunkBatchReusesByContentAndEmbedsOnlyMisses(t *testing.T) {
	embedder := &countingEmbedder{}
	service := &Service{embedder: embedder}

	chunks := []model.StoredChunk{
		{Content: "reused-A"},
		{Content: "fresh-B"},
		{Content: "reused-C"},
	}
	reuse := map[string][]float32{
		contentVectorKey("reused-A"): {7, 7},
		contentVectorKey("reused-C"): {9, 9},
	}

	vectors, reused, err := service.embedChunkBatch(context.Background(), chunks, reuse)
	if err != nil {
		t.Fatalf("embedChunkBatch returned error: %v", err)
	}
	if len(vectors) != 3 {
		t.Fatalf("got %d vectors, want 3", len(vectors))
	}
	if reused != 2 {
		t.Fatalf("reused = %d, want 2 (reused-A and reused-C)", reused)
	}
	if !slices.Equal(vectors[0], []float32{7, 7}) {
		t.Fatalf("vectors[0] = %v, want the reused {7,7}", vectors[0])
	}
	if !slices.Equal(vectors[2], []float32{9, 9}) {
		t.Fatalf("vectors[2] = %v, want the reused {9,9}", vectors[2])
	}
	// Only the single miss ("fresh-B") may reach the embedder, in one batch.
	if len(embedder.batches) != 1 {
		t.Fatalf("embedder called %d times, want 1", len(embedder.batches))
	}
	if want := []string{"fresh-B"}; !slices.Equal(embedder.batches[0], want) {
		t.Fatalf("embedded batch = %v, want %v (reused chunks must not be embedded)", embedder.batches[0], want)
	}
	// The embedded miss lands in its original position with the embedder's vector.
	if !slices.Equal(vectors[1], []float32{float32(len("fresh-B"))}) {
		t.Fatalf("vectors[1] = %v, want the embedded vector for fresh-B", vectors[1])
	}
}

func TestEmbedChunkBatchAllReusedSkipsEmbedderEntirely(t *testing.T) {
	embedder := &countingEmbedder{}
	service := &Service{embedder: embedder}

	chunks := []model.StoredChunk{{Content: "x"}, {Content: "y"}}
	reuse := map[string][]float32{
		contentVectorKey("x"): {1},
		contentVectorKey("y"): {2},
	}

	vectors, reused, err := service.embedChunkBatch(context.Background(), chunks, reuse)
	if err != nil {
		t.Fatalf("embedChunkBatch returned error: %v", err)
	}
	if reused != 2 {
		t.Fatalf("reused = %d, want 2 (every chunk reused)", reused)
	}
	if len(embedder.batches) != 0 {
		t.Fatalf("embedder was called %d time(s) for an all-reuse batch, want 0", len(embedder.batches))
	}
	if !slices.Equal(vectors[0], []float32{1}) || !slices.Equal(vectors[1], []float32{2}) {
		t.Fatalf("reused vectors not returned in order: %v", vectors)
	}
}

func TestEmbedChunkBatchNoReuseEmbedsEverything(t *testing.T) {
	embedder := &countingEmbedder{}
	service := &Service{embedder: embedder}

	chunks := []model.StoredChunk{{Content: "a"}, {Content: "bb"}}
	vectors, reused, err := service.embedChunkBatch(context.Background(), chunks, nil)
	if err != nil {
		t.Fatalf("embedChunkBatch returned error: %v", err)
	}
	if reused != 0 {
		t.Fatalf("reused = %d, want 0 (no reuse map)", reused)
	}
	if len(embedder.batches) != 1 || len(embedder.batches[0]) != 2 {
		t.Fatalf("embedder batches = %v, want one batch of 2", embedder.batches)
	}
	if !slices.Equal(vectors[0], []float32{1}) || !slices.Equal(vectors[1], []float32{2}) {
		t.Fatalf("vectors = %v, want lengths of the inputs", vectors)
	}
}

func TestBuildRelativePathPrefixFilterMatchesSubtree(t *testing.T) {
	if got := buildRelativePathPrefixFilter(""); got != "" {
		t.Fatalf("empty prefix produced a clause %q, want none", got)
	}
	if got := buildRelativePathPrefixFilter("."); got != "" {
		t.Fatalf("root prefix produced a clause %q, want none", got)
	}
	got := buildRelativePathPrefixFilter("codex-rs")
	want := `(relativePath == "codex-rs" or relativePath like "codex-rs/%")`
	if got != want {
		t.Fatalf("prefix filter = %q, want %q", got, want)
	}
}

func TestRelativePathExpressionMatchesOnlyExactPath(t *testing.T) {
	got := relativePathExpression(`src/file.go`)
	want := `relativePath == "src/file.go"`
	if got != want {
		t.Fatalf("relativePathExpression = %q, want %q", got, want)
	}
	if got == relativePathPrefixExpression(`src/file.go`) {
		t.Fatalf("exact-path expression matched prefix expression %q; code reuse must not read prefix neighbors", got)
	}
	quoted := relativePathExpression(`src/"quoted".go`)
	quotedWant := `relativePath == "src/\"quoted\".go"`
	if quoted != quotedWant {
		t.Fatalf("escaped exact-path expression = %q, want %q", quoted, quotedWant)
	}
}

func TestBuildSearchFilterCombinesExtensionAndPrefix(t *testing.T) {
	got := buildSearchFilter([]string{".go"}, []string{"codex-rs"})
	want := `fileExtension in [".go"] and (relativePath == "codex-rs" or relativePath like "codex-rs/%")`
	if got != want {
		t.Fatalf("combined filter = %q, want %q", got, want)
	}
	if got := buildSearchFilter(nil, nil); got != "" {
		t.Fatalf("no extension and no prefix produced %q, want empty", got)
	}
}

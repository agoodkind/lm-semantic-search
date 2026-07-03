package semantic

import (
	"context"
	"slices"
	"strings"
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

func TestReusePrefixFilterEscapesNewlineConversationPrefix(t *testing.T) {
	prefix := "conv/cursor:task-call_0mtc\nfc_00729/"

	got := relativePathPrefixExpression(prefix)
	// A literal wildcard reaches the parser as an escaped backslash plus the
	// wildcard, because the lexer has no \_ or \% escape sequence.
	want := `relativePath like "conv/cursor:task-call\\_0mtc\nfc\\_00729/%"`

	if got != want {
		t.Fatalf("reuse prefix expression = %q, want %q", got, want)
	}
	assertNoRawControlBytes(t, got)
}

func TestRelativePathExpressionsEscapeMilvusStringBytes(t *testing.T) {
	tests := []struct {
		name       string
		exactPath  string
		exactWant  string
		prefix     string
		prefixWant string
		filter     string
		filterWant string
	}{
		{
			name:       "newline",
			exactPath:  "conv/cursor:task-call_0mtc\nfc_00729/0",
			exactWant:  `relativePath == "conv/cursor:task-call_0mtc\nfc_00729/0"`,
			prefix:     "conv/cursor:task-call_0mtc\nfc_00729/",
			prefixWant: `relativePath like "conv/cursor:task-call\\_0mtc\nfc\\_00729/%"`,
			filter:     "conv/cursor:task-call_0mtc\nfc_00729",
			filterWant: `(relativePath == "conv/cursor:task-call_0mtc\nfc_00729" or relativePath like "conv/cursor:task-call\\_0mtc\nfc\\_00729/%")`,
		},
		{
			name:       "double quote",
			exactPath:  `conv/cursor:"quoted"/0`,
			exactWant:  `relativePath == "conv/cursor:\"quoted\"/0"`,
			prefix:     `conv/cursor:"quoted"/`,
			prefixWant: `relativePath like "conv/cursor:\"quoted\"/%"`,
			filter:     `conv/cursor:"quoted"`,
			filterWant: `(relativePath == "conv/cursor:\"quoted\"" or relativePath like "conv/cursor:\"quoted\"/%")`,
		},
		{
			name:       "backslash",
			exactPath:  `conv/cursor:\slash/0`,
			exactWant:  `relativePath == "conv/cursor:\\slash/0"`,
			prefix:     `conv/cursor:\slash/`,
			prefixWant: `relativePath like "conv/cursor:\\slash/%"`,
			filter:     `conv/cursor:\slash`,
			filterWant: `(relativePath == "conv/cursor:\\slash" or relativePath like "conv/cursor:\\slash/%")`,
		},
		{
			name:       "tab",
			exactPath:  "conv/cursor:\tthread/0",
			exactWant:  `relativePath == "conv/cursor:\tthread/0"`,
			prefix:     "conv/cursor:\tthread/",
			prefixWant: `relativePath like "conv/cursor:\tthread/%"`,
			filter:     "conv/cursor:\tthread",
			filterWant: `(relativePath == "conv/cursor:\tthread" or relativePath like "conv/cursor:\tthread/%")`,
		},
		{
			name:       "percent",
			exactPath:  "conv/cursor:100%/0",
			exactWant:  `relativePath == "conv/cursor:100%/0"`,
			prefix:     "conv/cursor:100%/",
			prefixWant: `relativePath like "conv/cursor:100\\%/%"`,
			filter:     "conv/cursor:100%",
			filterWant: `(relativePath == "conv/cursor:100%" or relativePath like "conv/cursor:100\\%/%")`,
		},
		{
			name:       "other control byte",
			exactPath:  "conv/cursor:\x01thread/0",
			exactWant:  `relativePath == "conv/cursor:\001thread/0"`,
			prefix:     "conv/cursor:\x01thread/",
			prefixWant: `relativePath like "conv/cursor:\001thread/%"`,
			filter:     "conv/cursor:\x01thread",
			filterWant: `(relativePath == "conv/cursor:\001thread" or relativePath like "conv/cursor:\001thread/%")`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			exact := relativePathExpression(test.exactPath)
			if exact != test.exactWant {
				t.Fatalf("relativePathExpression = %q, want %q", exact, test.exactWant)
			}
			prefix := relativePathPrefixExpression(test.prefix)
			if prefix != test.prefixWant {
				t.Fatalf("relativePathPrefixExpression = %q, want %q", prefix, test.prefixWant)
			}
			filter := buildRelativePathPrefixFilter(test.filter)
			if filter != test.filterWant {
				t.Fatalf("buildRelativePathPrefixFilter = %q, want %q", filter, test.filterWant)
			}
			assertNoRawControlBytes(t, exact)
			assertNoRawControlBytes(t, prefix)
			assertNoRawControlBytes(t, filter)
		})
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

func assertNoRawControlBytes(t *testing.T, expression string) {
	t.Helper()
	if strings.ContainsAny(expression, "\n\r\t") {
		t.Fatalf("expression contains a raw newline, carriage return, or tab: %q", expression)
	}
	for index := range len(expression) {
		if expression[index] < 0x20 {
			t.Fatalf("expression contains raw control byte 0x%02x: %q", expression[index], expression)
		}
	}
}

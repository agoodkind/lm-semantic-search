// Package importfixture imports the published lm-semantic-search library as an
// external module and exercises it through its public API only.
package importfixture

import (
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/embedded"
	"goodkind.io/lm-semantic-search/library/embedding"
	"goodkind.io/lm-semantic-search/library/embedding/onnx"
)

const (
	documentPrefix = "passage: "
	modelName      = "bge-small"
	// sharedModelCache is the os.TempDir entry that the LMS offline model tests
	// fill with the pinned bge-small artifacts.
	sharedModelCache = "lm-semantic-search-offline-model-test-cache"
	shortText        = "The shared search library prepares one short text."
	longSentence     = "The shared search library splits long source text at the model token limit. "
)

func newTokenizer(t *testing.T) *onnx.Tokenizer {
	t.Helper()
	tokenizer, err := onnx.NewTokenizer(t.Context(), onnx.Config{
		ModelName:      modelName,
		ModelCacheRoot: filepath.Join(os.TempDir(), sharedModelCache),
	})
	if err != nil {
		t.Fatalf("onnx.NewTokenizer: %v", err)
	}
	return tokenizer
}

func countTokens(t *testing.T, tokenizer *onnx.Tokenizer, text string) int {
	t.Helper()
	count, err := tokenizer.CountTokens(t.Context(), text)
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	return count
}

func prepare(t *testing.T, tokenizer *onnx.Tokenizer, text string) []library.PreparedPart {
	t.Helper()
	parts, err := library.PrepareText(t.Context(), library.PrepareRequest{
		Text:           text,
		DocumentPrefix: documentPrefix,
		MaxTokens:      tokenizer.MaxTokens(),
		MaxBytes:       tokenizer.MaxInputBytes(),
		Tokenizer:      tokenizer,
	})
	if err != nil {
		t.Fatalf("PrepareText: %v", err)
	}
	return parts
}

func TestPrepareTextReturnsOnePartBelowTheTokenLimit(t *testing.T) {
	tokenizer := newTokenizer(t)
	if count := countTokens(t, tokenizer, documentPrefix+shortText); count > tokenizer.MaxTokens() {
		t.Fatalf("short input measures %d tokens, over the %d-token limit", count, tokenizer.MaxTokens())
	}

	parts := prepare(t, tokenizer, shortText)
	if len(parts) != 1 {
		t.Fatalf("PrepareText returned %d parts, want 1", len(parts))
	}
	part := parts[0]
	if part.Suffix != "0" {
		t.Fatalf("Suffix = %q, want %q", part.Suffix, "0")
	}
	if part.ByteStart != 0 || part.ByteEnd != len(shortText) {
		t.Fatalf("span = [%d,%d), want [0,%d)", part.ByteStart, part.ByteEnd, len(shortText))
	}
	if part.EmbeddingInput != documentPrefix+shortText {
		t.Fatalf("EmbeddingInput = %q, want %q", part.EmbeddingInput, documentPrefix+shortText)
	}
}

func TestPrepareTextSplitsAboveTheTokenLimit(t *testing.T) {
	tokenizer := newTokenizer(t)
	maxTokens := tokenizer.MaxTokens()
	text := strings.TrimSpace(strings.Repeat(longSentence, maxTokens/4))
	if count := countTokens(t, tokenizer, documentPrefix+text); count <= maxTokens {
		t.Fatalf("long input measures %d tokens, want more than the %d-token limit", count, maxTokens)
	}

	parts := prepare(t, tokenizer, text)
	if len(parts) < 2 {
		t.Fatalf("PrepareText returned %d parts, want at least 2", len(parts))
	}
	next := 0
	for ordinal, part := range parts {
		if part.Suffix != strconv.Itoa(ordinal) {
			t.Fatalf("part %d Suffix = %q, want %q", ordinal, part.Suffix, strconv.Itoa(ordinal))
		}
		if part.ByteStart != next {
			t.Fatalf("part %d ByteStart = %d, want %d", ordinal, part.ByteStart, next)
		}
		if part.ByteEnd <= part.ByteStart {
			t.Fatalf("part %d span [%d,%d) is empty", ordinal, part.ByteStart, part.ByteEnd)
		}
		want := documentPrefix + text[part.ByteStart:part.ByteEnd]
		if part.EmbeddingInput != want {
			t.Fatalf("part %d EmbeddingInput is not the document prefix plus its span", ordinal)
		}
		if count := countTokens(t, tokenizer, part.EmbeddingInput); count > maxTokens {
			t.Fatalf("part %d measures %d tokens, over the %d-token limit", ordinal, count, maxTokens)
		}
		next = part.ByteEnd
	}
	if next != len(text) {
		t.Fatalf("parts end at byte %d, want %d", next, len(text))
	}
}

func TestNewOpenAIRejectsAnEmptyBaseURL(t *testing.T) {
	embedder, err := embedding.NewOpenAI(t.Context(), embedding.OpenAIConfig{
		BaseURL:   "",
		Model:     "text-embedding-3-small",
		Dimension: 384,
	})
	if !errors.Is(err, library.ErrInvalidRequest) {
		t.Fatalf("NewOpenAI error = %v, want library.ErrInvalidRequest", err)
	}
	if embedder != nil {
		t.Fatal("NewOpenAI returned an embedder with an error")
	}
}

// bgeSmallDimension is the vector width of the bge-small model.
const bgeSmallDimension = 384

func TestOpenOverTheEmbeddedPoolAppliesAndReplays(t *testing.T) {
	root := t.TempDir()
	embedder, err := onnx.New(t.Context(), onnx.Config{
		ModelName:      modelName,
		ModelCacheRoot: filepath.Join(os.TempDir(), sharedModelCache),
	})
	if err != nil {
		t.Fatalf("onnx.New: %v", err)
	}
	pool, err := embedded.New(embedded.Config{Root: filepath.Join(root, "pool")})
	if err != nil {
		t.Fatalf("embedded.New: %v", err)
	}
	opened, err := library.Open(t.Context(), library.Config{
		Store: library.StoreDescriptor{
			CatalogPath:       filepath.Join(root, "catalog.sqlite"),
			LockPath:          filepath.Join(root, "catalog.lock"),
			PoolID:            "import-fixture",
			EmbeddingModel:    modelName,
			EmbeddingRevision: "import-fixture",
			Dimension:         bgeSmallDimension,
			Normalization:     "none",
		},
		Vectors:  pool,
		Embedder: embedder,
	})
	if err != nil {
		t.Fatalf("library.Open: %v", err)
	}
	t.Cleanup(func() {
		if err := opened.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
	if err := opened.RegisterNamespace(t.Context(), library.NamespaceSpec{ID: "notes", Policy: library.AppendOnly}); err != nil {
		t.Fatalf("RegisterNamespace: %v", err)
	}
	batch := library.Batch{
		Namespace:        "notes",
		OwnerID:          "note-1",
		GenerationOrder:  1,
		IdempotencyToken: "first",
		Mode:             library.Append,
		Rows: []library.Occurrence{{
			RowKey:         "line-1",
			SortKey:        "1",
			SourceText:     shortText,
			SearchText:     shortText,
			EmbeddingInput: documentPrefix + shortText,
		}},
	}
	receipt, err := opened.Apply(t.Context(), batch)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	state, err := opened.GetOwnerState(t.Context(), "notes", "note-1")
	if err != nil || state.GenerationOrder != 1 || state.Fingerprint != receipt.Fingerprint {
		t.Fatalf("GetOwnerState = %+v, %v, want order 1 with fingerprint %s", state, err, receipt.Fingerprint)
	}
	if replay, err := opened.Apply(t.Context(), batch); err != nil || replay != receipt {
		t.Fatalf("replay = %+v, %v, want %+v", replay, err, receipt)
	}
}

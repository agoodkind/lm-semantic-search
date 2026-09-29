package library_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"

	"goodkind.io/lm-semantic-search/internal/config"
	internalonnx "goodkind.io/lm-semantic-search/internal/embedding/onnx"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/embedding"
	"goodkind.io/lm-semantic-search/library/embedding/onnx"
)

// testDocumentPrefix stands in for a model's document role prefix.
const testDocumentPrefix = "passage: "

// offlineONNXConfig selects the bge-small preset in the model cache that every
// offline model test shares. The pinned artifacts download once per machine.
func offlineONNXConfig(t *testing.T) onnx.Config {
	t.Helper()
	root := filepath.Join(os.TempDir(), "lm-semantic-search-offline-model-test-cache")
	if err := os.MkdirAll(root, 0o700); err != nil {
		t.Fatalf("create shared offline model cache %s: %v", root, err)
	}
	return onnx.Config{ModelName: offlinemodel.BGESmall, ModelCacheRoot: root}
}

// realTokenizer returns the bge-small tokenizer. It skips only when the pinned
// artifact host is unreachable, as the internal ONNX tests do.
func realTokenizer(t *testing.T) *onnx.Tokenizer {
	t.Helper()
	tokenizer, err := onnx.NewTokenizer(context.Background(), offlineONNXConfig(t))
	if errors.Is(err, internalonnx.ErrArtifactUnavailable) {
		t.Skipf("offline embedding artifact unavailable: %v", err)
	}
	if err != nil {
		t.Fatalf("NewONNXTokenizer: %v", err)
	}
	return tokenizer
}

func countTokens(t *testing.T, tokenizer library.Tokenizer, text string) int {
	t.Helper()
	count, err := tokenizer.CountTokens(context.Background(), text)
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	return count
}

// wordsWithTokenCount returns repeated words. The real tokenizer measures the
// prefixed words at exactly want tokens.
func wordsWithTokenCount(t *testing.T, tokenizer library.Tokenizer, want int) string {
	t.Helper()
	for words := 1; words <= want; words++ {
		text := strings.TrimSpace(strings.Repeat("hello ", words))
		count := countTokens(t, tokenizer, testDocumentPrefix+text)
		if count == want {
			return text
		}
		if count > want {
			break
		}
	}
	t.Fatalf("no repeated-word text measures exactly %d tokens", want)
	return ""
}

// assertCompleteParts checks that parts cover every byte of text in order,
// that each part's embedding input is the prefix plus its span, and that each
// input fits the token limit.
func assertCompleteParts(
	t *testing.T,
	tokenizer library.Tokenizer,
	maxTokens int,
	text string,
	parts []library.PreparedPart,
) {
	t.Helper()
	var covered strings.Builder
	next := 0
	for ordinal, part := range parts {
		if part.Suffix != strconv.Itoa(ordinal) {
			t.Fatalf("part %d Suffix = %q, want %q", ordinal, part.Suffix, strconv.Itoa(ordinal))
		}
		if part.ByteStart != next {
			t.Fatalf("part %d ByteStart = %d, want %d", ordinal, part.ByteStart, next)
		}
		span := text[part.ByteStart:part.ByteEnd]
		if !utf8.ValidString(span) {
			t.Fatalf("part %d span [%d,%d) cuts a UTF-8 character", ordinal, part.ByteStart, part.ByteEnd)
		}
		if part.EmbeddingInput != testDocumentPrefix+span {
			t.Fatalf("part %d EmbeddingInput is not the document prefix plus its span", ordinal)
		}
		if count := countTokens(t, tokenizer, part.EmbeddingInput); count > maxTokens {
			t.Fatalf("part %d measures %d tokens, over the %d-token limit", ordinal, count, maxTokens)
		}
		covered.WriteString(span)
		next = part.ByteEnd
	}
	if covered.String() != text {
		t.Fatalf("parts cover %d of %d bytes", covered.Len(), len(text))
	}
}

func TestPrepareTextSplitsAtTheRealTokenLimit(t *testing.T) {
	tokenizer := realTokenizer(t)
	maxTokens := tokenizer.MaxTokens()

	for _, testCase := range []struct {
		name      string
		tokens    int
		wantParts int
	}{
		{name: "below the limit", tokens: maxTokens - 1, wantParts: 1},
		{name: "at the limit", tokens: maxTokens, wantParts: 1},
		{name: "above the limit", tokens: maxTokens + 1, wantParts: 2},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			text := wordsWithTokenCount(t, tokenizer, testCase.tokens)
			request := library.PrepareRequest{
				Text:           text,
				DocumentPrefix: testDocumentPrefix,
				MaxTokens:      maxTokens,
				MaxBytes:       tokenizer.MaxInputBytes(),
				Tokenizer:      tokenizer,
			}
			parts, err := library.PrepareText(context.Background(), request)
			if err != nil {
				t.Fatalf("PrepareText: %v", err)
			}
			if len(parts) != testCase.wantParts {
				t.Fatalf("PrepareText returned %d parts, want %d", len(parts), testCase.wantParts)
			}
			assertCompleteParts(t, tokenizer, maxTokens, text, parts)

			if len(parts) > 1 {
				// The first cut is the largest fitting boundary: one more
				// character exceeds the limit.
				first := parts[0]
				_, size := utf8.DecodeRuneInString(text[first.ByteEnd:])
				longer := testDocumentPrefix + text[:first.ByteEnd+size]
				if count := countTokens(t, tokenizer, longer); count <= maxTokens {
					t.Fatalf("first part ends at byte %d, but one more character still fits at %d tokens", first.ByteEnd, count)
				}
			}

			again, err := library.PrepareText(context.Background(), request)
			if err != nil {
				t.Fatalf("PrepareText again: %v", err)
			}
			if !reflect.DeepEqual(parts, again) {
				t.Fatal("PrepareText returned different parts for the same request")
			}
		})
	}
}

func TestPrepareTextKeepsMultibyteCharactersWhole(t *testing.T) {
	tokenizer := realTokenizer(t)
	maxTokens := tokenizer.MaxTokens()
	text := strings.Repeat("naïve café 数据 🚀 ", maxTokens)
	parts, err := library.PrepareText(context.Background(), library.PrepareRequest{
		Text:           text,
		DocumentPrefix: testDocumentPrefix,
		MaxTokens:      maxTokens,
		MaxBytes:       tokenizer.MaxInputBytes(),
		Tokenizer:      tokenizer,
	})
	if err != nil {
		t.Fatalf("PrepareText: %v", err)
	}
	if len(parts) < 2 {
		t.Fatalf("PrepareText returned %d parts, want a split", len(parts))
	}
	assertCompleteParts(t, tokenizer, maxTokens, text, parts)
}

func TestPrepareTextWithoutTokenizerUsesTheConservativeByteBudget(t *testing.T) {
	t.Parallel()
	const maxTokens = 4096
	byteBudget := config.EmbedChunkByteBudgetForLimit(0, maxTokens)
	text := strings.Repeat("func alpha() { return beta }\n", 3*byteBudget/30)
	parts, err := library.PrepareText(context.Background(), library.PrepareRequest{
		Text:           text,
		DocumentPrefix: testDocumentPrefix,
		MaxTokens:      maxTokens,
		MaxBytes:       0,
		Tokenizer:      nil,
	})
	if err != nil {
		t.Fatalf("PrepareText: %v", err)
	}
	if len(parts) < 3 {
		t.Fatalf("PrepareText returned %d parts, want at least 3", len(parts))
	}
	var covered strings.Builder
	for ordinal, part := range parts {
		if len(part.EmbeddingInput) > byteBudget {
			t.Fatalf("part %d input is %d bytes, over the %d-byte budget", ordinal, len(part.EmbeddingInput), byteBudget)
		}
		if part.EmbeddingInput != testDocumentPrefix+text[part.ByteStart:part.ByteEnd] {
			t.Fatalf("part %d EmbeddingInput is not the document prefix plus its span", ordinal)
		}
		covered.WriteString(text[part.ByteStart:part.ByteEnd])
	}
	if covered.String() != text {
		t.Fatalf("parts cover %d of %d bytes", covered.Len(), len(text))
	}
}

func TestPrepareTextSkipsOnlyWhitespaceSpans(t *testing.T) {
	t.Parallel()
	const maxBytes = 16
	text := "alpha" + strings.Repeat(" ", 40) + "omega"
	parts, err := library.PrepareText(context.Background(), library.PrepareRequest{
		Text:      text,
		MaxBytes:  maxBytes,
		MaxTokens: 0,
	})
	if err != nil {
		t.Fatalf("PrepareText: %v", err)
	}
	var nonWhitespace strings.Builder
	for ordinal, part := range parts {
		if part.Suffix != strconv.Itoa(ordinal) {
			t.Fatalf("part %d Suffix = %q, want consecutive ordinals", ordinal, part.Suffix)
		}
		if strings.TrimSpace(part.EmbeddingInput) == "" {
			t.Fatalf("part %d contains only whitespace", ordinal)
		}
		nonWhitespace.WriteString(strings.Join(strings.Fields(part.EmbeddingInput), ""))
	}
	if nonWhitespace.String() != "alphaomega" {
		t.Fatalf("parts contain %q, want every non-whitespace byte once", nonWhitespace.String())
	}
}

func TestPrepareTextRejectsInvalidRequests(t *testing.T) {
	t.Parallel()
	for _, testCase := range []struct {
		name    string
		request library.PrepareRequest
		want    string
	}{
		{name: "whitespace text", request: library.PrepareRequest{Text: " \n\t", MaxBytes: 10}, want: "no non-whitespace"},
		{name: "invalid utf-8", request: library.PrepareRequest{Text: "a\xffb", MaxBytes: 10}, want: "not valid UTF-8"},
		{name: "no limit", request: library.PrepareRequest{Text: "alpha"}, want: "set MaxTokens, MaxBytes, or both"},
		{name: "negative limit", request: library.PrepareRequest{Text: "alpha", MaxTokens: -1}, want: "must not be negative"},
		{name: "prefix fills budget", request: library.PrepareRequest{Text: "alpha", DocumentPrefix: "0123456789", MaxBytes: 10}, want: "leaves no room"},
		{name: "character over budget", request: library.PrepareRequest{Text: "🚀", MaxBytes: 3}, want: "does not fit"},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			parts, err := library.PrepareText(context.Background(), testCase.request)
			if parts != nil {
				t.Fatalf("PrepareText returned %d parts with an error", len(parts))
			}
			assertInvalidRequest(t, err, testCase.want)
		})
	}
}

// TestPreparedPartsFailWholeWhenTheProviderRefusesOne embeds prepared parts
// with the production ONNX provider. A part the provider refuses fails the
// whole batch with a typed rejection and returns no vectors.
func TestPreparedPartsFailWholeWhenTheProviderRefusesOne(t *testing.T) {
	tokenizer := realTokenizer(t)
	embedder, err := onnx.New(context.Background(), offlineONNXConfig(t))
	if err != nil {
		t.Fatalf("NewONNX: %v", err)
	}
	prepare := func(text string) []library.PreparedPart {
		parts, prepareErr := library.PrepareText(context.Background(), library.PrepareRequest{
			Text:           text,
			DocumentPrefix: testDocumentPrefix,
			MaxTokens:      tokenizer.MaxTokens(),
			MaxBytes:       tokenizer.MaxInputBytes(),
			Tokenizer:      tokenizer,
		})
		if prepareErr != nil {
			t.Fatalf("PrepareText: %v", prepareErr)
		}
		return parts
	}

	accepted := prepare(wordsWithTokenCount(t, tokenizer, tokenizer.MaxTokens()+40))
	inputs := make([]string, 0, len(accepted))
	for _, part := range accepted {
		inputs = append(inputs, part.EmbeddingInput)
	}
	vectors, err := embedder.EmbedBatch(context.Background(), inputs)
	if err != nil {
		t.Fatalf("EmbedBatch(accepted parts): %v", err)
	}
	if len(vectors) != len(inputs) {
		t.Fatalf("EmbedBatch returned %d vectors for %d parts", len(vectors), len(inputs))
	}
	for index, vector := range vectors {
		if len(vector) != 384 {
			t.Fatalf("vector %d has %d values, want 384", index, len(vector))
		}
	}

	mixed := append(append([]string{}, inputs...), testDocumentPrefix+"alpha\x00beta")
	vectors, err = embedder.EmbedBatch(context.Background(), mixed)
	if !errors.Is(err, embedding.ErrEmbedderRejected) {
		t.Fatalf("EmbedBatch(with refused part) error = %v, want ErrEmbedderRejected", err)
	}
	if vectors != nil {
		t.Fatalf("EmbedBatch returned %d vectors with an error, want none", len(vectors))
	}
}

//go:build live

package live

import (
	"context"
	"encoding/base64"
	"encoding/hex"
	"math/rand/v2"
	"strings"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/library"
)

// denseSampleBytes sizes each dense sample to several parts at the 4096-token
// model limit.
const denseSampleBytes = 10000

// denseSamples returns texts that pack many tokens into few bytes for the
// production embedding model: random hex and base64, emoji and Deseret letters
// outside the vocabulary, and CJK text.
func denseSamples() map[string]string {
	generator := rand.New(rand.NewPCG(20260929, 1))
	randomBytes := make([]byte, denseSampleBytes)
	for index := range randomBytes {
		randomBytes[index] = byte(generator.UintN(256))
	}
	runes := func(first rune, span uint, count int) string {
		var builder strings.Builder
		for range count {
			builder.WriteRune(first + rune(generator.UintN(span)))
		}
		return builder.String()
	}
	return map[string]string{
		"hex":     hex.EncodeToString(randomBytes)[:denseSampleBytes],
		"base64":  base64.StdEncoding.EncodeToString(randomBytes)[:denseSampleBytes],
		"emoji":   runes(0x1F600, 80, denseSampleBytes/4),
		"deseret": runes(0x10400, 80, denseSampleBytes/4),
		"cjk":     runes(0x4E00, 20000, denseSampleBytes/3),
	}
}

// TestLibraryWritePreparedPartsFitTheEmbeddingModel prepares dense text
// without a tokenizer at the production model limit and embeds every part
// through the production endpoint. The endpoint refuses a whole batch when any
// part exceeds the model limit.
func TestLibraryWritePreparedPartsFitTheEmbeddingModel(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), libraryLiveTimeout)
	t.Cleanup(cancel)
	environment := resolveLibraryEnvironment(t)
	embedder := newLiveEmbedder(t, ctx, environment)
	t.Logf("embedding window start %s", time.Now().UTC().Format(time.RFC3339))
	defer func() { t.Logf("embedding window end %s", time.Now().UTC().Format(time.RFC3339)) }()

	const documentPrefix = "read_file\n"
	for name, text := range denseSamples() {
		t.Run(name, func(t *testing.T) {
			parts, err := library.PrepareText(ctx, library.PrepareRequest{
				Text:           text,
				DocumentPrefix: documentPrefix,
				MaxTokens:      libraryLiveModelTokenLimit,
				MaxBytes:       0,
				Tokenizer:      nil,
			})
			if err != nil {
				t.Fatalf("PrepareText: %v", err)
			}
			if len(parts) < 2 {
				t.Fatalf("PrepareText returned %d parts for %d bytes, want a split", len(parts), len(text))
			}
			inputs := make([]string, 0, len(parts))
			for _, part := range parts {
				inputs = append(inputs, part.EmbeddingInput)
			}
			vectors, err := embedder.EmbedBatch(ctx, inputs)
			if err != nil {
				t.Fatalf("embed %d prepared parts: %v", len(inputs), err)
			}
			if len(vectors) != len(inputs) {
				t.Fatalf("embedder returned %d vectors for %d parts", len(vectors), len(inputs))
			}
		})
	}
}

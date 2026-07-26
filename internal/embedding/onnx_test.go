package embedding

import (
	"context"
	"errors"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

// skipIfArtifactUnavailable skips a test that needs a downloaded offline model
// artifact when the remote host is unreachable. The project does not assume
// network access, so a fetch failure is an environment skip, not a test error.
func skipIfArtifactUnavailable(t *testing.T, err error) {
	t.Helper()
	if errors.Is(err, ErrArtifactUnavailable) {
		t.Skipf("offline embedding artifact unavailable: %v", err)
	}
}

// offlineModelCacheRoot returns a state root shared by every test in this
// package, so the pinned model artifacts download once per machine instead of
// once per test. Each artifact is checksum-verified on use, so a partial or
// stale file is replaced rather than trusted.
func offlineModelCacheRoot(t *testing.T) string {
	t.Helper()
	root := filepath.Join(os.TempDir(), "lm-semantic-search-offline-model-test-cache")
	if err := os.MkdirAll(root, 0o700); err != nil {
		t.Fatalf("create shared offline model cache %s: %v", root, err)
	}
	return root
}

func TestONNXBGEProviderDeterministicNormalizedAndConfigured(t *testing.T) {
	cfg := config.ApplyProfile(config.Config{
		Profile:               config.ProfileOffline,
		OfflineEmbeddingModel: offlinemodel.BGESmall,
		StateRoot:             offlineModelCacheRoot(t),
	})
	provider, err := NewProvider(context.Background(), cfg)
	if err != nil {
		skipIfArtifactUnavailable(t, err)
		t.Fatalf("NewProvider: %v", err)
	}
	if provider.ProviderName() != config.EmbeddingProviderONNX {
		t.Fatalf(
			"ProviderName = %q, want %q",
			provider.ProviderName(),
			config.EmbeddingProviderONNX,
		)
	}

	first, err := provider.Embed(context.Background(), "package main\nfunc main() {}")
	if err != nil {
		t.Fatalf("embed first: %v", err)
	}
	second, err := provider.Embed(context.Background(), "package main\nfunc main() {}")
	if err != nil {
		t.Fatalf("embed second: %v", err)
	}
	if len(first) != 384 {
		t.Fatalf("dimension = %d, want 384", len(first))
	}
	if !slices.Equal(first, second) {
		t.Fatal("repeated ONNX embeddings differ")
	}

	var squaredNorm float64
	for _, value := range first {
		squaredNorm += float64(value) * float64(value)
	}
	if math.Abs(squaredNorm-1.0) > 1e-3 {
		t.Fatalf("squared L2 norm = %v, want 1", squaredNorm)
	}
	if err := provider.Health(context.Background()); err != nil {
		t.Fatalf("Health: %v", err)
	}
}

func TestGenericTokenizerLoadsTokenizerJSONWithStableIDs(t *testing.T) {
	preset, err := offlinemodel.Resolve(offlinemodel.BGESmall)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	tokenizerPath := filepath.Join(t.TempDir(), "tokenizer.json")
	if err := ensureArtifact(
		context.Background(),
		http.DefaultClient,
		preset.TokenizerURL,
		preset.TokenizerSHA256,
		tokenizerPath,
	); err != nil {
		skipIfArtifactUnavailable(t, err)
		t.Fatalf("ensureArtifact: %v", err)
	}
	tokenizer, err := newGenericTokenizer(tokenizerPath, preset.MaximumTokens)
	if err != nil {
		t.Fatalf("newGenericTokenizer: %v", err)
	}
	t.Cleanup(func() {
		if closeErr := tokenizer.Close(); closeErr != nil {
			t.Errorf("Close: %v", closeErr)
		}
	})

	encoded, err := tokenizer.encode("package main\nfunc main() {}")
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	want := []int64{101, 7427, 2364, 4569, 2278, 2364, 1006, 1007, 1063, 1065, 102}
	if !slices.Equal(encoded.inputIDs, want) {
		t.Fatalf("token ids = %v, want %v", encoded.inputIDs, want)
	}
	if len(encoded.attentionMask) != len(encoded.inputIDs) {
		t.Fatalf(
			"attention mask length = %d, want %d",
			len(encoded.attentionMask),
			len(encoded.inputIDs),
		)
	}
}

// loadPresetTokenizer downloads one preset's tokenizer.json into the shared
// cache and loads it at that preset's token limit.
func loadPresetTokenizer(t *testing.T, presetName string) (*genericTokenizer, offlinemodel.Preset) {
	t.Helper()
	preset, err := offlinemodel.Resolve(presetName)
	if err != nil {
		t.Fatalf("Resolve %s: %v", presetName, err)
	}
	tokenizerPath := filepath.Join(offlineModelCacheRoot(t), presetName+"-tokenizer.json")
	if err := ensureArtifact(
		context.Background(),
		http.DefaultClient,
		preset.TokenizerURL,
		preset.TokenizerSHA256,
		tokenizerPath,
	); err != nil {
		skipIfArtifactUnavailable(t, err)
		t.Fatalf("ensureArtifact %s: %v", presetName, err)
	}
	tokenizer, err := newGenericTokenizer(tokenizerPath, preset.MaximumTokens)
	if err != nil {
		t.Fatalf("newGenericTokenizer %s: %v", presetName, err)
	}
	t.Cleanup(func() {
		if closeErr := tokenizer.Close(); closeErr != nil {
			t.Errorf("Close %s: %v", presetName, closeErr)
		}
	})
	return tokenizer, preset
}

func TestGenericTokenizerReportsOverLimitInsteadOfTruncating(t *testing.T) {
	for _, presetName := range offlinemodel.Names() {
		t.Run(presetName, func(t *testing.T) {
			tokenizer, preset := loadPresetTokenizer(t, presetName)

			// One repeated line tokenizes to far more tokens than the preset allows,
			// so the tokenizer must report the overflow rather than cutting the input
			// down to the limit and letting the model embed only the head of it.
			longText := strings.Repeat("func handleRequest(writer http.ResponseWriter) { return }\n", int(preset.MaximumTokens))
			encoded, err := tokenizer.encode(longText)
			if err != nil {
				t.Fatalf("encode long input: %v", err)
			}
			if !encoded.overLimit {
				t.Fatalf("long input reported overLimit = false with %d token ids; the tokenizer truncated it silently", len(encoded.inputIDs))
			}
			if encoded.tokenCount <= int(preset.MaximumTokens) {
				t.Fatalf("reported token count = %d, want more than the %d-token limit", encoded.tokenCount, preset.MaximumTokens)
			}
			if len(encoded.inputIDs) != 0 {
				t.Fatalf("over-limit input still carries %d token ids, which the model could embed as a truncated copy", len(encoded.inputIDs))
			}

			shortEncoded, err := tokenizer.encode("package main\nfunc main() {}")
			if err != nil {
				t.Fatalf("encode short input: %v", err)
			}
			if shortEncoded.overLimit {
				t.Fatal("short input reported as over the limit")
			}
			if shortEncoded.tokenCount != len(shortEncoded.inputIDs) {
				t.Fatalf("short input token count = %d, want %d", shortEncoded.tokenCount, len(shortEncoded.inputIDs))
			}
		})
	}
}

func TestONNXEmbedBatchSkipsOverLimitInputInsteadOfTruncating(t *testing.T) {
	preset, err := offlinemodel.Resolve(offlinemodel.BGESmall)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	cfg := config.ApplyProfile(config.Config{
		Profile:               config.ProfileOffline,
		OfflineEmbeddingModel: offlinemodel.BGESmall,
		StateRoot:             offlineModelCacheRoot(t),
	})
	provider, err := NewProvider(context.Background(), cfg)
	if err != nil {
		skipIfArtifactUnavailable(t, err)
		t.Fatalf("NewProvider: %v", err)
	}

	longText := strings.Repeat("func handleRequest(writer http.ResponseWriter) { return }\n", int(preset.MaximumTokens))
	result, err := provider.EmbedBatch(context.Background(), []string{"package main", longText})
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	if len(result.Vectors) != 2 {
		t.Fatalf("vectors = %d, want 2 (index-aligned with the inputs)", len(result.Vectors))
	}
	if result.Vectors[0] == nil {
		t.Fatal("the input that fits got no vector")
	}
	if result.Vectors[1] != nil {
		t.Fatal("the over-limit input got a vector, so its content was truncated and embedded anyway")
	}
	if len(result.Skipped) != 1 {
		t.Fatalf("skipped = %d, want 1 (the over-limit input)", len(result.Skipped))
	}
	skip := result.Skipped[0]
	if skip.Index != 1 {
		t.Fatalf("skipped index = %d, want 1", skip.Index)
	}
	if skip.Reason != embedCodeContextLengthExceeded {
		t.Fatalf("skipped reason = %q, want %q", skip.Reason, embedCodeContextLengthExceeded)
	}
	if skip.MaxTokens != int(preset.MaximumTokens) {
		t.Fatalf("skipped MaxTokens = %d, want %d", skip.MaxTokens, preset.MaximumTokens)
	}
	if skip.ReportedTokens <= skip.MaxTokens {
		t.Fatalf("skipped ReportedTokens = %d, want more than the %d-token limit", skip.ReportedTokens, skip.MaxTokens)
	}

	// A single-input Embed follows the same rule: an over-limit input is an error,
	// never a vector over a shortened copy of the content.
	if _, embedErr := provider.Embed(context.Background(), longText); embedErr == nil {
		t.Fatal("Embed returned a vector for an over-limit input")
	}
}

func TestPoolAndNormalizeUsesConfiguredPooling(t *testing.T) {
	const dimension = 2
	tokenEmbeddings := []float32{
		3, 4,
		0, 2,
		100, 100,
	}
	attentionMask := []int64{1, 1, 0}

	clsVector, err := poolAndNormalize(
		tokenEmbeddings,
		attentionMask,
		dimension,
		offlinemodel.PoolingCLS,
	)
	if err != nil {
		t.Fatalf("CLS pool: %v", err)
	}
	assertVectorClose(t, clsVector, []float32{0.6, 0.8})

	meanVector, err := poolAndNormalize(
		tokenEmbeddings,
		attentionMask,
		dimension,
		offlinemodel.PoolingMean,
	)
	if err != nil {
		t.Fatalf("mean pool: %v", err)
	}
	inverseNorm := float32(1 / math.Sqrt(5))
	assertVectorClose(t, meanVector, []float32{inverseNorm, 2 * inverseNorm})
}

func assertVectorClose(t *testing.T, got []float32, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("vector length = %d, want %d", len(got), len(want))
	}
	for index := range got {
		if math.Abs(float64(got[index]-want[index])) > 1e-6 {
			t.Fatalf("vector[%d] = %v, want %v", index, got[index], want[index])
		}
	}
}

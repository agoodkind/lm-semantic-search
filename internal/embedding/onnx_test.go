package embedding

import (
	"context"
	"math"
	"net/http"
	"path/filepath"
	"slices"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

func TestONNXBGEProviderDeterministicNormalizedAndConfigured(t *testing.T) {
	cfg := config.ApplyProfile(config.Config{
		Profile:               config.ProfileOffline,
		OfflineEmbeddingModel: offlinemodel.BGESmall,
		StateRoot:             t.TempDir(),
	})
	provider, err := NewProvider(context.Background(), cfg)
	if err != nil {
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

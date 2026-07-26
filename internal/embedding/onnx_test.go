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
	"sync"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/adapterr"
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
			if encoded.rejection != onnxInputOverTokenLimit {
				t.Fatalf("long input reported rejection %q with %d token ids; the tokenizer truncated it silently", encoded.rejection, len(encoded.inputIDs))
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
			if shortEncoded.rejection != onnxInputAccepted {
				t.Fatalf("short input reported rejection %q", shortEncoded.rejection)
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
	if skip.Reason != adapterr.EmbedRejectionContextLengthExceeded {
		t.Fatalf("skipped reason = %q, want %q", skip.Reason, adapterr.EmbedRejectionContextLengthExceeded)
	}
	if skip.MaxTokens != adapterr.ReportedFigure(int(preset.MaximumTokens)) {
		t.Fatalf("skipped MaxTokens = %+v, want a reported %d", skip.MaxTokens, preset.MaximumTokens)
	}
	if !skip.ReportedTokens.Reported || skip.ReportedTokens.Value <= skip.MaxTokens.Value {
		t.Fatalf("skipped ReportedTokens = %+v, want a reported count over the %d-token limit", skip.ReportedTokens, skip.MaxTokens.Value)
	}

	// A single-input Embed follows the same rule: an over-limit input is an error,
	// never a vector over a shortened copy of the content.
	if _, embedErr := provider.Embed(context.Background(), longText); embedErr == nil {
		t.Fatal("Embed returned a vector for an over-limit input")
	}
}

// newUnloadedONNXProvider builds a provider whose runtime carries a tokenizer
// with no loaded binding and no model session, so the only inputs it can answer
// for are the ones it refuses before tokenizing. Any path that reaches the
// binding or the model panics on the nil pointer, which is what makes a clean
// return from these tests evidence that neither was touched.
func newUnloadedONNXProvider(t *testing.T, presetName string) *onnxProvider {
	t.Helper()
	preset, err := offlinemodel.Resolve(presetName)
	if err != nil {
		t.Fatalf("Resolve %s: %v", presetName, err)
	}
	return &onnxProvider{
		runtime: &inProcessONNXRuntime{
			session: nil,
			tokenizer: &genericTokenizer{
				tokenizer:     nil,
				maximumTokens: int(preset.MaximumTokens),
			},
			preset: preset,
			mutex:  sync.Mutex{},
		},
	}
}

func TestONNXRejectsInputWithNULByteInsteadOfEmbeddingItsPrefix(t *testing.T) {
	provider := newUnloadedONNXProvider(t, offlinemodel.BGESmall)
	// The tokenizer binding passes the text as a NUL-terminated C string, so an
	// input like this one would measure and embed only "trusted prefix" while the
	// caller stored that vector under the whole input's identity.
	const nulInput = "trusted prefix\x00omitted tail"

	vector, err := provider.Embed(context.Background(), nulInput)
	if err == nil {
		t.Fatal("Embed returned a vector for an input containing a NUL byte; that vector could only cover the prefix before it")
	}
	if vector != nil {
		t.Fatalf("Embed returned %d values alongside the rejection", len(vector))
	}

	result, batchErr := provider.EmbedBatch(context.Background(), []string{nulInput})
	if batchErr != nil {
		t.Fatalf("EmbedBatch: %v", batchErr)
	}
	if len(result.Vectors) != 1 {
		t.Fatalf("vectors = %d, want 1 (index-aligned with the inputs)", len(result.Vectors))
	}
	if result.Vectors[0] != nil {
		t.Fatal("the NUL-carrying input got a vector, so its prefix was embedded anyway")
	}
	if len(result.Skipped) != 1 {
		t.Fatalf("skipped = %d, want 1; the input must be reported, never silently dropped", len(result.Skipped))
	}
	if result.Skipped[0].Reason != adapterr.EmbedRejectionInputContainsNUL {
		t.Fatalf("skipped reason = %q, want %q", result.Skipped[0].Reason, adapterr.EmbedRejectionInputContainsNUL)
	}
	if result.Skipped[0].Index != 0 {
		t.Fatalf("skipped index = %d, want 0", result.Skipped[0].Index)
	}
}

func TestONNXEmbedRejectsNULInputThatSharesAPrefixWithAnEmbeddableOne(t *testing.T) {
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

	const prefix = "package main\nfunc main() {}"
	prefixVector, err := provider.Embed(context.Background(), prefix)
	if err != nil {
		t.Fatalf("embed the prefix on its own: %v", err)
	}

	// Same prefix, then a NUL, then content the NUL-terminated binding would never
	// read. A vector here would be the prefix's vector stored under the whole
	// input's identity.
	fullInput := prefix + "\x00type Handler struct{ Name string }"
	fullVector, fullErr := provider.Embed(context.Background(), fullInput)
	if fullErr == nil {
		if slices.Equal(fullVector, prefixVector) {
			t.Fatal("Embed returned the prefix's own vector for an input whose tail follows a NUL byte")
		}
		t.Fatal("Embed returned a vector for an input containing a NUL byte")
	}
	if fullVector != nil {
		t.Fatalf("Embed returned %d values alongside the rejection", len(fullVector))
	}
}

func TestONNXEmbedRejectsOversizedInputWithoutTakingTheRuntimeLock(t *testing.T) {
	provider := newUnloadedONNXProvider(t, offlinemodel.BGESmall)
	oversized := strings.Repeat("a", provider.runtime.tokenizer.maximumInputBytes()+1)

	// Hold the runtime lock for the whole call. An implementation that tokenizes
	// before checking the input's size blocks here, which is the contention this
	// bound exists to prevent: one oversized query would otherwise stall every
	// other embedding and health probe while its encoding was built and discarded.
	provider.runtime.mutex.Lock()
	defer provider.runtime.mutex.Unlock()

	type embedAttempt struct {
		vector []float32
		err    error
	}
	attempts := make(chan embedAttempt, 1)
	go func() {
		vector, err := provider.Embed(context.Background(), oversized)
		attempts <- embedAttempt{vector: vector, err: err}
	}()

	select {
	case attempt := <-attempts:
		if attempt.err == nil {
			t.Fatal("Embed returned a vector for an input past the tokenizer byte bound")
		}
		if attempt.vector != nil {
			t.Fatalf("Embed returned %d values alongside the rejection", len(attempt.vector))
		}
		if !strings.Contains(attempt.err.Error(), "over the") {
			t.Fatalf("rejection does not name the bound it exceeded: %v", attempt.err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Embed blocked on the runtime lock for an input it must reject before tokenizing")
	}
}

func TestONNXEmbedBatchReportsOversizedInputAsSkipped(t *testing.T) {
	provider := newUnloadedONNXProvider(t, offlinemodel.BGESmall)
	maximumInputBytes := provider.runtime.tokenizer.maximumInputBytes()
	oversized := strings.Repeat("a", maximumInputBytes+1)

	result, err := provider.EmbedBatch(context.Background(), []string{oversized})
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	if len(result.Vectors) != 1 {
		t.Fatalf("vectors = %d, want 1 (index-aligned with the inputs)", len(result.Vectors))
	}
	if result.Vectors[0] != nil {
		t.Fatal("the oversized input got a vector, so part of its content was embedded")
	}
	if len(result.Skipped) != 1 {
		t.Fatalf("skipped = %d, want 1; the input must be reported, never silently dropped", len(result.Skipped))
	}
	if result.Skipped[0].Reason != adapterr.EmbedRejectionInputBytesExceeded {
		t.Fatalf("skipped reason = %q, want %q", result.Skipped[0].Reason, adapterr.EmbedRejectionInputBytesExceeded)
	}
	// The byte ceiling refused this input, not the model's token window, so the
	// skip carries no token limit. Reporting the model's 512 tokens here would
	// name a figure that had nothing to do with the refusal.
	if result.Skipped[0].MaxTokens.Reported {
		t.Fatalf("skipped MaxTokens = %+v, want unreported; the token limit is not the limit that refused this input", result.Skipped[0].MaxTokens)
	}
	if result.Skipped[0].ReportedTokens.Reported {
		t.Fatalf("skipped ReportedTokens = %+v, want unreported; the input was never tokenized", result.Skipped[0].ReportedTokens)
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

package config

import "testing"

func effectiveEmbedTokenCap(maxTokens int) int {
	return EffectiveEmbedTokenCapForLimit(maxTokens, EmbedModelInputTokenLimit)
}

func TestEffectiveEmbedTokenCap(t *testing.T) {
	t.Parallel()
	// int(4096 * 0.9) = 3686, the model-limit backstop after the safety margin.
	modelCap := 3686
	// An unset or non-positive value falls back to the model-limit backstop rather
	// than disabling the split, so an oversized input is divided instead of dropped.
	if got := effectiveEmbedTokenCap(0); got != modelCap {
		t.Fatalf("effectiveEmbedTokenCap(0) = %d, want %d (model-limit backstop)", got, modelCap)
	}
	if got := effectiveEmbedTokenCap(-5); got != modelCap {
		t.Fatalf("effectiveEmbedTokenCap(-5) = %d, want %d (model-limit backstop)", got, modelCap)
	}
	// A configured value below the model limit tightens the cap: 2560 * 0.9 = 2304.
	if got := effectiveEmbedTokenCap(2560); got != 2304 {
		t.Fatalf("effectiveEmbedTokenCap(2560) = %d, want 2304", got)
	}
	// A configured value at or above the model limit still caps at the model limit.
	if got := effectiveEmbedTokenCap(4096); got != modelCap {
		t.Fatalf("effectiveEmbedTokenCap(4096) = %d, want %d", got, modelCap)
	}
	if got := effectiveEmbedTokenCap(10000); got != modelCap {
		t.Fatalf("effectiveEmbedTokenCap(10000) = %d, want %d (capped at model limit)", got, modelCap)
	}
	// A tiny positive limit keeps a floor of one token.
	if got := effectiveEmbedTokenCap(1); got != 1 {
		t.Fatalf("effectiveEmbedTokenCap(1) = %d, want 1", got)
	}
}

func TestEffectiveEmbedTokenCapStaysBelowModelLimit(t *testing.T) {
	t.Parallel()
	// The cap must stay strictly below the model's hard input-token limit for every
	// input, so a sub-chunk sized to the cap never reaches the limit that would drop
	// it. This is the invariant a byte-budgeted sub-chunk relies on.
	for _, maxTokens := range []int{-1, 0, 1, 2560, 4096, 8192, 100000} {
		if got := effectiveEmbedTokenCap(maxTokens); got >= EmbedModelInputTokenLimit {
			t.Fatalf("effectiveEmbedTokenCap(%d) = %d, want < %d", maxTokens, got, EmbedModelInputTokenLimit)
		}
	}
}

func TestEffectiveEmbedTokenCapForLimitUsesPresetLimit(t *testing.T) {
	t.Parallel()
	// The local ONNX backend passes the offline preset's real token limit, so the
	// cap tracks the preset (2048 or 512) rather than the 4096 OpenAI-compatible
	// limit that would let the ONNX tokenizer truncate the input.
	// int(2048 * 0.9) = 1843, int(512 * 0.9) = 460 after the safety margin.
	if got := EffectiveEmbedTokenCapForLimit(0, 2048); got != 1843 {
		t.Fatalf("EffectiveEmbedTokenCapForLimit(0, 2048) = %d, want 1843", got)
	}
	if got := EffectiveEmbedTokenCapForLimit(0, 512); got != 460 {
		t.Fatalf("EffectiveEmbedTokenCapForLimit(0, 512) = %d, want 460", got)
	}
	// A non-positive model limit fails safe at the smallest known limit: 460.
	if got := EffectiveEmbedTokenCapForLimit(0, 0); got != 460 {
		t.Fatalf("EffectiveEmbedTokenCapForLimit(0, 0) = %d, want 460 (safe fallback)", got)
	}
}

func TestEmbedChunkByteBudget(t *testing.T) {
	t.Parallel()
	// The budget is always positive because the model limit is always enforced.
	// 3686 tokens * 5 / 2 = 9215 bytes at the conservative 2.5 bytes/token ratio.
	modelBudget := 3686 * conservativeEmbedBytesPerTokenNum / conservativeEmbedBytesPerTokenDen
	if got := EmbedChunkByteBudget(0); got != modelBudget {
		t.Fatalf("EmbedChunkByteBudget(0) = %d, want %d (model-limit backstop)", got, modelBudget)
	}
	if got := EmbedChunkByteBudget(4096); got != modelBudget {
		t.Fatalf("EmbedChunkByteBudget(4096) = %d, want %d", got, modelBudget)
	}
	// A tighter configured cap yields a smaller budget: 2304 * 5 / 2 = 5760 bytes.
	if got := EmbedChunkByteBudget(2560); got != 5760 {
		t.Fatalf("EmbedChunkByteBudget(2560) = %d, want 5760", got)
	}
}

func TestEmbedChunkByteBudgetForLimitTracksPreset(t *testing.T) {
	t.Parallel()
	// The bge-small preset caps at 512 tokens: int(512 * 0.9) = 460 tokens, and
	// 460 * 5 / 2 = 1150 bytes, well under the 9215-byte OpenAI-compatible budget.
	want := 460 * conservativeEmbedBytesPerTokenNum / conservativeEmbedBytesPerTokenDen
	if got := EmbedChunkByteBudgetForLimit(0, 512); got != want {
		t.Fatalf("EmbedChunkByteBudgetForLimit(0, 512) = %d, want %d", got, want)
	}
	if EmbedChunkByteBudgetForLimit(0, 512) >= EmbedChunkByteBudget(0) {
		t.Fatal("preset byte budget must be smaller than the OpenAI-compatible budget")
	}
}

func TestActiveEmbedTokenLimitSelectsPresetForONNX(t *testing.T) {
	t.Parallel()
	// The ONNX provider selects the offline preset's maximum tokens.
	onnx := Config{EmbeddingProvider: EmbeddingProviderONNX, OfflineEmbeddingModel: "bge-small"}
	if got := ActiveEmbedTokenLimit(onnx); got != 512 {
		t.Fatalf("ActiveEmbedTokenLimit(bge-small) = %d, want 512", got)
	}
	gemma := Config{EmbeddingProvider: EmbeddingProviderONNX, OfflineEmbeddingModel: "embeddinggemma"}
	if got := ActiveEmbedTokenLimit(gemma); got != 2048 {
		t.Fatalf("ActiveEmbedTokenLimit(embeddinggemma) = %d, want 2048", got)
	}
}

func TestActiveEmbedTokenLimitSelectsOpenAICompatibleModel(t *testing.T) {
	t.Parallel()
	cfg := Config{
		EmbeddingProvider: "OpenAI",
		EmbeddingModel:    "BAAI/bge-small-en-v1.5",
	}
	if got := ActiveEmbedTokenLimit(cfg); got != 512 {
		t.Fatalf("ActiveEmbedTokenLimit(BAAI/bge-small-en-v1.5) = %d, want 512", got)
	}
	if got := EmbedChunkByteBudgetForLimit(0, ActiveEmbedTokenLimit(cfg)); got >= EmbedChunkByteBudget(0) {
		t.Fatalf("hosted bge-small byte budget = %d, want < %d", got, EmbedChunkByteBudget(0))
	}
}

func TestActiveEmbedTokenLimitUnknownModelUsesSmallestKnownLimit(t *testing.T) {
	t.Parallel()
	cfg := Config{
		EmbeddingProvider: "OpenAI",
		EmbeddingModel:    "custom/unknown-embedding-model",
	}
	if got := ActiveEmbedTokenLimit(cfg); got != 512 {
		t.Fatalf("ActiveEmbedTokenLimit(unknown model) = %d, want safe fallback 512", got)
	}
}

func TestOpenAICompatibleConfiguredLimitOnlyTightensModelLimit(t *testing.T) {
	t.Parallel()
	for _, testCase := range []struct {
		name       string
		maxTokens  int
		wantTokens int
	}{
		{name: "larger configured limit", maxTokens: 1024, wantTokens: 460},
		{name: "smaller configured limit", maxTokens: 256, wantTokens: 230},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			cfg := Config{
				EmbeddingProvider:  "OpenAI",
				EmbeddingModel:     "BAAI/bge-small-en-v1.5",
				EmbeddingMaxTokens: testCase.maxTokens,
			}
			modelLimit := ActiveEmbedTokenLimit(cfg)
			if got := EffectiveEmbedTokenCapForLimit(cfg.EmbeddingMaxTokens, modelLimit); got != testCase.wantTokens {
				t.Fatalf(
					"effective cap for embeddingMaxTokens %d = %d, want %d",
					testCase.maxTokens,
					got,
					testCase.wantTokens,
				)
			}
		})
	}
}

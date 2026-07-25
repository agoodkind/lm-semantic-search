package config

import "testing"

func TestEffectiveEmbedTokenCap(t *testing.T) {
	t.Parallel()
	// int(4096 * 0.9) = 3686, the model-limit backstop after the safety margin.
	modelCap := 3686
	// An unset or non-positive value falls back to the model-limit backstop rather
	// than disabling the split, so an oversized input is divided instead of dropped.
	if got := EffectiveEmbedTokenCap(0); got != modelCap {
		t.Fatalf("EffectiveEmbedTokenCap(0) = %d, want %d (model-limit backstop)", got, modelCap)
	}
	if got := EffectiveEmbedTokenCap(-5); got != modelCap {
		t.Fatalf("EffectiveEmbedTokenCap(-5) = %d, want %d (model-limit backstop)", got, modelCap)
	}
	// A configured value below the model limit tightens the cap: 2560 * 0.9 = 2304.
	if got := EffectiveEmbedTokenCap(2560); got != 2304 {
		t.Fatalf("EffectiveEmbedTokenCap(2560) = %d, want 2304", got)
	}
	// A configured value at or above the model limit still caps at the model limit.
	if got := EffectiveEmbedTokenCap(4096); got != modelCap {
		t.Fatalf("EffectiveEmbedTokenCap(4096) = %d, want %d", got, modelCap)
	}
	if got := EffectiveEmbedTokenCap(10000); got != modelCap {
		t.Fatalf("EffectiveEmbedTokenCap(10000) = %d, want %d (capped at model limit)", got, modelCap)
	}
	// A tiny positive limit keeps a floor of one token.
	if got := EffectiveEmbedTokenCap(1); got != 1 {
		t.Fatalf("EffectiveEmbedTokenCap(1) = %d, want 1", got)
	}
}

func TestEffectiveEmbedTokenCapStaysBelowModelLimit(t *testing.T) {
	t.Parallel()
	// The cap must stay strictly below the model's hard input-token limit for every
	// input, so a sub-chunk sized to the cap never reaches the limit that would drop
	// it. This is the invariant a byte-budgeted sub-chunk relies on.
	for _, maxTokens := range []int{-1, 0, 1, 2560, 4096, 8192, 100000} {
		if got := EffectiveEmbedTokenCap(maxTokens); got >= EmbedModelInputTokenLimit {
			t.Fatalf("EffectiveEmbedTokenCap(%d) = %d, want < %d", maxTokens, got, EmbedModelInputTokenLimit)
		}
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

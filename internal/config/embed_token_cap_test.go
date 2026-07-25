package config

import "testing"

func TestEffectiveEmbedTokenCap(t *testing.T) {
	t.Parallel()
	if got := EffectiveEmbedTokenCap(0); got != 0 {
		t.Fatalf("EffectiveEmbedTokenCap(0) = %d, want 0 (disabled)", got)
	}
	if got := EffectiveEmbedTokenCap(-5); got != 0 {
		t.Fatalf("EffectiveEmbedTokenCap(-5) = %d, want 0 (disabled)", got)
	}
	// 4096 * 0.9 = 3686.4, truncated to 3686.
	if got := EffectiveEmbedTokenCap(4096); got != 3686 {
		t.Fatalf("EffectiveEmbedTokenCap(4096) = %d, want 3686", got)
	}
	// A tiny positive limit keeps a floor of one token.
	if got := EffectiveEmbedTokenCap(1); got != 1 {
		t.Fatalf("EffectiveEmbedTokenCap(1) = %d, want 1", got)
	}
}

func TestEmbedChunkByteBudget(t *testing.T) {
	t.Parallel()
	if got := EmbedChunkByteBudget(0); got != 0 {
		t.Fatalf("EmbedChunkByteBudget(0) = %d, want 0 (disabled)", got)
	}
	// 3686 tokens * 4 bytes/token = 14744 bytes.
	if got := EmbedChunkByteBudget(4096); got != 3686*4 {
		t.Fatalf("EmbedChunkByteBudget(4096) = %d, want %d", got, 3686*4)
	}
}

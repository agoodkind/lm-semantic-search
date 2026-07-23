package offlinemodel

import "testing"

func TestResolveDefaultsToEmbeddingGemma(t *testing.T) {
	t.Parallel()

	preset, err := Resolve("")
	if err != nil {
		t.Fatalf("Resolve returned error: %v", err)
	}
	if preset.Name != EmbeddingGemma {
		t.Fatalf("Name = %q, want %q", preset.Name, EmbeddingGemma)
	}
	if preset.Dimension != 768 {
		t.Fatalf("Dimension = %d, want 768", preset.Dimension)
	}
	if preset.Pooling != PoolingMean {
		t.Fatalf("Pooling = %q, want %q", preset.Pooling, PoolingMean)
	}
	if preset.QueryPrefix != "task: code retrieval | query: " {
		t.Fatalf("QueryPrefix = %q", preset.QueryPrefix)
	}
}

func TestResolveIncludesBGESmallFallback(t *testing.T) {
	t.Parallel()

	preset, err := Resolve(BGESmall)
	if err != nil {
		t.Fatalf("Resolve returned error: %v", err)
	}
	if preset.Dimension != 384 {
		t.Fatalf("Dimension = %d, want 384", preset.Dimension)
	}
	if preset.Pooling != PoolingCLS {
		t.Fatalf("Pooling = %q, want %q", preset.Pooling, PoolingCLS)
	}
	if preset.ModelDataURL != "" || preset.ModelDataSHA256 != "" {
		t.Fatalf("BGE fallback unexpectedly has external model data")
	}
}

func TestResolveRejectsUnknownPreset(t *testing.T) {
	t.Parallel()

	_, err := Resolve("unknown")
	if err == nil {
		t.Fatal("Resolve returned nil error for unknown preset")
	}
}

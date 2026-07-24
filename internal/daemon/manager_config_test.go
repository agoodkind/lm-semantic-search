package daemon

import (
	"slices"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
)

func TestEnrichIndexConfigReportsConfiguredVectorBackend(t *testing.T) {
	manager := &Manager{
		config: config.Config{
			EmbeddingProvider: config.EmbeddingProviderONNX,
			IndexBackend:      config.IndexBackendLocal,
		},
	}

	got := manager.enrichIndexConfig(model.IndexConfig{})

	if got.VectorBackend != config.IndexBackendLocal {
		t.Fatalf(
			"VectorBackend = %q, want %q",
			got.VectorBackend,
			config.IndexBackendLocal,
		)
	}
}

func TestNormalizeSubmoduleListRejectsRootEscapesAndAbsolutePaths(t *testing.T) {
	input := []string{
		"third_party/lib",
		"a/..",
		"..",
		"../outside",
		"/tmp/lib",
	}

	got := normalizeSubmoduleList(input)
	want := []string{"third_party/lib"}
	if !slices.Equal(got, want) {
		t.Fatalf("normalizeSubmoduleList() = %v, want %v", got, want)
	}
}

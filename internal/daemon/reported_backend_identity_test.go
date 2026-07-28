package daemon

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
)

// TestReportedIndexConfigNamesTheBuiltBackendNotTheConfiguredOne proves the
// daemon answers "which store am I using" from the store it built rather than
// from the setting that asked for one.
//
// The two can disagree. Reading the setting reports an intention, so a backend
// selected wrongly would still be reported as the backend that was requested,
// and every surface carrying that value would repeat it. The offline profile is
// where this matters most: its whole claim is that a hostile environment naming
// a remote store and a remote embedder produces neither.
//
// The fixture states the disagreement outright. The configuration asks for the
// remote store and the remote embedder while the built backend is the local one
// with the embedded model, so a report that follows the configuration and a
// report that follows the backend cannot both pass.
func TestReportedIndexConfigNamesTheBuiltBackendNotTheConfiguredOne(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	manager.config.IndexBackend = config.IndexBackendMilvus
	manager.config.EmbeddingProvider = "OpenAI"
	manager.semantic = &fakeSemantic{
		backendName:           config.IndexBackendLocal,
		embeddingProviderName: config.EmbeddingProviderONNX,
	}

	reported := manager.enrichIndexConfig(model.IndexConfig{})

	if reported.VectorBackend != config.IndexBackendLocal {
		t.Fatalf(
			"reported VectorBackend = %q, want %q from the built backend",
			reported.VectorBackend,
			config.IndexBackendLocal,
		)
	}
	if reported.EmbeddingProvider != config.EmbeddingProviderONNX {
		t.Fatalf(
			"reported EmbeddingProvider = %q, want %q from the built backend",
			reported.EmbeddingProvider,
			config.EmbeddingProviderONNX,
		)
	}
}

// TestReportedIndexConfigFallsBackToConfigurationWithNoBackend covers the window
// before a backend exists. A manager whose backend construction has not run, or
// failed, has nothing to ask, so the configured values are the only answer it
// has and reporting them is honest rather than a claim about what was built.
func TestReportedIndexConfigFallsBackToConfigurationWithNoBackend(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	manager.config.IndexBackend = config.IndexBackendMilvus
	manager.config.EmbeddingProvider = "OpenAI"
	manager.semantic = nil

	reported := manager.enrichIndexConfig(model.IndexConfig{})

	if reported.VectorBackend != config.IndexBackendMilvus {
		t.Fatalf(
			"reported VectorBackend = %q, want the configured %q with no backend built",
			reported.VectorBackend,
			config.IndexBackendMilvus,
		)
	}
	if reported.EmbeddingProvider != "OpenAI" {
		t.Fatalf(
			"reported EmbeddingProvider = %q, want the configured OpenAI with no backend built",
			reported.EmbeddingProvider,
		)
	}
}

package daemon

import (
	"testing"

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
	manager.config.IndexBackend = model.VectorBackendMilvus
	manager.config.EmbeddingProvider = model.EmbeddingProviderOpenAI
	manager.semantic = &fakeSemantic{
		backendName:           model.VectorBackendLocal,
		embeddingProviderName: model.EmbeddingProviderONNX,
	}

	reported := manager.enrichIndexConfig(model.IndexConfig{})

	if reported.VectorBackend != model.VectorBackendLocal {
		t.Fatalf(
			"reported VectorBackend = %q, want %q from the built backend",
			reported.VectorBackend,
			model.VectorBackendLocal,
		)
	}
	if reported.EmbeddingProvider != model.EmbeddingProviderONNX {
		t.Fatalf(
			"reported EmbeddingProvider = %q, want %q from the built backend",
			reported.EmbeddingProvider,
			model.EmbeddingProviderONNX,
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
	manager.config.IndexBackend = model.VectorBackendMilvus
	manager.config.EmbeddingProvider = model.EmbeddingProviderOpenAI
	manager.semantic = nil

	reported := manager.enrichIndexConfig(model.IndexConfig{})

	if reported.VectorBackend != model.VectorBackendMilvus {
		t.Fatalf(
			"reported VectorBackend = %q, want the configured %q with no backend built",
			reported.VectorBackend,
			model.VectorBackendMilvus,
		)
	}
	if reported.EmbeddingProvider != model.EmbeddingProviderOpenAI {
		t.Fatalf(
			"reported EmbeddingProvider = %q, want the configured %q with no backend built",
			reported.EmbeddingProvider,
			model.EmbeddingProviderOpenAI,
		)
	}
}

// TestReportedProviderKeepsTheConfiguredNameWhenNoEmbedderWasBuilt pins the one
// case where the report deliberately does not follow the built backend.
//
// The Milvus-backed service builds no embedder whenever the store address is
// empty. Reporting none there would make this field change the moment an address
// is configured, and the field is a config digest input, so the checkpoint that
// digest gates would be discarded and every file embedded again for a setting
// that does not change how anything is embedded.
//
// The vector backend gets no such fallback, because a backend object always
// exists once one is built.
func TestReportedProviderKeepsTheConfiguredNameWhenNoEmbedderWasBuilt(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	manager.config.EmbeddingProvider = model.EmbeddingProviderOpenAI
	manager.semantic = &fakeSemantic{
		backendName:            model.VectorBackendMilvus,
		embeddingProviderName:  model.EmbeddingProviderNone,
		reportProviderVerbatim: true,
	}

	reported := manager.enrichIndexConfig(model.IndexConfig{})

	if reported.EmbeddingProvider != model.EmbeddingProviderOpenAI {
		t.Fatalf(
			"reported EmbeddingProvider = %q, want the configured %q so setting a store address cannot move the digest",
			reported.EmbeddingProvider,
			model.EmbeddingProviderOpenAI,
		)
	}
}

// TestConfigDigestIsUnchangedByProviderSpelling proves a configured name that
// differs only in letter case cannot move the config digest.
//
// The digest gates checkpoint reuse: a merkle snapshot whose stored digest does
// not match the current one is discarded and every file is embedded again. The
// provider name reaches that digest, and the name a constructed provider reports
// is one fixed spelling, so an accepted variant spelling in the configuration
// used to produce a different digest from the same actual backend. Parsing the
// configured name into a closed set removes the variant before it can be hashed.
func TestConfigDigestIsUnchangedByProviderSpelling(t *testing.T) {
	t.Parallel()

	spellings := []string{"OpenAI", "openai", "OPENAI", "  OpenAI  "}
	digests := make(map[string]string, len(spellings))
	for _, spelling := range spellings {
		provider, err := model.ParseEmbeddingProvider(spelling)
		if err != nil {
			t.Fatalf("ParseEmbeddingProvider(%q) returned error: %v", spelling, err)
		}
		indexConfig := model.IndexConfig{EmbeddingProvider: provider}
		digests[spelling] = digestIndexConfig(indexConfig)
	}

	canonical := digests["OpenAI"]
	for spelling, digest := range digests {
		if digest != canonical {
			t.Fatalf(
				"digest for provider spelling %q = %s, want %s so an accepted variant cannot discard a checkpoint",
				spelling,
				digest,
				canonical,
			)
		}
	}
}

// TestUnsupportedProviderNameIsRejected proves an unknown name fails at the
// parse boundary rather than travelling onward as an unrecognised string that
// some later comparison silently treats as the default.
func TestUnsupportedProviderNameIsRejected(t *testing.T) {
	t.Parallel()

	if _, err := model.ParseEmbeddingProvider("cohere"); err == nil {
		t.Fatal("ParseEmbeddingProvider accepted an unsupported provider name")
	}
}

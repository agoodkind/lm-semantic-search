package config

import "goodkind.io/lm-semantic-search/internal/offlinemodel"

const (
	// ProfileStandard is the default profile: the Milvus store and the
	// OpenAI-compatible embedder.
	ProfileStandard = "standard"
	// ProfileOffline selects the embedded local store and the in-process ONNX
	// embedder so search runs with no external store or model server.
	ProfileOffline = "offline"
	// IndexBackendMilvus selects the Milvus-backed vector store (default).
	IndexBackendMilvus = "milvus"
	// IndexBackendLocal selects the embedded local vector store.
	IndexBackendLocal = "local"
	// EmbeddingProviderONNX selects the in-process ONNX embedding provider.
	EmbeddingProviderONNX = "onnx"
)

// ApplyProfile expands the user-facing Profile into the derived backend and
// embedder fields. It is a pure function: standard (or empty) leaves cfg's
// backend and embedder as given, offline sets the local store and the ONNX
// embedder and clears the Milvus requirement so an absent MilvusAddress is not
// treated as degraded.
func ApplyProfile(cfg Config) Config {
	if cfg.Profile != ProfileOffline {
		if cfg.IndexBackend == "" {
			cfg.IndexBackend = IndexBackendMilvus
		}
		return cfg
	}
	preset, err := offlinemodel.Resolve(cfg.OfflineEmbeddingModel)
	if err == nil {
		cfg.OfflineEmbeddingModel = preset.Name
		cfg.EmbeddingModel = preset.Name
		cfg.EmbeddingDimension = preset.Dimension
		cfg.QueryInstructionPrefix = preset.QueryPrefix
	} else {
		cfg.EmbeddingModel = cfg.OfflineEmbeddingModel
		cfg.EmbeddingDimension = 0
		cfg.QueryInstructionPrefix = ""
	}
	cfg.IndexBackend = IndexBackendLocal
	cfg.EmbeddingProvider = EmbeddingProviderONNX
	cfg.MilvusAddress = ""
	cfg.MilvusToken = ""
	return cfg
}

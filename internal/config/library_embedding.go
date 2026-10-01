package config

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log/slog"
	"strconv"
	"strings"

	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

// ResolveLibraryEmbeddingIdentity validates the shared library model identity and resolves pinned ONNX artifacts.
func ResolveLibraryEmbeddingIdentity(cfg Config) (Config, error) {
	if cfg.CodebaseStore != CodebaseStoreLibrary {
		return cfg, nil
	}
	if cfg.EmbeddingProvider == EmbeddingProviderONNX {
		preset, err := offlinemodel.Resolve(cfg.OfflineEmbeddingModel)
		if err != nil {
			slog.Error("resolve library offline model identity failed", "model", cfg.OfflineEmbeddingModel, "err", err)
			return Config{}, fmt.Errorf("resolve library offline model identity: %w", err)
		}
		identity := strings.Join([]string{preset.ModelSHA256, preset.ModelDataSHA256, preset.TokenizerSHA256, string(preset.Pooling), strconv.FormatBool(preset.UsesTokenTypeIDs)}, "\x00")
		digest := sha256.Sum256([]byte(identity))
		cfg.EmbeddingModel = preset.Name
		cfg.EmbeddingDimension = preset.Dimension
		cfg.EmbeddingRevision = hex.EncodeToString(digest[:])
		cfg.EmbeddingNormalization = "l2"
	}
	cfg.EmbeddingRevision = strings.TrimSpace(cfg.EmbeddingRevision)
	cfg.EmbeddingNormalization = strings.TrimSpace(cfg.EmbeddingNormalization)
	if strings.TrimSpace(cfg.EmbeddingModel) == "" || cfg.EmbeddingDimension <= 0 || cfg.EmbeddingRevision == "" || cfg.EmbeddingNormalization == "" {
		return Config{}, fmt.Errorf("the library codebase store requires EMBEDDING_MODEL, EMBEDDING_DIMENSION, EMBEDDING_REVISION, and EMBEDDING_NORMALIZATION")
	}
	return cfg, nil
}

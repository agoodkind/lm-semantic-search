// Package providers constructs the embedding provider that the daemon
// configuration selects.
package providers

import (
	"context"
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/embedding"
	"goodkind.io/lm-semantic-search/internal/embedding/onnx"
)

// New constructs the configured embedding provider. The ONNX provider runs the
// offline model in process. Every other configuration builds the
// OpenAI-compatible adapter through [embedding.NewHostedProvider], which
// rejects an unsupported provider name.
func New(ctx context.Context, cfg config.Config) (embedding.Provider, error) {
	if cfg.EmbeddingProvider == config.EmbeddingProviderONNX {
		provider, err := onnx.NewProvider(ctx, cfg.OfflineEmbeddingModel, cfg.ModelCacheRoot)
		if err != nil {
			slog.ErrorContext(ctx, "construct ONNX embedding provider failed", "model", cfg.OfflineEmbeddingModel, "err", err)
			return nil, fmt.Errorf("construct ONNX embedding provider: %w", err)
		}
		return provider, nil
	}
	provider, err := embedding.NewHostedProvider(ctx, cfg)
	if err != nil {
		slog.ErrorContext(ctx, "construct hosted embedding provider failed", "provider", cfg.EmbeddingProvider, "err", err)
		return nil, fmt.Errorf("construct hosted embedding provider: %w", err)
	}
	return provider, nil
}

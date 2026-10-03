// Package embeddingprovider builds the embedding provider the daemon
// configuration selects.
package embeddingprovider

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"goodkind.io/lm-semantic-search/embedding"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/embedding/onnx"
	"goodkind.io/lm-semantic-search/internal/model"
)

// New constructs the configured embedding provider.
//
// The ONNX provider runs the embedded offline model in process. The default
// OpenAI-compatible adapter sends requests to the configured embeddings API.
func New(ctx context.Context, cfg config.Config) (embedding.Provider, error) {
	switch cfg.EmbeddingProvider {
	case config.EmbeddingProviderONNX:
		provider, err := onnx.NewProvider(ctx, cfg)
		if err != nil {
			slog.ErrorContext(ctx, "create ONNX embedding provider failed", "err", err)
			return nil, fmt.Errorf("create ONNX embedding provider: %w", err)
		}
		return provider, nil
	case model.EmbeddingProviderNone, config.EmbeddingProviderOpenAI:
		// Both build the OpenAI-compatible adapter: an unnamed provider is the
		// historical default rather than an error.
	default:
		slog.ErrorContext(
			ctx,
			"embedding provider is not supported",
			"provider",
			cfg.EmbeddingProvider,
			"err",
			errors.New("only ONNX and OpenAI-compatible adapters are supported"),
		)
		return nil, fmt.Errorf(
			"embedding provider %q is not supported; use %q or %q",
			cfg.EmbeddingProvider,
			config.EmbeddingProviderONNX,
			config.EmbeddingProviderOpenAI,
		)
	}
	requestTimeout := time.Duration(cfg.EmbeddingRequestTimeoutMS) * time.Millisecond
	provider, err := embedding.NewOpenAICompatible(embedding.OpenAIOptions{
		APIKey:         cfg.OpenAIAPIKey,
		BaseURL:        cfg.OpenAIBaseURL,
		Model:          cfg.EmbeddingModel,
		Dimensions:     cfg.EmbeddingDimension,
		RequestTimeout: requestTimeout,
	})
	if err != nil {
		slog.ErrorContext(ctx, "create OpenAI-compatible embedding provider failed", "err", err)
		return nil, fmt.Errorf("create OpenAI-compatible embedding provider: %w", err)
	}
	return provider, nil
}

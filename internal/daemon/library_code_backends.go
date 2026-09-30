package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"path/filepath"
	"time"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/embedded"
	"goodkind.io/lm-semantic-search/library/embedding"
	"goodkind.io/lm-semantic-search/library/embedding/onnx"
	"goodkind.io/lm-semantic-search/library/milvus"
)

// libraryCodePoolDirectory is the libraryCodeDirectory subdirectory of the
// embedded vector pool that the offline profile uses.
const libraryCodePoolDirectory = "pool"

// libraryCodeBackends is the vector pool, embedder, and part limits of the
// codebase store. The offline profile has an exact tokenizer and no Milvus
// client; the standard profile has a Milvus client and no tokenizer, and
// PrepareText counts one byte as one token.
type libraryCodeBackends struct {
	vectors      library.VectorStore
	embedder     library.Embedder
	tokenizer    library.Tokenizer
	maxTokens    int
	maxBytes     int
	vectorClient *milvusclient.Client
}

// newLibraryCodeBackends opens the Milvus pool and the OpenAI-compatible
// embedder for the standard profile, and the embedded pool, the ONNX
// embedder, and the ONNX tokenizer for the offline profile.
func newLibraryCodeBackends(ctx context.Context, cfg config.Config) (libraryCodeBackends, error) {
	if cfg.EmbeddingProvider == config.EmbeddingProviderONNX {
		return newOfflineLibraryCodeBackends(ctx, cfg)
	}
	embedder, err := embedding.NewOpenAI(ctx, embedding.OpenAIConfig{
		BaseURL:        cfg.OpenAIBaseURL,
		APIKey:         cfg.OpenAIAPIKey,
		Model:          cfg.EmbeddingModel,
		Dimension:      int(cfg.EmbeddingDimension),
		RequestTimeout: time.Duration(cfg.EmbeddingRequestTimeoutMS) * time.Millisecond,
		MaxAttempts:    0,
		BackoffBase:    0,
	})
	if err != nil {
		slog.ErrorContext(ctx, "create library codebase embedder failed", "err", err)
		return libraryCodeBackends{}, fmt.Errorf("create library codebase embedder: %w", err)
	}
	vectorClient, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: cfg.MilvusAddress,
		APIKey:  cfg.MilvusToken,
		DBName:  cfg.MilvusDatabase,
	})
	if err != nil {
		slog.ErrorContext(ctx, "connect library codebase vector pool failed", "err", err)
		return libraryCodeBackends{}, fmt.Errorf("connect library codebase vector pool: %w", err)
	}
	vectors, err := milvus.New(vectorClient, milvus.Config{Database: milvusDatabaseName(cfg), Collection: libraryCodeCollection})
	if err != nil {
		slog.ErrorContext(ctx, "create library codebase vector adapter failed", "err", err)
		return libraryCodeBackends{}, errors.Join(fmt.Errorf("create library codebase vector adapter: %w", err), closeLibraryVectorClient(ctx, vectorClient))
	}
	return libraryCodeBackends{
		vectors:      vectors,
		embedder:     embedder,
		tokenizer:    nil,
		maxTokens:    libraryCodeMaxTokens(cfg),
		maxBytes:     0,
		vectorClient: vectorClient,
	}, nil
}

func newOfflineLibraryCodeBackends(ctx context.Context, cfg config.Config) (libraryCodeBackends, error) {
	modelConfig := onnx.Config{ModelName: cfg.OfflineEmbeddingModel, ModelCacheRoot: cfg.ModelCacheRoot}
	embedder, err := onnx.New(ctx, modelConfig)
	if err != nil {
		slog.ErrorContext(ctx, "create offline library codebase embedder failed", "err", err)
		return libraryCodeBackends{}, fmt.Errorf("create offline library codebase embedder: %w", err)
	}
	tokenizer, err := onnx.NewTokenizer(ctx, modelConfig)
	if err != nil {
		slog.ErrorContext(ctx, "create offline library codebase tokenizer failed", "err", err)
		return libraryCodeBackends{}, fmt.Errorf("create offline library codebase tokenizer: %w", err)
	}
	vectors, err := embedded.New(embedded.Config{Root: filepath.Join(cfg.StateRoot, libraryCodeDirectory, libraryCodePoolDirectory)})
	if err != nil {
		slog.ErrorContext(ctx, "create offline library codebase vector pool failed", "err", err)
		return libraryCodeBackends{}, fmt.Errorf("create offline library codebase vector pool: %w", err)
	}
	return libraryCodeBackends{
		vectors:      vectors,
		embedder:     embedder,
		tokenizer:    tokenizer,
		maxTokens:    tokenizer.MaxTokens(),
		maxBytes:     tokenizer.MaxInputBytes(),
		vectorClient: nil,
	}, nil
}

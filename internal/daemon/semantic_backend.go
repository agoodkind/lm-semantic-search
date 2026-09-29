package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/localvec"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// newSemanticIndex constructs the vector-store-and-embedder backend the manager
// depends on, selected by cfg.IndexBackend. The local backend is the offline
// profile's embedded store; every other value builds the Milvus-backed service,
// including the zero value, which a config assembled without ApplyProfile leaves
// unset. With cfg.CodebaseStore set to the library store, the returned index
// writes codebase chunks to the shared search library and passes every other
// call to that backend.
func newSemanticIndex(ctx context.Context, cfg config.Config) (semanticIndex, error) {
	inner, err := newBackendIndex(ctx, cfg)
	if err != nil {
		return nil, err
	}
	if cfg.CodebaseStore != config.CodebaseStoreLibrary {
		return inner, nil
	}
	wrapped, err := newLibraryCodeIndex(ctx, cfg, inner)
	if err != nil {
		if closer, ok := inner.(semanticCloser); ok {
			err = errors.Join(err, closer.Close(ctx))
		}
		slog.ErrorContext(ctx, "create library codebase store failed", "err", err)
		return nil, fmt.Errorf("create library codebase store: %w", err)
	}
	return wrapped, nil
}

func newBackendIndex(ctx context.Context, cfg config.Config) (semanticIndex, error) {
	switch cfg.IndexBackend {
	case config.IndexBackendLocal:
		store, err := localvec.New(ctx, cfg)
		if err != nil {
			slog.ErrorContext(ctx, "create local vector store failed", "err", err)
			return nil, fmt.Errorf("create local vector store: %w", err)
		}
		return store, nil
	case config.IndexBackendMilvus:
		fallthrough
	default:
		service, err := semantic.NewService(ctx, cfg)
		if err != nil {
			slog.ErrorContext(ctx, "create semantic service failed", "err", err)
			return nil, fmt.Errorf("create semantic service: %w", err)
		}
		return service, nil
	}
}

package daemon

import (
	"context"
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/localvec"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// newSemanticIndex constructs the vector-store-and-embedder backend the manager
// depends on, selected by cfg.IndexBackend. The local backend is the offline
// profile's embedded store; the default is the Milvus-backed service.
func newSemanticIndex(ctx context.Context, cfg config.Config) (semanticIndex, error) {
	switch cfg.IndexBackend {
	case config.IndexBackendLocal:
		store, err := localvec.New(ctx, cfg)
		if err != nil {
			slog.ErrorContext(ctx, "create local vector store failed", "err", err)
			return nil, fmt.Errorf("create local vector store: %w", err)
		}
		return store, nil
	default:
		service, err := semantic.NewService(ctx, cfg)
		if err != nil {
			slog.ErrorContext(ctx, "create semantic service failed", "err", err)
			return nil, fmt.Errorf("create semantic service: %w", err)
		}
		return service, nil
	}
}

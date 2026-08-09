package localvec

import (
	"context"
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/semantic"
)

// PrepareCollection is a no-op because the local backend has no online schema migration.
func (store *Store) PrepareCollection(context.Context, string) error {
	return nil
}

// PinStaging satisfies the shared staging lifecycle without external residency.
func (store *Store) PinStaging(
	ctx context.Context,
	codebasePath string,
) (semantic.CollectionPin, error) {
	_ = store
	_ = codebasePath
	if err := ctx.Err(); err != nil {
		wrappedErr := fmt.Errorf("pin local staging collection: %w", err)
		slog.WarnContext(ctx, "pin local staging collection cancelled", "error", wrappedErr)
		return nil, wrappedErr
	}
	return semantic.NoopCollectionPin{}, nil
}

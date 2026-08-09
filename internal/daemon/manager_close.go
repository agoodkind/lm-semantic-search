package daemon

import (
	"context"
	"fmt"
	"log/slog"
)

type semanticCloser interface {
	Close(ctx context.Context) error
}

// Close shuts down the manager's graph and semantic resources.
func (manager *Manager) Close(ctx context.Context) error {
	manager.CloseGraphEngines()
	closer, ok := manager.semantic.(semanticCloser)
	if !ok {
		return nil
	}
	if err := closer.Close(ctx); err != nil {
		slog.ErrorContext(ctx, "close semantic backend", "err", err)
		return fmt.Errorf("close semantic backend: %w", err)
	}
	return nil
}

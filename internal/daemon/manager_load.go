package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"os"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/store"
)

func (manager *Manager) load(ctx context.Context) error {
	registry, err := store.ReadRegistry(manager.config.RegistryPath)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		slog.ErrorContext(ctx, "read registry failed", "path", manager.config.RegistryPath, "err", err)
		return fmt.Errorf("read registry: %w", err)
	}
	for _, codebase := range registry.Codebases {
		if codebase.Kind == "" {
			codebase.Kind = model.CodebaseKindCode
		}
		manager.codebases[codebase.ID] = codebase
	}
	dropGhostURICodebases(manager.codebases)

	jobs, err := store.ReadJobEvents(manager.config.JobsPath)
	if err != nil {
		slog.ErrorContext(ctx, "read jobs failed", "path", manager.config.JobsPath, "err", err)
		return fmt.Errorf("read jobs: %w", err)
	}
	maps.Copy(manager.jobs, jobs)
	manager.reconcileJournalOnStartLocked()
	return nil
}

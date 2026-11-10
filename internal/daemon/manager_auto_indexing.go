package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/model"
)

type automaticIndexingContextKey uint8

const automaticIndexingKey automaticIndexingContextKey = 0

var errAutomaticIndexingPaused = errors.New("automatic indexing is paused")

func automaticIndexingContext(ctx context.Context) context.Context {
	return context.WithValue(ctx, automaticIndexingKey, true)
}

// prepareAutomaticIndexingLocked resumes indexing only for an explicit request.
// The caller holds manager.mu and serializes admission with policyMutationMutex.
func (manager *Manager) prepareAutomaticIndexingLocked(
	ctx context.Context,
	codebase model.Codebase,
) error {
	if !codebase.AutoIndexingPaused {
		return nil
	}

	automatic, _ := ctx.Value(automaticIndexingKey).(bool)
	if automatic {
		return fmt.Errorf("%s: %w", codebase.CanonicalPath, errAutomaticIndexingPaused)
	}

	original := codebase
	codebase.AutoIndexingPaused = false
	codebase.UpdatedAt = clock.Now()
	manager.codebases[codebase.ID] = codebase
	if err := manager.saveLocked(); err != nil {
		manager.codebases[original.ID] = original
		return fmt.Errorf("resume automatic indexing for %s: %w", codebase.ID, err)
	}

	slog.InfoContext(
		ctx,
		"automatic indexing resumed",
		"codebase_id",
		codebase.ID,
		"path",
		codebase.CanonicalPath,
	)

	return nil
}

func (manager *Manager) automaticIndexingPaused(codebaseID string) bool {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	return manager.codebases[codebaseID].AutoIndexingPaused
}

// pauseAutomaticIndexingLocked persists the operator's stop before cancellation
// can release scheduler capacity or dispatch a pending successor.
func (manager *Manager) pauseAutomaticIndexingLocked(ctx context.Context, job model.Job) error {
	codebase, found := manager.codebases[job.CodebaseID]
	if !found || codebase.Kind == model.CodebaseKindDocument || codebase.AutoIndexingPaused {
		return nil
	}

	original := codebase
	codebase.AutoIndexingPaused = true
	codebase.UpdatedAt = clock.Now()
	manager.codebases[codebase.ID] = codebase
	if err := manager.saveLocked(); err != nil {
		manager.codebases[original.ID] = original
		return fmt.Errorf("pause automatic indexing for %s: %w", codebase.ID, err)
	}

	delete(manager.pendingCodeJobs, codebase.ID)
	slog.InfoContext(
		ctx,
		"automatic indexing paused after cancellation",
		"codebase_id",
		codebase.ID,
		"job_id",
		job.ID,
		"path",
		codebase.CanonicalPath,
	)

	return nil
}

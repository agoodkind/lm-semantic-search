package daemon

import (
	"context"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// TestUpdateJobCompletedClearsDependencyOnlyWhenFilesEmbedded guards the status
// contract that an empty-diff no-op completion must not clear the global
// vector-store outage banner. After the convergence fix, an unchanged codebase
// reaches the no-op completion with the whole codebase file count but zero
// embedded files, so gating the clear on FilesEmbedded (not IndexedFiles) keeps
// a real outage on another codebase visible.
func TestUpdateJobCompletedClearsDependencyOnlyWhenFilesEmbedded(t *testing.T) {
	t.Parallel()

	completeWithEmbedded := func(t *testing.T, filesEmbedded int32) *Manager {
		t.Helper()

		manager, _, repoPath := newTestManager(t)
		canonicalPath, err := filepath.EvalSymlinks(repoPath)
		if err != nil {
			t.Fatalf("EvalSymlinks returned error: %v", err)
		}
		manager.semantic = nil

		indexConfig := defaultIndexConfig()
		indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)
		codebase := newCodebaseRecord(canonicalPath)
		codebase.ID = "cb-health-gate"
		codebase.EffectiveConfig = indexConfig
		codebase.UpdatedAt = clock.Now()

		job := newQueuedJob(codebase.ID, canonicalPath, canonicalPath, testClientInfo(), string(jobOperationSync), false, indexConfig, clock.Now())
		job.State = model.JobStateRunning
		job.Progress.FilesEmbedded = filesEmbedded

		manager.mu.Lock()
		manager.noteDependencyFailureLocked(semantic.ErrUnavailable)
		manager.codebases[codebase.ID] = codebase
		manager.jobs[job.ID] = job
		manager.mu.Unlock()

		manager.updateJobCompleted(context.Background(), job.ID, indexer.Result{
			IndexedFiles:      3,
			TotalChunks:       5,
			Chunks:            nil,
			FileHashes:        map[string]string{"main.go": "hash"},
			SkippedFiles:      nil,
			SkippedOversize:   0,
			SkippedUnreadable: 0,
			SkippedPending:    0,
		})
		return manager
	}

	t.Run("zero embedded files keeps the store outage banner", func(t *testing.T) {
		manager := completeWithEmbedded(t, 0)
		manager.mu.Lock()
		mode := manager.health.Mode
		manager.mu.Unlock()
		if mode != dependencyStoreUnavailable {
			t.Fatalf("dependency mode = %q after a zero-embed completion, want it to stay %q; a no-op sync must not clear a real outage", mode, dependencyStoreUnavailable)
		}
	})

	t.Run("embedded files clear the store outage banner", func(t *testing.T) {
		manager := completeWithEmbedded(t, 5)
		manager.mu.Lock()
		mode := manager.health.Mode
		manager.mu.Unlock()
		if mode == dependencyStoreUnavailable {
			t.Fatalf("dependency mode still %q after an embedding completion, want it cleared", mode)
		}
	})
}

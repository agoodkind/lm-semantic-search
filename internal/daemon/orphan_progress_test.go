package daemon

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/store"
)

// TestJobProgressIsJournaledAndThrottled proves a running job's progress is
// persisted to the journal (so a crash preserves it) and that consecutive
// updates inside the throttle interval are not re-journaled.
func TestJobProgressIsJournaledAndThrottled(t *testing.T) {
	t.Parallel()

	manager, _, repoPath := newTestManager(t)
	cfg := defaultIndexConfig()
	job := newQueuedJob("cb-progress", repoPath, repoPath, testClientInfo(), string(jobOperationSync), false, cfg, emptyAdmissionBudget, clock.Now())

	manager.mu.Lock()
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	// First progress update journals immediately (no prior journal time).
	manager.updateJobProgress(job.ID, indexer.Progress{Phase: "Reindexing", FilesTotal: 100, FilesProcessed: 42}, "file")
	// Second update inside the throttle interval stays in memory only.
	manager.updateJobProgress(job.ID, indexer.Progress{Phase: "Reindexing", FilesTotal: 100, FilesProcessed: 99}, "file")

	loaded, err := store.ReadJobEvents(manager.config.JobsPath)
	if err != nil {
		t.Fatalf("ReadJobEvents returned error: %v", err)
	}
	persisted := loaded[job.ID].Progress.FilesProcessed
	if persisted != 42 {
		t.Fatalf("journaled FilesProcessed = %d, want 42 (first update journals, second is throttled)", persisted)
	}

	manager.mu.Lock()
	live := manager.jobs[job.ID].Progress.FilesProcessed
	manager.mu.Unlock()
	if live != 99 {
		t.Fatalf("in-memory FilesProcessed = %d, want 99 (latest update applies in memory)", live)
	}
}

// TestReconcileJournalPreservesProgress proves the restart reconciler keeps an
// interrupted job's journaled progress, marking it cancelled without resetting
// the counts to zero.
func TestReconcileJournalPreservesProgress(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	job := model.Job{ID: "job-orphan", CodebaseID: "cb-orphan", State: model.JobStateRunning}
	job.Progress.FilesProcessed = 42
	job.Progress.ChunksEmbedded = 7

	manager.mu.Lock()
	manager.jobs[job.ID] = job
	manager.reconcileJournalOnStartLocked()
	recovered := manager.jobs[job.ID]
	manager.mu.Unlock()

	if recovered.State != model.JobStateCancelled {
		t.Fatalf("orphan job state = %q, want cancelled", recovered.State)
	}
	if recovered.Progress.FilesProcessed != 42 {
		t.Fatalf("orphan FilesProcessed = %d, want 42 preserved", recovered.Progress.FilesProcessed)
	}
	if recovered.Progress.ChunksEmbedded != 7 {
		t.Fatalf("orphan ChunksEmbedded = %d, want 7 preserved", recovered.Progress.ChunksEmbedded)
	}
}

package daemon

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

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
	manager.closeJobJournal()

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

func TestUpdateJobRunningPersistsIndexingForRestart(t *testing.T) {
	t.Parallel()

	manager, cfg, repoPath := newTestManager(t)
	job := newQueuedJob(
		"cb-running-persistence",
		repoPath,
		repoPath,
		testClientInfo(),
		string(jobOperationIndex),
		false,
		defaultIndexConfig(),
		emptyAdmissionBudget,
		clock.Now(),
	)
	manager.mu.Lock()
	manager.codebases[job.CodebaseID] = model.Codebase{
		ID:              job.CodebaseID,
		CanonicalPath:   repoPath,
		Status:          model.CodebaseStatusPending,
		ActiveJobID:     job.ID,
		EffectiveConfig: job.Config,
	}
	manager.jobs[job.ID] = job
	if err := manager.saveLocked(); err != nil {
		manager.mu.Unlock()
		t.Fatalf("saveLocked returned error: %v", err)
	}
	manager.mu.Unlock()

	if err := manager.updateJobRunning(job); err != nil {
		t.Fatalf("updateJobRunning returned error: %v", err)
	}

	registry, err := store.ReadRegistry(cfg.RegistryPath)
	if err != nil {
		t.Fatalf("ReadRegistry returned error: %v", err)
	}
	if len(registry.Codebases) != 1 {
		t.Fatalf("registry codebases = %d, want 1", len(registry.Codebases))
	}
	if registry.Codebases[0].Status != model.CodebaseStatusIndexing {
		t.Fatalf("persisted status = %q, want %q", registry.Codebases[0].Status, model.CodebaseStatusIndexing)
	}
}

func TestUpdateJobRunningDoesNotJournalRunningAfterRegistryFailure(t *testing.T) {
	t.Parallel()

	manager, cfg, repoPath := newTestManager(t)
	job := newQueuedJob(
		"cb-running-registry-failure",
		repoPath,
		repoPath,
		testClientInfo(),
		string(jobOperationIndex),
		false,
		defaultIndexConfig(),
		emptyAdmissionBudget,
		clock.Now(),
	)
	manager.mu.Lock()
	manager.codebases[job.CodebaseID] = model.Codebase{
		ID:              job.CodebaseID,
		CanonicalPath:   repoPath,
		Status:          model.CodebaseStatusPending,
		ActiveJobID:     job.ID,
		EffectiveConfig: job.Config,
	}
	if err := manager.saveLocked(); err != nil {
		manager.mu.Unlock()
		t.Fatalf("saveLocked returned error: %v", err)
	}
	if err := manager.appendJobLocked("start_index", job); err != nil {
		manager.mu.Unlock()
		t.Fatalf("appendJobLocked returned error: %v", err)
	}
	manager.config.RegistryPath = t.TempDir()
	manager.mu.Unlock()

	if err := manager.updateJobRunning(job); err == nil {
		t.Fatal("updateJobRunning returned nil error after registry failure")
	}
	manager.closeJobJournal()

	jobs, err := store.ReadJobEvents(cfg.JobsPath)
	if err != nil {
		t.Fatalf("ReadJobEvents returned error: %v", err)
	}
	if jobs[job.ID].State != model.JobStateQueued {
		t.Fatalf("journaled state = %q, want %q", jobs[job.ID].State, model.JobStateQueued)
	}
}

func TestUpdateJobRunningRestoresQueuedStateAfterJournalFailure(t *testing.T) {
	manager, cfg, repoPath := newTestManager(t)
	job := newQueuedJob(
		"cb-running-journal-failure",
		repoPath,
		repoPath,
		testClientInfo(),
		string(jobOperationIndex),
		false,
		defaultIndexConfig(),
		emptyAdmissionBudget,
		clock.Now(),
	)
	manager.mu.Lock()
	manager.codebases[job.CodebaseID] = model.Codebase{
		ID:              job.CodebaseID,
		CanonicalPath:   repoPath,
		Status:          model.CodebaseStatusPending,
		ActiveJobID:     job.ID,
		EffectiveConfig: job.Config,
	}
	manager.jobs[job.ID] = job
	if err := manager.saveLocked(); err != nil {
		manager.mu.Unlock()
		t.Fatalf("saveLocked returned error: %v", err)
	}
	manager.mu.Unlock()
	manager.jobJournal.close()
	manager.jobJournal = nil
	manager.appendJobEvent = func(string, model.JobEvent) error { return errors.New("journal unavailable") }

	if err := manager.updateJobRunning(job); err == nil {
		t.Fatal("updateJobRunning returned nil after journal failure")
	}
	manager.mu.Lock()
	gotJob := manager.jobs[job.ID]
	gotCodebase := manager.codebases[job.CodebaseID]
	manager.mu.Unlock()
	if gotJob.State != model.JobStateQueued {
		t.Fatalf("in-memory job state = %q, want %q", gotJob.State, model.JobStateQueued)
	}
	if gotCodebase.Status != model.CodebaseStatusPending {
		t.Fatalf("in-memory codebase status = %q, want %q", gotCodebase.Status, model.CodebaseStatusPending)
	}
	registry, err := store.ReadRegistry(cfg.RegistryPath)
	if err != nil {
		t.Fatalf("ReadRegistry returned error: %v", err)
	}
	if len(registry.Codebases) != 1 || registry.Codebases[0].Status != model.CodebaseStatusPending {
		t.Fatalf("registry after journal failure = %+v, want one pending codebase", registry.Codebases)
	}
}

func TestRunJobKeepsFirstBuildQueuedWhenRunningStateCannotPersist(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	job := newQueuedJob(
		"cb-run-journal-failure",
		repoPath,
		repoPath,
		testClientInfo(),
		string(jobOperationIndex),
		false,
		defaultIndexConfig(),
		emptyAdmissionBudget,
		clock.Now(),
	)
	manager.mu.Lock()
	manager.codebases[job.CodebaseID] = model.Codebase{
		ID:              job.CodebaseID,
		CanonicalPath:   repoPath,
		Status:          model.CodebaseStatusPending,
		ActiveJobID:     job.ID,
		EffectiveConfig: job.Config,
	}
	manager.jobs[job.ID] = job
	if err := manager.saveLocked(); err != nil {
		manager.mu.Unlock()
		t.Fatalf("saveLocked returned error: %v", err)
	}
	manager.mu.Unlock()
	manager.jobJournal.close()
	manager.jobJournal = nil
	manager.appendJobEvent = func(string, model.JobEvent) error { return errors.New("journal unavailable") }
	backendCalled := false
	manager.runner = fakeRunner{index: func(context.Context, string, model.IndexConfig, func(indexer.Progress)) (indexer.Result, error) {
		backendCalled = true
		return indexer.Result{}, nil
	}}

	manager.runJob(context.Background(), job.ID)

	if backendCalled {
		t.Fatal("runJob reached the backend after running state persistence failed")
	}
	gotJob, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("GetJob(%q) did not find queued job", job.ID)
	}
	if gotJob.State != model.JobStateQueued {
		t.Fatalf("job state = %q, want %q", gotJob.State, model.JobStateQueued)
	}
	manager.mu.Lock()
	gotCodebase := manager.codebases[job.CodebaseID]
	manager.mu.Unlock()
	if gotCodebase.Status != model.CodebaseStatusPending || gotCodebase.ActiveJobID != job.ID {
		t.Fatalf("codebase after failed running persistence = %+v, want pending with active job %q", gotCodebase, job.ID)
	}
}

func TestRunJobAsyncRetriesRunningStatePersistence(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	job := newQueuedJob(
		"cb-run-persistence-retry",
		repoPath,
		repoPath,
		testClientInfo(),
		string(jobOperationIndex),
		false,
		defaultIndexConfig(),
		emptyAdmissionBudget,
		clock.Now(),
	)
	manager.mu.Lock()
	manager.codebases[job.CodebaseID] = model.Codebase{
		ID:              job.CodebaseID,
		CanonicalPath:   repoPath,
		Status:          model.CodebaseStatusPending,
		ActiveJobID:     job.ID,
		EffectiveConfig: job.Config,
	}
	manager.jobs[job.ID] = job
	if err := manager.saveLocked(); err != nil {
		manager.mu.Unlock()
		t.Fatalf("saveLocked returned error: %v", err)
	}
	manager.mu.Unlock()
	manager.jobJournal.close()
	manager.jobJournal = nil
	originalAppend := manager.appendJobEvent
	var runningEvents atomic.Int32
	retried := make(chan struct{})
	manager.appendJobEvent = func(path string, event model.JobEvent) error {
		if event.Event == "job_running" {
			if runningEvents.Add(1) == 1 {
				return errors.New("journal temporarily unavailable")
			}
			select {
			case <-retried:
			default:
				close(retried)
			}
		}
		return originalAppend(path, event)
	}
	enteredBackend := make(chan struct{})
	manager.runner = fakeRunner{indexOne: func(ctx context.Context, _ string, _ string, _ model.IndexConfig) (indexer.OneFileResult, error) {
		select {
		case <-enteredBackend:
		default:
			close(enteredBackend)
		}
		<-ctx.Done()
		return indexer.OneFileResult{}, ctx.Err()
	}}
	manager.runJobAsync(context.Background(), job.ID)
	manager.mu.Lock()
	done := manager.done[job.ID]
	manager.mu.Unlock()
	defer func() {
		manager.mu.Lock()
		stop := manager.cancels[job.ID]
		manager.mu.Unlock()
		if stop != nil {
			stop()
		}
	}()

	select {
	case <-retried:
	case <-time.After(3 * time.Second):
		t.Fatal("runJobAsync did not retry running-state persistence")
	}
	select {
	case <-enteredBackend:
	case <-time.After(3 * time.Second):
		t.Fatal("retried job did not reach the backend")
	}
	manager.mu.Lock()
	stop := manager.cancels[job.ID]
	manager.mu.Unlock()
	if stop == nil {
		t.Fatal("retried job had no cancellation handle")
	}
	stop()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("retried job did not stop after cancellation")
	}
}

func TestRunJobAsyncStopsAfterPersistentRunningStateFailure(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	job := newQueuedJob(
		"cb-run-persistence-exhausted",
		repoPath,
		repoPath,
		testClientInfo(),
		string(jobOperationIndex),
		false,
		defaultIndexConfig(),
		emptyAdmissionBudget,
		clock.Now(),
	)
	manager.mu.Lock()
	manager.codebases[job.CodebaseID] = model.Codebase{
		ID:              job.CodebaseID,
		CanonicalPath:   repoPath,
		Status:          model.CodebaseStatusPending,
		ActiveJobID:     job.ID,
		EffectiveConfig: job.Config,
	}
	manager.jobs[job.ID] = job
	if err := manager.saveLocked(); err != nil {
		manager.mu.Unlock()
		t.Fatalf("saveLocked returned error: %v", err)
	}
	manager.mu.Unlock()
	manager.jobJournal.close()
	manager.jobJournal = nil
	var runningEvents atomic.Int32
	manager.appendJobEvent = func(_ string, event model.JobEvent) error {
		if event.Event == "job_running" {
			runningEvents.Add(1)
			return errors.New("journal unavailable")
		}
		return nil
	}
	backendCalled := false
	manager.runner = fakeRunner{index: func(context.Context, string, model.IndexConfig, func(indexer.Progress)) (indexer.Result, error) {
		backendCalled = true
		return indexer.Result{}, nil
	}}

	manager.runJobAsync(context.Background(), job.ID)
	manager.mu.Lock()
	done := manager.done[job.ID]
	manager.mu.Unlock()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("runJobAsync did not stop after persistent running-state failure")
	}

	if backendCalled {
		t.Fatal("persistently failing job reached the backend")
	}
	if got := runningEvents.Load(); got != jobStartRetryAttempts {
		t.Fatalf("job_running attempts = %d, want %d", got, jobStartRetryAttempts)
	}
	gotJob, found := manager.GetJob(job.ID)
	if !found || gotJob.State != model.JobStateCancelled {
		t.Fatalf("job after exhausted retries = %+v found=%v, want cancelled", gotJob, found)
	}
	manager.mu.Lock()
	gotCodebase := manager.codebases[job.CodebaseID]
	_, hasCancel := manager.cancels[job.ID]
	_, hasDone := manager.done[job.ID]
	manager.mu.Unlock()
	if gotCodebase.ActiveJobID != "" {
		t.Fatalf("codebase active job = %q, want empty", gotCodebase.ActiveJobID)
	}
	if hasCancel || hasDone {
		t.Fatalf("exhausted retry left cancel=%v done=%v, want both removed", hasCancel, hasDone)
	}
}

func TestUpdateJobRunningRejectsStalePendingOwner(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	job := newQueuedJob(
		"cb-stale-pending-owner",
		repoPath,
		repoPath,
		testClientInfo(),
		string(jobOperationIndex),
		false,
		defaultIndexConfig(),
		emptyAdmissionBudget,
		clock.Now(),
	)
	manager.mu.Lock()
	manager.codebases[job.CodebaseID] = model.Codebase{
		ID:              job.CodebaseID,
		CanonicalPath:   repoPath,
		Status:          model.CodebaseStatusPending,
		ActiveJobID:     "newer-job",
		EffectiveConfig: job.Config,
	}
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	if err := manager.updateJobRunning(job); err == nil {
		t.Fatal("updateJobRunning accepted stale pending ownership")
	}
	gotJob, found := manager.GetJob(job.ID)
	if !found || gotJob.State != model.JobStateQueued {
		t.Fatalf("stale job after rejection = %+v found=%v, want queued", gotJob, found)
	}
	manager.mu.Lock()
	gotCodebase := manager.codebases[job.CodebaseID]
	manager.mu.Unlock()
	if gotCodebase.Status != model.CodebaseStatusPending || gotCodebase.ActiveJobID != "newer-job" {
		t.Fatalf("codebase after stale job = %+v, want newer pending owner", gotCodebase)
	}
}

func TestUpdateJobRunningRejectsStaleIndexingOwner(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	job := newQueuedJob(
		"cb-stale-indexing-owner",
		repoPath,
		repoPath,
		testClientInfo(),
		string(jobOperationSync),
		false,
		defaultIndexConfig(),
		emptyAdmissionBudget,
		clock.Now(),
	)
	manager.mu.Lock()
	manager.codebases[job.CodebaseID] = model.Codebase{
		ID:              job.CodebaseID,
		CanonicalPath:   repoPath,
		Status:          model.CodebaseStatusIndexing,
		ActiveJobID:     "newer-job",
		EffectiveConfig: job.Config,
	}
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	if err := manager.updateJobRunning(job); err == nil {
		t.Fatal("updateJobRunning accepted stale indexing ownership")
	}
	gotJob, found := manager.GetJob(job.ID)
	if !found || gotJob.State != model.JobStateQueued {
		t.Fatalf("stale job after rejection = %+v found=%v, want queued", gotJob, found)
	}
	manager.mu.Lock()
	gotCodebase := manager.codebases[job.CodebaseID]
	manager.mu.Unlock()
	if gotCodebase.Status != model.CodebaseStatusIndexing || gotCodebase.ActiveJobID != "newer-job" {
		t.Fatalf("codebase after stale job = %+v, want newer indexing owner", gotCodebase)
	}
}

func TestUpdateJobRunningRejectsMissingJob(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	job := newQueuedJob(
		"cb-missing-running-job",
		repoPath,
		repoPath,
		testClientInfo(),
		string(jobOperationIndex),
		false,
		defaultIndexConfig(),
		emptyAdmissionBudget,
		clock.Now(),
	)

	if err := manager.updateJobRunning(job); err == nil {
		t.Fatal("updateJobRunning accepted a missing job")
	}
}

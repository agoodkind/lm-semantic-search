package daemon

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
)

// A codebase marked Failed by a past run is presented as Indexed when its
// semantic collection currently exists: a status check reflects the current
// usable state, not a stale failure that no longer blocks anything.
func TestGetIndexPresentsIndexedWhenCollectionPresentDespiteFailed(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}

	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusFailed
	codebase.LastFailedRun = &model.IndexRunFailure{
		Message:  "embedding endpoint is unreachable",
		FailedAt: time.Now(),
	}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	manager.semantic = &fakeSemantic{
		hasCollectionForPath: func(context.Context, string) (bool, error) { return true, nil },
	}

	got, _, found, _, err := manager.GetIndex(context.Background(), repoPath)
	if err != nil || !found {
		t.Fatalf("GetIndex returned err=%v found=%v", err, found)
	}
	if got.Status != model.CodebaseStatusIndexed {
		t.Fatalf("status = %q, want Indexed since the collection is present now", got.Status)
	}
	if got.LastFailedRun != nil {
		t.Fatalf("LastFailedRun = %+v, want nil when presented as current", got.LastFailedRun)
	}
}

// A Failed codebase whose collection is genuinely missing keeps showing Failed.
func TestGetIndexKeepsFailedWhenCollectionAbsent(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}

	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusFailed
	codebase.LastFailedRun = &model.IndexRunFailure{Message: "boom", FailedAt: time.Now()}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	manager.semantic = &fakeSemantic{
		hasCollectionForPath: func(context.Context, string) (bool, error) { return false, nil },
	}

	got, _, found, _, err := manager.GetIndex(context.Background(), repoPath)
	if err != nil || !found {
		t.Fatalf("GetIndex returned err=%v found=%v", err, found)
	}
	if got.Status != model.CodebaseStatusFailed {
		t.Fatalf("status = %q, want Failed since the collection is absent (a current blocker)", got.Status)
	}
}

// A transient failure (at-capacity embedder) does not mark the codebase failed;
// it stays at last-good, and the job carries a sanitized, retryable error.
func TestUpdateJobFailedTransientKeepsCodebaseUsable(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}

	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	job := model.Job{ID: "job-transient", CodebaseID: codebase.ID, State: model.JobStateRunning}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	manager.updateJobFailed(context.Background(), job.ID, adapterr.NewEmbedderBusy(errors.New("429 capacity_exceeded")))

	manager.mu.Lock()
	gotCodebase := manager.codebases[codebase.ID]
	gotJob := manager.jobs[job.ID]
	manager.mu.Unlock()

	if gotCodebase.Status != model.CodebaseStatusIndexed {
		t.Fatalf("codebase status = %q, want Indexed (a transient failure must not fail it)", gotCodebase.Status)
	}
	if gotCodebase.LastFailedRun != nil {
		t.Fatalf("LastFailedRun = %+v, want nil for a transient failure", gotCodebase.LastFailedRun)
	}
	if gotJob.State != model.JobStateFailed {
		t.Fatalf("job state = %q, want Failed", gotJob.State)
	}
	if gotJob.Error == nil || !gotJob.Error.Retryable {
		t.Fatalf("job error = %+v, want a retryable error", gotJob.Error)
	}
	if strings.Contains(gotJob.Error.Message, "429") || strings.Contains(gotJob.Error.Message, "capacity_exceeded") {
		t.Fatalf("job error message leaked implementation detail: %q", gotJob.Error.Message)
	}
}

// A terminal failure marks the codebase failed and persists only the sanitized
// class message, never the wrapped cause.
func TestUpdateJobFailedTerminalPersistsSanitizedMessage(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}

	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	job := model.Job{ID: "job-terminal", CodebaseID: codebase.ID, State: model.JobStateRunning}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	manager.updateJobFailed(context.Background(), job.ID, adapterr.NewEmbedderRejected(errors.New("400 invalid_request model=bad")))

	manager.mu.Lock()
	gotCodebase := manager.codebases[codebase.ID]
	manager.mu.Unlock()

	if gotCodebase.Status != model.CodebaseStatusFailed {
		t.Fatalf("status = %q, want Failed for a terminal failure", gotCodebase.Status)
	}
	if gotCodebase.LastFailedRun == nil {
		t.Fatal("LastFailedRun nil, want it recorded for a terminal failure")
	}
	if strings.Contains(gotCodebase.LastFailedRun.Message, "400") || strings.Contains(gotCodebase.LastFailedRun.Message, "invalid_request") {
		t.Fatalf("persisted message leaked implementation detail: %q", gotCodebase.LastFailedRun.Message)
	}
	if !strings.Contains(gotCodebase.LastFailedRun.Message, "rejected") {
		t.Fatalf("persisted message dropped the class message: %q", gotCodebase.LastFailedRun.Message)
	}
}

// A cancellation is not a failure: the codebase stays at last-good.
func TestUpdateJobCancelledDoesNotFailCodebase(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}

	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	job := model.Job{ID: "job-cancel", CodebaseID: codebase.ID, State: model.JobStateCancelling}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	manager.updateJobCancelled(context.Background(), job.ID)

	manager.mu.Lock()
	gotCodebase := manager.codebases[codebase.ID]
	manager.mu.Unlock()

	if gotCodebase.Status != model.CodebaseStatusIndexed {
		t.Fatalf("status = %q, want Indexed (a cancellation is not a failure)", gotCodebase.Status)
	}
	if gotCodebase.LastFailedRun != nil {
		t.Fatalf("LastFailedRun = %+v, want nil after a cancellation", gotCodebase.LastFailedRun)
	}
}

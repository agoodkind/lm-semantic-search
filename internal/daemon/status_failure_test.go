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

// A shared-infrastructure failure never marks the codebase failed, whether it is
// a self-healing outage (unreachable) or a global config error (rejected). The
// codebase stays resumable and the cause lives on the job, not the codebase. An
// unreachable outage is retryable; a rejected config error is not, yet still must
// not fail the codebase.
func TestUpdateJobFailedInfraKeepsCodebaseUsable(t *testing.T) {
	cases := []struct {
		name      string
		runErr    error
		retryable bool
	}{
		{"unreachable", adapterr.NewEmbedderUnreachable(errors.New("connection refused")), true},
		{"rejected", adapterr.NewEmbedderRejected(errors.New("400 invalid_request model=bad")), false},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			manager, _, repoPath := newTestManager(t)
			canonical, err := filepath.EvalSymlinks(repoPath)
			if err != nil {
				t.Fatalf("EvalSymlinks returned error: %v", err)
			}

			codebase := newCodebaseRecord(canonical)
			codebase.Status = model.CodebaseStatusIndexing
			job := model.Job{ID: "job-" + testCase.name, CodebaseID: codebase.ID, State: model.JobStateRunning}
			manager.mu.Lock()
			manager.codebases[codebase.ID] = codebase
			manager.jobs[job.ID] = job
			manager.mu.Unlock()

			manager.updateJobFailed(context.Background(), job.ID, testCase.runErr)

			manager.mu.Lock()
			gotCodebase := manager.codebases[codebase.ID]
			gotJob := manager.jobs[job.ID]
			manager.mu.Unlock()

			if gotCodebase.Status == model.CodebaseStatusFailed {
				t.Fatalf("codebase status = Failed, want it left usable for a shared-infrastructure failure")
			}
			if gotCodebase.LastFailedRun != nil {
				t.Fatalf("LastFailedRun = %+v, want nil for a shared-infrastructure failure", gotCodebase.LastFailedRun)
			}
			if gotJob.State != model.JobStateFailed {
				t.Fatalf("job state = %q, want Failed", gotJob.State)
			}
			if gotJob.Error == nil || gotJob.Error.Retryable != testCase.retryable {
				t.Fatalf("job error = %+v, want Retryable=%v", gotJob.Error, testCase.retryable)
			}
		})
	}
}

// A fault local to the codebase marks it failed and persists only the sanitized
// envelope, never the wrapped cause.
func TestUpdateJobFailedLocalErrorMarksFailed(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}

	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	job := model.Job{ID: "job-local", CodebaseID: codebase.ID, State: model.JobStateRunning}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	manager.updateJobFailed(context.Background(), job.ID, adapterr.NewInternal("boom: secret detail", errors.New("secret detail")))

	manager.mu.Lock()
	gotCodebase := manager.codebases[codebase.ID]
	manager.mu.Unlock()

	if gotCodebase.Status != model.CodebaseStatusFailed {
		t.Fatalf("status = %q, want Failed for a codebase-local error", gotCodebase.Status)
	}
	if gotCodebase.LastFailedRun == nil {
		t.Fatal("LastFailedRun nil, want it recorded for a codebase-local failure")
	}
	if strings.Contains(gotCodebase.LastFailedRun.Message, "secret detail") {
		t.Fatalf("persisted message leaked the wrapped cause: %q", gotCodebase.LastFailedRun.Message)
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

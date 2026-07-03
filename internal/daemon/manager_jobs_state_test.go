package daemon

import (
	"context"
	"testing"

	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/model"
)

func TestUpdateJobCompletedDoesNotOverwriteCancelledJob(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	codebase := newCodebaseRecord(repoPath)
	job := model.Job{ID: "job-cancelled", CodebaseID: codebase.ID, State: model.JobStateCancelled}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	manager.updateJobCompleted(context.Background(), job.ID, indexer.Result{IndexedFiles: 1, TotalChunks: 1})

	manager.mu.Lock()
	recorded := manager.jobs[job.ID]
	manager.mu.Unlock()
	if recorded.State != model.JobStateCancelled {
		t.Fatalf("job state = %q, want %q", recorded.State, model.JobStateCancelled)
	}
}

func TestUpdateJobCompletedTurnsCancellingIntoCancelled(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	codebase := newCodebaseRecord(repoPath)
	job := model.Job{ID: "job-cancelling", CodebaseID: codebase.ID, State: model.JobStateCancelling}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	manager.updateJobCompleted(context.Background(), job.ID, indexer.Result{IndexedFiles: 1, TotalChunks: 1})

	manager.mu.Lock()
	recorded := manager.jobs[job.ID]
	manager.mu.Unlock()
	if recorded.State != model.JobStateCancelled {
		t.Fatalf("job state = %q, want %q", recorded.State, model.JobStateCancelled)
	}
	if recorded.CompletedAt == nil {
		t.Fatal("CompletedAt is nil, want cancelled terminal timestamp")
	}
}

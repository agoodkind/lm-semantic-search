package daemon

import (
	"context"
	"errors"
	"path/filepath"
	"slices"
	"testing"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
)

// removedWorktreeSemantic is a fakeSemantic that reports the worktree collection
// as present so the repair pass exercises the auto-clean path rather than a
// rebuild.
func worktreeSemantic(name string) *fakeSemantic {
	return &fakeSemantic{
		collectionName:  func(string) string { return name },
		listCollections: func(context.Context) ([]string, error) { return []string{name}, nil },
	}
}

// TestPlanClassifiesRemovedWorktreeAsCleanup proves a codebase whose root was a
// linked worktree that git no longer tracks is routed to auto-clean, not left
// as a stale entry.
func TestPlanClassifiesRemovedWorktreeAsCleanup(t *testing.T) {
	manager, _, _ := newTestManager(t)
	// A common dir that exists but tracks no worktree (no worktrees/ subdir), so
	// WorktreeTracked returns false: the positive "removed" signal.
	commonDir := t.TempDir()
	codebase := newCodebaseRecord(filepath.Join(t.TempDir(), "gone-worktree"))
	codebase.Status = model.CodebaseStatusIndexed
	codebase.WorktreeCommonDir = commonDir
	codebase.CollectionName = "wt_collection"
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	manager.semantic = worktreeSemantic("wt_collection")

	_, cleanups, err := manager.planMissingCollectionRepairs(context.Background())
	if err != nil {
		t.Fatalf("planMissingCollectionRepairs returned error: %v", err)
	}
	if !slices.Contains(cleanups, codebase.CanonicalPath) {
		t.Fatalf("cleanups = %v, want it to contain %s", cleanups, codebase.CanonicalPath)
	}
}

// TestPlanMarksMissingNonWorktree proves a plain (non-worktree) missing
// directory becomes the missing state and is kept, not deleted.
func TestPlanMarksMissingNonWorktree(t *testing.T) {
	manager, _, _ := newTestManager(t)
	codebase := newCodebaseRecord(filepath.Join(t.TempDir(), "gone-plain"))
	codebase.Status = model.CodebaseStatusIndexed
	codebase.CollectionName = "plain_collection"
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	manager.semantic = worktreeSemantic("plain_collection")

	_, cleanups, err := manager.planMissingCollectionRepairs(context.Background())
	if err != nil {
		t.Fatalf("planMissingCollectionRepairs returned error: %v", err)
	}
	if len(cleanups) != 0 {
		t.Fatalf("cleanups = %v, want none for a plain missing directory", cleanups)
	}
	manager.mu.Lock()
	got := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if got.Status != model.CodebaseStatusMissing {
		t.Fatalf("status = %q, want missing", got.Status)
	}
	if got.LastFailedRun != nil {
		t.Fatalf("LastFailedRun = %+v, want nil; a missing directory is not a failure", got.LastFailedRun)
	}
}

// TestRepairAutoCleansRemovedWorktree proves the full pass drops a removed
// worktree's registration and collection.
func TestRepairAutoCleansRemovedWorktree(t *testing.T) {
	manager, _, _ := newTestManager(t)
	commonDir := t.TempDir()
	gonePath := filepath.Join(t.TempDir(), "gone-worktree")
	codebase := newCodebaseRecord(gonePath)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.WorktreeCommonDir = commonDir
	codebase.CollectionName = "wt_collection"
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	semantic := worktreeSemantic("wt_collection")
	manager.semantic = semantic

	manager.RepairMissingCollections(context.Background())

	manager.mu.Lock()
	_, stillTracked := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if stillTracked {
		t.Fatalf("removed-worktree codebase is still tracked, want it auto-cleaned")
	}
	if !slices.Contains(semantic.dropped, gonePath) {
		t.Fatalf("semantic.Drop was not called for the worktree path; dropped = %v", semantic.dropped)
	}
}

// TestUpdateJobFailedSourceMissingSetsMissing proves a run that fails because
// the source directory is gone records the missing state, not a failure.
func TestUpdateJobFailedSourceMissingSetsMissing(t *testing.T) {
	manager, _, _ := newTestManager(t)
	codebase := newCodebaseRecord(filepath.Join(t.TempDir(), "vanished"))
	codebase.Status = model.CodebaseStatusIndexing
	job := model.Job{ID: "job-missing", CodebaseID: codebase.ID, State: model.JobStateRunning}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	manager.updateJobFailed(context.Background(), job.ID, adapterr.NewInternal("read dir: no such file or directory", errors.New("boom")))

	manager.mu.Lock()
	got := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if got.Status != model.CodebaseStatusMissing {
		t.Fatalf("status = %q, want missing for a vanished source directory", got.Status)
	}
	if got.LastFailedRun != nil {
		t.Fatalf("LastFailedRun = %+v, want nil for a missing source", got.LastFailedRun)
	}
}

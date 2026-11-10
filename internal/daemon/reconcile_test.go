package daemon

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/model"
)

func reconcileTestCodebase(t *testing.T, manager *Manager, repoPath string, status model.CodebaseStatus, collectionName string) model.Codebase {
	t.Helper()
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonical)
	codebase.Status = status
	codebase.CollectionName = collectionName
	codebase.LastFailedRun = &model.IndexRunFailure{Message: "embedding endpoint is unreachable", FailedAt: time.Now()}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	return codebase
}

func presentCollectionSemantic(name string) *fakeSemantic {
	return &fakeSemantic{
		collectionName:  func(string) string { return name },
		listCollections: func(context.Context) ([]string, error) { return []string{name}, nil },
	}
}

// AC1, AC2, AC8: a Failed codebase whose collection is present heals to Indexed
// with no rebuild, and a second pass is a no-op.
func TestRepairHealsFailedWhenCollectionPresent(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	codebase := reconcileTestCodebase(t, manager, repoPath, model.CodebaseStatusFailed, "present_collection")
	manager.semantic = presentCollectionSemantic("present_collection")

	manager.RepairMissingCollections(context.Background())

	manager.mu.Lock()
	got := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if got.Status != model.CodebaseStatusIndexed {
		t.Fatalf("status = %q, want Indexed", got.Status)
	}
	if got.LastFailedRun != nil {
		t.Fatalf("LastFailedRun = %+v, want nil after heal", got.LastFailedRun)
	}
	if jobs := manager.ListJobs(""); len(jobs) != 0 {
		t.Fatalf("queued %d jobs, want 0 (heal is pure status reconciliation)", len(jobs))
	}

	manager.RepairMissingCollections(context.Background())
	manager.mu.Lock()
	again := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if again.Status != model.CodebaseStatusIndexed {
		t.Fatalf("status after a second pass = %q, want Indexed (idempotent)", again.Status)
	}
}

// AC5: a Stale codebase whose collection is present also heals to Indexed.
func TestRepairHealsStaleWhenCollectionPresent(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	codebase := reconcileTestCodebase(t, manager, repoPath, model.CodebaseStatusStale, "present_collection")
	manager.semantic = presentCollectionSemantic("present_collection")

	manager.RepairMissingCollections(context.Background())

	manager.mu.Lock()
	got := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if got.Status != model.CodebaseStatusIndexed {
		t.Fatalf("stale+present status = %q, want Indexed", got.Status)
	}
	if got.LastFailedRun != nil {
		t.Fatalf("LastFailedRun = %+v, want nil after heal", got.LastFailedRun)
	}
}

// AC3: a Failed codebase whose collection is genuinely absent stays Failed,
// keeps its failure record, and queues no rebuild.
func TestRepairKeepsFailedWhenCollectionAbsent(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	codebase := reconcileTestCodebase(t, manager, repoPath, model.CodebaseStatusFailed, "missing_collection")
	manager.semantic = &fakeSemantic{
		collectionName:  func(string) string { return "missing_collection" },
		listCollections: func(context.Context) ([]string, error) { return []string{}, nil },
	}

	manager.RepairMissingCollections(context.Background())

	manager.mu.Lock()
	got := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if got.Status != model.CodebaseStatusFailed {
		t.Fatalf("failed+missing status = %q, want Failed (a genuine current blocker)", got.Status)
	}
	if got.LastFailedRun == nil {
		t.Fatal("LastFailedRun cleared, want it preserved for a genuine failure")
	}
	if jobs := manager.ListJobs(""); len(jobs) != 0 {
		t.Fatalf("queued %d jobs for failed+missing, want 0", len(jobs))
	}
}

// AC4: a codebase with an active job is left untouched regardless of presence.
func TestRepairLeavesActiveJobCodebaseUntouched(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusFailed
	codebase.CollectionName = "present_collection"
	codebase.LastFailedRun = &model.IndexRunFailure{Message: "x", FailedAt: time.Now()}
	job := model.Job{ID: "active-job", CodebaseID: codebase.ID, State: model.JobStateRunning}
	codebase.ActiveJobID = job.ID
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.jobs[job.ID] = job
	manager.mu.Unlock()
	manager.semantic = presentCollectionSemantic("present_collection")

	manager.RepairMissingCollections(context.Background())

	manager.mu.Lock()
	got := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if got.Status != model.CodebaseStatusFailed {
		t.Fatalf("status = %q, want Failed left untouched while a job is active", got.Status)
	}
	if got.LastFailedRun == nil {
		t.Fatal("LastFailedRun cleared while a job is active, want it preserved")
	}
}

// TestPlanRepairsResumesPendingWithLostFirstBuild pins the strand this pass
// exists to heal: a first build cancelled while queued leaves its codebase at
// pending with no live job and no coalesced request, and nothing else ever
// re-queues it. The plan must include it so the sweep restarts the build.
func TestPlanRepairsResumesPendingWithLostFirstBuild(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusPending
	codebase.ActiveJobID = "job-cancelled-at-shutdown"
	codebase.CollectionName = "pending_lost_collection"
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.jobs["job-cancelled-at-shutdown"] = model.Job{
		ID:         "job-cancelled-at-shutdown",
		CodebaseID: codebase.ID,
		State:      model.JobStateCancelled,
	}
	manager.mu.Unlock()
	manager.semantic = &fakeSemantic{
		collectionName:  func(string) string { return "pending_lost_collection" },
		listCollections: func(context.Context) ([]string, error) { return []string{}, nil },
	}

	plans, _, err := manager.planMissingCollectionRepairs(context.Background())
	if err != nil {
		t.Fatalf("planMissingCollectionRepairs returned error: %v", err)
	}
	found := false
	for _, plan := range plans {
		if plan.codebaseID == codebase.ID {
			found = true
		}
	}
	if !found {
		t.Fatalf("pending codebase with a lost first build was not queued for retry; plans=%+v", plans)
	}
}

// TestPlanRepairsLeavesPendingWithLiveOrCoalescedBuildAlone pins the healthy
// side of the same gate: a pending codebase whose first build is live, or whose
// next build waits in the coalesced slot, is normal queued work the repair pass
// must not duplicate.
func TestPlanRepairsLeavesPendingWithLiveOrCoalescedBuildAlone(t *testing.T) {
	t.Run("live first build", func(t *testing.T) {
		manager, _, repoPath := newTestManager(t)
		canonical, err := filepath.EvalSymlinks(repoPath)
		if err != nil {
			t.Fatalf("EvalSymlinks returned error: %v", err)
		}
		codebase := newCodebaseRecord(canonical)
		codebase.Status = model.CodebaseStatusPending
		codebase.ActiveJobID = "job-live-first-build"
		codebase.CollectionName = "pending_live_collection"
		manager.mu.Lock()
		manager.codebases[codebase.ID] = codebase
		manager.jobs["job-live-first-build"] = model.Job{
			ID:         "job-live-first-build",
			CodebaseID: codebase.ID,
			State:      model.JobStateQueued,
		}
		manager.mu.Unlock()
		manager.semantic = &fakeSemantic{
			collectionName:  func(string) string { return "pending_live_collection" },
			listCollections: func(context.Context) ([]string, error) { return []string{}, nil },
		}

		plans, _, err := manager.planMissingCollectionRepairs(context.Background())
		if err != nil {
			t.Fatalf("planMissingCollectionRepairs returned error: %v", err)
		}
		if len(plans) != 0 {
			t.Fatalf("pending codebase with a live first build was planned for repair; plans=%+v", plans)
		}
	})

	t.Run("coalesced request", func(t *testing.T) {
		manager, _, repoPath := newTestManager(t)
		canonical, err := filepath.EvalSymlinks(repoPath)
		if err != nil {
			t.Fatalf("EvalSymlinks returned error: %v", err)
		}
		codebase := newCodebaseRecord(canonical)
		codebase.Status = model.CodebaseStatusPending
		codebase.CollectionName = "pending_coalesced_collection"
		manager.mu.Lock()
		manager.codebases[codebase.ID] = codebase
		manager.pendingCodeJobs[codebase.ID] = pendingCodeRequest{}
		manager.mu.Unlock()
		manager.semantic = &fakeSemantic{
			collectionName:  func(string) string { return "pending_coalesced_collection" },
			listCollections: func(context.Context) ([]string, error) { return []string{}, nil },
		}

		plans, _, err := manager.planMissingCollectionRepairs(context.Background())
		if err != nil {
			t.Fatalf("planMissingCollectionRepairs returned error: %v", err)
		}
		if len(plans) != 0 {
			t.Fatalf("pending codebase with a coalesced request was planned for repair; plans=%+v", plans)
		}
	})
}

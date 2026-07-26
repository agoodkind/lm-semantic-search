package daemon

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// indexedTestCodebase registers one indexed codebase on the manager and returns
// it, so a self-check has a real target to query.
func indexedTestCodebase(t *testing.T, manager *Manager, repoPath string) model.Codebase {
	t.Helper()

	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.LastSuccessfulRun = &model.IndexRunSummary{
		IndexedFiles: 1,
		TotalChunks:  1,
		Status:       "completed",
		CompletedAt:  clock.Now(),
	}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	return codebase
}

// healthModeNow reads the raw health record. DependencyHealth() applies a
// reconnect shortcut that clears a store-unavailable mode when the backend
// reports available, which would mask exactly what these tests assert, so they
// read the record the self-check wrote.
func healthModeNow(manager *Manager) dependencyMode {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	return manager.health.Mode
}

// A boot that can still answer a real query reports one passing line and clears
// the store banner the boot dial left behind.
func TestBootSelfCheckPassesAndClearsBootBanner(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	codebase := indexedTestCodebase(t, manager, repoPath)

	queried := make(chan string, 1)
	manager.semantic = &fakeSemantic{
		search: func(_ context.Context, codebasePath string, _ string, limit int32, _ []string, _ string) ([]model.StoredChunk, error) {
			queried <- codebasePath
			if limit != bootSelfCheckLimit {
				t.Errorf("self-check search limit = %d, want %d", limit, bootSelfCheckLimit)
			}
			return []model.StoredChunk{{Content: "package main", RelativePath: "main.go"}}, nil
		},
	}
	manager.mu.Lock()
	manager.health = dependencyHealth{Mode: dependencyStoreUnavailable, Since: clock.Now(), LastHealthyAt: time.Time{}}
	manager.mu.Unlock()

	outcome, err := manager.runBootSelfCheck(context.Background())
	if err != nil {
		t.Fatalf("runBootSelfCheck returned error: %v", err)
	}
	if outcome != bootSelfCheckPassed {
		t.Fatalf("outcome = %q, want %q", outcome, bootSelfCheckPassed)
	}
	select {
	case got := <-queried:
		if got != codebase.CanonicalPath {
			t.Fatalf("self-check queried %q, want the indexed codebase %q", got, codebase.CanonicalPath)
		}
	default:
		t.Fatal("self-check ran no query; it must exercise the real read path")
	}
	if got := healthModeNow(manager); got != dependencyHealthy {
		t.Fatalf("health mode = %q after a passing self-check, want %q", got, dependencyHealthy)
	}
}

// A restore that left the collection unusable must show up as a failed check
// that degrades the shared readiness record, so no surface reports ready while
// the data cannot answer.
func TestBootSelfCheckFailureDegradesReadiness(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	indexedTestCodebase(t, manager, repoPath)

	manager.semantic = &fakeSemantic{
		search: func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error) {
			return nil, semantic.ErrUnavailable
		},
	}

	outcome, err := manager.runBootSelfCheck(context.Background())
	if outcome != bootSelfCheckFailed {
		t.Fatalf("outcome = %q, want %q", outcome, bootSelfCheckFailed)
	}
	if !errors.Is(err, semantic.ErrUnavailable) {
		t.Fatalf("runBootSelfCheck error = %v, want it to carry the store outage", err)
	}
	if got := healthModeNow(manager); got != dependencyStoreUnavailable {
		t.Fatalf("health mode = %q after a failed self-check, want %q", got, dependencyStoreUnavailable)
	}
}

// A collection that is gone is the bad-restore case the check exists to catch,
// and it must be reported rather than swallowed.
func TestBootSelfCheckReportsMissingCollection(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	indexedTestCodebase(t, manager, repoPath)

	manager.semantic = &fakeSemantic{
		search: func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error) {
			return nil, semantic.ErrCollectionMissing
		},
	}

	outcome, err := manager.runBootSelfCheck(context.Background())
	if outcome != bootSelfCheckFailed {
		t.Fatalf("outcome = %q, want %q", outcome, bootSelfCheckFailed)
	}
	if !errors.Is(err, semantic.ErrCollectionMissing) {
		t.Fatalf("runBootSelfCheck error = %v, want the missing-collection classification", err)
	}
}

// A daemon with nothing indexed has no data to prove usable, so the check skips
// without touching the health record or issuing a query.
func TestBootSelfCheckSkipsWithNothingIndexed(t *testing.T) {
	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{
		search: func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error) {
			t.Error("self-check queried the store with no indexed codebase")
			return nil, nil
		},
	}

	outcome, err := manager.runBootSelfCheck(context.Background())
	if err != nil {
		t.Fatalf("runBootSelfCheck returned error: %v", err)
	}
	if outcome != bootSelfCheckSkipped {
		t.Fatalf("outcome = %q, want %q", outcome, bootSelfCheckSkipped)
	}
	if got := healthModeNow(manager); got != dependencyHealthy {
		t.Fatalf("health mode = %q after a skipped self-check, want it untouched (%q)", got, dependencyHealthy)
	}
}

// The check never gates startup: StartBootSelfCheck returns while a hung store
// still has the probe query parked, so the daemon goes on to serve.
func TestStartBootSelfCheckDoesNotBlockStartup(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	indexedTestCodebase(t, manager, repoPath)

	entered := make(chan struct{})
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	manager.bootSelfCheckDelay = time.Millisecond
	manager.semantic = &fakeSemantic{
		search: func(ctx context.Context, _ string, _ string, _ int32, _ []string, _ string) ([]model.StoredChunk, error) {
			close(entered)
			select {
			case <-release:
			case <-ctx.Done():
			}
			return nil, ctx.Err()
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	returnedAt := make(chan time.Duration, 1)
	startedAt := clock.Now()
	manager.StartBootSelfCheck(ctx)
	returnedAt <- clock.Now().Sub(startedAt)

	if elapsed := <-returnedAt; elapsed > time.Second {
		t.Fatalf("StartBootSelfCheck took %s to return; startup must not wait on the check", elapsed)
	}

	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("the background self-check never ran")
	}
	// The hung probe is still parked here, which is the point: startup already
	// returned and the daemon would be serving.
}

// The target is the most recently completed index, so the probe hits the data
// most likely to matter and two boots on the same registry pick the same one.
func TestBootSelfCheckTargetPicksMostRecentIndex(t *testing.T) {
	manager, _, _ := newTestManager(t)

	older := newCodebaseRecord("/tmp/older-repo")
	older.Status = model.CodebaseStatusIndexed
	older.LastSuccessfulRun = &model.IndexRunSummary{CompletedAt: clock.Now().Add(-2 * time.Hour)}
	newer := newCodebaseRecord("/tmp/newer-repo")
	newer.Status = model.CodebaseStatusIndexed
	newer.LastSuccessfulRun = &model.IndexRunSummary{CompletedAt: clock.Now()}
	failed := newCodebaseRecord("/tmp/failed-repo")
	failed.Status = model.CodebaseStatusFailed
	failed.LastSuccessfulRun = &model.IndexRunSummary{CompletedAt: clock.Now().Add(time.Hour)}

	manager.mu.Lock()
	for _, codebase := range []model.Codebase{older, newer, failed} {
		manager.codebases[codebase.ID] = codebase
	}
	manager.mu.Unlock()

	target, found := manager.bootSelfCheckTarget()
	if !found {
		t.Fatal("bootSelfCheckTarget found no indexed codebase")
	}
	if target.ID != newer.ID {
		t.Fatalf("target = %q (%s), want the most recently indexed %q", target.ID, target.CanonicalPath, newer.ID)
	}
}

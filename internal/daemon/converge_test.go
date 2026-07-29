package daemon

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

// seedConvergeCodebase registers an indexed codebase rooted at repoPath, so a
// converge test starts from a tracked codebase without repeating the registry
// setup.
func seedConvergeCodebase(t *testing.T, manager *Manager, repoPath string) model.Codebase {
	t.Helper()

	indexConfig := defaultIndexConfig()
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)
	codebase := newCodebaseRecord(repoPath)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = indexConfig
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	return codebase
}

// TestConvergePathsReportsWhatItConverged proves the caller learns how many of
// the paths it handed over actually reached the index. A job built around this
// call reports that count as its scope, so a wrong count is a wrong status.
func TestConvergePathsReportsWhatItConverged(t *testing.T) {
	t.Parallel()

	manager, _, repoPath := newTestManager(t)
	codebase := seedConvergeCodebase(t, manager, repoPath)
	manager.semantic = &fakeSemantic{
		reindex: func(context.Context, string, []model.StoredChunk, []string) error { return nil },
	}

	present := filepath.Join(repoPath, "present.go")
	if err := os.WriteFile(present, []byte("package main\n\nfunc Present() {}\n"), 0o644); err != nil {
		t.Fatalf("write the present file: %v", err)
	}

	outcome, err := manager.ConvergePaths(context.Background(), codebase.ID, []string{"present.go", "absent.go"})
	if err != nil {
		t.Fatalf("ConvergePaths returned error: %v", err)
	}
	if outcome.PathsGiven != 2 {
		t.Fatalf("PathsGiven = %d, want 2", outcome.PathsGiven)
	}
	if outcome.PathsConverged != 1 {
		t.Fatalf("PathsConverged = %d, want 1; only present.go exists on disk", outcome.PathsConverged)
	}
}

// TestConvergePathsStopsBetweenPathsOnCancel proves a cancelled converge stops
// rather than finishing its list. The paths it did not reach keep their previous
// index entries, which the periodic sync repairs on its next pass.
func TestConvergePathsStopsBetweenPathsOnCancel(t *testing.T) {
	t.Parallel()

	manager, _, repoPath := newTestManager(t)
	codebase := seedConvergeCodebase(t, manager, repoPath)

	ctx, cancel := context.WithCancel(context.Background())
	// Cancel as soon as the first path reaches the store, so the second path is
	// never read.
	manager.semantic = &fakeSemantic{
		reindex: func(context.Context, string, []model.StoredChunk, []string) error {
			cancel()
			return nil
		},
	}

	names := []string{"first.go", "second.go"}
	for _, name := range names {
		body := "package main\n\nfunc " + name[:len(name)-3] + "() {}\n"
		if err := os.WriteFile(filepath.Join(repoPath, name), []byte(body), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}

	outcome, err := manager.ConvergePaths(ctx, codebase.ID, names)
	if err != nil {
		t.Fatalf("ConvergePaths returned error: %v", err)
	}
	if outcome.PathsGiven != 2 {
		t.Fatalf("PathsGiven = %d, want 2", outcome.PathsGiven)
	}
	if outcome.PathsConverged >= 2 {
		t.Fatalf("PathsConverged = %d, want fewer than 2; the cancel did not stop the loop", outcome.PathsConverged)
	}
}

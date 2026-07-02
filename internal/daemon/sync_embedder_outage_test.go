package daemon

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/model"
)

// setupThreeChangedCodebases indexes three codebases, then edits a file in each
// so every one has a non-empty merkle diff on the next sweep. It returns the
// manager and the three repo paths.
func setupThreeChangedCodebases(t *testing.T) (*Manager, []string) {
	t.Helper()
	manager, _, repo1 := newTestManager(t)

	base := t.TempDir()
	repo2 := filepath.Join(base, "repo2")
	repo3 := filepath.Join(base, "repo3")
	for _, repo := range []string{repo2, repo3} {
		if err := os.MkdirAll(repo, 0o755); err != nil {
			t.Fatalf("MkdirAll %s: %v", repo, err)
		}
		if err := os.WriteFile(filepath.Join(repo, "main.go"), []byte("package main\n"), 0o644); err != nil {
			t.Fatalf("WriteFile %s: %v", repo, err)
		}
	}

	// A faithful fake: it hashes the real on-disk content, like the real indexer.
	// So change detection is driven by actual edits and the snapshot a completed
	// sync persists converges with the file, instead of a canned hash that would
	// leave the codebase reading as perpetually changed.
	manager.runner = fakeRunner{
		index: func(_ context.Context, root string, _ model.IndexConfig, _ func(indexer.Progress)) (indexer.Result, error) {
			content, err := os.ReadFile(filepath.Join(root, "main.go"))
			if err != nil {
				return indexer.Result{}, err
			}
			return indexer.Result{
				IndexedFiles: 1,
				TotalChunks:  1,
				Chunks: []model.StoredChunk{{
					Content:       string(content),
					RelativePath:  "main.go",
					StartLine:     1,
					EndLine:       1,
					Language:      "go",
					FileExtension: ".go",
				}},
				FileHashes: map[string]string{"main.go": hashText(string(content))},
			}, nil
		},
		indexOne: func(_ context.Context, root string, relativePath string, _ model.IndexConfig) (indexer.OneFileResult, error) {
			content, err := os.ReadFile(filepath.Join(root, relativePath))
			if err != nil {
				return indexer.OneFileResult{}, err
			}
			return indexer.OneFileResult{
				Chunks: []model.StoredChunk{{
					Content:       string(content),
					RelativePath:  relativePath,
					StartLine:     1,
					EndLine:       1,
					Language:      "go",
					FileExtension: ".go",
				}},
				FileHash: hashText(string(content)),
			}, nil
		},
	}

	repos := []string{repo1, repo2, repo3}
	for _, repo := range repos {
		if _, _, _, _, err := manager.StartIndex(context.Background(), repo, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget); err != nil {
			t.Fatalf("StartIndex(%s): %v", repo, err)
		}
		waitForCodebaseStatus(t, manager, repo, model.CodebaseStatusIndexed)
	}

	// Edit each file so its content differs from the initial snapshot, which makes
	// the next sweep see a real change. The faithful fake above hashes this same
	// content, so a completed sync converges the snapshot with the file.
	for _, repo := range repos {
		if err := os.WriteFile(filepath.Join(repo, "main.go"), []byte("package main\n// edited\n"), 0o644); err != nil {
			t.Fatalf("edit %s: %v", repo, err)
		}
	}

	return manager, repos
}

func countDaemonSyncJobs(manager *Manager) int {
	count := 0
	for _, job := range manager.ListJobs("") {
		if job.Client.Name == "daemon-sync" {
			count++
		}
	}
	return count
}

func setHealthMode(manager *Manager, mode dependencyMode) {
	manager.mu.Lock()
	manager.health.Mode = mode
	manager.mu.Unlock()
}

func drainToIndexed(t *testing.T, manager *Manager, repos []string) {
	t.Helper()
	for _, repo := range repos {
		waitForCodebaseStatus(t, manager, repo, model.CodebaseStatusIndexed)
	}
}

// TestRunSyncAllSkipsWhenEmbedderUnreachable proves the sweep enqueues nothing
// while the embedder outage is already recorded, so a sustained outage stops
// re-recording itself as a fresh sync job per changed codebase per interval.
func TestRunSyncAllSkipsWhenEmbedderUnreachable(t *testing.T) {
	manager, _ := setupThreeChangedCodebases(t)
	cfg := manager.config

	setHealthMode(manager, dependencyEmbedderUnreachable)

	syncer := NewBackgroundSync(cfg, manager)
	syncer.runSyncAll(context.Background(), "test")

	if got := countDaemonSyncJobs(manager); got != 0 {
		t.Fatalf("embedder-unreachable sweep started %d sync jobs, want 0", got)
	}
}

// TestRunSyncAllSyncsEveryChangedCodebaseWhenHealthy proves the skip is scoped to
// the outage: a healthy sweep still syncs every changed codebase.
func TestRunSyncAllSyncsEveryChangedCodebaseWhenHealthy(t *testing.T) {
	manager, repos := setupThreeChangedCodebases(t)
	cfg := manager.config

	syncer := NewBackgroundSync(cfg, manager)
	syncer.runSyncAll(context.Background(), "test")

	drainToIndexed(t, manager, repos)

	if got := countDaemonSyncJobs(manager); got != len(repos) {
		t.Fatalf("healthy sweep started %d sync jobs, want %d (one per changed codebase)", got, len(repos))
	}

	// A second sweep is a no-op: the synced snapshot converged with the on-disk
	// content, so no codebase reads as changed and no new sync job is started.
	syncer.runSyncAll(context.Background(), "test")
	drainToIndexed(t, manager, repos)
	if got := countDaemonSyncJobs(manager); got != len(repos) {
		t.Fatalf("second healthy sweep started %d sync jobs total, want %d (no re-sync of unchanged codebases)", got, len(repos))
	}
}

// TestRunSyncAllResumesAfterEmbedderRecovers proves the sweep resumes once the
// recorded outage clears: while unreachable it enqueues nothing, and after the
// health record returns to healthy the next sweep syncs every changed codebase.
func TestRunSyncAllResumesAfterEmbedderRecovers(t *testing.T) {
	manager, repos := setupThreeChangedCodebases(t)
	cfg := manager.config
	syncer := NewBackgroundSync(cfg, manager)

	setHealthMode(manager, dependencyEmbedderUnreachable)
	syncer.runSyncAll(context.Background(), "test")
	if got := countDaemonSyncJobs(manager); got != 0 {
		t.Fatalf("during outage sweep started %d sync jobs, want 0", got)
	}

	setHealthMode(manager, dependencyHealthy)
	syncer.runSyncAll(context.Background(), "test")
	drainToIndexed(t, manager, repos)
	if got := countDaemonSyncJobs(manager); got != len(repos) {
		t.Fatalf("post-recovery sweep started %d sync jobs, want %d", got, len(repos))
	}
}

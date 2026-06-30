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
// manager, its config, and the three canonical-ish repo paths.
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

	manager.runner = fakeRunner{
		index: func(_ context.Context, _ string, _ model.IndexConfig, _ func(indexer.Progress)) (indexer.Result, error) {
			return indexer.Result{
				IndexedFiles: 1,
				TotalChunks:  1,
				Chunks: []model.StoredChunk{{
					Content:       "package main\n",
					RelativePath:  "main.go",
					StartLine:     1,
					EndLine:       1,
					Language:      "go",
					FileExtension: ".go",
				}},
				FileHashes: map[string]string{"main.go": hashText("package main\n")},
			}, nil
		},
		indexOne: func(_ context.Context, _ string, relativePath string, _ model.IndexConfig) (indexer.OneFileResult, error) {
			content := "package main\n// edited\n"
			return indexer.OneFileResult{
				Chunks: []model.StoredChunk{{
					Content:       content,
					RelativePath:  relativePath,
					StartLine:     1,
					EndLine:       2,
					Language:      "go",
					FileExtension: ".go",
				}},
				FileHash: hashText(content),
			}, nil
		},
	}

	repos := []string{repo1, repo2, repo3}
	for _, repo := range repos {
		if _, _, _, _, err := manager.StartIndex(context.Background(), repo, testClientInfo(), defaultIndexConfig(), false); err != nil {
			t.Fatalf("StartIndex(%s): %v", repo, err)
		}
		waitForCodebaseStatus(t, manager, repo, model.CodebaseStatusIndexed)
	}

	for _, repo := range repos {
		if err := os.WriteFile(filepath.Join(repo, "main.go"), []byte("package main\nfunc Edited() {}\n"), 0o644); err != nil {
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

// TestRunSyncAllCanaryBackoffWhenEmbedderDegraded proves that when the embedding
// pipeline is degraded, one sweep starts a single canary sync rather than one per
// changed codebase, so a sustained outage costs one failed job per interval
// instead of one per codebase. Without the backoff all three changed codebases
// would enqueue a sync that fails and is superseded every interval.
func TestRunSyncAllCanaryBackoffWhenEmbedderDegraded(t *testing.T) {
	manager, repos := setupThreeChangedCodebases(t)
	cfg := manager.config

	manager.mu.Lock()
	manager.health.Mode = dependencyEmbedderUnreachable
	manager.mu.Unlock()

	syncer := NewBackgroundSync(cfg, manager)
	syncer.runSyncAll(context.Background(), "test")

	// Drain the canary's async job before asserting and before teardown, so the
	// job cannot race the temp-dir cleanup.
	for _, repo := range repos {
		waitForCodebaseStatus(t, manager, repo, model.CodebaseStatusIndexed)
	}

	if got := countDaemonSyncJobs(manager); got != 1 {
		t.Fatalf("degraded sweep started %d sync jobs, want 1 canary", got)
	}
}

// TestRunSyncAllSyncsEveryChangedCodebaseWhenHealthy proves the canary backoff is
// scoped to the degraded case: a healthy sweep still syncs every changed
// codebase, so the backoff never throttles normal operation.
func TestRunSyncAllSyncsEveryChangedCodebaseWhenHealthy(t *testing.T) {
	manager, repos := setupThreeChangedCodebases(t)
	cfg := manager.config

	syncer := NewBackgroundSync(cfg, manager)
	syncer.runSyncAll(context.Background(), "test")

	// Drain the async sync jobs before asserting and before teardown, so a job
	// still capturing a repo cannot race the temp-dir cleanup.
	for _, repo := range repos {
		waitForCodebaseStatus(t, manager, repo, model.CodebaseStatusIndexed)
	}

	if got := countDaemonSyncJobs(manager); got != len(repos) {
		t.Fatalf("healthy sweep started %d sync jobs, want %d (one per changed codebase)", got, len(repos))
	}
}

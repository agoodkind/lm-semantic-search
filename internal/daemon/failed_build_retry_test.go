package daemon

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/model"
)

// failedBuildRetryTestCap tracks the production cap so the test stays a valid
// regression guard if the cap changes.
const failedBuildRetryTestCap = maxFailedBuildRetries

func TestRunSyncAllRetriesFailedBuildUntilCap(t *testing.T) {
	t.Parallel()

	manager, cfg, repoPath := newTestManager(t)
	manager.semantic = &fakeSemantic{unavailable: true}
	manager.runner = failedBuildRetryRunner()
	codebase := registerFailedBuildCodebase(t, manager, repoPath)
	syncer := NewBackgroundSync(cfg, manager)

	for attempt := 1; attempt <= failedBuildRetryTestCap; attempt++ {
		syncer.runSyncAll(context.Background(), "test")
		waitForFailedBuildRetryJobs(t, manager, codebase.ID, attempt)
	}

	syncer.runSyncAll(context.Background(), "test")
	if got := len(manager.ListJobs(codebase.ID)); got != failedBuildRetryTestCap {
		t.Fatalf("retry jobs after capped sweep = %d, want %d", got, failedBuildRetryTestCap)
	}
}

func TestRunSyncAllDoesNotRetryFailedBuildWhenDirectoryMissing(t *testing.T) {
	t.Parallel()

	manager, cfg, _ := newTestManager(t)
	missingPath := filepath.Join(t.TempDir(), "missing")
	codebase := registerFailedBuildCodebaseAtPath(manager, "cb-failed-missing", missingPath)
	syncer := NewBackgroundSync(cfg, manager)

	syncer.runSyncAll(context.Background(), "test")

	if got := len(manager.ListJobs(codebase.ID)); got != 0 {
		t.Fatalf("retry jobs for missing directory = %d, want 0", got)
	}
}

func TestSuccessfulBuildCompletionClearsFailedBuildRetryCounter(t *testing.T) {
	t.Parallel()

	manager, cfg, repoPath := newTestManager(t)
	manager.semantic = &fakeSemantic{unavailable: true}
	manager.runner = failedBuildRetryRunner()
	codebase := registerFailedBuildCodebase(t, manager, repoPath)
	syncer := NewBackgroundSync(cfg, manager)

	for attempt := 1; attempt <= failedBuildRetryTestCap; attempt++ {
		syncer.runSyncAll(context.Background(), "test")
		waitForFailedBuildRetryJobs(t, manager, codebase.ID, attempt)
	}
	syncer.runSyncAll(context.Background(), "test")
	if got := len(manager.ListJobs(codebase.ID)); got != failedBuildRetryTestCap {
		t.Fatalf("retry jobs before completion reset = %d, want %d", got, failedBuildRetryTestCap)
	}

	jobs := manager.ListJobs(codebase.ID)
	manager.updateJobCompleted(context.Background(), jobs[0].ID, successfulFailedBuildRetryResult())
	markCodebaseFailedForRetry(manager, codebase.ID)

	syncer.runSyncAll(context.Background(), "test")
	waitForBuildRetryJobs(t, manager, codebase.ID, failedBuildRetryTestCap+1)
}

func failedBuildRetryRunner() fakeRunner {
	return fakeRunner{
		index:      nil,
		indexFiles: nil,
		indexOne: func(context.Context, string, string, model.IndexConfig) (indexer.OneFileResult, error) {
			return indexer.OneFileResult{
				Chunks:   nil,
				FileHash: "",
				Skipped:  false,
				Removed:  false,
			}, errors.New("retry still failing")
		},
	}
}

func registerFailedBuildCodebase(t *testing.T, manager *Manager, repoPath string) model.Codebase {
	t.Helper()

	canonicalPath, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	return registerFailedBuildCodebaseAtPath(manager, "cb-failed-retry", canonicalPath)
}

func registerFailedBuildCodebaseAtPath(manager *Manager, codebaseID string, canonicalPath string) model.Codebase {
	indexConfig := defaultIndexConfig()
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)
	codebase := newCodebaseRecord(canonicalPath)
	codebase.ID = codebaseID
	codebase.Status = model.CodebaseStatusFailed
	codebase.EffectiveConfig = indexConfig
	codebase.UpdatedAt = clock.Now()

	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	return codebase
}

func markCodebaseFailedForRetry(manager *Manager, codebaseID string) {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	codebase := manager.codebases[codebaseID]
	codebase.Status = model.CodebaseStatusFailed
	codebase.ActiveJobID = ""
	codebase.UpdatedAt = clock.Now()
	manager.codebases[codebaseID] = codebase
}

func waitForFailedBuildRetryJobs(t *testing.T, manager *Manager, codebaseID string, want int) {
	t.Helper()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		jobs := manager.ListJobs(codebaseID)
		failedJobs := 0
		for _, job := range jobs {
			if job.State == model.JobStateFailed {
				failedJobs++
			}
		}
		if len(jobs) == want && failedJobs == want {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("failed retry jobs for %s did not reach %d; got %d", codebaseID, want, len(manager.ListJobs(codebaseID)))
}

func waitForBuildRetryJobs(t *testing.T, manager *Manager, codebaseID string, want int) {
	t.Helper()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		jobs := manager.ListJobs(codebaseID)
		terminalJobs := 0
		for _, job := range jobs {
			if job.State == model.JobStateCompleted || job.State == model.JobStateFailed || job.State == model.JobStateCancelled {
				terminalJobs++
			}
		}
		if len(jobs) == want && terminalJobs == want {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("retry jobs for %s did not reach %d; got %d", codebaseID, want, len(manager.ListJobs(codebaseID)))
}

func successfulFailedBuildRetryResult() indexer.Result {
	content := "package main\n"
	return indexer.Result{
		IndexedFiles: 1,
		TotalChunks:  1,
		Chunks: []model.StoredChunk{{
			Content:              content,
			RelativePath:         "main.go",
			StartLine:            1,
			EndLine:              1,
			Language:             "go",
			FileExtension:        ".go",
			ConversationID:       "",
			ParentConversationID: "",
			MessageIndex:         0,
			Role:                 "",
			TimestampUnix:        0,
			WorkspaceRoot:        "",
			Archived:             false,
			Score:                0,
		}},
		FileHashes:        map[string]string{"main.go": hashText(content)},
		SkippedFiles:      nil,
		SkippedOversize:   0,
		SkippedUnreadable: 0,
		SkippedPending:    0,
	}
}

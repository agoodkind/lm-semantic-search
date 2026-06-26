package daemon

import (
	"context"
	"errors"
	"net"
	"path/filepath"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/store"
)

// A job failure on a shared dependency degrades the health record with the
// matching mode. A busy failure that survived the in-process retry is a real
// outage and degrades too; a cancellation is transient and leaves the banner off.
func TestDegradeModeFor(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		err  error
		want dependencyMode
	}{
		{"unreachable", adapterr.NewEmbedderUnreachable(nil), dependencyEmbedderUnreachable},
		{"rejected", adapterr.NewEmbedderRejected(nil), dependencyEmbedderRejected},
		{"busy", adapterr.NewEmbedderBusy(nil), dependencyEmbedderBusy},
		{"store unavailable", semantic.ErrUnavailable, dependencyStoreUnavailable},
		{"cancelled", adapterr.NewEmbedCancelled(nil), dependencyHealthy},
		{"context canceled", context.Canceled, dependencyHealthy},
		{"internal", adapterr.NewInternal("boom", nil), dependencyHealthy},
		{"nil", nil, dependencyHealthy},
	}
	for _, testCase := range cases {
		if got := degradeModeFor(testCase.err); got != testCase.want {
			t.Fatalf("%s: degradeModeFor = %q, want %q", testCase.name, got, testCase.want)
		}
	}
}

// A no-op completion (no files embedded) must not clear a degraded banner raised
// by a real outage, because it never exercised the embedder. Only a completion
// that actually embedded proves the dependency recovered.
func TestNoOpCompletionKeepsBanner(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}

	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	job := model.Job{ID: "job-noop", CodebaseID: codebase.ID, State: model.JobStateRunning}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.jobs[job.ID] = job
	manager.health = dependencyHealth{Mode: dependencyEmbedderUnreachable}
	manager.mu.Unlock()

	manager.updateJobCompleted(context.Background(), job.ID, indexer.Result{IndexedFiles: 0, TotalChunks: 0})

	if !manager.DependencyHealth().Degraded() {
		t.Fatal("a no-op completion cleared the banner; want it kept during the outage")
	}

	// A real completion records embedded files through the per-file progress loop;
	// model that so the clear gate sees genuine embed work rather than the codebase
	// file count a no-op completion also carries.
	manager.mu.Lock()
	embeddedJob := manager.jobs[job.ID]
	embeddedJob.Progress.FilesEmbedded = 3
	manager.jobs[job.ID] = embeddedJob
	manager.mu.Unlock()

	manager.updateJobCompleted(context.Background(), job.ID, indexer.Result{IndexedFiles: 3, TotalChunks: 12})

	if manager.DependencyHealth().Degraded() {
		t.Fatal("a completion that embedded did not clear the banner")
	}
}

// A progress update that embedded a file clears the banner immediately, so a long
// recovering build stops showing a stale "paused" banner while it makes progress;
// a progress update that embedded nothing leaves the banner up.
func TestEmbedProgressClearsBanner(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}

	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexing
	job := model.Job{ID: "job-progress", CodebaseID: codebase.ID, State: model.JobStateRunning}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.jobs[job.ID] = job
	manager.health = dependencyHealth{Mode: dependencyEmbedderBusy}
	manager.mu.Unlock()

	manager.updateJobProgress(job.ID, indexer.Progress{FilesTotal: 10, FilesProcessed: 1, FilesEmbedded: 0}, "file")
	if !manager.DependencyHealth().Degraded() {
		t.Fatal("a no-embed progress update cleared the banner; want it kept")
	}

	manager.updateJobProgress(job.ID, indexer.Progress{FilesTotal: 10, FilesProcessed: 2, FilesEmbedded: 1}, "file")
	if manager.DependencyHealth().Degraded() {
		t.Fatal("an embed-progress update did not clear the banner")
	}
}

// A failed infra job degrades the daemon health record; the next completed job
// clears it, so the banner appears on the outage and clears on recovery.
func TestDependencyHealthFollowsJobOutcomes(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}

	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexing
	job := model.Job{ID: "job-health", CodebaseID: codebase.ID, State: model.JobStateRunning}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	if manager.DependencyHealth().Degraded() {
		t.Fatal("health degraded before any failure, want healthy")
	}

	manager.updateJobFailed(context.Background(), job.ID, adapterr.NewEmbedderUnreachable(errors.New("connection refused")))

	health := manager.DependencyHealth()
	if health.Mode != dependencyEmbedderUnreachable {
		t.Fatalf("health mode = %q, want %q after an unreachable failure", health.Mode, dependencyEmbedderUnreachable)
	}
	if health.Since.IsZero() {
		t.Fatal("health Since is zero after degrading, want a timestamp")
	}

	// A real recovery completion records embedded files through the per-file
	// progress loop; model that so the clear gate sees genuine embed work.
	manager.mu.Lock()
	recoveredJob := manager.jobs[job.ID]
	recoveredJob.Progress.FilesEmbedded = 1
	manager.jobs[job.ID] = recoveredJob
	manager.mu.Unlock()

	manager.updateJobCompleted(context.Background(), job.ID, indexer.Result{IndexedFiles: 1, TotalChunks: 1})

	if manager.DependencyHealth().Degraded() {
		t.Fatal("health still degraded after a completed job, want cleared")
	}
	if manager.DependencyHealth().LastHealthyAt.IsZero() {
		t.Fatal("LastHealthyAt is zero after a completed job, want a timestamp")
	}
}

func TestNewManagerMarksStoreUnavailableWhenMilvusBootDialFails(t *testing.T) {
	stateRoot := t.TempDir()
	cfg := config.Config{
		StateRoot:              stateRoot,
		SocketPath:             filepath.Join(stateRoot, "sockets", "lm-semantic-search-daemon.sock"),
		RegistryPath:           filepath.Join(stateRoot, "registry.json"),
		JobsPath:               filepath.Join(stateRoot, "jobs.jsonl"),
		EventsPath:             filepath.Join(stateRoot, "events.jsonl"),
		LogsDir:                filepath.Join(stateRoot, "logs"),
		LogPath:                filepath.Join(stateRoot, "logs", "lm-semantic-search-daemon.log"),
		MerkleDir:              filepath.Join(stateRoot, "merkle"),
		LocksDir:               filepath.Join(stateRoot, "locks"),
		SocketsDir:             filepath.Join(stateRoot, "sockets"),
		ChunksDir:              filepath.Join(stateRoot, "chunks"),
		ContextRoot:            filepath.Join(stateRoot, "context"),
		EmbeddingProvider:      "OpenAI",
		EmbeddingModel:         "text-embedding-3-small",
		OpenAIAPIKey:           "test-key",
		MilvusAddress:          closedDaemonMilvusAddress(t),
		HybridMode:             true,
		SyncIntervalMS:         300000,
		SyncLockStaleMS:        600000,
		MaxConcurrentIndexJobs: 1,
	}
	for _, path := range []string{cfg.StateRoot, cfg.LogsDir, cfg.MerkleDir, cfg.LocksDir, cfg.SocketsDir, cfg.ChunksDir, cfg.ContextRoot} {
		if err := store.EnsureDir(path); err != nil {
			t.Fatalf("EnsureDir returned error: %v", err)
		}
	}
	if err := store.WriteRegistry(cfg.RegistryPath, model.RegistryFile{}); err != nil {
		t.Fatalf("WriteRegistry returned error: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	manager, err := NewManager(ctx, cfg)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}
	t.Cleanup(func() {
		if service, ok := manager.semantic.(*semantic.Service); ok {
			if closeErr := service.Close(context.Background()); closeErr != nil {
				t.Fatalf("Close returned error: %v", closeErr)
			}
		}
	})

	health := manager.DependencyHealth()
	if health.Mode != dependencyStoreUnavailable {
		t.Fatalf("DependencyHealth().Mode = %q, want %q", health.Mode, dependencyStoreUnavailable)
	}
	if health.Since.IsZero() {
		t.Fatal("DependencyHealth().Since is zero, want the boot failure timestamp")
	}
}

func closedDaemonMilvusAddress(t *testing.T) string {
	t.Helper()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen returned error: %v", err)
	}
	address := listener.Addr().String()
	if err := listener.Close(); err != nil {
		t.Fatalf("Close listener returned error: %v", err)
	}
	return address
}

package daemon

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// A hard outage degrades the health record with the matching mode; a busy
// endpoint and a cancellation are transient and leave the banner off.
func TestDegradeModeFor(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		err  error
		want dependencyMode
	}{
		{"unreachable", adapterr.NewEmbedderUnreachable(nil), dependencyEmbedderUnreachable},
		{"rejected", adapterr.NewEmbedderRejected(nil), dependencyEmbedderRejected},
		{"store unavailable", semantic.ErrUnavailable, dependencyStoreUnavailable},
		{"busy", adapterr.NewEmbedderBusy(nil), dependencyHealthy},
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

	manager.updateJobCompleted(job.ID, indexer.Result{IndexedFiles: 1, TotalChunks: 1})

	if manager.DependencyHealth().Degraded() {
		t.Fatal("health still degraded after a completed job, want cleared")
	}
	if manager.DependencyHealth().LastHealthyAt.IsZero() {
		t.Fatal("LastHealthyAt is zero after a completed job, want a timestamp")
	}
}

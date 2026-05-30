package daemon

import (
	"strings"
	"testing"
	"time"

	"goodkind.io/claude-context-go/internal/model"
)

// TestRenderReconcileMagnitudeEmpty proves a queued run with no recorded counts
// adds no magnitude line, so the status output stays quiet until work starts.
func TestRenderReconcileMagnitudeEmpty(t *testing.T) {
	t.Parallel()
	if got := renderReconcileMagnitude(model.Progress{}); got != "" {
		t.Fatalf("expected empty magnitude for zero progress, got %q", got)
	}
}

// TestRenderReconcileMagnitudeFullReindex proves a full reindex (no delta
// breakdown) reports the embed progress but no change line.
func TestRenderReconcileMagnitudeFullReindex(t *testing.T) {
	t.Parallel()
	got := renderReconcileMagnitude(model.Progress{FilesTotal: 100, FilesProcessed: 50, ChunksGenerated: 800})
	if !strings.Contains(got, "📦 50/100 files embedded, 800 chunks") {
		t.Fatalf("expected embed progress line, got %q", got)
	}
	if strings.Contains(got, "🔀") {
		t.Fatalf("did not expect a change-breakdown line for a full reindex, got %q", got)
	}
}

// TestRenderReconcileMagnitudeDelta proves a delta sync reports both the embed
// progress and the added/modified/removed breakdown.
func TestRenderReconcileMagnitudeDelta(t *testing.T) {
	t.Parallel()
	got := renderReconcileMagnitude(model.Progress{
		FilesTotal:      480,
		FilesProcessed:  168,
		ChunksGenerated: 5400,
		FilesAdded:      12,
		FilesModified:   30,
		FilesRemoved:    5,
	})
	if !strings.Contains(got, "📦 168/480 files embedded, 5400 chunks") {
		t.Fatalf("expected embed progress line, got %q", got)
	}
	if !strings.Contains(got, "🔀 Changes: 12 added, 30 modified, 5 removed") {
		t.Fatalf("expected change-breakdown line, got %q", got)
	}
}

// TestRenderIndexingActiveShowsMagnitude proves the in-progress status banner
// surfaces the reconcile magnitude, so a large merge is visibly distinct from a
// one-file edit.
func TestRenderIndexingActiveShowsMagnitude(t *testing.T) {
	t.Parallel()
	codebase := &model.Codebase{CanonicalPath: "/repo", UpdatedAt: time.Unix(1700000000, 0)}
	job := &model.Job{Progress: model.Progress{
		OverallPercent:  35,
		FilesTotal:      480,
		FilesProcessed:  168,
		ChunksGenerated: 5400,
		FilesAdded:      12,
		FilesModified:   30,
		FilesRemoved:    5,
		LastEventAt:     time.Unix(1700000000, 0),
	}}
	out := renderIndexingActive(codebase, job)
	if !strings.Contains(out, "Progress: 35.0%") {
		t.Fatalf("expected progress percent, got %q", out)
	}
	if !strings.Contains(out, "📦 168/480 files embedded, 5400 chunks") {
		t.Fatalf("expected embed progress line in active banner, got %q", out)
	}
	if !strings.Contains(out, "🔀 Changes: 12 added, 30 modified, 5 removed") {
		t.Fatalf("expected change-breakdown line in active banner, got %q", out)
	}
}

// TestRenderGetJobShowsMagnitude proves the per-job view carries the same
// magnitude detail.
func TestRenderGetJobShowsMagnitude(t *testing.T) {
	t.Parallel()
	job := &model.Job{
		ID:            "job_x",
		CanonicalPath: "/repo",
		Operation:     "sync",
		State:         model.JobStateRunning,
		Progress: model.Progress{
			OverallPercent:  35,
			FilesTotal:      480,
			FilesProcessed:  168,
			ChunksGenerated: 5400,
			FilesAdded:      12,
			FilesModified:   30,
			FilesRemoved:    5,
		},
	}
	out := renderGetJob(job)
	if !strings.Contains(out, "📦 168/480 files embedded, 5400 chunks") {
		t.Fatalf("expected embed progress line in job view, got %q", out)
	}
	if !strings.Contains(out, "🔀 Changes: 12 added, 30 modified, 5 removed") {
		t.Fatalf("expected change-breakdown line in job view, got %q", out)
	}
}

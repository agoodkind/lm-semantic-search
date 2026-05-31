package daemon

import (
	"strings"
	"testing"
	"time"

	"goodkind.io/claude-context-go/internal/model"
)

var renderTestTime = time.Unix(1700000000, 0)

// TestRenderIndexedDetailReady proves the ready status leads with the repo
// title, states readiness, and shows the standing index totals.
func TestRenderIndexedDetailReady(t *testing.T) {
	t.Parallel()
	codebase := &model.Codebase{
		CanonicalPath: "/Users/agoodkind/Sites/swift-makefile",
		LastSuccessfulRun: &model.IndexRunSummary{
			IndexedFiles: 58,
			TotalChunks:  600,
			Status:       "completed",
			CompletedAt:  renderTestTime,
		},
	}
	out := renderIndexedDetail(codebase)
	for _, want := range []string{"📁 swift-makefile", "✅ Ready to search", "📊 58 files, 600 chunks"} {
		if !strings.Contains(out, want) {
			t.Fatalf("ready status missing %q in:\n%s", want, out)
		}
	}
}

// TestRenderIndexingActivePreparingSync proves a watcher-driven sync that has
// not started embedding reads as "Changes detected, preparing to index".
func TestRenderIndexingActivePreparingSync(t *testing.T) {
	t.Parallel()
	codebase := &model.Codebase{CanonicalPath: "/Users/agoodkind/Sites/swift-makefile"}
	job := &model.Job{Operation: "sync", Progress: model.Progress{FilesTotal: 0, LastEventAt: renderTestTime}}
	out := renderIndexingActive(codebase, job)
	if !strings.Contains(out, "📁 swift-makefile") {
		t.Fatalf("missing title in:\n%s", out)
	}
	if !strings.Contains(out, "⚙️ Changes detected, preparing to index") {
		t.Fatalf("expected changes-detected prepare line in:\n%s", out)
	}
	if strings.Contains(out, "🔄 Indexing") {
		t.Fatalf("did not expect indexing line during prepare in:\n%s", out)
	}
}

// TestRenderIndexingActivePreparingForced proves a full or forced reindex that
// has not started embedding reads as a plain "Preparing to index".
func TestRenderIndexingActivePreparingForced(t *testing.T) {
	t.Parallel()
	codebase := &model.Codebase{CanonicalPath: "/Users/agoodkind/Sites/swift-makefile"}
	job := &model.Job{Operation: "index", Progress: model.Progress{FilesTotal: 0}}
	out := renderIndexingActive(codebase, job)
	if !strings.Contains(out, "⚙️ Preparing to index") {
		t.Fatalf("expected plain prepare line in:\n%s", out)
	}
	if strings.Contains(out, "Changes detected") {
		t.Fatalf("did not expect changes-detected for a forced reindex in:\n%s", out)
	}
}

// TestRenderIndexingActiveEmbedding proves the embedding phase shows the file
// progress as "X of N" and the chunk count as a running tally.
func TestRenderIndexingActiveEmbedding(t *testing.T) {
	t.Parallel()
	codebase := &model.Codebase{CanonicalPath: "/Users/agoodkind/Sites/swift-makefile"}
	job := &model.Job{
		Operation: "streaming_reindex",
		Progress:  model.Progress{FilesTotal: 58, FilesProcessed: 7, ChunksGenerated: 84, LastEventAt: renderTestTime},
	}
	out := renderIndexingActive(codebase, job)
	for _, want := range []string{"📁 swift-makefile", "🔄 Indexing", "📄 7 of 58 files", "🧩 84 chunks so far"} {
		if !strings.Contains(out, want) {
			t.Fatalf("embedding status missing %q in:\n%s", want, out)
		}
	}
}

// TestRenderReconcileMagnitude proves the job-view magnitude shows files and
// chunks, plus the change breakdown for a delta, and nothing when empty.
func TestRenderReconcileMagnitude(t *testing.T) {
	t.Parallel()
	if got := renderReconcileMagnitude(model.Progress{}); got != "" {
		t.Fatalf("expected empty magnitude for zero progress, got %q", got)
	}
	got := renderReconcileMagnitude(model.Progress{
		FilesTotal: 58, FilesProcessed: 7, ChunksGenerated: 84,
		FilesAdded: 12, FilesModified: 30, FilesRemoved: 5,
	})
	if !strings.Contains(got, "📄 7 of 58 files · 🧩 84 chunks") {
		t.Fatalf("expected files and chunks line, got %q", got)
	}
	if !strings.Contains(got, "Added 12 · Modified 30 · Removed 5") {
		t.Fatalf("expected change breakdown, got %q", got)
	}
}

// TestRenderGetJobShowsMagnitude proves the job view carries the magnitude.
func TestRenderGetJobShowsMagnitude(t *testing.T) {
	t.Parallel()
	job := &model.Job{
		ID:            "job_x",
		CanonicalPath: "/repo",
		Operation:     "sync",
		State:         model.JobStateRunning,
		Progress:      model.Progress{FilesTotal: 58, FilesProcessed: 7, ChunksGenerated: 84},
	}
	out := renderGetJob(job)
	if !strings.Contains(out, "📄 7 of 58 files · 🧩 84 chunks") {
		t.Fatalf("expected magnitude in job view, got:\n%s", out)
	}
}

// TestStatusTemplateNoBlankLines proves the embedded templates produce a tidy
// block with no blank lines and the expected line count, guarding against
// template whitespace regressions.
func TestStatusTemplateNoBlankLines(t *testing.T) {
	t.Parallel()
	codebase := &model.Codebase{
		CanonicalPath: "/Users/agoodkind/Sites/swift-makefile",
		LastSuccessfulRun: &model.IndexRunSummary{
			IndexedFiles: 58,
			TotalChunks:  600,
			Status:       "completed",
			CompletedAt:  renderTestTime,
		},
	}
	cases := map[string]string{
		"ready":     renderIndexedDetail(codebase),
		"preparing": renderIndexingActive(codebase, &model.Job{Operation: "sync", Progress: model.Progress{FilesTotal: 0}}),
		"indexing":  renderIndexingActive(codebase, &model.Job{Operation: "sync", Progress: model.Progress{FilesTotal: 58, FilesProcessed: 7, ChunksGenerated: 84}}),
	}
	wantLines := map[string]int{"ready": 4, "preparing": 3, "indexing": 5}
	for name, out := range cases {
		for _, line := range strings.Split(out, "\n") {
			if strings.TrimSpace(line) == "" {
				t.Fatalf("%s has a blank line:\n%s", name, out)
			}
		}
		if got := len(strings.Split(out, "\n")); got != wantLines[name] {
			t.Fatalf("%s line count = %d, want %d:\n%s", name, got, wantLines[name], out)
		}
	}
}

package daemon

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/view"
)

var renderTestTime = time.Unix(1700000000, 0)

func renderListJobsForTest(jobs []model.Job, degraded bool) string {
	successors := buildJobSuccessors(jobs)
	summary := resolveListSummary(jobs, degraded)
	active := make([]view.JobEntryView, 0, len(jobs))
	terminal := make([]view.JobEntryView, 0, len(jobs))
	for _, job := range jobs {
		entry := resolveJobEntry(job, degraded, successors[job.ID])
		if isTerminalJobState(job.State) {
			terminal = append(terminal, entry)
		} else {
			active = append(active, entry)
		}
	}
	return renderListJobs(summary, active, terminal)
}

// TestRenderSymlinkResolution proves the status output names the real path a
// symlinked query path resolves to, and adds nothing for a non-symlink path.
func TestRenderSymlinkResolution(t *testing.T) {
	t.Parallel()
	// Resolve the temp dir first: on macOS t.TempDir lives under /var, itself a
	// symlink to /private/var, so the resolved form is the true non-symlink path.
	realRoot, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatalf("resolve temp dir: %v", err)
	}
	link := filepath.Join(filepath.Dir(realRoot), "codebase-link")
	if err := os.Symlink(realRoot, link); err != nil {
		t.Fatalf("create symlink: %v", err)
	}
	t.Cleanup(func() { _ = os.Remove(link) })

	if got := renderSymlinkResolution(link); got != "🔗 symlink resolved to: "+realRoot {
		t.Fatalf("symlink path: got %q, want resolution to %q", got, realRoot)
	}
	if got := renderSymlinkResolution(realRoot); got != "" {
		t.Fatalf("non-symlink path should add no line, got %q", got)
	}
}

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

// TestRenderIndexingActiveBuilding proves a from-scratch build reads as
// "Building initial index" with the percent, files embedded, and chunks so far.
func TestRenderIndexingActiveBuilding(t *testing.T) {
	t.Parallel()
	codebase := &model.Codebase{CanonicalPath: "/Users/agoodkind/Sites/swift-makefile"}
	job := &model.Job{
		Operation: "index",
		Progress:  model.Progress{OverallPercent: 42, FilesTotal: 58, FilesProcessed: 24, ChunksGenerated: 71, LastEventAt: renderTestTime},
	}
	out := renderIndexingActive(codebase, job)
	for _, want := range []string{"📁 swift-makefile", "🔄 Building initial index: 42%", "📥 24 of 58 files processed", "🧩 71 chunks total", "♻️ 0 reused", "➕ 71 embedded this run"} {
		if !strings.Contains(out, want) {
			t.Fatalf("building status missing %q in:\n%s", want, out)
		}
	}
}

// TestHeadingFor proves the in-progress heading derives from whether a completed
// run exists and from the trigger: a first build, a forced reindex over a good
// index, or a changed-files sync.
func TestHeadingFor(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		cb   model.Codebase
		job  model.Job
		want string
	}{
		{"initial", model.Codebase{}, model.Job{Operation: "index", Forced: true}, "Building initial index"},
		{"forced", model.Codebase{LastSuccessfulRun: &model.IndexRunSummary{}}, model.Job{Operation: "index", Forced: true}, "Forced reindex"},
		{"forced-sync", model.Codebase{LastSuccessfulRun: &model.IndexRunSummary{}}, model.Job{Operation: "sync", Forced: true}, "Forced reindex"},
		{"changed", model.Codebase{LastSuccessfulRun: &model.IndexRunSummary{}}, model.Job{Operation: "sync"}, "Indexing new changes"},
	}
	for _, testCase := range cases {
		job := testCase.job
		if got := headingFor(testCase.cb, &job); got != testCase.want {
			t.Errorf("%s: headingFor = %q, want %q", testCase.name, got, testCase.want)
		}
	}
}

// TestRenderIndexingActiveIncremental proves an incremental run reads as
// "Indexing new changes" with the percent, the scanned/unchanged/re-embedded
// breakdown, the live chunk total, and the chunks added this scan.
func TestRenderIndexingActiveIncremental(t *testing.T) {
	t.Parallel()
	codebase := &model.Codebase{
		CanonicalPath:     "/Users/agoodkind/Sites/swift-makefile",
		LastSuccessfulRun: &model.IndexRunSummary{TotalChunks: 600},
	}
	job := &model.Job{
		Operation: "streaming_reindex",
		Progress: model.Progress{
			OverallPercent: 37, FilesTotal: 452, FilesProcessed: 285,
			FilesInCodebase: 4292, FilesAdded: 29, FilesModified: 423, FilesRemoved: 10,
			FilesEmbedded: 285, FilesSkippedOversize: 3, FilesSkippedUnreadable: 2,
			ChunksGenerated: 1043, ChunksTotal: 57240, LastEventAt: renderTestTime,
		},
	}
	out := renderIndexingActive(codebase, job)
	for _, want := range []string{
		"📁 swift-makefile",
		"🔄 Indexing new changes: 37%",
		"🔢 4292 files: 462 changed, 3830 unchanged",
		"📄 300 of 462 changed files processed",
		"♻️ 285 re-embedded",
		"🗑️ 10 removed",
		"📏 3 skipped, oversize",
		"🚫 2 skipped, unreadable",
		"🧩 57240 chunks total",
		"➕ 1043 chunks added this scan",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("incremental status missing %q in:\n%s", want, out)
		}
	}
}

// TestRenderIndexingActiveIncrementalZeroProcessed proves that once the diff is
// known the status shows the indexing view with zero processed, rather than
// "Preparing", so a slow first embed does not read as a stall.
func TestRenderIndexingActiveIncrementalZeroProcessed(t *testing.T) {
	t.Parallel()
	codebase := &model.Codebase{CanonicalPath: "/Users/agoodkind/Sites/swift-makefile"}
	// Diff just captured: FilesInCodebase and the changed counts are set, but the
	// embed loop has not reported a FilesTotal yet. The renderer flips to the
	// indexing view off the diff-known fact, not the first per-file update.
	job := &model.Job{
		Operation: "sync",
		Progress: model.Progress{
			OverallPercent: 0, FilesTotal: 0, FilesProcessed: 0,
			FilesInCodebase: 4292, FilesAdded: 0, FilesModified: 118, FilesRemoved: 0,
			FilesEmbedded: 0, ChunksGenerated: 0,
		},
	}
	out := renderIndexingActive(codebase, job)
	for _, want := range []string{
		"🔄 Indexing new changes: 0%",
		"🔢 4292 files: 118 changed, 4174 unchanged",
		"📄 0 of 118 changed files processed",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("zero-processed incremental status missing %q in:\n%s", want, out)
		}
	}
	if strings.Contains(out, "Preparing") {
		t.Fatalf("expected indexing view, not preparing, in:\n%s", out)
	}
}

// TestRenderIndexingActiveIncrementalFallsBackToLastTotal proves the chunk total
// falls back to the last recorded run total when the live count is unpopulated.
func TestRenderIndexingActiveIncrementalFallsBackToLastTotal(t *testing.T) {
	t.Parallel()
	codebase := &model.Codebase{
		CanonicalPath:     "/Users/agoodkind/Sites/swift-makefile",
		LastSuccessfulRun: &model.IndexRunSummary{TotalChunks: 600},
	}
	job := &model.Job{
		Operation: "sync",
		Progress:  model.Progress{OverallPercent: 10, FilesTotal: 58, FilesProcessed: 1, ChunksTotal: 0},
	}
	out := renderIndexingActive(codebase, job)
	if !strings.Contains(out, "🧩 600 chunks total") {
		t.Fatalf("expected fallback to last recorded total in:\n%s", out)
	}
}

// TestRenderGetIndexBodySyncKeepsReady proves a background sync over an
// already-indexed codebase holds the ready view with a background note rather
// than the busy "Indexing new changes" takeover, because the live collection
// stays searchable while the delta runs.
func TestRenderGetIndexBodySyncKeepsReady(t *testing.T) {
	t.Parallel()
	codebase := &model.Codebase{
		CanonicalPath:     "/Users/agoodkind/Sites/swift-makefile",
		Status:            model.CodebaseStatusIndexing,
		LastSuccessfulRun: &model.IndexRunSummary{IndexedFiles: 58, TotalChunks: 600, CompletedAt: renderTestTime},
	}
	job := &model.Job{
		Operation: "sync",
		Progress:  model.Progress{OverallPercent: 33, FilesInCodebase: 58, FilesModified: 3, FilesProcessed: 1, LastEventAt: renderTestTime},
	}
	out := renderGetIndexBody("/Users/agoodkind/Sites/swift-makefile", true, codebase, job, dependencyHealth{})
	for _, want := range []string{"✅ Ready to search", "📊 58 files, 600 chunks", "🔄 Syncing 3 changed files in the background (33%)"} {
		if !strings.Contains(out, want) {
			t.Fatalf("sync-reconcile status missing %q in:\n%s", want, out)
		}
	}
	for _, reject := range []string{"Indexing new changes", "Building initial index"} {
		if strings.Contains(out, reject) {
			t.Fatalf("sync-reconcile status should not show %q in:\n%s", reject, out)
		}
	}
}

// TestRenderGetIndexBodySyncPreDiffKeepsReady proves a sync that has not yet
// captured its diff still reads as ready, with a generic background note.
func TestRenderGetIndexBodySyncPreDiffKeepsReady(t *testing.T) {
	t.Parallel()
	codebase := &model.Codebase{
		CanonicalPath:     "/Users/agoodkind/Sites/swift-makefile",
		Status:            model.CodebaseStatusIndexing,
		LastSuccessfulRun: &model.IndexRunSummary{IndexedFiles: 58, TotalChunks: 600, CompletedAt: renderTestTime},
	}
	job := &model.Job{Operation: "sync", Progress: model.Progress{FilesInCodebase: 0, LastEventAt: renderTestTime}}
	out := renderGetIndexBody("/Users/agoodkind/Sites/swift-makefile", true, codebase, job, dependencyHealth{})
	for _, want := range []string{"✅ Ready to search", "🔄 Checking for changes in the background"} {
		if !strings.Contains(out, want) {
			t.Fatalf("pre-diff sync status missing %q in:\n%s", want, out)
		}
	}
}

// TestRenderGetIndexBodyBuildingTakesOver proves a from-scratch build still
// owns the display, because its staging collection is not promoted yet.
func TestRenderGetIndexBodyBuildingTakesOver(t *testing.T) {
	t.Parallel()
	codebase := &model.Codebase{CanonicalPath: "/Users/agoodkind/Sites/swift-makefile", Status: model.CodebaseStatusIndexing}
	job := &model.Job{Operation: "index", Progress: model.Progress{OverallPercent: 42, FilesTotal: 58, FilesProcessed: 24, ChunksGenerated: 71}}
	out := renderGetIndexBody("/Users/agoodkind/Sites/swift-makefile", true, codebase, job, dependencyHealth{})
	if !strings.Contains(out, "🔄 Building initial index") {
		t.Fatalf("expected building takeover in:\n%s", out)
	}
	if strings.Contains(out, "✅ Ready to search") {
		t.Fatalf("a from-scratch build must not read as ready in:\n%s", out)
	}
}

// TestRenderGetIndexBodyStreamingReindexTakesOver proves a streaming_reindex
// keeps the busy takeover, scoping the ready-during-sync change to "sync" only.
func TestRenderGetIndexBodyStreamingReindexTakesOver(t *testing.T) {
	t.Parallel()
	codebase := &model.Codebase{
		CanonicalPath:     "/Users/agoodkind/Sites/swift-makefile",
		Status:            model.CodebaseStatusIndexing,
		LastSuccessfulRun: &model.IndexRunSummary{IndexedFiles: 58, TotalChunks: 600, CompletedAt: renderTestTime},
	}
	job := &model.Job{Operation: "streaming_reindex", Progress: model.Progress{OverallPercent: 37, FilesInCodebase: 58, FilesModified: 58, FilesProcessed: 20}}
	out := renderGetIndexBody("/Users/agoodkind/Sites/swift-makefile", true, codebase, job, dependencyHealth{})
	if !strings.Contains(out, "🔄 Indexing new changes") {
		t.Fatalf("expected streaming_reindex takeover in:\n%s", out)
	}
	if strings.Contains(out, "✅ Ready to search") {
		t.Fatalf("a streaming_reindex must not read as ready in:\n%s", out)
	}
}

// TestRenderProgressLines proves the job-view progress lines show typed
// denominators, chunks, and the typed change breakdown.
func TestRenderProgressLines(t *testing.T) {
	t.Parallel()
	empty := view.ProgressSurface{
		Heading:            "",
		HasScope:           false,
		Checked:            0,
		ScopeTotal:         0,
		ScopeLabel:         "",
		CheckVerb:          "",
		Embedded:           0,
		AlreadyIndexed:     0,
		ChunksThisRun:      0,
		ChunksReused:       0,
		ChunksInCollection: 0,
		ScopeLine:          "",
		PercentLabel:       "",
	}
	if got := renderProgressLines(empty); len(got) != 0 {
		t.Fatalf("expected no progress lines for an empty view, got %q", got)
	}
	progress := view.ProgressSurface{
		Heading:            "",
		HasScope:           true,
		Checked:            7,
		ScopeTotal:         58,
		ScopeLabel:         "files",
		CheckVerb:          "embedded",
		Embedded:           0,
		AlreadyIndexed:     0,
		ChunksThisRun:      84,
		ChunksReused:       0,
		ChunksInCollection: 84,
		ScopeLine:          "Changed since last sync: 12 files added · 30 modified · 5 removed",
		PercentLabel:       "12.1%",
	}
	got := strings.Join(renderProgressLines(progress), "\n")
	if !strings.Contains(got, "📄 7 of 58 files embedded") {
		t.Fatalf("expected files and chunks line, got %q", got)
	}
	if !strings.Contains(got, "🧩 84 chunks added this run") {
		t.Fatalf("expected chunk line, got %q", got)
	}
	if !strings.Contains(got, "Changed since last sync: 12 files added · 30 modified · 5 removed") {
		t.Fatalf("expected change breakdown, got %q", got)
	}
}

func TestRenderGetJobNotFound(t *testing.T) {
	t.Parallel()
	if got := renderGetJob(view.JobEntryView{}, false); got != "Job not found." {
		t.Fatalf("renderGetJob not found = %q, want %q", got, "Job not found.")
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
	out := renderGetJob(resolveJobEntry(*job, false, ""), true)
	if !strings.Contains(out, "📄 7 of 58 files embedded") {
		t.Fatalf("expected magnitude in job view, got:\n%s", out)
	}
	if !strings.Contains(out, "🧩 84 chunks added this run") {
		t.Fatalf("expected chunk line in job view, got:\n%s", out)
	}
}

func TestRenderGetJobUsesAmericanCanceledSpelling(t *testing.T) {
	t.Parallel()
	completedAt := renderTestTime.Add(90 * time.Second)
	job := &model.Job{
		ID:            "job_x",
		CanonicalPath: "/repo",
		Operation:     "sync",
		State:         model.JobStateCancelled,
		StartedAt:     renderTestTime,
		UpdatedAt:     completedAt,
		CompletedAt:   &completedAt,
		Progress:      model.Progress{Phase: "cancelled"},
	}
	out := renderGetJob(resolveJobEntry(*job, false, ""), true)
	if strings.Contains(out, "cancelled") {
		t.Fatalf("job view should use American spelling, got:\n%s", out)
	}
	for _, want := range []string{"State: canceled", "Phase: canceled", "Duration: 1m30s"} {
		if !strings.Contains(out, want) {
			t.Fatalf("job view missing %q in:\n%s", want, out)
		}
	}
}

func TestRenderListJobsSummarizesHistory(t *testing.T) {
	t.Parallel()
	activeUpdatedAt := renderTestTime.Add(2 * time.Minute)
	completedAt := renderTestTime.Add(45 * time.Minute)
	jobs := []model.Job{
		{
			ID:            "job_running",
			CanonicalPath: "/repo/running",
			Operation:     "index",
			State:         model.JobStateRunning,
			StartedAt:     renderTestTime,
			UpdatedAt:     activeUpdatedAt,
			Progress:      model.Progress{OverallPercent: 22.5, FilesTotal: 58, FilesProcessed: 7, ChunksGenerated: 84},
		},
		{
			ID:            "job_completed",
			CanonicalPath: "/repo/completed",
			Operation:     "sync",
			State:         model.JobStateCompleted,
			StartedAt:     renderTestTime,
			UpdatedAt:     completedAt,
			CompletedAt:   &completedAt,
			Progress:      model.Progress{OverallPercent: 100, FilesTotal: 58, FilesProcessed: 58, ChunksGenerated: 144},
		},
		{
			ID:            "job_cancelled",
			CanonicalPath: "/repo/cancelled",
			Operation:     "sync",
			State:         model.JobStateCancelled,
			StartedAt:     renderTestTime,
			UpdatedAt:     completedAt,
			CompletedAt:   &completedAt,
			Progress:      model.Progress{OverallPercent: 0, Phase: "cancelled"},
		},
	}
	out := renderListJobsForTest(jobs, false)
	for _, want := range []string{
		"Tracked jobs: 3 total",
		"Active: 0 queued, 1 running, 0 canceling",
		"Terminal: 1 completed, 0 failed, 0 superseded, 1 canceled",
		"Active jobs:",
		"Terminal jobs: 2",
		"Duration: 45m0s",
		"Elapsed: 2m0s",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("job list missing %q in:\n%s", want, out)
		}
	}
	if strings.Contains(out, "[cancelled") {
		t.Fatalf("job list should use American spelling for states, got:\n%s", out)
	}
}

// An earlier failure for a codebase that has a later terminal job is tallied as
// superseded, not failed, and the entry names the successor.
func TestRenderListJobsSeparatesSupersededFailures(t *testing.T) {
	t.Parallel()
	t0 := renderTestTime
	older := t0.Add(1 * time.Minute)
	newer := t0.Add(2 * time.Minute)
	jobs := []model.Job{
		{
			ID:            "job_old",
			CodebaseID:    "A",
			CanonicalPath: "/repo/a",
			Operation:     "sync",
			State:         model.JobStateFailed,
			StartedAt:     t0,
			CompletedAt:   &older,
			Error:         &model.JobError{Message: "embedding endpoint is unreachable", Retryable: true},
		},
		{
			ID:            "job_new",
			CodebaseID:    "A",
			CanonicalPath: "/repo/a",
			Operation:     "sync",
			State:         model.JobStateFailed,
			StartedAt:     t0,
			CompletedAt:   &newer,
			Error:         &model.JobError{Message: "internal error", Retryable: false},
		},
	}
	out := renderListJobsForTest(jobs, false)
	if want := "Terminal: 0 completed, 1 failed, 1 superseded, 0 canceled"; !strings.Contains(out, want) {
		t.Fatalf("summary did not separate superseded from failed, want %q in:\n%s", want, out)
	}
	if want := "superseded by job_new"; !strings.Contains(out, want) {
		t.Fatalf("superseded entry did not name its successor, want %q in:\n%s", want, out)
	}
}

// TestRenderGetJobPreparingNotZeroPercent proves a running index job whose work
// scope is not measured yet (A1) shows the preparing label, not a 0.0%.
func TestRenderGetJobPreparingNotZeroPercent(t *testing.T) {
	t.Parallel()
	job := &model.Job{
		ID:            "job_prep",
		CanonicalPath: "/repo",
		Operation:     "index",
		State:         model.JobStateRunning,
		Progress:      model.Progress{FilesTotal: 0, FilesInCodebase: 0, OverallPercent: 0},
	}
	out := renderGetJob(resolveJobEntry(*job, false, ""), true)
	if !strings.Contains(out, "Progress: Preparing to index") {
		t.Fatalf("expected preparing label, got:\n%s", out)
	}
	if strings.Contains(out, "0.0%") {
		t.Fatalf("running job with unknown scope must not show 0.0%%, got:\n%s", out)
	}
}

// TestRenderListJobsPreparingNotZeroPercent proves the same for the list view (A1).
func TestRenderListJobsPreparingNotZeroPercent(t *testing.T) {
	t.Parallel()
	jobs := []model.Job{{
		ID:            "job_prep",
		CanonicalPath: "/repo",
		Operation:     "index",
		State:         model.JobStateRunning,
		Progress:      model.Progress{FilesTotal: 0, FilesInCodebase: 0},
	}}
	out := renderListJobsForTest(jobs, false)
	if !strings.Contains(out, "Preparing to index") {
		t.Fatalf("expected preparing label in list, got:\n%s", out)
	}
	if strings.Contains(out, "0.0%") {
		t.Fatalf("list entry with unknown scope must not show 0.0%%, got:\n%s", out)
	}
}

// TestRenderGetJobSyncPreparingWording proves a sync job with unknown scope uses
// the sync-specific preparing wording (A2).
func TestRenderGetJobSyncPreparingWording(t *testing.T) {
	t.Parallel()
	job := &model.Job{
		ID:            "job_sync",
		CanonicalPath: "/repo",
		Operation:     "sync",
		State:         model.JobStateRunning,
		Progress:      model.Progress{FilesTotal: 0, FilesInCodebase: 0},
	}
	out := renderGetJob(resolveJobEntry(*job, false, ""), true)
	if !strings.Contains(out, "Changes detected, preparing to index") {
		t.Fatalf("expected sync preparing wording, got:\n%s", out)
	}
}

// TestRenderGetJobKeepsRealZeroPercent proves a running job whose scope IS known
// still shows 0.0% (a genuine "0 of N"), not the preparing label (A3).
func TestRenderGetJobKeepsRealZeroPercent(t *testing.T) {
	t.Parallel()
	job := &model.Job{
		ID:            "job_zero",
		CanonicalPath: "/repo",
		Operation:     "index",
		State:         model.JobStateRunning,
		Progress:      model.Progress{FilesTotal: 58, OverallPercent: 0},
	}
	out := renderGetJob(resolveJobEntry(*job, false, ""), true)
	if !strings.Contains(out, "Progress: 0.0%") {
		t.Fatalf("known-scope zero should render 0.0%%, got:\n%s", out)
	}
	if strings.Contains(out, "Preparing to index") {
		t.Fatalf("known-scope job should not show preparing, got:\n%s", out)
	}
}

// TestRenderGetJobShowsMeasuredPercent proves a measured percent renders as-is (A4).
func TestRenderGetJobShowsMeasuredPercent(t *testing.T) {
	t.Parallel()
	job := &model.Job{
		ID:            "job_mid",
		CanonicalPath: "/repo",
		Operation:     "index",
		State:         model.JobStateRunning,
		Progress:      model.Progress{FilesTotal: 4292, FilesProcessed: 2139, OverallPercent: 49.8},
	}
	out := renderGetJob(resolveJobEntry(*job, false, ""), true)
	if !strings.Contains(out, "Progress: 49.8%") {
		t.Fatalf("expected 49.8%%, got:\n%s", out)
	}
}

// TestRenderGetJobFailedShowsPercentAndError proves a terminal failed job keeps
// its percent and error line, never the preparing label, even at 0% (A5).
func TestRenderGetJobFailedShowsPercentAndError(t *testing.T) {
	t.Parallel()
	job := &model.Job{
		ID:            "job_fail",
		CanonicalPath: "/repo",
		Operation:     "index",
		State:         model.JobStateFailed,
		Progress:      model.Progress{FilesTotal: 0, OverallPercent: 0},
		Error:         &model.JobError{Message: "embedder_unreachable: dial tcp [::1]:5400: connect: connection refused"},
	}
	out := renderGetJob(resolveJobEntry(*job, false, ""), true)
	if !strings.Contains(out, "Progress: 0.0%") {
		t.Fatalf("failed job should show its percent, got:\n%s", out)
	}
	if strings.Contains(out, "Preparing to index") {
		t.Fatalf("failed (terminal) job must not show preparing, got:\n%s", out)
	}
	if !strings.Contains(out, "Error: embedder_unreachable") {
		t.Fatalf("failed job should show error line, got:\n%s", out)
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
		"ready":       renderIndexedDetail(codebase),
		"preparing":   renderIndexingActive(codebase, &model.Job{Operation: "sync", Progress: model.Progress{FilesTotal: 0}}),
		"building":    renderIndexingActive(codebase, &model.Job{Operation: "index", Progress: model.Progress{FilesTotal: 58, FilesProcessed: 7, ChunksGenerated: 84}}),
		"incremental": renderIndexingActive(codebase, &model.Job{Operation: "sync", Progress: model.Progress{FilesTotal: 58, FilesProcessed: 7, FilesInCodebase: 100, FilesAdded: 5, FilesModified: 50, FilesRemoved: 3, FilesEmbedded: 2, ChunksGenerated: 84, ChunksTotal: 620}}),
	}
	wantLines := map[string]int{"ready": 4, "preparing": 3, "building": 7, "incremental": 11}
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

// TestRenderClearIndexHasNoRemainLine proves the clear output is just the
// success line, with no trailing "other ... remain" count.
func TestRenderClearIndexHasNoRemainLine(t *testing.T) {
	t.Parallel()
	codebase := model.Codebase{CanonicalPath: "/Users/agoodkind/Sites/swift-makefile"}
	out := renderClearIndex(codebase)
	want := "Successfully cleared codebase '/Users/agoodkind/Sites/swift-makefile'"
	if out != want {
		t.Fatalf("renderClearIndex = %q, want %q", out, want)
	}
	if strings.Contains(out, "remain") {
		t.Fatalf("renderClearIndex still contains a remain line:\n%s", out)
	}
}

// TestRenderListIndexesShowsIDAndPluralHeader proves each row carries the
// copy-pasteable id and the header pluralizes the codebase count correctly.
func TestRenderListIndexesShowsIDAndPluralHeader(t *testing.T) {
	t.Parallel()
	single := []CodebaseView{
		{Codebase: model.Codebase{ID: "cb_1_aaaa", CanonicalPath: "/tmp/alpha", Status: model.CodebaseStatusIndexed}, Display: displayIndexed},
	}
	out := renderListIndexes(single)
	if !strings.Contains(out, "Tracked 1 codebase:") {
		t.Fatalf("single header not singular:\n%s", out)
	}
	if !strings.Contains(out, "cb_1_aaaa") || !strings.Contains(out, "/tmp/alpha") || !strings.Contains(out, "[indexed]") {
		t.Fatalf("row missing id, path, or display status:\n%s", out)
	}

	many := []CodebaseView{
		{Codebase: model.Codebase{ID: "cb_1_aaaa", CanonicalPath: "/tmp/alpha", Status: model.CodebaseStatusIndexed}, Display: displayIndexed},
		{Codebase: model.Codebase{ID: "cb_2_bbbb", CanonicalPath: "/tmp/beta", Status: model.CodebaseStatusIndexing}, Display: displayPreparing},
	}
	outMany := renderListIndexes(many)
	if !strings.Contains(outMany, "Tracked 2 codebases:") {
		t.Fatalf("plural header wrong:\n%s", outMany)
	}
	if !strings.Contains(outMany, "[preparing]") {
		t.Fatalf("list should show the display status, not raw status:\n%s", outMany)
	}
	if strings.Contains(outMany, "(s)") {
		t.Fatalf("list still uses a (s) suffix:\n%s", outMany)
	}
}

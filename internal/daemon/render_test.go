package daemon

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/pbconv"
	render "goodkind.io/lm-semantic-search/internal/render"
	"goodkind.io/lm-semantic-search/internal/view"
	"google.golang.org/protobuf/encoding/protojson"
)

// treeLines keeps only the file-and-chunk tree lines (the breakdown block) from
// a rendered status or job output, raw and in order, so two surfaces can be
// compared byte-for-byte. A divergence in wording or indentation fails.
func treeLines(out string) []string {
	kept := []string{}
	for _, line := range strings.Split(out, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "📄") || strings.HasPrefix(trimmed, "🧩") ||
			strings.HasPrefix(trimmed, "├─") || strings.HasPrefix(trimmed, "└─") {
			kept = append(kept, line)
		}
	}
	return kept
}

// TestStatusTreeIdenticalAcrossSurfaces is the structural guard against
// diverging status output: the compact job surface and the codebase status
// surface must carry the same resolved tree and render it byte-identically, so
// a status can never read one way under `job get` and another under
// `get_indexing_status`.
func TestStatusTreeIdenticalAcrossSurfaces(t *testing.T) {
	t.Parallel()
	codebase := &model.Codebase{
		CanonicalPath:     "/Users/agoodkind/Sites/swift-makefile",
		LastSuccessfulRun: &model.IndexRunSummary{TotalChunks: 57240},
	}
	job := model.Job{
		ID:            "job_ident",
		CanonicalPath: codebase.CanonicalPath,
		State:         model.JobStateRunning,
		Operation:     "streaming_reindex",
		Progress: model.Progress{
			RunMode:        model.RunModeChanged,
			OverallPercent: 37, FilesTotal: 452, FilesProcessed: 290,
			FilesInCodebase: 4292, FilesAdded: 29, FilesModified: 423, FilesRemoved: 10,
			FilesEmbedded: 285, FilesSkippedOversize: 3, FilesSkippedUnreadable: 2,
			ChunksGenerated: 1043, ChunksTotal: 57240, LastEventAt: renderTestTime,
		},
	}
	compact := resolveProgressSurface(job).Breakdown
	status, _ := resolveStatusView(*codebase, &job, displayIndexing, "")
	if !reflect.DeepEqual(compact, status.Breakdown) {
		t.Fatalf("breakdown differs between surfaces:\ncompact=%+v\nstatus =%+v", compact, status.Breakdown)
	}
	jobTree := treeLines(render.GetJob(resolveJobEntry(job, false, ""), true))
	statusTree := treeLines(renderActiveStatusForTest(codebase, &job))
	if !reflect.DeepEqual(jobTree, statusTree) {
		t.Fatalf("rendered tree differs between surfaces:\njob:\n%s\nstatus:\n%s",
			strings.Join(jobTree, "\n"), strings.Join(statusTree, "\n"))
	}
	if len(jobTree) == 0 {
		t.Fatal("no tree lines extracted; the guard would pass vacuously")
	}
}

// TestBreakdownIdenticalAcrossAllSurfaces is the single-SOT guard. The daemon
// text, the proto breakdown rendered back (the TUI path), and the JSON-decoded
// breakdown rendered back must all equal the one render.BreakdownLines over the
// one view.ResolveBreakdown. Any surface that re-derives breaks this test.
func TestBreakdownIdenticalAcrossAllSurfaces(t *testing.T) {
	t.Parallel()
	progress := model.Progress{
		RunMode: model.RunModeChanged, Unit: "document", ScopeUnit: "conversation",
		FilesTotal: 72, FilesProcessed: 70, FilesAdded: 63, FilesModified: 9,
		FilesEmbedded: 9, FilesPending: 61,
		ChunksProcessed: 3516, ChunksGenerated: 1204, ChunksEmbedded: 1204,
		ChunksReused: 2312, ChunksTotal: 3516, ReuseVectorsLoaded: 2400,
		LastEventAt: renderTestTime,
	}

	// The single source of truth: one resolver, one renderer.
	want := render.BreakdownLines(view.ResolveBreakdown(pbconv.ProgressCounts(progress)))
	if len(want) == 0 {
		t.Fatal("breakdown produced no lines; the guard would pass vacuously")
	}

	job := model.Job{ID: "j", State: model.JobStateRunning, Operation: "conversation_ingest", Progress: progress}

	// Daemon compact text (job get / job list path).
	daemonText := treeLines(render.GetJob(resolveJobEntry(job, false, ""), true))
	if !reflect.DeepEqual(daemonText, want) {
		t.Fatalf("daemon text differs from SOT:\n%s\nwant\n%s", strings.Join(daemonText, "\n"), strings.Join(want, "\n"))
	}

	// Proto breakdown rendered back: the path the TUI uses from active_progress.
	pbJob := pbconv.ToJob(job)
	protoRender := render.BreakdownLines(pbconv.BreakdownFromProto(pbJob.GetProgress().GetBreakdown()))
	if !reflect.DeepEqual(protoRender, want) {
		t.Fatalf("proto-rendered breakdown differs from SOT:\n%s", strings.Join(protoRender, "\n"))
	}

	// JSON (--json) decoded and rendered back.
	data, err := protojson.Marshal(pbJob)
	if err != nil {
		t.Fatalf("protojson.Marshal returned error: %v", err)
	}
	var decoded pb.Job
	if err := protojson.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("protojson.Unmarshal returned error: %v", err)
	}
	jsonRender := render.BreakdownLines(pbconv.BreakdownFromProto(decoded.GetProgress().GetBreakdown()))
	if !reflect.DeepEqual(jsonRender, want) {
		t.Fatalf("JSON-decoded breakdown differs from SOT:\n%s", strings.Join(jsonRender, "\n"))
	}

	// The undelivered conversation is a first-class wire counter, not inferred.
	if got := decoded.GetProgress().GetBreakdown(); !breakdownHasPending(got) {
		t.Fatalf("JSON breakdown missing a pending row: %+v", got.GetFileRows())
	}
	if got := decoded.GetProgress().GetChunksProcessed(); got != 3516 {
		t.Fatalf("JSON chunks_processed = %d, want 3516", got)
	}
	if got := decoded.GetProgress().GetChunksEmbedded(); got != 1204 {
		t.Fatalf("JSON chunks_embedded = %d, want 1204", got)
	}
	if got := decoded.GetProgress().GetReuseVectorsLoaded(); got != 2400 {
		t.Fatalf("JSON reuse_vectors_loaded = %d, want 2400", got)
	}
}

// breakdownHasPending reports whether the proto breakdown carries a pending row.
func breakdownHasPending(breakdown *pb.OutcomeBreakdown) bool {
	for _, row := range breakdown.GetFileRows() {
		if row.GetKind() == pb.OutcomeKind_OUTCOME_KIND_PENDING {
			return true
		}
	}
	return false
}

// TestStatusTreeMatchesSessionCases locks the two real cases from the design
// session to their exact trees: a code delta and a conversation ingest. The
// file rows sum to the processed count in both.
func TestStatusTreeMatchesSessionCases(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		job  model.Job
		want []string
	}{
		{
			name: "code delta 1 of 2",
			job: model.Job{
				State: model.JobStateRunning, Operation: "streaming_reindex",
				Progress: model.Progress{
					RunMode: model.RunModeChanged, Unit: "file",
					FilesTotal: 2, FilesProcessed: 1, FilesAdded: 2, FilesEmbedded: 1,
					ChunksGenerated: 778, ChunksTotal: 778, LastEventAt: renderTestTime,
				},
			},
			want: []string{
				"📄 1 of 2 changed files processed",
				"└─ ➕ 1 embedded",
				"🧩 778 chunks total",
				"├─ ➕ 778 added",
				"└─ ♻️ 0 reused",
			},
		},
		{
			name: "conversation ingest 70 of 72",
			job: model.Job{
				State: model.JobStateRunning, Operation: "conversation_ingest",
				Progress: model.Progress{
					RunMode: model.RunModeChanged, Unit: "document", ScopeUnit: "conversation",
					FilesTotal: 72, FilesProcessed: 70, FilesAdded: 63, FilesModified: 9,
					FilesEmbedded: 9, FilesPending: 61,
					ChunksGenerated: 1204, ChunksReused: 2312, ChunksTotal: 3516, LastEventAt: renderTestTime,
				},
			},
			want: []string{
				"📄 70 of 72 changed documents processed",
				"├─ ➕ 9 embedded",
				"└─ ⏳ 61 pending, not sent yet",
				"🧩 3,516 chunks total",
				"├─ ➕ 1,204 added",
				"└─ ♻️ 2,312 reused",
			},
		},
	}
	for _, testCase := range cases {
		breakdown := resolveProgressSurface(testCase.job).Breakdown
		if sum := sumRowCounts(breakdown.FileRows); sum != breakdown.Processed {
			t.Fatalf("%s: file rows sum to %d, want processed %d", testCase.name, sum, breakdown.Processed)
		}
		got := treeLines(render.GetJob(resolveJobEntry(testCase.job, false, ""), true))
		want := testCase.want
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("%s tree =\n%s\nwant\n%s", testCase.name, strings.Join(got, "\n"), strings.Join(want, "\n"))
		}
	}
}

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
	return render.ListJobs(summary, active, terminal)
}

func renderStatusForTest(codebase *model.Codebase, activeJob *model.Job, display displayStatus) string {
	statusView, templateName := resolveStatusView(*codebase, activeJob, display, waitingLabel(dependencyHealthy))
	return render.GetIndex(view.GetIndexView{
		Tracked:      true,
		TemplateName: templateName,
		Status:       statusView,
	})
}

func renderActiveStatusForTest(codebase *model.Codebase, activeJob *model.Job) string {
	return renderStatusForTest(codebase, activeJob, displayIndexing)
}

func renderReadyStatusForTest(codebase *model.Codebase) string {
	return renderStatusForTest(codebase, nil, displayIndexed)
}

func renderGetIndexBodyForTest(requestedPath string, tracked bool, codebase *model.Codebase, activeJob *model.Job, health dependencyHealth) string {
	getIndex := view.GetIndexView{
		Tracked:            tracked,
		RequestedPath:      requestedPath,
		CanonicalPath:      "",
		Display:            "",
		TemplateName:       "",
		Status:             view.StatusView{},
		Failure:            view.FailureSurface{},
		Quarantine:         view.QuarantineSurface{},
		WaitLabel:          "",
		ClassificationLine: "",
		ResolutionLines:    nil,
		CoverageLine:       "",
		DescendantsHint:    "",
		SyncNote:           "",
	}
	if tracked && codebase != nil {
		getIndex.CanonicalPath = codebase.CanonicalPath
		display := computeDisplayStatus(*codebase, activeJob, health.Mode, collectionNotApplicable)
		getIndex.Display = view.Display(display)
		getIndex.Failure = resolveCodebaseFailure(*codebase)
		getIndex.Quarantine = resolveQuarantineSurface(*codebase)
		statusView, templateName := resolveStatusView(*codebase, activeJob, display, waitingLabel(health.Mode))
		getIndex.Status = statusView
		getIndex.TemplateName = templateName
		getIndex.Narrative = resolveStatusNarrative(display, codebase.CanonicalPath, getIndex.Failure, getIndex.Quarantine, statusView)
	}
	return render.GetIndex(getIndex)
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
	out := renderReadyStatusForTest(codebase)
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
	out := renderActiveStatusForTest(codebase, job)
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
	out := renderActiveStatusForTest(codebase, job)
	if !strings.Contains(out, "⚙️ Preparing to index") {
		t.Fatalf("expected plain prepare line in:\n%s", out)
	}
	if strings.Contains(out, "Changes detected") {
		t.Fatalf("did not expect changes-detected for a forced reindex in:\n%s", out)
	}
}

// TestRenderIndexingActiveBuilding proves a cold from-scratch build reads as
// "Building initial index" with the percent, files embedded, and chunks so far.
func TestRenderIndexingActiveBuilding(t *testing.T) {
	t.Parallel()
	codebase := &model.Codebase{CanonicalPath: "/Users/agoodkind/Sites/swift-makefile"}
	job := &model.Job{
		Operation: "index",
		Progress:  model.Progress{RunMode: model.RunModeFirstBuild, OverallPercent: 42, FilesTotal: 58, FilesProcessed: 24, FilesEmbedded: 24, ChunksGenerated: 71, LastEventAt: renderTestTime},
	}
	out := renderActiveStatusForTest(codebase, job)
	for _, want := range []string{"📁 swift-makefile", "🔄 Building initial index: 42%", "📄 24 of 58 files (full build) processed", "└─ ➕ 24 embedded", "🧩 71 chunks total", "└─ ➕ 71 added"} {
		if !strings.Contains(out, want) {
			t.Fatalf("building status missing %q in:\n%s", want, out)
		}
	}
	// A cold first build loads and serves no reuse vectors, so the chunk tree
	// omits the reused row; a seeded first build shows it (see the next test).
	if strings.Contains(out, "reused") {
		t.Fatalf("a cold first build should omit the reused row in:\n%s", out)
	}
}

// TestRenderIndexingActiveSeededFirstBuild proves a first build that loads
// sibling vectors does not read like a cold full build.
func TestRenderIndexingActiveSeededFirstBuild(t *testing.T) {
	t.Parallel()
	codebase := &model.Codebase{CanonicalPath: "/Users/agoodkind/Sites/swift-makefile"}
	job := &model.Job{
		Operation: "index",
		Progress: model.Progress{
			RunMode:            model.RunModeFirstBuild,
			OverallPercent:     42,
			FilesTotal:         58,
			FilesProcessed:     1,
			FilesEmbedded:      1,
			ChunksProcessed:    678,
			ChunksReused:       623,
			ChunksEmbedded:     55,
			ReuseVectorsLoaded: 2316,
			LastEventAt:        renderTestTime,
		},
	}
	out := renderActiveStatusForTest(codebase, job)
	for _, want := range []string{
		"📁 swift-makefile",
		"🔄 Building initial index: 42%",
		"📄 1 of 58 files (first build, reusing prior vectors) processed",
		"└─ ➕ 1 embedded",
		"🧩 678 chunks total",
		"├─ ➕ 55 added",
		"└─ ♻️ 623 reused",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("seeded building status missing %q in:\n%s", want, out)
		}
	}
	if strings.Contains(out, "(full build)") {
		t.Fatalf("seeded first build should not use the cold full-build label in:\n%s", out)
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
			RunMode:        model.RunModeChanged,
			OverallPercent: 37, FilesTotal: 452, FilesProcessed: 290,
			FilesInCodebase: 4292, FilesAdded: 29, FilesModified: 423, FilesRemoved: 10,
			FilesEmbedded: 285, FilesSkippedOversize: 3, FilesSkippedUnreadable: 2,
			ChunksGenerated: 1043, ChunksTotal: 57240, LastEventAt: renderTestTime,
		},
	}
	out := renderActiveStatusForTest(codebase, job)
	for _, want := range []string{
		"📁 swift-makefile",
		"🔄 Indexing new changes: 37%",
		"🔢 4292 files: 462 changed, 3830 unchanged",
		"📄 300 of 462 changed files processed",
		"➕ 285 embedded",
		"🗑️ 10 removed",
		"📏 3 skipped, too large",
		"⚠️ 2 error, unreadable",
		"🧩 57,240 chunks total",
		"➕ 1,043 added",
		"♻️ 0 reused",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("incremental status missing %q in:\n%s", want, out)
		}
	}
	// The old derived "already indexed" column is gone, and reuse is visible.
	if strings.Contains(out, "already indexed") {
		t.Fatalf("status still emits 'already indexed' in:\n%s", out)
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
			RunMode:        model.RunModeChanged,
			OverallPercent: 0, FilesTotal: 0, FilesProcessed: 0,
			FilesInCodebase: 4292, FilesAdded: 0, FilesModified: 118, FilesRemoved: 0,
			FilesEmbedded: 0, ChunksGenerated: 0,
		},
	}
	out := renderActiveStatusForTest(codebase, job)
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
	out := renderActiveStatusForTest(codebase, job)
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
	out := renderGetIndexBodyForTest("/Users/agoodkind/Sites/swift-makefile", true, codebase, job, dependencyHealth{})
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
	out := renderGetIndexBodyForTest("/Users/agoodkind/Sites/swift-makefile", true, codebase, job, dependencyHealth{})
	for _, want := range []string{"✅ Ready to search", "🔄 Checking for changes in the background"} {
		if !strings.Contains(out, want) {
			t.Fatalf("pre-diff sync status missing %q in:\n%s", want, out)
		}
	}
}

func TestRenderGetIndexBodyQuarantinedPreservesSearchabilityMessage(t *testing.T) {
	t.Parallel()
	codebase := &model.Codebase{
		CanonicalPath:     "/Users/agoodkind/Sites/swift-makefile",
		Status:            model.CodebaseStatusQuarantined,
		LastSuccessfulRun: &model.IndexRunSummary{IndexedFiles: 58, TotalChunks: 600, CompletedAt: renderTestTime},
		Quarantine: &model.QuarantineState{
			Reason:           quarantineReasonWatcherLargeDelete,
			FirstObservedAt:  renderTestTime,
			LastObservedAt:   renderTestTime,
			ObservationCount: 1,
			LastTrigger:      quarantineTriggerWatcher,
			LastMissingCount: 400,
			LastTotalCount:   4292,
		},
	}
	out := renderGetIndexBodyForTest("/Users/agoodkind/Sites/swift-makefile", true, codebase, nil, dependencyHealth{})
	for _, want := range []string{
		"is quarantined after a suspicious large disappearance",
		"Search continues to serve the last known-good index",
		"Last known good index: 58 files, 600 chunks",
		"400 of 4,292 tracked files",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("quarantined status missing %q in:\n%s", want, out)
		}
	}
}

// TestRenderGetIndexBodyNonTemplateEmptyNarrativeFallsBack proves the render
// fallback: a non-template display that arrives without a narrative (a caller
// that skipped resolveStatusNarrative) surfaces the status word and path rather
// than a blank body.
func TestRenderGetIndexBodyNonTemplateEmptyNarrativeFallsBack(t *testing.T) {
	t.Parallel()
	out := render.GetIndex(view.GetIndexView{
		Tracked:       true,
		RequestedPath: "/repo",
		CanonicalPath: "/repo",
		Display:       view.Display(displayQuarantined),
	})
	if !strings.Contains(out, "Codebase '/repo' status: quarantined") {
		t.Fatalf("empty-narrative fallback missing status word in:\n%s", out)
	}
}

// TestRenderGetIndexBodyBuildingTakesOver proves a from-scratch build still
// owns the display, because its staging collection is not promoted yet.
func TestRenderGetIndexBodyBuildingTakesOver(t *testing.T) {
	t.Parallel()
	codebase := &model.Codebase{CanonicalPath: "/Users/agoodkind/Sites/swift-makefile", Status: model.CodebaseStatusIndexing}
	job := &model.Job{Operation: "index", Progress: model.Progress{OverallPercent: 42, FilesTotal: 58, FilesProcessed: 24, ChunksGenerated: 71}}
	out := renderGetIndexBodyForTest("/Users/agoodkind/Sites/swift-makefile", true, codebase, job, dependencyHealth{})
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
	out := renderGetIndexBodyForTest("/Users/agoodkind/Sites/swift-makefile", true, codebase, job, dependencyHealth{})
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
		Heading: "",
		Breakdown: view.OutcomeBreakdown{
			ScopeLabel:  "",
			Processed:   0,
			ScopeTotal:  0,
			FileRows:    nil,
			ChunksTotal: 0,
			ChunkRows:   nil,
		},
		ScopeLine:    "",
		PercentLabel: "",
	}
	emptyOut := render.GetJob(view.JobEntryView{ID: "job_empty", Progress: empty}, true)
	if strings.Contains(emptyOut, "📄") || strings.Contains(emptyOut, "🧩") {
		t.Fatalf("expected no progress lines for an empty view, got:\n%s", emptyOut)
	}
	progress := view.ProgressSurface{
		Heading: "",
		Breakdown: view.OutcomeBreakdown{
			ScopeLabel:  "files (full build)",
			Processed:   7,
			ScopeTotal:  58,
			FileRows:    []view.OutcomeRow{view.NewOutcomeRow(view.KindEmbedded, 7)},
			ChunksTotal: 84,
			ChunkRows:   []view.OutcomeRow{view.NewOutcomeRow(view.KindAdded, 84)},
		},
		ScopeLine:    "Changed since last sync: 12 files added · 30 modified · 5 removed",
		PercentLabel: "12.1%",
	}
	got := render.GetJob(view.JobEntryView{ID: "job_progress", Progress: progress}, true)
	for _, want := range []string{
		"📄 7 of 58 files (full build) processed",
		"└─ ➕ 7 embedded",
		"🧩 84 chunks total",
		"└─ ➕ 84 added",
		"Changed since last sync: 12 files added · 30 modified · 5 removed",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("progress lines missing %q in:\n%s", want, got)
		}
	}
}

func TestRenderGetJobNotFound(t *testing.T) {
	t.Parallel()
	if got := render.GetJob(view.JobEntryView{}, false); got != "Job not found." {
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
		Progress:      model.Progress{FilesTotal: 58, FilesProcessed: 7, FilesEmbedded: 7, ChunksGenerated: 84},
	}
	out := render.GetJob(resolveJobEntry(*job, false, ""), true)
	if !strings.Contains(out, "📄 7 of 58 files processed") {
		t.Fatalf("expected magnitude in job view, got:\n%s", out)
	}
	if !strings.Contains(out, "🧩 84 chunks total") {
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
	out := render.GetJob(resolveJobEntry(*job, false, ""), true)
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
	if want := strings.Repeat("─", 40); !strings.Contains(out, want) {
		t.Fatalf("two jobs in a section must be separated by the divider in:\n%s", out)
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
	out := render.GetJob(resolveJobEntry(*job, false, ""), true)
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
	if strings.Contains(out, strings.Repeat("─", 40)) {
		t.Fatalf("a single job must not be wrapped by a divider, got:\n%s", out)
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
	out := render.GetJob(resolveJobEntry(*job, false, ""), true)
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
	out := render.GetJob(resolveJobEntry(*job, false, ""), true)
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
	out := render.GetJob(resolveJobEntry(*job, false, ""), true)
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
	out := render.GetJob(resolveJobEntry(*job, false, ""), true)
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
		"ready":       renderReadyStatusForTest(codebase),
		"preparing":   renderActiveStatusForTest(codebase, &model.Job{Operation: "sync", Progress: model.Progress{FilesTotal: 0}}),
		"building":    renderActiveStatusForTest(codebase, &model.Job{Operation: "index", Progress: model.Progress{RunMode: model.RunModeForcedReindex, FilesTotal: 58, FilesProcessed: 7, FilesEmbedded: 7, ChunksGenerated: 84}}),
		"incremental": renderActiveStatusForTest(codebase, &model.Job{Operation: "sync", Progress: model.Progress{RunMode: model.RunModeChanged, FilesTotal: 58, FilesProcessed: 7, FilesInCodebase: 100, FilesAdded: 5, FilesModified: 50, FilesRemoved: 3, FilesEmbedded: 2, ChunksGenerated: 84, ChunksTotal: 620}}),
	}
	wantLines := map[string]int{"ready": 4, "preparing": 3, "building": 8, "incremental": 11}
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
	out := render.MutationAck(view.MutationAckView{Kind: view.AckClear, Path: codebase.CanonicalPath})
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
	single := []view.CodebaseRowView{
		{ID: "cb_1_aaaa", CanonicalPath: "/tmp/alpha", Display: view.Display(displayIndexed)},
	}
	out := render.ListIndexes(single)
	if !strings.Contains(out, "Tracked 1 codebase:") {
		t.Fatalf("single header not singular:\n%s", out)
	}
	if !strings.Contains(out, "cb_1_aaaa") || !strings.Contains(out, "/tmp/alpha") || !strings.Contains(out, "[indexed]") {
		t.Fatalf("row missing id, path, or display status:\n%s", out)
	}

	many := []view.CodebaseRowView{
		{ID: "cb_1_aaaa", CanonicalPath: "/tmp/alpha", Display: view.Display(displayIndexed)},
		{ID: "cb_2_bbbb", CanonicalPath: "/tmp/beta", Display: view.Display(displayPreparing)},
	}
	outMany := render.ListIndexes(many)
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

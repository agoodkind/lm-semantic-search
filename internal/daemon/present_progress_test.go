package daemon

import (
	"strings"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/view"
)

// findRow returns the outcome row with the given kind, or fails the test.
func findRow(t *testing.T, rows []view.OutcomeRow, kind view.OutcomeKind) view.OutcomeRow {
	t.Helper()
	for _, row := range rows {
		if row.Kind() == kind {
			return row
		}
	}
	t.Fatalf("row %q not found in %+v", kind, rows)
	return view.NewOutcomeRow("", 0)
}

// hasRow reports whether a row with the given kind is present.
func hasRow(rows []view.OutcomeRow, kind view.OutcomeKind) bool {
	for _, row := range rows {
		if row.Kind() == kind {
			return true
		}
	}
	return false
}

// sumRowCounts totals the counts of every row, the invariant the file tree must
// satisfy against Processed.
func sumRowCounts(rows []view.OutcomeRow) int32 {
	var total int32
	for _, row := range rows {
		total += row.Count()
	}
	return total
}

func TestFormatCount(t *testing.T) {
	t.Parallel()
	cases := map[int32]string{
		0:      "0",
		29:     "29",
		1011:   "1,011",
		33240:  "33,240",
		124754: "124,754",
	}
	for in, want := range cases {
		if got := formatCount(in); got != want {
			t.Errorf("formatCount(%d) = %q, want %q", in, got, want)
		}
	}
}

func TestResolveProgressSurfaceResumingIngest(t *testing.T) {
	t.Parallel()
	job := model.Job{
		ID:        "job-p",
		State:     model.JobStateRunning,
		Operation: "conversation_ingest",
		Progress: model.Progress{
			RunMode:         model.RunModeResuming,
			Unit:            "document",
			ScopeUnit:       "conversation",
			OverallPercent:  23.5,
			FilesTotal:      1011,
			FilesProcessed:  238,
			FilesEmbedded:   12,
			FilesAdded:      1004,
			FilesModified:   7,
			ChunksGenerated: 29,
			ChunksTotal:     33240,
		},
	}
	got := resolveProgressSurface(job)
	breakdown := got.Breakdown
	if got.Heading == "" || !strings.Contains(got.Heading, "Resuming after restart") {
		t.Fatalf("heading = %q, want the resume heading", got.Heading)
	}
	if breakdown.ScopeLabel != "changed documents" {
		t.Fatalf("scope label = %q, want %q", breakdown.ScopeLabel, "changed documents")
	}
	// The 226 that the old derived "already indexed" column carried is now the
	// honest seed-reuse remainder, shown as an unchanged row, and the file rows
	// sum to Processed rather than hiding three buckets.
	if unchanged := findRow(t, breakdown.FileRows, view.KindUnchanged).Count(); unchanged != 226 {
		t.Fatalf("unchanged = %d, want 226 (238 processed minus 12 embedded)", unchanged)
	}
	if embedded := findRow(t, breakdown.FileRows, view.KindEmbedded).Count(); embedded != 12 {
		t.Fatalf("embedded = %d, want 12", embedded)
	}
	if breakdown.Processed != 238 || sumRowCounts(breakdown.FileRows) != 238 {
		t.Fatalf("processed = %d, file rows sum = %d, want both 238", breakdown.Processed, sumRowCounts(breakdown.FileRows))
	}
	if breakdown.ChunksTotal != 33240 {
		t.Fatalf("collection total = %d, want 33240", breakdown.ChunksTotal)
	}
	// A resuming pass is reuse-capable, so the reused chunk row shows even at zero.
	if !hasRow(breakdown.ChunkRows, view.KindReused) {
		t.Fatalf("reused row missing on a reuse-capable pass: %+v", breakdown.ChunkRows)
	}
	if !strings.Contains(got.ScopeLine, "1,004 conversations added · 7 modified") {
		t.Fatalf("scope line = %q, want the typed classification", got.ScopeLine)
	}
}

func TestResolveProgressSurfaceColdFirstBuild(t *testing.T) {
	t.Parallel()
	job := model.Job{
		ID:        "job-fb",
		State:     model.JobStateRunning,
		Operation: "index",
		Progress: model.Progress{
			RunMode:        model.RunModeFirstBuild,
			FilesTotal:     100,
			FilesProcessed: 10,
			FilesEmbedded:  10,
			OverallPercent: 10,
		},
	}
	got := resolveProgressSurface(job)
	breakdown := got.Breakdown
	if breakdown.ScopeLabel != "files (full build)" {
		t.Fatalf("scope label = %q, want %q", breakdown.ScopeLabel, "files (full build)")
	}
	if embedded := findRow(t, breakdown.FileRows, view.KindEmbedded).Count(); embedded != 10 {
		t.Fatalf("embedded = %d, want 10 for a full build", embedded)
	}
	if hasRow(breakdown.FileRows, view.KindUnchanged) {
		t.Fatalf("a full build should have no unchanged row: %+v", breakdown.FileRows)
	}
	if hasRow(breakdown.ChunkRows, view.KindReused) {
		t.Fatalf("a cold first build should omit the reused row: %+v", breakdown.ChunkRows)
	}
}

func TestResolveProgressSurfaceSeededFirstBuild(t *testing.T) {
	t.Parallel()
	job := model.Job{
		ID:        "job-fb-seeded",
		State:     model.JobStateRunning,
		Operation: "index",
		Progress: model.Progress{
			RunMode:            model.RunModeFirstBuild,
			FilesTotal:         100,
			FilesProcessed:     1,
			FilesEmbedded:      1,
			ChunksProcessed:    678,
			ChunksReused:       623,
			ChunksEmbedded:     55,
			ReuseVectorsLoaded: 2316,
			OverallPercent:     10,
		},
	}
	got := resolveProgressSurface(job)
	breakdown := got.Breakdown
	if breakdown.ScopeLabel != "files (first build, reusing prior vectors)" {
		t.Fatalf("scope label = %q, want seeded first build label", breakdown.ScopeLabel)
	}
	if reused := findRow(t, breakdown.ChunkRows, view.KindReused).Count(); reused != 623 {
		t.Fatalf("reused chunks = %d, want 623", reused)
	}
}

func TestResolveListSummarySplitsSuperseded(t *testing.T) {
	t.Parallel()
	older := time.Now().Add(-2 * time.Minute)
	newer := time.Now().Add(-1 * time.Minute)
	jobs := []model.Job{
		{
			ID:          "a1",
			CodebaseID:  "A",
			State:       model.JobStateFailed,
			StartedAt:   older,
			CompletedAt: &older,
			Error:       &model.JobError{Message: "x", Retryable: true},
		},
		{
			ID:          "a2",
			CodebaseID:  "A",
			State:       model.JobStateFailed,
			StartedAt:   older,
			CompletedAt: &newer,
			Error:       &model.JobError{Message: "y", Retryable: false},
		},
		{ID: "b1", CodebaseID: "B", State: model.JobStateCompleted, StartedAt: older, CompletedAt: &newer},
	}
	got := resolveListSummary(jobs, false)
	if got.Failed != 1 || got.Superseded != 1 || got.Completed != 1 {
		t.Fatalf("summary = %+v, want 1 failed, 1 superseded, 1 completed", got)
	}
}

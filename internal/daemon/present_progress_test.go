package daemon

import (
	"strings"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/model"
)

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
	if got.Heading == "" || !strings.Contains(got.Heading, "Resuming after restart") {
		t.Fatalf("heading = %q, want the resume heading", got.Heading)
	}
	if got.ScopeLabel != "changed documents" {
		t.Fatalf("scope label = %q, want %q", got.ScopeLabel, "changed documents")
	}
	if got.CheckVerb != "checked" {
		t.Fatalf("check verb = %q, want checked", got.CheckVerb)
	}
	if got.AlreadyIndexed != 226 {
		t.Fatalf("already indexed = %d, want 226 (238 checked minus 12 embedded)", got.AlreadyIndexed)
	}
	if got.ChunksInCollection != 33240 {
		t.Fatalf("collection total = %d, want 33240", got.ChunksInCollection)
	}
	if !strings.Contains(got.ScopeLine, "1,004 conversations added · 7 modified") {
		t.Fatalf("scope line = %q, want the typed classification", got.ScopeLine)
	}
}

func TestResolveProgressSurfaceFirstBuild(t *testing.T) {
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
	if got.ScopeLabel != "files (full build)" {
		t.Fatalf("scope label = %q, want %q", got.ScopeLabel, "files (full build)")
	}
	if got.CheckVerb != "embedded" {
		t.Fatalf("check verb = %q, want embedded for a full build", got.CheckVerb)
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

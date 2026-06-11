package daemon

import (
	"context"
	"slices"
	"strings"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/view"
)

func TestComputeDroppedCodebasesReportsCompletedUntrackedOnDisk(t *testing.T) {
	t.Parallel()

	jobs := []model.Job{
		{CanonicalPath: "/repo/dropped", State: model.JobStateCompleted},
		{CanonicalPath: "/repo/tracked", State: model.JobStateCompleted},
		{CanonicalPath: "/repo/gone", State: model.JobStateCompleted},
		{CanonicalPath: "/repo/failed", State: model.JobStateFailed},
		{CanonicalPath: "/repo/running", State: model.JobStateRunning},
	}
	codebases := []model.Codebase{
		{CanonicalPath: "/repo/tracked"},
	}
	onDisk := map[string]bool{
		"/repo/dropped": true,
		"/repo/tracked": true,
		"/repo/gone":    false,
		"/repo/failed":  true,
		"/repo/running": true,
	}
	exists := func(path string) bool {
		return onDisk[path]
	}

	dropped := computeDroppedCodebases(jobs, codebases, exists)

	want := []string{"/repo/dropped"}
	if !slices.Equal(dropped, want) {
		t.Fatalf("computeDroppedCodebases = %v, want %v", dropped, want)
	}
}

func TestComputeDroppedCodebasesIgnoresNeverIndexed(t *testing.T) {
	t.Parallel()

	dropped := computeDroppedCodebases(nil, nil, func(string) bool { return true })
	if len(dropped) != 0 {
		t.Fatalf("computeDroppedCodebases = %v, want empty", dropped)
	}
}

func TestComputeDroppedCodebasesSortsAndDeduplicates(t *testing.T) {
	t.Parallel()

	jobs := []model.Job{
		{CanonicalPath: "/repo/b", State: model.JobStateCompleted},
		{CanonicalPath: "/repo/a", State: model.JobStateCompleted},
		{CanonicalPath: "/repo/a", State: model.JobStateCompleted},
	}

	dropped := computeDroppedCodebases(jobs, nil, func(string) bool { return true })

	want := []string{"/repo/a", "/repo/b"}
	if !slices.Equal(dropped, want) {
		t.Fatalf("computeDroppedCodebases = %v, want %v", dropped, want)
	}
}

func TestRenderDroppedSectionStatesNoneWhenEmpty(t *testing.T) {
	t.Parallel()

	if section := renderDroppedSection(view.DoctorView{}); !strings.Contains(section, "none") {
		t.Fatalf("renderDroppedSection empty = %q, want a none statement", section)
	}
}

// TestDoctorDisplayTextIncludesDroppedSection proves the daemon's Doctor surface
// carries the dropped-codebases section, so the MCP adapter no longer computes it
// client-side. A completed job whose codebase is no longer tracked, while its
// source directory still exists on disk, reads as a dropped codebase.
func TestDoctorDisplayTextIncludesDroppedSection(t *testing.T) {
	t.Parallel()

	manager, _, repoPath := newTestManager(t)
	manager.mu.Lock()
	manager.jobs["job-dropped"] = model.Job{
		ID:            "job-dropped",
		CanonicalPath: repoPath,
		State:         model.JobStateCompleted,
	}
	manager.mu.Unlock()

	server := NewGRPCServer(manager, nil)
	response, err := server.Doctor(context.Background(), &pb.DoctorRequest{})
	if err != nil {
		t.Fatalf("Doctor returned error: %v", err)
	}
	text := response.GetDisplayText()
	if !strings.Contains(text, "Dropped codebases") {
		t.Fatalf("Doctor display_text missing dropped section:\n%s", text)
	}
	if !strings.Contains(text, repoPath) {
		t.Fatalf("Doctor display_text missing dropped path %q:\n%s", repoPath, text)
	}
}

package modeldownload_test

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/modeldownload"
)

func TestEnsureResumesFromPartialFileAfterDroppedConnection(t *testing.T) {
	fixture := newArtifactServer(
		t,
		behaviorDropMidway,
		behaviorDropMidway,
		behaviorDropMidway,
		behaviorDropMidway,
	)
	run := newDownloadRun(t)

	partial := leavePartial(t, run, fixture)
	wantSleeps := []time.Duration{time.Second, 2 * time.Second, 4 * time.Second}
	if !slices.Equal(run.sleeps, wantSleeps) {
		t.Fatalf("sleeps = %v, want %v", run.sleeps, wantSleeps)
	}
	firstCallRanges := fixture.ranges()
	if len(firstCallRanges) != maximumAttempts {
		t.Fatalf("requests = %d, want %d", len(firstCallRanges), maximumAttempts)
	}
	if firstCallRanges[0] != "" {
		t.Fatalf("first Range header = %q, want none", firstCallRanges[0])
	}
	for _, rangeValue := range firstCallRanges[1:] {
		if !strings.HasPrefix(rangeValue, "bytes=") {
			t.Fatalf("retry Range header = %q, want a bytes range", rangeValue)
		}
	}

	run.progress = nil
	if err := modeldownload.Ensure(context.Background(), run.request(fixture, fixture.sha256())); err != nil {
		t.Fatalf("Ensure after partial: %v", err)
	}

	assertInstalled(t, run, fixture)
	assertFinalProgress(t, run)
	ranges := fixture.ranges()
	wantRange := "bytes=" + strconv.Itoa(len(partial)) + "-"
	if len(ranges) != maximumAttempts+1 || ranges[maximumAttempts] != wantRange {
		t.Fatalf("Range headers = %q, want a fifth request with %q", ranges, wantRange)
	}
}

func TestEnsureRestartsFromZeroWhenServerIgnoresRange(t *testing.T) {
	fixture := newArtifactServer(
		t,
		behaviorDropMidway,
		behaviorDropMidway,
		behaviorDropMidway,
		behaviorDropMidway,
		behaviorIgnoreRange,
	)
	run := newDownloadRun(t)
	leavePartial(t, run, fixture)

	run.progress = nil
	if err := modeldownload.Ensure(context.Background(), run.request(fixture, fixture.sha256())); err != nil {
		t.Fatalf("Ensure after partial: %v", err)
	}

	assertInstalled(t, run, fixture)
	assertFinalProgress(t, run)
	ranges := fixture.ranges()
	if len(ranges) != maximumAttempts+1 || ranges[maximumAttempts] == "" {
		t.Fatalf("Range headers = %q, want a fifth request with Range", ranges)
	}
}

func TestEnsureRestartsOnceWhenServerRejectsRange(t *testing.T) {
	fixture := newArtifactServer(t)
	run := newDownloadRun(t)
	if err := os.MkdirAll(filepath.Dir(run.partialPath), 0o700); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	oversized := append(slices.Clone(fixture.content), 1, 2, 3)
	if err := os.WriteFile(run.partialPath, oversized, fixtureFilePermission); err != nil {
		t.Fatalf("WriteFile %s: %v", run.partialPath, err)
	}

	if err := modeldownload.Ensure(context.Background(), run.request(fixture, fixture.sha256())); err != nil {
		t.Fatalf("Ensure: %v", err)
	}

	assertInstalled(t, run, fixture)
	wantRanges := []string{"bytes=" + strconv.Itoa(len(oversized)) + "-", ""}
	if ranges := fixture.ranges(); !slices.Equal(ranges, wantRanges) {
		t.Fatalf("Range headers = %q, want %q", ranges, wantRanges)
	}
	if len(run.sleeps) != 0 {
		t.Fatalf("sleeps = %v, want none", run.sleeps)
	}
}

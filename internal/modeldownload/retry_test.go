package modeldownload_test

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/modeldownload"
)

func TestEnsureRetriesTransientServerError(t *testing.T) {
	fixture := newArtifactServer(t, behaviorServiceUnavailable)
	run := newDownloadRun(t)

	if err := modeldownload.Ensure(context.Background(), run.request(fixture, fixture.sha256())); err != nil {
		t.Fatalf("Ensure: %v", err)
	}

	assertInstalled(t, run, fixture)
	if requestCount := len(fixture.ranges()); requestCount != 2 {
		t.Fatalf("requests = %d, want 2", requestCount)
	}
	if !slices.Equal(run.sleeps, []time.Duration{time.Second}) {
		t.Fatalf("sleeps = %v, want [1s]", run.sleeps)
	}
}

func TestEnsureFailsWithoutRetryOnNotFound(t *testing.T) {
	fixture := newArtifactServer(t, behaviorNotFound)
	run := newDownloadRun(t)

	err := modeldownload.Ensure(context.Background(), run.request(fixture, fixture.sha256()))
	if !errors.Is(err, modeldownload.ErrUnavailable) {
		t.Fatalf("Ensure err = %v, want ErrUnavailable", err)
	}
	if requestCount := len(fixture.ranges()); requestCount != 1 {
		t.Fatalf("requests = %d, want 1", requestCount)
	}
	if len(run.sleeps) != 0 {
		t.Fatalf("sleeps = %v, want none", run.sleeps)
	}
	assertAbsent(t, run.destinationPath)
}

func TestEnsureStopsWhenContextIsCancelledDuringRetryDelay(t *testing.T) {
	fixture := newArtifactServer(t, behaviorServiceUnavailable)
	run := newDownloadRun(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	request := run.request(fixture, fixture.sha256())
	request.Sleep = func(_ context.Context, delay time.Duration) {
		run.sleeps = append(run.sleeps, delay)
		cancel()
	}

	err := modeldownload.Ensure(ctx, request)
	if !errors.Is(err, modeldownload.ErrUnavailable) {
		t.Fatalf("Ensure err = %v, want ErrUnavailable", err)
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Ensure err = %v, want context.Canceled", err)
	}
	if requestCount := len(fixture.ranges()); requestCount != 1 {
		t.Fatalf("requests = %d, want 1", requestCount)
	}
	if !slices.Equal(run.sleeps, []time.Duration{time.Second}) {
		t.Fatalf("sleeps = %v, want [1s]", run.sleeps)
	}
	assertAbsent(t, run.partialPath)
	assertAbsent(t, run.destinationPath)
}

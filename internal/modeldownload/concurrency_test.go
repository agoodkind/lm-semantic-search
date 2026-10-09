package modeldownload_test

import (
	"context"
	"errors"
	"testing"

	"goodkind.io/lm-semantic-search/internal/modeldownload"
)

func TestEnsureSerializesConcurrentCallsForOneDestination(t *testing.T) {
	fixture := newArtifactServer(t)
	run := newDownloadRun(t)
	request := run.request(fixture, fixture.sha256())
	request.Progress = nil
	request.Sleep = nil

	results := make(chan error, concurrentCalls)
	for range concurrentCalls {
		go func() {
			results <- modeldownload.Ensure(context.Background(), request)
		}()
	}
	for range concurrentCalls {
		if err := <-results; err != nil {
			t.Fatalf("Ensure: %v", err)
		}
	}

	assertInstalled(t, run, fixture)
	if requestCount := len(fixture.ranges()); requestCount != 1 {
		t.Fatalf("requests = %d, want 1", requestCount)
	}
}

func TestEnsureStopsOnCancellationAndLeavesPartialFile(t *testing.T) {
	fixture := newArtifactServer(t, behaviorStallMidway)
	run := newDownloadRun(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	request := run.request(fixture, fixture.sha256())
	request.Progress = func(string, int64, int64) {
		cancel()
	}

	err := modeldownload.Ensure(ctx, request)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Ensure err = %v, want context.Canceled", err)
	}
	readPartial(t, run, fixture)
	if requestCount := len(fixture.ranges()); requestCount != 1 {
		t.Fatalf("requests = %d, want 1", requestCount)
	}
	if len(run.sleeps) != 0 {
		t.Fatalf("sleeps = %v, want none", run.sleeps)
	}
}

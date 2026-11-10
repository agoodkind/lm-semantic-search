//go:build offlinelive

package offlinelive

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"google.golang.org/protobuf/proto"
)

func TestCancelPausesAutomaticIndexingAndKeepsSearchAvailable(t *testing.T) {
	harness := newUnstartedHarness(t)
	fixture := t.TempDir()
	if err := os.CopyFS(fixture, os.DirFS(harness.fixturePath)); err != nil {
		t.Fatalf("copy fixture: %v", err)
	}
	harness.fixturePath = fixture
	options := harnessOptions{fileWatcher: true, backgroundSync: true, syncIntervalMS: 1000, resumeOnBoot: true}
	harness.start(options, true)
	t.Cleanup(harness.teardown)
	requireCompleted(t, harness.indexFixture())
	response, err := harness.client.StartIndex(correlatedContext(), &pb.StartIndexRequest{
		Path: fixture, Force: true, Client: harnessClientInfo(),
		SchedulingPolicy: &pb.SchedulingPolicyPatch{Quiet: proto.Bool(true), IdleAfterSeconds: proto.Int32(86400)},
	})
	if err != nil {
		t.Fatalf("queue quiet index: %v", err)
	}
	canceled, err := harness.client.CancelJob(correlatedContext(), &pb.CancelJobRequest{JobId: response.GetJobId()})
	if err != nil || !canceled.GetCancelled() {
		t.Fatalf("cancel index: response=%v err=%v", canceled, err)
	}
	if err := os.WriteFile(filepath.Join(fixture, "changed.go"), []byte("package fixture\nfunc afterCancellation() string { return \"paused change\" }\n"), 0o644); err != nil {
		t.Fatalf("write changed file: %v", err)
	}
	assertAutomaticIndexingRemainsPaused(t, harness, response.GetCodebaseId(), 7*time.Second)
	harness.restart(options)
	assertAutomaticIndexingRemainsPaused(t, harness, response.GetCodebaseId(), 6*time.Second)
	search := harness.search(fixtureQuery, searchResultLimit)
	if !containsTargetResult(search.GetResults(), targetRelativePath, targetFunctionName) {
		t.Fatal("cancellation prevented search of the completed index")
	}
	resumed, err := harness.client.SyncIndex(correlatedContext(), &pb.SyncIndexRequest{
		Path: fixture, Client: harnessClientInfo(),
		SchedulingPolicy: &pb.SchedulingPolicyPatch{Quiet: proto.Bool(false)},
	})
	if err != nil {
		t.Fatalf("resume indexing: %v", err)
	}
	requireCompleted(t, harness.waitForJob(resumed.GetJobId()))
	if harness.indexStatus().GetCodebase().GetAutoIndexingPaused() {
		t.Fatal("explicit sync did not resume automatic indexing")
	}
}

func assertAutomaticIndexingRemainsPaused(t *testing.T, harness *harness, codebaseID string, duration time.Duration) {
	t.Helper()
	jobs, err := harness.client.ListJobs(correlatedContext(), &pb.ListJobsRequest{CodebaseId: codebaseID})
	if err != nil {
		t.Fatalf("list canceled codebase jobs: %v", err)
	}
	jobCount := len(jobs.GetJobs())
	deadline := time.NewTimer(duration)
	defer deadline.Stop()
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		if !harness.indexStatus().GetCodebase().GetAutoIndexingPaused() {
			t.Fatal("codebase did not retain its automatic indexing pause")
		}
		current, err := harness.client.ListJobs(correlatedContext(), &pb.ListJobsRequest{CodebaseId: codebaseID})
		if err != nil {
			t.Fatalf("list jobs while paused: %v", err)
		}
		if len(current.GetJobs()) != jobCount {
			t.Fatalf("automatic indexing created replacement jobs: before=%d after=%d", jobCount, len(current.GetJobs()))
		}
		select {
		case <-deadline.C:
			return
		case <-ticker.C:
		}
	}
}

//go:build live

package live

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/model"
)

const residencyLiveIdleTimeout = 500 * time.Millisecond

func TestPublicCodeSearchLoadsAnIdleCollection(t *testing.T) {
	harness := newResidencyHarness(t, residencyLiveIdleTimeout)
	codebasePath := filepath.Join(harness.stateRoot, "idle-code-"+randomID())
	if err := os.MkdirAll(codebasePath, 0o755); err != nil {
		t.Fatalf("create live codebase: %v", err)
	}
	if err := os.WriteFile(
		filepath.Join(codebasePath, "residency.go"),
		[]byte("package residency\n\nfunc IdleSearchSentinel() string {\n\treturn \"public idle collection residency sentinel\"\n}\n"),
		0o644,
	); err != nil {
		t.Fatalf("write live codebase fixture: %v", err)
	}
	collectionName := harness.trackCodebasePath(codebasePath)

	startResponse, err := harness.client.StartIndex(
		correlatedContext(),
		&pb.StartIndexRequest{
			Path: codebasePath,
			Splitter: &pb.SplitterConfig{
				Type: "ast",
			},
			Client: &pb.ClientInfo{Name: "residency-live-harness"},
		},
	)
	if err != nil {
		t.Fatalf("start public live index: %v", err)
	}
	job := waitForPublicJob(t, harness, startResponse.GetJobId())
	if job.GetState() != string(model.JobStateCompleted) {
		t.Fatalf("public live index state = %q, want completed: %s", job.GetState(), job.GetError().GetMessage())
	}

	statusResponse, err := harness.client.GetIndex(
		correlatedContext(),
		&pb.GetIndexRequest{
			Path:   codebasePath,
			Client: &pb.ClientInfo{Name: "residency-live-harness"},
		},
	)
	if err != nil {
		t.Fatalf("get public live index: %v", err)
	}
	if got := statusResponse.GetCodebase().GetCollectionName(); got != collectionName {
		t.Fatalf("public collection name = %q, want %q", got, collectionName)
	}
	waitForLoadState(t, harness, collectionName, entity.LoadStateNotLoad)

	searchResponse, err := harness.client.SearchCode(
		correlatedContext(),
		&pb.SearchCodeRequest{
			Path:   codebasePath,
			Query:  "public idle collection residency sentinel",
			Limit:  5,
			Client: &pb.ClientInfo{Name: "residency-live-harness"},
		},
	)
	if err != nil {
		t.Fatalf("public code search from idle: %v", err)
	}
	if len(searchResponse.GetResults()) == 0 {
		t.Fatalf("public code search from idle returned no results: %s", searchResponse.GetDisplayText())
	}
}

func waitForPublicJob(t *testing.T, harness *harness, jobID string) *pb.Job {
	t.Helper()
	if jobID == "" {
		t.Fatal("public live index returned an empty job id")
	}
	deadline := time.Now().Add(jobPollTimeout)
	for time.Now().Before(deadline) {
		response, err := harness.client.GetJob(
			correlatedContext(),
			&pb.GetJobRequest{JobId: jobID},
		)
		if err != nil {
			t.Fatalf("get public live job %s: %v", jobID, err)
		}
		job := response.GetJob()
		if job != nil {
			switch job.GetState() {
			case string(model.JobStateCompleted),
				string(model.JobStateFailed),
				string(model.JobStateCancelled):
				return job
			}
		}
		time.Sleep(jobPollInterval)
	}
	t.Fatalf("public live job %s did not finish within %s", jobID, jobPollTimeout)
	return nil
}

func waitForLoadState(
	t *testing.T,
	harness *harness,
	collectionName string,
	want entity.LoadStateCode,
) {
	t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		state, err := harness.milvus.GetLoadState(
			context.Background(),
			milvusclient.NewGetLoadStateOption(collectionName),
		)
		if err != nil {
			t.Fatalf("get load state for %s: %v", collectionName, err)
		}
		if state.State == want {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("collection %s did not reach load state %v", collectionName, want)
}

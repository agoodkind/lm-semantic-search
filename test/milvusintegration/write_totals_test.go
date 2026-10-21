//go:build milvusintegration

package milvusintegration

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/metrics"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func TestMilvusJobTotalsCountProviderCallsAndReuse(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, "4g")
	_ = dialMilvus(t, stack)
	cfg := integrationConfig(t, stack, map[string]string{"EMBEDDING_BATCH_SIZE": "2"})
	manager, err := daemon.NewManager(context.Background(), cfg)
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err := manager.Close(ctx); err != nil {
			t.Errorf("Close manager: %v", err)
		}
	})
	server := daemon.NewGRPCServer(manager, nil)
	directory := t.TempDir()
	var content strings.Builder
	content.WriteString("package example\n\n")
	for index := range 40 {
		fmt.Fprintf(&content, "func Value%d() string { return %q }\n", index, strings.Repeat("sample ", 30)+fmt.Sprint(index))
	}
	source := []byte(content.String())
	if err := os.WriteFile(filepath.Join(directory, "values.go"), source, 0o644); err != nil {
		t.Fatalf("Write initial file: %v", err)
	}
	before := metrics.Read()
	started, err := server.StartIndex(context.Background(), &pb.StartIndexRequest{Path: directory, Client: &pb.ClientInfo{Name: "integration"}})
	if err != nil {
		t.Fatalf("StartIndex: %v", err)
	}
	initial := waitForWriteTotalsJob(t, manager, started.GetJobId())
	after := metrics.Read()
	providerCalls := after.EmbedBatchesTotal - before.EmbedBatchesTotal
	if providerCalls <= 1 {
		t.Fatalf("The real ONNX provider completed %d batches, want more than 1", providerCalls)
	}
	rows := int64(initial.Progress.ChunksProcessed)
	if initial.Progress.WriteTotals == nil || initial.Progress.WriteTotals.EmbeddingBatchesCompleted != providerCalls || initial.Progress.WriteTotals.CollectionRowsWritten != rows {
		t.Fatalf("Initial Milvus write totals = %+v, want %d rows and %d provider calls", initial.Progress.WriteTotals, rows, providerCalls)
	}
	if err := os.WriteFile(filepath.Join(directory, "copied.go"), source, 0o644); err != nil {
		t.Fatalf("Write copied file: %v", err)
	}
	before = metrics.Read()
	startedSync, err := server.SyncIndex(context.Background(), &pb.SyncIndexRequest{Path: directory, Client: &pb.ClientInfo{Name: "integration"}})
	if err != nil {
		t.Fatalf("SyncIndex: %v", err)
	}
	reused := waitForWriteTotalsJob(t, manager, startedSync.GetJobId())
	after = metrics.Read()
	if calls := after.EmbedBatchesTotal - before.EmbedBatchesTotal; calls != 0 {
		t.Fatalf("The reuse job made %d provider calls, want 0", calls)
	}
	if reused.Progress.ChunksReused == 0 || reused.Progress.EmbeddingBatchesCompleted == 0 {
		t.Fatalf("The test must complete a reused Milvus pack: %+v", reused.Progress)
	}
	if reused.Progress.WriteTotals == nil || reused.Progress.WriteTotals.EmbeddingBatchesCompleted != 0 || reused.Progress.WriteTotals.CollectionRowsWritten != rows {
		t.Fatalf("Reused Milvus write totals = %+v, want %d rows and 0 provider calls", reused.Progress.WriteTotals, rows)
	}
	t.Logf("The initial Milvus job wrote %d rows in %d real provider calls. The reuse job wrote %d rows in 0 provider calls while completing %d legacy packs.", rows, providerCalls, rows, reused.Progress.EmbeddingBatchesCompleted)
}

func TestMilvusFailedWriteRetainsCompletedProviderCalls(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStackWithMilvusConfig(t, "4g", readMilvusTestConfig(t, "write-limit.yaml"))
	_ = dialMilvus(t, stack)
	service := newService(t, integrationConfig(t, stack, nil), &milvusCallRecorder{})
	var failedProgress semantic.Progress
	before := metrics.Read()
	writeErr := service.StageReindex(
		context.Background(),
		filepath.Join(t.TempDir(), "limited-write"),
		[]model.StoredChunk{{Content: "valid content for an acknowledged embedding call", RelativePath: "sample.go", StartLine: 1, EndLine: 1}},
		semantic.Removal{},
		func(progress semantic.Progress) { failedProgress = progress },
		nil,
		semantic.CodeColumns(),
	)
	after := metrics.Read()
	if writeErr == nil || !strings.Contains(strings.ToLower(writeErr.Error()), "rate limit") {
		t.Fatalf("Milvus must reject the insert at its configured rate limit: %v", writeErr)
	}
	if calls := after.EmbedBatchesTotal - before.EmbedBatchesTotal; calls != 1 || failedProgress.EmbeddingCallsCompleted != 1 || failedProgress.CollectionRowsWritten != 0 {
		t.Fatalf("The failed insert must preserve 1 completed provider call and 0 acknowledged rows: calls=%d progress=%+v", calls, failedProgress)
	}
	t.Logf("Milvus rejected the insert after 1 provider call and reported 0 acknowledged rows: %v", writeErr)
}

func waitForWriteTotalsJob(t *testing.T, manager *daemon.Manager, jobID string) model.Job {
	t.Helper()
	deadline := time.Now().Add(5 * time.Minute)
	for {
		job, found := manager.GetJob(jobID)
		if found {
			switch job.State {
			case model.JobStateCompleted:
				return job
			case model.JobStateFailed, model.JobStateCancelled:
				t.Fatalf("Job %s ended in %s: %+v", jobID, job.State, job.Error)
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("Job %s did not complete", jobID)
		}
		time.Sleep(25 * time.Millisecond)
	}
}

func readMilvusTestConfig(t *testing.T, name string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("Read Milvus configuration %s: %v", name, err)
	}
	return string(data)
}

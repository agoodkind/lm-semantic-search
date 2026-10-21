//go:build milvusintegration

package milvusintegration

import (
	"context"
	"errors"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/grpc"
)

const catalogDaemonPollInterval = 25 * time.Millisecond

func TestDaemonAutomaticallyRecoversAfterCatalogRelease(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, catalogRecoveryMemory)
	milvus := dialMilvus(t, stack)
	cfg := integrationConfig(t, stack, map[string]string{
		"CLAUDE_CONTEXT_BACKGROUND_SYNC":  "true",
		"CLAUDE_CONTEXT_SYNC_INTERVAL_MS": "1000",
	})
	manager, err := daemon.NewManager(context.Background(), cfg)
	if err != nil {
		t.Fatalf("create daemon manager: %v", err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err := manager.Close(ctx); err != nil {
			t.Errorf("close daemon manager: %v", err)
		}
	})
	client := catalogDaemonClient(t, manager, cfg.SocketPath)
	ctx, cancel := context.WithTimeout(context.Background(), catalogRecoveryTimeout)
	defer cancel()
	seedPath := t.TempDir()
	for _, name := range []string{"first.go", "second.go"} {
		if err := os.WriteFile(filepath.Join(seedPath, name), []byte(catalogRecoveryContent), 0o644); err != nil {
			t.Fatalf("write seed source: %v", err)
		}
	}
	seed, err := client.StartIndex(ctx, &pb.StartIndexRequest{Path: seedPath})
	if err != nil {
		t.Fatalf("index catalog seed: %v", err)
	}
	seedJob := waitForCatalogDaemonJob(t, ctx, client, seed.GetJobId())
	if seedJob.GetState() != "completed" || seedJob.GetProgress().GetChunksReused() == 0 {
		t.Fatalf("seed job = %v, want completed indexing with a catalog reuse read", seedJob)
	}
	catalogName := semantic.ReuseCatalogCollectionName(cfg)
	if err := milvus.ReleaseCollection(ctx, milvusclient.NewReleaseCollectionOption(catalogName)); err != nil {
		t.Fatalf("release catalog after the daemon cached readiness: %v", err)
	}
	waitForLoadState(t, milvus, catalogName, entity.LoadStateNotLoad)
	targetPath := t.TempDir()
	if err := os.WriteFile(filepath.Join(targetPath, "catalog.go"), []byte(catalogRecoveryContent), 0o644); err != nil {
		t.Fatalf("write target source: %v", err)
	}
	started, err := client.StartIndex(ctx, &pb.StartIndexRequest{Path: targetPath})
	if err != nil {
		t.Fatalf("start indexing against the released catalog: %v", err)
	}
	failed := waitForCatalogDaemonJob(t, ctx, client, started.GetJobId())
	if failed.GetState() != "failed" || failed.GetError().GetCode() != "milvus_unavailable" || !failed.GetError().GetRetryable() {
		t.Fatalf("released-catalog job = %v, want a retryable Milvus failure", failed)
	}
	waitForCatalogRepairState(t, ctx, client, targetPath)
	for _, suffix := range []string{".json", ".staging.json"} {
		checkpointPath := filepath.Join(cfg.MerkleDir, started.GetCodebaseId()+suffix)
		if _, err := os.Stat(checkpointPath); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("checkpoint %s: %v, want absent after the catalog read failed", checkpointPath, err)
		}
	}
	t.Logf("job %s reported retryable milvus_unavailable and left its codebase resumable without a successful run or checkpoint", failed.GetId())

	backgroundContext, stopBackground := context.WithCancel(ctx)
	defer stopBackground()
	daemon.NewBackgroundSync(cfg, manager).Start(backgroundContext)
	recovered := waitForCatalogAutomaticJob(t, ctx, client, started.GetCodebaseId(), failed.GetId())
	status, err := client.GetIndex(ctx, &pb.GetIndexRequest{Path: targetPath})
	if err != nil {
		t.Fatalf("read automatically recovered index: %v", err)
	}
	if !status.GetSearchable() || status.GetCodebase().GetLastSuccessfulRun().GetTotalChunks() != 1 {
		t.Fatalf("recovered index = %v, want one searchable chunk", status)
	}
	if recovered.GetClient().GetName() != "daemon-repair" || recovered.GetProgress().GetChunksReused() != 1 || recovered.GetProgress().GetChunksEmbedded() != 0 {
		t.Fatalf("recovered job = %v, want the automatic repair to reuse the stored vector", recovered)
	}
	t.Logf("the production background sweep completed successor job %s as %s with one reused vector and a searchable index without another StartIndex request or daemon restart", recovered.GetId(), recovered.GetClient().GetName())
}

func catalogDaemonClient(t *testing.T, manager *daemon.Manager, socketPath string) pb.SemanticSearchDaemonServiceClient {
	t.Helper()
	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatalf("listen on daemon socket: %v", err)
	}
	server := grpc.NewServer()
	pb.RegisterSemanticSearchDaemonServiceServer(server, daemon.NewGRPCServer(manager, nil))
	go func() {
		if err := server.Serve(listener); err != nil {
			t.Errorf("serve daemon gRPC: %v", err)
		}
	}()
	connection, client, err := grpcutil.DialDaemon(context.Background(), socketPath)
	if err != nil {
		t.Fatalf("dial daemon gRPC: %v", err)
	}
	t.Cleanup(func() {
		if err := connection.Close(); err != nil {
			t.Errorf("close daemon client: %v", err)
		}
		server.GracefulStop()
	})
	return client
}

func waitForCatalogDaemonJob(t *testing.T, ctx context.Context, client pb.SemanticSearchDaemonServiceClient, jobID string) *pb.Job {
	t.Helper()
	ticker := time.NewTicker(catalogDaemonPollInterval)
	defer ticker.Stop()
	for {
		response, err := client.GetJob(ctx, &pb.GetJobRequest{JobId: jobID})
		if err != nil {
			t.Fatalf("read indexing job: %v", err)
		}
		job := response.GetJob()
		if job.GetState() == "completed" || job.GetState() == "failed" || job.GetState() == "cancelled" {
			return job
		}
		select {
		case <-ctx.Done():
			t.Fatalf("job %s did not finish: %v", jobID, ctx.Err())
		case <-ticker.C:
		}
	}
}

func waitForCatalogRepairState(t *testing.T, ctx context.Context, client pb.SemanticSearchDaemonServiceClient, path string) {
	t.Helper()
	ticker := time.NewTicker(catalogDaemonPollInterval)
	defer ticker.Stop()
	for {
		status, err := client.GetIndex(ctx, &pb.GetIndexRequest{Path: path})
		if err != nil {
			t.Fatalf("read resumable codebase: %v", err)
		}
		if status.GetCodebase().GetActiveJobId() == "" {
			if status.GetCodebase().GetStatus() != "indexing" || status.GetCodebase().GetLastSuccessfulRun() != nil || status.GetCodebase().GetLastFailedRun() != nil {
				t.Fatalf("codebase after catalog failure = %v, want resumable indexing", status)
			}
			return
		}
		select {
		case <-ctx.Done():
			t.Fatalf("codebase retained failed job ownership: %v", ctx.Err())
		case <-ticker.C:
		}
	}
}

func waitForCatalogAutomaticJob(t *testing.T, ctx context.Context, client pb.SemanticSearchDaemonServiceClient, codebaseID, failedJobID string) *pb.Job {
	t.Helper()
	ticker := time.NewTicker(catalogDaemonPollInterval)
	defer ticker.Stop()
	for {
		response, err := client.ListJobs(ctx, &pb.ListJobsRequest{CodebaseId: codebaseID})
		if err != nil {
			t.Fatalf("read automatic indexing jobs: %v", err)
		}
		for _, job := range response.GetJobs() {
			if job.GetId() == failedJobID {
				continue
			}
			if job.GetState() == "completed" {
				return job
			}
			if job.GetState() == "failed" || job.GetState() == "cancelled" {
				t.Fatalf("automatic repair job = %v, want completed", job)
			}
		}
		select {
		case <-ctx.Done():
			t.Fatalf("the background sweep did not complete an automatic repair: %v", ctx.Err())
		case <-ticker.C:
		}
	}
}

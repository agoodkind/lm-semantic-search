package daemon_test

import (
	"bytes"
	"context"
	"errors"
	"net"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/sandbox"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"
)

func TestConfiguredUnavailableStoreCannotCompleteIndexOrSync(t *testing.T) {
	for _, variable := range sandbox.Env(t.TempDir()) {
		t.Setenv(variable.Name, variable.Value)
	}
	t.Setenv("CLAUDE_CONTEXT_PROFILE", config.ProfileStandard)
	t.Setenv("MILVUS_ADDRESS", "")
	t.Setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
	t.Setenv("OPENAI_API_KEY", "local-test-placeholder")
	cfg, err := config.Default()
	if err != nil {
		t.Fatalf("load sandbox config: %v", err)
	}
	for _, directory := range sandbox.Directories(cfg) {
		if err := os.MkdirAll(directory, 0o755); err != nil {
			t.Fatalf("create sandbox directory: %v", err)
		}
	}
	seedPath := writeUnavailableStoreSource(t, "before")
	client, stop := startUnavailableStoreDaemon(t, cfg)
	seed, err := client.StartIndex(context.Background(), &pb.StartIndexRequest{Path: seedPath})
	if err != nil {
		t.Fatalf("start unconfigured metadata index: %v", err)
	}
	if job := waitForUnavailableStoreJob(t, client, seed.GetJobId()); job.GetState() != "completed" {
		t.Fatalf("unconfigured metadata job = %v, want completed", job)
	}
	seedStatus, err := client.GetIndex(context.Background(), &pb.GetIndexRequest{Path: seedPath})
	if err != nil {
		t.Fatalf("read seed status: %v", err)
	}
	snapshotPath := seedStatus.GetCodebase().GetMerkleSnapshotPath()
	checkpoint, err := os.ReadFile(snapshotPath)
	if err != nil {
		t.Fatalf("read seed checkpoint: %v", err)
	}
	stop()

	probe, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("allocate unavailable store address: %v", err)
	}
	cfg.MilvusAddress = probe.Addr().String()
	if err := probe.Close(); err != nil {
		t.Fatalf("close unavailable store endpoint: %v", err)
	}
	client, _ = startUnavailableStoreDaemon(t, cfg)

	t.Run("bootstrap", func(t *testing.T) {
		newPath := writeUnavailableStoreSource(t, "new")
		started, err := client.StartIndex(context.Background(), &pb.StartIndexRequest{Path: newPath})
		if err != nil {
			t.Fatalf("start bootstrap with unavailable store: %v", err)
		}
		assertUnavailableStoreJob(t, waitForUnavailableStoreJob(t, client, started.GetJobId()))
		status, err := client.GetIndex(context.Background(), &pb.GetIndexRequest{Path: newPath})
		if err != nil {
			t.Fatalf("read bootstrap status: %v", err)
		}
		if status.GetSearchable() || status.GetCodebase().GetLastSuccessfulRun() != nil || status.GetCodebase().GetStatus() != "pending" {
			t.Fatalf("bootstrap status = %v, want pending without a successful run", status)
		}
		for _, suffix := range []string{".json", ".staging.json"} {
			path := filepath.Join(cfg.MerkleDir, started.GetCodebaseId()+suffix)
			if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("bootstrap checkpoint %s: %v, want absent", path, err)
			}
		}
	})

	t.Run("delta", func(t *testing.T) {
		if err := os.WriteFile(filepath.Join(seedPath, "main.go"), unavailableStoreSource("after"), 0o644); err != nil {
			t.Fatalf("edit source before sync: %v", err)
		}
		started, err := client.SyncIndex(context.Background(), &pb.SyncIndexRequest{Path: seedPath})
		if err != nil {
			t.Fatalf("start delta with unavailable store: %v", err)
		}
		assertUnavailableStoreJob(t, waitForUnavailableStoreJob(t, client, started.GetJobId()))
		after, err := os.ReadFile(snapshotPath)
		if err != nil {
			t.Fatalf("read checkpoint after failed sync: %v", err)
		}
		if !bytes.Equal(after, checkpoint) {
			t.Fatal("sync advanced the checkpoint without writing semantic data")
		}
		status, err := client.GetIndex(context.Background(), &pb.GetIndexRequest{Path: seedPath})
		if err != nil {
			t.Fatalf("read delta status: %v", err)
		}
		if status.GetSearchable() || !proto.Equal(status.GetCodebase().GetLastSuccessfulRun(), seedStatus.GetCodebase().GetLastSuccessfulRun()) {
			t.Fatalf("delta status = %v, want the previous successful run", status)
		}
		if status.GetCodebase().GetLastFailedRun() != nil || status.GetCodebase().GetStatus() != "indexing" {
			t.Fatalf("delta status = %v, want resumable indexing without a terminal codebase failure", status)
		}
	})
}

func assertUnavailableStoreJob(t *testing.T, job *pb.Job) {
	t.Helper()
	if job.GetState() != "failed" || job.GetError().GetCode() != "milvus_unavailable" || !job.GetError().GetRetryable() {
		t.Fatalf("job = %v, want retryable Milvus failure", job)
	}
	if job.GetProgress().GetFilesProcessed() != 0 || job.GetProgress().GetChunksProcessed() != 0 {
		t.Fatalf("job progress = %v, want unchanged file and chunk progress", job.GetProgress())
	}
	t.Logf("job=%s state=%s code=%s retryable=%t files=%d chunks=%d", job.GetId(), job.GetState(), job.GetError().GetCode(), job.GetError().GetRetryable(), job.GetProgress().GetFilesProcessed(), job.GetProgress().GetChunksProcessed())
}

func waitForUnavailableStoreJob(t *testing.T, client pb.SemanticSearchDaemonServiceClient, jobID string) *pb.Job {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	ticker := time.NewTicker(10 * time.Millisecond)
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

func startUnavailableStoreDaemon(t *testing.T, cfg config.Config) (pb.SemanticSearchDaemonServiceClient, func()) {
	t.Helper()
	manager, err := daemon.NewManager(context.Background(), cfg)
	if err != nil {
		t.Fatalf("create sandbox manager: %v", err)
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen for sandbox gRPC: %v", err)
	}
	server := grpc.NewServer()
	pb.RegisterSemanticSearchDaemonServiceServer(server, daemon.NewGRPCServer(manager, nil))
	go func() {
		if err := server.Serve(listener); err != nil {
			t.Errorf("serve sandbox gRPC: %v", err)
		}
	}()
	connection, err := grpc.NewClient(listener.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("create sandbox gRPC client: %v", err)
	}
	var once sync.Once
	stop := func() {
		once.Do(func() {
			if err := connection.Close(); err != nil {
				t.Errorf("close sandbox client: %v", err)
			}
			server.GracefulStop()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if err := manager.Close(ctx); err != nil {
				t.Errorf("close sandbox manager: %v", err)
			}
		})
	}
	t.Cleanup(stop)
	return pb.NewSemanticSearchDaemonServiceClient(connection), stop
}

func writeUnavailableStoreSource(t *testing.T, version string) string {
	t.Helper()
	path := t.TempDir()
	if err := os.WriteFile(filepath.Join(path, "main.go"), unavailableStoreSource(version), 0o644); err != nil {
		t.Fatalf("write source file: %v", err)
	}
	return path
}

func unavailableStoreSource(version string) []byte {
	return []byte("package main\n\nfunc Main() string { return \"" + version + "\" }\n")
}

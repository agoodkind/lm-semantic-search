package daemon_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/store"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

func TestCancelJobPauseSurvivesRestartAndExplicitIndexResumes(t *testing.T) {
	manager, server, cfg, repoPath := newCancellationManager(t)
	job := queueCancellationIndex(t, server, repoPath)
	if _, err := server.CancelJob(
		context.Background(),
		&pb.CancelJobRequest{JobId: job.GetJobId()},
	); err != nil {
		t.Fatalf("CancelJob: %v", err)
	}

	registry, err := store.ReadRegistry(cfg.RegistryPath)
	if err != nil {
		t.Fatalf("ReadRegistry: %v", err)
	}

	if len(registry.Codebases) != 1 || !registry.Codebases[0].AutoIndexingPaused {
		t.Fatal("explicit cancellation did not persist the automatic indexing pause")
	}

	if err := manager.Close(context.Background()); err != nil {
		t.Fatalf("close manager: %v", err)
	}

	restarted, err := daemon.NewManager(context.Background(), cfg)
	if err != nil {
		t.Fatalf("restart manager: %v", err)
	}

	t.Cleanup(func() { _ = restarted.Close(context.Background()) })
	server = daemon.NewGRPCServer(restarted, nil)
	assertCancellationPause(t, server, repoPath, true)
	queueCancellationIndex(t, server, repoPath)
	assertCancellationPause(t, server, repoPath, false)
}

func TestInternalCancellationDoesNotPauseAutomaticIndexing(t *testing.T) {
	manager, server, cfg, repoPath := newCancellationManager(t)
	queueCancellationIndex(t, server, repoPath)
	if err := manager.Close(context.Background()); err != nil {
		t.Fatalf("close manager: %v", err)
	}

	registry, err := store.ReadRegistry(cfg.RegistryPath)
	if err != nil {
		t.Fatalf("ReadRegistry: %v", err)
	}

	if len(registry.Codebases) != 1 || registry.Codebases[0].AutoIndexingPaused {
		t.Fatal("shutdown paused automatic indexing")
	}
}

func assertCancellationPause(t *testing.T, server *daemon.GRPCServer, path string, paused bool) {
	t.Helper()
	response, err := server.GetIndex(context.Background(), &pb.GetIndexRequest{Path: path})
	if err != nil {
		t.Fatalf("GetIndex: %v", err)
	}

	if response.GetCodebase().GetAutoIndexingPaused() != paused {
		t.Fatal("explicit index did not resume automatic indexing")
	}
}

func TestCancelJobReportsPausePersistenceFailure(t *testing.T) {
	_, server, cfg, repoPath := newCancellationManager(t)
	job := queueCancellationIndex(t, server, repoPath)
	backupPath := cfg.RegistryPath + ".saved"
	if err := os.Rename(cfg.RegistryPath, backupPath); err != nil {
		t.Fatalf("move registry: %v", err)
	}

	if err := os.Mkdir(cfg.RegistryPath, 0o755); err != nil {
		t.Fatalf("block registry writes: %v", err)
	}

	t.Cleanup(func() {
		if err := os.Remove(cfg.RegistryPath); err != nil {
			t.Errorf("remove registry obstruction: %v", err)
		}

		if err := os.Rename(backupPath, cfg.RegistryPath); err != nil {
			t.Errorf("restore registry: %v", err)
		}
	})
	_, err := server.CancelJob(context.Background(), &pb.CancelJobRequest{JobId: job.GetJobId()})
	if status.Code(err) != codes.Internal {
		t.Fatalf("CancelJob did not report the persistence failure: %v", err)
	}

	assertCancellationPause(t, server, repoPath, false)
}

func newCancellationManager(
	t *testing.T,
) (*daemon.Manager, *daemon.GRPCServer, config.Config, string) {
	t.Helper()
	cfg := newSourceTestConfig(t)
	path := t.TempDir()
	if err := os.WriteFile(
		filepath.Join(path, "main.go"),
		[]byte("package main\n"),
		0o644,
	); err != nil {
		t.Fatalf("write source: %v", err)
	}

	manager, err := daemon.NewManager(context.Background(), cfg)
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}

	t.Cleanup(func() { _ = manager.Close(context.Background()) })

	return manager, daemon.NewGRPCServer(manager, nil), cfg, path
}

func queueCancellationIndex(
	t *testing.T,
	server *daemon.GRPCServer,
	path string,
) *pb.StartIndexResponse {
	t.Helper()
	response, err := server.StartIndex(
		context.Background(),
		&pb.StartIndexRequest{
			Path:  path,
			Force: true,
			SchedulingPolicy: &pb.SchedulingPolicyPatch{
				Quiet:            proto.Bool(true),
				IdleAfterSeconds: proto.Int32(86400),
			},
		},
	)
	if err != nil {
		t.Fatalf("StartIndex: %v", err)
	}

	return response
}

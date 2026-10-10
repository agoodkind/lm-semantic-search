package daemon_test

import (
	"context"
	"net"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/networkcost"
	"goodkind.io/lm-semantic-search/internal/sandbox"
	"google.golang.org/grpc"
)

const (
	testModelArtifactName = "model.onnx"
	testTokenizerName     = "tokenizer.json"
	testModelBytes        = 256 * 1024
	testDownloadTimeout   = 30 * time.Second
	testPollInterval      = 20 * time.Millisecond
	testAttemptBackoff    = 10 * time.Millisecond

	stateDownloading = "downloading"
	stateDeferred    = "deferred"
	stateFailed      = "failed"
	stateComplete    = "complete"

	jobStateQueued        = "queued"
	jobPhaseModelDownload = "model_download"
)

type modelDownloadHarness struct {
	config config.Config
	client pb.SemanticSearchDaemonServiceClient
}

func offlineDownloadTestConfig(t *testing.T) config.Config {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	root, err := os.MkdirTemp("", "lms-dl-root")
	if err != nil {
		t.Fatalf("MkdirTemp returned error: %v", err)
	}
	t.Cleanup(func() {
		if removeErr := os.RemoveAll(root); removeErr != nil {
			t.Errorf("remove daemon root %s: %v", root, removeErr)
		}
	})
	socketDirectory, err := os.MkdirTemp("", "lms-dl")
	if err != nil {
		t.Fatalf("MkdirTemp returned error: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(socketDirectory) })
	t.Setenv("CLAUDE_CONTEXTD_SOCKET_PATH", filepath.Join(socketDirectory, "d.sock"))
	t.Setenv("CLAUDE_CONTEXTD_MODEL_CACHE_ROOT", filepath.Join(root, "models"))
	t.Setenv("CLAUDE_CONTEXT_MODEL_DOWNLOAD_NETWORK_POLICY", "")
	t.Setenv("CLAUDE_CONTEXT_MODEL_DOWNLOAD_NETWORK_OVERRIDE", "")
	t.Setenv("CLAUDE_CONTEXT_PROFILE", config.ProfileOffline)
	t.Setenv("OFFLINE_EMBEDDING_MODEL", "")
	t.Setenv("EMBEDDING_PROVIDER", "")
	for _, variable := range sandbox.Env(root) {
		if _, alreadySet := os.LookupEnv(variable.Name); alreadySet {
			continue
		}
		t.Setenv(variable.Name, variable.Value)
	}
	cfg, err := config.Default()
	if err != nil {
		t.Fatalf("config.Default returned error: %v", err)
	}
	for _, directory := range sandbox.Directories(cfg) {
		if err := os.MkdirAll(directory, 0o755); err != nil {
			t.Fatalf("MkdirAll(%s) returned error: %v", directory, err)
		}
	}
	return cfg
}

func startModelDownloadDaemon(
	t *testing.T,
	cfg config.Config,
	artifacts *modelArtifactServer,
	classification networkcost.Classification,
	interval time.Duration,
) *modelDownloadHarness {
	t.Helper()
	manager, err := daemon.NewManagerWithOptions(context.Background(), cfg, daemon.ManagerOptions{
		ModelDownload: artifacts.options(classification, interval),
	})
	if err != nil {
		t.Fatalf("NewManagerWithOptions returned error: %v", err)
	}
	listener, err := (&net.ListenConfig{}).Listen(context.Background(), "unix", cfg.SocketPath)
	if err != nil {
		t.Fatalf("listen on %s returned error: %v", cfg.SocketPath, err)
	}
	server := grpc.NewServer(
		grpc.MaxRecvMsgSize(grpcutil.MaxMessageBytes),
		grpc.MaxSendMsgSize(grpcutil.MaxMessageBytes),
	)
	pb.RegisterSemanticSearchDaemonServiceServer(server, daemon.NewGRPCServer(manager, nil))
	go func() {
		_ = server.Serve(listener)
	}()
	connection, client, err := grpcutil.DialDaemon(context.Background(), cfg.SocketPath)
	if err != nil {
		server.Stop()
		t.Fatalf("DialDaemon returned error: %v", err)
	}
	var closeOnce sync.Once
	closeDaemon := func() {
		closeOnce.Do(func() {
			_ = connection.Close()
			server.GracefulStop()
			_ = listener.Close()
			closeContext, cancel := context.WithTimeout(context.Background(), testDownloadTimeout)
			defer cancel()
			if closeErr := manager.Close(closeContext); closeErr != nil {
				t.Errorf("Manager.Close returned error: %v", closeErr)
			}
		})
	}
	t.Cleanup(func() {
		artifacts.release()
		closeDaemon()
	})
	return &modelDownloadHarness{config: cfg, client: client}
}

func (harness *modelDownloadHarness) startIndex(t *testing.T, repoPath string) string {
	t.Helper()
	startContext, cancel := context.WithTimeout(grpcutil.WithCorrelation(context.Background()), 10*time.Second)
	defer cancel()
	started, err := harness.client.StartIndex(startContext, &pb.StartIndexRequest{
		Path:     repoPath,
		Splitter: &pb.SplitterConfig{Type: "ast"},
		Client:   &pb.ClientInfo{Name: "model-download-test"},
	})
	if err != nil {
		t.Fatalf("StartIndex returned error: %v", err)
	}
	return started.GetJobId()
}

//go:build live

package live

import (
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"google.golang.org/grpc"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/sandbox"
	"goodkind.io/lm-semantic-search/internal/semantic/milvusgrpc"
	"goodkind.io/lm-semantic-search/internal/store"
)

func prepareLiveDaemonConfig(t *testing.T, gate *embedGate, milvusAddress string, token string, databaseName string, harnessID string, idleTimeout time.Duration, realEmbedding bool) (config.Config, string, *embeddingCallRecorder) {
	t.Helper()
	slog.Debug("prepare isolated daemon configuration", "database", databaseName)
	stateRoot := t.TempDir()
	// The unix socket path must fit macOS's ~104-char sun_path limit, and
	// t.TempDir lives under a long /var/folders path that overflows it, so the
	// socket gets a short /tmp dir instead. State and merkle can use the long temp
	// root.
	socketDir := shortLiveSocketDirectory(t)
	t.Cleanup(func() { _ = os.RemoveAll(socketDir) })
	socketPath := filepath.Join(socketDir, "daemon.sock")

	embeddingRecorder := &embeddingCallRecorder{}
	var embeddingConfig []config.Config
	var embeddingURL string
	if realEmbedding {
		resolved := resolveHarnessConfig(t, true)
		if resolved.OpenAIBaseURL == "" || resolved.EmbeddingModel == "" || resolved.EmbeddingDimension <= 0 {
			t.Fatal("real embedding requires the resolved endpoint, model, and dimension")
		}
		embeddingConfig = []config.Config{resolved}
		embeddingURL = resolved.OpenAIBaseURL
		t.Logf("Real embedding endpoint=%s model=%s dimension=%d", embeddingURL, resolved.EmbeddingModel, resolved.EmbeddingDimension)
	} else {
		embedServer := newFakeEmbeddingServerWithRecorder(t, gate, fakeEmbeddingDimension, embeddingRecorder)
		embeddingURL = embedServer.URL
	}

	cfg := resolveLiveConfig(
		t,
		stateRoot,
		socketPath,
		embeddingURL,
		milvusAddress,
		token,
		databaseName,
		harnessID,
		idleTimeout,
		embeddingConfig...,
	)
	for _, dir := range sandbox.Directories(cfg) {
		if err := store.EnsureDir(dir); err != nil {
			t.Fatalf("EnsureDir(%s) returned error: %v", dir, err)
		}
	}
	if err := store.WriteRegistry(cfg.RegistryPath, model.RegistryFile{Codebases: nil, UpdatedAt: time.Time{}}); err != nil {
		t.Fatalf("WriteRegistry returned error: %v", err)
	}

	return cfg, stateRoot, embeddingRecorder
}

func shortLiveSocketDirectory(t *testing.T) string {
	t.Helper()
	slog.Debug("create isolated Unix socket directory")
	directory := filepath.Join("/tmp", "lms-live-"+randomHex(t, 16))
	if err := os.Mkdir(directory, 0o700); err != nil {
		t.Fatalf("create short Unix socket directory: %v", err)
	}
	return directory
}

func connectLiveSandbox(t *testing.T, milvusAddress string, token string, databaseName string, callRecorder *milvusCallRecorder) (context.Context, *milvusclient.Client, milvusInventory) {
	t.Helper()
	sandboxContext := context.WithValue(
		context.Background(),
		milvusgrpc.CallObserverContextKey{},
		milvusgrpc.CallObserver(callRecorder.observe),
	)
	dialCtx, dialCancel := context.WithTimeout(sandboxContext, 5*time.Second)
	sandboxMilvus, err := milvusclient.New(dialCtx, &milvusclient.ClientConfig{
		Address:     milvusAddress,
		APIKey:      token,
		DBName:      databaseName,
		DialOptions: milvusgrpc.DialOptions(sandboxContext, slog.Default(), milvusgrpc.DefaultCallTimeouts()),
	})
	dialCancel()
	if err != nil {
		t.Fatalf("connect to temporary Milvus database %s: %v", databaseName, err)
	}
	initialized := false
	defer func() {
		if !initialized {
			closeContext, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			if err := sandboxMilvus.Close(closeContext); err != nil {
				t.Errorf("close incomplete sandbox connection: %v", err)
			}
		}
	}()
	sandboxBefore, err := readMilvusInventory(sandboxMilvus)
	if err != nil {
		t.Fatalf("read sandbox Milvus inventory before: %v", err)
	}
	if len(sandboxBefore) != 0 {
		t.Fatalf(
			"temporary Milvus database %q started with collections: %v",
			databaseName,
			sandboxBefore,
		)
	}

	initialized = true
	return sandboxContext, sandboxMilvus, sandboxBefore
}

func resolveHarnessConfig(t *testing.T, requireMilvus bool) config.Config {
	t.Helper()
	slog.Debug("resolve isolated harness configuration")
	defaultConfig, err := config.Default()
	if err != nil {
		t.Fatalf("config.Default returned error: %v", err)
	}
	milvusAddress := strings.TrimSpace(defaultConfig.MilvusAddress)
	if milvusAddress == "" {
		if requireMilvus {
			t.Fatal("BLOCKED: MilvusAddress is empty; set MILVUS_ADDRESS or local config before running the residency suite")
		}
		t.Skip("BLOCKED: MilvusAddress is empty; set MILVUS_ADDRESS or local config before running the live suite")
	}

	return defaultConfig
}

func connectLiveOperator(t *testing.T, operatorContext context.Context, milvusAddress string, token string, requireMilvus bool) *milvusclient.Client {
	t.Helper()
	// Probe Milvus directly first. A dial failure here means the backend is down,
	// so the whole scenario is blocked on the environment rather than the code.
	dialCtx, dialCancel := context.WithTimeout(operatorContext, 5*time.Second)
	operatorMilvus, err := milvusclient.New(dialCtx, &milvusclient.ClientConfig{
		Address:     milvusAddress,
		APIKey:      token,
		DialOptions: milvusgrpc.DialOptions(operatorContext, slog.Default(), milvusgrpc.DefaultCallTimeouts()),
	})
	dialCancel()
	if err != nil {
		if requireMilvus {
			t.Fatalf("BLOCKED: Milvus unreachable at %s: %v", milvusAddress, err)
		}
		t.Skipf("BLOCKED: Milvus unreachable at %s: %v", milvusAddress, err)
	}

	return operatorMilvus
}

type liveDaemonParts struct {
	manager        *daemon.Manager
	conn           *grpc.ClientConn
	client         pb.SemanticSearchDaemonServiceClient
	stopServer     func()
	collectionName string
	codebaseID     string
}

func startLiveHarnessDaemon(t *testing.T, ctx context.Context, cfg config.Config, collectionID string) liveDaemonParts {
	t.Helper()
	slog.Debug("start isolated daemon", "socket", cfg.SocketPath)
	manager, err := daemon.NewManager(ctx, cfg)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}
	initialized := false
	var conn *grpc.ClientConn
	var stopServer func()
	defer func() {
		if initialized {
			return
		}
		if conn != nil {
			if err := conn.Close(); err != nil {
				t.Errorf("close incomplete daemon connection: %v", err)
			}
		}
		if stopServer != nil {
			stopServer()
		}
		closeContext, cancel := context.WithTimeout(context.WithoutCancel(ctx), 30*time.Second)
		defer cancel()
		if err := manager.Close(closeContext); err != nil {
			t.Errorf("close incomplete daemon manager: %v", err)
		}
	}()

	stopServer = startInProcessServer(t, ctx, manager, cfg.SocketPath)

	conn, client, err := grpcutil.DialDaemon(ctx, cfg.SocketPath)
	if err != nil {
		t.Fatalf("DialDaemon returned error: %v", err)
	}

	// A fresh random id derives a unique conv_chunks_<hash> collection name, so
	// the throwaway collection can never be the production one.
	codebase, err := manager.RegisterCollection(ctx, daemon.CollectionRegistration{CollectionID: collectionID, Declaration: liveCollectionDeclaration()})
	if err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}
	if codebase.CollectionName == "" {
		t.Fatal("RegisterCollection returned an empty collection name")
	}
	if codebase.CollectionName == productionProtectedCollection {
		t.Fatalf("throwaway collection name equals production %q; refusing to run", productionProtectedCollection)
	}

	initialized = true
	return liveDaemonParts{
		manager: manager, conn: conn, client: client, stopServer: stopServer,
		collectionName: codebase.CollectionName, codebaseID: codebase.ID,
	}
}

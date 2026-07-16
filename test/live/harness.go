//go:build live

// Package live holds the build-tagged, end-to-end validation of the merged
// conversation-marker feature against a real Milvus. Every run boots the daemon
// gRPC server in-process on a throwaway unix socket, points embedding at a local
// fake, and ingests into a per-test UUID collection whose Milvus name can never
// collide with the operator's production collection. Teardown drops the throwaway
// collection and asserts the production daemon was never touched. Run with:
//
//	go test -tags live -count=1 ./test/live/
//
// or `make live`. Milvus must be reachable; when it is not, each test skips with a
// clear environment note rather than failing.
package live

import (
	"context"
	cryptorand "crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/store"
	"google.golang.org/grpc"
)

const (
	// productionConversationCollection is the operator's real conversation
	// collection. The harness asserts every throwaway collection differs from it,
	// so a live run can never read, write, or drop production conversation rows.
	productionConversationCollection = "conv_chunks_09cfca5e"

	// fakeEmbeddingDimension is the width of every vector the fake embedder
	// returns. It defines the throwaway collection's dimension, learned lazily on
	// first insert, so a small fixed width keeps the collection cheap.
	fakeEmbeddingDimension = 16

	// daemonProcessName is the installed production daemon's process name. The
	// production-safety guard snapshots these pids before boot and asserts none
	// disappeared at teardown, proving the in-process server never signalled the
	// operator's daemon.
	daemonProcessName = "lm-semantic-search-daemon"

	// relativePathField mirrors the collection's scalar column name
	// (internal/semantic), so the scenario-4 direct query can count rows by
	// relative-path prefix without importing unexported constants.
	relativePathField = "relativePath"
	countOutputField  = "count(*)"

	jobPollTimeout  = 90 * time.Second
	jobPollInterval = 100 * time.Millisecond
)

// harness owns one live test's isolated daemon, its throwaway collection, the
// gRPC client that drives ingest, and a direct Milvus client for row-level
// assertions and teardown. Every field is scoped to this test; nothing is shared
// with the operator's running daemon.
type harness struct {
	t              *testing.T
	manager        *daemon.Manager
	conn           *grpc.ClientConn
	client         pb.SemanticSearchDaemonServiceClient
	milvus         *milvusclient.Client
	collectionID   string
	collectionName string
	codebaseID     string
	stateRoot      string
	merkleDir      string
	prodPidsPre    map[int]bool
	embedGate      *embedGate
}

// newHarness builds the isolated daemon and returns a ready harness, or skips the
// test when Milvus is unreachable (a BLOCKED environment condition, not a code
// failure). It registers a per-test UUID conversation collection and asserts the
// derived Milvus name is not the production collection before any ingest runs.
func newHarness(t *testing.T) *harness {
	return newHarnessWithGate(t, nil)
}

// newHarnessWithGate builds the isolated daemon like newHarness but installs an
// embedGate so a test can pace embedding requests and read the job's progress
// between batches. A nil gate is the normal, ungated path.
func newHarnessWithGate(t *testing.T, gate *embedGate) *harness {
	t.Helper()

	defaultConfig, err := config.Default()
	if err != nil {
		t.Fatalf("config.Default returned error: %v", err)
	}
	milvusAddress := strings.TrimSpace(defaultConfig.MilvusAddress)
	if milvusAddress == "" {
		t.Skip("BLOCKED: MilvusAddress is empty; set MILVUS_ADDRESS or local config before running the live suite")
	}

	// Probe Milvus directly first. A dial failure here means the backend is down,
	// so the whole scenario is blocked on the environment rather than the code.
	dialCtx, dialCancel := context.WithTimeout(context.Background(), 5*time.Second)
	milvus, err := milvusclient.New(dialCtx, &milvusclient.ClientConfig{
		Address: milvusAddress,
		APIKey:  defaultConfig.MilvusToken,
	})
	dialCancel()
	if err != nil {
		t.Skipf("BLOCKED: Milvus unreachable at %s: %v", milvusAddress, err)
	}

	prodPidsPre := snapshotProductionDaemonPids()

	stateRoot := t.TempDir()
	// The unix socket path must fit macOS's ~104-char sun_path limit, and
	// t.TempDir lives under a long /var/folders path that overflows it, so the
	// socket gets a short /tmp dir instead. State and merkle can use the long temp
	// root.
	socketDir, err := os.MkdirTemp("/tmp", "lms-live-")
	if err != nil {
		t.Fatalf("mkdir short socket dir returned error: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(socketDir) })
	socketPath := filepath.Join(socketDir, "daemon.sock")

	embedServer := newFakeEmbeddingServer(t, gate)

	cfg := config.Config{
		StateRoot:                 stateRoot,
		SocketPath:                socketPath,
		RegistryPath:              filepath.Join(stateRoot, "registry.json"),
		JobsPath:                  filepath.Join(stateRoot, "jobs.jsonl"),
		EventsPath:                filepath.Join(stateRoot, "events.jsonl"),
		LogsDir:                   filepath.Join(stateRoot, "logs"),
		LogPath:                   filepath.Join(stateRoot, "logs", "daemon.log"),
		MerkleDir:                 filepath.Join(stateRoot, "merkle"),
		LocksDir:                  filepath.Join(stateRoot, "locks"),
		SocketsDir:                filepath.Join(stateRoot, "sockets"),
		ChunksDir:                 filepath.Join(stateRoot, "chunks"),
		GraphDir:                  filepath.Join(stateRoot, "graph"),
		ContextRoot:               filepath.Join(stateRoot, "context"),
		EmbeddingProvider:         "OpenAI",
		EmbeddingModel:            "text-embedding-3-small",
		EmbeddingBatchSize:        8,
		EmbeddingBatchTokenBudget: 1000,
		EmbeddingDimension:        fakeEmbeddingDimension,
		OpenAIAPIKey:              "live-harness-dummy-key", //gitleaks:allow // not a secret: fake embedder needs any non-empty key
		OpenAIBaseURL:             embedServer.URL,
		QueryInstructionPrefix:    "",
		MilvusAddress:             milvusAddress,
		MilvusToken:               defaultConfig.MilvusToken,
		HybridMode:                true,
		BackgroundSyncEnabled:     false,
		SyncIntervalMS:            300000,
		TriggerWatcherEnabled:     false,
		FileWatcherEnabled:        false,
		SyncLockStaleMS:           600000,
		DebugListenerEnabled:      false,
		DebugListenAddr:           "127.0.0.1:0",
		PerfCountersIntervalMS:    0,
		MaxConcurrentIndexJobs:    1,
		ResumeIndexingOnBoot:      false,
	}
	for _, dir := range []string{
		cfg.StateRoot, cfg.LogsDir, cfg.MerkleDir, cfg.LocksDir,
		cfg.SocketsDir, cfg.ChunksDir, cfg.GraphDir, cfg.ContextRoot,
	} {
		if err := store.EnsureDir(dir); err != nil {
			t.Fatalf("EnsureDir(%s) returned error: %v", dir, err)
		}
	}
	if err := store.WriteRegistry(cfg.RegistryPath, model.RegistryFile{}); err != nil {
		t.Fatalf("WriteRegistry returned error: %v", err)
	}

	manager, err := daemon.NewManager(context.Background(), cfg)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}

	stopServer := startInProcessServer(t, manager, socketPath)

	conn, client, err := grpcutil.DialDaemon(context.Background(), socketPath)
	if err != nil {
		stopServer()
		t.Fatalf("DialDaemon returned error: %v", err)
	}

	// A fresh random id derives a unique conv_chunks_<hash> collection name, so
	// the throwaway collection can never be the production one.
	collectionID := "live-marker-" + randomID()
	codebase, err := manager.RegisterConversationCollection(context.Background(), collectionID)
	if err != nil {
		_ = conn.Close()
		stopServer()
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	if codebase.CollectionName == "" {
		_ = conn.Close()
		stopServer()
		t.Fatal("RegisterConversationCollection returned an empty collection name")
	}
	if codebase.CollectionName == productionConversationCollection {
		_ = conn.Close()
		stopServer()
		t.Fatalf("throwaway collection name equals production %q; refusing to run", productionConversationCollection)
	}

	h := &harness{
		t:              t,
		manager:        manager,
		conn:           conn,
		client:         client,
		milvus:         milvus,
		collectionID:   collectionID,
		collectionName: codebase.CollectionName,
		codebaseID:     codebase.ID,
		stateRoot:      stateRoot,
		merkleDir:      cfg.MerkleDir,
		prodPidsPre:    prodPidsPre,
		embedGate:      gate,
	}
	t.Cleanup(func() { h.teardown(stopServer) })
	return h
}

// teardown drops the throwaway collection, closes the clients, stops the
// in-process server, and asserts the production daemon set is unchanged. It
// re-asserts the collection name was never the production one, so an isolation
// breach fails the test even on the teardown path.
func (h *harness) teardown(stopServer func()) {
	if h.collectionName == productionConversationCollection {
		h.t.Errorf("isolation breach: harness collection was the production collection %q", productionConversationCollection)
	} else {
		dropCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		if err := h.milvus.DropCollection(dropCtx, milvusclient.NewDropCollectionOption(h.collectionName)); err != nil {
			// A missing collection (a scenario that never inserted) is not an error.
			if !strings.Contains(err.Error(), "not exist") && !strings.Contains(err.Error(), "not found") {
				h.t.Errorf("DropCollection(%s) returned error: %v", h.collectionName, err)
			}
		}
		cancel()
	}
	if err := h.conn.Close(); err != nil {
		h.t.Errorf("close gRPC connection returned error: %v", err)
	}
	closeCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	_ = h.milvus.Close(closeCtx)
	cancel()
	stopServer()
	h.assertProductionUntouched()
}

// assertProductionUntouched confirms every production daemon pid seen before boot
// is still alive after the run, proving the in-process server never signalled the
// operator's daemon.
func (h *harness) assertProductionUntouched() {
	post := snapshotProductionDaemonPids()
	for pid := range h.prodPidsPre {
		if !post[pid] {
			h.t.Errorf("production daemon pid %d disappeared during the live run; isolation breached", pid)
		}
	}
}

// startInProcessServer serves the daemon gRPC service on a throwaway unix socket
// in a goroutine and returns a stop closure that GracefulStops the server and
// removes the socket. Readiness is a successful dial by the caller, so no log
// tailing is needed. It mirrors internal/daemon's own test helper.
func startInProcessServer(t *testing.T, manager *daemon.Manager, socketPath string) func() {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(socketPath), 0o755); err != nil {
		t.Fatalf("mkdir socket dir returned error: %v", err)
	}
	_ = os.Remove(socketPath)

	listener, err := (&net.ListenConfig{}).Listen(context.Background(), "unix", socketPath)
	if err != nil {
		t.Fatalf("listen on unix socket returned error: %v", err)
	}
	server := grpc.NewServer(
		grpc.MaxRecvMsgSize(grpcutil.MaxMessageBytes),
		grpc.MaxSendMsgSize(grpcutil.MaxMessageBytes),
	)
	pb.RegisterSemanticSearchDaemonServiceServer(server, daemon.NewGRPCServer(manager, nil))
	go func() {
		_ = server.Serve(listener)
	}()
	return func() {
		server.GracefulStop()
		_ = listener.Close()
		_ = os.Remove(socketPath)
	}
}

// snapshotProductionDaemonPids returns the pids of every running installed
// production daemon, meaning those whose executable lives outside a temp root. A
// live-test server boots in-process (no daemon subprocess), so an empty set is
// the common, correct result on a host with no production daemon running.
func snapshotProductionDaemonPids() map[int]bool {
	pids := map[int]bool{}
	out, err := exec.Command("pgrep", "-f", daemonProcessName).Output()
	if err != nil {
		return pids
	}
	for _, field := range strings.Fields(string(out)) {
		pid, convErr := strconv.Atoi(field)
		if convErr != nil || pid <= 0 {
			continue
		}
		if isProductionDaemonPid(pid) {
			pids[pid] = true
		}
	}
	return pids
}

// isProductionDaemonPid reports whether pid runs an installed production daemon
// rather than a temp-rooted test artifact. It resolves the command with ps and
// keeps only pids whose executable path sits outside every temp root, so a
// concurrent live test's process is never mistaken for production.
func isProductionDaemonPid(pid int) bool {
	out, err := exec.Command("ps", "-o", "command=", "-p", strconv.Itoa(pid)).Output()
	if err != nil {
		return false
	}
	command := strings.TrimSpace(string(out))
	if command == "" {
		return false
	}
	execPath := command
	if idx := strings.IndexByte(command, ' '); idx >= 0 {
		execPath = command[:idx]
	}
	return !underTempRoot(execPath)
}

// underTempRoot reports whether path lives under a temp root macOS resolves the
// system temp dir through, so a test artifact is never counted as production.
func underTempRoot(path string) bool {
	clean := filepath.Clean(path)
	roots := []string{"/tmp", "/private/tmp", "/private/var/folders", "/var/folders"}
	if osTemp := filepath.Clean(os.TempDir()); osTemp != "" && osTemp != "." {
		roots = append(roots, osTemp)
	}
	for _, root := range roots {
		if clean == root || strings.HasPrefix(clean, root+string(os.PathSeparator)) {
			return true
		}
	}
	return false
}

// newFakeEmbeddingServer starts a local OpenAI-compatible embedding endpoint. It
// answers the health probe (GET .../models) with a minimal models list and every
// embed request (POST .../embeddings) with one fixed-width vector per input,
// keyed by a content hash so identical content yields an identical vector and the
// engine's content-hash reuse path stays exercised.
// embedGate lets a test pace embedding requests. When installed, every embed
// request announces its batch size on arrived, then blocks until the test sends
// on release, so the test can read job progress between batches. The models
// (health) route is never gated.
type embedGate struct {
	arrived chan int
	release chan struct{}
}

func newFakeEmbeddingServer(t *testing.T, gate *embedGate) *httptest.Server {
	t.Helper()
	handler := http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		switch {
		case strings.HasSuffix(request.URL.Path, "/models"):
			writeModelsList(writer)
		case strings.HasSuffix(request.URL.Path, "/embeddings"):
			writeEmbeddings(t, writer, request, gate)
		default:
			http.Error(writer, "unexpected path "+request.URL.Path, http.StatusNotFound)
		}
	})
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return server
}

func writeModelsList(writer http.ResponseWriter) {
	writer.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(writer).Encode(map[string]any{
		"object": "list",
		"data": []map[string]any{
			{"id": "text-embedding-3-small", "object": "model", "created": 0, "owned_by": "live-harness"},
		},
	})
}

func writeEmbeddings(t *testing.T, writer http.ResponseWriter, request *http.Request, gate *embedGate) {
	inputs, err := decodeEmbeddingInputs(request)
	if err != nil {
		http.Error(writer, err.Error(), http.StatusBadRequest)
		return
	}
	if gate != nil {
		gate.arrived <- len(inputs)
		<-gate.release
	}
	type row struct {
		Object    string    `json:"object"`
		Index     int       `json:"index"`
		Embedding []float64 `json:"embedding"`
	}
	rows := make([]row, 0, len(inputs))
	for index, text := range inputs {
		rows = append(rows, row{Object: "embedding", Index: index, Embedding: deterministicVector(text)})
	}
	writer.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(writer).Encode(map[string]any{
		"object": "list",
		"model":  "text-embedding-3-small",
		"data":   rows,
		"usage":  map[string]int{"prompt_tokens": 1, "total_tokens": 1},
	}); err != nil {
		t.Logf("encode embedding response failed: %v", err)
	}
}

// decodeEmbeddingInputs reads the request's input field, accepting both the array
// form the batch embedder sends and a bare single string, so the fake is robust
// to either shape.
func decodeEmbeddingInputs(request *http.Request) ([]string, error) {
	var body struct {
		Input json.RawMessage `json:"input"`
	}
	if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("decode embedding request: %w", err)
	}
	var asArray []string
	if err := json.Unmarshal(body.Input, &asArray); err == nil {
		return asArray, nil
	}
	var asString string
	if err := json.Unmarshal(body.Input, &asString); err == nil {
		return []string{asString}, nil
	}
	return nil, fmt.Errorf("embedding request input was neither an array nor a string")
}

// deterministicVector maps content to a fixed-width unit vector derived from its
// SHA-256 digest, so identical content always yields an identical vector (reuse
// works) and distinct content yields a distinct one.
func deterministicVector(content string) []float64 {
	digest := sha256.Sum256([]byte(content))
	vector := make([]float64, fakeEmbeddingDimension)
	var norm float64
	for i := 0; i < fakeEmbeddingDimension; i++ {
		value := (float64(digest[i]) - 128.0) / 128.0
		vector[i] = value
		norm += value * value
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		vector[0] = 1
		return vector
	}
	for i := range vector {
		vector[i] /= norm
	}
	return vector
}

// correlatedContext wraps ctx with the trace/span identity the daemon requires in
// strict mode, so every RPC and manager read carries a correlation.
func correlatedContext() context.Context {
	return grpcutil.WithCorrelation(context.Background())
}

// randomID returns a hex token unique per test, so each run's collection id (and
// therefore its derived Milvus collection name) is fresh and never collides with
// another run or with production.
func randomID() string {
	buffer := make([]byte, 16)
	if _, err := cryptorand.Read(buffer); err != nil {
		return strconv.FormatInt(time.Now().UnixNano(), 16)
	}
	return hex.EncodeToString(buffer)
}

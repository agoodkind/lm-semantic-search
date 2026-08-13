//go:build restartacceptance

package restartacceptance

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"maps"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"google.golang.org/grpc"
)

func TestStartCaseProxiesWarmsEmbeddingBackendBeforeScenario(t *testing.T) {
	type readinessRequest struct {
		path       string
		authority  string
		model      string
		dimensions int32
		input      []string
		err        error
	}
	requestStarted := make(chan readinessRequest, 1)
	releaseResponse := make(chan struct{})
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(releaseResponse) }) }
	t.Cleanup(release)
	backend := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var payload struct {
			Model      string   `json:"model"`
			Input      []string `json:"input"`
			Dimensions int32    `json:"dimensions"`
		}
		decodeErr := json.NewDecoder(request.Body).Decode(&payload)
		requestStarted <- readinessRequest{
			path:       request.URL.Path,
			authority:  request.Header.Get("Authorization"),
			model:      payload.Model,
			dimensions: payload.Dimensions,
			input:      payload.Input,
			err:        decodeErr,
		}
		<-releaseResponse
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}],"model":"readiness-model","usage":{"prompt_tokens":1,"total_tokens":1}}`))
	}))
	t.Cleanup(backend.Close)
	t.Setenv("OPENAI_API_KEY", "test-key")
	t.Setenv("OPENAI_BASE_URL", backend.URL+"/v1")
	t.Setenv("EMBEDDING_MODEL", "readiness-model")
	t.Setenv("EMBEDDING_DIMENSION", "2")

	type proxyResult struct {
		proxies caseProxies
		err     error
	}
	result := make(chan proxyResult, 1)
	go func() {
		proxies, err := startCaseProxies(context.Background())
		result <- proxyResult{proxies: proxies, err: err}
	}()

	select {
	case request := <-requestStarted:
		if request.err != nil {
			t.Fatalf("decode embedding readiness request: %v", request.err)
		}
		if request.path != "/v1/embeddings" {
			t.Fatalf("embedding readiness path = %q, want /v1/embeddings", request.path)
		}
		if request.authority != "Bearer test-key" {
			t.Fatalf("embedding readiness authorization = %q, want Bearer test-key", request.authority)
		}
		if request.model != "readiness-model" || request.dimensions != 2 {
			t.Fatalf("embedding readiness model = %q dimensions = %d, want readiness-model and 2", request.model, request.dimensions)
		}
		if !slices.Equal(request.input, []string{"restart acceptance embedding readiness"}) {
			t.Fatalf("embedding readiness input = %v", request.input)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("embedding readiness request did not start")
	}
	select {
	case returned := <-result:
		_ = returned.proxies.close()
		t.Fatalf("startCaseProxies returned before embedding readiness completed: %v", returned.err)
	default:
	}
	release()
	returned := <-result
	if returned.err != nil {
		t.Fatal(returned.err)
	}
	if err := returned.proxies.close(); err != nil {
		t.Fatal(err)
	}
}

func TestInstalledLMSForcesProductionMetadataTimeoutAcrossProcessBoundary(t *testing.T) {
	configHome := t.TempDir()
	configRoot := filepath.Join(configHome, "lm-semantic-search")
	if err := os.MkdirAll(configRoot, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(configRoot, "config.json"),
		[]byte(`{"milvusMetadataCallTimeoutMs":2}`),
		0o600,
	); err != nil {
		t.Fatal(err)
	}
	t.Setenv("XDG_CONFIG_HOME", configHome)
	t.Setenv("CLAUDE_CONTEXT_MILVUS_METADATA_CALL_TIMEOUT_MS", "1")

	run := acceptanceRun{Paths: pathsForRun(t.TempDir())}
	process := installedLMSProcess(run)
	probePath := filepath.Join(t.TempDir(), "metadata-timeout")
	process.Path = os.Args[0]
	process.Args = []string{"-test.run=^TestRestartAcceptanceConfigProbe$"}
	process.Environment["LMS_RESTART_CONFIG_PROBE"] = probePath
	command, err := startInstalledProcess(process)
	if err != nil {
		t.Fatal(err)
	}
	if err := command.Wait(); err != nil {
		t.Fatalf("wait for config probe: %v", err)
	}
	body, err := os.ReadFile(probePath)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(body)); got != "60000" {
		t.Fatalf("effective metadata call timeout = %q, want 60000", got)
	}
}

func TestInstalledScenarioBKeepsMetadataTimeoutBelowFailureBound(t *testing.T) {
	configHome := t.TempDir()
	configRoot := filepath.Join(configHome, "lm-semantic-search")
	if err := os.MkdirAll(configRoot, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(configRoot, "config.json"),
		[]byte(`{"milvusMetadataCallTimeoutMs":60000}`),
		0o600,
	); err != nil {
		t.Fatal(err)
	}
	t.Setenv("XDG_CONFIG_HOME", configHome)
	t.Setenv("CLAUDE_CONTEXT_MILVUS_METADATA_CALL_TIMEOUT_MS", "60000")

	run := acceptanceRun{Paths: pathsForRun(t.TempDir())}
	process := installedLMSProcessForScenarioB(run)
	probePath := filepath.Join(t.TempDir(), "scenario-b-metadata-timeout")
	process.Path = os.Args[0]
	process.Args = []string{"-test.run=^TestRestartAcceptanceConfigProbe$"}
	process.Environment["LMS_RESTART_CONFIG_PROBE"] = probePath
	command, err := startInstalledProcess(process)
	if err != nil {
		t.Fatal(err)
	}
	if err := command.Wait(); err != nil {
		t.Fatalf("wait for config probe: %v", err)
	}
	body, err := os.ReadFile(probePath)
	if err != nil {
		t.Fatal(err)
	}
	timeoutMilliseconds, err := strconv.Atoi(strings.TrimSpace(string(body)))
	if err != nil {
		t.Fatalf("parse effective metadata timeout: %v", err)
	}
	if timeoutMilliseconds <= 0 || timeoutMilliseconds >= 15000 {
		t.Fatalf("effective metadata call timeout = %dms, want between 1ms and 14999ms", timeoutMilliseconds)
	}
}

func TestInstalledScenarioBDisablesAutomaticSyncAdmissionAcrossProcessBoundary(t *testing.T) {
	t.Setenv("CLAUDE_CONTEXT_BACKGROUND_SYNC", "true")
	t.Setenv("CLAUDE_CONTEXT_TRIGGER_WATCHER", "true")
	t.Setenv("CLAUDE_CONTEXT_FILE_WATCHER", "true")

	run := acceptanceRun{Paths: pathsForRun(t.TempDir())}
	process := installedLMSProcessForScenarioB(run)
	probePath := filepath.Join(t.TempDir(), "scenario-b-sync-admission")
	process.Path = os.Args[0]
	process.Args = []string{"-test.run=^TestRestartAcceptanceSyncAdmissionConfigProbe$"}
	process.Environment["LMS_RESTART_CONFIG_PROBE"] = probePath
	command, err := startInstalledProcess(process)
	if err != nil {
		t.Fatal(err)
	}
	if err := command.Wait(); err != nil {
		t.Fatalf("wait for config probe: %v", err)
	}
	body, err := os.ReadFile(probePath)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(body)); got != "background=false trigger=false file=false" {
		t.Fatalf("effective sync admission = %q, want all disabled", got)
	}
}

func TestPrepareScenarioBRebuildAddsTwoFilesWithoutChangingIndexedFiles(t *testing.T) {
	fixture, err := createAcceptanceFixture(t.TempDir(), "b")
	if err != nil {
		t.Fatal(err)
	}
	before := make(map[string]string, len(fixture.files))
	for _, name := range fixture.files {
		body, readErr := os.ReadFile(filepath.Join(fixture.root, name))
		if readErr != nil {
			t.Fatal(readErr)
		}
		before[name] = string(body)
	}
	added, err := prepareScenarioBRebuild(fixture)
	if err != nil {
		t.Fatalf("prepare scenario B rebuild: %v", err)
	}
	changed := 0
	for _, name := range fixture.files {
		body, readErr := os.ReadFile(filepath.Join(fixture.root, name))
		if readErr != nil {
			t.Fatal(readErr)
		}
		if string(body) != before[name] {
			changed++
		}
	}
	if changed != 0 {
		t.Fatalf("changed existing fixture files = %d, want 0", changed)
	}
	if !slices.Equal(slices.Sorted(maps.Keys(added)), []string{"04.go", "05.go"}) {
		t.Fatalf("added fixture IDs = %v, want [04.go 05.go]", maps.Keys(added))
	}
	for name, expectedHash := range added {
		if _, err := os.Stat(filepath.Join(fixture.root, name)); err != nil {
			t.Fatalf("stat added fixture %s: %v", name, err)
		}
		body, err := os.ReadFile(filepath.Join(fixture.root, name))
		if err != nil {
			t.Fatal(err)
		}
		digest := sha256.Sum256(body)
		if actual := hex.EncodeToString(digest[:]); actual != expectedHash {
			t.Fatalf("added fixture hash = %q, want %q", actual, expectedHash)
		}
	}
}

func TestPrepareScenarioBRebuildRejectsExistingTargetBeforeWriting(t *testing.T) {
	fixture, err := createAcceptanceFixture(t.TempDir(), "b")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(fixture.root, "05.go"), 0o700); err != nil {
		t.Fatal(err)
	}
	if _, err := prepareScenarioBRebuild(fixture); err == nil {
		t.Fatal("prepareScenarioBRebuild succeeded with an existing target")
	}
	if _, err := os.Stat(filepath.Join(fixture.root, "04.go")); !os.IsNotExist(err) {
		t.Fatalf("04.go exists after rejected preparation: %v", err)
	}
}

func TestRunInstalledScenarioBExercisesSeedFaultRecoveryAndPreservation(t *testing.T) {
	embeddingBackend := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"data":[{"embedding":[1]}]}`))
	}))
	t.Cleanup(embeddingBackend.Close)
	embeddingListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	embeddingProxy, err := newEmbeddingProxy(embeddingListener, embeddingBackend.URL)
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = embeddingProxy.Serve() }()
	t.Cleanup(func() { _ = embeddingProxy.Close() })

	milvusBackendListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	milvusBackend := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(milvusBackend, &proxyTestMilvusServer{})
	go func() { _ = milvusBackend.Serve(milvusBackendListener) }()
	t.Cleanup(milvusBackend.Stop)
	milvusListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	milvusProxy, err := newMilvusProxy(milvusListener, milvusBackendListener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = milvusProxy.Serve() }()
	t.Cleanup(func() { _ = milvusProxy.Close() })

	server := newScenarioFakeServer()
	server.embeddingURL = "http://" + embeddingListener.Addr().String()
	server.milvusAddress = milvusListener.Addr().String()
	server.requireSearchBeforeIndex = true
	server.requireSyncForRebuild = true
	server.requireExplicitRecovery = true
	server.allowInitialSeed = true
	server.orchestrateScenarioB = true
	server.scenarioBEmbeddingDelay = 50 * time.Millisecond
	client := startScenarioFakeGRPC(t, server)
	fixture, err := createAcceptanceFixture(t.TempDir(), "b")
	if err != nil {
		t.Fatal(err)
	}
	run := acceptanceRun{Paths: pathsForRun(t.TempDir())}
	server.expectedPath = fixture.root
	recorder, _ := scenarioTestRecorder(t)
	driver := &realAcceptanceDriver{
		startScenarioBDaemon: func(_ context.Context, process installedProcess, socket string) (*daemonRuntime, error) {
			if socket != run.Paths.LMSSocket {
				t.Fatalf("scenario B socket = %q, want %q", socket, run.Paths.LMSSocket)
			}
			for _, name := range []string{"CLAUDE_CONTEXT_BACKGROUND_SYNC", "CLAUDE_CONTEXT_TRIGGER_WATCHER", "CLAUDE_CONTEXT_FILE_WATCHER"} {
				if process.Environment[name] != "false" {
					t.Fatalf("%s = %q, want false", name, process.Environment[name])
				}
			}
			return &daemonRuntime{client: client}, nil
		},
		openScenarioBObservers: func(_ context.Context, _ *daemonRuntime, _ acceptanceRun, observedFixture acceptanceFixture) (rowSnapshotObserver, checkpointSnapshotObserver, func(), error) {
			server.checkpointRoot = observedFixture.root
			return server.observeRows, server.observeCheckpoint, func() {}, nil
		},
	}
	err = driver.runInstalledScenarioB(context.Background(), run, caseProxies{embedding: embeddingProxy, milvus: milvusProxy}, fixture, recorder)
	if err != nil {
		t.Fatalf("runInstalledScenarioB: %v", err)
	}
	for _, name := range []string{"04.go", "05.go"} {
		if _, err := os.Stat(filepath.Join(fixture.root, name)); err != nil {
			t.Fatalf("stat %s: %v", name, err)
		}
	}
	if inputs := embeddingProxy.Inputs(); !slices.Equal(inputs, []string{"01.go", "04.go", "04.go", "05.go"}) {
		t.Fatalf("scenario B embedding inputs = %v, want [01.go 04.go 04.go 05.go]", inputs)
	}
	if reached := embeddingProxy.GateReachedCount(); reached != 1 {
		t.Fatalf("scenario B embedding gate reached = %d, want 1", reached)
	}
	server.mutex.Lock()
	jobs := append([]*pb.Job(nil), server.jobs...)
	server.mutex.Unlock()
	if len(jobs) != 3 || jobs[1].GetState() != "failed" || jobs[2].GetState() != "completed" {
		t.Fatalf("scenario B jobs = %+v, want seeded, failed, recovered", jobs)
	}
}

func TestInstalledLMSForcesProductionCollectionLoadTimeoutAcrossProcessBoundary(t *testing.T) {
	configHome := t.TempDir()
	configRoot := filepath.Join(configHome, "lm-semantic-search")
	if err := os.MkdirAll(configRoot, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(configRoot, "config.json"),
		[]byte(`{"milvusCollectionLoadTimeoutMs":2}`),
		0o600,
	); err != nil {
		t.Fatal(err)
	}
	t.Setenv("XDG_CONFIG_HOME", configHome)
	t.Setenv("CLAUDE_CONTEXT_MILVUS_COLLECTION_LOAD_TIMEOUT_MS", "1")

	run := acceptanceRun{Paths: pathsForRun(t.TempDir())}
	process := installedLMSProcess(run)
	probePath := filepath.Join(t.TempDir(), "collection-load-timeout")
	process.Path = os.Args[0]
	process.Args = []string{"-test.run=^TestRestartAcceptanceCollectionLoadConfigProbe$"}
	process.Environment["LMS_RESTART_CONFIG_PROBE"] = probePath
	command, err := startInstalledProcess(process)
	if err != nil {
		t.Fatal(err)
	}
	if err := command.Wait(); err != nil {
		t.Fatalf("wait for config probe: %v", err)
	}
	body, err := os.ReadFile(probePath)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(body)); got != "90000" {
		t.Fatalf("effective collection load timeout = %q, want 90000", got)
	}
}

func TestRestartAcceptanceConfigProbe(t *testing.T) {
	probePath := os.Getenv("LMS_RESTART_CONFIG_PROBE")
	if probePath == "" {
		return
	}
	cfg, err := config.Default()
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(strconv.Itoa(cfg.MilvusMetadataCallTimeoutMS) + "\n")
	if err := os.WriteFile(probePath, body, 0o600); err != nil {
		t.Fatal(fmt.Errorf("write config probe: %w", err))
	}
}

func TestRestartAcceptanceCollectionLoadConfigProbe(t *testing.T) {
	probePath := os.Getenv("LMS_RESTART_CONFIG_PROBE")
	if probePath == "" {
		return
	}
	cfg, err := config.Default()
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(strconv.Itoa(cfg.MilvusCollectionLoadTimeoutMS) + "\n")
	if err := os.WriteFile(probePath, body, 0o600); err != nil {
		t.Fatal(fmt.Errorf("write config probe: %w", err))
	}
}

func TestRestartAcceptanceSyncAdmissionConfigProbe(t *testing.T) {
	probePath := os.Getenv("LMS_RESTART_CONFIG_PROBE")
	if probePath == "" {
		return
	}
	cfg, err := config.Default()
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(fmt.Sprintf(
		"background=%t trigger=%t file=%t\n",
		cfg.BackgroundSyncEnabled,
		cfg.TriggerWatcherEnabled,
		cfg.FileWatcherEnabled,
	))
	if err := os.WriteFile(probePath, body, 0o600); err != nil {
		t.Fatal(fmt.Errorf("write config probe: %w", err))
	}
}

func TestPreserveCaseDiagnosticsCopiesTheDaemonLogBeforeCleanup(t *testing.T) {
	paths := pathsForRun(t.TempDir())
	logPath := installedLMSLogPath(paths)
	if err := os.MkdirAll(filepath.Dir(logPath), 0o700); err != nil {
		t.Fatal(err)
	}
	const body = `{"msg":"hybrid search failed","err":"raw Milvus failure"}`
	if err := os.WriteFile(logPath, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}

	if err := preserveCaseDiagnostics(paths, "b"); err != nil {
		t.Fatalf("preserve diagnostics: %v", err)
	}
	preserved, err := os.ReadFile(filepath.Join(paths.Artifacts, "scenario-b-lms.log"))
	if err != nil {
		t.Fatalf("read preserved log: %v", err)
	}
	if string(preserved) != body {
		t.Fatalf("preserved log = %q, want %q", preserved, body)
	}
}

func TestPreserveCaseDiagnosticsRejectsAnUnsafeScenarioName(t *testing.T) {
	paths := pathsForRun(t.TempDir())
	if err := preserveCaseDiagnostics(paths, "../escape"); err == nil {
		t.Fatal("preserve diagnostics accepted an unsafe scenario name")
	}
}

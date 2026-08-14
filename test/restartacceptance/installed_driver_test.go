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
	"sync/atomic"
	"testing"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"google.golang.org/grpc"
)

func TestStartCaseProxiesUsesOnlyLocalEmbeddingBackend(t *testing.T) {
	t.Setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/forbidden-host-backend")
	proxies, err := startCaseProxies(context.Background())
	if err != nil {
		t.Fatalf("start isolated proxies: %v", err)
	}
	t.Cleanup(func() {
		if closeErr := proxies.close(); closeErr != nil {
			t.Errorf("close isolated proxies: %v", closeErr)
		}
	})
	body := strings.NewReader(`{"model":"restart-acceptance","input":["same","same"],"dimensions":16}`)
	response, err := http.Post(
		fmt.Sprintf("http://127.0.0.1:%d/v1/embeddings", embeddingProxyPort),
		"application/json",
		body,
	)
	if err != nil {
		t.Fatalf("request local embedding proxy: %v", err)
	}
	defer func() { _ = response.Body.Close() }()
	var decoded struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		t.Fatalf("decode local embedding response: %v", err)
	}
	if response.StatusCode != http.StatusOK || len(decoded.Data) != 2 {
		t.Fatalf("local embedding status = %d rows = %d", response.StatusCode, len(decoded.Data))
	}
	if len(decoded.Data[0].Embedding) != 16 {
		t.Fatalf("local embedding width = %d, want 16", len(decoded.Data[0].Embedding))
	}
	if !slices.Equal(decoded.Data[0].Embedding, decoded.Data[1].Embedding) {
		t.Fatal("identical local embedding inputs returned different vectors")
	}
}

func TestVerifyEmbeddingReadinessReusesConnectionAfterRejectedProbe(t *testing.T) {
	var requestCount atomic.Int32
	var connectionCount atomic.Int32
	backend := httptest.NewUnstartedServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		if requestCount.Add(1) == 1 {
			writer.WriteHeader(http.StatusServiceUnavailable)
			_, _ = writer.Write([]byte(strings.Repeat("unavailable", 4096)))
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1]}]}`))
	}))
	backend.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			connectionCount.Add(1)
		}
	}
	backend.Start()
	t.Cleanup(backend.Close)
	if err := verifyEmbeddingReadiness(
		context.Background(),
		backend.URL+"/v1",
		"readiness-model",
		0,
	); err == nil {
		t.Fatal("rejected embedding readiness probe succeeded")
	}
	if err := verifyEmbeddingReadiness(
		context.Background(),
		backend.URL+"/v1",
		"readiness-model",
		0,
	); err != nil {
		t.Fatal(err)
	}
	if got := connectionCount.Load(); got != 1 {
		t.Fatalf("embedding readiness connections = %d, want 1", got)
	}
}

func TestVerifyEmbeddingReadinessRejectsOversizedResponse(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1]}]}`))
		_, _ = writer.Write([]byte(strings.Repeat("x", embeddingReadinessResponseLimit)))
	}))
	t.Cleanup(backend.Close)
	if err := verifyEmbeddingReadiness(
		context.Background(),
		backend.URL+"/v1",
		"readiness-model",
		0,
	); err == nil {
		t.Fatal("oversized embedding readiness response succeeded")
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

func TestInstalledScenariosCanDisablePeriodicMaintenance(t *testing.T) {
	run := acceptanceRun{Paths: pathsForRun(t.TempDir())}
	process := installedLMSProcessWithoutPeriodicMaintenance(run)
	if got := process.Environment["CLAUDE_CONTEXT_BACKGROUND_SYNC"]; got != "false" {
		t.Fatalf("background sync = %q, want false", got)
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

func TestInstalledScenarioGKeepsCallerWaitBelowInternalLoadBoundAcrossProcessBoundary(t *testing.T) {
	t.Setenv("CLAUDE_CONTEXT_MILVUS_COLLECTION_LOAD_TIMEOUT_MS", "1")
	t.Setenv("CLAUDE_CONTEXT_MILVUS_COLLECTION_LOAD_WAIT_TIMEOUT_MS", "90000")
	t.Setenv("CLAUDE_CONTEXT_MILVUS_COLLECTION_IDLE_TIMEOUT_MS", "1")

	run := acceptanceRun{Paths: pathsForRun(t.TempDir())}
	process := installedLMSProcessForScenarioG(run)
	probePath := filepath.Join(t.TempDir(), "scenario-g-load-timeouts")
	process.Path = os.Args[0]
	process.Args = []string{"-test.run=^TestRestartAcceptanceScenarioGLoadConfigProbe$"}
	process.Environment["LMS_RESTART_CONFIG_PROBE"] = probePath
	command, err := startInstalledProcess(process)
	if err != nil {
		t.Fatal(err)
	}
	if err := command.Wait(); err != nil {
		t.Fatalf("wait for scenario G config probe: %v", err)
	}
	body, err := os.ReadFile(probePath)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(body)); got != "12000 7000 60000" {
		t.Fatalf("effective scenario G timeouts = %q, want internal 12000 caller 7000 idle 60000", got)
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

func TestRestartAcceptanceScenarioGLoadConfigProbe(t *testing.T) {
	probePath := os.Getenv("LMS_RESTART_CONFIG_PROBE")
	if probePath == "" {
		return
	}
	cfg, err := config.Default()
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(fmt.Sprintf(
		"%d %d %d\n",
		cfg.MilvusCollectionLoadTimeoutMS,
		cfg.MilvusCollectionLoadWaitTimeoutMS,
		cfg.MilvusCollectionIdleTimeoutMS,
	))
	if err := os.WriteFile(probePath, body, 0o600); err != nil {
		t.Fatal(fmt.Errorf("write scenario G config probe: %w", err))
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

func TestWaitForSeededClydeSearchWaitsForFeederConvergence(t *testing.T) {
	statusCalls := 0
	searchCalls := 0
	result, err := waitForSeededClydeSearch(
		context.Background(),
		func(context.Context) (clydeStatusObservation, error) {
			statusCalls++
			if statusCalls == 1 {
				return clydeStatusObservation{
					PID: 7, Responding: true, LastSyncUnix: 1,
					Manifest: 1, Needed: 1, Pending: 1,
				}, nil
			}
			return clydeStatusObservation{
				PID: 8, Responding: true, LastSyncUnix: 2,
				Manifest: 1, Embedded: 1,
			}, nil
		},
		func(context.Context) (semanticSearchObservation, error) {
			searchCalls++
			return semanticSearchObservation{
				Succeeded: true, Source: "semantic", Matches: 1,
				ResultIDs: []string{"conversation:0"},
			}, nil
		},
		50*time.Millisecond,
		10*time.Millisecond,
		time.Millisecond,
	)
	if err != nil {
		t.Fatalf("waitForSeededClydeSearch returned error: %v", err)
	}
	if statusCalls < 2 {
		t.Fatalf("status calls = %d, want at least 2", statusCalls)
	}
	if searchCalls != 1 {
		t.Fatalf("search calls = %d, want 1 after convergence", searchCalls)
	}
	if result.Matches != 1 {
		t.Fatalf("matches = %d, want 1", result.Matches)
	}
}

func TestWaitForSeededClydeSearchRetriesAfterOneBoundedSearch(t *testing.T) {
	searchCalls := 0
	result, err := waitForSeededClydeSearch(
		context.Background(),
		func(context.Context) (clydeStatusObservation, error) {
			return clydeStatusObservation{
				PID: 7, Responding: true, LastSyncUnix: 2,
				Manifest: 1, Embedded: 1,
			}, nil
		},
		func(searchContext context.Context) (semanticSearchObservation, error) {
			searchCalls++
			if searchCalls == 1 {
				<-searchContext.Done()
				return semanticSearchObservation{}, searchContext.Err()
			}
			return semanticSearchObservation{Succeeded: true, Source: "semantic", Matches: 1}, nil
		},
		100*time.Millisecond,
		5*time.Millisecond,
		time.Millisecond,
	)
	if err != nil {
		t.Fatalf("waitForSeededClydeSearch returned error: %v", err)
	}
	if searchCalls != 2 {
		t.Fatalf("search calls = %d, want 2", searchCalls)
	}
	if result.Matches != 1 {
		t.Fatalf("matches = %d, want 1", result.Matches)
	}
}

func TestRequireClonePayloadAllowsAnEmptyUnusedMinioVolume(t *testing.T) {
	caseRoot := t.TempDir()
	for _, name := range []string{"milvus", "minio", "minio-default"} {
		if err := os.MkdirAll(filepath.Join(caseRoot, name), 0o700); err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{"milvus", "minio"} {
		if err := os.WriteFile(filepath.Join(caseRoot, name, "payload"), []byte("data"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := requireClonePayload(caseRoot); err != nil {
		t.Fatalf("requireClonePayload returned error: %v", err)
	}
}

func TestStartScenarioGTargetJobWaitsForCallerLoad(t *testing.T) {
	loadCount := 0
	callerStarted := false
	jobStarted := false
	jobID, err := startScenarioGTargetJob(
		context.Background(),
		func() {
			callerStarted = true
			loadCount = 1
		},
		func() int { return loadCount },
		func(context.Context) (string, error) {
			if !callerStarted || loadCount == 0 {
				t.Fatal("target job started before the public caller load")
			}
			jobStarted = true
			return "target-job", nil
		},
		50*time.Millisecond,
		time.Millisecond,
	)
	if err != nil {
		t.Fatalf("start scenario G target job: %v", err)
	}
	if jobID != "target-job" || !jobStarted {
		t.Fatalf("job id = %q started = %t", jobID, jobStarted)
	}
}

func TestPrepareScenarioGColdTargetStopsDaemonBeforeRelease(t *testing.T) {
	events := make([]string, 0, 2)
	err := prepareScenarioGColdTarget(
		context.Background(),
		&daemonRuntime{},
		"hybrid_code_chunks_target",
		func(*daemonRuntime) error {
			events = append(events, "stop")
			return nil
		},
		func(_ context.Context, collectionName string) error {
			if len(events) != 1 || events[0] != "stop" {
				t.Fatal("collection released before the daemon stopped")
			}
			events = append(events, "release:"+collectionName)
			return nil
		},
	)
	if err != nil {
		t.Fatalf("prepare scenario G cold target: %v", err)
	}
	want := []string{"stop", "release:hybrid_code_chunks_target"}
	if !slices.Equal(events, want) {
		t.Fatalf("events = %v, want %v", events, want)
	}
}

func TestSnapshotRowCountUsesStrongRowsForDetailedCollection(t *testing.T) {
	rows := []rowStateObservation{{Identity: "conversation-row"}}
	if got := snapshotRowCount(0, rows, true); got != 1 {
		t.Fatalf("detailed row count = %d, want 1", got)
	}
	if got := snapshotRowCount(7, nil, false); got != 7 {
		t.Fatalf("summary row count = %d, want 7", got)
	}
}

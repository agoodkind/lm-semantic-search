//go:build live

package live

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/test/sandboxharness"
)

const codebaseRestartChildEnvironment = "LMS_CODEBASE_RESTART_CHILD"

type codebaseRestartRequest struct {
	Config config.Config
	Ready  string
}

func TestCodebaseRestartProcess(t *testing.T) {
	encoded := os.Getenv(codebaseRestartChildEnvironment)
	if encoded == "" {
		return
	}
	var request codebaseRestartRequest
	if err := json.Unmarshal([]byte(encoded), &request); err != nil {
		t.Fatalf("decode child configuration: %v", err)
	}
	credentials, err := config.Default()
	if err != nil {
		t.Fatalf("resolve child credentials: %v", err)
	}
	request.Config.OpenAIAPIKey = credentials.OpenAIAPIKey
	request.Config.MilvusToken = credentials.MilvusToken
	ctx, cancel := context.WithTimeout(context.Background(), libraryLiveTimeout)
	defer cancel()
	manager, err := daemon.NewManager(ctx, request.Config)
	if err != nil {
		t.Fatalf("open child daemon: %v", err)
	}
	stopServer := startInProcessServer(t, ctx, manager, request.Config.SocketPath)
	defer stopServer()
	defer func() { _ = manager.Close(context.Background()) }()
	if err := os.WriteFile(request.Ready, []byte("ready"), 0o600); err != nil {
		t.Fatalf("publish child readiness: %v", err)
	}
	<-ctx.Done()
	t.Fatal("the parent did not terminate the child within its deadline")
}

func TestLibraryCodebaseRecoversProcessBeforeAndAfterVectorPublication(t *testing.T) {
	for _, phase := range []sandboxharness.StorePublicationPhase{
		sandboxharness.BeforeStorePublication, sandboxharness.AfterStorePublication,
	} {
		t.Run(phaseName(phase), func(t *testing.T) {
			codebaseDaemon, proxy := newProxiedCodebaseDaemon(t)
			harness := codebaseDaemon.harness
			root := t.TempDir()
			writeCodebaseFile(t, root, "recovery.go", goFile(goFunction("OldVersion", "oldversionmarker")))
			codebaseDaemon.index(t, root)
			before := codebaseDaemon.readCodebaseOwners(t)["recovery.go"]
			vectorsBefore := harness.backendVectorIDs("lms_library_codebase")
			codebaseDaemon.stop()
			child := startCodebaseRestartChild(t, codebaseDaemon)
			writeCodebaseFile(t, root, "recovery.go", goFile(goFunction("NewVersion", "newversionmarker")))
			observed, release := proxy.PausePublication(harness.database, "lms_library_codebase", phase)
			t.Cleanup(release)
			codebaseDaemon.startSync(t, root)
			var boundary sandboxharness.StorePublicationObservation
			select {
			case boundary = <-observed:
			case <-harness.context().Done():
				t.Fatal("the changed codebase did not request vector publication")
			}
			if boundary.Method != "Upsert" || boundary.Database != harness.database || boundary.Collection != "lms_library_codebase" || boundary.Phase != phase {
				t.Fatalf("unexpected publication boundary: %+v", boundary)
			}
			vectorsAtBoundary := harness.backendVectorIDs("lms_library_codebase")
			if phase == sandboxharness.BeforeStorePublication {
				if boundary.ResponseReceived || !slices.Equal(vectorsBefore, vectorsAtBoundary) {
					t.Fatal("the before-publication barrier forwarded or changed backend vectors")
				}
			} else if !boundary.ResponseReceived || boundary.BackendCode != 0 || len(vectorsAtBoundary) <= len(vectorsBefore) {
				t.Fatalf("the backend did not confirm a new published vector: %+v, before=%v after=%v", boundary, vectorsBefore, vectorsAtBoundary)
			}
			if !slices.Equal(before, codebaseDaemon.readCodebaseOwners(t)["recovery.go"]) {
				t.Fatal("uncommitted vector publication changed the visible owner generation")
			}
			killCodebaseRestartChild(t, child)
			release()
			t.Logf("SIGKILL child PID=%d at phase=%s method=%s database=%s collection=%s response_received=%t backend_code=%d", child.Process.Pid, phaseName(phase), boundary.Method, boundary.Database, boundary.Collection, boundary.ResponseReceived, boundary.BackendCode)
			reopened := startCodebaseRestartChild(t, codebaseDaemon)
			if !slices.Equal(before, codebaseDaemon.readCodebaseOwners(t)["recovery.go"]) {
				t.Fatal("catalog reopen changed the committed owner generation")
			}
			requireRestartSource(t, codebaseDaemon, root, "oldversionmarker")
			codebaseDaemon.sync(t, root)
			after := codebaseDaemon.readCodebaseOwners(t)["recovery.go"]
			if len(before) != 1 || len(after) != 1 || after[0].generationOrder != before[0].generationOrder+1 {
				t.Fatalf("recovered generation before=%v after=%v", before, after)
			}
			requireRestartSource(t, codebaseDaemon, root, "newversionmarker")
			t.Logf("reopened PID=%d generation=%d; public SearchCode returned the old source before retry and new source after retry", reopened.Process.Pid, after[0].generationOrder)
			killCodebaseRestartChild(t, reopened)
		})
	}
}

func newProxiedCodebaseDaemon(t *testing.T) (*libraryCodebaseDaemon, *sandboxharness.EmbeddingStoreProxy) {
	t.Helper()
	harness := newLibraryHarness(t)
	proxy, err := sandboxharness.StartEmbeddingStoreProxy(sandboxharness.EmbeddingStoreProxyOptions{
		BackendAddress: harness.environment.MilvusAddress, Start: true,
	})
	if err != nil {
		t.Fatalf("start real Milvus forwarding proxy: %v", err)
	}
	t.Cleanup(func() {
		if err := proxy.Close(); err != nil {
			t.Errorf("close forwarding proxy: %v", err)
		}
	})
	harness.environment.MilvusAddress = proxy.Address()
	return newCodebaseLiveDaemonWithHarness(t, harness, config.CodebaseStoreLibrary), proxy
}

func phaseName(phase sandboxharness.StorePublicationPhase) string {
	if phase == sandboxharness.BeforeStorePublication {
		return "before_backend"
	}
	return "after_backend"
}

func requireRestartSource(t *testing.T, codebaseDaemon *libraryCodebaseDaemon, root string, marker string) {
	t.Helper()
	response, err := codebaseDaemon.client.SearchCode(codebaseDaemon.harness.context(), &pb.SearchCodeRequest{Path: root, Query: marker, Limit: codebaseSearchLimit})
	if err != nil || len(response.GetResults()) != 1 {
		t.Fatalf("reopened SearchCode returned %d results, error=%v", len(response.GetResults()), err)
	}
	assertExcerpts(t, codebaseDaemon.readCodebaseOwners(t), "recovery.go", response.GetResults()[0].GetContent())
	if response.GetResults()[0].GetRelativePath() != "recovery.go" {
		t.Fatalf("reopened search returned %s", response.GetResults()[0].GetRelativePath())
	}
	if !strings.Contains(response.GetResults()[0].GetContent(), marker) {
		t.Fatalf("reopened search excerpt does not contain %s", marker)
	}
}

func startCodebaseRestartChild(t *testing.T, codebaseDaemon *libraryCodebaseDaemon) *exec.Cmd {
	t.Helper()
	cfg := codebaseDaemon.config
	cfg.OpenAIAPIKey = ""
	cfg.MilvusToken = ""
	ready := filepath.Join(t.TempDir(), "ready")
	encoded, err := json.Marshal(codebaseRestartRequest{Config: cfg, Ready: ready})
	if err != nil {
		t.Fatalf("encode child configuration: %v", err)
	}
	log, err := os.Create(filepath.Join(t.TempDir(), "child.log"))
	if err != nil {
		t.Fatalf("create child log: %v", err)
	}
	t.Cleanup(func() { _ = log.Close() })
	command := exec.Command(os.Args[0], "-test.run=^TestCodebaseRestartProcess$", "-test.timeout=15m")
	command.Env = append(os.Environ(), codebaseRestartChildEnvironment+"="+string(encoded))
	command.Stdout, command.Stderr = log, log
	if err := command.Start(); err != nil {
		t.Fatalf("start daemon child: %v", err)
	}
	t.Cleanup(func() {
		if command.ProcessState == nil {
			_ = command.Process.Kill()
			_ = command.Wait()
		}
	})
	deadline := time.Now().Add(time.Minute)
	for {
		if _, err := os.Stat(ready); err == nil {
			break
		} else if !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("read child readiness: %v", err)
		}
		if time.Now().After(deadline) {
			t.Fatalf("child PID=%d did not open its daemon; log=%s", command.Process.Pid, log.Name())
		}
		time.Sleep(20 * time.Millisecond)
	}
	connection, client, err := grpcutil.DialDaemon(codebaseDaemon.harness.context(), cfg.SocketPath)
	if err != nil {
		t.Fatalf("dial reopened daemon: %v", err)
	}
	t.Cleanup(func() { _ = connection.Close() })
	codebaseDaemon.client = client
	return command
}

func killCodebaseRestartChild(t *testing.T, command *exec.Cmd) {
	t.Helper()
	if err := command.Process.Kill(); err != nil {
		t.Fatalf("SIGKILL child PID=%d: %v", command.Process.Pid, err)
	}
	exit, signal := waitLibraryChild(t, command)
	if exit != -1 || signal != syscall.SIGKILL {
		t.Fatalf("child PID=%d exit=%d signal=%v; want SIGKILL", command.Process.Pid, exit, signal)
	}
}

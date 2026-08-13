//go:build restartacceptance

package restartacceptance

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"golang.org/x/sys/unix"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func TestRunScenarioAResumesAfterEmbeddingOutageWithoutDuplicateWork(t *testing.T) {
	var backendRequests atomic.Int32
	var backendMutex sync.Mutex
	backendIDs := make([]string, 0)
	backend := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		backendRequests.Add(1)
		backendMutex.Lock()
		backendIDs = append(backendIDs, request.Header.Get("X-Acceptance-Row-ID"))
		backendMutex.Unlock()
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"data":[{"embedding":[1]}]}`))
	}))
	t.Cleanup(backend.Close)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	proxy, err := newEmbeddingProxy(listener, backend.URL)
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = proxy.Serve() }()
	t.Cleanup(func() { _ = proxy.Close() })

	server := newScenarioFakeServer()
	server.embeddingURL = "http://" + listener.Addr().String()
	client := startScenarioFakeGRPC(t, server)
	evidenceRoot := t.TempDir()
	evidencePaths := pathsForRun(evidenceRoot)
	if err := os.MkdirAll(evidencePaths.Artifacts, 0o700); err != nil {
		t.Fatal(err)
	}
	recorder := newEvidenceRecorder(evidencePaths, func() time.Time {
		return time.Date(2026, 8, 12, 0, 0, 0, 0, time.UTC)
	})
	result, err := runScenarioA(context.Background(), scenarioAInput{
		Client:                client,
		Path:                  t.TempDir(),
		Proxy:                 proxy,
		ExpectedUnfinishedIDs: []string{"row-2", "row-3"},
		ObserveRows:           server.observeRows,
		ObserveCheckpoint:     server.observeCheckpoint,
		ObserveEmbeddingCalls: func() []string {
			backendMutex.Lock()
			defer backendMutex.Unlock()
			return slices.Clone(backendIDs)
		},
		Recorder: recorder,
		Timeouts: focusedScenarioTimeouts(),
	})
	if err != nil {
		t.Fatalf("runScenarioA returned error: %v", err)
	}
	if result.FailureCode != "embedder_busy" || !result.Retryable {
		t.Fatalf("failure = %q retryable=%v, want typed retryable embedder_busy", result.FailureCode, result.Retryable)
	}
	if result.FailureElapsed > 500*time.Millisecond {
		t.Fatalf("typed failure took %s, want at most 500ms in focused test", result.FailureElapsed)
	}
	if result.Rows != 3 || result.UniqueRows != 3 {
		t.Fatalf("rows = %d unique = %d, want 3 and 3", result.Rows, result.UniqueRows)
	}
	if result.ResumedChunksEmbedded != 2 || backendRequests.Load() != 3 {
		t.Fatalf("resume embedded = %d backend requests = %d, want 2 and 3", result.ResumedChunksEmbedded, backendRequests.Load())
	}
	if !slices.Equal(result.EmbeddedAfterFault, []string{"row-2", "row-3"}) {
		t.Fatalf("embedded after fault = %v", result.EmbeddedAfterFault)
	}
	evidence, err := os.ReadFile(evidencePaths.EventsJSONL)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(evidence, []byte(`"phase":"scenario_a"`)) ||
		!bytes.Contains(evidence, []byte(`"unique_rows":"3"`)) {
		t.Fatalf("scenario A evidence = %s", evidence)
	}
	assertSingleScenarioRecord(t, evidencePaths.EventsJSONL, "scenario_a")
}

func TestRunScenarioAAllowsCloneStartupBeforeMidIngestCheckpoint(t *testing.T) {
	var backendMutex sync.Mutex
	backendIDs := make([]string, 0)
	backend := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		backendMutex.Lock()
		backendIDs = append(backendIDs, request.Header.Get("X-Acceptance-Row-ID"))
		backendMutex.Unlock()
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"data":[{"embedding":[1]}]}`))
	}))
	t.Cleanup(backend.Close)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	proxy, err := newEmbeddingProxy(listener, backend.URL)
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = proxy.Serve() }()
	t.Cleanup(func() { _ = proxy.Close() })

	server := newScenarioFakeServer()
	server.embeddingURL = "http://" + listener.Addr().String()
	server.initialCheckpointDelay = 80 * time.Millisecond
	client := startScenarioFakeGRPC(t, server)
	recorder, _ := scenarioTestRecorder(t)
	timeouts := focusedScenarioTimeouts()
	timeouts.Ready = 20 * time.Millisecond
	timeouts.Recovery = time.Second

	if _, err := runScenarioA(context.Background(), scenarioAInput{
		Client:                client,
		Path:                  t.TempDir(),
		Proxy:                 proxy,
		ExpectedUnfinishedIDs: []string{"row-2", "row-3"},
		ObserveRows:           server.observeRows,
		ObserveCheckpoint:     server.observeCheckpoint,
		ObserveEmbeddingCalls: func() []string {
			backendMutex.Lock()
			defer backendMutex.Unlock()
			return slices.Clone(backendIDs)
		},
		Recorder: recorder,
		Timeouts: timeouts,
	}); err != nil {
		t.Fatalf("runScenarioA returned error after delayed checkpoint: %v", err)
	}
}

func TestRunScenarioBReconnectsAfterMilvusOutageWithoutDaemonRestart(t *testing.T) {
	backendListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	backendServer := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(backendServer, &proxyTestMilvusServer{})
	go func() { _ = backendServer.Serve(backendListener) }()
	t.Cleanup(backendServer.Stop)

	proxyListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	proxy, err := newMilvusProxy(proxyListener, backendListener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = proxy.Serve() }()
	t.Cleanup(func() { _ = proxy.Close() })

	server := newScenarioFakeServer()
	server.milvusAddress = proxyListener.Addr().String()
	server.requireSearchBeforeIndex = true
	server.requireSyncForRebuild = true
	server.requireExplicitRecovery = true
	server.rows["row-1"] = testRow("row-1", 0, "hash-1")
	client := startScenarioFakeGRPC(t, server)
	recorder, evidencePaths := scenarioTestRecorder(t)
	result, err := runScenarioB(context.Background(), scenarioBInput{
		Client:               client,
		Path:                 t.TempDir(),
		Proxy:                proxy,
		EmbeddingGateReached: closedScenarioSignal(),
		ExpectedAddedHashes:  scenarioBAddedHashes(),
		ObserveRows:          server.observeRows,
		ObserveCheckpoint:    server.observeCheckpoint,
		Recorder:             recorder,
		Timeouts:             focusedScenarioTimeouts(),
	})
	if err != nil {
		t.Fatalf("runScenarioB returned error: %v", err)
	}
	if result.SearchCode != codes.Unavailable || result.FailureCode != "milvus_unavailable" {
		t.Fatalf("search code = %v failure = %q, want Unavailable and milvus_unavailable", result.SearchCode, result.FailureCode)
	}
	if result.UnhealthyMode != "store_unavailable" || result.DaemonPIDBefore != result.DaemonPIDAfter {
		t.Fatalf("health = %q pids = %d/%d", result.UnhealthyMode, result.DaemonPIDBefore, result.DaemonPIDAfter)
	}
	if result.FailureElapsed > 500*time.Millisecond {
		t.Fatalf("typed failure took %s, want at most 500ms in focused test", result.FailureElapsed)
	}
	assertSingleScenarioRecord(t, evidencePaths.EventsJSONL, "scenario_b")
}

func TestRunScenarioBRejectsEmptySearchAfterMilvusRecovery(t *testing.T) {
	backendListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	backendServer := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(backendServer, &proxyTestMilvusServer{})
	go func() { _ = backendServer.Serve(backendListener) }()
	t.Cleanup(backendServer.Stop)

	proxyListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	proxy, err := newMilvusProxy(proxyListener, backendListener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = proxy.Serve() }()
	t.Cleanup(func() { _ = proxy.Close() })

	server := newScenarioFakeServer()
	server.milvusAddress = proxyListener.Addr().String()
	server.requireSearchBeforeIndex = true
	server.requireSyncForRebuild = true
	server.requireExplicitRecovery = true
	server.emptyRecoveredSearch = true
	server.rows["row-1"] = testRow("row-1", 0, "hash-1")
	client := startScenarioFakeGRPC(t, server)
	recorder, _ := scenarioTestRecorder(t)
	_, err = runScenarioB(context.Background(), scenarioBInput{
		Client:               client,
		Path:                 t.TempDir(),
		Proxy:                proxy,
		EmbeddingGateReached: closedScenarioSignal(),
		ExpectedAddedHashes:  scenarioBAddedHashes(),
		ObserveRows:          server.observeRows,
		ObserveCheckpoint:    server.observeCheckpoint,
		Recorder:             recorder,
		Timeouts:             focusedScenarioTimeouts(),
	})
	if err == nil || !strings.Contains(err.Error(), "search after reconnect returned no results") {
		t.Fatalf("runScenarioB error = %v, want empty recovered search rejection", err)
	}
}

func TestRunScenarioBRejectsCompletedRowLossAfterRecovery(t *testing.T) {
	backendListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	backendServer := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(backendServer, &proxyTestMilvusServer{})
	go func() { _ = backendServer.Serve(backendListener) }()
	t.Cleanup(backendServer.Stop)
	proxyListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	proxy, err := newMilvusProxy(proxyListener, backendListener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = proxy.Serve() }()
	t.Cleanup(func() { _ = proxy.Close() })

	server := newScenarioFakeServer()
	server.milvusAddress = proxyListener.Addr().String()
	server.requireSearchBeforeIndex = true
	server.requireSyncForRebuild = true
	server.requireExplicitRecovery = true
	server.dropCompletedRowOnFailure = true
	server.rows["row-1"] = testRow("row-1", 0, "hash-1")
	client := startScenarioFakeGRPC(t, server)
	recorder, _ := scenarioTestRecorder(t)
	_, err = runScenarioB(context.Background(), scenarioBInput{
		Client:               client,
		Path:                 t.TempDir(),
		Proxy:                proxy,
		EmbeddingGateReached: closedScenarioSignal(),
		ExpectedAddedHashes:  scenarioBAddedHashes(),
		ObserveRows:          server.observeRows,
		ObserveCheckpoint:    server.observeCheckpoint,
		Recorder:             recorder,
		Timeouts:             focusedScenarioTimeouts(),
	})
	if err == nil || !strings.Contains(err.Error(), "completed row \"row-1\" changed or disappeared") {
		t.Fatalf("runScenarioB error = %v, want completed row loss rejection", err)
	}
}

func TestRunScenarioBRejectsIngestFailureAfterSharedDeadline(t *testing.T) {
	backendListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	backendServer := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(backendServer, &proxyTestMilvusServer{})
	go func() { _ = backendServer.Serve(backendListener) }()
	t.Cleanup(backendServer.Stop)
	proxyListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	proxy, err := newMilvusProxy(proxyListener, backendListener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = proxy.Serve() }()
	t.Cleanup(func() { _ = proxy.Close() })

	server := newScenarioFakeServer()
	server.milvusAddress = proxyListener.Addr().String()
	server.requireSearchBeforeIndex = true
	server.requireSyncForRebuild = true
	server.requireExplicitRecovery = true
	server.milvusJobDelay = 700 * time.Millisecond
	server.rows["row-1"] = testRow("row-1", 0, "hash-1")
	client := startScenarioFakeGRPC(t, server)
	recorder, _ := scenarioTestRecorder(t)
	started := time.Now()
	_, err = runScenarioB(context.Background(), scenarioBInput{
		Client:               client,
		Path:                 t.TempDir(),
		Proxy:                proxy,
		EmbeddingGateReached: closedScenarioSignal(),
		ExpectedAddedHashes:  scenarioBAddedHashes(),
		ObserveRows:          server.observeRows,
		ObserveCheckpoint:    server.observeCheckpoint,
		Recorder:             recorder,
		Timeouts:             focusedScenarioTimeouts(),
	})
	if err == nil || !strings.Contains(err.Error(), "wait for failed ingest") {
		t.Fatalf("runScenarioB error = %v, want bounded ingest failure", err)
	}
	if elapsed := time.Since(started); elapsed > 900*time.Millisecond {
		t.Fatalf("bounded failure took %s, want at most 900ms", elapsed)
	}
}

func TestRunScenarioBRejectsRecoveryPastSharedDeadline(t *testing.T) {
	backendListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	backendServer := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(backendServer, &proxyTestMilvusServer{})
	go func() { _ = backendServer.Serve(backendListener) }()
	t.Cleanup(backendServer.Stop)
	proxyListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	proxy, err := newMilvusProxy(proxyListener, backendListener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = proxy.Serve() }()
	t.Cleanup(func() { _ = proxy.Close() })

	server := newScenarioFakeServer()
	server.milvusAddress = proxyListener.Addr().String()
	server.requireSearchBeforeIndex = true
	server.requireSyncForRebuild = true
	server.requireExplicitRecovery = true
	server.recoveredJobDelay = 700 * time.Millisecond
	server.rows["row-1"] = testRow("row-1", 0, "hash-1")
	client := startScenarioFakeGRPC(t, server)
	recorder, _ := scenarioTestRecorder(t)
	timeouts := focusedScenarioTimeouts()
	timeouts.Recovery = 500 * time.Millisecond
	started := time.Now()
	_, err = runScenarioB(context.Background(), scenarioBInput{
		Client:               client,
		Path:                 t.TempDir(),
		Proxy:                proxy,
		EmbeddingGateReached: closedScenarioSignal(),
		ExpectedAddedHashes:  scenarioBAddedHashes(),
		ObserveRows:          server.observeRows,
		ObserveCheckpoint:    server.observeCheckpoint,
		Recorder:             recorder,
		Timeouts:             timeouts,
	})
	if err == nil || !strings.Contains(err.Error(), "wait for recovered ingest") {
		t.Fatalf("runScenarioB error = %v, want bounded recovery failure", err)
	}
	if elapsed := time.Since(started); elapsed > 1200*time.Millisecond {
		t.Fatalf("bounded recovery took %s, want at most 1.2s", elapsed)
	}
}

func TestSearchCodeWithinBoundsPublicRequest(t *testing.T) {
	server := newScenarioFakeServer()
	server.blockSearchCall = 1
	client := startScenarioFakeGRPC(t, server)
	started := time.Now()
	_, err := searchCodeWithin(context.Background(), client, t.TempDir(), 40*time.Millisecond)
	if status.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("search code = %s, want DeadlineExceeded", status.Code(err))
	}
	if elapsed := time.Since(started); elapsed > 500*time.Millisecond {
		t.Fatalf("bounded search took %s, want at most 500ms", elapsed)
	}
}

func TestRunScenarioBRejectsGateReleaseBeforeMilvusFault(t *testing.T) {
	proxy := &milvusProxy{}
	if err := releaseScenarioBEmbeddingGate(proxy, func() {}); err == nil || !strings.Contains(err.Error(), "not active") {
		t.Fatalf("release error = %v, want inactive fault rejection", err)
	}
}

func TestValidateRecoverySnapshotsRejectsCheckpointHashChange(t *testing.T) {
	rows := rowSnapshot{Entries: map[string]rowSnapshotEntry{"row-1": testRow("row-1", 0, "hash-1")}}
	before := checkpointSnapshot{FileHashes: map[string]string{"row-1": "checkpoint-before"}, CompletedCount: 1}
	after := checkpointSnapshot{FileHashes: map[string]string{"row-1": "checkpoint-after"}, CompletedCount: 1}
	err := validateRecoverySnapshotsWithHashes(rows, before, rows, after, nil)
	if err == nil || !strings.Contains(err.Error(), "checkpoint hash") {
		t.Fatalf("validateRecoverySnapshotsWithHashes error = %v, want checkpoint hash rejection", err)
	}
}

func TestCaptureRecoverySnapshotsRejectsRowWithoutCheckpoint(t *testing.T) {
	observeRows := func(context.Context) (rowSnapshot, error) {
		return rowSnapshot{Entries: map[string]rowSnapshotEntry{
			"row-1": testRow("row-1", 0, "hash-1"),
			"row-2": testRow("row-2", 1, "hash-2"),
		}}, nil
	}
	observeCheckpoint := func(context.Context) (checkpointSnapshot, error) {
		return checkpointSnapshot{FileHashes: map[string]string{"row-1": "checkpoint-row-1"}, CompletedCount: 1}, nil
	}
	_, _, err := captureRecoverySnapshots(context.Background(), observeRows, observeCheckpoint)
	if err == nil || !strings.Contains(err.Error(), "identities differ") {
		t.Fatalf("captureRecoverySnapshots error = %v, want identity mismatch", err)
	}
}

func TestRunScenarioCKillReclaimsKernelLockAndResumesCheckpoint(t *testing.T) {
	root := shortTempDir(t)
	socketPath := filepath.Join(root, "daemon.sock")
	lockPath := filepath.Join(root, "context", "mcp-sync.flock")
	statePath := filepath.Join(root, "state")
	recorder, evidencePaths := scenarioTestRecorder(t)
	snapshots := newRecoverySnapshotSequence()
	result, err := runScenarioC(context.Background(), scenarioCInput{
		Process: installedProcess{
			Path: currentTestExecutable(t),
			Args: []string{"-test.run=TestRestartAcceptanceHelperProcess"},
			Environment: map[string]string{
				"LMS_RESTART_HELPER":        "crash-resume",
				"LMS_RESTART_HELPER_STATE":  statePath,
				"LMS_RESTART_HELPER_LOCK":   lockPath,
				"LMS_RESTART_HELPER_SOCKET": socketPath,
			},
		},
		SocketPath:            socketPath,
		LockPath:              lockPath,
		Path:                  root,
		ExpectedUnfinishedIDs: []string{"row-2", "row-3"},
		ObserveRows:           snapshots.observeRows,
		ObserveCheckpoint:     snapshots.observeCheckpoint,
		Recorder:              recorder,
		Timeouts:              focusedScenarioTimeouts(),
	})
	if err != nil {
		t.Fatalf("runScenarioC returned error: %v", err)
	}
	if !result.LockBusyBeforeKill || !result.LockReclaimedAfterKill {
		t.Fatalf("lock evidence = busy %v reclaimed %v", result.LockBusyBeforeKill, result.LockReclaimedAfterKill)
	}
	if result.CheckpointChunks != 1 || result.ResumedChunksEmbedded != 2 || result.Rows != 3 {
		t.Fatalf("checkpoint=%d resumed=%d rows=%d, want 1, 2, 3", result.CheckpointChunks, result.ResumedChunksEmbedded, result.Rows)
	}
	if result.FirstExecutable != result.RestartExecutable {
		t.Fatalf("executables differ: %q and %q", result.FirstExecutable, result.RestartExecutable)
	}
	assertSingleScenarioRecord(t, evidencePaths.EventsJSONL, "scenario_c")
}

func TestRunScenarioDIgnoresRetiredOwnerPIDArtifact(t *testing.T) {
	unrelated := exec.Command("sleep", "30")
	if err := unrelated.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = unrelated.Process.Kill()
		_, _ = unrelated.Process.Wait()
	})
	root := shortTempDir(t)
	socketPath := filepath.Join(root, "daemon.sock")
	paths := pathsForRun(root)
	if err := os.MkdirAll(paths.LMSContext, 0o700); err != nil {
		t.Fatal(err)
	}
	recorder, evidencePaths := scenarioTestRecorder(t)
	result, err := runScenarioD(context.Background(), scenarioDInput{
		Process: installedProcess{
			Path: currentTestExecutable(t),
			Args: []string{"-test.run=TestRestartAcceptanceHelperProcess"},
			Environment: map[string]string{
				"LMS_RESTART_HELPER":        "retired-lock",
				"LMS_RESTART_HELPER_SOCKET": socketPath,
			},
		},
		SocketPath: socketPath,
		Paths:      paths,
		OwnerPID:   unrelated.Process.Pid,
		Path:       root,
		Recorder:   recorder,
		Timeouts:   focusedScenarioTimeouts(),
	})
	if err != nil {
		t.Fatalf("runScenarioD returned error: %v", err)
	}
	if !result.CompletedWhileArtifactPresent || !result.UnrelatedProcessAlive {
		t.Fatalf("artifact present = %v process alive = %v", result.CompletedWhileArtifactPresent, result.UnrelatedProcessAlive)
	}
	legacyRoot := filepath.Join(paths.LMSContext, "mcp-sync.lock")
	if _, err := os.Stat(legacyRoot); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("retired artifact remains after scenario cleanup: %v", err)
	}
	if err := unrelated.Process.Signal(syscall.Signal(0)); err != nil {
		t.Fatalf("unrelated process was disturbed: %v", err)
	}
	assertSingleScenarioRecord(t, evidencePaths.EventsJSONL, "scenario_d")
}

func TestWaitForSuccessorRejectsStaleAndUntimestampedJobs(t *testing.T) {
	server := newScenarioFakeServer()
	activation := time.Date(2026, 8, 12, 1, 0, 0, 0, time.UTC)
	server.jobs = []*pb.Job{
		{Id: "old-completed", CodebaseId: "cb-1", State: "completed", StartedAt: timestamppb.New(activation.Add(-time.Minute)), UpdatedAt: timestamppb.New(activation.Add(-time.Minute))},
		{Id: "missing-time", CodebaseId: "cb-1", State: "completed"},
	}
	client := startScenarioFakeGRPC(t, server)
	preFault, err := captureJobSet(context.Background(), client, "cb-1")
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		time.Sleep(50 * time.Millisecond)
		server.mutex.Lock()
		createdAt := activation.Add(time.Second)
		server.jobs = append(server.jobs, &pb.Job{Id: "new-job", CodebaseId: "cb-1", State: "running", StartedAt: timestamppb.New(createdAt), UpdatedAt: timestamppb.New(createdAt)})
		server.mutex.Unlock()
		time.Sleep(50 * time.Millisecond)
		server.mutex.Lock()
		server.jobs[len(server.jobs)-1] = &pb.Job{Id: "new-job", CodebaseId: "cb-1", State: "completed", StartedAt: timestamppb.New(createdAt), UpdatedAt: timestamppb.New(createdAt.Add(time.Second))}
		server.mutex.Unlock()
	}()
	job, err := waitForSuccessor(context.Background(), client, "cb-1", preFault, activation, time.Second, 10*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	if job.GetId() != "new-job" {
		t.Fatalf("successor = %q, want new-job", job.GetId())
	}
}

func TestWaitForSuccessorRequiresObservedNonterminalState(t *testing.T) {
	server := newScenarioFakeServer()
	activation := time.Date(2026, 8, 12, 1, 0, 0, 0, time.UTC)
	createdAt := activation.Add(time.Second)
	server.jobs = []*pb.Job{{
		Id:         "completed-without-observation",
		CodebaseId: "cb-1",
		State:      "completed",
		StartedAt:  timestamppb.New(createdAt),
		UpdatedAt:  timestamppb.New(createdAt),
	}}
	client := startScenarioFakeGRPC(t, server)
	_, err := waitForSuccessor(
		context.Background(),
		client,
		"cb-1",
		map[string]struct{}{},
		activation,
		75*time.Millisecond,
		10*time.Millisecond,
	)
	if err == nil {
		t.Fatal("completed successor passed without an observed nonterminal state")
	}
}

func TestScenariosRejectMissingRecorderBeforeFaultActivation(t *testing.T) {
	proxy := &embeddingProxy{}
	client := startScenarioFakeGRPC(t, newScenarioFakeServer())
	_, err := runScenarioA(context.Background(), scenarioAInput{Client: client, Proxy: proxy})
	if err == nil || !strings.Contains(err.Error(), "evidence recorder") {
		t.Fatalf("scenario A error = %v", err)
	}
	milvus := &milvusProxy{}
	_, err = runScenarioB(context.Background(), scenarioBInput{Client: client, Proxy: milvus})
	if err == nil || !strings.Contains(err.Error(), "evidence recorder") {
		t.Fatalf("scenario B error = %v", err)
	}
	_, err = runScenarioC(context.Background(), scenarioCInput{Process: installedProcess{Path: "unused"}})
	if err == nil || !strings.Contains(err.Error(), "evidence recorder") {
		t.Fatalf("scenario C error = %v", err)
	}
	_, err = runScenarioD(context.Background(), scenarioDInput{Process: installedProcess{Path: "unused"}})
	if err == nil || !strings.Contains(err.Error(), "evidence recorder") {
		t.Fatalf("scenario D error = %v", err)
	}
}

func TestRunScenarioDRejectsEscapedAndSymlinkedContextRoots(t *testing.T) {
	recorder, _ := scenarioTestRecorder(t)
	runRoot := shortTempDir(t)
	outside := shortTempDir(t)
	paths := pathsForRun(runRoot)
	paths.LMSContext = outside
	_, err := runScenarioD(context.Background(), scenarioDInput{Process: installedProcess{Path: "unused"}, Paths: paths, OwnerPID: os.Getpid(), Recorder: recorder})
	if err == nil || !strings.Contains(err.Error(), "exact isolated context") {
		t.Fatalf("escaped context error = %v", err)
	}
	symlinkTarget := filepath.Join(runRoot, "target")
	if err := os.MkdirAll(symlinkTarget, 0o700); err != nil {
		t.Fatal(err)
	}
	symlinkPath := pathsForRun(runRoot).LMSContext
	if err := os.MkdirAll(filepath.Dir(symlinkPath), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(symlinkTarget, symlinkPath); err != nil {
		t.Fatal(err)
	}
	paths = pathsForRun(runRoot)
	_, err = runScenarioD(context.Background(), scenarioDInput{Process: installedProcess{Path: "unused"}, Paths: paths, OwnerPID: os.Getpid(), Recorder: recorder})
	if err == nil || !strings.Contains(err.Error(), "symlink") {
		t.Fatalf("symlink context error = %v", err)
	}
}

func focusedScenarioTimeouts() scenarioTimeouts {
	return scenarioTimeouts{
		Failure:  500 * time.Millisecond,
		Recovery: 3 * time.Second,
		Ready:    3 * time.Second,
		Poll:     10 * time.Millisecond,
	}
}

func closedScenarioSignal() <-chan struct{} {
	reached := make(chan struct{})
	close(reached)
	return reached
}

func currentTestExecutable(t *testing.T) string {
	t.Helper()
	path, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return path
}

func shortTempDir(t *testing.T) string {
	t.Helper()
	root, err := os.MkdirTemp("/tmp", "lms-ra-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	return root
}

func scenarioTestRecorder(t *testing.T) (*evidenceRecorder, runPaths) {
	t.Helper()
	paths := pathsForRun(t.TempDir())
	if err := os.MkdirAll(paths.Artifacts, 0o700); err != nil {
		t.Fatal(err)
	}
	return newEvidenceRecorder(paths, func() time.Time {
		return time.Date(2026, 8, 12, 0, 0, 0, 0, time.UTC)
	}), paths
}

func assertSingleScenarioRecord(t *testing.T, path string, phase string) {
	t.Helper()
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Count(body, []byte(`"phase":"`+phase+`"`)) != 1 {
		t.Fatalf("%s records = %s", phase, body)
	}
}

type recoverySnapshotSequence struct {
	mutex           sync.Mutex
	rowCalls        int
	checkpointCalls int
}

func newRecoverySnapshotSequence() *recoverySnapshotSequence {
	return &recoverySnapshotSequence{}
}

func (sequence *recoverySnapshotSequence) observeRows(context.Context) (rowSnapshot, error) {
	sequence.mutex.Lock()
	defer sequence.mutex.Unlock()
	sequence.rowCalls++
	rows := map[string]rowSnapshotEntry{
		"row-1": testRow("row-1", 0, "hash-1"),
	}
	if sequence.rowCalls > 1 {
		rows["row-2"] = testRow("row-2", 1, "hash-2")
		rows["row-3"] = testRow("row-3", 2, "hash-3")
	}
	return rowSnapshot{Entries: rows}, nil
}

func (sequence *recoverySnapshotSequence) observeCheckpoint(context.Context) (checkpointSnapshot, error) {
	sequence.mutex.Lock()
	defer sequence.mutex.Unlock()
	sequence.checkpointCalls++
	tracked := map[string]string{"row-1": "checkpoint-row-1"}
	if sequence.checkpointCalls > 1 {
		tracked["row-2"] = "checkpoint-row-2"
		tracked["row-3"] = "checkpoint-row-3"
	}
	return checkpointSnapshot{FileHashes: tracked, CompletedCount: len(tracked)}, nil
}

func testRow(id string, splitPosition int, vectorHash string) rowSnapshotEntry {
	return rowSnapshotEntry{
		ID:              id,
		RelativePath:    id + ".go",
		StartLine:       splitPosition*10 + 1,
		EndLine:         splitPosition*10 + 5,
		SplitPosition:   splitPosition,
		EmbeddingModel:  "acceptance-model",
		DenseVectorHash: vectorHash,
	}
}

type scenarioFakeServer struct {
	pb.UnimplementedSemanticSearchDaemonServiceServer
	mutex                     sync.Mutex
	embeddingURL              string
	milvusAddress             string
	jobs                      []*pb.Job
	rows                      map[string]rowSnapshotEntry
	health                    string
	startedAt                 time.Time
	initialCheckpointDelay    time.Duration
	requireSearchBeforeIndex  bool
	requireSyncForRebuild     bool
	requireExplicitRecovery   bool
	emptyRecoveredSearch      bool
	allowInitialSeed          bool
	orchestrateScenarioB      bool
	dropCompletedRowOnFailure bool
	blockSearchCall           int
	searchCalls               int
	milvusJobDelay            time.Duration
	recoveredJobDelay         time.Duration
	checkpointRoot            string
	expectedPath              string
	scenarioBEmbeddingDelay   time.Duration
	preflightSearches         int
}

func newScenarioFakeServer() *scenarioFakeServer {
	return &scenarioFakeServer{rows: make(map[string]rowSnapshotEntry), startedAt: time.Now()}
}

func (server *scenarioFakeServer) StartIndex(_ context.Context, request *pb.StartIndexRequest) (*pb.StartIndexResponse, error) {
	server.mutex.Lock()
	jobCount := len(server.jobs)
	server.mutex.Unlock()
	if server.requireSyncForRebuild && (!server.allowInitialSeed || jobCount > 0) {
		return nil, status.Error(codes.FailedPrecondition, "scenario B rebuild used StartIndex instead of SyncIndex")
	}
	return server.startJob(request.GetPath())
}

func (server *scenarioFakeServer) SyncIndex(_ context.Context, request *pb.SyncIndexRequest) (*pb.SyncIndexResponse, error) {
	started, err := server.startJob(request.GetPath())
	if err != nil {
		return nil, err
	}
	return &pb.SyncIndexResponse{JobId: started.GetJobId(), CodebaseId: started.GetCodebaseId(), State: started.GetState()}, nil
}

func (server *scenarioFakeServer) startJob(path string) (*pb.StartIndexResponse, error) {
	server.mutex.Lock()
	if server.expectedPath != "" && path != server.expectedPath {
		server.mutex.Unlock()
		return nil, status.Errorf(codes.InvalidArgument, "scenario B path = %q, want %q", path, server.expectedPath)
	}
	if server.requireSearchBeforeIndex && server.preflightSearches == 0 && !(server.allowInitialSeed && len(server.jobs) == 0) {
		server.mutex.Unlock()
		return nil, status.Error(codes.FailedPrecondition, "scenario B target was not searchable before ingest")
	}
	now := time.Now()
	jobID := fmt.Sprintf("job-%d", len(server.jobs)+1)
	job := &pb.Job{Id: jobID, CodebaseId: "cb-1", State: "running", Progress: &pb.Progress{}, StartedAt: timestamppb.New(now), UpdatedAt: timestamppb.New(now)}
	server.jobs = append(server.jobs, job)
	jobNumber := len(server.jobs)
	orchestrateScenarioB := server.orchestrateScenarioB
	if server.initialCheckpointDelay == 0 && !(server.dropCompletedRowOnFailure && len(server.jobs) > 1) {
		job.Progress.ChunksEmbedded = 1
		server.rows["row-1"] = testRow("row-1", 0, "hash-1")
	}
	result := &pb.StartIndexResponse{JobId: job.Id, CodebaseId: job.CodebaseId, State: job.State}
	server.mutex.Unlock()
	if orchestrateScenarioB {
		if jobNumber == 1 {
			response, err := postScenarioBEmbedding(server.embeddingURL, "01.go")
			if err != nil {
				return nil, fmt.Errorf("seed scenario B embedding: %w", err)
			}
			if response != nil {
				_ = response.Body.Close()
			}
			if response == nil || response.StatusCode != http.StatusOK {
				return nil, fmt.Errorf("seed scenario B embedding returned an invalid response")
			}
			go server.runMilvusJob(job)
			return result, nil
		}
		go server.runScenarioBEmbeddingJob(job, path, jobNumber)
		return result, nil
	}
	if server.embeddingURL != "" {
		if server.initialCheckpointDelay > 0 {
			go server.startEmbeddingJobAfterDelay(job)
			return result, nil
		}
		response, err := postEmbedding(server.embeddingURL, "row-1")
		if err != nil {
			return nil, err
		}
		_ = response.Body.Close()
		go server.runEmbeddingJob(job)
	}
	if server.milvusAddress != "" {
		go server.runMilvusJob(job)
	}
	return result, nil
}

func (server *scenarioFakeServer) runScenarioBEmbeddingJob(job *pb.Job, path string, jobNumber int) {
	if server.scenarioBEmbeddingDelay > 0 {
		time.Sleep(server.scenarioBEmbeddingDelay)
	}
	ids := []string{"04.go"}
	if jobNumber > 2 {
		ids = append(ids, "05.go")
	}
	for _, id := range ids {
		if _, err := os.Stat(filepath.Join(path, id)); err != nil {
			server.failScenarioBJob(job, "fixture_missing")
			return
		}
		response, err := postScenarioBEmbedding(server.embeddingURL, id)
		if response != nil {
			_ = response.Body.Close()
		}
		if err != nil || response == nil || response.StatusCode != http.StatusOK {
			server.failScenarioBJob(job, "embedder_busy")
			return
		}
	}
	server.completeScenarioBMilvusJob(job, ids)
}

func (server *scenarioFakeServer) completeScenarioBMilvusJob(job *pb.Job, ids []string) {
	if callMilvusLoadState(server.milvusAddress) != nil {
		server.failScenarioBJob(job, "milvus_unavailable")
		return
	}
	server.mutex.Lock()
	defer server.mutex.Unlock()
	server.health = ""
	for index, id := range ids {
		server.rows[id] = testRow(id, index+1, fmt.Sprintf("hash-%d", index+2))
	}
	job.Progress.ChunksEmbedded = int32(len(ids))
	job.State = "completed"
	job.UpdatedAt = timestamppb.Now()
}

func (server *scenarioFakeServer) failScenarioBJob(job *pb.Job, code string) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	job.State = "failed"
	job.Error = &pb.JobError{Code: code, Retryable: true}
	if code == "milvus_unavailable" {
		server.health = "store_unavailable"
	}
}

func (server *scenarioFakeServer) startEmbeddingJobAfterDelay(job *pb.Job) {
	time.Sleep(server.initialCheckpointDelay)
	response, err := postEmbedding(server.embeddingURL, "row-1")
	if response != nil {
		_ = response.Body.Close()
	}
	if err != nil || response == nil || response.StatusCode != http.StatusOK {
		return
	}
	server.recordInitialCheckpoint(job)
	server.runEmbeddingJob(job)
}

func (server *scenarioFakeServer) recordInitialCheckpoint(job *pb.Job) {
	server.mutex.Lock()
	job.Progress.ChunksEmbedded = 1
	job.UpdatedAt = timestamppb.Now()
	server.rows["row-1"] = testRow("row-1", 0, "hash-1")
	server.mutex.Unlock()
}

func (server *scenarioFakeServer) runEmbeddingJob(job *pb.Job) {
	time.Sleep(80 * time.Millisecond)
	response, err := postEmbedding(server.embeddingURL, "row-2")
	if response != nil {
		_ = response.Body.Close()
	}
	server.mutex.Lock()
	defer server.mutex.Unlock()
	if err != nil || response == nil || response.StatusCode != http.StatusOK {
		job.State = "failed"
		job.Error = &pb.JobError{Code: "embedder_busy", Retryable: true}
		go server.resumeEmbeddingWhenHealthy()
		return
	}
}

func (server *scenarioFakeServer) resumeEmbeddingWhenHealthy() {
	for {
		time.Sleep(10 * time.Millisecond)
		response, err := postEmbedding(server.embeddingURL, "row-2")
		if response != nil {
			_ = response.Body.Close()
		}
		if err != nil || response == nil || response.StatusCode != http.StatusOK {
			continue
		}
		response, err = postEmbedding(server.embeddingURL, "row-3")
		if response != nil {
			_ = response.Body.Close()
		}
		if err != nil || response == nil || response.StatusCode != http.StatusOK {
			continue
		}
		server.mutex.Lock()
		server.rows["row-2"] = testRow("row-2", 1, "hash-2")
		server.rows["row-3"] = testRow("row-3", 2, "hash-3")
		now := time.Now()
		server.jobs = append(server.jobs, &pb.Job{Id: "job-2", CodebaseId: "cb-1", State: "running", Progress: &pb.Progress{ChunksEmbedded: 2}, StartedAt: timestamppb.New(now), UpdatedAt: timestamppb.New(now)})
		server.mutex.Unlock()
		time.Sleep(30 * time.Millisecond)
		server.mutex.Lock()
		server.jobs[len(server.jobs)-1] = &pb.Job{Id: "job-2", CodebaseId: "cb-1", State: "completed", Progress: &pb.Progress{ChunksEmbedded: 2}, StartedAt: timestamppb.New(now), UpdatedAt: timestamppb.New(time.Now())}
		server.mutex.Unlock()
		return
	}
}

func (server *scenarioFakeServer) runMilvusJob(job *pb.Job) {
	delay := server.milvusJobDelay
	if job.GetId() != "job-1" && server.recoveredJobDelay > 0 {
		delay = server.recoveredJobDelay
	}
	if delay <= 0 {
		delay = 80 * time.Millisecond
	}
	time.Sleep(delay)
	if err := callMilvusLoadState(server.milvusAddress); err == nil {
		server.mutex.Lock()
		server.health = ""
		if job.GetId() != "job-1" {
			server.rows["04.go"] = testRow("04.go", 1, "hash-2")
			server.rows["05.go"] = testRow("05.go", 2, "hash-3")
			job.Progress.ChunksEmbedded = 2
		}
		job.State = "completed"
		job.UpdatedAt = timestamppb.Now()
		server.mutex.Unlock()
		return
	}
	server.mutex.Lock()
	job.State = "failed"
	job.Error = &pb.JobError{Code: "milvus_unavailable", Retryable: true}
	server.health = "store_unavailable"
	if server.dropCompletedRowOnFailure {
		delete(server.rows, "row-1")
	}
	explicitRecovery := server.requireExplicitRecovery
	server.mutex.Unlock()
	if explicitRecovery {
		return
	}
	go server.resumeMilvusWhenHealthy()
}

func (server *scenarioFakeServer) resumeMilvusWhenHealthy() {
	for {
		time.Sleep(10 * time.Millisecond)
		if callMilvusLoadState(server.milvusAddress) != nil {
			continue
		}
		server.mutex.Lock()
		server.health = ""
		now := time.Now()
		server.jobs = append(server.jobs, &pb.Job{Id: "job-2", CodebaseId: "cb-1", State: "running", Progress: &pb.Progress{ChunksEmbedded: 2}, StartedAt: timestamppb.New(now), UpdatedAt: timestamppb.New(now)})
		server.mutex.Unlock()
		time.Sleep(30 * time.Millisecond)
		server.mutex.Lock()
		server.jobs[len(server.jobs)-1] = &pb.Job{Id: "job-2", CodebaseId: "cb-1", State: "completed", Progress: &pb.Progress{ChunksEmbedded: 2}, StartedAt: timestamppb.New(now), UpdatedAt: timestamppb.New(time.Now())}
		server.mutex.Unlock()
		return
	}
}

func callMilvusLoadState(address string) error {
	connection, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return err
	}
	defer func() { _ = connection.Close() }()
	_, err = milvuspb.NewMilvusServiceClient(connection).GetLoadState(context.Background(), &milvuspb.GetLoadStateRequest{DbName: "default", CollectionName: "case"})
	return err
}

func (server *scenarioFakeServer) GetJob(_ context.Context, request *pb.GetJobRequest) (*pb.GetJobResponse, error) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	for _, job := range server.jobs {
		if job.Id == request.JobId {
			return &pb.GetJobResponse{Job: proto.Clone(job).(*pb.Job), DependencyHealth: &pb.DependencyHealth{Degraded: server.health != "", Mode: server.health}}, nil
		}
	}
	return nil, status.Error(codes.NotFound, "job not found")
}

func (server *scenarioFakeServer) ListJobs(_ context.Context, _ *pb.ListJobsRequest) (*pb.ListJobsResponse, error) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	jobs := make([]*pb.Job, 0, len(server.jobs))
	for _, job := range server.jobs {
		jobs = append(jobs, proto.Clone(job).(*pb.Job))
	}
	return &pb.ListJobsResponse{Jobs: jobs, DependencyHealth: &pb.DependencyHealth{Degraded: server.health != "", Mode: server.health}}, nil
}

func (server *scenarioFakeServer) SearchCode(ctx context.Context, _ *pb.SearchCodeRequest) (*pb.SearchCodeResponse, error) {
	server.mutex.Lock()
	server.searchCalls++
	blockSearch := server.searchCalls == server.blockSearchCall
	if blockSearch {
		server.mutex.Unlock()
		<-ctx.Done()
		return nil, status.FromContextError(ctx.Err()).Err()
	}
	if server.requireSearchBeforeIndex && server.preflightSearches == 0 {
		server.preflightSearches++
		server.mutex.Unlock()
		return &pb.SearchCodeResponse{Results: []*pb.SearchResult{{RelativePath: "01.go"}}}, nil
	}
	server.mutex.Unlock()
	if server.milvusAddress != "" {
		if err := callMilvusLoadState(server.milvusAddress); err != nil {
			server.mutex.Lock()
			server.health = "store_unavailable"
			server.mutex.Unlock()
			return nil, status.Error(codes.Unavailable, "vector store is unavailable")
		}
		server.mutex.Lock()
		emptyRecoveredSearch := server.emptyRecoveredSearch && len(server.jobs) > 1
		server.health = ""
		server.mutex.Unlock()
		if emptyRecoveredSearch {
			return &pb.SearchCodeResponse{}, nil
		}
	}
	return &pb.SearchCodeResponse{Results: []*pb.SearchResult{{RelativePath: "01.go"}}}, nil
}

func (server *scenarioFakeServer) GetIndex(_ context.Context, _ *pb.GetIndexRequest) (*pb.GetIndexResponse, error) {
	if server.milvusAddress != "" {
		_ = callMilvusLoadState(server.milvusAddress)
	}
	server.mutex.Lock()
	defer server.mutex.Unlock()
	return &pb.GetIndexResponse{Tracked: true, DependencyHealth: &pb.DependencyHealth{Degraded: server.health != "", Mode: server.health}}, nil
}

func (server *scenarioFakeServer) GetStatus(_ context.Context, _ *pb.GetStatusRequest) (*pb.GetStatusResponse, error) {
	return &pb.GetStatusResponse{Daemon: &pb.DaemonIdentity{Pid: 4242}}, nil
}

func (server *scenarioFakeServer) observeRows(context.Context) (rowSnapshot, error) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	entries := make(map[string]rowSnapshotEntry, len(server.rows))
	for id, entry := range server.rows {
		entries[id] = entry
	}
	return rowSnapshot{Entries: entries}, nil
}

func (server *scenarioFakeServer) observeCheckpoint(context.Context) (checkpointSnapshot, error) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	tracked := make(map[string]string, len(server.rows))
	for id := range server.rows {
		tracked[id] = "checkpoint-" + id
		if server.checkpointRoot != "" {
			body, err := os.ReadFile(filepath.Join(server.checkpointRoot, id))
			if err == nil {
				digest := sha256.Sum256(body)
				tracked[id] = hex.EncodeToString(digest[:])
			}
		}
	}
	return checkpointSnapshot{FileHashes: tracked, CompletedCount: len(tracked)}, nil
}

func scenarioBAddedHashes() map[string]string {
	return map[string]string{"04.go": "checkpoint-04.go", "05.go": "checkpoint-05.go"}
}

func postEmbedding(baseURL string, id string) (*http.Response, error) {
	request, err := http.NewRequest(http.MethodPost, baseURL, http.NoBody)
	if err != nil {
		return nil, err
	}
	request.Header.Set("X-Acceptance-Row-ID", id)
	return http.DefaultClient.Do(request)
}

func postScenarioBEmbedding(baseURL string, id string) (*http.Response, error) {
	body := strings.NewReader(`{"input":"restart_acceptance_id:` + id + `"}`)
	request, err := http.NewRequest(http.MethodPost, baseURL, body)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	return http.DefaultClient.Do(request)
}

func startScenarioFakeGRPC(t *testing.T, implementation pb.SemanticSearchDaemonServiceServer) pb.SemanticSearchDaemonServiceClient {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := grpc.NewServer()
	pb.RegisterSemanticSearchDaemonServiceServer(server, implementation)
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(server.Stop)
	connection, err := grpc.NewClient(listener.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = connection.Close() })
	return pb.NewSemanticSearchDaemonServiceClient(connection)
}

func TestRestartAcceptanceHelperProcess(t *testing.T) {
	mode := os.Getenv("LMS_RESTART_HELPER")
	if mode == "" {
		return
	}
	socketPath := os.Getenv("LMS_RESTART_HELPER_SOCKET")
	_ = os.Remove(socketPath)
	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		os.Exit(10)
	}
	server := grpc.NewServer()
	helper := &helperProcessServer{mode: mode, statePath: os.Getenv("LMS_RESTART_HELPER_STATE")}
	if mode == "crash-resume" {
		lockPath := os.Getenv("LMS_RESTART_HELPER_LOCK")
		if err := os.MkdirAll(filepath.Dir(lockPath), 0o700); err != nil {
			os.Exit(11)
		}
		file, openErr := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o600)
		if openErr != nil || unix.Flock(int(file.Fd()), unix.LOCK_EX) != nil {
			os.Exit(12)
		}
		helper.lock = file
		if _, statErr := os.Stat(helper.statePath); statErr == nil {
			helper.resumed = true
		}
	}
	pb.RegisterSemanticSearchDaemonServiceServer(server, helper)
	if err := server.Serve(listener); err != nil {
		os.Exit(13)
	}
}

type helperProcessServer struct {
	pb.UnimplementedSemanticSearchDaemonServiceServer
	mode           string
	statePath      string
	lock           *os.File
	resumed        bool
	successorPolls atomic.Int32
}

func (server *helperProcessServer) StartIndex(_ context.Context, _ *pb.StartIndexRequest) (*pb.StartIndexResponse, error) {
	if server.mode == "crash-resume" && !server.resumed {
		_ = os.WriteFile(server.statePath, []byte("1"), 0o600)
	}
	return &pb.StartIndexResponse{JobId: "job-1", CodebaseId: "cb-1", State: "running"}, nil
}

func (server *helperProcessServer) GetJob(_ context.Context, request *pb.GetJobRequest) (*pb.GetJobResponse, error) {
	if request.JobId == "job-1" {
		state := "running"
		if server.resumed {
			state = "cancelled"
		}
		if server.mode == "retired-lock" {
			state = "completed"
		}
		return &pb.GetJobResponse{Job: &pb.Job{Id: "job-1", CodebaseId: "cb-1", State: state, Progress: &pb.Progress{ChunksEmbedded: 1}}}, nil
	}
	return nil, status.Error(codes.NotFound, "job not found")
}

func (server *helperProcessServer) ListJobs(_ context.Context, _ *pb.ListJobsRequest) (*pb.ListJobsResponse, error) {
	startedAt := time.Now().Add(-time.Minute)
	jobs := []*pb.Job{
		{Id: "stale-completed", CodebaseId: "cb-1", State: "completed", StartedAt: timestamppb.New(startedAt), UpdatedAt: timestamppb.New(startedAt)},
		{Id: "job-1", CodebaseId: "cb-1", State: "running", Progress: &pb.Progress{ChunksEmbedded: 1}, StartedAt: timestamppb.New(startedAt), UpdatedAt: timestamppb.New(time.Now())},
	}
	if server.resumed {
		jobs[1].State = "cancelled"
		now := time.Now()
		successorState := "running"
		if server.successorPolls.Add(1) > 1 {
			successorState = "completed"
		}
		jobs = append(jobs, &pb.Job{Id: "job-2", CodebaseId: "cb-1", State: successorState, Progress: &pb.Progress{ChunksEmbedded: 2, CollectionRowsWritten: 3}, StartedAt: timestamppb.New(now), UpdatedAt: timestamppb.New(now)})
	}
	if server.mode == "retired-lock" {
		jobs[1].State = "completed"
	}
	return &pb.ListJobsResponse{Jobs: jobs, DependencyHealth: &pb.DependencyHealth{}}, nil
}

func (server *helperProcessServer) GetStatus(_ context.Context, _ *pb.GetStatusRequest) (*pb.GetStatusResponse, error) {
	pid, _ := strconv.Atoi(strconv.Itoa(os.Getpid()))
	return &pb.GetStatusResponse{Daemon: &pb.DaemonIdentity{Pid: int32(pid)}}, nil
}

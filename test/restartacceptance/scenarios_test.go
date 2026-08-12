//go:build restartacceptance

package restartacceptance

import (
	"bytes"
	"context"
	"errors"
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
	client := startScenarioFakeGRPC(t, server)
	recorder, evidencePaths := scenarioTestRecorder(t)
	result, err := runScenarioB(context.Background(), scenarioBInput{
		Client:   client,
		Path:     t.TempDir(),
		Proxy:    proxy,
		Recorder: recorder,
		Timeouts: focusedScenarioTimeouts(),
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
	tracked := map[string]struct{}{"row-1": {}}
	if sequence.checkpointCalls > 1 {
		tracked["row-2"] = struct{}{}
		tracked["row-3"] = struct{}{}
	}
	return checkpointSnapshot{TrackedIDs: tracked, CompletedCount: len(tracked)}, nil
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
	mutex         sync.Mutex
	embeddingURL  string
	milvusAddress string
	jobs          []*pb.Job
	rows          map[string]rowSnapshotEntry
	health        string
	startedAt     time.Time
}

func newScenarioFakeServer() *scenarioFakeServer {
	return &scenarioFakeServer{rows: make(map[string]rowSnapshotEntry), startedAt: time.Now()}
}

func (server *scenarioFakeServer) StartIndex(_ context.Context, _ *pb.StartIndexRequest) (*pb.StartIndexResponse, error) {
	server.mutex.Lock()
	now := time.Now()
	job := &pb.Job{Id: "job-1", CodebaseId: "cb-1", State: "running", Progress: &pb.Progress{ChunksEmbedded: 1}, StartedAt: timestamppb.New(now), UpdatedAt: timestamppb.New(now)}
	server.jobs = append(server.jobs, job)
	server.rows["row-1"] = testRow("row-1", 0, "hash-1")
	result := &pb.StartIndexResponse{JobId: job.Id, CodebaseId: job.CodebaseId, State: job.State}
	server.mutex.Unlock()
	if server.embeddingURL != "" {
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
	time.Sleep(80 * time.Millisecond)
	if err := callMilvusLoadState(server.milvusAddress); err == nil {
		return
	}
	server.mutex.Lock()
	job.State = "failed"
	job.Error = &pb.JobError{Code: "milvus_unavailable", Retryable: true}
	server.health = "store_unavailable"
	server.mutex.Unlock()
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

func (server *scenarioFakeServer) SearchCode(_ context.Context, _ *pb.SearchCodeRequest) (*pb.SearchCodeResponse, error) {
	if server.milvusAddress != "" {
		if err := callMilvusLoadState(server.milvusAddress); err != nil {
			server.mutex.Lock()
			server.health = "store_unavailable"
			server.mutex.Unlock()
			return nil, status.Error(codes.Unavailable, "vector store is unavailable")
		}
	}
	return &pb.SearchCodeResponse{}, nil
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
	tracked := make(map[string]struct{}, len(server.rows))
	for id := range server.rows {
		tracked[id] = struct{}{}
	}
	return checkpointSnapshot{TrackedIDs: tracked, CompletedCount: len(tracked)}, nil
}

func postEmbedding(baseURL string, id string) (*http.Response, error) {
	request, err := http.NewRequest(http.MethodPost, baseURL, http.NoBody)
	if err != nil {
		return nil, err
	}
	request.Header.Set("X-Acceptance-Row-ID", id)
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

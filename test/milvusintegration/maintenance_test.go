//go:build milvusintegration

package milvusintegration

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	maintenanceMilvusMemory  = "4g"
	maintenanceGateName      = "lmstest_maintenance_gate"
	maintenanceGateRows      = 2_000
	maintenanceGateDimension = 1024
	maintenanceReason        = "milvus restore"
	// maintenanceSweepIntervalMS is the periodic sweep interval for the daemon
	// under test; the sweep's fixed initial delay is five seconds on top of it.
	maintenanceSweepIntervalMS = "1000"
	// maintenanceSweepWindow is how long the test lets the periodic sweep run
	// while the mode is on: the five second initial delay plus several ticks.
	maintenanceSweepWindow = 9 * time.Second
	// maintenanceIdleTimeoutMS releases a searched collection after two idle
	// seconds, so the maintenance phase starts from a cold collection without
	// the test reaching around the daemon to release it.
	maintenanceIdleTimeoutMS = "2000"
	maintenanceJobWait       = 3 * time.Minute
	maintenanceJobPoll       = 250 * time.Millisecond
	maintenanceSearchLimit   = 10
	sourceFileMode           = 0o644
	sourceDirMode            = 0o755
	// maintenanceAlphaFile and maintenanceBetaFile are the codebase files the
	// test writes: alpha before the first build, beta while maintenance is on.
	maintenanceAlphaFile = "alpha.go"
	maintenanceBetaFile  = "beta.go"
)

// While the maintenance gate is closed a cold collection is refused before
// any LoadCollection request reaches Milvus, and Milvus keeps reporting the
// collection not loaded. Opening the gate lets the same acquire load it.
func TestMaintenanceGateRefusesColdLoadsInMilvus(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, maintenanceMilvusMemory)
	milvus := dialMilvus(t, stack)
	seedCollection(t, milvus, seedSpec{name: maintenanceGateName, rows: maintenanceGateRows, dimension: maintenanceGateDimension, batchRows: maintenanceGateRows, flushEvery: 0})

	recorder := &milvusCallRecorder{}
	service := newService(t, integrationConfig(t, stack, nil), recorder)

	service.SetMaintenance(true)
	sampler := startLoadStateSampler(t, milvus, []string{maintenanceGateName})
	started := time.Now()
	lease, err := service.AcquireCollection(context.Background(), maintenanceGateName)
	if lease != nil {
		lease.Release()
	}
	if !errors.Is(err, semantic.ErrMaintenance) {
		t.Fatalf("AcquireCollection during maintenance returned %v after %s, want ErrMaintenance", err, time.Since(started).Round(time.Millisecond))
	}
	time.Sleep(backoffDeferWindow)
	sampler.finish()
	if calls := recorder.count("LoadCollection", ""); calls != 0 {
		t.Fatalf("daemon sent %d LoadCollection requests during maintenance, want 0", calls)
	}
	samples := sampler.snapshot()
	if len(samples) == 0 {
		t.Fatal("the sampler recorded no load states during maintenance")
	}
	for _, sample := range samples {
		if len(sample.loading) != 0 || len(sample.loaded) != 0 {
			t.Fatalf("Milvus reported %s loading=%v loaded=%v during maintenance at %s, want not loaded", maintenanceGateName, sample.loading, sample.loaded, sample.at.Format(time.RFC3339Nano))
		}
	}

	service.SetMaintenance(false)
	lease, err = service.AcquireCollection(context.Background(), maintenanceGateName)
	if err != nil {
		t.Fatalf("AcquireCollection after maintenance returned error: %v", err)
	}
	defer lease.Release()
	if state := loadState(t, milvus, maintenanceGateName); state.State != entity.LoadStateLoaded {
		t.Fatalf("Milvus reports %s load state %v after maintenance, want loaded", maintenanceGateName, state.State)
	}
	if calls := recorder.count("LoadCollection", maintenanceGateName); calls != 1 {
		t.Fatalf("daemon sent %d LoadCollection requests for %s after maintenance, want 1", calls, maintenanceGateName)
	}
}

// testDaemon is one in-process daemon over a state root: the manager, its
// background sweeps, the gRPC server on the configured unix socket, and a
// client dialed to it. Every Milvus call its client makes is recorded on the
// wire; the store itself is the throwaway Milvus.
type testDaemon struct {
	manager  *daemon.Manager
	client   pb.SemanticSearchDaemonServiceClient
	recorder *milvusCallRecorder
	stop     func()
}

// startTestDaemon runs a daemon over cfg the way the daemon binary does,
// with the periodic sweep on. A second call over the same cfg after stop is
// a restart over the persisted state.
func startTestDaemon(t *testing.T, cfg config.Config) *testDaemon {
	t.Helper()
	recorder := &milvusCallRecorder{}
	manager, err := daemon.NewManager(observedContext(recorder), cfg)
	if err != nil {
		t.Fatalf("daemon.NewManager: %v", err)
	}
	backgroundCtx, stopBackground := context.WithCancel(context.Background())
	manager.ResumeOrphanedJobs(backgroundCtx)
	daemon.NewBackgroundSync(cfg, manager).Start(backgroundCtx)

	_ = os.Remove(cfg.SocketPath)
	listener, err := (&net.ListenConfig{}).Listen(context.Background(), "unix", cfg.SocketPath)
	if err != nil {
		stopBackground()
		t.Fatalf("listen on %s: %v", cfg.SocketPath, err)
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
		stopBackground()
		server.Stop()
		t.Fatalf("dial daemon socket %s: %v", cfg.SocketPath, err)
	}
	stopped := false
	stop := func() {
		if stopped {
			return
		}
		stopped = true
		_ = connection.Close()
		stopBackground()
		server.GracefulStop()
		_ = listener.Close()
		closeCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if closeErr := manager.Close(closeCtx); closeErr != nil {
			t.Errorf("close daemon manager: %v", closeErr)
		}
	}
	t.Cleanup(stop)
	return &testDaemon{manager: manager, client: client, recorder: recorder, stop: stop}
}

func rpcContext() context.Context {
	return grpcutil.WithCorrelation(context.Background())
}

func testClient() *pb.ClientInfo {
	return &pb.ClientInfo{Name: "milvus-integration", Pid: int32(os.Getpid())}
}

// writeSourceFile writes one Go file whose function returns sentinel, so a
// search for the sentinel finds that file.
func writeSourceFile(t *testing.T, directory string, name string, sentinel string) {
	t.Helper()
	source := fmt.Sprintf("package maintenance\n\n// %s\nfunc %s() string {\n\treturn %q\n}\n", sentinel, strings.TrimSuffix(name, ".go"), sentinel)
	if err := os.WriteFile(filepath.Join(directory, name), []byte(source), sourceFileMode); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
}

// waitForJob polls the daemon until jobID reaches a terminal state and
// returns it.
func waitForJob(t *testing.T, client pb.SemanticSearchDaemonServiceClient, jobID string) *pb.Job {
	t.Helper()
	deadline := time.Now().Add(maintenanceJobWait)
	for {
		response, err := client.GetJob(rpcContext(), &pb.GetJobRequest{JobId: jobID})
		if err != nil {
			t.Fatalf("GetJob(%s): %v", jobID, err)
		}
		job := response.GetJob()
		switch job.GetState() {
		case string(model.JobStateCompleted), string(model.JobStateFailed), string(model.JobStateCancelled):
			return job
		}
		if time.Now().After(deadline) {
			t.Fatalf("job %s still %q after %s", jobID, job.GetState(), maintenanceJobWait)
		}
		time.Sleep(maintenanceJobPoll)
	}
}

// waitForNewCompletedJob polls the daemon's job list until a job other than
// the ones in seen completes, and returns it.
func waitForNewCompletedJob(t *testing.T, client pb.SemanticSearchDaemonServiceClient, seen map[string]struct{}) *pb.Job {
	t.Helper()
	deadline := time.Now().Add(maintenanceJobWait)
	for {
		response, err := client.ListJobs(rpcContext(), &pb.ListJobsRequest{CodebaseId: ""})
		if err != nil {
			t.Fatalf("ListJobs: %v", err)
		}
		for _, job := range response.GetJobs() {
			if _, known := seen[job.GetId()]; known {
				continue
			}
			switch job.GetState() {
			case string(model.JobStateCompleted):
				return job
			case string(model.JobStateFailed), string(model.JobStateCancelled):
				t.Fatalf("the sweep's job %s ended %s: %s", job.GetId(), job.GetState(), job.GetError().GetMessage())
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("no new job completed within %s of maintenance ending", maintenanceJobWait)
		}
		time.Sleep(maintenanceJobPoll)
	}
}

func jobIDs(t *testing.T, client pb.SemanticSearchDaemonServiceClient) map[string]struct{} {
	t.Helper()
	response, err := client.ListJobs(rpcContext(), &pb.ListJobsRequest{CodebaseId: ""})
	if err != nil {
		t.Fatalf("ListJobs: %v", err)
	}
	ids := make(map[string]struct{}, len(response.GetJobs()))
	for _, job := range response.GetJobs() {
		ids[job.GetId()] = struct{}{}
	}
	return ids
}

// storeMutations lists the recorded calls that would change or load the
// store, the calls maintenance mode must keep away from Milvus.
func storeMutations(recorder *milvusCallRecorder) []string {
	mutating := map[string]struct{}{
		"LoadCollection": {}, "Insert": {}, "Upsert": {}, "Delete": {}, "Flush": {},
		"CreateCollection": {}, "DropCollection": {}, "AlterCollection": {},
		"CreateIndex": {}, "DropIndex": {}, "RenameCollection": {},
	}
	found := make([]string, 0)
	for _, call := range recorder.snapshot() {
		if _, mutates := mutating[call.method]; mutates {
			found = append(found, fmt.Sprintf("%s %v at %s", call.method, call.collectionNames, call.recordedAt.Format(time.RFC3339Nano)))
		}
	}
	return found
}

func assertMaintenanceRefusal(t *testing.T, operation string, err error) {
	t.Helper()
	if err == nil {
		t.Fatalf("%s succeeded during maintenance, want the maintenance refusal", operation)
	}
	grpcStatus, ok := status.FromError(err)
	if !ok {
		t.Fatalf("%s error %v is not a gRPC status", operation, err)
	}
	if grpcStatus.Code() != codes.FailedPrecondition {
		t.Fatalf("%s code = %s, want FailedPrecondition", operation, grpcStatus.Code())
	}
	if !strings.Contains(grpcStatus.Message(), "maintenance mode") || !strings.Contains(grpcStatus.Message(), maintenanceReason) {
		t.Fatalf("%s message %q does not name maintenance mode and its reason", operation, grpcStatus.Message())
	}
}

func searchCode(client pb.SemanticSearchDaemonServiceClient, path string, query string) (*pb.SearchCodeResponse, error) {
	return client.SearchCode(rpcContext(), &pb.SearchCodeRequest{
		Path:            path,
		Query:           query,
		Limit:           maintenanceSearchLimit,
		ExtensionFilter: nil,
		Client:          testClient(),
	})
}

func resultPaths(response *pb.SearchCodeResponse) []string {
	paths := make([]string, 0, len(response.GetResults()))
	for _, result := range response.GetResults() {
		paths = append(paths, result.GetRelativePath())
	}
	return paths
}

// assertDaemonQuietDuringMaintenance lets the periodic sweep run for a window
// and proves, from the wire and from Milvus, that the daemon started no work
// against the store: no job, no mutating or loading call, and the collection
// still not loaded. It then proves every request surface reports the mode.
func assertDaemonQuietDuringMaintenance(t *testing.T, running *testDaemon, milvus *milvusclient.Client, codebasePath string, collectionName string, jobsBefore map[string]struct{}) {
	t.Helper()
	running.recorder.reset()
	sampler := startLoadStateSampler(t, milvus, []string{collectionName})
	time.Sleep(maintenanceSweepWindow)
	sampler.finish()

	if mutations := storeMutations(running.recorder); len(mutations) != 0 {
		t.Fatalf("daemon reached the store during maintenance:\n%s", strings.Join(mutations, "\n"))
	}
	for id := range jobIDs(t, running.client) {
		if _, known := jobsBefore[id]; !known {
			t.Fatalf("daemon registered job %s during maintenance, want none", id)
		}
	}
	samples := sampler.snapshot()
	if len(samples) == 0 {
		t.Fatal("the sampler recorded no load states during maintenance")
	}
	for _, sample := range samples {
		if len(sample.loading) != 0 || len(sample.loaded) != 0 {
			t.Fatalf("Milvus reported %s loading=%v loaded=%v during maintenance at %s, want not loaded", collectionName, sample.loading, sample.loaded, sample.at.Format(time.RFC3339Nano))
		}
	}
	t.Logf("sweeps ran for %s during maintenance: no store mutation, no job, %s not loaded in %d samples", maintenanceSweepWindow, collectionName, len(samples))

	_, searchErr := searchCode(running.client, codebasePath, "alpha sentinel")
	assertMaintenanceRefusal(t, "SearchCode", searchErr)
	_, indexErr := running.client.StartIndex(rpcContext(), &pb.StartIndexRequest{Path: codebasePath, Client: testClient()})
	assertMaintenanceRefusal(t, "StartIndex", indexErr)
	if calls := running.recorder.count("LoadCollection", ""); calls != 0 {
		t.Fatalf("the refused search and index sent %d LoadCollection requests, want 0", calls)
	}

	getIndex, err := running.client.GetIndex(rpcContext(), &pb.GetIndexRequest{Path: codebasePath, Client: testClient()})
	if err != nil {
		t.Fatalf("GetIndex during maintenance: %v", err)
	}
	if !getIndex.GetMaintenance().GetEnabled() || getIndex.GetMaintenance().GetReason() != maintenanceReason {
		t.Fatalf("GetIndex maintenance = %v, want enabled with reason %q", getIndex.GetMaintenance(), maintenanceReason)
	}
	if getIndex.Searchable == nil || getIndex.GetSearchable() {
		t.Fatalf("GetIndex searchable = %v during maintenance, want present and false", getIndex.Searchable)
	}
	if !strings.Contains(getIndex.GetDisplayText(), "Maintenance mode is on") || !strings.Contains(getIndex.GetDisplayText(), maintenanceReason) {
		t.Fatalf("GetIndex text lacks the maintenance banner:\n%s", getIndex.GetDisplayText())
	}
	getStatus, err := running.client.GetStatus(rpcContext(), &pb.GetStatusRequest{})
	if err != nil {
		t.Fatalf("GetStatus during maintenance: %v", err)
	}
	if !getStatus.GetMaintenance().GetEnabled() {
		t.Fatal("GetStatus did not report maintenance mode")
	}
}

// Maintenance mode keeps a running daemon away from a real Milvus until it is
// turned off. The test indexes a real codebase into the throwaway store, lets
// the searched collection go idle, turns the mode on, and changes the codebase
// while the periodic sweep keeps running: the sweep starts no job and sends
// Milvus no load or write, the collection stays not loaded, search and index
// requests fail fast with the maintenance status, and the status surfaces
// report the mode. A daemon restarted over the same state comes back in the
// mode with the same guarantees. Turning the mode off lets the next sweep sync
// the change into Milvus, and a search then loads the collection and finds
// the file written during maintenance.
func TestMaintenanceModePausesDaemonAgainstMilvus(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, maintenanceMilvusMemory)
	milvus := dialMilvus(t, stack)
	cfg := integrationConfig(t, stack, map[string]string{
		"CLAUDE_CONTEXT_BACKGROUND_SYNC":                   "true",
		"CLAUDE_CONTEXT_SYNC_INTERVAL_MS":                  maintenanceSweepIntervalMS,
		"CLAUDE_CONTEXT_MILVUS_COLLECTION_IDLE_TIMEOUT_MS": maintenanceIdleTimeoutMS,
	})
	codebasePath := filepath.Join(t.TempDir(), "maintenance-codebase")
	if err := os.MkdirAll(codebasePath, sourceDirMode); err != nil {
		t.Fatalf("create codebase: %v", err)
	}
	writeSourceFile(t, codebasePath, maintenanceAlphaFile, "alpha sentinel written before the first build")

	first := startTestDaemon(t, cfg)
	startResponse, err := first.client.StartIndex(rpcContext(), &pb.StartIndexRequest{Path: codebasePath, Client: testClient()})
	if err != nil {
		t.Fatalf("StartIndex: %v", err)
	}
	if job := waitForJob(t, first.client, startResponse.GetJobId()); job.GetState() != string(model.JobStateCompleted) {
		t.Fatalf("first build ended %s: %s", job.GetState(), job.GetError().GetMessage())
	}
	getIndex, err := first.client.GetIndex(rpcContext(), &pb.GetIndexRequest{Path: codebasePath, Client: testClient()})
	if err != nil {
		t.Fatalf("GetIndex after the first build: %v", err)
	}
	collectionName := getIndex.GetCodebase().GetCollectionName()
	if collectionName == "" {
		t.Fatal("GetIndex reports no collection name after the first build")
	}
	alphaSearch, err := searchCode(first.client, codebasePath, "alpha sentinel")
	if err != nil {
		t.Fatalf("SearchCode before maintenance: %v", err)
	}
	if paths := resultPaths(alphaSearch); !slices.Contains(paths, maintenanceAlphaFile) {
		t.Fatalf("search before maintenance returned %v, want %s", paths, maintenanceAlphaFile)
	}
	if state := loadState(t, milvus, collectionName); state.State != entity.LoadStateLoaded {
		t.Fatalf("Milvus reports %s load state %v after a search, want loaded", collectionName, state.State)
	}
	// The daemon releases the collection itself once it sits idle, so the
	// maintenance phase starts cold without the test touching Milvus.
	waitForLoadState(t, milvus, collectionName, entity.LoadStateNotLoad)
	t.Logf("indexed %s into %s and let it go idle", codebasePath, collectionName)

	on, err := first.client.SetMaintenanceMode(rpcContext(), &pb.SetMaintenanceModeRequest{Enabled: true, Reason: maintenanceReason, Client: testClient()})
	if err != nil {
		t.Fatalf("SetMaintenanceMode(true): %v", err)
	}
	if !on.GetMaintenance().GetEnabled() || on.GetMaintenance().GetReason() != maintenanceReason {
		t.Fatalf("SetMaintenanceMode reply = %v, want enabled with the reason", on.GetMaintenance())
	}
	if !strings.Contains(on.GetDisplayText(), "Maintenance mode is on") {
		t.Fatalf("SetMaintenanceMode reply text lacks the banner:\n%s", on.GetDisplayText())
	}
	writeSourceFile(t, codebasePath, maintenanceBetaFile, "beta sentinel written during maintenance")
	jobsBefore := jobIDs(t, first.client)
	assertDaemonQuietDuringMaintenance(t, first, milvus, codebasePath, collectionName, jobsBefore)

	// Restart the daemon over the same state root the way launchd would after
	// a crash mid-restore: it must come back in maintenance mode.
	first.stop()
	restarted := startTestDaemon(t, cfg)
	if state := restarted.manager.Maintenance(); !state.Enabled || state.Reason != maintenanceReason || state.Since.IsZero() {
		t.Fatalf("restarted daemon maintenance state = %+v, want enabled with the reason and start time", state)
	}
	assertDaemonQuietDuringMaintenance(t, restarted, milvus, codebasePath, collectionName, jobsBefore)

	off, err := restarted.client.SetMaintenanceMode(rpcContext(), &pb.SetMaintenanceModeRequest{Enabled: false, Reason: "", Client: testClient()})
	if err != nil {
		t.Fatalf("SetMaintenanceMode(false): %v", err)
	}
	if off.GetMaintenance().GetEnabled() {
		t.Fatal("SetMaintenanceMode(false) left the mode on")
	}
	restarted.recorder.reset()
	syncJob := waitForNewCompletedJob(t, restarted.client, jobsBefore)
	t.Logf("the sweep after maintenance ended ran job %s (%s) in %s", syncJob.GetId(), syncJob.GetOperation(), syncJob.GetCompletedAt().AsTime().Sub(syncJob.GetStartedAt().AsTime()).Round(time.Millisecond))
	if writes := restarted.recorder.count("Insert", collectionName) + restarted.recorder.count("Upsert", collectionName); writes == 0 {
		t.Fatalf("the sweep's sync sent no Insert or Upsert for %s; recorded calls: %v", collectionName, storeMutations(restarted.recorder))
	}

	betaSearch, err := searchCode(restarted.client, codebasePath, "beta sentinel")
	if err != nil {
		t.Fatalf("SearchCode after maintenance ended: %v", err)
	}
	if paths := resultPaths(betaSearch); !slices.Contains(paths, maintenanceBetaFile) {
		t.Fatalf("search after maintenance returned %v, want the file written during maintenance %s", paths, maintenanceBetaFile)
	}
	if state := loadState(t, milvus, collectionName); state.State != entity.LoadStateLoaded {
		t.Fatalf("Milvus reports %s load state %v after the search, want loaded", collectionName, state.State)
	}
	getIndex, err = restarted.client.GetIndex(rpcContext(), &pb.GetIndexRequest{Path: codebasePath, Client: testClient()})
	if err != nil {
		t.Fatalf("GetIndex after maintenance ended: %v", err)
	}
	if getIndex.GetMaintenance().GetEnabled() || strings.Contains(getIndex.GetDisplayText(), "Maintenance mode is on") {
		t.Fatalf("GetIndex still reports maintenance after it ended:\n%s", getIndex.GetDisplayText())
	}
}

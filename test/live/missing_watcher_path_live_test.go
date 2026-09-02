//go:build live

package live

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/semantic/milvusgrpc"
	"google.golang.org/grpc"
)

const (
	sandboxDaemonBinary = "lm-semantic-search-daemon"
	watcherJobTimeout   = 30 * time.Second
)

type watcherIndexSnapshot struct {
	codebaseID      string
	collectionName  string
	checkpointPath  string
	checkpointBytes []byte
	checkpointInfo  os.FileInfo
	rowCount        int
	vectorChecksums []string
}

type watcherProgressUpdate struct {
	state      string
	processed  int32
	filesTotal int32
}

func TestWatcherRetainsRemovedPathAtPublicBoundary(t *testing.T) {
	harness := newSandboxWatcherHarness(t)
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "main.go"), []byte("package watcherfixture\n\nfunc Existing() string { return \"indexed\" }\n"), 0o600); err != nil {
		t.Fatalf("write indexed source: %v", err)
	}

	initial := startPublicCodebaseIndex(t, harness, root)
	requirePublicCompleted(t, initial, "initial codebase index")
	before := captureWatcherIndexSnapshot(t, harness, root)
	if before.rowCount == 0 {
		t.Fatal("initial codebase index wrote no rows")
	}

	for attempt := 1; attempt <= 2; attempt++ {
		knownJobs := publicWatcherJobIDs(t, harness)
		job := triggerRemovedPathConverge(t, harness, root, knownJobs, attempt)
		requirePublicCompleted(t, job, fmt.Sprintf("watcher converge %d", attempt))
		assertWatcherConvergeResult(t, job)
		assertNoSemanticRemovalTrace(t, job)
		assertWatcherIndexSnapshot(t, harness, root, before)
	}
}

func TestWatcherProgressRejectsDuplicateRunningOneOfOne(t *testing.T) {
	first := watcherProgressUpdate{state: "running", processed: 1, filesTotal: 1}
	second := watcherProgressUpdate{state: "running", processed: 1, filesTotal: 1}
	if !duplicateNonFinalWatcherProgress(first, second) {
		t.Fatal("duplicate running 1 of 1 watcher progress was not detected")
	}
}

func TestWatcherProgressAllowsTotalChange(t *testing.T) {
	first := watcherProgressUpdate{state: "running", processed: 0, filesTotal: 0}
	second := watcherProgressUpdate{state: "running", processed: 0, filesTotal: 1}
	if duplicateNonFinalWatcherProgress(first, second) {
		t.Fatal("watcher progress treated a changed total as a duplicate")
	}
}

func newSandboxWatcherHarness(t *testing.T) *harness {
	t.Helper()
	defaultConfig, err := config.Default()
	if err != nil {
		t.Fatalf("resolve live configuration: %v", err)
	}
	milvusAddress := strings.TrimSpace(defaultConfig.MilvusAddress)
	if milvusAddress == "" {
		t.Skip("BLOCKED: MILVUS_ADDRESS is empty; the sandbox watcher acceptance needs local Milvus")
	}

	callRecorder := &milvusCallRecorder{}
	operatorContext := context.WithValue(context.Background(), milvusgrpc.CallObserverContextKey{}, milvusgrpc.CallObserver(callRecorder.observe))
	dialContext, cancelDial := context.WithTimeout(operatorContext, 5*time.Second)
	operatorMilvus, err := milvusclient.New(dialContext, &milvusclient.ClientConfig{
		Address:     milvusAddress,
		APIKey:      defaultConfig.MilvusToken,
		DialOptions: milvusgrpc.DialOptions(operatorContext, slog.Default(), milvusgrpc.DefaultCallTimeouts()),
	})
	cancelDial()
	if err != nil {
		t.Skipf("BLOCKED: Milvus is unreachable at %s: %v", milvusAddress, err)
	}
	beforeDatabases, err := listMilvusDatabases(operatorMilvus)
	if err != nil {
		closeMilvusClient(operatorMilvus)
		t.Fatalf("list operator Milvus databases: %v", err)
	}
	operatorBefore, err := readMilvusInventory(operatorMilvus)
	if err != nil {
		closeMilvusClient(operatorMilvus)
		t.Fatalf("read operator Milvus inventory: %v", err)
	}

	databaseName := liveDatabasePrefix + randomID()
	if slices.Contains(beforeDatabases, databaseName) {
		closeMilvusClient(operatorMilvus)
		t.Fatalf("temporary Milvus database %q already exists", databaseName)
	}
	createContext, cancelCreate := context.WithTimeout(operatorContext, 15*time.Second)
	if err := operatorMilvus.CreateDatabase(createContext, milvusclient.NewCreateDatabaseOption(databaseName)); err != nil {
		cancelCreate()
		closeMilvusClient(operatorMilvus)
		t.Fatalf("create temporary Milvus database %s: %v", databaseName, err)
	}
	cancelCreate()

	sandboxContext := context.WithValue(context.Background(), milvusgrpc.CallObserverContextKey{}, milvusgrpc.CallObserver(callRecorder.observe))
	dialContext, cancelDial = context.WithTimeout(sandboxContext, 5*time.Second)
	sandboxMilvus, err := milvusclient.New(dialContext, &milvusclient.ClientConfig{
		Address:     milvusAddress,
		APIKey:      defaultConfig.MilvusToken,
		DBName:      databaseName,
		DialOptions: milvusgrpc.DialOptions(sandboxContext, slog.Default(), milvusgrpc.DefaultCallTimeouts()),
	})
	cancelDial()
	if err != nil {
		dropOwnedDatabase(operatorMilvus, databaseName)
		closeMilvusClient(operatorMilvus)
		t.Fatalf("connect temporary Milvus database %s: %v", databaseName, err)
	}
	sandboxBefore, err := readMilvusInventory(sandboxMilvus)
	if err != nil || len(sandboxBefore) != 0 {
		closeMilvusClient(sandboxMilvus)
		dropOwnedDatabase(operatorMilvus, databaseName)
		closeMilvusClient(operatorMilvus)
		if err != nil {
			t.Fatalf("read temporary Milvus inventory: %v", err)
		}
		t.Fatalf("temporary Milvus database %q started nonempty: %v", databaseName, sandboxBefore)
	}

	harness := &harness{
		t:               t,
		config:          config.Config{},
		operatorMilvus:  operatorMilvus,
		milvus:          sandboxMilvus,
		databaseName:    databaseName,
		beforeDatabases: beforeDatabases,
		operatorBefore:  operatorBefore,
		sandboxBefore:   sandboxBefore,
		temporaryNames:  make(map[string]struct{}),
		callRecorder:    callRecorder,
		milvusContext:   sandboxContext,
	}
	t.Cleanup(func() {
		for _, cleanupErr := range harness.cleanupMilvus() {
			t.Error(cleanupErr)
		}
	})
	sandboxRoot, err := os.MkdirTemp("/tmp", "lms-watcher-sandbox-")
	if err != nil {
		closeMilvusClient(sandboxMilvus)
		dropOwnedDatabase(operatorMilvus, databaseName)
		closeMilvusClient(operatorMilvus)
		t.Fatalf("create short sandbox root: %v", err)
	}
	t.Cleanup(func() {
		if removeErr := os.RemoveAll(sandboxRoot); removeErr != nil {
			t.Errorf("remove owned sandbox root: %v", removeErr)
		}
	})
	harness.trackTemporaryCollection(semantic.ReuseCatalogCollectionName(config.Config{
		StateRoot:          filepath.Join(sandboxRoot, "state"),
		EmbeddingDimension: fakeEmbeddingDimension,
	}))
	embedder := newFakeEmbeddingServer(t, nil)
	process := startSandboxDaemon(t, sandboxRoot, databaseName, defaultConfig, embedder.URL)
	t.Cleanup(func() {
		if harness.conn != nil {
			_ = harness.conn.Close()
		}
		stopSandboxDaemon(t, process)
	})

	connection, client := waitForSandboxDaemon(t, filepath.Join(sandboxRoot, "daemon.sock"), process)
	harness.conn = connection
	harness.client = client
	return harness
}

func dropOwnedDatabase(client *milvusclient.Client, databaseName string) {
	cleanupContext, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	_ = client.DropDatabase(cleanupContext, milvusclient.NewDropDatabaseOption(databaseName))
}

func startSandboxDaemon(t *testing.T, sandboxRoot string, databaseName string, defaultConfig config.Config, embedderURL string) *exec.Cmd {
	t.Helper()
	command := exec.Command(builtSandboxDaemonPath(t), "sandbox", "--root", sandboxRoot)
	command.Env = sandboxDaemonEnvironment(sandboxRoot, databaseName, defaultConfig, embedderURL)
	output := &bytes.Buffer{}
	command.Stdout = output
	command.Stderr = output
	if err := command.Start(); err != nil {
		t.Fatalf("start built sandbox daemon: %v", err)
	}
	return command
}

func builtSandboxDaemonPath(t *testing.T) string {
	t.Helper()
	_, sourcePath, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve sandbox acceptance source path")
	}
	path := filepath.Join(filepath.Dir(sourcePath), "..", "..", "dist", sandboxDaemonBinary)
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("built sandbox daemon %s is unavailable; run make build first: %v", path, err)
	}
	if !info.Mode().IsRegular() {
		t.Fatalf("built sandbox daemon %s is not a regular file", path)
	}
	return path
}

func sandboxDaemonEnvironment(sandboxRoot string, databaseName string, defaultConfig config.Config, embedderURL string) []string {
	values := make(map[string]string)
	for _, item := range os.Environ() {
		name, value, found := strings.Cut(item, "=")
		if found {
			values[name] = value
		}
	}
	for name, value := range map[string]string{
		"CLAUDE_CONTEXTD_STATE_ROOT":               filepath.Join(sandboxRoot, "state"),
		"CLAUDE_CONTEXTD_CONFIG_ROOT":              filepath.Join(sandboxRoot, "config"),
		"CLAUDE_CONTEXTD_CONTEXT_ROOT":             filepath.Join(sandboxRoot, "context"),
		"CLAUDE_CONTEXTD_SOCKET_PATH":              filepath.Join(sandboxRoot, "daemon.sock"),
		"CLAUDE_CONTEXTD_LOG_PATH":                 filepath.Join(sandboxRoot, "logs", "daemon.log"),
		"CLAUDE_CONTEXTD_MODEL_CACHE_ROOT":         filepath.Join(sandboxRoot, "models"),
		"CLAUDE_CONTEXT_PROFILE":                   config.ProfileStandard,
		"CLAUDE_CONTEXT_DEBUG_LISTENER":            "false",
		"CLAUDE_CONTEXT_DEBUG_LISTEN_ADDR":         "127.0.0.1:0",
		"CLAUDE_CONTEXT_BACKGROUND_SYNC":           "false",
		"CLAUDE_CONTEXT_TRIGGER_WATCHER":           "false",
		"CLAUDE_CONTEXT_FILE_WATCHER":              "true",
		"CLAUDE_CONTEXT_MAX_CONCURRENT_INDEX_JOBS": "1",
		"MILVUS_ADDRESS":                           defaultConfig.MilvusAddress,
		"MILVUS_TOKEN":                             defaultConfig.MilvusToken,
		"MILVUS_DATABASE":                          databaseName,
		"EMBEDDING_PROVIDER":                       "OpenAI",
		"EMBEDDING_MODEL":                          "sandbox-watcher-" + databaseName,
		"EMBEDDING_DIMENSION":                      fmt.Sprintf("%d", fakeEmbeddingDimension),
		"EMBEDDING_BATCH_SIZE":                     "8",
		"OPENAI_BASE_URL":                          embedderURL,
		"OPENAI_API_KEY":                           "sandbox-watcher-local", //gitleaks:allow // not a secret: the local fake accepts any non-empty key
	} {
		values[name] = value
	}
	names := make([]string, 0, len(values))
	for name := range values {
		names = append(names, name)
	}
	slices.Sort(names)
	result := make([]string, 0, len(names))
	for _, name := range names {
		result = append(result, name+"="+values[name])
	}
	return result
}

func waitForSandboxDaemon(t *testing.T, socketPath string, process *exec.Cmd) (*grpc.ClientConn, pb.SemanticSearchDaemonServiceClient) {
	t.Helper()
	deadline := time.Now().Add(watcherJobTimeout)
	var lastError error
	for time.Now().Before(deadline) {
		connection, client, err := grpcutil.DialDaemon(correlatedContext(), socketPath)
		if err == nil {
			_, err = client.ListIndexes(correlatedContext(), &pb.ListIndexesRequest{})
			if err == nil {
				return connection, client
			}
			_ = connection.Close()
		}
		lastError = err
		time.Sleep(jobPollInterval)
	}
	t.Fatalf("built sandbox daemon did not serve %s: %v\n%s", socketPath, lastError, sandboxProcessOutput(process))
	return nil, nil
}

func stopSandboxDaemon(t *testing.T, process *exec.Cmd) {
	t.Helper()
	if process.ProcessState != nil {
		return
	}
	if err := process.Process.Signal(os.Interrupt); err != nil {
		t.Errorf("interrupt sandbox daemon: %v", err)
		return
	}
	done := make(chan error, 1)
	go func() { done <- process.Wait() }()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("wait for sandbox daemon: %v\n%s", err, sandboxProcessOutput(process))
		}
	case <-time.After(15 * time.Second):
		if err := process.Process.Kill(); err != nil {
			t.Errorf("kill stalled sandbox daemon: %v", err)
			return
		}
		select {
		case err := <-done:
			if err != nil {
				t.Errorf("wait for killed sandbox daemon: %v\n%s", err, sandboxProcessOutput(process))
			}
		case <-time.After(5 * time.Second):
			t.Errorf("wait for killed sandbox daemon timed out\n%s", sandboxProcessOutput(process))
		}
	}
}

func sandboxProcessOutput(process *exec.Cmd) string {
	if output, ok := process.Stdout.(*bytes.Buffer); ok {
		return output.String()
	}
	return "sandbox process output unavailable"
}

func startPublicCodebaseIndex(t *testing.T, harness *harness, root string) *pb.Job {
	t.Helper()
	response, err := harness.client.StartIndex(correlatedContext(), &pb.StartIndexRequest{
		Path: root, Splitter: &pb.SplitterConfig{Type: "ast"}, Client: &pb.ClientInfo{Name: "missing-watcher-live-harness"},
	})
	if err != nil {
		t.Fatalf("start codebase index: %v", err)
	}
	if response.GetJobId() == "" {
		t.Fatal("start codebase index returned an empty job id")
	}
	return waitForCodebasePublicJob(t, harness, response.GetJobId())
}

func triggerRemovedPathConverge(t *testing.T, harness *harness, root string, knownJobs map[string]struct{}, attempt int) *pb.Job {
	t.Helper()
	path := filepath.Join(root, fmt.Sprintf("removed-%d.go", attempt))
	startedAt := time.Now()
	if err := os.WriteFile(path, []byte("package watcherfixture\n"), 0o600); err != nil {
		t.Fatalf("write transient source %d: %v", attempt, err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatalf("remove transient source %d: %v", attempt, err)
	}
	if time.Since(startedAt) >= 2*time.Second {
		t.Fatalf("transient source %d outlived the watcher debounce window", attempt)
	}
	return waitForNewWatcherJob(t, harness, knownJobs)
}

func waitForCodebasePublicJob(t *testing.T, harness *harness, jobID string) *pb.Job {
	t.Helper()
	deadline := time.Now().Add(watcherJobTimeout)
	for time.Now().Before(deadline) {
		response, err := harness.client.GetJob(correlatedContext(), &pb.GetJobRequest{JobId: jobID})
		if err != nil {
			t.Fatalf("get public job %s: %v", jobID, err)
		}
		job := response.GetJob()
		if job == nil {
			t.Fatalf("get public job %s returned no job", jobID)
		}
		if publicJobTerminal(job) {
			return job
		}
		time.Sleep(jobPollInterval)
	}
	t.Fatalf("public job %s did not finish within %s", jobID, watcherJobTimeout)
	return nil
}

func waitForNewWatcherJob(t *testing.T, harness *harness, knownJobs map[string]struct{}) *pb.Job {
	t.Helper()
	deadline := time.Now().Add(watcherJobTimeout)
	watcherJobID := ""
	var previous watcherProgressUpdate
	var previousUpdatedAt time.Time
	havePrevious := false
	for time.Now().Before(deadline) {
		if watcherJobID == "" {
			response, err := harness.client.ListJobs(correlatedContext(), &pb.ListJobsRequest{})
			if err != nil {
				t.Fatalf("list public watcher jobs: %v", err)
			}
			for _, job := range response.GetJobs() {
				if _, known := knownJobs[job.GetId()]; known {
					continue
				}
				if job.GetOperation() == "converge" && job.GetClient().GetName() == "daemon-watcher" {
					watcherJobID = job.GetId()
					break
				}
			}
		}
		if watcherJobID != "" {
			response, err := harness.client.GetJob(correlatedContext(), &pb.GetJobRequest{JobId: watcherJobID})
			if err != nil {
				t.Fatalf("get watcher job %s: %v", watcherJobID, err)
			}
			job := response.GetJob()
			if job == nil {
				t.Fatalf("get watcher job %s returned no job", watcherJobID)
			}
			updatedAt := job.GetUpdatedAt().AsTime()
			current := watcherProgressUpdate{state: job.GetState(), processed: job.GetProgress().GetFilesProcessed(), filesTotal: job.GetProgress().GetFilesTotal()}
			if havePrevious && !updatedAt.Equal(previousUpdatedAt) && duplicateNonFinalWatcherProgress(previous, current) {
				t.Fatalf("watcher job repeated non-final progress: previous=%+v current=%+v", previous, current)
			}
			if !updatedAt.Equal(previousUpdatedAt) {
				previous = current
				previousUpdatedAt = updatedAt
				havePrevious = true
			}
			if publicJobTerminal(job) {
				return job
			}
		}
		time.Sleep(jobPollInterval)
	}
	t.Fatal("watcher converge job did not finish within the public timeout")
	return nil
}

func duplicateNonFinalWatcherProgress(previous watcherProgressUpdate, current watcherProgressUpdate) bool {
	return previous.state == "running" && current.state == "running" && previous.processed == current.processed && previous.filesTotal == current.filesTotal
}

func publicWatcherJobIDs(t *testing.T, harness *harness) map[string]struct{} {
	t.Helper()
	response, err := harness.client.ListJobs(correlatedContext(), &pb.ListJobsRequest{})
	if err != nil {
		t.Fatalf("list public jobs before watcher event: %v", err)
	}
	jobIDs := make(map[string]struct{}, len(response.GetJobs()))
	for _, job := range response.GetJobs() {
		jobIDs[job.GetId()] = struct{}{}
	}
	return jobIDs
}

func publicJobTerminal(job *pb.Job) bool {
	switch job.GetState() {
	case "completed", "failed", "cancelled":
		return true
	default:
		return false
	}
}

func requirePublicCompleted(t *testing.T, job *pb.Job, label string) {
	t.Helper()
	if job.GetState() == "completed" {
		return
	}
	t.Fatalf("%s state = %q, want completed: %s", label, job.GetState(), job.GetDisplayError())
}

func assertWatcherConvergeResult(t *testing.T, job *pb.Job) {
	t.Helper()
	progress := job.GetProgress()
	if progress.GetUnit() != "path" {
		t.Fatalf("watcher progress unit = %q, want path", progress.GetUnit())
	}
	if progress.GetFilesTotal() != 1 || progress.GetFilesProcessed() != 1 {
		t.Fatalf("watcher progress = %d of %d, want 1 of 1 path", progress.GetFilesProcessed(), progress.GetFilesTotal())
	}
	filesEmbedded := outcomeRowCount(progress.GetBreakdown().GetFileRows(), pb.OutcomeKind_OUTCOME_KIND_EMBEDDED)
	if filesEmbedded != 0 || progress.GetChunksEmbedded() != 0 {
		t.Fatalf("watcher embedded files/chunks = %d/%d, want 0/0", filesEmbedded, progress.GetChunksEmbedded())
	}
	if !progress.GetHeartbeatAt().AsTime().After(job.GetStartedAt().AsTime()) {
		t.Fatalf("watcher heartbeat = %s, want after start %s", progress.GetHeartbeatAt().AsTime(), job.GetStartedAt().AsTime())
	}
}

func outcomeRowCount(rows []*pb.OutcomeRow, kind pb.OutcomeKind) int32 {
	for _, row := range rows {
		if row.GetKind() == kind {
			return row.GetCount()
		}
	}
	return 0
}

func assertNoSemanticRemovalTrace(t *testing.T, job *pb.Job) {
	t.Helper()
	breakdown := job.GetProgress().GetBreakdown()
	for _, rows := range [][]*pb.OutcomeRow{breakdown.GetFileRows(), breakdown.GetChunkRows()} {
		for _, row := range rows {
			if row.GetKind() == pb.OutcomeKind_OUTCOME_KIND_REMOVED {
				t.Fatalf("watcher job reported semantic removal: %+v", breakdown)
			}
		}
	}
}

func captureWatcherIndexSnapshot(t *testing.T, harness *harness, root string) watcherIndexSnapshot {
	t.Helper()
	response, err := harness.client.GetIndex(correlatedContext(), &pb.GetIndexRequest{Path: root})
	if err != nil {
		t.Fatalf("get public codebase index: %v", err)
	}
	codebase := response.GetCodebase()
	if codebase == nil || codebase.GetCollectionName() == "" {
		t.Fatal("public codebase index omitted collection identity")
	}
	checkpointPath := codebase.GetMerkleSnapshotPath()
	if checkpointPath == "" {
		t.Fatal("public codebase index omitted checkpoint identity")
	}
	checkpointBytes, err := os.ReadFile(checkpointPath)
	if err != nil {
		t.Fatalf("read checkpoint %s: %v", checkpointPath, err)
	}
	checkpointInfo, err := os.Stat(checkpointPath)
	if err != nil {
		t.Fatalf("stat checkpoint %s: %v", checkpointPath, err)
	}
	rows := queryRowSnapshots(t, harness, codebase.GetCollectionName(), `id != ""`)
	checksums := make([]string, 0, len(rows))
	for _, row := range rows {
		checksums = append(checksums, row.vectorChecksum)
	}
	slices.Sort(checksums)
	harness.trackCollectionFamily(codebase.GetCollectionName())
	return watcherIndexSnapshot{codebaseID: codebase.GetId(), collectionName: codebase.GetCollectionName(), checkpointPath: checkpointPath, checkpointBytes: checkpointBytes, checkpointInfo: checkpointInfo, rowCount: len(rows), vectorChecksums: checksums}
}

func assertWatcherIndexSnapshot(t *testing.T, harness *harness, root string, before watcherIndexSnapshot) {
	t.Helper()
	after := captureWatcherIndexSnapshot(t, harness, root)
	if after.codebaseID != before.codebaseID || after.collectionName != before.collectionName {
		t.Fatalf("codebase identity changed: before=%+v after=%+v", before, after)
	}
	if after.checkpointPath != before.checkpointPath {
		t.Fatalf("checkpoint path changed: before=%s after=%s", before.checkpointPath, after.checkpointPath)
	}
	if !os.SameFile(before.checkpointInfo, after.checkpointInfo) {
		t.Fatal("watcher replaced the checkpoint file")
	}
	if !bytes.Equal(before.checkpointBytes, after.checkpointBytes) {
		t.Fatal("watcher changed checkpoint bytes")
	}
	if after.rowCount != before.rowCount {
		t.Fatalf("collection rows changed: before=%d after=%d", before.rowCount, after.rowCount)
	}
	if !slices.Equal(after.vectorChecksums, before.vectorChecksums) {
		t.Fatalf("collection vector SHA-256 values changed: before=%v after=%v", before.vectorChecksums, after.vectorChecksums)
	}
}

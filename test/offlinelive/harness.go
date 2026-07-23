//go:build offlinelive

// Package offlinelive validates the offline daemon profile against an isolated
// fixture and throwaway state.
package offlinelive

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/store"
	"google.golang.org/grpc"
)

const (
	daemonProcessName = "lm-semantic-search-daemon"

	milvusAddress = "127.0.0.1:19530"
	lmdAddress    = "127.0.0.1:5400"

	// The query describes transferred-data corruption without reusing the target
	// identifier or its archive, checksum, validation, and storage vocabulary.
	fixtureQuery       = "detect corruption in transferred bytes prior to acceptance"
	targetRelativePath = "checksum.go"
	targetFunctionName = "VerifyArchiveChecksum"
	callerFunctionName = "BuildArchiveReport"

	indexedStatus         = "indexed"
	collectionReady       = "ready"
	localCollectionPrefix = "local_code_chunks_"

	searchResultLimit     int32 = 3
	queryMeasurementCount       = 7
	queryLatencyBound           = 200 * time.Millisecond

	jobPollTimeout   = 2 * time.Minute
	graphPollTimeout = 30 * time.Second
	pollInterval     = 100 * time.Millisecond
)

type harness struct {
	t           *testing.T
	config      config.Config
	manager     *daemon.Manager
	connection  *grpc.ClientConn
	client      pb.SemanticSearchDaemonServiceClient
	fixturePath string
	prodPidsPre map[int]bool
	dialGuard   *netDialGuard
}

func newHarness(t *testing.T) *harness {
	t.Helper()
	slog.Debug("offline live harness setup started")

	prodPidsPre, err := snapshotProductionDaemonPids(
		context.Background(),
		runCommandOutput,
	)
	if err != nil {
		t.Fatalf("snapshot production daemons before offline validation: %v", err)
	}
	dialGuard := newNetDialGuard(t, []string{milvusAddress, lmdAddress})
	stateRoot := t.TempDir()
	socketDirectory, err := os.MkdirTemp("/tmp", "lms-offline-live-")
	if err != nil {
		t.Fatalf("create short socket directory: %v", err)
	}
	t.Cleanup(func() {
		if removeErr := os.RemoveAll(socketDirectory); removeErr != nil {
			t.Errorf("remove socket directory: %v", removeErr)
		}
	})

	socketPath := filepath.Join(socketDirectory, "daemon.sock")
	offlineConfig := resolveOfflineConfig(t, stateRoot, socketPath)
	prepareState(t, offlineConfig)

	manager, err := daemon.NewManager(context.Background(), offlineConfig)
	if err != nil {
		t.Fatalf("create offline daemon manager: %v", err)
	}
	stopServer := startInProcessServer(t, manager, offlineConfig.SocketPath)

	connection, client, err := grpcutil.DialDaemon(
		context.Background(),
		offlineConfig.SocketPath,
	)
	if err != nil {
		stopServer()
		manager.CloseGraphEngines()
		t.Fatalf("dial isolated daemon: %v", err)
	}

	offlineHarness := &harness{
		t:           t,
		config:      offlineConfig,
		manager:     manager,
		connection:  connection,
		client:      client,
		fixturePath: fixtureDirectory(t),
		prodPidsPre: prodPidsPre,
		dialGuard:   dialGuard,
	}
	t.Cleanup(func() {
		offlineHarness.teardown(stopServer)
	})
	slog.Debug(
		"offline live harness setup completed",
		"state_root",
		stateRoot,
		"socket_path",
		offlineConfig.SocketPath,
	)
	return offlineHarness
}

func resolveOfflineConfig(
	t *testing.T,
	stateRoot string,
	socketPath string,
) config.Config {
	t.Helper()

	environment := []struct {
		name  string
		value string
	}{
		{name: "HOME", value: t.TempDir()},
		{name: "CLAUDE_CONTEXTD_CONFIG_ROOT", value: t.TempDir()},
		{name: "CLAUDE_CONTEXTD_STATE_ROOT", value: stateRoot},
		{name: "CLAUDE_CONTEXTD_SOCKET_PATH", value: socketPath},
		{
			name:  "CLAUDE_CONTEXTD_LOG_PATH",
			value: filepath.Join(stateRoot, "logs", "daemon.log"),
		},
		{name: "CLAUDE_CONTEXT_PROFILE", value: config.ProfileOffline},
		{name: "EMBEDDING_PROVIDER", value: "OpenAI"},
		{name: "EMBEDDING_MODEL", value: "BAAI/bge-small-en-v1.5"},
		{name: "EMBEDDING_BATCH_SIZE", value: "8"},
		{name: "OPENAI_API_KEY", value: ""},
		{name: "OPENAI_BASE_URL", value: "http://" + lmdAddress + "/v1"},
		{name: "MILVUS_ADDRESS", value: milvusAddress},
		{name: "MILVUS_TOKEN", value: ""},
		{name: "HYBRID_MODE", value: "true"},
		{name: "CUSTOM_IGNORE_PATTERNS", value: ""},
		{name: "CLAUDE_CONTEXT_INCLUDE_SUBMODULES", value: ""},
		{name: "CLAUDE_CONTEXT_BACKGROUND_SYNC", value: "false"},
		{name: "CLAUDE_CONTEXT_SYNC_INTERVAL_MS", value: "300000"},
		{name: "CLAUDE_CONTEXT_TRIGGER_WATCHER", value: "false"},
		{name: "CLAUDE_CONTEXT_FILE_WATCHER", value: "false"},
		{name: "CLAUDE_CONTEXT_SYNC_LOCK_STALE_MS", value: "600000"},
		{name: "CLAUDE_CONTEXT_DEBUG_LISTENER", value: "false"},
		{name: "CLAUDE_CONTEXT_DEBUG_LISTEN_ADDR", value: "127.0.0.1:0"},
		{name: "CLAUDE_CONTEXT_PERF_COUNTERS_INTERVAL_MS", value: "0"},
		{name: "CLAUDE_CONTEXT_MAX_CONCURRENT_INDEX_JOBS", value: "1"},
		{name: "CLAUDE_CONTEXT_RESUME_ON_BOOT", value: "false"},
	}
	for _, variable := range environment {
		t.Setenv(variable.name, variable.value)
	}

	resolvedConfig, err := config.Default()
	if err != nil {
		t.Fatalf("resolve offline config through config.Default: %v", err)
	}
	requireOfflineConfig(t, resolvedConfig)
	return resolvedConfig
}

func prepareState(t *testing.T, daemonConfig config.Config) {
	t.Helper()
	slog.Debug(
		"offline live state preparation started",
		"state_root",
		daemonConfig.StateRoot,
	)

	directories := []string{
		daemonConfig.StateRoot,
		daemonConfig.LogsDir,
		daemonConfig.MerkleDir,
		daemonConfig.LocksDir,
		daemonConfig.SocketsDir,
		daemonConfig.ChunksDir,
		daemonConfig.GraphDir,
		daemonConfig.ContextRoot,
	}
	for _, directory := range directories {
		if err := store.EnsureDir(directory); err != nil {
			t.Fatalf("create state directory %s: %v", directory, err)
		}
	}
	if err := store.WriteRegistry(
		daemonConfig.RegistryPath,
		model.RegistryFile{},
	); err != nil {
		t.Fatalf("write empty registry: %v", err)
	}
	slog.Debug(
		"offline live state preparation completed",
		"state_root",
		daemonConfig.StateRoot,
	)
}

func fixtureDirectory(t *testing.T) string {
	t.Helper()
	slog.Debug("offline live fixture resolution started")

	_, sourcePath, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve offline harness source path")
	}
	fixturePath := filepath.Join(filepath.Dir(sourcePath), "fixture")
	info, err := os.Stat(fixturePath)
	if err != nil {
		t.Fatalf("stat fixture directory: %v", err)
	}
	if !info.IsDir() {
		t.Fatalf("fixture path %s is not a directory", fixturePath)
	}
	slog.Debug(
		"offline live fixture resolution completed",
		"fixture_path",
		fixturePath,
	)
	return fixturePath
}

func requireOfflineConfig(t *testing.T, daemonConfig config.Config) {
	t.Helper()

	if daemonConfig.Profile != config.ProfileOffline {
		t.Fatalf(
			"Profile = %q, want %q",
			daemonConfig.Profile,
			config.ProfileOffline,
		)
	}
	if daemonConfig.MilvusAddress != "" {
		t.Fatalf(
			"offline MilvusAddress = %q, want empty",
			daemonConfig.MilvusAddress,
		)
	}
	if daemonConfig.EmbeddingProvider != config.EmbeddingProviderONNX {
		t.Fatalf(
			"offline EmbeddingProvider = %q, want %q",
			daemonConfig.EmbeddingProvider,
			config.EmbeddingProviderONNX,
		)
	}
	if daemonConfig.IndexBackend != config.IndexBackendLocal {
		t.Fatalf(
			"offline IndexBackend = %q, want %q",
			daemonConfig.IndexBackend,
			config.IndexBackendLocal,
		)
	}
}

func (harness *harness) teardown(stopServer func()) {
	slog.Debug(
		"offline live harness teardown started",
		"state_root",
		harness.config.StateRoot,
	)
	if err := harness.connection.Close(); err != nil {
		harness.t.Errorf("close isolated daemon connection: %v", err)
	}
	stopServer()
	harness.manager.CloseGraphEngines()
	harness.dialGuard.close()
	harness.assertNoExternalDials()
	harness.assertProductionUntouched()
	slog.Debug(
		"offline live harness teardown completed",
		"state_root",
		harness.config.StateRoot,
	)
}

func (harness *harness) indexFixture() *pb.Job {
	harness.t.Helper()

	response, err := harness.client.StartIndex(
		correlatedContext(),
		&pb.StartIndexRequest{
			Path: harness.fixturePath,
			Splitter: &pb.SplitterConfig{
				Type: "ast",
			},
			Client: harnessClientInfo(),
		},
	)
	if err != nil {
		harness.t.Fatalf("start offline fixture index: %v", err)
	}
	if response.GetJobId() == "" {
		harness.t.Fatal("start offline fixture index returned an empty job id")
	}
	return harness.waitForJob(response.GetJobId())
}

func (harness *harness) waitForJob(jobID string) *pb.Job {
	harness.t.Helper()

	waitContext, cancel := context.WithTimeout(context.Background(), jobPollTimeout)
	defer cancel()
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()
	for {
		response, err := harness.client.GetJob(
			correlatedContext(),
			&pb.GetJobRequest{JobId: jobID},
		)
		if err != nil {
			harness.t.Fatalf("get offline index job %s: %v", jobID, err)
		}
		job := response.GetJob()
		if job != nil {
			switch job.GetState() {
			case string(model.JobStateCompleted),
				string(model.JobStateFailed),
				string(model.JobStateCancelled):
				return job
			}
		}
		select {
		case <-waitContext.Done():
			harness.t.Fatalf(
				"offline index job %s did not finish within %s",
				jobID,
				jobPollTimeout,
			)
		case <-ticker.C:
		}
	}
}

func requireCompleted(t *testing.T, job *pb.Job) {
	t.Helper()

	if job.GetState() == string(model.JobStateCompleted) {
		return
	}
	errorMessage := ""
	if job.GetError() != nil {
		errorMessage = job.GetError().GetMessage()
	}
	t.Fatalf(
		"offline index job state = %q, want %q (error: %s)",
		job.GetState(),
		model.JobStateCompleted,
		errorMessage,
	)
}

func (harness *harness) indexStatus() *pb.GetIndexResponse {
	harness.t.Helper()

	response, err := harness.client.GetIndex(
		correlatedContext(),
		&pb.GetIndexRequest{
			Path:   harness.fixturePath,
			Client: harnessClientInfo(),
		},
	)
	if err != nil {
		harness.t.Fatalf("get offline fixture index status: %v", err)
	}
	if !response.GetTracked() {
		harness.t.Fatalf(
			"offline fixture is not tracked:\n%s",
			response.GetDisplayText(),
		)
	}
	return response
}

func (harness *harness) assertOfflineRuntime(
	job *pb.Job,
	status *pb.GetIndexResponse,
) {
	harness.t.Helper()
	requireOfflineConfig(harness.t, harness.config)

	requireRuntimeIndexConfig(harness.t, "completed job", job.GetConfig())
	if job.GetProgress().GetEmbeddingBatchesCompleted() <= 0 {
		harness.t.Fatal(
			"completed offline job reported no embedding batches",
		)
	}

	codebase := status.GetCodebase()
	requireRuntimeIndexConfig(
		harness.t,
		"indexed codebase",
		codebase.GetEffectiveConfig(),
	)
	collectionName := codebase.GetCollectionName()
	if !strings.HasPrefix(collectionName, localCollectionPrefix) {
		harness.t.Fatalf(
			"offline collection name = %q, want prefix %q",
			collectionName,
			localCollectionPrefix,
		)
	}
	collectionPath := filepath.Join(
		harness.config.StateRoot,
		"localvec",
		collectionName,
	)
	collectionInfo, err := os.Stat(collectionPath)
	if err != nil {
		harness.t.Fatalf(
			"stat offline local collection %s: %v",
			collectionPath,
			err,
		)
	}
	if !collectionInfo.IsDir() {
		harness.t.Fatalf(
			"offline local collection %s is not a directory",
			collectionPath,
		)
	}
}

func requireRuntimeIndexConfig(
	t *testing.T,
	source string,
	indexConfig *pb.IndexConfig,
) {
	t.Helper()

	if indexConfig == nil {
		t.Fatalf("%s did not report effective index config", source)
	}
	if indexConfig.GetVectorBackend() != config.IndexBackendLocal {
		t.Fatalf(
			"%s vector backend = %q, want %q",
			source,
			indexConfig.GetVectorBackend(),
			config.IndexBackendLocal,
		)
	}
	if indexConfig.GetEmbeddingProvider() != config.EmbeddingProviderONNX {
		t.Fatalf(
			"%s embedding provider = %q, want %q",
			source,
			indexConfig.GetEmbeddingProvider(),
			config.EmbeddingProviderONNX,
		)
	}
}

func (harness *harness) search(
	query string,
	limit int32,
) *pb.SearchCodeResponse {
	harness.t.Helper()

	response, err := harness.client.SearchCode(
		correlatedContext(),
		&pb.SearchCodeRequest{
			Path:   harness.fixturePath,
			Query:  query,
			Limit:  limit,
			Client: harnessClientInfo(),
		},
	)
	if err != nil {
		harness.t.Fatalf("search offline fixture: %v", err)
	}
	return response
}

func containsTargetResult(
	results []*pb.SearchResult,
	targetPath string,
	targetFunction string,
) bool {
	for _, result := range results {
		if result.GetRelativePath() == targetPath &&
			strings.Contains(result.GetContent(), targetFunction) {
			return true
		}
	}
	return false
}

func (harness *harness) measureQueryP50(
	query string,
	limit int32,
	measurementCount int,
) time.Duration {
	harness.t.Helper()

	if measurementCount <= 0 {
		harness.t.Fatal("query measurement count must be positive")
	}
	durations := make([]time.Duration, 0, measurementCount)
	for measurementIndex := 0; measurementIndex < measurementCount; measurementIndex++ {
		startedAt := clock.Now()
		response := harness.search(query, limit)
		duration := clock.Now().Sub(startedAt)
		if !containsTargetResult(
			response.GetResults(),
			targetRelativePath,
			targetFunctionName,
		) {
			harness.t.Fatalf(
				"measured offline search %d omitted target %q",
				measurementIndex+1,
				targetRelativePath,
			)
		}
		durations = append(durations, duration)
	}
	sort.Slice(durations, func(leftIndex int, rightIndex int) bool {
		return durations[leftIndex] < durations[rightIndex]
	})
	return durations[len(durations)/2]
}

func (harness *harness) waitForGraphTrace(
	functionName string,
) *pb.GraphToolResponse {
	harness.t.Helper()

	arguments, err := json.Marshal(struct {
		FunctionName string `json:"function_name"`
		Direction    string `json:"direction"`
		Depth        int    `json:"depth"`
	}{
		FunctionName: functionName,
		Direction:    "inbound",
		Depth:        1,
	})
	if err != nil {
		harness.t.Fatalf("marshal graph trace arguments: %v", err)
	}

	waitContext, cancel := context.WithTimeout(
		context.Background(),
		graphPollTimeout,
	)
	defer cancel()
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()
	var lastResponse *pb.GraphToolResponse
	var lastError error
	for {
		lastResponse, lastError = harness.client.GraphTool(
			correlatedContext(),
			&pb.GraphToolRequest{
				Path:     harness.fixturePath,
				Client:   harnessClientInfo(),
				ToolName: "trace_path",
				ArgsJson: string(arguments),
			},
		)
		if lastError == nil &&
			strings.Contains(lastResponse.GetResultJson(), callerFunctionName) {
			return lastResponse
		}
		select {
		case <-waitContext.Done():
			if lastError != nil {
				harness.t.Fatalf(
					"offline graph trace did not become ready within %s: %v",
					graphPollTimeout,
					lastError,
				)
			}
			harness.t.Fatalf(
				"offline graph trace did not include caller %q within %s:\n%s",
				callerFunctionName,
				graphPollTimeout,
				lastResponse.GetResultJson(),
			)
		case <-ticker.C:
		}
	}
}

func startInProcessServer(
	t *testing.T,
	manager *daemon.Manager,
	socketPath string,
) func() {
	t.Helper()

	if err := os.MkdirAll(filepath.Dir(socketPath), 0o755); err != nil {
		t.Fatalf("create socket directory: %v", err)
	}
	if err := os.Remove(socketPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("remove stale socket: %v", err)
	}
	listener, err := (&net.ListenConfig{}).Listen(
		context.Background(),
		"unix",
		socketPath,
	)
	if err != nil {
		t.Fatalf("listen on isolated unix socket: %v", err)
	}

	server := grpc.NewServer(
		grpc.MaxRecvMsgSize(grpcutil.MaxMessageBytes),
		grpc.MaxSendMsgSize(grpcutil.MaxMessageBytes),
	)
	pb.RegisterSemanticSearchDaemonServiceServer(
		server,
		daemon.NewGRPCServer(manager, nil),
	)
	serverContext := context.Background()
	slog.InfoContext(
		serverContext,
		"offline live gRPC server started",
		"socket_path",
		socketPath,
	)
	goSafe(serverContext, "offline live gRPC server panicked", func() {
		if serveErr := server.Serve(listener); serveErr != nil &&
			!errors.Is(serveErr, grpc.ErrServerStopped) {
			slog.ErrorContext(
				serverContext,
				"offline live gRPC server failed",
				"socket_path",
				socketPath,
				"err",
				serveErr,
			)
		}
	})
	return func() {
		server.GracefulStop()
		if closeErr := listener.Close(); closeErr != nil &&
			!errors.Is(closeErr, net.ErrClosed) {
			t.Errorf("close isolated unix listener: %v", closeErr)
		}
		if removeErr := os.Remove(socketPath); removeErr != nil &&
			!errors.Is(removeErr, os.ErrNotExist) {
			t.Errorf("remove isolated unix socket: %v", removeErr)
		}
		slog.InfoContext(
			serverContext,
			"offline live gRPC server stopped",
			"socket_path",
			socketPath,
		)
	}
}

func goSafe(ctx context.Context, panicMessage string, run func()) {
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.ErrorContext(
					ctx,
					panicMessage,
					"err",
					fmt.Errorf("panic: %v", recovered),
				)
			}
		}()
		run()
	}()
}

func correlatedContext() context.Context {
	return grpcutil.WithCorrelation(context.Background())
}

func harnessClientInfo() *pb.ClientInfo {
	return &pb.ClientInfo{Name: "offline-live-harness"}
}

func (harness *harness) assertProductionUntouched() {
	postPids, err := snapshotProductionDaemonPids(
		context.Background(),
		runCommandOutput,
	)
	if err != nil {
		harness.t.Errorf(
			"snapshot production daemons after offline validation: %v",
			err,
		)
		return
	}
	for processID := range harness.prodPidsPre {
		if !postPids[processID] {
			harness.t.Errorf(
				"production daemon pid %d disappeared during offline validation",
				processID,
			)
		}
	}
}

type commandOutputFunction func(
	context.Context,
	string,
	...string,
) ([]byte, error)

type exitCodeError interface {
	error
	ExitCode() int
}

func snapshotProductionDaemonPids(
	ctx context.Context,
	commandOutput commandOutputFunction,
) (map[int]bool, error) {
	processIDs := make(map[int]bool)
	output, err := commandOutput(
		ctx,
		"pgrep",
		"-f",
		daemonProcessName,
	)
	if err != nil {
		var exitError exitCodeError
		if errors.As(err, &exitError) && exitError.ExitCode() == 1 {
			slog.DebugContext(
				ctx,
				"offline live production daemon snapshot",
				"process_count",
				0,
			)
			return processIDs, nil
		}
		slog.ErrorContext(
			ctx,
			"snapshot production daemons failed",
			"executable",
			"pgrep",
			"err",
			err,
		)
		return nil, fmt.Errorf("run pgrep for production daemon: %w", err)
	}
	fields := strings.Fields(string(output))
	if len(fields) == 0 {
		slog.ErrorContext(
			ctx,
			"snapshot production daemons failed",
			"executable",
			"pgrep",
			"err",
			errors.New("pgrep returned no process ids"),
		)
		return nil, errors.New("pgrep succeeded without returning a process id")
	}
	for _, field := range fields {
		processID, conversionError := strconv.Atoi(field)
		if conversionError != nil {
			slog.ErrorContext(
				ctx,
				"parse production daemon process id failed",
				"value",
				field,
				"err",
				conversionError,
			)
			return nil, fmt.Errorf(
				"parse pgrep process id %q: %w",
				field,
				conversionError,
			)
		}
		if processID <= 0 {
			slog.ErrorContext(
				ctx,
				"validate production daemon process id failed",
				"pid",
				processID,
				"err",
				errors.New("process id must be positive"),
			)
			return nil, fmt.Errorf("pgrep returned invalid process id %d", processID)
		}
		isProduction, inspectError := isProductionDaemonPid(
			ctx,
			commandOutput,
			processID,
		)
		if inspectError != nil {
			return nil, inspectError
		}
		if isProduction {
			processIDs[processID] = true
		}
	}
	slog.DebugContext(
		ctx,
		"offline live production daemon snapshot",
		"process_count",
		len(processIDs),
	)
	return processIDs, nil
}

func isProductionDaemonPid(
	ctx context.Context,
	commandOutput commandOutputFunction,
	processID int,
) (bool, error) {
	output, err := commandOutput(
		ctx,
		"ps",
		"-o",
		"command=",
		"-p",
		strconv.Itoa(processID),
	)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"inspect production daemon candidate failed",
			"pid",
			processID,
			"executable",
			"ps",
			"err",
			err,
		)
		return false, fmt.Errorf(
			"inspect production daemon candidate pid %d with ps: %w",
			processID,
			err,
		)
	}
	command := strings.TrimSpace(string(output))
	if command == "" {
		slog.ErrorContext(
			ctx,
			"inspect production daemon candidate failed",
			"pid",
			processID,
			"executable",
			"ps",
			"err",
			errors.New("ps returned an empty command"),
		)
		return false, fmt.Errorf(
			"ps returned an empty command for production daemon candidate pid %d",
			processID,
		)
	}
	executablePath := command
	if separatorIndex := strings.IndexByte(command, ' '); separatorIndex >= 0 {
		executablePath = command[:separatorIndex]
	}
	return !underTempRoot(executablePath), nil
}

func underTempRoot(path string) bool {
	cleanPath := filepath.Clean(path)
	tempRoots := []string{
		"/tmp",
		"/private/tmp",
		"/private/var/folders",
		"/var/folders",
	}
	osTempRoot := filepath.Clean(os.TempDir())
	if osTempRoot != "" && osTempRoot != "." {
		tempRoots = append(tempRoots, osTempRoot)
	}
	for _, tempRoot := range tempRoots {
		if cleanPath == tempRoot ||
			strings.HasPrefix(
				cleanPath,
				tempRoot+string(os.PathSeparator),
			) {
			return true
		}
	}
	return false
}

type netDialGuard struct {
	mutex        sync.Mutex
	listeners    []net.Listener
	observations []string
	waitGroup    sync.WaitGroup
	closeOnce    sync.Once
}

func newNetDialGuard(t *testing.T, addresses []string) *netDialGuard {
	t.Helper()
	slog.Debug(
		"offline live dial guard setup started",
		"address_count",
		len(addresses),
	)

	guard := &netDialGuard{
		mutex:        sync.Mutex{},
		listeners:    make([]net.Listener, 0, len(addresses)),
		observations: make([]string, 0),
		waitGroup:    sync.WaitGroup{},
		closeOnce:    sync.Once{},
	}
	t.Cleanup(guard.close)
	skippedAddressCount := 0
	for _, address := range addresses {
		listener, err := (&net.ListenConfig{}).Listen(
			context.Background(),
			"tcp",
			address,
		)
		if err != nil {
			if errors.Is(err, syscall.EADDRINUSE) {
				skippedAddressCount++
				t.Logf(
					"offline dial tripwire skipped for %s because the address is occupied; daemon runtime config assertions cover dependency construction",
					address,
				)
				continue
			}
			t.Fatalf("bind offline dial tripwire for %s: %v", address, err)
		}
		guard.listeners = append(guard.listeners, listener)
		guard.waitGroup.Add(1)
		goSafe(
			context.Background(),
			"offline live dial tripwire panicked",
			func() {
				guard.acceptDials(listener)
			},
		)
	}
	slog.Debug(
		"offline live dial guard setup completed",
		"listener_count",
		len(guard.listeners),
		"skipped_address_count",
		skippedAddressCount,
	)
	return guard
}

func (guard *netDialGuard) acceptDials(listener net.Listener) {
	defer guard.waitGroup.Done()
	slog.Debug(
		"offline live dial tripwire started",
		"address",
		listener.Addr().String(),
	)
	defer slog.Debug(
		"offline live dial tripwire stopped",
		"address",
		listener.Addr().String(),
	)
	for {
		connection, err := listener.Accept()
		if err != nil {
			if !errors.Is(err, net.ErrClosed) {
				guard.record(
					fmt.Sprintf(
						"accept on %s failed: %v",
						listener.Addr(),
						err,
					),
				)
			}
			return
		}
		guard.record(
			fmt.Sprintf(
				"dial from %s to %s",
				connection.RemoteAddr(),
				connection.LocalAddr(),
			),
		)
		if closeErr := connection.Close(); closeErr != nil {
			guard.record(
				fmt.Sprintf(
					"close guarded connection to %s failed: %v",
					connection.LocalAddr(),
					closeErr,
				),
			)
		}
	}
}

func runCommandOutput(
	ctx context.Context,
	name string,
	arguments ...string,
) ([]byte, error) {
	slog.DebugContext(ctx, "offline live command started", "executable", name)
	output, err := exec.CommandContext(ctx, name, arguments...).Output()
	slog.DebugContext(
		ctx,
		"offline live command completed",
		"executable",
		name,
		"err",
		err,
	)
	return output, err
}

func (guard *netDialGuard) record(observation string) {
	guard.mutex.Lock()
	defer guard.mutex.Unlock()
	guard.observations = append(guard.observations, observation)
}

func (guard *netDialGuard) snapshot() []string {
	guard.mutex.Lock()
	defer guard.mutex.Unlock()

	return append([]string(nil), guard.observations...)
}

func (guard *netDialGuard) close() {
	guard.closeOnce.Do(func() {
		slog.Debug(
			"offline live dial guard teardown started",
			"listener_count",
			len(guard.listeners),
		)
		for _, listener := range guard.listeners {
			if err := listener.Close(); err != nil &&
				!errors.Is(err, net.ErrClosed) {
				guard.record(
					fmt.Sprintf(
						"close dial tripwire %s failed: %v",
						listener.Addr(),
						err,
					),
				)
			}
		}
		guard.waitGroup.Wait()
		slog.Debug("offline live dial guard teardown completed")
	})
}

func (harness *harness) assertNoExternalDials() {
	harness.t.Helper()

	observations := harness.dialGuard.snapshot()
	if len(observations) == 0 {
		return
	}
	harness.t.Errorf(
		"offline validation observed forbidden external dependency activity:\n%s",
		strings.Join(observations, "\n"),
	)
}

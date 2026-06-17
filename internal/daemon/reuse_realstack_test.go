package daemon

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/store"
	"google.golang.org/grpc"
)

const realStackReuseEnv = "LMS_REAL_STACK_TESTS"

func TestSameCodebaseReuseIntegrationRealSemanticService(t *testing.T) {
	requireRealStack(t)

	embedServer := newTestEmbeddingServer(t)
	manager, repoPath := newRealSemanticManager(t, embedServer.URL)
	initialContent := buildReuseFixture("changed-before")
	writeReuseFixture(t, repoPath, initialContent)

	indexConfig := defaultIndexConfig()
	indexConfig.SplitterChunkSize = 180
	indexConfig.SplitterOverlap = 40

	_, _, _, _, err := manager.StartIndex(context.Background(), repoPath, testClientInfo(), indexConfig, false)
	if err != nil {
		t.Fatalf("StartIndex returned error: %v", err)
	}
	waitForCodebaseSettled(t, manager, repoPath)

	updatedContent := buildReuseFixture("changed-after")
	if updatedContent == initialContent {
		t.Fatal("updated fixture matches initial fixture; want one changed region")
	}
	writeReuseFixture(t, repoPath, updatedContent)

	job, _, _, err := manager.SyncIndex(context.Background(), repoPath, testClientInfo())
	if err != nil {
		t.Fatalf("SyncIndex returned error: %v", err)
	}
	completed := waitForJobTerminal(t, manager, job.ID)

	if completed.State != model.JobStateCompleted {
		t.Fatalf("job state = %q, want completed", completed.State)
	}
	if completed.Progress.ChunksReused <= 0 {
		t.Fatalf("ChunksReused = %d, want > 0 for same-codebase unchanged chunks", completed.Progress.ChunksReused)
	}
	if completed.Progress.ChunksEmbedded <= 0 {
		t.Fatalf("ChunksEmbedded = %d, want > 0 for the changed chunk set", completed.Progress.ChunksEmbedded)
	}
	if completed.Progress.ChunksProcessed != completed.Progress.ChunksReused+completed.Progress.ChunksEmbedded {
		t.Fatalf(
			"ChunksProcessed = %d, want reused+embedded = %d+%d",
			completed.Progress.ChunksProcessed,
			completed.Progress.ChunksReused,
			completed.Progress.ChunksEmbedded,
		)
	}
	if completed.Progress.ReuseVectorsLoaded <= 0 {
		t.Fatalf("ReuseVectorsLoaded = %d, want > 0 after exact-path reuse load", completed.Progress.ReuseVectorsLoaded)
	}
}

func TestSameCodebaseReuseGRPCE2ERealSemanticService(t *testing.T) {
	requireRealStack(t)

	embedServer := newTestEmbeddingServer(t)
	manager, repoPath := newRealSemanticManager(t, embedServer.URL)
	socketPath := filepath.Join(os.TempDir(), fmt.Sprintf("lms-%d.sock", time.Now().UnixNano()))
	stopServer := startGRPCServerForTest(t, manager, socketPath)
	defer stopServer()

	initialContent := buildReuseFixture("rpc-before")
	writeReuseFixture(t, repoPath, initialContent)

	connection, client, err := grpcutil.DialDaemon(context.Background(), socketPath)
	if err != nil {
		t.Fatalf("DialDaemon returned error: %v", err)
	}
	defer connection.Close()

	startResponse, err := client.StartIndex(grpcutil.WithCorrelation(context.Background()), &pb.StartIndexRequest{
		Path:  repoPath,
		Force: false,
		Splitter: &pb.SplitterConfig{
			Type:      "ast",
			ChunkSize: 180,
			Overlap:   40,
		},
		Client: &pb.ClientInfo{Name: "grpc-e2e"},
	})
	if err != nil {
		t.Fatalf("StartIndex RPC returned error: %v", err)
	}
	waitForRPCJobTerminal(t, client, startResponse.GetJobId())

	updatedContent := buildReuseFixture("rpc-after")
	if updatedContent == initialContent {
		t.Fatal("updated fixture matches initial fixture; want one changed region")
	}
	writeReuseFixture(t, repoPath, updatedContent)

	syncResponse, err := client.SyncIndex(grpcutil.WithCorrelation(context.Background()), &pb.SyncIndexRequest{
		Path:   repoPath,
		Client: &pb.ClientInfo{Name: "grpc-e2e"},
	})
	if err != nil {
		t.Fatalf("SyncIndex RPC returned error: %v", err)
	}
	completed := waitForRPCJobTerminal(t, client, syncResponse.GetJobId())
	progress := completed.GetProgress()

	if completed.GetState() != string(model.JobStateCompleted) {
		t.Fatalf("job state = %q, want completed", completed.GetState())
	}
	if progress.GetChunksReused() <= 0 {
		t.Fatalf("ChunksReused = %d, want > 0 for unchanged chunks", progress.GetChunksReused())
	}
	if progress.GetChunksEmbedded() <= 0 {
		t.Fatalf("ChunksEmbedded = %d, want > 0 for the changed chunk set", progress.GetChunksEmbedded())
	}
	if progress.GetChunksProcessed() != progress.GetChunksReused()+progress.GetChunksEmbedded() {
		t.Fatalf(
			"ChunksProcessed = %d, want reused+embedded = %d+%d",
			progress.GetChunksProcessed(),
			progress.GetChunksReused(),
			progress.GetChunksEmbedded(),
		)
	}
	if progress.GetReuseVectorsLoaded() <= 0 {
		t.Fatalf("ReuseVectorsLoaded = %d, want > 0 after exact-path reuse load", progress.GetReuseVectorsLoaded())
	}
}

func TestQuarantineLargeDeleteIntegrationRealSemanticService(t *testing.T) {
	requireRealStack(t)

	embedServer := newTestEmbeddingServer(t)
	manager, repoPath := newRealSemanticManager(t, embedServer.URL)
	service, ok := manager.semantic.(*semantic.Service)
	if !ok {
		t.Fatalf("manager.semantic type = %T, want *semantic.Service", manager.semantic)
	}

	seeded := seedRealIndexedCodebase(t, manager, repoPath, 120)

	beforeCount, err := service.Count(context.Background(), seeded.CanonicalPath)
	if err != nil {
		t.Fatalf("Count(before) returned error: %v", err)
	}
	if beforeCount <= 0 {
		t.Fatalf("Count(before) = %d, want > 0", beforeCount)
	}

	removeLargeDeleteFixturePrefix(t, repoPath, 110)

	job, _, _, err := manager.SyncIndex(context.Background(), repoPath, testClientInfo())
	if err != nil {
		t.Fatalf("SyncIndex returned error: %v", err)
	}
	completed := waitForJobTerminal(t, manager, job.ID)
	if completed.State != model.JobStateFailed {
		t.Fatalf("job state = %q, want failed quarantine terminal state", completed.State)
	}
	if completed.Error == nil || !strings.Contains(completed.Error.Message, "quarantined") {
		t.Fatalf("job error = %+v, want quarantined message", completed.Error)
	}

	codebase, _, found, _, err := manager.GetIndex(context.Background(), repoPath)
	if err != nil || !found {
		t.Fatalf("GetIndex returned err=%v found=%v", err, found)
	}
	if codebase.ID != seeded.ID {
		t.Fatalf("codebase ID = %q, want seeded ID %q", codebase.ID, seeded.ID)
	}
	if codebase.Status != model.CodebaseStatusQuarantined {
		t.Fatalf("codebase status = %q, want quarantined", codebase.Status)
	}
	if codebase.Quarantine == nil {
		t.Fatal("codebase.Quarantine = nil, want recorded quarantine state")
	}
	if codebase.Quarantine.LastMissingCount != 110 || codebase.Quarantine.LastTotalCount != 120 {
		t.Fatalf("quarantine = %+v, want 110 of 120", codebase.Quarantine)
	}

	afterCount, err := service.Count(context.Background(), seeded.CanonicalPath)
	if err != nil {
		t.Fatalf("Count(after) returned error: %v", err)
	}
	if afterCount != beforeCount {
		t.Fatalf("Count(after) = %d, want unchanged %d while quarantined", afterCount, beforeCount)
	}
}

func TestQuarantineLargeDeleteGRPCE2ERealSemanticService(t *testing.T) {
	requireRealStack(t)

	embedServer := newTestEmbeddingServer(t)
	manager, repoPath := newRealSemanticManager(t, embedServer.URL)
	socketPath := filepath.Join(os.TempDir(), fmt.Sprintf("lms-%d.sock", time.Now().UnixNano()))
	stopServer := startGRPCServerForTest(t, manager, socketPath)
	defer stopServer()

	seeded := seedRealIndexedCodebase(t, manager, repoPath, 120)

	connection, client, err := grpcutil.DialDaemon(context.Background(), socketPath)
	if err != nil {
		t.Fatalf("DialDaemon returned error: %v", err)
	}
	defer connection.Close()

	removeLargeDeleteFixturePrefix(t, repoPath, 110)

	syncResponse, err := client.SyncIndex(grpcutil.WithCorrelation(context.Background()), &pb.SyncIndexRequest{
		Path:   repoPath,
		Client: &pb.ClientInfo{Name: "grpc-e2e"},
	})
	if err != nil {
		t.Fatalf("SyncIndex RPC returned error: %v", err)
	}
	completed := waitForRPCJobTerminal(t, client, syncResponse.GetJobId())
	if completed.GetState() != string(model.JobStateFailed) {
		t.Fatalf("job state = %q, want failed quarantine terminal state", completed.GetState())
	}

	indexResponse, err := client.GetIndex(grpcutil.WithCorrelation(context.Background()), &pb.GetIndexRequest{
		Path: repoPath,
	})
	if err != nil {
		t.Fatalf("GetIndex RPC returned error: %v", err)
	}
	if indexResponse.GetCodebase().GetId() != seeded.ID {
		t.Fatalf("codebase id = %q, want seeded ID %q", indexResponse.GetCodebase().GetId(), seeded.ID)
	}
	if indexResponse.GetCodebase().GetStatus() != string(model.CodebaseStatusQuarantined) {
		t.Fatalf("codebase status = %q, want quarantined", indexResponse.GetCodebase().GetStatus())
	}
	if !strings.Contains(indexResponse.GetDisplayText(), "quarantined after a suspicious large disappearance") {
		t.Fatalf("GetIndex display text missing quarantine explanation:\n%s", indexResponse.GetDisplayText())
	}
	if !strings.Contains(indexResponse.GetDisplayText(), "last known-good index") {
		t.Fatalf("GetIndex display text missing last-known-good wording:\n%s", indexResponse.GetDisplayText())
	}

	searchResponse, err := client.SearchCode(grpcutil.WithCorrelation(context.Background()), &pb.SearchCodeRequest{
		Path:   repoPath,
		Query:  "package main",
		Limit:  5,
		Client: &pb.ClientInfo{Name: "grpc-e2e"},
	})
	if err != nil {
		t.Fatalf("SearchCode RPC returned error while quarantined: %v", err)
	}
	if !strings.Contains(searchResponse.GetDisplayText(), "last known-good index") {
		t.Fatalf("SearchCode display text missing quarantine search note:\n%s", searchResponse.GetDisplayText())
	}
}

func requireRealStack(t *testing.T) {
	t.Helper()
	if os.Getenv(realStackReuseEnv) != "1" {
		t.Skipf("%s is not set; skipping real-stack reuse test", realStackReuseEnv)
	}
}

func newRealSemanticManager(t *testing.T, openAIBaseURL string) (*Manager, string) {
	t.Helper()

	defaultConfig, err := config.Default()
	if err != nil {
		t.Fatalf("config.Default returned error: %v", err)
	}
	if strings.TrimSpace(defaultConfig.MilvusAddress) == "" {
		t.Fatal("MilvusAddress is empty; set machine-local Clyde/lm-semantic-search config before running real-stack tests")
	}

	stateRoot := t.TempDir()
	repoPath := filepath.Join(t.TempDir(), "repo")
	if err := os.MkdirAll(repoPath, 0o755); err != nil {
		t.Fatalf("MkdirAll(repo) returned error: %v", err)
	}

	cfg := config.Config{
		StateRoot:                 stateRoot,
		SocketPath:                filepath.Join(stateRoot, "sockets", "lm-semantic-search-daemon.sock"),
		RegistryPath:              filepath.Join(stateRoot, "registry.json"),
		JobsPath:                  filepath.Join(stateRoot, "jobs.jsonl"),
		EventsPath:                filepath.Join(stateRoot, "events.jsonl"),
		LogsDir:                   filepath.Join(stateRoot, "logs"),
		LogPath:                   filepath.Join(stateRoot, "logs", "lm-semantic-search-daemon.log"),
		MerkleDir:                 filepath.Join(stateRoot, "merkle"),
		LocksDir:                  filepath.Join(stateRoot, "locks"),
		SocketsDir:                filepath.Join(stateRoot, "sockets"),
		ChunksDir:                 filepath.Join(stateRoot, "chunks"),
		ContextRoot:               filepath.Join(stateRoot, "context"),
		EmbeddingProvider:         "OpenAI",
		EmbeddingModel:            "text-embedding-3-small",
		EmbeddingBatchSize:        8,
		EmbeddingBatchTokenBudget: 1000,
		EmbeddingDimension:        3,
		OpenAIAPIKey:              "test-key",
		OpenAIBaseURL:             openAIBaseURL,
		QueryInstructionPrefix:    "",
		MilvusAddress:             defaultConfig.MilvusAddress,
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
	for _, path := range []string{
		cfg.StateRoot,
		cfg.LogsDir,
		cfg.MerkleDir,
		cfg.LocksDir,
		cfg.SocketsDir,
		cfg.ChunksDir,
		cfg.ContextRoot,
	} {
		if err := store.EnsureDir(path); err != nil {
			t.Fatalf("EnsureDir(%s) returned error: %v", path, err)
		}
	}
	if err := store.WriteRegistry(cfg.RegistryPath, model.RegistryFile{}); err != nil {
		t.Fatalf("WriteRegistry returned error: %v", err)
	}

	manager, err := NewManager(context.Background(), cfg)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}
	service, ok := manager.semantic.(*semantic.Service)
	if !ok {
		t.Fatalf("manager.semantic type = %T, want *semantic.Service", manager.semantic)
	}
	if !service.Available() {
		t.Fatal("semantic service is not available; verify the local Milvus stack before running real-stack tests")
	}
	t.Cleanup(func() {
		if _, clearErr := manager.ClearIndex(context.Background(), repoPath, testClientInfo()); clearErr != nil && !strings.Contains(clearErr.Error(), "codebase not tracked") {
			t.Fatalf("ClearIndex cleanup returned error: %v", clearErr)
		}
		if closeErr := service.Close(context.Background()); closeErr != nil {
			t.Fatalf("semantic service Close returned error: %v", closeErr)
		}
	})
	return manager, repoPath
}

func newTestEmbeddingServer(t *testing.T) *httptest.Server {
	t.Helper()

	handler := http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var payload struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}
		type responseRow struct {
			Embedding []float64 `json:"embedding"`
		}
		type responseBody struct {
			Data []responseRow `json:"data"`
		}
		rows := make([]responseRow, 0, len(payload.Input))
		for _, text := range payload.Input {
			rows = append(rows, responseRow{Embedding: deterministicTestVector(text)})
		}
		writer.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(writer).Encode(responseBody{Data: rows}); err != nil {
			http.Error(writer, err.Error(), http.StatusInternalServerError)
		}
	})
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return server
}

func deterministicTestVector(text string) []float64 {
	sum := 0
	for _, byteValue := range []byte(text) {
		sum += int(byteValue)
	}
	return []float64{
		float64(len(text)),
		float64(strings.Count(text, "\n") + 1),
		float64(sum % 997),
	}
}

func buildReuseFixture(changedToken string) string {
	var builder strings.Builder
	builder.WriteString("package main\n\n")
	for index := 0; index < 8; index++ {
		builder.WriteString(fmt.Sprintf("func keepPrefix%d() string {\n", index))
		builder.WriteString(fmt.Sprintf("    value := %q\n", strings.Repeat(fmt.Sprintf("keep-prefix-%d-", index), 18)))
		builder.WriteString("    return value\n")
		builder.WriteString("}\n\n")
	}
	builder.WriteString("func changedMiddle() string {\n")
	builder.WriteString(fmt.Sprintf("    value := %q\n", strings.Repeat(changedToken+"-", 24)))
	builder.WriteString("    return value\n")
	builder.WriteString("}\n\n")
	for index := 8; index < 16; index++ {
		builder.WriteString(fmt.Sprintf("func keepSuffix%d() string {\n", index))
		builder.WriteString(fmt.Sprintf("    value := %q\n", strings.Repeat(fmt.Sprintf("keep-suffix-%d-", index), 18)))
		builder.WriteString("    return value\n")
		builder.WriteString("}\n\n")
	}
	return builder.String()
}

func writeReuseFixture(t *testing.T, repoPath string, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(repoPath, "main.go"), []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile(main.go) returned error: %v", err)
	}
}

func writeLargeDeleteFixture(t *testing.T, repoPath string, fileCount int) {
	t.Helper()
	for index := 0; index < fileCount; index++ {
		path := filepath.Join(repoPath, fmt.Sprintf("f%03d.go", index))
		content := fmt.Sprintf("package main\n\nfunc fixture_%03d() string {\n\treturn %q\n}\n", index, strings.Repeat(fmt.Sprintf("fixture-%03d-", index), 12))
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatalf("WriteFile(%s) returned error: %v", path, err)
		}
	}
}

func seedRealIndexedCodebase(t *testing.T, manager *Manager, repoPath string, fileCount int) model.Codebase {
	t.Helper()

	writeLargeDeleteFixture(t, repoPath, fileCount)
	service, ok := manager.semantic.(*semantic.Service)
	if !ok {
		t.Fatalf("manager.semantic type = %T, want *semantic.Service", manager.semantic)
	}
	canonicalPath, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks(%s) returned error: %v", repoPath, err)
	}

	indexConfig := manager.enrichIndexConfig(defaultIndexConfig())
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)
	chunks := make([]model.StoredChunk, 0, fileCount)
	fileHashes := make(map[string]string, fileCount)
	for index := 0; index < fileCount; index++ {
		relativePath := fmt.Sprintf("f%03d.go", index)
		content := fmt.Sprintf("package main\n\nfunc fixture_%03d() string {\n\treturn %q\n}\n", index, strings.Repeat(fmt.Sprintf("fixture-%03d-", index), 12))
		fileHashes[relativePath] = hashText(content)
		chunks = append(chunks, model.StoredChunk{
			Content:       content,
			RelativePath:  relativePath,
			StartLine:     1,
			EndLine:       5,
			Language:      "go",
			FileExtension: ".go",
		})
	}

	if err := service.StageReindex(context.Background(), canonicalPath, chunks, semantic.RemovePaths(nil), nil, nil); err != nil {
		t.Fatalf("semantic.StageReindex seed returned error: %v", err)
	}
	if err := service.PromoteStaging(context.Background(), canonicalPath); err != nil {
		t.Fatalf("semantic.PromoteStaging seed returned error: %v", err)
	}

	codebase := newCodebaseRecord(canonicalPath)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = indexConfig
	codebase.CollectionName = service.CollectionName(canonicalPath)
	codebase.MerkleSnapshotPath = manager.snapshotPathForCodebase(codebase)
	codebase.LastSuccessfulRun = &model.IndexRunSummary{
		IndexedFiles: int32(fileCount),
		TotalChunks:  int32(len(chunks)),
		Status:       "completed",
		CompletedAt:  time.Now(),
	}
	codebase.LiveFileTotal = int32(fileCount)
	codebase.LiveChunkTotal = int32(len(chunks))

	snapshot := merkle.Snapshot{
		ConfigDigest: indexConfig.IgnoreDigest,
		Files:        fileHashes,
		Inodes:       nil,
	}
	if err := merkle.WriteSnapshot(manager.snapshotPathForCodebase(codebase), snapshot); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	if err := manager.saveLocked(); err != nil {
		manager.mu.Unlock()
		t.Fatalf("saveLocked returned error: %v", err)
	}
	manager.mu.Unlock()

	return codebase
}

func removeLargeDeleteFixturePrefix(t *testing.T, repoPath string, removeCount int) {
	t.Helper()
	for index := 0; index < removeCount; index++ {
		path := filepath.Join(repoPath, fmt.Sprintf("f%03d.go", index))
		if err := os.Remove(path); err != nil {
			t.Fatalf("Remove(%s) returned error: %v", path, err)
		}
	}
}

func waitForJobTerminal(t *testing.T, manager *Manager, jobID string) model.Job {
	t.Helper()
	var settled model.Job
	waitForCondition(t, func() bool {
		job, found := manager.GetJob(jobID)
		if !found {
			return false
		}
		settled = job
		return job.State == model.JobStateCompleted || job.State == model.JobStateFailed || job.State == model.JobStateCancelled
	})
	return settled
}

func startGRPCServerForTest(t *testing.T, manager *Manager, socketPath string) func() {
	t.Helper()

	if err := os.MkdirAll(filepath.Dir(socketPath), 0o755); err != nil {
		t.Fatalf("MkdirAll(socket dir) returned error: %v", err)
	}
	_ = os.Remove(socketPath)

	listener, err := (&net.ListenConfig{}).Listen(context.Background(), "unix", socketPath)
	if err != nil {
		t.Fatalf("Listen(unix) returned error: %v", err)
	}
	server := grpc.NewServer(
		grpc.MaxRecvMsgSize(grpcutil.MaxMessageBytes),
		grpc.MaxSendMsgSize(grpcutil.MaxMessageBytes),
	)
	pb.RegisterSemanticSearchDaemonServiceServer(server, NewGRPCServer(manager, nil))
	go func() {
		_ = server.Serve(listener)
	}()
	return func() {
		server.GracefulStop()
		_ = listener.Close()
		_ = os.Remove(socketPath)
	}
}

func waitForRPCJobTerminal(t *testing.T, client pb.SemanticSearchDaemonServiceClient, jobID string) *pb.Job {
	t.Helper()

	deadline := time.Now().Add(2 * time.Minute)
	for time.Now().Before(deadline) {
		response, err := client.GetJob(grpcutil.WithCorrelation(context.Background()), &pb.GetJobRequest{JobId: jobID})
		if err != nil {
			t.Fatalf("GetJob(%s) returned error: %v", jobID, err)
		}
		job := response.GetJob()
		if job != nil {
			switch job.GetState() {
			case string(model.JobStateCompleted), string(model.JobStateFailed), string(model.JobStateCancelled):
				return job
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("job %s did not reach a terminal state before the timeout", jobID)
	return nil
}

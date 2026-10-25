package mcpserver_test

import (
	"context"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	mcpclient "github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/mcpserver"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/response"
	"goodkind.io/lm-semantic-search/internal/sandbox"
	"goodkind.io/lm-semantic-search/internal/store"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/encoding/protojson"
)

const (
	statusResponseLimit  = time.Second
	statusProcessEnv     = "LMS_STATUS_TEST_PROCESS"
	statusTestCodebaseID = "status-test"
)

func TestStatusMCPProcess(t *testing.T) {
	if os.Getenv(statusProcessEnv) != "1" {
		t.Skip("The stdio MCP client starts this subprocess helper.")
	}
	if err := mcpserver.Run(context.Background()); err != nil {
		t.Fatalf("serve stdio MCP: %v", err)
	}
}

func TestIndexingStatusReturnsCurrentProgressWithoutWaiting(t *testing.T) {
	for _, state := range []model.CodebaseStatus{
		model.CodebaseStatusDiscovered,
		model.CodebaseStatusPending,
		model.CodebaseStatusIndexing,
	} {
		t.Run(string(state), func(t *testing.T) {
			socketPath, codebasePath := startStatusTestDaemon(t, state)
			client := newStatusTestClient(t, socketPath)
			started := time.Now()
			status := callStatusTestTool(t, client, "get_indexing_status", map[string]any{
				"absolutePath": codebasePath,
			})
			elapsed := time.Since(started)
			t.Logf("get_indexing_status state=%s searchable=%t elapsed=%s", status.GetCodebase().GetStatus(), status.GetSearchable(), elapsed)
			if elapsed >= statusResponseLimit {
				t.Fatalf("get_indexing_status took %s; it must return current progress within %s", elapsed, statusResponseLimit)
			}
			if !status.GetTracked() || status.GetSearchable() || status.GetCodebase().GetStatus() != string(state) {
				t.Fatalf("status = %v, want tracked %s without a searchable index", status, state)
			}
		})
	}
}

func TestWaitForIndexingStillWaitsForCurrentProgress(t *testing.T) {
	socketPath, codebasePath := startStatusTestDaemon(t, model.CodebaseStatusPending)
	client := newStatusTestClient(t, socketPath)
	started := time.Now()
	status := callStatusTestTool(t, client, "wait_for_indexing", map[string]any{
		"absolutePath":         codebasePath,
		"wait_timeout_seconds": 1,
	})
	elapsed := time.Since(started)
	t.Logf("wait_for_indexing state=%s searchable=%t elapsed=%s", status.GetCodebase().GetStatus(), status.GetSearchable(), elapsed)
	if elapsed < statusResponseLimit {
		t.Fatalf("wait_for_indexing returned after %s before its one-second timeout", elapsed)
	}
	if status.GetSearchable() || status.GetCodebase().GetStatus() != string(model.CodebaseStatusPending) {
		t.Fatalf("status = %v, want current pending progress after timeout", status)
	}
}

func TestIndexingStatusReturnsQueuedJobWithoutWaiting(t *testing.T) {
	socketPath, codebasePath := startStatusTestDaemon(t, model.CodebaseStatusNotIndexed)
	client := newStatusTestClient(t, socketPath)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	request := statusToolRequest(map[string]any{
		"absolutePath":       codebasePath,
		"priority":           "normal",
		"quiet":              true,
		"idle_after_seconds": 2147483647,
	})
	request.Params.Name = "index_codebase"
	result, err := client.CallTool(ctx, request)
	if err != nil || result == nil || result.IsError {
		t.Fatalf("queue index_codebase result=%v error=%v", result, err)
	}
	started := time.Now()
	status := callStatusTestTool(t, client, "get_indexing_status", map[string]any{
		"absolutePath": codebasePath,
	})
	elapsed := time.Since(started)
	t.Logf("get_indexing_status state=%s job=%s searchable=%t elapsed=%s", status.GetCodebase().GetStatus(), status.GetActiveJob().GetState(), status.GetSearchable(), elapsed)
	if elapsed >= statusResponseLimit || status.GetSearchable() || status.GetActiveJob().GetState() != string(model.JobStateQueued) {
		t.Fatalf("status after %s = %v, want queued progress within %s", elapsed, status, statusResponseLimit)
	}
}

func TestIndexingStatusReportsUnavailableDaemon(t *testing.T) {
	client := newStatusTestClient(t, statusTestSocketPath(t))
	ctx, cancel := context.WithTimeout(context.Background(), statusResponseLimit)
	defer cancel()
	request := statusToolRequest(map[string]any{"absolutePath": t.TempDir()})
	request.Params.Name = "get_indexing_status"
	result, err := client.CallTool(ctx, request)
	if err != nil {
		t.Fatalf("call get_indexing_status: %v", err)
	}
	if !result.IsError || len(result.Content) != 1 {
		t.Fatalf("result = %v, want one tool error", result)
	}
	text := result.Content[0].(mcp.TextContent).Text
	if !strings.Contains(text, "dial unix") || !strings.Contains(text, "no such file") {
		t.Fatalf("tool error = %q, want daemon socket failure", text)
	}
	t.Logf("get_indexing_status unavailable daemon: %s", text)
}

func callStatusTestTool(t *testing.T, client *mcpclient.Client, toolName string, arguments map[string]any) *pb.GetIndexResponse {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*statusResponseLimit)
	defer cancel()
	request := statusToolRequest(arguments)
	request.Params.Name = toolName
	result, err := client.CallTool(ctx, request)
	if err != nil {
		t.Fatalf("call %s: %v", toolName, err)
	}
	if result.IsError || len(result.Content) != 1 {
		t.Fatalf("%s result = %#v, want one successful status", toolName, result)
	}
	text, ok := result.Content[0].(mcp.TextContent)
	if !ok {
		t.Fatalf("%s content type = %T, want text", toolName, result.Content[0])
	}
	status := &pb.GetIndexResponse{}
	if err := protojson.Unmarshal([]byte(text.Text), status); err != nil {
		t.Fatalf("decode %s status: %v", toolName, err)
	}
	return status
}

func newStatusTestClient(t *testing.T, socketPath string) *mcpclient.Client {
	t.Helper()
	return newMCPProcessClient(t, socketPath, response.ModeJSON)
}

func newMCPProcessClient(t *testing.T, socketPath string, outputMode response.Mode) *mcpclient.Client {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatalf("find test executable: %v", err)
	}
	client, err := mcpclient.NewStdioMCPClient(executable, []string{
		"CLAUDE_CONTEXTD_SOCKET_PATH=" + socketPath,
		"CLAUDE_CONTEXT_MCP_OUTPUT=" + string(outputMode),
		statusProcessEnv + "=1",
	}, "-test.run=^TestStatusMCPProcess$")
	if err != nil {
		t.Fatalf("create MCP client: %v", err)
	}
	t.Cleanup(func() {
		if err := client.Close(); err != nil {
			t.Errorf("close MCP client: %v", err)
		}
	})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err = client.Initialize(ctx, mcp.InitializeRequest{
		Params: mcp.InitializeParams{ProtocolVersion: mcp.LATEST_PROTOCOL_VERSION},
	})
	if err != nil {
		t.Fatalf("initialize MCP client: %v", err)
	}
	return client
}

func statusToolRequest(arguments map[string]any) mcp.CallToolRequest {
	return mcp.CallToolRequest{Params: mcp.CallToolParams{Arguments: arguments}}
}

func startStatusTestDaemon(t *testing.T, state model.CodebaseStatus) (string, string) {
	t.Helper()
	for _, variable := range sandbox.Env(t.TempDir()) {
		t.Setenv(variable.Name, variable.Value)
	}
	t.Setenv("CLAUDE_CONTEXT_PROFILE", config.ProfileStandard)
	t.Setenv("MILVUS_ADDRESS", "")
	cfg, err := config.Default()
	if err != nil {
		t.Fatalf("load sandbox config: %v", err)
	}
	for _, directory := range sandbox.Directories(cfg) {
		if err := os.MkdirAll(directory, 0o755); err != nil {
			t.Fatalf("create sandbox directory: %v", err)
		}
	}
	codebasePath, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatalf("resolve codebase directory: %v", err)
	}
	if err := store.WriteRegistry(cfg.RegistryPath, model.RegistryFile{
		Codebases: []model.Codebase{{
			ID:               statusTestCodebaseID,
			CanonicalPath:    codebasePath,
			Status:           state,
			SchedulingPolicy: model.DefaultSchedulingPolicy(),
		}},
	}); err != nil {
		t.Fatalf("write sandbox registry: %v", err)
	}
	manager, err := daemon.NewManager(context.Background(), cfg)
	if err != nil {
		t.Fatalf("start sandbox manager: %v", err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := manager.Close(ctx); err != nil {
			t.Errorf("close sandbox manager: %v", err)
		}
	})
	socketPath := statusTestSocketPath(t)
	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatalf("listen on daemon socket: %v", err)
	}
	grpcServer := grpc.NewServer()
	pb.RegisterSemanticSearchDaemonServiceServer(grpcServer, daemon.NewGRPCServer(manager, nil))
	go func() {
		if err := grpcServer.Serve(listener); err != nil {
			t.Errorf("serve sandbox daemon: %v", err)
		}
	}()
	t.Cleanup(grpcServer.GracefulStop)
	return socketPath, codebasePath
}

func statusTestSocketPath(t *testing.T) string {
	t.Helper()
	// A short socket directory avoids the macOS Unix socket path limit.
	socketDirectory, err := os.MkdirTemp("", "lms-status-")
	if err != nil {
		t.Fatalf("create socket directory: %v", err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(socketDirectory); err != nil {
			t.Errorf("remove socket directory: %v", err)
		}
	})
	return filepath.Join(socketDirectory, "d.sock")
}

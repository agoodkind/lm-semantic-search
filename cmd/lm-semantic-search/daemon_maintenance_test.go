package main

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/sandbox"
	"google.golang.org/grpc"
)

// startMaintenanceTestDaemon serves a real daemon, rooted in a throwaway
// directory with no vector store configured, on a unix socket the CLI under
// test dials. It returns the socket path.
func startMaintenanceTestDaemon(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	for _, variable := range sandbox.Env(root) {
		t.Setenv(variable.Name, variable.Value)
	}
	// The standard profile with no store address builds the Milvus-backed
	// service in its not-configured shape, so the daemon needs neither a store
	// nor a model download.
	t.Setenv("CLAUDE_CONTEXT_PROFILE", config.ProfileStandard)
	t.Setenv("MILVUS_ADDRESS", "")
	t.Setenv("CLAUDE_CONTEXT_BACKGROUND_SYNC", "false")
	t.Setenv("CLAUDE_CONTEXT_FILE_WATCHER", "false")
	t.Setenv("CLAUDE_CONTEXT_TRIGGER_WATCHER", "false")
	t.Setenv("CLAUDE_CONTEXT_DEBUG_LISTENER", "false")
	// The sandbox socket path sits under the temp root, which can exceed the
	// platform's socket path limit, so the socket lives in a short directory.
	socketDir, err := os.MkdirTemp("", "lms-cli")
	if err != nil {
		t.Fatalf("MkdirTemp returned error: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(socketDir) })
	socketPath := filepath.Join(socketDir, "d.sock")
	t.Setenv("CLAUDE_CONTEXTD_SOCKET_PATH", socketPath)

	cfg, err := config.Default()
	if err != nil {
		t.Fatalf("config.Default returned error: %v", err)
	}
	for _, directory := range []string{cfg.StateRoot, cfg.LogsDir, cfg.MerkleDir, cfg.LocksDir, cfg.SocketsDir, cfg.ChunksDir, cfg.GraphDir, cfg.ContextRoot} {
		if err := os.MkdirAll(directory, 0o755); err != nil {
			t.Fatalf("MkdirAll(%s) returned error: %v", directory, err)
		}
	}
	manager, err := daemon.NewManager(context.Background(), cfg)
	if err != nil {
		t.Fatalf("daemon.NewManager returned error: %v", err)
	}
	listener, err := (&net.ListenConfig{}).Listen(context.Background(), "unix", socketPath)
	if err != nil {
		t.Fatalf("listen on %s returned error: %v", socketPath, err)
	}
	server := grpc.NewServer(
		grpc.MaxRecvMsgSize(grpcutil.MaxMessageBytes),
		grpc.MaxSendMsgSize(grpcutil.MaxMessageBytes),
	)
	pb.RegisterSemanticSearchDaemonServiceServer(server, daemon.NewGRPCServer(manager, nil))
	go func() {
		_ = server.Serve(listener)
	}()
	t.Cleanup(func() {
		server.GracefulStop()
		_ = listener.Close()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := manager.Close(ctx); err != nil {
			t.Errorf("manager.Close returned error: %v", err)
		}
	})
	return socketPath
}

// runCLICapturingStdout runs one CLI invocation and returns what it printed.
// The response printer writes to the process stdout rather than cobra's
// writer, so the test swaps that stream for a pipe while the command runs.
func runCLICapturingStdout(t *testing.T, args []string) string {
	t.Helper()
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe returned error: %v", err)
	}
	previousStdout := os.Stdout
	os.Stdout = writer
	root, _, _ := testRoot()
	root.SetArgs(args)
	runErr := root.Execute()
	os.Stdout = previousStdout
	if closeErr := writer.Close(); closeErr != nil {
		t.Fatalf("close pipe writer: %v", closeErr)
	}
	output, readErr := io.ReadAll(reader)
	if readErr != nil {
		t.Fatalf("read captured stdout: %v", readErr)
	}
	if runErr != nil {
		t.Fatalf("%s returned error: %v\n%s", strings.Join(args, " "), runErr, output)
	}
	return string(output)
}

// The operator turns maintenance mode on and off through the CLI, and the
// daemon's status reports the mode the same way in between.
func TestDaemonMaintenanceCommandsSwitchTheRunningDaemon(t *testing.T) {
	socketPath := startMaintenanceTestDaemon(t)

	onOutput := runCLICapturingStdout(t, []string{"--socket", socketPath, "daemon", "maintenance", "on", "--reason", "milvus backup"})
	if !strings.Contains(onOutput, "Maintenance mode is on") || !strings.Contains(onOutput, "milvus backup") {
		t.Fatalf("daemon maintenance on output lacks the banner and reason:\n%s", onOutput)
	}

	statusOutput := runCLICapturingStdout(t, []string{"--socket", socketPath, "--json", "status"})
	var statusReply struct {
		Maintenance struct {
			Enabled bool   `json:"enabled"`
			Reason  string `json:"reason"`
		} `json:"maintenance"`
	}
	if err := json.Unmarshal([]byte(statusOutput), &statusReply); err != nil {
		t.Fatalf("status JSON did not parse: %v\n%s", err, statusOutput)
	}
	if !statusReply.Maintenance.Enabled || statusReply.Maintenance.Reason != "milvus backup" {
		t.Fatalf("status maintenance = %+v, want enabled with the reason", statusReply.Maintenance)
	}

	offOutput := runCLICapturingStdout(t, []string{"--socket", socketPath, "daemon", "maintenance", "off"})
	if !strings.Contains(offOutput, "Maintenance mode is off") {
		t.Fatalf("daemon maintenance off output lacks the acknowledgement:\n%s", offOutput)
	}
	afterOutput := runCLICapturingStdout(t, []string{"--socket", socketPath, "--json", "status"})
	if strings.Contains(afterOutput, `"enabled":true`) {
		t.Fatalf("status still reports maintenance after it was turned off:\n%s", afterOutput)
	}
}

// The group itself is not a command: it names the two switches.
func TestDaemonMaintenanceRequiresOnOrOff(t *testing.T) {
	root, _, _ := testRoot()
	root.SetArgs([]string{"daemon", "maintenance"})
	err := root.Execute()
	if err == nil || err.Error() != "daemon maintenance requires on or off" {
		t.Fatalf("error = %v, want the on-or-off message", err)
	}
}

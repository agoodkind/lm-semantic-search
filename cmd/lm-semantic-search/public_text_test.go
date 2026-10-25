package main_test

import (
	"bytes"
	"context"
	"errors"
	"net"
	"os"
	"os/exec"
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

const (
	cliPackage          = "goodkind.io/lm-semantic-search/cmd/lm-semantic-search"
	traceMarker         = "🔎"
	traceIDField        = "trace_id="
	missingJobID        = "job_missing"
	cliCommandTimeout   = time.Minute
	daemonCloseTimeout  = 5 * time.Second
	socketDirectoryName = "lms-cli-text"
)

type cliRun struct {
	exitCode int
	stdout   string
	stderr   string
}

// TestCLIPublicTextOmitsTraceAnnotations builds lm-semantic-search and starts
// a real daemon on a temporary Unix socket. It runs the binary with these
// commands:
//
//   - codebase status
//   - codebase list
//   - job list
//   - daemon doctor
//
// The test requires each command to exit with status 0 and print a result.
// It then runs job get with a job ID that does not exist. That command must
// exit with an error and print a line on stderr that starts with "Error: "
// and contains the job ID.
//
// The test checks that every command's output does not contain the magnifying
// glass character or "trace_id=".
func TestCLIPublicTextOmitsTraceAnnotations(t *testing.T) {
	binaryPath := buildCLI(t)
	socketPath := startPublicTextDaemon(t)
	commands := [][]string{
		{"codebase", "status", t.TempDir()},
		{"codebase", "list"},
		{"job", "list"},
		{"daemon", "doctor"},
	}
	for _, command := range commands {
		name := strings.Join(command, " ")
		run := runCLI(t, binaryPath, append([]string{"--socket", socketPath}, command...)...)
		if run.exitCode != 0 {
			t.Fatalf("%s exited %d:\n%s%s", name, run.exitCode, run.stdout, run.stderr)
		}
		assertNoTraceAnnotation(t, name, run.stdout+run.stderr)
		if strings.TrimSpace(run.stdout) == "" {
			t.Fatalf("%s printed nothing", name)
		}
	}

	run := runCLI(t, binaryPath, "--socket", socketPath, "job", "get", missingJobID)
	if run.exitCode == 0 {
		t.Fatalf("job get %s succeeded, want a job-not-found error:\n%s", missingJobID, run.stdout)
	}
	assertNoTraceAnnotation(t, "job get", run.stdout+run.stderr)
	if !hasErrorLine(run.stderr, missingJobID) {
		t.Fatalf("job get stderr = %q, want an Error: line that contains %q", run.stderr, missingJobID)
	}
}

func hasErrorLine(stderr string, want string) bool {
	for line := range strings.SplitSeq(stderr, "\n") {
		if strings.HasPrefix(line, "Error: ") && strings.Contains(line, want) {
			return true
		}
	}
	return false
}

func buildCLI(t *testing.T) string {
	t.Helper()
	binaryPath := filepath.Join(t.TempDir(), "lm-semantic-search")
	build := exec.CommandContext(context.Background(), "go", "build", "-o", binaryPath, cliPackage)
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build %s: %v\n%s", cliPackage, err, output)
	}
	return binaryPath
}

func runCLI(t *testing.T, binaryPath string, args ...string) cliRun {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), cliCommandTimeout)
	defer cancel()
	command := exec.CommandContext(ctx, binaryPath, args...)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr
	exitCode := 0
	if err := command.Run(); err != nil {
		var exitError *exec.ExitError
		if !errors.As(err, &exitError) {
			t.Fatalf("run %s %v: %v", binaryPath, args, err)
		}
		exitCode = exitError.ExitCode()
	}
	return cliRun{exitCode: exitCode, stdout: stdout.String(), stderr: stderr.String()}
}

// startPublicTextDaemon points every daemon path at a temporary sandbox.
// It disables background sync, file watchers, and the debug listener.
// It leaves the vector store address empty.
//
// It starts a daemon manager and serves its gRPC service on a Unix socket
// in a short temporary directory. It returns the socket path and registers
// test cleanup that stops the server and closes the manager.
func startPublicTextDaemon(t *testing.T) string {
	t.Helper()
	for _, variable := range sandbox.Env(t.TempDir()) {
		t.Setenv(variable.Name, variable.Value)
	}
	t.Setenv("CLAUDE_CONTEXT_PROFILE", config.ProfileStandard)
	t.Setenv("MILVUS_ADDRESS", "")
	t.Setenv("CLAUDE_CONTEXT_BACKGROUND_SYNC", "false")
	t.Setenv("CLAUDE_CONTEXT_FILE_WATCHER", "false")
	t.Setenv("CLAUDE_CONTEXT_TRIGGER_WATCHER", "false")
	t.Setenv("CLAUDE_CONTEXT_DEBUG_LISTENER", "false")
	socketDirectory, err := os.MkdirTemp("", socketDirectoryName)
	if err != nil {
		t.Fatalf("create socket directory: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(socketDirectory) })
	socketPath := filepath.Join(socketDirectory, "d.sock")
	t.Setenv("CLAUDE_CONTEXTD_SOCKET_PATH", socketPath)

	cfg, err := config.Default()
	if err != nil {
		t.Fatalf("load sandbox config: %v", err)
	}
	for _, directory := range sandbox.Directories(cfg) {
		if err := os.MkdirAll(directory, 0o755); err != nil {
			t.Fatalf("create sandbox directory %s: %v", directory, err)
		}
	}
	manager, err := daemon.NewManager(context.Background(), cfg)
	if err != nil {
		t.Fatalf("start sandbox manager: %v", err)
	}
	listener, err := (&net.ListenConfig{}).Listen(context.Background(), "unix", socketPath)
	if err != nil {
		t.Fatalf("listen on %s: %v", socketPath, err)
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
		ctx, cancel := context.WithTimeout(context.Background(), daemonCloseTimeout)
		defer cancel()
		if err := manager.Close(ctx); err != nil {
			t.Errorf("close sandbox manager: %v", err)
		}
	})
	return socketPath
}

func assertNoTraceAnnotation(t *testing.T, command string, output string) {
	t.Helper()
	if strings.Contains(output, traceMarker) || strings.Contains(output, traceIDField) {
		t.Fatalf("%s output contains a trace annotation:\n%s", command, output)
	}
}

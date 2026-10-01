//go:build live

package live

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
)

func TestDaemonShutdownDuringNativeGraphWork(t *testing.T) {
	instance := newLibraryCodebaseDaemon(t)
	instance.stop()
	binary, err := filepath.Abs(filepath.Join("..", "..", "dist", "lm-semantic-search-daemon"))
	if err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(t.TempDir(), "daemon.log")
	if retained := os.Getenv("LMS_GRAPH_SHUTDOWN_LOG"); retained != "" {
		logPath = retained
	}
	joined := startGraphShutdownDaemon(t, instance, binary, logPath)
	idle := t.TempDir()
	writeCodebaseFile(t, idle, "idle.go", "package idle\nfunc IdleTarget() string { return \"idlegraphmarker\" }\n")
	instance.index(t, idle)
	if _, err := instance.client.GraphTool(t.Context(), &pb.GraphToolRequest{
		Path: idle, ToolName: "query_graph", ArgsJson: `{"query":"MATCH (f:Function) RETURN f.name LIMIT 5"}`,
		Client: &pb.ClientInfo{Name: "graph-shutdown-live"},
	}); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	active := t.TempDir()
	var sourceBuilder strings.Builder
	sourceBuilder.WriteString("package active\n")
	for ordinal := range 2500 {
		fmt.Fprintf(&sourceBuilder, "func FixtureTarget%d() {}\n", ordinal)
	}
	sourceBuilder.WriteString("func ActiveTarget() {\n")
	sourceBuilder.WriteString(strings.Repeat("UnknownTarget()\n", 1000000))
	sourceBuilder.WriteString("}\n")
	source := sourceBuilder.String()
	writeCodebaseFile(t, active, "active.go", source)
	if _, err := instance.client.StartIndex(t.Context(), &pb.StartIndexRequest{
		Path: active, Splitter: &pb.SplitterConfig{Type: "ast"}, Client: &pb.ClientInfo{Name: "graph-shutdown-live"},
	}); err != nil {
		t.Fatal(err)
	}
	waitNativeGraphCalls(t, logPath, len(before))
	started := time.Now()
	response, err := instance.client.Shutdown(t.Context(), &pb.ShutdownRequest{})
	if err != nil || !response.GetAccepted() {
		t.Fatalf("public shutdown accepted=%t error=%v", response.GetAccepted(), err)
	}
	select {
	case err := <-joined:
		if err != nil {
			t.Fatalf("daemon shutdown returned %v; log=%s", err, logPath)
		}
	case <-time.After(15 * time.Second):
		t.Fatalf("daemon did not join after public shutdown; log=%s", logPath)
	}
	requireGraphShutdownWarning(t, instance.config.LogsDir, logPath)
	unchanged, err := os.ReadFile(filepath.Join(active, "active.go"))
	if err != nil || string(unchanged) != source {
		t.Fatal("the source changed during indexing or shutdown")
	}
	t.Logf("public Shutdown joined exit0 in %s with a distinct idle graph handle and observed native calls pass; log=%s", time.Since(started), logPath)
}

func requireGraphShutdownWarning(t *testing.T, logsDir, logPath string) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(logsDir, "daemon.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if os.Getenv("LMS_GRAPH_SHUTDOWN_LOG") != "" {
		file, err := os.OpenFile(logPath+".structured.jsonl", os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := file.Write(data); err != nil {
			_ = file.Close()
			t.Fatal(err)
		}
		if err := file.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if !strings.Contains(string(data), "graph handles remain open at shutdown; graph operation still active") {
		t.Fatal("shutdown did not encounter the actual active graph operation")
	}
}

func startGraphShutdownDaemon(t *testing.T, instance *libraryCodebaseDaemon, binary, logPath string) <-chan error {
	t.Helper()
	logFile, err := os.OpenFile(logPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = logFile.Close() })
	command := exec.Command(binary)
	command.Env = os.Environ()
	command.Stdout, command.Stderr = logFile, logFile
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	joined := make(chan error, 1)
	done := make(chan struct{})
	go func() {
		joined <- command.Wait()
		close(done)
	}()
	t.Cleanup(func() {
		select {
		case <-done:
		default:
			_ = command.Process.Kill()
			<-done
		}
	})
	deadline := time.Now().Add(time.Minute)
	for time.Now().Before(deadline) {
		ctx, cancel := context.WithTimeout(t.Context(), time.Second)
		connection, client, err := grpcutil.DialDaemon(ctx, instance.config.SocketPath)
		if err == nil {
			_, err = client.Version(ctx, &pb.VersionRequest{})
			if err == nil {
				instance.client = client
				t.Cleanup(func() { _ = connection.Close() })
				cancel()
				return joined
			}
			_ = connection.Close()
		}
		cancel()
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("daemon PID%d did not serve its private socket; log=%s", command.Process.Pid, logPath)
	return nil
}

func waitNativeGraphCalls(t *testing.T, logPath string, offset int) {
	t.Helper()
	deadline := time.Now().Add(time.Minute)
	for time.Now().Before(deadline) {
		data, err := os.ReadFile(logPath)
		if err != nil {
			t.Fatal(err)
		}
		if len(data) >= offset && strings.Contains(string(data[offset:]), "pass.start pass=calls") {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("the actual native calls pass did not start; log=%s", logPath)
}

//go:build offlinelive

package offlinelive

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/exec"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"google.golang.org/grpc"
)

const (
	// crashDaemonEnv makes this test binary run a daemon instead of the tests,
	// so a test can kill that daemon the way a crash or a forced stop does.
	crashDaemonEnv = "LMS_OFFLINE_LIVE_CRASH_DAEMON"

	crashDaemonReadyTimeout = 30 * time.Second
)

func TestMain(m *testing.M) {
	if os.Getenv(crashDaemonEnv) != "" {
		os.Exit(runCrashDaemon())
	}
	os.Exit(m.Run())
}

// runCrashDaemon serves a daemon over the configuration the parent test put in
// the environment until the process is killed.
func runCrashDaemon() int {
	ctx := context.Background()
	daemonConfig, err := config.Default()
	if err != nil {
		slog.Error("crash daemon config failed", "err", err)
		return 1
	}
	manager, err := daemon.NewManager(ctx, daemonConfig)
	if err != nil {
		slog.Error("crash daemon manager failed", "err", err)
		return 1
	}
	manager.ResumeOrphanedJobs(ctx)
	daemon.NewBackgroundSync(daemonConfig, manager).Start(ctx)

	if err := os.Remove(daemonConfig.SocketPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		slog.Error("crash daemon remove stale socket failed", "err", err)
		return 1
	}
	listener, err := (&net.ListenConfig{}).Listen(ctx, "unix", daemonConfig.SocketPath)
	if err != nil {
		slog.Error("crash daemon listen failed", "err", err)
		return 1
	}
	server := grpc.NewServer(
		grpc.MaxRecvMsgSize(grpcutil.MaxMessageBytes),
		grpc.MaxSendMsgSize(grpcutil.MaxMessageBytes),
	)
	pb.RegisterSemanticSearchDaemonServiceServer(server, daemon.NewGRPCServer(manager, nil))
	if err := server.Serve(listener); err != nil {
		slog.Error("crash daemon serve failed", "err", err)
		return 1
	}
	return 0
}

// startCrashableDaemon runs a daemon with options over the harness state root in
// a child process and connects the harness client to it. crash kills that
// process without letting it shut down, so every build it was running is left
// mid-flight in the persisted state.
func (harness *harness) startCrashableDaemon(options harnessOptions) (crash func()) {
	harness.t.Helper()

	daemonConfig := resolveOfflineConfig(harness.t, harness.stateRoot, harness.socketPath, options)
	prepareState(harness.t, daemonConfig)

	command := exec.CommandContext(context.Background(), os.Args[0])
	command.Env = append(os.Environ(), crashDaemonEnv+"=1")
	if err := command.Start(); err != nil {
		harness.t.Fatalf("start crashable daemon: %v", err)
	}
	killed := false
	kill := func() {
		if killed {
			return
		}
		killed = true
		if err := command.Process.Kill(); err != nil {
			harness.t.Errorf("kill crashable daemon: %v", err)
		}
		// The kill makes Wait report the signal, which is the expected exit.
		_ = command.Wait()
	}
	harness.t.Cleanup(kill)

	connection, client, err := harness.dialWhenReady()
	if err != nil {
		harness.t.Fatalf("connect to crashable daemon: %v", err)
	}
	harness.config = daemonConfig
	harness.connection = connection
	harness.client = client
	return func() {
		if closeErr := connection.Close(); closeErr != nil {
			harness.t.Errorf("close crashable daemon connection: %v", closeErr)
		}
		kill()
	}
}

// dialWhenReady connects to the harness socket once a daemon answers on it.
func (harness *harness) dialWhenReady() (*grpc.ClientConn, pb.SemanticSearchDaemonServiceClient, error) {
	deadline := time.Now().Add(crashDaemonReadyTimeout)
	var lastErr error
	for time.Now().Before(deadline) {
		connection, client, err := grpcutil.DialDaemon(context.Background(), harness.socketPath)
		if err != nil {
			lastErr = err
			time.Sleep(pollInterval)
			continue
		}
		_, lastErr = client.Version(correlatedContext(), &pb.VersionRequest{})
		if lastErr == nil {
			return connection, client, nil
		}
		// A failed probe leaves nothing to report beyond lastErr.
		_ = connection.Close()
		time.Sleep(pollInterval)
	}
	return nil, nil, fmt.Errorf("daemon did not answer within %s: %w", crashDaemonReadyTimeout, lastErr)
}

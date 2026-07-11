// Command lm-semantic-search-daemon runs the local gRPC daemon.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"goodkind.io/gklog"
	"goodkind.io/gklog/correlation"
	"goodkind.io/gklog/version"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/debugserver"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/logcleanup"
	"goodkind.io/lm-semantic-search/internal/metrics"
	"goodkind.io/lm-semantic-search/internal/store"
	"goodkind.io/lm-semantic-search/internal/updateopts"
	"google.golang.org/grpc"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "version" {
		if err := writeVersion(os.Stdout); err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "write version output: %v\n", err)
			os.Exit(1)
		}
		return
	}
	rootContext := installCorrelationLogger("daemon-boot")
	if err := run(rootContext); err != nil {
		slog.ErrorContext(rootContext, "daemon failed", "err", err)
		os.Exit(1)
	}
}

func writeVersion(writer io.Writer) error {
	_, err := fmt.Fprintf(writer, "version: %s commit=%s build_time=%s\n", version.String(), version.Commit, version.BuildTime)
	if err != nil {
		slog.Error("write version output failed", "err", err)
		return fmt.Errorf("write version output: %w", err)
	}
	return nil
}

func correlationHandlerOptions() correlation.HandlerOptions {
	return correlation.HandlerOptions{
		Strict:   true,
		Required: []string{"trace_id", "span_id"},
	}
}

// installCorrelationLogger wraps the default JSON slog handler with a
// correlation handler in strict mode and returns a root context that
// carries the given origin so boot records inherit a trace_id. The boot
// logger writes only to stderr; once the state paths are known,
// installConcernRouter swaps in per-concern files.
func installCorrelationLogger(origin string) context.Context {
	jsonHandler := slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo})
	slog.SetDefault(slog.New(correlation.SlogHandler(jsonHandler, correlationHandlerOptions())))
	rootCorrelation := correlation.New("").WithIdentityAttributes(
		correlation.IdentityAttribute{Key: "origin", Value: origin},
	)
	return correlation.WithContext(context.Background(), rootCorrelation)
}

// minRotationMB is the smallest whole-megabyte cap gklog rotation accepts, so
// a sub-megabyte configured cap still rotates rather than falling back to
// gklog's zero-value default.
const minRotationMB = 1

// rotationConfig converts the configured per-file byte cap into gklog's
// whole-megabyte RotationConfig. It is populated (non-empty) so both the
// per-concern files and the combined service log rotate on write instead of
// growing unbounded.
func rotationConfig(cfg config.Config) gklog.RotationConfig {
	maxSizeMB := max(int(cfg.LogRotationMaxBytes/(1024*1024)), minRotationMB)
	return gklog.RotationConfig{MaxSizeMB: maxSizeMB}
}

// installConcernRouter swaps the default logger so records fan out to
// per-concern rotating JSONL files under logsDir, and routes the combined
// service stream through a rotating gklog file at logPath instead of raw
// stderr, so the top-level combined log is bounded too. The combined file
// handle is released at process exit, matching the per-concern files the router
// opens lazily. The concern is the first dot-separated segment of each message;
// the daemon concern catches anything without a dot.
func installConcernRouter(logsDir string, logPath string, rot gklog.RotationConfig) {
	combined := gklog.FileJSON(logPath, slog.LevelInfo, rot)
	router := gklog.NewRouter(logsDir, slog.LevelInfo, combined, gklog.RouterOptions{FallbackConcern: "daemon", Rotation: rot})
	slog.SetDefault(slog.New(correlation.SlogHandler(router, correlationHandlerOptions())))
}

func run(rootContext context.Context) error {
	slog.InfoContext(rootContext, "start daemon")

	cfg, err := config.Default()
	if err != nil {
		slog.ErrorContext(rootContext, "load config failed", "err", err)
		return fmt.Errorf("load default config: %w", err)
	}

	socketPath := flag.String("socket", cfg.SocketPath, "unix socket path")
	stateRoot := flag.String("state-root", cfg.StateRoot, "state root")
	flag.Parse()

	cfg = applyStatePaths(cfg, *stateRoot, *socketPath)

	for _, path := range []string{cfg.StateRoot, cfg.SocketsDir, cfg.LogsDir, cfg.MerkleDir, cfg.LocksDir, cfg.ChunksDir} {
		if err := store.EnsureDir(path); err != nil {
			slog.ErrorContext(rootContext, "ensure state directory failed", "path", path, "err", err)
			return fmt.Errorf("ensure state directory %s: %w", path, err)
		}
	}

	installConcernRouter(cfg.LogsDir, cfg.LogPath, rotationConfig(cfg))
	metrics.Register()

	slog.InfoContext(rootContext, "daemon identity", "build", version.String(), "commit", version.Commit, "socket", cfg.SocketPath, "state_root", cfg.StateRoot, "pid", os.Getpid())

	if err := refuseIfDaemonAlreadyServing(rootContext, cfg.SocketPath); err != nil {
		return err
	}
	if err := os.RemoveAll(cfg.SocketPath); err != nil {
		slog.ErrorContext(rootContext, "remove stale socket failed", "path", cfg.SocketPath, "err", err)
		return fmt.Errorf("remove stale socket %s: %w", cfg.SocketPath, err)
	}

	listenConfig := net.ListenConfig{}
	listener, err := listenConfig.Listen(rootContext, "unix", cfg.SocketPath)
	if err != nil {
		slog.ErrorContext(rootContext, "listen on unix socket failed", "path", cfg.SocketPath, "err", err)
		return fmt.Errorf("listen on unix socket %s: %w", cfg.SocketPath, err)
	}
	defer func() { _ = listener.Close() }()

	manager, err := daemon.NewManager(rootContext, cfg)
	if err != nil {
		slog.ErrorContext(rootContext, "create manager failed", "err", err)
		return fmt.Errorf("create manager: %w", err)
	}
	defer manager.CloseGraphEngines()

	runtimeContext, cancelRuntime := context.WithCancel(rootContext)
	defer cancelRuntime()
	manager.ResumeOrphanedJobs(runtimeContext)
	daemon.NewBackgroundSync(cfg, manager).Start(runtimeContext)
	startLogRetentionSweep(runtimeContext, cfg)

	metrics.StartReporter(runtimeContext, time.Duration(cfg.PerfCountersIntervalMS)*time.Millisecond)
	var debugSrv *debugserver.Server
	if cfg.DebugListenerEnabled {
		if debugSrv, err = startDebugServer(runtimeContext, cfg); err != nil {
			return err
		}
		defer stopDebugServer(rootContext, debugSrv)
	}

	server := grpc.NewServer(
		grpc.MaxRecvMsgSize(grpcutil.MaxMessageBytes),
		grpc.MaxSendMsgSize(grpcutil.MaxMessageBytes),
	)
	shutdownCh := make(chan struct{}, 1)
	pb.RegisterSemanticSearchDaemonServiceServer(server, daemon.NewGRPCServer(manager, func() {
		select {
		case shutdownCh <- struct{}{}:
		default:
		}
	}))
	startUpdateScheduler(runtimeContext, cfg, shutdownCh)

	serveErrCh := make(chan error, 1)
	goSafe(rootContext, func() {
		if serveErr := server.Serve(listener); serveErr != nil {
			serveErrCh <- fmt.Errorf("serve gRPC on %s: %w", cfg.SocketPath, serveErr)
		}
	})

	signalCh := make(chan os.Signal, 1)
	signal.Notify(signalCh, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-serveErrCh:
		cancelRuntime()
		server.Stop()
		return err
	case <-signalCh:
	case <-shutdownCh:
	}

	cancelRuntime()
	server.GracefulStop()
	return nil
}

// startLogRetentionSweep launches the background log retention walker. It runs
// off the log-write path: rotation happens on write via gklog, while this
// walker deletes rotated backups past the retention budget on ctx cancellation
// or its interval. The walker stops when the runtime context cancels at
// shutdown.
func startLogRetentionSweep(ctx context.Context, cfg config.Config) {
	interval := time.Duration(cfg.LogCleanupIntervalMS) * time.Millisecond
	// The combined log lives under LogsDir today, but sweep its own directory
	// explicitly too so an overridden CLAUDE_CONTEXTD_LOG_PATH pointing outside
	// LogsDir still has its rotated backups retained rather than growing unbounded.
	// Dedupe so the common case (both resolve to LogsDir) runs a single walker.
	roots := map[string]struct{}{
		cfg.LogsDir:               {},
		filepath.Dir(cfg.LogPath): {},
	}
	for root := range roots {
		logcleanup.Start(ctx, logcleanup.Policy{
			Root:           root,
			RetentionBytes: cfg.LogRetentionBytes,
			Enabled:        cfg.LogCleanupEnabled,
		}, interval)
	}
}

func startUpdateScheduler(ctx context.Context, cfg config.Config, shutdownCh chan<- struct{}) {
	executablePath, err := os.Executable()
	if err != nil {
		slog.WarnContext(ctx, "update scheduler disabled; executable path unavailable", "err", err)
		return
	}
	overrides := updateopts.Overrides{
		Client:     nil,
		InstallDir: filepath.Dir(executablePath),
		StateRoot:  cfg.StateRoot,
		CacheDir:   "",
		DryRun:     false,
		Log:        slog.Default(),
	}
	goSafe(ctx, func() {
		updateopts.RunApplyScheduler(ctx, overrides, func() {
			select {
			case shutdownCh <- struct{}{}:
			default:
			}
		})
	})
}

// applyStatePaths derives every state-relative path from the resolved state
// root and socket so the rest of startup reads a fully-populated config.
func applyStatePaths(cfg config.Config, stateRoot string, socketPath string) config.Config {
	cfg.StateRoot = stateRoot
	cfg.SocketPath = socketPath
	cfg.RegistryPath = filepath.Join(cfg.StateRoot, "registry.json")
	cfg.JobsPath = filepath.Join(cfg.StateRoot, "jobs.jsonl")
	cfg.EventsPath = filepath.Join(cfg.StateRoot, "events.jsonl")
	cfg.SocketsDir = filepath.Dir(cfg.SocketPath)
	cfg.LogsDir = filepath.Join(cfg.StateRoot, "logs")
	cfg.LogPath = filepath.Join(cfg.LogsDir, "lm-semantic-search-daemon.log")
	cfg.MerkleDir = filepath.Join(cfg.StateRoot, "merkle")
	cfg.LocksDir = filepath.Join(cfg.StateRoot, "locks")
	cfg.ChunksDir = filepath.Join(cfg.StateRoot, "chunks")
	return cfg
}

// debugShutdownTimeout bounds how long daemon exit waits on the debug
// listener so an in-flight profile request cannot delay shutdown.
const debugShutdownTimeout = 2 * time.Second

// startDebugServer binds the loopback pprof and expvar listener. The bind is
// eager so a port conflict surfaces during startup rather than inside the
// serving goroutine.
// refuseIfDaemonAlreadyServing returns an error when another lm-semantic-search-daemon is
// already accepting connections on socketPath, so a second instance never
// clobbers the live one's socket and steals its clients. A stale socket file
// with no listener (the normal case after a crash or a kill-and-restart) fails
// to dial and is cleared by the caller before binding.
func refuseIfDaemonAlreadyServing(ctx context.Context, socketPath string) error {
	dialContext, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
	defer cancel()

	var dialer net.Dialer
	conn, err := dialer.DialContext(dialContext, "unix", socketPath)
	if err != nil {
		return nil
	}
	if closeErr := conn.Close(); closeErr != nil {
		slog.WarnContext(ctx, "close probe connection failed", "path", socketPath, "err", closeErr)
	}
	conflict := fmt.Errorf("another lm-semantic-search-daemon is already listening on %s", socketPath)
	slog.ErrorContext(ctx, "another lm-semantic-search-daemon is already serving this socket; refusing to start", "path", socketPath, "err", conflict)
	return conflict
}

func startDebugServer(ctx context.Context, cfg config.Config) (*debugserver.Server, error) {
	srv, err := debugserver.New(cfg.DebugListenAddr)
	if err != nil {
		slog.ErrorContext(ctx, "create debug listener failed", "addr", cfg.DebugListenAddr, "err", err)
		return nil, fmt.Errorf("create debug listener on %s: %w", cfg.DebugListenAddr, err)
	}
	if err := srv.Start(ctx); err != nil {
		slog.ErrorContext(ctx, "start debug listener failed", "addr", cfg.DebugListenAddr, "err", err)
		return nil, fmt.Errorf("start debug listener on %s: %w", cfg.DebugListenAddr, err)
	}
	slog.InfoContext(ctx, "debug listener started", "addr", srv.Addr())
	return srv, nil
}

// stopDebugServer shuts the debug listener within a bounded window. A nil
// server is a no-op so callers need not branch on whether it was started.
func stopDebugServer(ctx context.Context, srv *debugserver.Server) {
	if srv == nil {
		return
	}
	shutdownCtx, cancel := context.WithTimeout(ctx, debugShutdownTimeout)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		slog.WarnContext(ctx, "debug listener shutdown failed", "err", err)
	}
}

func goSafe(ctx context.Context, run func()) {
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.ErrorContext(ctx, "goroutine panic", "err", fmt.Errorf("panic: %v", recovered))
			}
		}()
		run()
	}()
}

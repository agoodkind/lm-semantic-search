//go:build updatelive

// Package updatelive runs real daemon processes against a local release API
// server and records which daemons ask that server for releases. It builds the
// daemon twice with the same release version stamp, once with the release
// build tag and once without it.
package updatelive

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/sandbox"
)

const (
	daemonBinary  = "lm-semantic-search-daemon"
	daemonPackage = "goodkind.io/lm-semantic-search/cmd/lm-semantic-search-daemon"

	releaseBuildTag     = "lmsrelease"
	gklogVersionPackage = "goodkind.io/gklog/version"
	stampedVersion      = "202609290000-1-0123abc"
	stampedCommit       = "0123abc"
	stampedBuildTime    = "2026-09-29T00:00:00Z"
	updateAPIBaseURLEnv = "LM_SEMANTIC_SEARCH_UPDATE_API_BASE_URL"
	releaseListPath     = "/repos/agoodkind/lm-semantic-search/releases"

	localBuildDisabledLog = "update scheduler disabled; binary is not a release artifact"
	sandboxDisabledLog    = "update scheduler disabled; sandbox daemon"
	daemonIdentityLog     = "daemon identity"

	// The scheduler waits one minute before its first check when the update
	// state has no next check time. A daemon that must not check stays silent
	// for longer than that wait.
	firstCheckDelay   = time.Minute
	silenceMargin     = 15 * time.Second
	firstCheckTimeout = 3 * time.Minute
	startupTimeout    = time.Minute
	pollInterval      = 250 * time.Millisecond
	stopTimeout       = 15 * time.Second

	// Each daemon root sits under /tmp instead of the longer platform temp
	// directory. The kernel caps a socket path near 104 bytes.
	daemonRootParent  = "/tmp"
	daemonRootPattern = "lms-updatelive-"
	// The router writes each record without a dotted concern prefix to the
	// daemon concern file under the state root, which sandbox.Env places at
	// root/state.
	daemonLogPath = "state/logs/daemon.jsonl"
)

var (
	releaseDaemonPath string
	localDaemonPath   string
)

func TestMain(m *testing.M) {
	buildDirectory, err := os.MkdirTemp("", "lms-update-live-")
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "create build directory: %v\n", err)
		os.Exit(1)
	}
	releaseDaemonPath = filepath.Join(buildDirectory, "release", daemonBinary)
	localDaemonPath = filepath.Join(buildDirectory, "local", daemonBinary)
	buildErr := buildDaemon(releaseDaemonPath, releaseBuildTag)
	if buildErr == nil {
		buildErr = buildDaemon(localDaemonPath, "")
	}
	if buildErr != nil {
		_, _ = fmt.Fprintf(os.Stderr, "%v\n", buildErr)
		_ = os.RemoveAll(buildDirectory)
		os.Exit(1)
	}
	exitCode := m.Run()
	_ = os.RemoveAll(buildDirectory)
	os.Exit(exitCode)
}

// buildDaemon stamps the same release-shaped version into every build. The
// build tag is the only input that differs between the two daemons.
func buildDaemon(outputPath string, tags string) error {
	ldflags := strings.Join([]string{
		"-X", gklogVersionPackage + ".Version=" + stampedVersion,
		"-X", gklogVersionPackage + ".Commit=" + stampedCommit,
		"-X", gklogVersionPackage + ".Dirty=false",
		"-X", gklogVersionPackage + ".BuildTime=" + stampedBuildTime,
	}, " ")
	arguments := []string{"build", "-ldflags", ldflags, "-o", outputPath}
	if tags != "" {
		arguments = append(arguments, "-tags", tags)
	}
	arguments = append(arguments, daemonPackage)
	output, err := exec.Command("go", arguments...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("build daemon with tags %q: %w\n%s", tags, err, output)
	}
	return nil
}

// releaseAPIServer answers the daemon's release API requests with an empty
// release list and records every request path.
type releaseAPIServer struct {
	server *httptest.Server
	mutex  sync.Mutex
	paths  []string
}

func startReleaseAPIServer(t *testing.T) *releaseAPIServer {
	t.Helper()
	api := &releaseAPIServer{}
	api.server = httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		api.mutex.Lock()
		api.paths = append(api.paths, request.URL.Path)
		api.mutex.Unlock()
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte("[]"))
	}))
	t.Cleanup(api.server.Close)
	return api
}

func (api *releaseAPIServer) requestPaths() []string {
	api.mutex.Lock()
	defer api.mutex.Unlock()
	return append([]string(nil), api.paths...)
}

func (api *releaseAPIServer) receivedReleaseListRequest() bool {
	for _, path := range api.requestPaths() {
		if strings.HasPrefix(path, releaseListPath) {
			return true
		}
	}
	return false
}

// daemonMode selects how the test starts a daemon.
type daemonMode int

const (
	// installedMode runs the daemon command itself, the way the service
	// manager starts the installed daemon.
	installedMode daemonMode = iota + 1
	// sandboxMode runs the sandbox subcommand.
	sandboxMode
)

type runningDaemon struct {
	command *exec.Cmd
	root    string
	started time.Time
	done    chan error
}

// startDaemon roots both modes in a fresh directory with the sandbox
// environment defaults. The installed mode passes them as environment
// variables and runs no sandbox code.
func startDaemon(t *testing.T, binaryPath string, mode daemonMode, api *releaseAPIServer) *runningDaemon {
	t.Helper()
	root, err := os.MkdirTemp(daemonRootParent, daemonRootPattern)
	if err != nil {
		t.Fatalf("create daemon root: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(root) })

	environment := append(os.Environ(),
		updateAPIBaseURLEnv+"="+api.server.URL,
		"CLAUDE_CONTEXT_BACKGROUND_SYNC=false",
		"CLAUDE_CONTEXT_FILE_WATCHER=false",
		"CLAUDE_CONTEXT_RESUME_ON_BOOT=false",
		"CLAUDE_CONTEXT_TRIGGER_WATCHER=false",
	)
	var command *exec.Cmd
	switch mode {
	case installedMode:
		for _, variable := range sandbox.Env(root) {
			if variable.Value != "" {
				environment = append(environment, variable.Name+"="+variable.Value)
			}
		}
		if err := os.MkdirAll(filepath.Join(root, "state"), 0o700); err != nil {
			t.Fatalf("create state root: %v", err)
		}
		command = exec.Command(binaryPath)
	case sandboxMode:
		command = exec.Command(binaryPath, "sandbox", "--root", root)
	default:
		t.Fatalf("unknown daemon mode %d", mode)
	}
	command.Env = environment
	command.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := command.Start(); err != nil {
		t.Fatalf("start %s: %v", binaryPath, err)
	}
	daemon := &runningDaemon{command: command, root: root, started: time.Now(), done: make(chan error, 1)}
	go func() { daemon.done <- command.Wait() }()
	t.Cleanup(func() { daemon.stop(t) })
	return daemon
}

func (daemon *runningDaemon) logContents() string {
	contents, err := os.ReadFile(filepath.Join(daemon.root, daemonLogPath))
	if err != nil {
		return ""
	}
	return string(contents)
}

func (daemon *runningDaemon) waitForLog(t *testing.T, message string) {
	t.Helper()
	deadline := time.Now().Add(startupTimeout)
	for time.Now().Before(deadline) {
		if strings.Contains(daemon.logContents(), message) {
			return
		}
		daemon.failIfExited(t)
		time.Sleep(pollInterval)
	}
	t.Fatalf("daemon log did not contain %q within %s:\n%s", message, startupTimeout, daemon.logContents())
}

func (daemon *runningDaemon) failIfExited(t *testing.T) {
	t.Helper()
	select {
	case err := <-daemon.done:
		daemon.done <- err
		t.Fatalf("daemon exited early: %v\n%s", err, daemon.logContents())
	default:
	}
}

func (daemon *runningDaemon) stop(t *testing.T) {
	t.Helper()
	processGroup := -daemon.command.Process.Pid
	if err := syscall.Kill(processGroup, syscall.SIGTERM); err != nil && !errors.Is(err, syscall.ESRCH) {
		t.Errorf("signal daemon: %v", err)
	}
	select {
	case <-daemon.done:
		return
	case <-time.After(stopTimeout):
	}
	if err := syscall.Kill(processGroup, syscall.SIGKILL); err != nil && !errors.Is(err, syscall.ESRCH) {
		t.Errorf("kill daemon: %v", err)
	}
	<-daemon.done
}

// TestOnlyInstalledReleaseArtifactChecksForUpdates starts three daemons at
// once. The installed release-tagged daemon asks its server for releases after
// the first check delay. The installed untagged daemon and the release-tagged
// sandbox daemon each log that the scheduler is disabled and send their
// servers no request for longer than that delay.
func TestOnlyInstalledReleaseArtifactChecksForUpdates(t *testing.T) {
	releaseAPI := startReleaseAPIServer(t)
	localAPI := startReleaseAPIServer(t)
	sandboxAPI := startReleaseAPIServer(t)
	releaseDaemon := startDaemon(t, releaseDaemonPath, installedMode, releaseAPI)
	localDaemon := startDaemon(t, localDaemonPath, installedMode, localAPI)
	sandboxDaemon := startDaemon(t, releaseDaemonPath, sandboxMode, sandboxAPI)

	releaseDaemon.waitForLog(t, daemonIdentityLog)
	localDaemon.waitForLog(t, daemonIdentityLog)
	sandboxDaemon.waitForLog(t, daemonIdentityLog)
	localDaemon.waitForLog(t, localBuildDisabledLog)
	sandboxDaemon.waitForLog(t, sandboxDisabledLog)

	deadline := time.Now().Add(firstCheckTimeout)
	for !releaseAPI.receivedReleaseListRequest() {
		if time.Now().After(deadline) {
			t.Fatalf("installed release-tagged daemon sent no release list request within %s; paths %v\n%s",
				firstCheckTimeout, releaseAPI.requestPaths(), releaseDaemon.logContents())
		}
		releaseDaemon.failIfExited(t)
		time.Sleep(pollInterval)
	}
	releaseLog := releaseDaemon.logContents()
	if strings.Contains(releaseLog, localBuildDisabledLog) || strings.Contains(releaseLog, sandboxDisabledLog) {
		t.Fatalf("installed release-tagged daemon logged a disabled scheduler:\n%s", releaseLog)
	}

	silentDaemons := []struct {
		name   string
		daemon *runningDaemon
		api    *releaseAPIServer
	}{
		{name: "installed untagged daemon", daemon: localDaemon, api: localAPI},
		{name: "release-tagged sandbox daemon", daemon: sandboxDaemon, api: sandboxAPI},
	}
	for _, silent := range silentDaemons {
		silentUntil := silent.daemon.started.Add(firstCheckDelay + silenceMargin)
		for time.Now().Before(silentUntil) {
			silent.daemon.failIfExited(t)
			time.Sleep(pollInterval)
		}
		if paths := silent.api.requestPaths(); len(paths) != 0 {
			t.Fatalf("%s sent update requests %v", silent.name, paths)
		}
	}
}

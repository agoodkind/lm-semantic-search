// Command ci-auto-update verifies LMS's published automatic update path.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"goodkind.io/go-makefile/selfupdate"
)

const (
	githubAPIBaseURL    = "https://api.github.com"
	maxReleaseAttempts  = 10
	maxUpdateAttempts   = 120
	pollInterval        = 5 * time.Second
	requestTimeout      = 30 * time.Second
	processStopTimeout  = 15 * time.Second
	maxGitHubBodyBytes  = 4 << 20
	diagnosticLineCount = 200
	testRootParent      = "/tmp"
	testRootPattern     = "lms-auto-update-"

	cliBinary    = "lm-semantic-search"
	mcpBinary    = "lm-semantic-search-mcp"
	daemonBinary = "lm-semantic-search-daemon"
)

var releaseBinaries = []string{cliBinary, mcpBinary, daemonBinary}

type environment struct {
	repository string
	commit     string
	refType    string
	refName    string
	token      string
	manual     bool
}

type githubRelease struct {
	TagName     string    `json:"tag_name"`
	Draft       bool      `json:"draft"`
	PublishedAt time.Time `json:"published_at"`
}

type releaseSelection struct {
	target   githubRelease
	previous githubRelease
}

type childProcess struct {
	process  *os.Process
	done     chan error
	finished bool
	result   error
}

type updateCheck struct {
	environment environment
	repository  string
	testRoot    string
	testHome    string
	installDir  string
	daemonLog   string
	stateRoot   string
	configRoot  string
	cacheRoot   string
	runtimeRoot string
	contextRoot string
	stdout      io.Writer
	stderr      io.Writer
	child       *childProcess
	apiProxy    *httptest.Server
}

func main() {
	slog.Info("ci.auto_update.invoked")
	os.Exit(realMain())
}

func realMain() int {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := run(ctx, os.Getenv, os.Stdout, os.Stderr); err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "auto-update check failed: %v\n", err)
		if errors.Is(err, context.Canceled) {
			return 130
		}
		return 1
	}
	return 0
}

func run(ctx context.Context, getenv func(string) string, stdout io.Writer, stderr io.Writer) (runErr error) {
	environment, err := loadEnvironment(getenv)
	if err != nil {
		return err
	}
	repository, err := os.Getwd()
	if err != nil {
		slog.ErrorContext(ctx, "ci.auto_update.repository_resolve_failed", "err", err)
		return fmt.Errorf("resolve repository root: %w", err)
	}
	testRoot, err := os.MkdirTemp(testRootParent, testRootPattern)
	if err != nil {
		slog.ErrorContext(ctx, "ci.auto_update.root_create_failed", "err", err)
		return fmt.Errorf("create test root: %w", err)
	}
	slog.InfoContext(ctx, "ci.auto_update.started", "repository", environment.repository)
	check := &updateCheck{
		environment: environment,
		repository:  repository,
		testRoot:    testRoot,
		testHome:    filepath.Join(testRoot, "home"),
		installDir:  filepath.Join(testRoot, "bin"),
		daemonLog:   filepath.Join(testRoot, "daemon.log"),
		stateRoot:   filepath.Join(testRoot, "state"),
		configRoot:  filepath.Join(testRoot, "config"),
		cacheRoot:   filepath.Join(testRoot, "cache"),
		runtimeRoot: filepath.Join(testRoot, "run"),
		contextRoot: filepath.Join(testRoot, "context"),
		stdout:      stdout,
		stderr:      stderr,
	}
	defer func() {
		if cleanupErr := check.cleanup(); cleanupErr != nil {
			if runErr == nil {
				runErr = cleanupErr
				return
			}
			_, _ = fmt.Fprintf(stderr, "auto-update cleanup failed: %v\n", cleanupErr)
		}
	}()
	if err := check.prepareDirectories(); err != nil {
		return err
	}
	selection, err := check.fetchReleaseSelection(ctx)
	if err != nil {
		return err
	}
	_, _ = fmt.Fprintf(stdout, "testing automatic update from %s to %s\n", selection.previous.TagName, selection.target.TagName)
	if err := check.installOldRelease(ctx, selection.previous); err != nil {
		return err
	}
	if err := check.startAuthenticatedProxy(ctx); err != nil {
		return err
	}
	if err := check.startDaemon(ctx); err != nil {
		return err
	}
	if err := check.waitForUpdate(ctx, selection.target.TagName); err != nil {
		check.printDiagnostics()
		return err
	}
	_, _ = fmt.Fprintf(stdout, "automatic update applied to all release binaries: %s\n", selection.target.TagName)
	return nil
}

func loadEnvironment(getenv func(string) string) (environment, error) {
	environment := environment{
		repository: strings.TrimSpace(getenv("GITHUB_REPOSITORY")),
		commit:     strings.TrimSpace(getenv("GITHUB_SHA")),
		refType:    strings.TrimSpace(getenv("GITHUB_REF_TYPE")),
		refName:    strings.TrimSpace(getenv("GITHUB_REF_NAME")),
		token:      strings.TrimSpace(getenv("GH_TOKEN")),
		manual:     strings.TrimSpace(getenv("GITHUB_EVENT_NAME")) == "workflow_dispatch",
	}
	required := []struct {
		name  string
		value string
	}{
		{name: "GITHUB_REPOSITORY", value: environment.repository},
		{name: "GITHUB_SHA", value: environment.commit},
		{name: "GITHUB_REF_TYPE", value: environment.refType},
		{name: "GITHUB_REF_NAME", value: environment.refName},
		{name: "GH_TOKEN", value: environment.token},
	}
	for _, variable := range required {
		if variable.value == "" {
			return environment, fmt.Errorf("%s is required", variable.name)
		}
	}
	if len(environment.commit) < 8 {
		return environment, fmt.Errorf("GITHUB_SHA must contain at least eight characters")
	}
	return environment, nil
}

func selectReleases(releases []githubRelease, environment environment) (releaseSelection, error) {
	eligible := make([]githubRelease, 0, len(releases))
	for _, release := range releases {
		if !release.Draft && strings.TrimSpace(release.TagName) != "" {
			eligible = append(eligible, release)
		}
	}
	sort.SliceStable(eligible, func(i int, j int) bool {
		return eligible[i].PublishedAt.After(eligible[j].PublishedAt)
	})
	if environment.manual {
		if len(eligible) < 2 {
			return releaseSelection{}, fmt.Errorf("manual run requires at least two published releases")
		}
		return releaseSelection{target: eligible[0], previous: eligible[1]}, nil
	}
	targetTag := environment.refName
	if environment.refType != "tag" {
		targetTag = ""
		for _, release := range eligible {
			if releaseMatchesCommit(release.TagName, environment.commit) {
				targetTag = release.TagName
				break
			}
		}
	}
	if targetTag == "" {
		return releaseSelection{}, fmt.Errorf("published release for commit %s was not found", environment.commit)
	}
	for i, release := range eligible {
		if release.TagName != targetTag {
			continue
		}
		if i+1 >= len(eligible) {
			return releaseSelection{}, fmt.Errorf("release %s has no preceding release", targetTag)
		}
		return releaseSelection{target: release, previous: eligible[i+1]}, nil
	}
	return releaseSelection{}, fmt.Errorf("target release %s was not found", targetTag)
}

func releaseMatchesCommit(tag string, commit string) bool {
	fields := strings.Split(strings.TrimSpace(tag), "-")
	if len(fields) < 2 {
		return false
	}
	suffix := fields[len(fields)-1]
	return len(suffix) >= 7 && strings.HasPrefix(commit, suffix)
}

func (check *updateCheck) fetchReleaseSelection(ctx context.Context) (releaseSelection, error) {
	client := &http.Client{Timeout: requestTimeout}
	var lastErr error
	for attempt := 1; attempt <= maxReleaseAttempts; attempt++ {
		releases, err := fetchGitHubReleases(ctx, client, check.environment)
		if err == nil {
			selection, selectErr := selectReleases(releases, check.environment)
			if selectErr == nil {
				return selection, nil
			}
			lastErr = selectErr
		} else {
			lastErr = err
		}
		_, _ = fmt.Fprintf(check.stderr, "release selection attempt %d failed: %v\n", attempt, lastErr)
		if err := wait(ctx, pollInterval); err != nil {
			return releaseSelection{}, err
		}
	}
	slog.WarnContext(ctx, "ci.auto_update.release_selection_failed", "err", lastErr)
	return releaseSelection{}, fmt.Errorf("resolve target and preceding releases: %w", lastErr)
}

func fetchGitHubReleases(ctx context.Context, client *http.Client, environment environment) ([]githubRelease, error) {
	requestURL := githubAPIBaseURL + "/repos/" + environment.repository + "/releases?per_page=100"
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL, nil)
	if err != nil {
		slog.WarnContext(ctx, "ci.auto_update.release_request_failed", "err", err)
		return nil, fmt.Errorf("build GitHub releases request: %w", err)
	}
	request.Header.Set("Accept", "application/vnd.github+json")
	request.Header.Set("Authorization", "Bearer "+environment.token)
	request.Header.Set("X-Github-Api-Version", "2022-11-28")
	response, err := client.Do(request)
	if err != nil {
		slog.WarnContext(ctx, "ci.auto_update.release_fetch_failed", "err", err)
		return nil, fmt.Errorf("query GitHub releases: %w", err)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		slog.WarnContext(ctx, "ci.auto_update.release_fetch_rejected", "status", response.StatusCode)
		return nil, fmt.Errorf("query GitHub releases: HTTP %d", response.StatusCode)
	}
	var releases []githubRelease
	if err := json.NewDecoder(io.LimitReader(response.Body, maxGitHubBodyBytes)).Decode(&releases); err != nil {
		slog.WarnContext(ctx, "ci.auto_update.release_decode_failed", "err", err)
		return nil, fmt.Errorf("decode GitHub releases: %w", err)
	}
	return releases, nil
}

func (check *updateCheck) prepareDirectories() error {
	for _, directory := range []string{check.testHome, check.installDir, check.stateRoot, check.configRoot, check.cacheRoot, check.runtimeRoot, check.contextRoot} {
		if err := os.MkdirAll(directory, 0o700); err != nil {
			slog.Warn("ci.auto_update.directory_create_failed", "path", directory, "err", err)
			return fmt.Errorf("create test directory %s: %w", directory, err)
		}
	}
	return nil
}

func (check *updateCheck) installOldRelease(ctx context.Context, release githubRelease) error {
	installerPath := filepath.Join(check.repository, "install.sh")
	command := exec.CommandContext(ctx, installerPath,
		"--bin-dir", check.installDir,
		"--no-service",
		"--version", release.TagName,
		"--require-attestation",
	)
	command.Env = replaceEnvironment(os.Environ(), map[string]string{
		"HOME":            check.testHome,
		"XDG_CACHE_HOME":  check.cacheRoot,
		"XDG_CONFIG_HOME": check.configRoot,
		"XDG_RUNTIME_DIR": check.runtimeRoot,
		"XDG_STATE_HOME":  check.stateRoot,
	})
	command.Stdout = check.stdout
	command.Stderr = check.stderr
	if err := command.Run(); err != nil {
		slog.WarnContext(ctx, "ci.auto_update.old_release_install_failed", "tag", release.TagName, "err", err)
		return fmt.Errorf("install old release %s: %w", release.TagName, err)
	}
	for _, binary := range releaseBinaries {
		destination := filepath.Join(check.installDir, binary)
		installedVersion, err := binaryVersion(ctx, destination)
		if err != nil {
			return err
		}
		if installedVersion != release.TagName {
			slog.WarnContext(ctx, "ci.auto_update.old_release_version_mismatch", "binary", binary, "got", installedVersion, "want", release.TagName)
			return fmt.Errorf("installed %s version is %q, expected %q", binary, installedVersion, release.TagName)
		}
	}
	return nil
}

func authenticatedProxy(target *url.URL, token string) *httputil.ReverseProxy {
	return &httputil.ReverseProxy{Rewrite: func(request *httputil.ProxyRequest) {
		request.SetURL(target)
		request.Out.Header.Set("Authorization", "Bearer "+token)
	}}
}

func (check *updateCheck) startAuthenticatedProxy(ctx context.Context) error {
	target, err := url.Parse(githubAPIBaseURL)
	if err != nil {
		slog.WarnContext(ctx, "ci.auto_update.api_proxy_url_invalid", "err", err)
		return fmt.Errorf("parse GitHub API URL: %w", err)
	}
	listener, err := (&net.ListenConfig{}).Listen(ctx, "tcp", "localhost:0")
	if err != nil {
		slog.WarnContext(ctx, "ci.auto_update.api_proxy_listen_failed", "err", err)
		return fmt.Errorf("listen for authenticated GitHub API proxy: %w", err)
	}
	server := httptest.NewUnstartedServer(authenticatedProxy(target, check.environment.token))
	server.Listener = listener
	server.Start()
	check.apiProxy = server
	return nil
}

func (check *updateCheck) startDaemon(ctx context.Context) error {
	slog.InfoContext(ctx, "ci.auto_update.daemon_starting")
	if check.apiProxy == nil {
		return fmt.Errorf("authenticated GitHub API proxy is not running")
	}
	logFile, err := os.OpenFile(check.daemonLog, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		slog.WarnContext(ctx, "ci.auto_update.daemon_log_create_failed", "err", err)
		return fmt.Errorf("create daemon log: %w", err)
	}
	daemonPath := filepath.Join(check.installDir, daemonBinary)
	socketPath := filepath.Join(check.stateRoot, "sockets", "lm-semantic-search-daemon.sock")
	command := exec.CommandContext(context.WithoutCancel(ctx), daemonPath,
		"--profile", "offline",
		"--offline-embedding-model", "bge-small",
		"--state-root", check.stateRoot,
		"--socket", socketPath,
	)
	command.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	command.Env = replaceEnvironment(os.Environ(), map[string]string{
		"CLAUDE_CONTEXTD_CONFIG_ROOT":            check.configRoot,
		"CLAUDE_CONTEXTD_CONTEXT_ROOT":           check.contextRoot,
		"CLAUDE_CONTEXTD_MODEL_CACHE_ROOT":       filepath.Join(check.cacheRoot, "models"),
		"CLAUDE_CONTEXT_BACKGROUND_SYNC":         "false",
		"CLAUDE_CONTEXT_DEBUG_LISTENER":          "false",
		"CLAUDE_CONTEXT_FILE_WATCHER":            "false",
		"CLAUDE_CONTEXT_RESUME_ON_BOOT":          "false",
		"CLAUDE_CONTEXT_TRIGGER_WATCHER":         "false",
		"HOME":                                   check.testHome,
		"LM_SEMANTIC_SEARCH_UPDATE_API_BASE_URL": check.apiProxy.URL,
		"XDG_CACHE_HOME":                         check.cacheRoot,
		"XDG_CONFIG_HOME":                        check.configRoot,
		"XDG_RUNTIME_DIR":                        check.runtimeRoot,
		"XDG_STATE_HOME":                         check.stateRoot,
	})
	command.Stdout = logFile
	command.Stderr = logFile
	if err := command.Start(); err != nil {
		_ = logFile.Close()
		slog.WarnContext(ctx, "ci.auto_update.daemon_start_failed", "err", err)
		return fmt.Errorf("start daemon: %w", err)
	}
	done := make(chan error, 1)
	go func() {
		var waitErr error
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.ErrorContext(ctx, "ci.auto_update.daemon_wait_panic", "err", fmt.Sprintf("panic: %v", recovered))
				waitErr = fmt.Errorf("wait for daemon panic: %v", recovered)
			}
			if closeErr := logFile.Close(); waitErr == nil {
				waitErr = closeErr
			}
			done <- waitErr
		}()
		waitErr = command.Wait()
	}()
	check.child = &childProcess{process: command.Process, done: done}
	return nil
}

func (process *childProcess) status() (error, bool) {
	if process == nil {
		return nil, true
	}
	if process.finished {
		return process.result, true
	}
	select {
	case process.result = <-process.done:
		process.finished = true
		return process.result, true
	default:
		return nil, false
	}
}

func (process *childProcess) stop() error {
	if process == nil || process.finished {
		return nil
	}
	processGroupID := -process.process.Pid
	if err := syscall.Kill(processGroupID, syscall.SIGTERM); err != nil && !errors.Is(err, syscall.ESRCH) {
		slog.Warn("ci.auto_update.daemon_signal_failed", "pid", process.process.Pid, "err", err)
		return fmt.Errorf("signal daemon process group: %w", err)
	}
	timer := time.NewTimer(processStopTimeout)
	defer timer.Stop()
	select {
	case process.result = <-process.done:
		process.finished = true
		return nil
	case <-timer.C:
	}
	if err := syscall.Kill(processGroupID, syscall.SIGKILL); err != nil && !errors.Is(err, syscall.ESRCH) {
		slog.Warn("ci.auto_update.daemon_kill_failed", "pid", process.process.Pid, "err", err)
		return fmt.Errorf("kill daemon process group: %w", err)
	}
	process.result = <-process.done
	process.finished = true
	return nil
}

func (check *updateCheck) waitForUpdate(ctx context.Context, targetTag string) error {
	for attempt := 1; attempt <= maxUpdateAttempts; attempt++ {
		allCurrent := true
		for _, binary := range releaseBinaries {
			current, err := binaryVersion(ctx, filepath.Join(check.installDir, binary))
			if err != nil || current != targetTag {
				allCurrent = false
				break
			}
		}
		applied, stateErr := stateReportsApplied(check.updateStatePath())
		if allCurrent && stateErr == nil && applied {
			return nil
		}
		if processErr, exited := check.child.status(); exited {
			if processErr == nil {
				slog.WarnContext(ctx, "ci.auto_update.daemon_exited_early")
				return fmt.Errorf("daemon exited before every binary reported the target version")
			}
			slog.WarnContext(ctx, "ci.auto_update.daemon_exited", "err", processErr)
			return fmt.Errorf("daemon exited before every binary reported the target version: %w", processErr)
		}
		if err := wait(ctx, pollInterval); err != nil {
			return err
		}
	}
	slog.WarnContext(ctx, "ci.auto_update.timed_out")
	return fmt.Errorf("automatic update did not apply within ten minutes")
}

func stateReportsApplied(path string) (bool, error) {
	state, err := selfupdate.LoadState(path)
	if err != nil {
		slog.Warn("ci.auto_update.state_load_failed", "path", path, "err", err)
		return false, fmt.Errorf("load update state: %w", err)
	}
	return state.LastResult == "applied", nil
}

func (check *updateCheck) updateStatePath() string {
	return filepath.Join(check.stateRoot, "update-state.json")
}

func binaryVersion(ctx context.Context, binary string) (string, error) {
	command := exec.CommandContext(ctx, binary, "version")
	var stdout strings.Builder
	var stderr strings.Builder
	command.Stdout = &stdout
	command.Stderr = &stderr
	if err := command.Run(); err != nil {
		slog.WarnContext(ctx, "ci.auto_update.version_failed", "binary", binary, "err", err)
		return "", fmt.Errorf("run %s version: %w: %s", binary, err, strings.TrimSpace(stderr.String()))
	}
	return parseVersion(stdout.String())
}

func parseVersion(output string) (string, error) {
	for line := range strings.SplitSeq(output, "\n") {
		value, found := strings.CutPrefix(line, "version:")
		if !found {
			continue
		}
		fields := strings.Fields(value)
		if len(fields) > 0 {
			return fields[0], nil
		}
	}
	return "", fmt.Errorf("version output has no version field")
}

func (check *updateCheck) printDiagnostics() {
	_, _ = fmt.Fprintln(check.stderr, "daemon output:")
	lines, err := tailLines(check.daemonLog, diagnosticLineCount)
	if err != nil {
		_, _ = fmt.Fprintf(check.stderr, "(unavailable: %v)\n", err)
	} else {
		for _, line := range lines {
			_, _ = fmt.Fprintln(check.stderr, line)
		}
	}
	_, _ = fmt.Fprintln(check.stderr, "daemon structured log:")
	structuredLog := filepath.Join(check.stateRoot, "logs", "lm-semantic-search-daemon.log")
	lines, err = tailLines(structuredLog, diagnosticLineCount)
	if err != nil {
		_, _ = fmt.Fprintf(check.stderr, "(unavailable: %v)\n", err)
	} else {
		for _, line := range lines {
			_, _ = fmt.Fprintln(check.stderr, line)
		}
	}
	_, _ = fmt.Fprintln(check.stderr, "update state:")
	content, err := os.ReadFile(check.updateStatePath())
	if err != nil {
		_, _ = fmt.Fprintf(check.stderr, "(unavailable: %v)\n", err)
		return
	}
	_, _ = check.stderr.Write(content)
}

func tailLines(path string, count int) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		slog.Warn("ci.auto_update.diagnostic_open_failed", "path", path, "err", err)
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer func() { _ = file.Close() }()
	lines := make([]string, 0, count)
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		if len(lines) == count {
			copy(lines, lines[1:])
			lines[count-1] = scanner.Text()
			continue
		}
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		slog.Warn("ci.auto_update.diagnostic_scan_failed", "path", path, "err", err)
		return nil, fmt.Errorf("scan %s: %w", path, err)
	}
	return lines, nil
}

func (check *updateCheck) cleanup() error {
	var cleanupErrors []error
	if check.child != nil {
		if err := check.child.stop(); err != nil {
			cleanupErrors = append(cleanupErrors, err)
		}
	}
	if check.apiProxy != nil {
		check.apiProxy.Close()
	}
	if err := removeTestRoot(check.testRoot); err != nil {
		cleanupErrors = append(cleanupErrors, err)
	}
	err := errors.Join(cleanupErrors...)
	if err != nil {
		slog.Warn("ci.auto_update.cleanup_failed", "err", err)
	}
	return err
}

func removeTestRoot(path string) error {
	cleaned := filepath.Clean(path)
	relative, err := filepath.Rel(testRootParent, cleaned)
	if err != nil {
		slog.Warn("ci.auto_update.root_resolve_failed", "path", path, "err", err)
		return fmt.Errorf("resolve test root %s: %w", path, err)
	}
	if relative == "." || relative == ".." || strings.HasPrefix(relative, ".."+string(os.PathSeparator)) {
		return fmt.Errorf("refuse to remove test root outside the temp directory: %s", path)
	}
	if !strings.HasPrefix(filepath.Base(cleaned), testRootPattern) {
		return fmt.Errorf("refuse to remove unexpected test root %s", path)
	}
	if err := os.RemoveAll(cleaned); err != nil {
		slog.Warn("ci.auto_update.root_remove_failed", "path", cleaned, "err", err)
		return fmt.Errorf("remove test root %s: %w", cleaned, err)
	}
	return nil
}

func replaceEnvironment(current []string, replacements map[string]string) []string {
	result := make([]string, 0, len(current)+len(replacements))
	for _, entry := range current {
		name, _, found := strings.Cut(entry, "=")
		if found {
			if _, replaced := replacements[name]; replaced {
				continue
			}
		}
		result = append(result, entry)
	}
	for name, value := range replacements {
		result = append(result, name+"="+value)
	}
	return result
}

func wait(ctx context.Context, duration time.Duration) error {
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		slog.WarnContext(ctx, "ci.auto_update.wait_interrupted", "err", ctx.Err())
		return fmt.Errorf("wait interrupted: %w", ctx.Err())
	case <-timer.C:
		return nil
	}
}

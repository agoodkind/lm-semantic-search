//go:build installlive

package installlive

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	"goodkind.io/lm-semantic-search/internal/onnxruntimedist"
	"goodkind.io/lm-semantic-search/internal/sandbox"
)

const (
	gklogVersionPackage = "goodkind.io/gklog/version"
	// oldReleaseVersion predates every published release. update apply always
	// finds a release newer than the stamped CLI.
	oldReleaseVersion = "202601010000-1-0000000"

	oldMCPContent    = "installed mcp before the update\n"
	oldDaemonContent = "installed daemon before the update\n"

	githubAPIBaseURL       = "https://api.github.com"
	updateAPIBaseURLEnv    = "LM_SEMANTIC_SEARCH_UPDATE_API_BASE_URL"
	releaseListPath        = "/repos/agoodkind/lm-semantic-search/releases"
	releaseAssetPathPrefix = "/repos/agoodkind/lm-semantic-search/releases/assets/"
	githubAPIHost          = "api.github.com"
	// The updater also fetches the release commit attestation at
	// attestations/sha1:<commit>, which every archive shares. Only the
	// sha256 paths identify one archive each.
	attestationPathPrefix = "/repos/agoodkind/lm-semantic-search/attestations/sha256:"
)

var releaseBinaries = []string{cliBinary, mcpBinary, daemonBinary}

// updateApplyDirectory is a temporary install directory holding a CLI stamped
// with an old release version and placeholder MCP and daemon files. Running
// that CLI's update apply makes updateopts.ApplyAll update the three binaries
// in this directory from the newest GitHub release.
type updateApplyDirectory struct {
	installDir string
	root       string
	cliPath    string
	cliBytes   []byte
}

func newUpdateApplyDirectory(t *testing.T, withONNXRuntime bool) updateApplyDirectory {
	t.Helper()
	// The root sits under /tmp instead of the longer platform temp directory.
	// The kernel caps a socket path near 104 bytes.
	root, err := os.MkdirTemp("/tmp", "lms-update-apply-")
	if err != nil {
		t.Fatalf("create root: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	installDir := filepath.Join(root, "bin")
	if err := os.MkdirAll(installDir, 0o755); err != nil {
		t.Fatalf("create install dir: %v", err)
	}

	cliPath := filepath.Join(installDir, cliBinary)
	ldflags := strings.Join([]string{
		"-X", gklogVersionPackage + ".Version=" + oldReleaseVersion,
		"-X", gklogVersionPackage + ".Commit=0000000",
		"-X", gklogVersionPackage + ".Dirty=false",
	}, " ")
	build := exec.Command("go", "build", "-ldflags", ldflags, "-o", cliPath, cliPackage)
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build stamped CLI: %v\n%s", err, output)
	}
	cliBytes, err := os.ReadFile(cliPath)
	if err != nil {
		t.Fatalf("read stamped CLI: %v", err)
	}
	writeFile(t, filepath.Join(installDir, mcpBinary), []byte(oldMCPContent))
	writeFile(t, filepath.Join(installDir, daemonBinary), []byte(oldDaemonContent))

	if withONNXRuntime {
		archive, err := onnxruntimedist.ArchiveFor(runtime.GOOS, runtime.GOARCH)
		if err != nil {
			t.Fatalf("resolve ONNX Runtime archive: %v", err)
		}
		names, err := onnxruntimedist.LibraryNamesFor(runtime.GOOS)
		if err != nil {
			t.Fatalf("resolve ONNX Runtime library names: %v", err)
		}
		archiveDirectory, err := onnxruntimedist.FetchArchive(context.Background(), http.DefaultClient, archive, t.TempDir())
		if err != nil {
			t.Fatalf("fetch ONNX Runtime archive: %v", err)
		}
		if err := onnxruntimedist.InstallLibrary(archiveDirectory, names, installDir); err != nil {
			t.Fatalf("install ONNX Runtime beside the binaries: %v", err)
		}
	}
	return updateApplyDirectory{installDir: installDir, root: root, cliPath: cliPath, cliBytes: cliBytes}
}

func writeFile(t *testing.T, path string, content []byte) {
	t.Helper()
	if err := os.WriteFile(path, content, 0o755); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// runUpdateApply runs the stamped CLI with its state, socket, config, and
// context roots under the temporary root, using the sandbox isolation table.
func (directory updateApplyDirectory) runUpdateApply(t *testing.T, proxy *githubAPIProxy) commandResult {
	t.Helper()
	command := exec.Command(directory.cliPath, "update", "apply")
	environment := os.Environ()
	for _, variable := range sandbox.Env(directory.root) {
		if variable.Value != "" {
			environment = append(environment, variable.Name+"="+variable.Value)
		}
	}
	environment = append(environment, updateAPIBaseURLEnv+"="+proxy.server.URL)
	command.Env = environment
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr
	err := command.Run()
	exitCode := 0
	if err != nil {
		var exitError *exec.ExitError
		if !errors.As(err, &exitError) {
			t.Fatalf("run update apply: %v", err)
		}
		exitCode = exitError.ExitCode()
	}
	return commandResult{exitCode: exitCode, stdout: stdout.String(), stderr: stderr.String()}
}

// TestUpdateApplyReplacesAllBinariesWhenDaemonLoadsLibraryFromInstallDir runs
// update apply with ONNX Runtime installed beside the binaries. The daemon
// candidate loads libonnxruntime from its own directory during validation.
// The CLI, MCP, and daemon must all be replaced with the release binaries.
func TestUpdateApplyReplacesAllBinariesWhenDaemonLoadsLibraryFromInstallDir(t *testing.T) {
	directory := newUpdateApplyDirectory(t, true)
	proxy := startGitHubAPIProxy(t)

	result := directory.runUpdateApply(t, proxy)
	failOnGitHubRefusal(t, result)
	if result.exitCode != 0 {
		t.Fatalf("update apply exit = %d\nstdout:\n%s\nstderr:\n%s", result.exitCode, result.stdout, result.stderr)
	}
	proxy.assertReleaseVerifiedThroughProxy(t)
	for _, binary := range releaseBinaries {
		version := runCommand(t, filepath.Join(directory.installDir, binary), "version")
		if version.exitCode != 0 || !strings.Contains(version.stdout, "version:") {
			t.Fatalf("%s version after update exit = %d\nstdout:\n%s\nstderr:\n%s",
				binary, version.exitCode, version.stdout, version.stderr)
		}
		if strings.Contains(version.stdout, oldReleaseVersion) {
			t.Fatalf("%s still reports the stamped old version:\n%s", binary, version.stdout)
		}
	}
	assertNoHiddenUpdateFiles(t, directory.installDir)
}

// TestUpdateApplyLeavesAllBinariesWhenDaemonCandidateFails runs update apply
// without ONNX Runtime in the install directory. The daemon candidate cannot
// load libonnxruntime, the 2026-09-28 production failure. update apply must
// fail and leave the CLI, MCP, and daemon files unchanged.
func TestUpdateApplyLeavesAllBinariesWhenDaemonCandidateFails(t *testing.T) {
	directory := newUpdateApplyDirectory(t, false)
	proxy := startGitHubAPIProxy(t)

	result := directory.runUpdateApply(t, proxy)
	failOnGitHubRefusal(t, result)
	if result.exitCode == 0 {
		t.Fatalf("update apply succeeded without ONNX Runtime\nstdout:\n%s", result.stdout)
	}
	if !strings.Contains(result.stderr, "candidate version failed") {
		t.Fatalf("update apply stderr does not report the daemon candidate failure:\n%s", result.stderr)
	}
	proxy.assertReleaseVerifiedThroughProxy(t)
	assertFileContent(t, directory.cliPath, directory.cliBytes)
	assertFileContent(t, filepath.Join(directory.installDir, mcpBinary), []byte(oldMCPContent))
	assertFileContent(t, filepath.Join(directory.installDir, daemonBinary), []byte(oldDaemonContent))
	assertNoHiddenUpdateFiles(t, directory.installDir)
}

// githubAPIProxy forwards the updater's GitHub API requests to api.github.com
// unmodified, including the Authorization header the updater sends. It
// records each request path, whether the request had an Authorization header,
// and the response status. It never records the header value.
type githubAPIProxy struct {
	server    *httptest.Server
	mutex     sync.Mutex
	responses []apiResponse
}

type apiResponse struct {
	path       string
	authorized bool
	status     int
	// redirectHost is the Location host of a redirect response.
	redirectHost string
}

func startGitHubAPIProxy(t *testing.T) *githubAPIProxy {
	t.Helper()
	target, err := url.Parse(githubAPIBaseURL)
	if err != nil {
		t.Fatalf("parse GitHub API URL: %v", err)
	}
	proxy := &githubAPIProxy{}
	proxy.server = httptest.NewServer(&httputil.ReverseProxy{
		Rewrite: func(request *httputil.ProxyRequest) {
			request.SetURL(target)
		},
		ModifyResponse: func(response *http.Response) error {
			redirectHost := ""
			if location, err := response.Location(); err == nil {
				redirectHost = location.Hostname()
			}
			proxy.mutex.Lock()
			proxy.responses = append(proxy.responses, apiResponse{
				path:         response.Request.URL.Path,
				authorized:   response.Request.Header.Get("Authorization") != "",
				status:       response.StatusCode,
				redirectHost: redirectHost,
			})
			proxy.mutex.Unlock()
			return nil
		},
	})
	t.Cleanup(proxy.server.Close)
	return proxy
}

// assertReleaseVerifiedThroughProxy fails unless the updater listed the
// releases and fetched the attestations of all three release archives through
// the proxy, with every response HTTP 200. Each archive digest has its own
// attestation path.
func (proxy *githubAPIProxy) assertReleaseVerifiedThroughProxy(t *testing.T) {
	t.Helper()
	proxy.mutex.Lock()
	responses := append([]apiResponse(nil), proxy.responses...)
	proxy.mutex.Unlock()
	// The updater resolves GH_TOKEN and GITHUB_TOKEN before `gh auth token`.
	// When either variable is set, every request must carry the updater's own
	// Authorization header.
	tokenInEnvironment := false
	for _, name := range []string{"GH_TOKEN", "GITHUB_TOKEN"} {
		tokenInEnvironment = tokenInEnvironment || strings.TrimSpace(os.Getenv(name)) != ""
	}
	listedReleases := false
	attestationPaths := map[string]bool{}
	for _, response := range responses {
		// With a token the updater downloads each archive through the API
		// asset endpoint, which answers with a redirect to the storage host.
		assetRedirect := strings.HasPrefix(response.path, releaseAssetPathPrefix) && response.status == http.StatusFound
		if response.status != http.StatusOK && !assetRedirect {
			t.Fatalf("GitHub API %s returned HTTP %d through the proxy; all responses %v", response.path, response.status, responses)
		}
		// net/http drops Authorization on a redirect to a host that is not
		// the original host or its subdomain. In production the original host
		// is api.github.com. The storage host must be outside api.github.com.
		if assetRedirect && (response.redirectHost == "" || response.redirectHost == githubAPIHost ||
			strings.HasSuffix(response.redirectHost, "."+githubAPIHost)) {
			t.Fatalf("asset %s redirects to host %q, want a host outside %s", response.path, response.redirectHost, githubAPIHost)
		}
		if tokenInEnvironment && !response.authorized {
			t.Fatalf("updater sent %s without an Authorization header while a token variable is set; all responses %v",
				response.path, responses)
		}
		if response.path == releaseListPath {
			listedReleases = true
		}
		if strings.HasPrefix(response.path, attestationPathPrefix) {
			attestationPaths[response.path] = true
		}
	}
	if !listedReleases {
		t.Fatalf("updater sent no release list request through the proxy; responses %v", responses)
	}
	if len(attestationPaths) < len(releaseBinaries) {
		t.Fatalf("updater fetched attestations for %d archive digests, want %d; responses %v",
			len(attestationPaths), len(releaseBinaries), responses)
	}
}

// failOnGitHubRefusal fails with a quota error when GitHub refused an API
// request with HTTP 403 or 429. A refused request stops update apply before
// it stages any candidate, and the file assertions would then report the
// refusal as a replaced or unchanged binary.
func failOnGitHubRefusal(t *testing.T, result commandResult) {
	t.Helper()
	for _, status := range []string{"HTTP 403", "HTTP 429"} {
		if strings.Contains(result.stderr, status) {
			t.Fatalf("GitHub refused an update API request with %s (rate limit or missing credentials); "+
				"set GH_TOKEN or GITHUB_TOKEN, or run `gh auth login`\nstderr:\n%s", status, result.stderr)
		}
	}
}

func assertFileContent(t *testing.T, path string, want []byte) {
	t.Helper()
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("%s changed: %d bytes, want the original %d bytes", path, len(got), len(want))
	}
}

// assertNoHiddenUpdateFiles fails when a staged candidate or backup stays in
// the install directory after update apply returns.
func assertNoHiddenUpdateFiles(t *testing.T, installDir string) {
	t.Helper()
	for _, name := range directoryEntryNames(t, installDir) {
		if strings.HasPrefix(name, ".") {
			t.Fatalf("install directory keeps hidden update file %s", name)
		}
	}
}

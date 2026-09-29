//go:build installlive

package installlive

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"

	"goodkind.io/go-makefile/selfupdate"
	"goodkind.io/lm-semantic-search/internal/sandbox"
	"goodkind.io/lm-semantic-search/internal/updateopts"
)

const (
	refusedDownloadPath = "/refused-download"
	stateRootEnv        = "CLAUDE_CONTEXTD_STATE_ROOT"
	updateStateFileName = "update-state.json"
)

// installedState records every binary and library file and every library
// symlink target in a bin directory after an install.
type installedState struct {
	files    map[string][]byte
	symlinks map[string]string
	entries  []string
}

func readInstalledState(t *testing.T, binDir string) installedState {
	t.Helper()
	names := expectedONNXLibraryNames(t)
	state := installedState{files: map[string][]byte{}, symlinks: map[string]string{}, entries: directoryEntryNames(t, binDir)}
	for _, name := range []string{cliBinary, mcpBinary, daemonBinary, names.versioned} {
		content, err := os.ReadFile(filepath.Join(binDir, name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		state.files[name] = content
	}
	for _, name := range []string{names.soname, names.unversioned, cliAlias} {
		target, err := os.Readlink(filepath.Join(binDir, name))
		if err != nil {
			t.Fatalf("read %s symlink: %v", name, err)
		}
		state.symlinks[name] = target
	}
	return state
}

func assertInstalledStateUnchanged(t *testing.T, binDir string, want installedState) {
	t.Helper()
	got := readInstalledState(t, binDir)
	for name, content := range want.files {
		if !bytes.Equal(got.files[name], content) {
			t.Fatalf("%s changed: %d bytes, want the installed %d bytes", name, len(got.files[name]), len(content))
		}
	}
	for name, target := range want.symlinks {
		if got.symlinks[name] != target {
			t.Fatalf("%s -> %q, want the installed target %q", name, got.symlinks[name], target)
		}
	}
	if strings.Join(got.entries, ",") != strings.Join(want.entries, ",") {
		t.Fatalf("bin dir = %v, want the installed entries %v", got.entries, want.entries)
	}
}

// installExistingRelease installs the newest release into a new bin directory
// under root, then replaces each binary with distinct placeholder bytes, the
// state of an install from an older release. A second install that replaces
// any binary then changes its bytes. It returns the directory and its state.
func installExistingRelease(t *testing.T, root string) (string, installedState) {
	t.Helper()
	binDir := t.TempDir()
	result := runInstallInRoot(t, root, binDir)
	failOnGitHubRefusal(t, result)
	if result.exitCode != 0 {
		t.Fatalf("first install exit = %d\nstdout:\n%s\nstderr:\n%s", result.exitCode, result.stdout, result.stderr)
	}
	for _, binary := range []string{cliBinary, mcpBinary, daemonBinary} {
		writeFile(t, filepath.Join(binDir, binary), []byte("installed "+binary+" from an older release\n"))
	}
	return binDir, readInstalledState(t, binDir)
}

// cliDownloadRefusingProxy forwards release API requests to api.github.com and
// refuses the CLI archive download with HTTP 404. It rewrites the CLI asset
// download URL in each release list to its own refused path, and it answers
// the API asset path of that asset with HTTP 404.
type cliDownloadRefusingProxy struct {
	server   *httptest.Server
	mutex    sync.Mutex
	assetIDs map[int64]bool
	refused  int
}

func startCLIDownloadRefusingProxy(t *testing.T) *cliDownloadRefusingProxy {
	t.Helper()
	target, err := url.Parse(githubAPIBaseURL)
	if err != nil {
		t.Fatalf("parse GitHub API URL: %v", err)
	}
	cliAssetName := cliBinary + "_" + runtime.GOOS + "_" + runtime.GOARCH + ".tar.gz"
	proxy := &cliDownloadRefusingProxy{assetIDs: map[int64]bool{}}
	reverseProxy := &httputil.ReverseProxy{
		Rewrite: func(request *httputil.ProxyRequest) {
			request.SetURL(target)
			request.Out.Header.Del("Accept-Encoding")
		},
		ModifyResponse: func(response *http.Response) error {
			if response.Request.URL.Path != releaseListPath || response.StatusCode != http.StatusOK {
				return nil
			}
			body, err := io.ReadAll(response.Body)
			_ = response.Body.Close()
			if err != nil {
				return err
			}
			var releases []map[string]any
			if err := json.Unmarshal(body, &releases); err != nil {
				return err
			}
			for _, release := range releases {
				assets, _ := release["assets"].([]any)
				for _, rawAsset := range assets {
					asset, _ := rawAsset.(map[string]any)
					if asset["name"] != cliAssetName {
						continue
					}
					asset["browser_download_url"] = proxy.server.URL + refusedDownloadPath
					if id, ok := asset["id"].(float64); ok {
						proxy.mutex.Lock()
						proxy.assetIDs[int64(id)] = true
						proxy.mutex.Unlock()
					}
				}
			}
			rewritten, err := json.Marshal(releases)
			if err != nil {
				return err
			}
			response.Body = io.NopCloser(bytes.NewReader(rewritten))
			response.ContentLength = int64(len(rewritten))
			response.Header.Set("Content-Length", strconv.Itoa(len(rewritten)))
			return nil
		},
	}
	proxy.server = httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if proxy.refuses(request.URL.Path) {
			proxy.mutex.Lock()
			proxy.refused++
			proxy.mutex.Unlock()
			http.NotFound(writer, request)
			return
		}
		reverseProxy.ServeHTTP(writer, request)
	}))
	t.Cleanup(proxy.server.Close)
	return proxy
}

func (proxy *cliDownloadRefusingProxy) refuses(path string) bool {
	if path == refusedDownloadPath {
		return true
	}
	idText, found := strings.CutPrefix(path, releaseAssetPathPrefix)
	if !found {
		return false
	}
	id, err := strconv.ParseInt(idText, 10, 64)
	if err != nil {
		return false
	}
	proxy.mutex.Lock()
	defer proxy.mutex.Unlock()
	return proxy.assetIDs[id]
}

func (proxy *cliDownloadRefusingProxy) refusedRequests() int {
	proxy.mutex.Lock()
	defer proxy.mutex.Unlock()
	return proxy.refused
}

// TestInstallLeavesInstalledSetWhenOneDownloadFails installs the newest
// release, then installs again while the CLI archive download answers HTTP
// 404. The second install must fail and leave every binary, the library file,
// and every library symlink target unchanged, with no hidden file left.
func TestInstallLeavesInstalledSetWhenOneDownloadFails(t *testing.T) {
	root := newIsolatedRoot(t)
	binDir, installed := installExistingRelease(t, root)
	proxy := startCLIDownloadRefusingProxy(t)

	environment := append(isolatedEnvironment(root), updateAPIBaseURLEnv+"="+proxy.server.URL)
	result := runCommandWithEnvironment(t, environment, builtCLIPath, "install", "--no-service", "--bin-dir", binDir)
	failOnGitHubRefusal(t, result)
	if result.exitCode == 0 {
		t.Fatalf("install with a refused CLI download succeeded\nstdout:\n%s", result.stdout)
	}
	if proxy.refusedRequests() == 0 {
		t.Fatalf("install never requested the refused CLI archive\nstderr:\n%s", result.stderr)
	}
	if !strings.Contains(result.stderr, "HTTP 404") {
		t.Fatalf("install stderr does not report the refused download:\n%s", result.stderr)
	}
	assertInstalledStateUnchanged(t, binDir, installed)
}

// TestInstallFailsWhileUpdateLockIsHeld installs the newest release, then runs
// install again while this process takes the update lock at the state path
// that updateopts.StatePath returns, the path update apply locks. The second
// install must fail with the lock error and leave the installed set
// unchanged. An install that locked a different path would succeed.
func TestInstallFailsWhileUpdateLockIsHeld(t *testing.T) {
	root := newIsolatedRoot(t)
	binDir, installed := installExistingRelease(t, root)
	isolatedStateRoot := ""
	for _, variable := range sandbox.Env(root) {
		t.Setenv(variable.Name, variable.Value)
		if variable.Name == stateRootEnv {
			isolatedStateRoot = variable.Value
		}
	}
	statePath, err := updateopts.StatePath(updateopts.Overrides{})
	if err != nil {
		t.Fatalf("updateopts.StatePath: %v", err)
	}
	if want := filepath.Join(isolatedStateRoot, updateStateFileName); statePath != want {
		t.Fatalf("updateopts.StatePath = %q, want the isolated %q", statePath, want)
	}

	var result commandResult
	lockErr := selfupdate.WithLock(context.Background(), statePath, func() error {
		result = runInstallInRoot(t, root, binDir)
		return nil
	})
	if lockErr != nil {
		t.Fatalf("take update lock: %v", lockErr)
	}
	if result.exitCode == 0 {
		t.Fatalf("install succeeded while the update lock was held\nstdout:\n%s", result.stdout)
	}
	if !strings.Contains(result.stderr, "update already running") {
		t.Fatalf("install stderr does not report the held lock:\n%s", result.stderr)
	}
	assertInstalledStateUnchanged(t, binDir, installed)
}

package updateopts

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"goodkind.io/go-makefile/selfupdate"
)

const (
	ghEnvToken     = "gh-env-test-token"
	githubEnvToken = "github-env-test-token"
	ghCLIToken     = "gh-cli-test-token"
)

// releaseQueryRecorder answers every release query with an empty release list
// and records the Authorization header of each request.
type releaseQueryRecorder struct {
	mutex          sync.Mutex
	authorizations []string
}

func (recorder *releaseQueryRecorder) ServeHTTP(writer http.ResponseWriter, request *http.Request) {
	recorder.mutex.Lock()
	recorder.authorizations = append(recorder.authorizations, request.Header.Get("Authorization"))
	recorder.mutex.Unlock()
	writer.Header().Set("Content-Type", "application/json")
	_, _ = writer.Write([]byte("[]"))
}

// writeGitHubCLI writes a gh executable that prints token for `gh auth token`
// into a new directory and returns that directory.
func writeGitHubCLI(t *testing.T, token string) string {
	t.Helper()
	directory := t.TempDir()
	script := "#!/bin/sh\nif [ \"$1 $2\" = \"auth token\" ]; then printf '%s\\n' '" + token + "'; exit 0; fi\nexit 1\n"
	if err := os.WriteFile(filepath.Join(directory, "gh"), []byte(script), 0o755); err != nil {
		t.Fatalf("write gh: %v", err)
	}
	return directory
}

// updatePaths are the two update operations that query the release API: the
// check that the CLI `update check` command runs, and ApplyAll, which the CLI
// `update apply` command and the daemon scheduler run.
var updatePaths = []struct {
	name string
	run  func(t *testing.T, overrides Overrides)
}{
	{name: "check", run: func(t *testing.T, overrides Overrides) {
		t.Helper()
		option, err := NetworkCheckOptions(context.Background(), overrides)
		if err != nil {
			t.Fatalf("NetworkCheckOptions: %v", err)
		}
		_, _ = selfupdate.Check(context.Background(), option)
	}},
	{name: "apply", run: func(t *testing.T, overrides Overrides) {
		t.Helper()
		_, _ = ApplyAll(context.Background(), overrides)
	}},
}

// TestUpdateSendsResolvedGitHubToken runs the update check and ApplyAll
// against a local release API. It asserts the Authorization header for each
// credential source in priority order: GH_TOKEN, GITHUB_TOKEN,
// `gh auth token`, and no header when none resolves.
func TestUpdateSendsResolvedGitHubToken(t *testing.T) {
	cases := []struct {
		name        string
		ghToken     string
		githubToken string
		ghCLIToken  string
		want        string
	}{
		{name: "GH_TOKEN wins over GITHUB_TOKEN and gh", ghToken: ghEnvToken, githubToken: githubEnvToken, ghCLIToken: ghCLIToken, want: "Bearer " + ghEnvToken},
		{name: "GITHUB_TOKEN wins over gh", githubToken: githubEnvToken, ghCLIToken: ghCLIToken, want: "Bearer " + githubEnvToken},
		{name: "gh auth token when no variable is set", ghCLIToken: ghCLIToken, want: "Bearer " + ghCLIToken},
		{name: "no header when nothing resolves", want: ""},
	}
	for _, path := range updatePaths {
		for _, testCase := range cases {
			t.Run(path.name+"/"+testCase.name, func(t *testing.T) {
				recorder := &releaseQueryRecorder{}
				server := httptest.NewServer(recorder)
				t.Cleanup(server.Close)
				t.Setenv(updateAPIBaseURLEnv, server.URL)
				t.Setenv("GH_TOKEN", testCase.ghToken)
				t.Setenv("GITHUB_TOKEN", testCase.githubToken)
				pathDirectory := t.TempDir()
				if testCase.ghCLIToken != "" {
					pathDirectory = writeGitHubCLI(t, testCase.ghCLIToken)
				}
				t.Setenv("PATH", pathDirectory)

				path.run(t, Overrides{InstallDir: t.TempDir(), StateRoot: t.TempDir(), CacheDir: t.TempDir()})

				recorder.mutex.Lock()
				defer recorder.mutex.Unlock()
				if len(recorder.authorizations) == 0 {
					t.Fatalf("update %s sent no release query", path.name)
				}
				for _, authorization := range recorder.authorizations {
					if authorization != testCase.want {
						t.Fatalf("Authorization = %q, want %q", authorization, testCase.want)
					}
				}
			})
		}
	}
}

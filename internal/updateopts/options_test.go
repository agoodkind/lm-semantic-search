package updateopts

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"goodkind.io/gklog/version"
)

func TestOptionsForInstallDirBuildsSharedStateOptionsInApplyOrder(t *testing.T) {
	client := &http.Client{}
	log := slog.Default()
	stateRoot := t.TempDir()
	cacheDir := t.TempDir()

	options, err := OptionsForInstallDir("/opt/lm/bin", Overrides{
		Client:    client,
		StateRoot: stateRoot,
		CacheDir:  cacheDir,
		DryRun:    true,
		Log:       log,
	})
	if err != nil {
		t.Fatalf("OptionsForInstallDir returned error: %v", err)
	}

	gotOrder := make([]string, 0, len(options))
	for _, option := range options {
		gotOrder = append(gotOrder, option.Config.Binary)
		if option.Config.Repo != "agoodkind/lm-semantic-search" {
			t.Fatalf("Repo = %q, want agoodkind/lm-semantic-search", option.Config.Repo)
		}
		if option.Config.CurrentVersion != version.Version {
			t.Fatalf("CurrentVersion = %q, want %q", option.Config.CurrentVersion, version.Version)
		}
		if option.Config.CurrentCommit != version.Commit {
			t.Fatalf("CurrentCommit = %q, want %q", option.Config.CurrentCommit, version.Commit)
		}
		if option.Config.CurrentBuildHash != version.BuildHash() {
			t.Fatalf("CurrentBuildHash = %q, want %q", option.Config.CurrentBuildHash, version.BuildHash())
		}
		wantLocal := isLocalBuild(version.Version, version.Dirty == "true")
		if option.Config.CurrentDirty != wantLocal {
			t.Fatalf("CurrentDirty = %v, want %v", option.Config.CurrentDirty, wantLocal)
		}
		if option.Config.AllowPrerelease != nil {
			t.Fatalf("AllowPrerelease = %v, want nil", option.Config.AllowPrerelease)
		}
		if !reflect.DeepEqual(option.Config.ValidateArgs, []string{"version"}) {
			t.Fatalf("ValidateArgs = %#v, want version", option.Config.ValidateArgs)
		}
		if option.Config.ValidateMatch != "version:" {
			t.Fatalf("ValidateMatch = %q, want version:", option.Config.ValidateMatch)
		}
		if option.StatePath != filepath.Join(stateRoot, "update-state.json") {
			t.Fatalf("StatePath = %q, want shared daemon state path", option.StatePath)
		}
		if option.CacheDir != cacheDir {
			t.Fatalf("CacheDir = %q, want override", option.CacheDir)
		}
		if option.Client != client {
			t.Fatalf("Client override was not preserved")
		}
		if option.Log != log {
			t.Fatalf("Log override was not preserved")
		}
		if !option.DryRun {
			t.Fatalf("DryRun = false, want true")
		}
	}

	wantOrder := []string{
		"lm-semantic-search",
		"lm-semantic-search-mcp",
		"lm-semantic-search-daemon",
	}
	if !reflect.DeepEqual(gotOrder, wantOrder) {
		t.Fatalf("binary order = %#v, want %#v", gotOrder, wantOrder)
	}
}

// TestApplyAllLogsApplyFailuresWithOperationLogger runs ApplyAll against a
// local release API that answers the release query with HTTP 500. ApplyAll
// must return the error and write the failure to the operation logger.
func TestApplyAllLogsApplyFailuresWithOperationLogger(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		http.Error(writer, "injected failure", http.StatusInternalServerError)
	}))
	t.Cleanup(server.Close)
	t.Setenv(updateAPIBaseURLEnv, server.URL)
	t.Setenv("GH_TOKEN", "")
	t.Setenv("GITHUB_TOKEN", "")
	t.Setenv("PATH", t.TempDir())

	var output bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&output, nil)).With("component", "update")
	_, err := ApplyAll(context.Background(), Overrides{
		InstallDir: t.TempDir(),
		StateRoot:  t.TempDir(),
		CacheDir:   t.TempDir(),
		Log:        logger,
	})
	if err == nil || !strings.Contains(err.Error(), "HTTP 500") {
		t.Fatalf("ApplyAll error = %v, want the release query HTTP 500", err)
	}

	logOutput := output.String()
	if !strings.Contains(logOutput, "apply binary updates failed") {
		t.Fatalf("operation logger output = %q, want apply failure message", logOutput)
	}
	if !strings.Contains(logOutput, "component=update") {
		t.Fatalf("operation logger output = %q, want component field", logOutput)
	}
}

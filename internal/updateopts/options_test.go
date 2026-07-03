package updateopts

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"path/filepath"
	"reflect"
	"testing"

	"goodkind.io/gklog/version"
	"goodkind.io/go-makefile/selfupdate"
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
		if option.Config.CurrentBuildHash != version.BinHash {
			t.Fatalf("CurrentBuildHash = %q, want %q", option.Config.CurrentBuildHash, version.BinHash)
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

func TestApplyAllAbortsBeforeDaemonWhenClientApplyFails(t *testing.T) {
	originalApply := applyBinary
	t.Cleanup(func() {
		applyBinary = originalApply
	})
	clientFailure := errors.New("client apply failed")
	calls := []string{}
	applyBinary = func(ctx context.Context, options selfupdate.Options) (selfupdate.ApplyResult, error) {
		_ = ctx
		calls = append(calls, options.Config.Binary)
		if options.Config.Binary == "lm-semantic-search-mcp" {
			return selfupdate.ApplyResult{}, clientFailure
		}
		return selfupdate.ApplyResult{Applied: true}, nil
	}

	_, err := ApplyAll(context.Background(), Overrides{
		InstallDir: t.TempDir(),
		StateRoot:  t.TempDir(),
		CacheDir:   t.TempDir(),
	})
	if !errors.Is(err, clientFailure) {
		t.Fatalf("ApplyAll error = %v, want client failure", err)
	}

	wantCalls := []string{"lm-semantic-search", "lm-semantic-search-mcp"}
	if !reflect.DeepEqual(calls, wantCalls) {
		t.Fatalf("apply calls = %#v, want %#v", calls, wantCalls)
	}
}

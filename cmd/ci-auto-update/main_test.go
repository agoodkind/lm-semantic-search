package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"goodkind.io/go-makefile/selfupdate"
)

func TestSelectReleasesForBranchBuild(t *testing.T) {
	t.Parallel()
	releases := []githubRelease{
		{TagName: "202608122141-d2-abcdef1"},
		{TagName: "202608122028-d1-1234567"},
	}
	environment := environment{commit: "abcdef1234567890", refType: "branch"}

	selection, err := selectReleases(releases, environment)
	if err != nil {
		t.Fatalf("selectReleases() error = %v", err)
	}
	if selection.target.TagName != releases[0].TagName {
		t.Fatalf("target = %q, want %q", selection.target.TagName, releases[0].TagName)
	}
	if selection.previous.TagName != releases[1].TagName {
		t.Fatalf("previous = %q, want %q", selection.previous.TagName, releases[1].TagName)
	}
}

func TestSelectReleasesForManualRunUsesLatestRelease(t *testing.T) {
	t.Parallel()
	releases := []githubRelease{
		{TagName: "202608122141-d2-abcdef1", PublishedAt: time.Date(2026, time.August, 12, 21, 41, 0, 0, time.UTC)},
		{TagName: "202608122028-d1-1234567", PublishedAt: time.Date(2026, time.August, 12, 20, 28, 0, 0, time.UTC)},
	}
	environment := environment{
		commit:  "unreleased123456",
		refType: "branch",
		manual:  true,
	}

	selection, err := selectReleases(releases, environment)
	if err != nil {
		t.Fatalf("selectReleases() error = %v", err)
	}
	if selection.target.TagName != releases[0].TagName {
		t.Fatalf("target = %q, want %q", selection.target.TagName, releases[0].TagName)
	}
	if selection.previous.TagName != releases[1].TagName {
		t.Fatalf("previous = %q, want %q", selection.previous.TagName, releases[1].TagName)
	}
}

func TestLoadEnvironmentDetectsManualRun(t *testing.T) {
	t.Parallel()
	values := map[string]string{
		"GITHUB_REPOSITORY": "agoodkind/lm-semantic-search",
		"GITHUB_SHA":        "abcdef1234567890",
		"GITHUB_REF_TYPE":   "branch",
		"GITHUB_REF_NAME":   "main",
		"GITHUB_EVENT_NAME": "workflow_dispatch",
		"GH_TOKEN":          "token",
	}

	environment, err := loadEnvironment(func(name string) string { return values[name] })
	if err != nil {
		t.Fatalf("loadEnvironment() error = %v", err)
	}
	if !environment.manual {
		t.Fatal("manual = false, want true")
	}
}

func TestSelectReleasesSkipsDrafts(t *testing.T) {
	t.Parallel()
	releases := []githubRelease{
		{TagName: "draft", Draft: true},
		{TagName: "202608122141-d2-abcdef1"},
		{TagName: "202608122028-d1-1234567"},
	}
	environment := environment{commit: "abcdef1234567890", refType: "branch"}

	selection, err := selectReleases(releases, environment)
	if err != nil {
		t.Fatalf("selectReleases() error = %v", err)
	}
	if selection.previous.TagName != releases[2].TagName {
		t.Fatalf("previous = %q, want %q", selection.previous.TagName, releases[2].TagName)
	}
}

func TestSelectReleasesUsesPublishedOrder(t *testing.T) {
	t.Parallel()
	releases := []githubRelease{
		{TagName: "202608122028-d1-1234567", PublishedAt: time.Date(2026, time.August, 12, 20, 28, 0, 0, time.UTC)},
		{TagName: "202608122141-d2-abcdef1", PublishedAt: time.Date(2026, time.August, 12, 21, 41, 0, 0, time.UTC)},
	}
	environment := environment{commit: "abcdef1234567890", refType: "branch"}

	selection, err := selectReleases(releases, environment)
	if err != nil {
		t.Fatalf("selectReleases() error = %v", err)
	}
	if selection.previous.TagName != releases[0].TagName {
		t.Fatalf("previous = %q, want %q", selection.previous.TagName, releases[0].TagName)
	}
}

func TestStateReportsApplied(t *testing.T) {
	t.Parallel()
	statePath := filepath.Join(t.TempDir(), "update-state.json")
	if err := selfupdate.SaveState(statePath, selfupdate.State{LastResult: "applied"}); err != nil {
		t.Fatalf("SaveState() error = %v", err)
	}
	applied, err := stateReportsApplied(statePath)
	if err != nil {
		t.Fatalf("stateReportsApplied() error = %v", err)
	}
	if !applied {
		t.Fatal("stateReportsApplied() = false, want true")
	}
}

func TestParseVersion(t *testing.T) {
	t.Parallel()
	output := "version: 202608122141-d2-9504f44 commit=9504f44 build_time=now\n"
	got, err := parseVersion(output)
	if err != nil {
		t.Fatalf("parseVersion() error = %v", err)
	}
	if got != "202608122141-d2-9504f44" {
		t.Fatalf("parseVersion() = %q", got)
	}
}

func TestRemoveTestRootRejectsTempDirectory(t *testing.T) {
	t.Parallel()
	if err := removeTestRoot(os.TempDir()); err == nil {
		t.Fatal("removeTestRoot() error = nil, want refusal")
	}
}

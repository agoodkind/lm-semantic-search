package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"goodkind.io/go-makefile/selfupdate"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

func TestAuthenticatedProxyAddsTokenAndPreservesRequest(t *testing.T) {
	t.Parallel()
	target, err := url.Parse("https://api.github.com")
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	var receivedRequest *http.Request
	proxy := authenticatedProxy(target, "ci-token")
	proxy.Transport = roundTripFunc(func(request *http.Request) (*http.Response, error) {
		receivedRequest = request.Clone(request.Context())
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader("ok")),
		}, nil
	})
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "http://localhost/repos/fork/lms/releases?per_page=100", nil)

	proxy.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	if receivedRequest == nil {
		t.Fatal("proxy did not forward request")
	}
	if got := receivedRequest.Header.Get("Authorization"); got != "Bearer ci-token" {
		t.Fatalf("Authorization = %q, want bearer token", got)
	}
	if got := receivedRequest.URL.String(); got != "https://api.github.com/repos/fork/lms/releases?per_page=100" {
		t.Fatalf("URL = %q", got)
	}
}

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

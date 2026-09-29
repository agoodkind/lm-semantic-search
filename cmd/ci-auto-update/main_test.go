package main

import (
	"encoding/json"
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
	proxy := authenticatedProxy(target, "ci-token", githubRelease{TagName: "202609291328-112-582d81e"})
	proxy.Transport = roundTripFunc(func(request *http.Request) (*http.Response, error) {
		receivedRequest = request.Clone(request.Context())
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader("[]")),
			Request:    request,
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

const (
	proxyTestNewerTag  = "202609291344-114-73e8f0b"
	proxyTestTargetTag = "202609291328-112-582d81e"
	proxyTestOlderTag  = "202609291250-111-1e81c83"
	proxyTestReleases  = `[
{"tag_name":"` + proxyTestNewerTag + `","draft":false,"prerelease":true,"published_at":"2026-09-29T13:53:08Z","assets":[{"name":"newer.tar.gz"}]},
{"tag_name":"` + proxyTestTargetTag + `","draft":false,"prerelease":true,"published_at":"2026-09-29T13:41:35Z","assets":[{"name":"target.tar.gz","digest":"sha256:abc"}]},
{"tag_name":"` + proxyTestOlderTag + `","draft":false,"prerelease":true,"published_at":"2026-09-29T12:57:00Z","assets":[{"name":"older.tar.gz"}]}
]`
	proxyTestAttestationPath = "/repos/fork/lms/attestations/sha256:abc"
	proxyTestAttestationBody = `{"attestations":[{"bundle":{"mediaType":"application/vnd.dev.sigstore.bundle.v0.3+json"}}]}`
)

// TestAuthenticatedProxyHidesReleasesPublishedAfterTarget sends the daemon's
// release list and attestation requests through the proxy to a local GitHub
// API server. The release list from the proxy must contain the target and the
// older release, with their assets, and omit the release published after the
// target. The attestation response must pass through unchanged.
func TestAuthenticatedProxyHidesReleasesPublishedAfterTarget(t *testing.T) {
	t.Parallel()
	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/repos/fork/lms/releases":
			_, _ = io.WriteString(writer, proxyTestReleases)
		case proxyTestAttestationPath:
			_, _ = io.WriteString(writer, proxyTestAttestationBody)
		default:
			http.NotFound(writer, request)
		}
	}))
	t.Cleanup(upstream.Close)
	target, err := url.Parse(upstream.URL)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	newest := githubRelease{
		TagName:     proxyTestTargetTag,
		PublishedAt: time.Date(2026, 9, 29, 13, 41, 35, 0, time.UTC),
	}
	proxy := httptest.NewServer(authenticatedProxy(target, "ci-token", newest))
	t.Cleanup(proxy.Close)

	listBody := getProxyBody(t, proxy.URL+"/repos/fork/lms/releases?per_page=100")
	var releases []struct {
		TagName string `json:"tag_name"`
		Assets  []struct {
			Name   string `json:"name"`
			Digest string `json:"digest"`
		} `json:"assets"`
	}
	if err := json.Unmarshal([]byte(listBody), &releases); err != nil {
		t.Fatalf("decode proxied release list %q: %v", listBody, err)
	}
	tags := make([]string, 0, len(releases))
	for _, release := range releases {
		tags = append(tags, release.TagName)
	}
	if strings.Join(tags, ",") != proxyTestTargetTag+","+proxyTestOlderTag {
		t.Fatalf("proxied release tags = %v, want the target and the older release", tags)
	}
	if len(releases[0].Assets) != 1 || releases[0].Assets[0].Digest != "sha256:abc" {
		t.Fatalf("proxied target assets = %+v, want the target archive digest", releases[0].Assets)
	}

	if got := getProxyBody(t, proxy.URL+proxyTestAttestationPath); got != proxyTestAttestationBody {
		t.Fatalf("proxied attestation body = %q, want the upstream body", got)
	}
}

func getProxyBody(t *testing.T, requestURL string) string {
	t.Helper()
	response, err := http.Get(requestURL)
	if err != nil {
		t.Fatalf("GET %s: %v", requestURL, err)
	}
	defer func() { _ = response.Body.Close() }()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatalf("read %s: %v", requestURL, err)
	}
	if response.StatusCode != http.StatusOK {
		t.Fatalf("GET %s status = %d, body %q", requestURL, response.StatusCode, body)
	}
	return string(body)
}

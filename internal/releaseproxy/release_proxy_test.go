package releaseproxy_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/releaseproxy"
)

const (
	proxyTestToken       = "ci-token"
	proxyTestReleasePath = "/repos/fork/lms/releases"
	proxyTestQuery       = "per_page=100"
	proxyTestNewerTag    = "202609291344-114-73e8f0b"
	proxyTestTargetTag   = "202609291328-112-582d81e"
	proxyTestOlderTag    = "202609291250-111-1e81c83"
	proxyTestReleases    = `[
{"tag_name":"` + proxyTestNewerTag + `","draft":false,"prerelease":true,"published_at":"2026-09-29T13:53:08Z","assets":[{"name":"newer.tar.gz"}]},
{"tag_name":"` + proxyTestTargetTag + `","draft":false,"prerelease":true,"published_at":"2026-09-29T13:41:35Z","assets":[{"name":"target.tar.gz","digest":"sha256:abc"}]},
{"tag_name":"` + proxyTestOlderTag + `","draft":false,"prerelease":true,"published_at":"2026-09-29T12:57:00Z","assets":[{"name":"older.tar.gz"}]}
]`
	proxyTestAttestationPath = "/repos/fork/lms/attestations/sha256:abc"
	proxyTestAttestationBody = `{"attestations":[{"bundle":{"mediaType":"application/vnd.dev.sigstore.bundle.v0.3+json"}}]}`
)

var proxyTestCutoff = time.Date(2026, 9, 29, 13, 41, 35, 0, time.UTC)

type receivedRequest struct {
	authorization string
	path          string
	rawQuery      string
}

func TestAuthenticatedProxyAddsTokenAndPreservesRequest(t *testing.T) {
	t.Parallel()
	received := make(chan receivedRequest, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		received <- receivedRequest{
			authorization: request.Header.Get("Authorization"),
			path:          request.URL.Path,
			rawQuery:      request.URL.RawQuery,
		}
		_, _ = io.WriteString(writer, "[]")
	}))
	t.Cleanup(upstream.Close)
	proxy := startProxy(t, upstream.URL)

	getProxyBody(t, proxy.URL+proxyTestReleasePath+"?"+proxyTestQuery)

	var forwarded receivedRequest
	select {
	case forwarded = <-received:
	default:
		t.Fatal("proxy did not forward request")
	}
	if forwarded.authorization != "Bearer "+proxyTestToken {
		t.Fatalf("Authorization = %q, want bearer token", forwarded.authorization)
	}
	if forwarded.path != proxyTestReleasePath || forwarded.rawQuery != proxyTestQuery {
		t.Fatalf("forwarded path = %q, query = %q", forwarded.path, forwarded.rawQuery)
	}
}

// TestAuthenticatedProxyHidesReleasesPublishedAfterTarget sends the daemon's
// release list and attestation requests through the proxy to a local GitHub
// API server. The release list from the proxy must contain the target and the
// older release, with their assets, and omit the release published after the
// target. The attestation response must pass through unchanged.
func TestAuthenticatedProxyHidesReleasesPublishedAfterTarget(t *testing.T) {
	t.Parallel()
	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case proxyTestReleasePath:
			_, _ = io.WriteString(writer, proxyTestReleases)
		case proxyTestAttestationPath:
			_, _ = io.WriteString(writer, proxyTestAttestationBody)
		default:
			http.NotFound(writer, request)
		}
	}))
	t.Cleanup(upstream.Close)
	proxy := startProxy(t, upstream.URL)

	listBody := getProxyBody(t, proxy.URL+proxyTestReleasePath+"?"+proxyTestQuery)
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

func startProxy(t *testing.T, upstreamURL string) *httptest.Server {
	t.Helper()
	target, err := url.Parse(upstreamURL)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	proxy := httptest.NewServer(releaseproxy.New(target, proxyTestToken, proxyTestCutoff))
	t.Cleanup(proxy.Close)
	return proxy
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

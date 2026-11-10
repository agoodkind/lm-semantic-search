package updateopts

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"runtime"
	"sync"
	"testing"

	"goodkind.io/go-makefile/selfupdate"
)

const (
	redirectTestTag     = "202609290000-1-0123abc"
	redirectTestAssetID = 7
)

// authorizationRecorder records the Authorization header of every request a
// handler receives.
type authorizationRecorder struct {
	mutex          sync.Mutex
	authorizations []string
}

func (recorder *authorizationRecorder) record(request *http.Request) {
	recorder.mutex.Lock()
	recorder.authorizations = append(recorder.authorizations, request.Header.Get("Authorization"))
	recorder.mutex.Unlock()
}

func (recorder *authorizationRecorder) snapshot() []string {
	recorder.mutex.Lock()
	defer recorder.mutex.Unlock()
	return append([]string(nil), recorder.authorizations...)
}

// TestAssetRedirectDropsGitHubToken downloads a release archive with a token
// the way the updater does. A local release API answers the asset request with
// HTTP 302 to a storage server on a different host name, as api.github.com
// does. The storage request must arrive without the Authorization header.
// selfupdate downloads through options.Client, and the default net/http
// client drops Authorization on a redirect to a host that is not the original
// host or its subdomain.
func TestAssetRedirectDropsGitHubToken(t *testing.T) {
	archive := []byte("release archive bytes")
	digest := sha256.Sum256(archive)

	storageRequests := &authorizationRecorder{}
	storage := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		storageRequests.record(request)
		_, _ = writer.Write(archive)
	}))
	t.Cleanup(storage.Close)
	storageURL, err := url.Parse(storage.URL)
	if err != nil {
		t.Fatalf("parse storage URL: %v", err)
	}
	// The API server listens on 127.0.0.1; the storage URL uses the host name
	// localhost, a different host for the redirect rule.
	storageLocation := "http://localhost:" + storageURL.Port() + "/archive"

	assetName := daemonBinary + "_" + runtime.GOOS + "_" + runtime.GOARCH + ".tar.gz"
	assetPath := "/repos/" + repository + "/releases/assets/7"
	assetRequests := &authorizationRecorder{}
	api := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/repos/" + repository + "/releases/tags/" + redirectTestTag:
			release := map[string]any{
				"tag_name": redirectTestTag,
				"assets": []map[string]any{{
					"id":                   redirectTestAssetID,
					"name":                 assetName,
					"browser_download_url": storage.URL + "/unused",
					"digest":               "sha256:" + hex.EncodeToString(digest[:]),
				}},
			}
			_ = json.NewEncoder(writer).Encode(release)
		case assetPath:
			assetRequests.record(request)
			http.Redirect(writer, request, storageLocation, http.StatusFound)
		default:
			http.NotFound(writer, request)
		}
	}))
	t.Cleanup(api.Close)

	t.Setenv(updateAPIBaseURLEnv, api.URL)
	t.Setenv("GH_TOKEN", ghEnvToken)
	t.Setenv("GITHUB_TOKEN", "")
	option, err := NetworkCheckOptions(context.Background(), Overrides{InstallDir: t.TempDir(), StateRoot: t.TempDir(), CacheDir: t.TempDir()})
	if err != nil {
		t.Fatalf("NetworkCheckOptions: %v", err)
	}
	// Attestation verification fails against the local API after the download,
	// so the returned error is expected and ignored.
	_ = selfupdate.VerifyReleaseAssets(context.Background(), option, redirectTestTag)

	assetAuthorizations := assetRequests.snapshot()
	if len(assetAuthorizations) != 1 || assetAuthorizations[0] != "Bearer "+ghEnvToken {
		t.Fatalf("API asset request authorizations = %q, want one request with the token", assetAuthorizations)
	}
	storageAuthorizations := storageRequests.snapshot()
	if len(storageAuthorizations) != 1 {
		t.Fatalf("storage requests = %d, want the one redirected download", len(storageAuthorizations))
	}
	if storageAuthorizations[0] != "" {
		t.Fatal("redirected storage request carried an Authorization header")
	}
}

// Package releaseproxy filters GitHub release lists for automatic update checks.
package releaseproxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"strings"
	"time"
)

const maxReleaseListBytes = 4 << 20

type githubRelease struct {
	TagName     string    `json:"tag_name"`
	PublishedAt time.Time `json:"published_at"`
}

// New forwards token to target as a bearer token.
// The proxy removes releases published after cutoff from successful /releases
// lists because the daemon applies the newest listed release.
// The proxy returns /releases/latest responses unchanged.
// The proxy answers upstream or response rewrite failures with HTTP 502.
func New(target *url.URL, token string, cutoff time.Time) *httputil.ReverseProxy {
	return &httputil.ReverseProxy{
		Rewrite: func(request *httputil.ProxyRequest) {
			request.SetURL(target)
			request.Out.Header.Set("Authorization", "Bearer "+token)
			// The transport then requests and decodes gzip itself, and
			// ModifyResponse reads a plain release list body.
			request.Out.Header.Del("Accept-Encoding")
		},
		ModifyResponse: func(response *http.Response) error {
			return dropReleasesPublishedAfter(response, cutoff)
		},
		ErrorHandler: func(writer http.ResponseWriter, _ *http.Request, _ error) {
			writer.WriteHeader(http.StatusBadGateway)
		},
	}
}

// dropReleasesPublishedAfter rewrites a successful release list response to
// omit releases published after cutoff. It leaves every other response
// unchanged.
func dropReleasesPublishedAfter(response *http.Response, cutoff time.Time) error {
	if response.StatusCode != http.StatusOK || !strings.HasSuffix(response.Request.URL.Path, "/releases") {
		return nil
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, maxReleaseListBytes))
	_ = response.Body.Close()
	if err != nil {
		slog.Warn("ci.auto_update.release_list_read_failed", "err", err)
		return fmt.Errorf("read release list: %w", err)
	}
	var entries []json.RawMessage
	if err := json.Unmarshal(body, &entries); err != nil {
		slog.Warn("ci.auto_update.release_list_decode_failed", "err", err)
		return fmt.Errorf("decode release list: %w", err)
	}
	kept := make([]json.RawMessage, 0, len(entries))
	hiddenTags := make([]string, 0)
	for _, entry := range entries {
		var release githubRelease
		if err := json.Unmarshal(entry, &release); err != nil {
			slog.Warn("ci.auto_update.release_decode_failed", "err", err)
			return fmt.Errorf("decode release: %w", err)
		}
		if release.PublishedAt.After(cutoff) {
			hiddenTags = append(hiddenTags, release.TagName)
			continue
		}
		kept = append(kept, entry)
	}
	if len(hiddenTags) > 0 {
		slog.Info("ci.auto_update.releases_hidden", "tags", hiddenTags, "cutoff", cutoff)
	}
	rewritten, err := json.Marshal(kept)
	if err != nil {
		slog.Warn("ci.auto_update.release_list_encode_failed", "err", err)
		return fmt.Errorf("encode release list: %w", err)
	}
	response.Body = io.NopCloser(bytes.NewReader(rewritten))
	response.ContentLength = int64(len(rewritten))
	response.Header.Set("Content-Length", strconv.Itoa(len(rewritten)))
	return nil
}

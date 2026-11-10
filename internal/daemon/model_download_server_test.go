package daemon_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/networkcost"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

//testdouble:external operating system network cost classification source
type fixedNetworkSource struct {
	classification networkcost.Classification
}

func (source fixedNetworkSource) Classify(context.Context) networkcost.Classification {
	return source.classification
}

type modelArtifactServer struct {
	server           *httptest.Server
	modelContent     []byte
	tokenizerContent []byte
	requests         atomic.Int64
	modelMissing     atomic.Bool
	dropModel        atomic.Bool
	holding          atomic.Bool
	resumedRange     atomic.Value
	holdModel        chan struct{}
	releaseOnce      sync.Once
}

func newModelArtifactServer(t *testing.T, holdModel bool) *modelArtifactServer {
	t.Helper()
	artifacts := &modelArtifactServer{
		modelContent:     bytes.Repeat([]byte("lms-model-bytes."), testModelBytes/16),
		tokenizerContent: []byte(`{"version":"lms-test"}`),
		holdModel:        make(chan struct{}),
	}
	artifacts.holding.Store(holdModel)
	artifacts.resumedRange.Store("")
	artifacts.server = httptest.NewServer(http.HandlerFunc(artifacts.serve))
	t.Cleanup(func() {
		artifacts.release()
		artifacts.server.Close()
	})
	return artifacts
}

func (artifacts *modelArtifactServer) release() {
	artifacts.releaseOnce.Do(func() {
		close(artifacts.holdModel)
	})
}

func (artifacts *modelArtifactServer) serveModelRange(
	writer http.ResponseWriter,
	rangeHeader string,
) {
	artifacts.resumedRange.Store(rangeHeader)
	offsetText := strings.TrimSuffix(strings.TrimPrefix(rangeHeader, "bytes="), "-")
	offset, err := strconv.Atoi(offsetText)
	if err != nil || offset <= 0 || offset >= len(artifacts.modelContent) {
		http.Error(writer, "unsatisfiable range", http.StatusRequestedRangeNotSatisfiable)
		return
	}
	total := len(artifacts.modelContent)
	writer.Header().Set("Content-Range", "bytes "+strconv.Itoa(offset)+"-"+strconv.Itoa(total-1)+"/"+strconv.Itoa(total))
	writer.Header().Set("Content-Length", strconv.Itoa(total-offset))
	writer.WriteHeader(http.StatusPartialContent)
	_, _ = writer.Write(artifacts.modelContent[offset:])
}

func (artifacts *modelArtifactServer) serve(writer http.ResponseWriter, request *http.Request) {
	artifacts.requests.Add(1)
	switch request.URL.Path {
	case "/" + testTokenizerName:
		_, _ = writer.Write(artifacts.tokenizerContent)
	case "/" + testModelArtifactName:
		if artifacts.modelMissing.Load() {
			http.NotFound(writer, request)
			return
		}
		rangeHeader := request.Header.Get("Range")
		dropping := artifacts.dropModel.Load()
		if rangeHeader != "" && dropping {
			panic(http.ErrAbortHandler)
		}
		if rangeHeader != "" {
			artifacts.serveModelRange(writer, rangeHeader)
			return
		}
		half := len(artifacts.modelContent) / 2
		writer.Header().Set("Content-Length", strconv.Itoa(len(artifacts.modelContent)))
		_, _ = writer.Write(artifacts.modelContent[:half])
		if flusher, ok := writer.(http.Flusher); ok {
			flusher.Flush()
		}
		if dropping {
			panic(http.ErrAbortHandler)
		}
		if artifacts.holding.Load() {
			select {
			case <-artifacts.holdModel:
			case <-request.Context().Done():
				return
			}
		}
		_, _ = writer.Write(artifacts.modelContent[half:])
	default:
		http.NotFound(writer, request)
	}
}

func (artifacts *modelArtifactServer) registry() *offlinemodel.Registry {
	modelSum := sha256.Sum256(artifacts.modelContent)
	tokenizerSum := sha256.Sum256(artifacts.tokenizerContent)
	return offlinemodel.NewRegistry(offlinemodel.EmbeddingGemma, offlinemodel.Preset{
		Name:            offlinemodel.EmbeddingGemma,
		ModelONNXURL:    artifacts.server.URL + "/" + testModelArtifactName,
		ModelSHA256:     hex.EncodeToString(modelSum[:]),
		TokenizerURL:    artifacts.server.URL + "/" + testTokenizerName,
		TokenizerSHA256: hex.EncodeToString(tokenizerSum[:]),
		Dimension:       4,
		Pooling:         offlinemodel.PoolingMean,
		MaximumTokens:   16,
	})
}

func (artifacts *modelArtifactServer) options(
	classification networkcost.Classification,
	interval time.Duration,
) daemon.ModelDownloadOptions {
	return daemon.ModelDownloadOptions{
		NetworkSource:   fixedNetworkSource{classification: classification},
		HTTPClient:      artifacts.server.Client(),
		Presets:         artifacts.registry(),
		RecheckInterval: interval,
		RetryInterval:   interval,
		AttemptBackoff:  testAttemptBackoff,
	}
}

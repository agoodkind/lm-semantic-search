package modeldownload_test

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
)

const (
	artifactName          = "model.onnx"
	artifactSizeBytes     = 96 * 1024
	fixtureFilePermission = 0o600
	maximumAttempts       = 4
	concurrentCalls       = 6
)

type serverBehavior int

const (
	behaviorServe serverBehavior = iota
	behaviorDropMidway
	behaviorIgnoreRange
	behaviorServiceUnavailable
	behaviorNotFound
	behaviorStallMidway
)

type artifactServer struct {
	server       *httptest.Server
	content      []byte
	contentPath  string
	mutex        sync.Mutex
	behaviors    []serverBehavior
	rangeHeaders []string
}

func newArtifactServer(t *testing.T, behaviors ...serverBehavior) *artifactServer {
	t.Helper()
	content := make([]byte, artifactSizeBytes)
	for i := range content {
		content[i] = byte((i * 31) % 251)
	}
	contentPath := filepath.Join(t.TempDir(), artifactName)
	if err := os.WriteFile(contentPath, content, fixtureFilePermission); err != nil {
		t.Fatalf("WriteFile %s: %v", contentPath, err)
	}
	fixture := &artifactServer{
		server:       nil,
		content:      content,
		contentPath:  contentPath,
		mutex:        sync.Mutex{},
		behaviors:    behaviors,
		rangeHeaders: nil,
	}
	fixture.server = httptest.NewServer(http.HandlerFunc(fixture.serve))
	t.Cleanup(fixture.server.Close)
	return fixture
}

func (fixture *artifactServer) nextBehavior(request *http.Request) serverBehavior {
	fixture.mutex.Lock()
	defer fixture.mutex.Unlock()
	fixture.rangeHeaders = append(fixture.rangeHeaders, request.Header.Get("Range"))
	if len(fixture.behaviors) == 0 {
		return behaviorServe
	}
	behavior := fixture.behaviors[0]
	fixture.behaviors = fixture.behaviors[1:]
	return behavior
}

func (fixture *artifactServer) serve(writer http.ResponseWriter, request *http.Request) {
	switch fixture.nextBehavior(request) {
	case behaviorServe:
		http.ServeFile(writer, request, fixture.contentPath)
	case behaviorIgnoreRange:
		request.Header.Del("Range")
		http.ServeFile(writer, request, fixture.contentPath)
	case behaviorServiceUnavailable:
		writer.WriteHeader(http.StatusServiceUnavailable)
	case behaviorNotFound:
		writer.WriteHeader(http.StatusNotFound)
	case behaviorDropMidway:
		fixture.writeHalf(writer, request)
		panic(http.ErrAbortHandler)
	case behaviorStallMidway:
		fixture.writeHalf(writer, request)
		<-request.Context().Done()
	}
}

func (fixture *artifactServer) writeHalf(writer http.ResponseWriter, request *http.Request) {
	start := 0
	rangeValue := request.Header.Get("Range")
	if rangeValue != "" {
		first := strings.TrimSuffix(strings.TrimPrefix(rangeValue, "bytes="), "-")
		parsed, err := strconv.Atoi(first)
		if err != nil {
			writer.WriteHeader(http.StatusBadRequest)
			return
		}
		start = parsed
	}
	remaining := fixture.content[start:]
	writer.Header().Set("Content-Length", strconv.Itoa(len(remaining)))
	if start > 0 {
		contentRange := "bytes " + strconv.Itoa(start) + "-" +
			strconv.Itoa(len(fixture.content)-1) + "/" + strconv.Itoa(len(fixture.content))
		writer.Header().Set("Content-Range", contentRange)
		writer.WriteHeader(http.StatusPartialContent)
	}
	_, _ = writer.Write(remaining[:len(remaining)/2])
	flusher, ok := writer.(http.Flusher)
	if ok {
		flusher.Flush()
	}
}

func (fixture *artifactServer) ranges() []string {
	fixture.mutex.Lock()
	defer fixture.mutex.Unlock()
	return slices.Clone(fixture.rangeHeaders)
}

func (fixture *artifactServer) sha256() string {
	digest := sha256.Sum256(fixture.content)
	return hex.EncodeToString(digest[:])
}

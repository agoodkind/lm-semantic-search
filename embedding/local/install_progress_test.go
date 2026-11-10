package local_test

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"sync/atomic"
	"testing"

	"goodkind.io/lm-semantic-search/embedding/local"
)

const (
	servedArtifactBytes   = 4096
	presetModelFilename   = "model.onnx"
	testCertificateServer = "example.com"
)

type reportedProgress struct {
	artifact        string
	downloadedBytes int64
	totalBytes      int64
}

func TestInstallModelWithProgressReportsBytesAndRejectsWrongChecksum(t *testing.T) {
	var requestCount atomic.Int64
	served := make([]byte, servedArtifactBytes)
	server := httptest.NewTLSServer(http.HandlerFunc(
		func(writer http.ResponseWriter, _ *http.Request) {
			requestCount.Add(1)
			writer.Header().Set("Content-Length", strconv.Itoa(len(served)))
			_, _ = writer.Write(served)
		},
	))
	t.Cleanup(server.Close)

	client := server.Client()
	transport, ok := client.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("client transport type = %T, want *http.Transport", client.Transport)
	}
	var dialer net.Dialer
	serverAddress := server.Listener.Addr().String()
	transport.DialContext = func(ctx context.Context, network string, _ string) (net.Conn, error) {
		return dialer.DialContext(ctx, network, serverAddress)
	}
	transport.TLSClientConfig.ServerName = testCertificateServer

	cacheRoot := t.TempDir()
	var reported []reportedProgress
	err := local.InstallModelWithProgress(
		context.Background(),
		client,
		cacheRoot,
		testModel,
		func(artifact string, downloadedBytes int64, totalBytes int64) {
			reported = append(reported, reportedProgress{
				artifact:        artifact,
				downloadedBytes: downloadedBytes,
				totalBytes:      totalBytes,
			})
		},
	)
	if !errors.Is(err, local.ErrModelUnavailable) {
		t.Fatalf("InstallModelWithProgress err = %v, want ErrModelUnavailable", err)
	}
	if requestCount.Load() != 1 {
		t.Fatalf("requests = %d, want 1", requestCount.Load())
	}
	if len(reported) == 0 {
		t.Fatal("progress calls = 0, want at least 1")
	}
	want := reportedProgress{
		artifact:        presetModelFilename,
		downloadedBytes: servedArtifactBytes,
		totalBytes:      servedArtifactBytes,
	}
	if final := reported[len(reported)-1]; final != want {
		t.Fatalf("final progress = %+v, want %+v", final, want)
	}
	modelDirectory := filepath.Join(cacheRoot, "embedding-models", testModel)
	for _, name := range []string{presetModelFilename, presetModelFilename + ".partial"} {
		_, statErr := os.Stat(filepath.Join(modelDirectory, name))
		if !errors.Is(statErr, os.ErrNotExist) {
			t.Fatalf("Stat %s err = %v, want os.ErrNotExist", name, statErr)
		}
	}
}

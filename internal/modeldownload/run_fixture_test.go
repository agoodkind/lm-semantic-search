package modeldownload_test

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/modeldownload"
)

type progressCall struct {
	artifact        string
	downloadedBytes int64
	totalBytes      int64
}

type downloadRun struct {
	destinationPath string
	partialPath     string
	sleeps          []time.Duration
	progress        []progressCall
}

func newDownloadRun(t *testing.T) *downloadRun {
	t.Helper()
	destinationPath := filepath.Join(t.TempDir(), "cache", artifactName)
	return &downloadRun{
		destinationPath: destinationPath,
		partialPath:     destinationPath + ".partial",
		sleeps:          nil,
		progress:        nil,
	}
}

func (run *downloadRun) request(fixture *artifactServer, expectedSHA256 string) modeldownload.Request {
	return modeldownload.Request{
		HTTPClient:      fixture.server.Client(),
		URL:             fixture.server.URL + "/" + artifactName,
		SHA256:          expectedSHA256,
		DestinationPath: run.destinationPath,
		Progress: func(artifact string, downloadedBytes int64, totalBytes int64) {
			run.progress = append(run.progress, progressCall{
				artifact:        artifact,
				downloadedBytes: downloadedBytes,
				totalBytes:      totalBytes,
			})
		},
		Sleep: func(_ context.Context, delay time.Duration) {
			run.sleeps = append(run.sleeps, delay)
		},
	}
}

func assertInstalled(t *testing.T, run *downloadRun, fixture *artifactServer) {
	t.Helper()
	installed, err := os.ReadFile(run.destinationPath)
	if err != nil {
		t.Fatalf("ReadFile %s: %v", run.destinationPath, err)
	}
	if !bytes.Equal(installed, fixture.content) {
		t.Fatalf("installed bytes = %d, want the served %d bytes", len(installed), len(fixture.content))
	}
	assertAbsent(t, run.partialPath)
}

func assertAbsent(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("Stat %s err = %v, want os.ErrNotExist", path, err)
	}
}

func assertFinalProgress(t *testing.T, run *downloadRun) {
	t.Helper()
	if len(run.progress) == 0 {
		t.Fatal("progress calls = 0, want at least 1")
	}
	want := progressCall{
		artifact:        artifactName,
		downloadedBytes: artifactSizeBytes,
		totalBytes:      artifactSizeBytes,
	}
	final := run.progress[len(run.progress)-1]
	if final != want {
		t.Fatalf("final progress = %+v, want %+v", final, want)
	}
}

func readPartial(t *testing.T, run *downloadRun, fixture *artifactServer) []byte {
	t.Helper()
	partial, err := os.ReadFile(run.partialPath)
	if err != nil {
		t.Fatalf("ReadFile %s: %v", run.partialPath, err)
	}
	if len(partial) == 0 || len(partial) >= len(fixture.content) {
		t.Fatalf("partial bytes = %d, want between 1 and %d", len(partial), len(fixture.content)-1)
	}
	if !bytes.HasPrefix(fixture.content, partial) {
		t.Fatal("partial file is not a prefix of the served artifact")
	}
	return partial
}

func leavePartial(t *testing.T, run *downloadRun, fixture *artifactServer) []byte {
	t.Helper()
	err := modeldownload.Ensure(context.Background(), run.request(fixture, fixture.sha256()))
	if !errors.Is(err, modeldownload.ErrUnavailable) {
		t.Fatalf("Ensure err = %v, want ErrUnavailable", err)
	}
	return readPartial(t, run, fixture)
}

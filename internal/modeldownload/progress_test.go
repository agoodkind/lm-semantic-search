package modeldownload_test

import (
	"context"
	"slices"
	"testing"

	"goodkind.io/lm-semantic-search/internal/modeldownload"
)

func TestEnsureDownloadsWholeArtifactAndReportsProgress(t *testing.T) {
	fixture := newArtifactServer(t)
	run := newDownloadRun(t)

	if err := modeldownload.Ensure(context.Background(), run.request(fixture, fixture.sha256())); err != nil {
		t.Fatalf("Ensure: %v", err)
	}

	assertInstalled(t, run, fixture)
	assertFinalProgress(t, run)
	for _, call := range run.progress {
		if call.artifact != artifactName || call.totalBytes != artifactSizeBytes {
			t.Fatalf("progress = %+v, want artifact %s and total %d", call, artifactName, artifactSizeBytes)
		}
	}
	if ranges := fixture.ranges(); !slices.Equal(ranges, []string{""}) {
		t.Fatalf("Range headers = %q, want one request without Range", ranges)
	}
}

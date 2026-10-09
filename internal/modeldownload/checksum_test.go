package modeldownload_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"testing"

	"goodkind.io/lm-semantic-search/internal/modeldownload"
)

func TestEnsureRemovesPartialFileOnChecksumMismatch(t *testing.T) {
	fixture := newArtifactServer(t)
	run := newDownloadRun(t)
	otherDigest := sha256.Sum256([]byte("other artifact"))

	err := modeldownload.Ensure(
		context.Background(),
		run.request(fixture, hex.EncodeToString(otherDigest[:])),
	)
	if !errors.Is(err, modeldownload.ErrUnavailable) {
		t.Fatalf("Ensure err = %v, want ErrUnavailable", err)
	}
	assertAbsent(t, run.partialPath)
	assertAbsent(t, run.destinationPath)
	if requestCount := len(fixture.ranges()); requestCount != 1 {
		t.Fatalf("requests = %d, want 1", requestCount)
	}
}

func TestEnsureSendsNoRequestForInstalledArtifact(t *testing.T) {
	fixture := newArtifactServer(t)
	run := newDownloadRun(t)
	if err := modeldownload.Ensure(context.Background(), run.request(fixture, fixture.sha256())); err != nil {
		t.Fatalf("Ensure: %v", err)
	}
	requestsAfterInstall := len(fixture.ranges())

	if err := modeldownload.Ensure(context.Background(), run.request(fixture, fixture.sha256())); err != nil {
		t.Fatalf("Ensure on installed artifact: %v", err)
	}

	if requestCount := len(fixture.ranges()); requestCount != requestsAfterInstall {
		t.Fatalf("requests = %d, want %d", requestCount, requestsAfterInstall)
	}
	assertInstalled(t, run, fixture)
}

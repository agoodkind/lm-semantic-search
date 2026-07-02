package daemon

import (
	"context"
	"log/slog"
	"os"
	"sync"
	"testing"

	"goodkind.io/lm-semantic-search/internal/merkle"
)

func TestResolveSeedLiveUsesLiveMerkle(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 1)
	canonical := newMultiFileRepo(t, "live.go")
	config := defaultIndexConfig()
	config.IgnoreDigest = "sha256:resolve-seed-live"
	codebaseID, job := seedBootstrapCodebase(t, manager, canonical, config)

	liveSeed := merkle.Snapshot{
		ConfigDigest: config.IgnoreDigest,
		Files:        map[string]string{"live.go": hashText("live")},
		Inodes:       nil,
	}
	if err := merkle.WriteSnapshot(manager.merklePath(codebaseID), liveSeed); err != nil {
		t.Fatalf("WriteSnapshot live returned error: %v", err)
	}
	stagingSeed := merkle.Snapshot{
		ConfigDigest: config.IgnoreDigest,
		Files:        map[string]string{"staging.go": hashText("staging")},
		Inodes:       nil,
	}
	if err := merkle.WriteSnapshot(manager.stagingMerklePath(codebaseID), stagingSeed); err != nil {
		t.Fatalf("WriteSnapshot staging returned error: %v", err)
	}

	decision := manager.resolveSeed(context.Background(), job, codebaseID, false, false)

	if decision.snapshotPath != manager.merklePath(codebaseID) {
		t.Fatalf("snapshotPath = %q, want %q", decision.snapshotPath, manager.merklePath(codebaseID))
	}
	if decision.resumed {
		t.Fatal("resumed = true, want false for live seed")
	}
	if decision.reason != "" {
		t.Fatalf("reason = %q, want empty for live seed", decision.reason)
	}
	if !merkle.Equal(decision.seed, liveSeed) {
		t.Fatalf("seed = %+v, want live seed %+v", decision.seed, liveSeed)
	}
}

func TestResolveSeedStagingResumeStampsBootstrapReason(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 1)
	manager.semantic = &fakeSemantic{
		hasStaging: func(context.Context, string) (bool, error) { return true, nil },
	}
	canonical := newMultiFileRepo(t, "resume.go")
	config := defaultIndexConfig()
	config.IgnoreDigest = "sha256:resolve-seed-staging-resume"
	codebaseID, job := seedBootstrapCodebase(t, manager, canonical, config)

	seed := merkle.Snapshot{
		ConfigDigest: config.IgnoreDigest,
		Files:        map[string]string{"resume.go": hashText("resume")},
		Inodes:       nil,
	}
	if err := merkle.WriteSnapshot(manager.stagingMerklePath(codebaseID), seed); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	decision := manager.resolveSeed(context.Background(), job, codebaseID, true, true)

	if !decision.resumed {
		t.Fatal("resumed = false, want true")
	}
	if decision.reason != string(bootstrapReasonStagingResume) {
		t.Fatalf("reason = %q, want %q", decision.reason, bootstrapReasonStagingResume)
	}
	if !merkle.Equal(decision.seed, seed) {
		t.Fatalf("seed = %+v, want staging seed %+v", decision.seed, seed)
	}
	updated, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("GetJob(%s) not found", job.ID)
	}
	if updated.Progress.BootstrapReason != string(bootstrapReasonStagingResume) {
		t.Fatalf(
			"BootstrapReason = %q, want %q",
			updated.Progress.BootstrapReason,
			bootstrapReasonStagingResume,
		)
	}
}

func TestResolveSeedStagingDiscardedNoCollectionClearsCheckpoint(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 1)
	fake := &fakeSemantic{
		hasStaging: func(context.Context, string) (bool, error) { return false, nil },
	}
	manager.semantic = fake
	canonical := newMultiFileRepo(t, "discard.go")
	config := defaultIndexConfig()
	config.IgnoreDigest = "sha256:resolve-seed-staging-discard"
	codebaseID, job := seedBootstrapCodebase(t, manager, canonical, config)

	seed := merkle.Snapshot{
		ConfigDigest: config.IgnoreDigest,
		Files:        map[string]string{"discard.go": hashText("discard")},
		Inodes:       nil,
	}
	stagingPath := manager.stagingMerklePath(codebaseID)
	if err := merkle.WriteSnapshot(stagingPath, seed); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	decision := manager.resolveSeed(context.Background(), job, codebaseID, true, true)

	if decision.resumed {
		t.Fatal("resumed = true, want false")
	}
	if decision.reason != "staging_discarded_no_collection" {
		t.Fatalf("reason = %q, want staging_discarded_no_collection", decision.reason)
	}
	if len(decision.seed.Files) != 0 {
		t.Fatalf("seed files = %d, want 0", len(decision.seed.Files))
	}
	if _, err := os.Stat(stagingPath); !os.IsNotExist(err) {
		t.Fatalf("staging checkpoint still exists: %v", err)
	}
	if dropped := fake.droppedStagingSnapshot(); len(dropped) != 1 || dropped[0] != canonical {
		t.Fatalf("DropStaging calls = %v, want [%s]", dropped, canonical)
	}
}

func TestResolveSeedStagingResumeLogsSeedReason(t *testing.T) {
	handler := &seedLogHandler{}
	previousLogger := slog.Default()
	slog.SetDefault(slog.New(handler))
	t.Cleanup(func() { slog.SetDefault(previousLogger) })

	manager, _ := newTestManagerWithCap(t, 1)
	manager.semantic = &fakeSemantic{
		hasStaging: func(context.Context, string) (bool, error) { return true, nil },
	}
	canonical := newMultiFileRepo(t, "logged.go")
	config := defaultIndexConfig()
	config.IgnoreDigest = "sha256:resolve-seed-staging-log"
	codebaseID, job := seedBootstrapCodebase(t, manager, canonical, config)
	seed := merkle.Snapshot{
		ConfigDigest: config.IgnoreDigest,
		Files:        map[string]string{"logged.go": hashText("logged")},
		Inodes:       nil,
	}
	if err := merkle.WriteSnapshot(manager.stagingMerklePath(codebaseID), seed); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	manager.resolveSeed(context.Background(), job, codebaseID, true, true)

	reason, found := handler.reasonForMessage("bootstrap.seed")
	if !found {
		t.Fatal("bootstrap.seed log record not found")
	}
	if reason != string(bootstrapReasonStagingResume) {
		t.Fatalf("bootstrap.seed reason = %q, want %q", reason, bootstrapReasonStagingResume)
	}
}

type seedLogHandler struct {
	mu      sync.Mutex
	records []slog.Record
}

func (handler *seedLogHandler) Enabled(context.Context, slog.Level) bool {
	return true
}

func (handler *seedLogHandler) Handle(_ context.Context, record slog.Record) error {
	handler.mu.Lock()
	defer handler.mu.Unlock()
	handler.records = append(handler.records, record.Clone())
	return nil
}

func (handler *seedLogHandler) WithAttrs([]slog.Attr) slog.Handler {
	return handler
}

func (handler *seedLogHandler) WithGroup(string) slog.Handler {
	return handler
}

func (handler *seedLogHandler) reasonForMessage(message string) (string, bool) {
	handler.mu.Lock()
	defer handler.mu.Unlock()

	for _, record := range handler.records {
		if record.Message != message {
			continue
		}
		reason := ""
		record.Attrs(func(attr slog.Attr) bool {
			if attr.Key == "reason" {
				reason = attr.Value.String()
				return false
			}
			return true
		})
		if reason != "" {
			return reason, true
		}
	}
	return "", false
}

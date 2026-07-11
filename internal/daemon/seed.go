package daemon

import (
	"context"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/store"
)

const (
	seedReasonFreshNoCheckpoint            = "fresh_no_checkpoint"
	seedReasonStagingDiscardedNoCollection = "staging_discarded_no_collection"
	seedReasonStagingCheckFailed           = "staging_check_failed"
)

type seedDecision struct {
	seed         merkle.Snapshot
	snapshotPath string
	resumed      bool
	reason       string
}

// resolveSeed owns this invariant: A build's seed snapshot may only cause
// per-file skips for rows provably present in the collection this build writes to.
func (manager *Manager) resolveSeed(ctx context.Context, job model.Job, codebaseID string, staging bool, semanticReady bool) seedDecision {
	configDigest := job.Config.IgnoreDigest
	legacyDigest := manager.legacyDigestForCodebase(codebaseID)
	if !staging {
		snapshotPath := manager.merklePath(codebaseID)
		return seedDecision{
			seed:         merkle.LoadSnapshotForConfig(snapshotPath, configDigest, legacyDigest),
			snapshotPath: snapshotPath,
			resumed:      false,
			reason:       "",
		}
	}

	snapshotPath := manager.stagingMerklePath(codebaseID)
	seed := merkle.LoadSnapshotForConfig(snapshotPath, configDigest, legacyDigest)
	resumed, reason := manager.canResumeStaging(ctx, job.CanonicalPath, seed, semanticReady)
	if resumed {
		decision := seedDecision{
			seed:         seed,
			snapshotPath: snapshotPath,
			resumed:      true,
			reason:       reason,
		}
		manager.logBootstrapSeed(ctx, job, codebaseID, decision)
		manager.routeToBootstrap(ctx, job.ID, bootstrapReasonStagingResume)
		return decision
	}

	if semanticReady {
		if dropErr := manager.semantic.DropStaging(ctx, job.CanonicalPath); dropErr != nil {
			slog.WarnContext(ctx, "drop stale staging before bootstrap failed", "path", job.CanonicalPath, "err", dropErr)
		}
	}
	seed = merkle.Snapshot{ConfigDigest: configDigest, Files: nil, Inodes: nil}
	if removeErr := store.RemoveFile(snapshotPath); removeErr != nil {
		slog.WarnContext(ctx, "remove stale bootstrap checkpoint failed", "path", snapshotPath, "err", removeErr)
	}
	removeConversationDerivedMarkers(ctx, snapshotPath, "remove stale bootstrap derived markers failed")
	decision := seedDecision{
		seed:         seed,
		snapshotPath: snapshotPath,
		resumed:      false,
		reason:       reason,
	}
	manager.logBootstrapSeed(ctx, job, codebaseID, decision)
	return decision
}

// canResumeStaging reports whether a persisted checkpoint can seed a resumed
// build. A checkpoint with no files cannot. When the semantic backend is
// configured the staging collection must still exist, because that is where
// the embedded vectors for the checkpointed files live; without it the
// checkpoint describes work whose vectors were lost, so the build restarts.
// When the backend is unavailable the checkpoint is the only state and is
// trusted on its own.
func (manager *Manager) canResumeStaging(ctx context.Context, canonicalPath string, seed merkle.Snapshot, semanticReady bool) (bool, string) {
	if len(seed.Files) == 0 {
		return false, seedReasonFreshNoCheckpoint
	}
	if !semanticReady {
		return true, string(bootstrapReasonStagingResume)
	}
	hasStaging, err := manager.semantic.HasStaging(ctx, canonicalPath)
	if err != nil {
		slog.WarnContext(ctx, "check staging for resume failed; restarting build", "path", canonicalPath, "err", err)
		return false, seedReasonStagingCheckFailed
	}
	if !hasStaging {
		return false, seedReasonStagingDiscardedNoCollection
	}
	return true, string(bootstrapReasonStagingResume)
}

func (manager *Manager) logBootstrapSeed(ctx context.Context, job model.Job, codebaseID string, decision seedDecision) {
	slog.InfoContext(
		ctx,
		"bootstrap.seed",
		"reason",
		decision.reason,
		"job_id",
		job.ID,
		"codebase_id",
		codebaseID,
		"snapshot_path",
		decision.snapshotPath,
		"files",
		len(decision.seed.Files),
	)
}

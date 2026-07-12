package daemon

import (
	"context"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/gitworktree"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/metrics"
	"goodkind.io/lm-semantic-search/internal/model"
)

type resumeCheckpointKind int

const (
	resumeCheckpointNone resumeCheckpointKind = iota
	resumeCheckpointLive
	resumeCheckpointStaging
)

type resumePlan struct {
	canonicalPath string
	config        model.IndexConfig
	codebaseID    string
	checkpoint    resumeCheckpointKind
}

// ResumeOrphanedJobs re-queues indexing for every codebase whose previous job
// was still running when the daemon exited, but only when a merkle checkpoint
// records the work already done. Delta, streaming, and bootstrap builds all
// checkpoint per file, so an interrupted run can skip files already embedded.
// A bootstrap resume also requires its staging collection to still exist; when
// the checkpoint is missing, the daemon leaves the codebase re-queueable so the
// background repair pass restarts the build, rather than parking it as failed.
// Call this once after NewManager returns and before the daemon advertises ready.
func (manager *Manager) ResumeOrphanedJobs(ctx context.Context) {
	manager.mu.Lock()
	plans := make([]resumePlan, 0)
	for _, codebase := range manager.codebases {
		if codebase.Kind == model.CodebaseKindDocument {
			// A conversation codebase recovers through its own ingest trigger;
			// its chat:// path is not a directory the index runner can walk.
			continue
		}
		if codebase.Status != model.CodebaseStatusIndexing {
			continue
		}
		plans = append(plans, resumePlan{
			canonicalPath: codebase.CanonicalPath,
			config:        codebase.EffectiveConfig,
			codebaseID:    codebase.ID,
			checkpoint:    resumeCheckpointNone,
		})
	}
	manager.mu.Unlock()

	if !manager.config.ResumeIndexingOnBoot {
		for _, plan := range plans {
			manager.logResumeSkipped(ctx, plan.codebaseID, plan.canonicalPath)
		}
		return
	}

	resumable := make([]resumePlan, 0, len(plans))
	for _, plan := range plans {
		plan.checkpoint = manager.resumableCheckpointKind(plan.codebaseID, plan.config.IgnoreDigest)
		if plan.checkpoint != resumeCheckpointNone {
			resumable = append(resumable, plan)
			continue
		}
		manager.logResumeUnresumable(ctx, plan.codebaseID, plan.canonicalPath)
		manager.parkUnresumableForRetry(ctx, plan.codebaseID)
	}
	if len(resumable) == 0 {
		return
	}

	paths := make([]string, 0, len(resumable))
	for _, plan := range resumable {
		paths = append(paths, plan.canonicalPath)
	}
	slog.InfoContext(ctx, "resuming orphaned indexing jobs", "count", len(resumable), "paths", paths)
	for _, plan := range resumable {
		client := model.ClientInfo{Name: "daemon-resume", PID: 0}
		var err error
		if plan.checkpoint == resumeCheckpointStaging {
			err = manager.startStagingResume(ctx, plan, client)
		} else {
			_, _, _, _, err = manager.StartIndex(ctx, plan.canonicalPath, client, plan.config, false, emptyAdmissionBudget)
		}
		if err != nil {
			slog.ErrorContext(ctx, "resume orphaned job failed", "codebase_id", plan.codebaseID, "path", plan.canonicalPath, "err", err)
			continue
		}
		metrics.JobResumed()
		manager.logResumeLaunched(ctx, plan.codebaseID, plan.canonicalPath)
	}
}

// resumableCheckpointKind reports which merkle checkpoint a codebase left
// mid-index persisted for its current config: the live snapshot, the staging
// bootstrap snapshot, or none. Resuming without one would re-embed every file
// from scratch, so the daemon treats a missing checkpoint as not resumable.
func (manager *Manager) resumableCheckpointKind(codebaseID string, configDigest string) resumeCheckpointKind {
	legacyDigest := manager.legacyDigestForCodebase(codebaseID)
	stagingSnapshot := merkle.LoadSnapshotForConfig(manager.stagingMerklePath(codebaseID), configDigest, legacyDigest)
	if len(stagingSnapshot.Files) > 0 {
		return resumeCheckpointStaging
	}
	liveSnapshot := merkle.LoadSnapshotForConfig(manager.merklePath(codebaseID), configDigest, legacyDigest)
	if len(liveSnapshot.Files) > 0 {
		return resumeCheckpointLive
	}
	return resumeCheckpointNone
}

func (manager *Manager) startStagingResume(ctx context.Context, plan resumePlan, client model.ClientInfo) error {
	indexConfig := manager.enrichIndexConfig(plan.config)
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)

	manager.mu.Lock()
	codebase, found := manager.codebases[plan.codebaseID]
	if !found {
		manager.mu.Unlock()
		return nil
	}
	_, resolution, err := manager.activeJobLocked(codebase, indexConfig)
	if err != nil {
		manager.mu.Unlock()
		return err
	}
	// A staging resume is a startup best-effort: if any job is already active for
	// this codebase (a matching-config dedup or a non-matching coalesce), skip the
	// resume rather than fight the in-flight run.
	if resolution != activeJobNone {
		manager.mu.Unlock()
		return nil
	}

	codebase.Status = model.CodebaseStatusPending
	if codebase.LastSuccessfulRun != nil {
		codebase.Status = model.CodebaseStatusIndexing
	}
	codebase.EffectiveConfig = indexConfig
	if manager.semantic != nil && manager.semantic.Available() {
		codebase.CollectionName = manager.semantic.CollectionName(plan.canonicalPath)
	}
	if info, ok := gitworktree.Resolve(plan.canonicalPath); ok && info.Linked {
		codebase.WorktreeCommonDir = info.CommonDir
	}
	codebase.UpdatedAt = clock.Now()

	job := newQueuedJob(codebase.ID, plan.canonicalPath, plan.canonicalPath, client, string(jobOperationIndex), false, indexConfig, emptyAdmissionBudget, clock.Now())
	codebase.ActiveJobID = job.ID
	manager.codebases[codebase.ID] = codebase
	if err := manager.saveLocked(); err != nil {
		manager.mu.Unlock()
		return err
	}
	if err := manager.appendJobLocked("resume_staging_index", job); err != nil {
		manager.mu.Unlock()
		return err
	}
	manager.observer.Invalidate(codebase.ID)
	manager.mu.Unlock()

	manager.runJobAsync(ctx, job.ID)
	return nil
}

// parkUnresumableForRetry leaves an interrupted codebase that has no checkpoint
// at "indexing" with its active job cleared, so the background repair pass
// re-queues a fresh build rather than parking it as a failure. A per-file build
// checkpoints after each file, so a missing checkpoint means almost nothing was
// embedded and the re-queued build restarts cleanly. Clearing the index is the
// only way to stop the retry.
func (manager *Manager) parkUnresumableForRetry(ctx context.Context, codebaseID string) {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	codebase, found := manager.codebases[codebaseID]
	if !found {
		return
	}
	if codebase.Status != model.CodebaseStatusIndexing {
		return
	}

	codebase.ActiveJobID = ""
	codebase.UpdatedAt = clock.Now()
	manager.codebases[codebaseID] = codebase
	if err := manager.saveLocked(); err != nil {
		slog.ErrorContext(ctx, "write registry after parking unresumable index failed", "codebase_id", codebaseID, "err", err)
	}
}

// logResumeSkipped records that boot resume is disabled for one tracked
// codebase. It exists as a method so the per-codebase line is not emitted
// lexically inside the ResumeOrphanedJobs loop.
func (manager *Manager) logResumeSkipped(ctx context.Context, codebaseID string, path string) {
	slog.InfoContext(ctx, "skipping orphaned indexing job resume", "codebase_id", codebaseID, "path", path, "reason", "resume_on_boot_disabled")
}

// logResumeUnresumable records that an interrupted index had no checkpoint to
// resume from, so the daemon leaves it tracked rather than re-embedding the
// whole codebase on boot. Re-run index_codebase to finish it.
func (manager *Manager) logResumeUnresumable(ctx context.Context, codebaseID string, path string) {
	slog.InfoContext(ctx, "skipping unresumable interrupted index; re-run index_codebase to finish", "codebase_id", codebaseID, "path", path, "reason", "no_checkpoint")
}

// logResumeLaunched records that boot resume re-queued one codebase. It
// exists as a method so the per-codebase line is not emitted lexically
// inside the ResumeOrphanedJobs loop.
func (manager *Manager) logResumeLaunched(ctx context.Context, codebaseID string, path string) {
	slog.InfoContext(ctx, "resumed orphaned indexing job", "codebase_id", codebaseID, "path", path)
}

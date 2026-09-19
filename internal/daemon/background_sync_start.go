package daemon

import (
	"context"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/model"
)

// startSweepSync starts the periodic sweep's sync of an indexed codebase whose
// files changed. It starts nothing while the codebase is a worktree that waits
// for a sibling's first build, which only happens when its last run indexed no
// file. The change stays on disk, so the next sweep sees it again.
func (syncer *BackgroundSync) startSweepSync(ctx context.Context, codebase model.Codebase) {
	if syncer.manager.waitsForSiblingFirstBuild(codebase.CanonicalPath) {
		slog.InfoContext(ctx, "sweep sync held for sibling first build", "codebase_id", codebase.ID, "path", codebase.CanonicalPath)
		return
	}
	_, _, _, err := syncer.manager.SyncIndex(
		ctx,
		codebase.CanonicalPath,
		model.ClientInfo{Name: "daemon-sync", PID: 0},
	)
	if err != nil && !syncConflictError(err) {
		slog.ErrorContext(ctx, "start sync job failed", "path", codebase.CanonicalPath, "err", err)
	}
}

// startBuildAfterEmptyRun starts a sync in place of a watcher converge for a
// codebase whose last completed run indexed no file, and reports whether it
// handled the batch. That run created no collection, so a per-path converge
// would drop every path as collection_missing. A sync routes the missing
// collection to a full build of the whole tree. The caller checks for an active
// job first, so a batch that arrives while that build runs is requeued rather
// than folded into a build whose walk may already have passed its paths. A
// worktree that waits for a sibling's first build requeues the batch the same
// way, so the build starts once that first build has content to reuse.
func (syncer *BackgroundSync) startBuildAfterEmptyRun(ctx context.Context, codebase model.Codebase, relativePaths []string) bool {
	if !ranWithoutCreatingACollection(codebase.LastSuccessfulRun) {
		return false
	}
	if syncer.manager.waitsForSiblingFirstBuild(codebase.CanonicalPath) {
		syncer.requeuePaths(codebase.ID, relativePaths)
		return true
	}
	job, _, deduplicated, err := syncer.manager.SyncIndex(ctx, codebase.CanonicalPath, model.ClientInfo{Name: "daemon-watcher", PID: 0})
	if err != nil {
		if !syncConflictError(err) {
			slog.ErrorContext(ctx, "start build after empty run failed", "codebase_id", codebase.ID, "path", codebase.CanonicalPath, "err", err)
		}
		return true
	}
	slog.InfoContext(ctx, "watcher started build after empty run", "codebase_id", codebase.ID, "job_id", job.ID, "deduplicated", deduplicated)
	return true
}

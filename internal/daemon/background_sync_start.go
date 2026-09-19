package daemon

import (
	"context"

	"goodkind.io/lm-semantic-search/internal/model"
)

// startSweepSync starts the periodic sweep's sync of an indexed codebase whose
// files changed. A worktree that waits for a sibling's first build is held
// instead, which only happens when its last run indexed no file; the release
// syncs it when that first build ends.
func (syncer *BackgroundSync) startSweepSync(ctx context.Context, codebase model.Codebase) {
	syncer.manager.startAutomaticSync(ctx, codebase, model.ClientInfo{Name: "daemon-sync", PID: 0})
}

// startBuildAfterEmptyRun starts a sync in place of a watcher converge for a
// codebase whose last completed run indexed no file, and reports whether it
// handled the batch. That run created no collection, so a per-path converge
// would drop every path as collection_missing. A sync routes the missing
// collection to a full build of the whole tree. The caller checks for an active
// job first, so a batch that arrives while that build runs is requeued rather
// than folded into a build whose walk may already have passed its paths. A
// worktree that waits for a sibling's first build is held rather than
// requeued: the sync the release starts walks the whole tree, so it covers
// this batch's paths without the watcher retrying them on every debounce.
func (syncer *BackgroundSync) startBuildAfterEmptyRun(ctx context.Context, codebase model.Codebase) bool {
	if !ranWithoutCreatingACollection(codebase.LastSuccessfulRun) {
		return false
	}
	syncer.manager.startAutomaticSync(ctx, codebase, model.ClientInfo{Name: "daemon-watcher", PID: 0})
	return true
}

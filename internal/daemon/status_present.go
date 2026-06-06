package daemon

import "goodkind.io/lm-semantic-search/internal/model"

// displayStatus is the user-facing status a codebase presents. It is derived,
// not persisted: the registry keeps the lifecycle model.CodebaseStatus, and this
// adds the one presentation fold (live job phase) on top. "preparing" exists
// only here because the job phase carries it.
type displayStatus string

const (
	displayPreparing displayStatus = "preparing"
	displayIndexing  displayStatus = "indexing"
	displayIndexed   displayStatus = "indexed"
	displayStale     displayStatus = "stale"
	displayFailed    displayStatus = "failed"
)

// computeDisplayStatus is the single source of truth for the status every
// surface shows (list, detail, MCP, CLI). It folds the live job into the
// persisted status and never returns "not_indexed": a tracked codebase is
// always preparing, indexing, indexed, stale, or failed.
//
// An interrupted build (persisted "indexing" or "not_indexed" with no live job)
// reads as "preparing" because the background pass re-queues it; it is never a
// phantom "indexing" with nothing running.
func computeDisplayStatus(codebase model.Codebase, activeJob *model.Job) displayStatus {
	// activeJob is the live job for this codebase or nil; the lifecycle clears
	// ActiveJobID on every terminal transition, so a non-nil job here is always
	// in flight. That is what turns a cancelled or transiently-failed build
	// (job cleared, status left at "indexing") into "preparing" rather than a
	// phantom "indexing" with nothing running.
	if activeJob != nil {
		if isBackgroundSyncReconcile(&codebase, activeJob) {
			return displayIndexed
		}
		if jobScopeKnown(activeJob.Progress) {
			return displayIndexing
		}
		return displayPreparing
	}
	switch codebase.Status {
	case model.CodebaseStatusIndexed:
		return displayIndexed
	case model.CodebaseStatusStale:
		return displayStale
	case model.CodebaseStatusFailed:
		return displayFailed
	case model.CodebaseStatusIndexing, model.CodebaseStatusNotIndexed:
		return displayPreparing
	default:
		return displayPreparing
	}
}

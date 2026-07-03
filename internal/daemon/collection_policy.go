package daemon

import "goodkind.io/lm-semantic-search/internal/model"

// Collection presence drives rebuild behavior only. Collection loss must never
// implicitly prune a tracked registration; only explicit clear or another
// intentional registration-deletion flow may remove daemon-owned state.
type collectionPresence int

const (
	collectionPresenceUnknown collectionPresence = iota
	collectionPresencePresent
	collectionPresenceMissing
)

type startIndexMode int

const (
	startIndexModeAlreadyIndexed startIndexMode = iota
	startIndexModeIncremental
	startIndexModeBootstrap
)

type emptyDiffMode int

const (
	emptyDiffModeCompleteNoop emptyDiffMode = iota
	emptyDiffModeFallbackBootstrap
)

type searchCollectionMode int

const (
	searchCollectionModeProceed searchCollectionMode = iota
	searchCollectionModeAutomaticRepair
	searchCollectionModeMissing
)

func decideStartIndexMode(codebaseFound bool, status model.CodebaseStatus, configMatches bool, force bool, presence collectionPresence) startIndexMode {
	if !codebaseFound {
		if presence == collectionPresencePresent {
			return startIndexModeIncremental
		}
		return startIndexModeBootstrap
	}

	switch status {
	case model.CodebaseStatusFailed, model.CodebaseStatusStale, model.CodebaseStatusIndexing,
		model.CodebaseStatusMissing, model.CodebaseStatusQuarantined:
		if presence == collectionPresenceMissing {
			return startIndexModeBootstrap
		}
		return startIndexModeIncremental
	case model.CodebaseStatusNotIndexed, model.CodebaseStatusPending, model.CodebaseStatusDiscovered:
		return startIndexModeBootstrap
	case model.CodebaseStatusIndexed:
		if !force && configMatches && presence != collectionPresenceMissing {
			return startIndexModeAlreadyIndexed
		}
		if presence == collectionPresenceMissing {
			return startIndexModeBootstrap
		}
		return startIndexModeIncremental
	default:
		return startIndexModeBootstrap
	}
}

func decideEmptyDiffMode(evidence collectionEvidence, seedFileCount int) emptyDiffMode {
	if evidence.presence == collectionPresenceMissing {
		return emptyDiffModeFallbackBootstrap
	}
	if evidence.presence == collectionPresencePresent && evidence.rowsKnown && evidence.rows == 0 && seedFileCount > 0 {
		return emptyDiffModeFallbackBootstrap
	}
	return emptyDiffModeCompleteNoop
}

// shouldResumeInterruptedBuild reports whether a codebase represents a build
// that never finished and has no live job, so the background pass should
// re-queue it. A cancelled or transiently-failed from-scratch build is left at
// "indexing" (or "not_indexed") with the active job cleared; re-queuing it is
// the auto-retry that keeps an interrupted build from sitting stuck. A codebase
// whose missing source directory has returned (status missing, dir present
// again) is likewise re-queued to rebuild. A genuine terminal failure (status
// failed) is excluded; it waits for an explicit re-index or clear.
func shouldResumeInterruptedBuild(codebase model.Codebase, hasActiveJob bool) bool {
	if hasActiveJob {
		return false
	}
	switch codebase.Status {
	case model.CodebaseStatusPending, model.CodebaseStatusIndexing, model.CodebaseStatusNotIndexed, model.CodebaseStatusMissing:
		return true
	case model.CodebaseStatusIndexed, model.CodebaseStatusStale, model.CodebaseStatusFailed,
		model.CodebaseStatusDiscovered, model.CodebaseStatusQuarantined:
		// A discovered worktree's build is driven by the deferred timer and the
		// periodic sweep, not the interrupted-build resume path.
		return false
	default:
		return false
	}
}

func shouldDeferWatcherConvergeForFirstBuild(codebase model.Codebase) bool {
	if codebase.Kind == model.CodebaseKindDocument {
		return false
	}
	if codebase.LastSuccessfulRun != nil {
		return false
	}
	return isPreIndexedStatus(codebase.Status)
}

// shouldSkipForActiveFirstBuildStaging reports whether a codebase has an
// active first build whose collection is still staging, so callers should skip
// work that assumes a live collection.
func shouldSkipForActiveFirstBuildStaging(codebase model.Codebase, hasActiveJob bool) bool {
	if !hasActiveJob {
		return false
	}
	return shouldDeferWatcherConvergeForFirstBuild(codebase)
}

func isPreIndexedStatus(status model.CodebaseStatus) bool {
	switch status {
	case model.CodebaseStatusNotIndexed, model.CodebaseStatusPending, model.CodebaseStatusIndexing,
		model.CodebaseStatusDiscovered:
		// Discovered worktrees have no collection yet and the watcher is active
		// for them, so a watcher converge before the first build promotes would
		// hit collection_missing; treat discovered as pre-indexed so it defers.
		return true
	case model.CodebaseStatusIndexed, model.CodebaseStatusStale, model.CodebaseStatusFailed,
		model.CodebaseStatusMissing, model.CodebaseStatusQuarantined:
		return false
	default:
		return false
	}
}

func shouldQueueMissingCollectionRepair(codebase model.Codebase, hasActiveJob bool, presence collectionPresence) bool {
	if hasActiveJob || presence != collectionPresenceMissing {
		return false
	}
	switch codebase.Status {
	case model.CodebaseStatusIndexed, model.CodebaseStatusStale:
		return true
	case model.CodebaseStatusNotIndexed, model.CodebaseStatusPending, model.CodebaseStatusIndexing,
		model.CodebaseStatusFailed, model.CodebaseStatusMissing, model.CodebaseStatusDiscovered,
		model.CodebaseStatusQuarantined:
		return false
	default:
		return false
	}
}

func decideSearchCollectionMode(codebase model.Codebase, activeJob *model.Job, presence collectionPresence) searchCollectionMode {
	if presence != collectionPresenceMissing {
		return searchCollectionModeProceed
	}
	if activeJob != nil || codebase.Status == model.CodebaseStatusStale || codebase.Status == model.CodebaseStatusIndexing {
		return searchCollectionModeAutomaticRepair
	}
	return searchCollectionModeMissing
}

func presenceFromCollectionSet(collectionName string, collectionSet map[string]struct{}) collectionPresence {
	if collectionName == "" {
		return collectionPresenceUnknown
	}
	if _, found := collectionSet[collectionName]; found {
		return collectionPresencePresent
	}
	return collectionPresenceMissing
}

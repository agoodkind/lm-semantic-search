package daemon

import "goodkind.io/lm-semantic-search/internal/model"

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
	case model.CodebaseStatusFailed, model.CodebaseStatusStale, model.CodebaseStatusIndexing:
		if presence == collectionPresenceMissing {
			return startIndexModeBootstrap
		}
		return startIndexModeIncremental
	case model.CodebaseStatusNotIndexed:
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

func decideEmptyDiffMode(presence collectionPresence) emptyDiffMode {
	if presence == collectionPresenceMissing {
		return emptyDiffModeFallbackBootstrap
	}
	return emptyDiffModeCompleteNoop
}

func shouldQueueMissingCollectionRepair(codebase model.Codebase, hasActiveJob bool, presence collectionPresence) bool {
	if hasActiveJob || presence != collectionPresenceMissing {
		return false
	}
	switch codebase.Status {
	case model.CodebaseStatusIndexed, model.CodebaseStatusStale:
		return true
	case model.CodebaseStatusNotIndexed, model.CodebaseStatusIndexing, model.CodebaseStatusFailed:
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

package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/status"
)

// SearchCode performs a semantic search over indexed chunk content.
func (manager *Manager) SearchCode(ctx context.Context, requestedPath string, query string, limit int32, extensionFilter []string) (SearchOutcome, error) {
	normalizedExtensions, err := semantic.ValidateExtensionFilter(extensionFilter)
	if err != nil {
		return SearchOutcome{}, adapterr.NewInvalidPath(err.Error(), err)
	}

	codebase, activeJob, found, _, err := manager.GetIndex(ctx, requestedPath)
	if err != nil {
		return SearchOutcome{}, err
	}
	if !found {
		return SearchOutcome{}, adapterr.NewNotIndexed(requestedPath, nil)
	}

	// A worktree the daemon just discovered on this read has no collection yet,
	// so there is nothing to search and serving the parent's collection would
	// return wrong-branch content. Return the discovered note with no results; the
	// deferred build (already scheduled by GetIndex) makes it searchable shortly.
	if codebase.Status == model.CodebaseStatusDiscovered {
		return SearchOutcome{
			Codebase:  codebase,
			ActiveJob: activeJob,
			Results:   []model.StoredChunk{},
			StateNote: discoveredSearchNote(manager.worktreeReuseForecast(codebase)),
		}, nil
	}
	stateNote := ""
	if codebase.Status == model.CodebaseStatusQuarantined {
		stateNote = quarantinedSearchNote(codebase.Quarantine)
	}

	// When the query targets a nested directory of a larger covering index, scope
	// the search to that subtree so results come only from the requested path,
	// not the whole parent index.
	relativePathPrefix := subtreePrefix(requestedPath, codebase.CanonicalPath)

	if manager.semantic == nil || !manager.semantic.Available() {
		storeErr := adapterr.NewMilvusUnavailable(semantic.ErrUnavailable)
		manager.noteDependencyFailure(storeErr)
		slog.ErrorContext(ctx, "semantic search unavailable", "codebase_path", codebase.CanonicalPath, "err", storeErr)
		return SearchOutcome{}, storeErr
	}
	collectionName := manager.semantic.CollectionName(codebase.CanonicalPath)
	lease, leaseErr := manager.semantic.AcquireCollection(ctx, collectionName)
	if leaseErr != nil {
		manager.noteDependencyFailure(leaseErr)
		return SearchOutcome{}, fmt.Errorf(
			"acquire search collection %s: %w",
			collectionName,
			leaseErr,
		)
	}
	defer lease.Release()

	chunks, semanticErr := manager.semantic.Search(ctx, codebase.CanonicalPath, query, limit, normalizedExtensions, relativePathPrefix)
	switch {
	case semanticErr == nil:
		// A search that returns no error exercised both shared dependencies, which
		// is why it may clear any mode rather than only the store's. Both backends
		// read the collection, embed the query against the live endpoint, then run
		// the ranked query, and each step returns its own error, so reaching here
		// is positive evidence that the store answered and the embedder answered.
		// The result count is not part of that evidence: an indexed codebase with
		// no match for this query returns zero rows from a perfectly healthy
		// pipeline. This mirrors the indexing rule that only a real embed clears
		// the banner.
		manager.noteDependencyHealthy()
		return SearchOutcome{
			Codebase:  codebase,
			ActiveJob: activeJob,
			Results:   semantic.DeduplicateChunks(chunks),
			StateNote: stateNote,
		}, nil
	case (errors.Is(semanticErr, semantic.ErrCollectionNotReady) ||
		errors.Is(semanticErr, semantic.ErrSearchResultIncomplete)) &&
		codebase.Status == model.CodebaseStatusIndexing:
		return SearchOutcome{Codebase: codebase, ActiveJob: activeJob, Results: []model.StoredChunk{}, StateNote: stateNote}, nil
	case errors.Is(semanticErr, semantic.ErrCollectionMissing):
		switch decideSearchCollectionMode(codebase, activeJob, collectionPresenceMissing) {
		case searchCollectionModeAutomaticRepair:
			return SearchOutcome{
				Codebase:  codebase,
				ActiveJob: activeJob,
				Results:   []model.StoredChunk{},
				StateNote: status.StateNoteFor(status.SearchRepairing),
			}, nil
		case searchCollectionModeMissing:
			return SearchOutcome{}, adapterr.NewIndexDataLost(codebase.CanonicalPath, nil)
		case searchCollectionModeProceed:
			return SearchOutcome{}, adapterr.NewIndexDataLost(codebase.CanonicalPath, nil)
		default:
			return SearchOutcome{}, adapterr.NewIndexDataLost(codebase.CanonicalPath, nil)
		}
	case errors.Is(semanticErr, semantic.ErrUnavailable):
		storeErr := adapterr.NewMilvusUnavailable(semanticErr)
		manager.noteDependencyFailure(storeErr)
		slog.ErrorContext(ctx, "semantic search unavailable", "codebase_path", codebase.CanonicalPath, "err", storeErr)
		return SearchOutcome{}, storeErr
	default:
		// Record a shared-infrastructure outage from the search path too, not
		// only from index jobs, so a failed search trips the same banner. The
		// recorder no-ops for any error that is not a real outage.
		manager.noteDependencyFailure(semanticErr)
		slog.ErrorContext(ctx, "semantic search failed", "codebase_path", codebase.CanonicalPath, "err", semanticErr)
		return SearchOutcome{}, fmt.Errorf("semantic search for %s: %w", codebase.CanonicalPath, semanticErr)
	}
}

// discoveredSearchNote is the read-only note a search returns for a worktree the
// daemon just discovered and has not built yet. It names the reuse the deferred
// build will get so the cheapness is visible, and tells the caller to retry.
func discoveredSearchNote(siblingCount int32) string {
	note := "🔎 This worktree was just discovered and is not indexed yet; its build is starting now"
	if siblingCount > 0 {
		note += fmt.Sprintf(" (reuses embeddings from %d indexed sibling %s)", siblingCount, plural("worktree", int(siblingCount)))
	}
	return note + ". Search again shortly."
}

func quarantinedSearchNote(quarantine *model.QuarantineState) string {
	if quarantine == nil {
		return "🔎 Search is serving the last known-good index while destructive sync is paused after a suspicious large disappearance."
	}
	return fmt.Sprintf(
		"🔎 Search is serving the last known-good index while destructive sync is paused after a suspicious large disappearance (%d of %d tracked files in the last %s observation).",
		quarantine.LastMissingCount,
		quarantine.LastTotalCount,
		defaultQuarantineTrigger(quarantine.LastTrigger),
	)
}

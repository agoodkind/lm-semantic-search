package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"sort"
	"strings"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/status"
	"goodkind.io/lm-semantic-search/internal/store"
)

// SearchCode performs a local ranked search over persisted chunk content.
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

	if manager.semantic != nil && manager.semantic.Available() {
		chunks, semanticErr := manager.semantic.Search(ctx, codebase.CanonicalPath, query, limit, normalizedExtensions, relativePathPrefix)
		switch {
		case semanticErr == nil:
			// The query embed succeeded, which proves the embedder is reachable, so
			// clear any degraded banner a prior outage left up. This mirrors the
			// indexing rule that only a real embed clears the banner.
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
			}
		case errors.Is(semanticErr, semantic.ErrUnavailable):
		default:
			// Record a shared-infrastructure outage from the search path too, not
			// only from index jobs, so a failed search trips the same banner. The
			// recorder no-ops for any error that is not a real outage.
			manager.noteDependencyFailure(semanticErr)
			return SearchOutcome{}, fmt.Errorf("semantic search for %s: %w", codebase.CanonicalPath, semanticErr)
		}
	}

	chunks, err := store.ReadChunks(manager.chunkPath(codebase.ID))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) && codebase.Status == model.CodebaseStatusIndexing {
			return SearchOutcome{Codebase: codebase, ActiveJob: activeJob, Results: []model.StoredChunk{}, StateNote: stateNote}, nil
		}
		slog.ErrorContext(ctx, "read chunk cache failed", "codebase_id", codebase.ID, "err", err)
		return SearchOutcome{}, fmt.Errorf("read chunk cache for %s: %w", codebase.ID, err)
	}
	return SearchOutcome{Codebase: codebase, ActiveJob: activeJob, Results: rankChunks(chunks, query, limit, normalizedExtensions, relativePathPrefix), StateNote: stateNote}, nil
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

// chunkUnderPrefix reports whether a chunk's relative path equals scopePrefix
// or descends from it, matching the Milvus prefix filter the semantic path
// applies so the in-memory fallback scopes a nested-directory search the same
// way.
func chunkUnderPrefix(relativePath string, scopePrefix string) bool {
	relativePath = strings.Trim(relativePath, "/")
	return relativePath == scopePrefix || strings.HasPrefix(relativePath, scopePrefix+"/")
}

func rankChunks(chunks []model.StoredChunk, query string, limit int32, extensionFilter []string, relativePathPrefix string) []model.StoredChunk {
	filteredChunks := make([]model.StoredChunk, 0, len(chunks))
	filterSet := map[string]struct{}{}
	for _, extension := range extensionFilter {
		filterSet[extension] = struct{}{}
	}
	scopePrefix := strings.Trim(strings.TrimSpace(relativePathPrefix), "/")

	queryLower := strings.ToLower(query)
	queryTerms := strings.Fields(queryLower)
	type scoredChunk struct {
		chunk model.StoredChunk
		score int
	}
	scored := make([]scoredChunk, 0, len(chunks))
	for _, chunk := range chunks {
		if scopePrefix != "" && !chunkUnderPrefix(chunk.RelativePath, scopePrefix) {
			continue
		}
		if len(filterSet) > 0 {
			if _, found := filterSet[chunk.FileExtension]; !found {
				continue
			}
		}

		contentLower := strings.ToLower(chunk.Content)
		score := 0
		if strings.Contains(contentLower, queryLower) {
			score += 100
		}
		for _, term := range queryTerms {
			if strings.Contains(contentLower, term) {
				score++
			}
		}
		if score == 0 {
			continue
		}
		scored = append(scored, scoredChunk{chunk: chunk, score: score})
	}

	sort.SliceStable(scored, func(i int, j int) bool {
		if scored[i].score == scored[j].score {
			if scored[i].chunk.RelativePath == scored[j].chunk.RelativePath {
				return scored[i].chunk.StartLine < scored[j].chunk.StartLine
			}
			return scored[i].chunk.RelativePath < scored[j].chunk.RelativePath
		}
		return scored[i].score > scored[j].score
	})

	maxResults := int(limit)
	if maxResults <= 0 {
		maxResults = 10
	}
	if maxResults > len(scored) {
		maxResults = len(scored)
	}
	for _, item := range scored[:maxResults] {
		// Carry the keyword rank as the chunk score so literal-fallback results
		// keep an ordering signal alongside semantic results' similarity scores.
		rankedChunk := item.chunk
		rankedChunk.Score = float64(item.score)
		filteredChunks = append(filteredChunks, rankedChunk)
	}
	return filteredChunks
}

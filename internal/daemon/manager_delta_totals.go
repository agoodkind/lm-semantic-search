package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"

	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/store"
)

// codebaseTotals reports the file and chunk totals that represent the
// codebase as a whole at the moment a run completes, so the registry's
// LastSuccessfulRun describes current state rather than the per-run delta.
// fileCount is the size of the working merkle set, which matches the codebase
// under the active config digest. chunkCount comes from semantic.Service.Count,
// a live count(*) of the collection, when the backend is available; on
// unavailability or any error it falls back to fallbackChunks, which the
// caller passes as either the loop's running TotalChunks (incremental
// path) or zero (empty-diff fast path).
func (manager *Manager) codebaseTotals(ctx context.Context, canonicalPath string, working map[string]string, fallbackChunks int32) (int32, int32) {
	fileCount := safeInt32(len(working))
	if manager.semantic == nil || !manager.semantic.Available() {
		return fileCount, fallbackChunks
	}
	count, err := manager.semantic.Count(ctx, canonicalPath)
	if err != nil {
		if !errors.Is(err, semantic.ErrUnavailable) {
			slog.WarnContext(ctx, "semantic count failed; using fallback chunk total", "path", canonicalPath, "err", err)
		}
		return fileCount, fallbackChunks
	}
	return fileCount, count
}

func (manager *Manager) normalizeDeltaTotalBytes(ctx context.Context, codebase model.Codebase, state deltaState, result indexer.Result) (indexer.Result, error) {
	if state.source.tracksByteTotals() {
		normalizedChunks, ok, err := manager.deltaChunkCache(ctx, codebase, state, result.Chunks)
		if err != nil {
			return result, err
		}
		if ok {
			result.Chunks = normalizedChunks
			result.TotalBytes = storedChunkBytes(normalizedChunks)
			return result, nil
		}
	}
	if codebase.LastSuccessfulRun != nil && result.TotalBytes < codebase.LastSuccessfulRun.TotalBytes {
		result.TotalBytes = codebase.LastSuccessfulRun.TotalBytes
	}
	return result, nil
}

func (manager *Manager) deltaChunkCache(ctx context.Context, codebase model.Codebase, state deltaState, changedChunks []model.StoredChunk) ([]model.StoredChunk, bool, error) {
	existingChunks, err := store.ReadChunks(manager.chunkPath(codebase.ID))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			// A delta runs only against a previously-indexed codebase, so a
			// missing chunk cache means it was deleted or never written, not that
			// the codebase is empty. Report the cache as unavailable so the caller
			// carries forward the prior whole-codebase byte total instead of
			// computing one from only the delta chunks and claiming it as the
			// whole-codebase total.
			return nil, false, nil
		}
		slog.ErrorContext(ctx, "read chunk cache for delta byte total failed", "codebase_id", codebase.ID, "path", codebase.CanonicalPath, "err", err)
		return nil, false, fmt.Errorf("read chunk cache for delta byte total: %w", err)
	}

	replacedPaths := replacedDeltaPaths(state, changedChunks)
	normalizedChunks := make([]model.StoredChunk, 0, len(existingChunks)+len(changedChunks))
	for _, chunk := range existingChunks {
		if _, replaced := replacedPaths[chunk.RelativePath]; replaced {
			continue
		}
		normalizedChunks = append(normalizedChunks, chunk)
	}
	normalizedChunks = append(normalizedChunks, changedChunks...)
	return normalizedChunks, true, nil
}

func replacedDeltaPaths(state deltaState, changedChunks []model.StoredChunk) map[string]struct{} {
	replacedPaths := make(map[string]struct{}, len(state.plan.diff.Added)+len(state.plan.diff.Modified)+len(state.plan.diff.Removed))
	for _, relativePath := range state.plan.diff.Removed {
		replacedPaths[relativePath] = struct{}{}
	}
	for _, chunk := range changedChunks {
		replacedPaths[chunk.RelativePath] = struct{}{}
	}
	for _, relativePath := range state.plan.diff.Added {
		if _, present := state.working[relativePath]; !present {
			replacedPaths[relativePath] = struct{}{}
		}
	}
	for _, relativePath := range state.plan.diff.Modified {
		seedHash, seeded := state.plan.seedSnapshot.Files[relativePath]
		workingHash, present := state.working[relativePath]
		if !present {
			replacedPaths[relativePath] = struct{}{}
			continue
		}
		if seeded && workingHash != seedHash {
			replacedPaths[relativePath] = struct{}{}
		}
	}
	return replacedPaths
}

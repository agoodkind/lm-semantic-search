package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"path/filepath"

	"goodkind.io/claude-context-go/internal/model"
)

// GetIndex resolves one tracked codebase whose canonical path covers
// requestedPath. The classification return describes how the daemon sees
// the queried path (covered, excluded, out-of-scope) for fine-grained
// callers; tracked callers can ignore it. GetIndex returns Indexed for any
// path whose Milvus collection exists, synthesizing a record when the
// registry has no entry for it.
func (manager *Manager) GetIndex(ctx context.Context, requestedPath string) (model.Codebase, *model.Job, bool, *model.PathClassification, error) {
	manager.reconcileIndexedCodebases(ctx)

	canonicalPath, err := canonicalizePath(requestedPath)
	if err != nil {
		slog.ErrorContext(ctx, "canonicalize path failed", "path", requestedPath, "err", err)
		return model.Codebase{}, nil, false, nil, fmt.Errorf("canonicalize path %s: %w", requestedPath, err)
	}

	manager.mu.Lock()
	matches := manager.findCodebasesByCoverage(canonicalPath)
	if len(matches) > 0 {
		codebase := matches[0]
		activeJob := manager.activeJobSnapshotLocked(codebase)
		manager.mu.Unlock()
		classification := classifyTrackedPath(codebase, canonicalPath)
		return codebase, activeJob, true, classification, nil
	}
	manager.mu.Unlock()

	if manager.semantic != nil && manager.semantic.Available() {
		hasCollection, hasErr := manager.semantic.HasCollectionForPath(ctx, canonicalPath)
		if hasErr == nil && hasCollection {
			synthetic := manager.synthesizeUnregisteredCodebase(canonicalPath)
			classification := &model.PathClassification{
				Kind:                model.PathClassificationInScopeIndexed,
				ExcludedByPattern:   "",
				ExcludedByGitignore: "",
				CoveringCodebaseID:  synthetic.ID,
			}
			return synthetic, nil, true, classification, nil
		}
		if hasErr != nil {
			slog.WarnContext(ctx, "Milvus HasCollection failed during GetIndex", "path", canonicalPath, "err", hasErr)
		}
	}

	var emptyCodebase model.Codebase
	classification := &model.PathClassification{
		Kind:                model.PathClassificationOutOfScope,
		ExcludedByPattern:   "",
		ExcludedByGitignore: "",
		CoveringCodebaseID:  "",
	}
	return emptyCodebase, nil, false, classification, nil
}

// classifyTrackedPath maps a covered canonical path into a classification.
// When the path equals the codebase root, it is treated as in-scope and
// indexed. Otherwise the resolved ignore rules decide between excluded and
// unindexed; the full discovery walk runs in convergence, so the status
// surface reports unindexed until the converge marks the file as present.
func classifyTrackedPath(codebase model.Codebase, canonicalPath string) *model.PathClassification {
	classification := &model.PathClassification{
		Kind:                model.PathClassificationInScopeIndexed,
		ExcludedByPattern:   "",
		ExcludedByGitignore: "",
		CoveringCodebaseID:  codebase.ID,
	}
	if canonicalPath == codebase.CanonicalPath {
		return classification
	}
	relative, err := filepath.Rel(codebase.CanonicalPath, canonicalPath)
	if err != nil || relative == "" || relative == "." {
		return classification
	}
	classification.Kind = model.PathClassificationInScopeUnindexed
	return classification
}

// synthesizeUnregisteredCodebase builds an in-memory codebase record for a
// path whose Milvus collection exists without a registry entry.
func (manager *Manager) synthesizeUnregisteredCodebase(canonicalPath string) model.Codebase {
	collectionName := ""
	if manager.semantic != nil {
		collectionName = manager.semantic.CollectionName(canonicalPath)
	}
	codebase := newCodebaseRecord(canonicalPath)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.CollectionName = collectionName
	codebase.EffectiveConfig.Hybrid = manager.config.HybridMode
	return codebase
}

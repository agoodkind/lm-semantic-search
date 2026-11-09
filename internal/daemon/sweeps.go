package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
)

func (syncer *BackgroundSync) codebaseChanged(
	ctx context.Context,
	codebase model.Codebase,
) (bool, error) {
	if codebase.Kind == model.CodebaseKindDocument {
		return false, nil
	}

	// The checkpoint read goes through the manager's one live-checkpoint reader,
	// so an empty codebase that never wrote a file is not reported as damaged on
	// every sweep. Its absent checkpoint reads as the empty snapshot it would
	// have written, which matches the empty capture below, so the sweep also
	// stops enqueuing a sync for a codebase where nothing changed. A codebase
	// that did index files and lost its checkpoint still reports the loss, and
	// its empty snapshot differs from a non-empty capture, so the resync it
	// needs still starts.
	existingSnapshot := syncer.manager.loadLiveCheckpoint(
		ctx,
		codebase,
		codebase.EffectiveConfig.IgnoreDigest,
	).snapshot

	currentSnapshot, err := merkle.Capture(
		ctx,
		syncer.manager.indexability,
		codebase.ID,
		codebase.CanonicalPath,
		codebase.EffectiveConfig,
	)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"capture Merkle snapshot failed",
			"path",
			codebase.CanonicalPath,
			"err",
			err,
		)

		return false, fmt.Errorf("capture Merkle snapshot for %s: %w", codebase.CanonicalPath, err)
	}

	if !merkle.Equal(existingSnapshot, currentSnapshot) {
		return true, nil
	}

	presence := syncer.manager.probeCollectionEvidence(
		ctx,
		codebase.CanonicalPath,
		"backgroundSync",
	).presence
	currentSnapshotHash := snapshotHashForGraph(
		currentSnapshot,
		codebase.EffectiveConfig.IgnoreDigest,
	)

	return syncer.manager.shouldReconcileGraph(codebase.ID, currentSnapshotHash, presence), nil
}

func (syncer *BackgroundSync) runPeriodicLoop(ctx context.Context, action func()) {
	initialTimer := time.NewTimer(defaultInitialSyncDelay)
	defer initialTimer.Stop()
	interval := time.Duration(syncer.cfg.SyncIntervalMS) * time.Millisecond
	if syncer.cfg.SyncIntervalMS < minimumSyncIntervalMS {
		interval = time.Duration(minimumSyncIntervalMS) * time.Millisecond
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-initialTimer.C:
			action()
		case <-ticker.C:
			action()
		}
	}
}

func syncConflictError(err error) bool {
	if err == nil {
		return false
	}

	message := err.Error()

	return strings.Contains(message, "conflicting active job") ||
		strings.Contains(message, "codebase not tracked")
}

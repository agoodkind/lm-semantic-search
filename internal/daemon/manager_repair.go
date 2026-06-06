package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"

	"goodkind.io/gklog/correlation"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/model"
)

type missingCollectionRepair struct {
	codebaseID    string
	canonicalPath string
	config        model.IndexConfig
}

// RepairMissingCollections is the daemon-owned anti-entropy pass that
// reconciles the registry to collection reality in both directions. When the
// registry says a codebase is indexed but Milvus no longer has its collection,
// it keeps the codebase tracked, marks it stale, and re-queues a full bootstrap
// rebuild. When the registry says a codebase failed or is stale but its
// collection is present, it heals the codebase back to indexed and clears the
// stale failure, so the registry is the single source of truth every reader
// agrees with. Read paths stay side-effect free; this pass owns the mutation.
func (manager *Manager) RepairMissingCollections(ctx context.Context) {
	plans, err := manager.planMissingCollectionRepairs(ctx)
	if err != nil {
		slog.ErrorContext(ctx, "repair missing collections failed", "err", err)
		return
	}

	queuedPaths := make([]string, 0, len(plans))
	for _, plan := range plans {
		_, _, _, _, err := manager.StartIndex(
			ctx,
			plan.canonicalPath,
			model.ClientInfo{Name: "daemon-repair", PID: 0},
			plan.config,
			false,
		)
		if err != nil {
			manager.noteAutomaticRepairStartFailure(ctx, plan.codebaseID, err)
			slog.WarnContext(
				ctx,
				"automatic rebuild enqueue failed for missing Milvus collection",
				"codebase_id",
				plan.codebaseID,
				"path",
				plan.canonicalPath,
				"err",
				err,
			)
			continue
		}
		queuedPaths = append(queuedPaths, plan.canonicalPath)
	}
	if len(queuedPaths) > 0 {
		slog.InfoContext(
			ctx,
			"queued automatic rebuilds for missing Milvus collections",
			"count",
			len(queuedPaths),
			"paths",
			queuedPaths,
		)
	}
}

func (manager *Manager) planMissingCollectionRepairs(ctx context.Context) ([]missingCollectionRepair, error) {
	if manager.semantic == nil || !manager.semantic.Available() {
		return nil, nil
	}

	collections, err := manager.semantic.ListCollections(ctx)
	if err != nil {
		return nil, fmt.Errorf("list semantic collections: %w", err)
	}

	collectionSet := make(map[string]struct{}, len(collections))
	for _, collectionName := range collections {
		collectionSet[collectionName] = struct{}{}
	}

	manager.mu.Lock()
	defer manager.mu.Unlock()

	plans := make([]missingCollectionRepair, 0)
	changed := false
	for codebaseID, codebase := range manager.codebases {
		switch codebase.Status {
		case model.CodebaseStatusIndexed, model.CodebaseStatusStale, model.CodebaseStatusFailed,
			model.CodebaseStatusIndexing, model.CodebaseStatusNotIndexed:
		default:
			continue
		}
		if _, err := os.Stat(codebase.CanonicalPath); errors.Is(err, os.ErrNotExist) {
			continue
		}

		expectedCollectionName := codebase.CollectionName
		if expectedCollectionName == "" {
			expectedCollectionName = manager.semantic.CollectionName(codebase.CanonicalPath)
			if expectedCollectionName != "" {
				codebase.CollectionName = expectedCollectionName
				manager.codebases[codebaseID] = codebase
				changed = true
			}
		}
		if expectedCollectionName == "" {
			continue
		}
		presence := presenceFromCollectionSet(expectedCollectionName, collectionSet)
		hasActiveJob := manager.activeJobSnapshotLocked(codebase) != nil

		// Reconcile the other direction: a codebase marked failed or stale whose
		// collection is present now is usable, so heal it to indexed and clear the
		// stale failure rather than leaving the registry divergent from reality.
		if presence == collectionPresencePresent && !hasActiveJob &&
			(codebase.Status == model.CodebaseStatusFailed || codebase.Status == model.CodebaseStatusStale) {
			codebase.Status = model.CodebaseStatusIndexed
			codebase.LastFailedRun = nil
			codebase.UpdatedAt = clock.Now()
			manager.codebases[codebaseID] = codebase
			changed = true
			continue
		}

		// An interrupted build (indexing or not_indexed with no live job) never
		// finished, so re-queue it to resume from its checkpoint or restart. This
		// is the auto-retry that makes a "preparing" presentation honest; only
		// clearing the index stops it.
		if shouldResumeInterruptedBuild(codebase, hasActiveJob) {
			plans = append(plans, missingCollectionRepair{
				codebaseID:    codebaseID,
				canonicalPath: codebase.CanonicalPath,
				config:        codebase.EffectiveConfig,
			})
			continue
		}

		if !shouldQueueMissingCollectionRepair(codebase, hasActiveJob, presence) {
			continue
		}

		if codebase.Status != model.CodebaseStatusStale || codebase.ActiveJobID != "" {
			codebase.Status = model.CodebaseStatusStale
			codebase.ActiveJobID = ""
			codebase.UpdatedAt = clock.Now()
			manager.codebases[codebaseID] = codebase
			changed = true
		}
		plans = append(plans, missingCollectionRepair{
			codebaseID:    codebaseID,
			canonicalPath: codebase.CanonicalPath,
			config:        codebase.EffectiveConfig,
		})
	}

	if changed {
		if err := manager.saveLocked(); err != nil {
			slog.ErrorContext(ctx, "persist stale codebases before automatic rebuild failed", "err", err)
			return nil, fmt.Errorf("persist stale codebases before automatic rebuild: %w", err)
		}
	}
	return plans, nil
}

func (manager *Manager) noteAutomaticRepairStartFailure(ctx context.Context, codebaseID string, startErr error) {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	codebase, found := manager.codebases[codebaseID]
	if !found {
		return
	}

	now := clock.Now()
	// Keep the raw enqueue error in the log; the persisted message stays a clean
	// summary so the status surface carries no implementation detail.
	slog.ErrorContext(ctx, "automatic rebuild could not start for missing collection", "codebase_id", codebaseID, "err", startErr)
	codebase.Status = model.CodebaseStatusStale
	codebase.ActiveJobID = ""
	codebase.LastFailedRun = &model.IndexRunFailure{
		Message:                 "semantic collection is missing and automatic rebuild could not start",
		LastAttemptedPercentage: 0,
		FailedAt:                now,
		TraceID:                 string(correlation.FromContext(ctx).TraceID),
		JobID:                   "",
	}
	codebase.UpdatedAt = now
	manager.codebases[codebaseID] = codebase
	if err := manager.saveLocked(); err != nil {
		slog.ErrorContext(ctx, "write registry after automatic rebuild enqueue failure failed", "codebase_id", codebaseID, "err", err)
	}
}

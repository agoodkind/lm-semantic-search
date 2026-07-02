package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"goodkind.io/gklog/correlation"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/metrics"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/store"
)

const (
	defaultInitialSyncDelay  = 5 * time.Second
	defaultTriggerPollPeriod = 1 * time.Second
	defaultTriggerDebounce   = 2 * time.Second
	minimumSyncIntervalMS    = 1000
)

// BackgroundSync owns daemon-driven file-watch, periodic, and trigger-based
// sync. The file watcher is the steady-state driver: it converges changed
// paths within the debounce window. The periodic sweep is the anti-entropy
// backstop that repairs drift from missed events or downtime.
//
// Watcher converges run through the manager's index-slot semaphore, so several
// codebases converge at once up to the cap and a single heavily-edited
// repository never blocks the others. Per-codebase serialization, plus the
// shared advisory lock held for the embed window, keeps two converges of the
// same codebase from racing and keeps the upstream TS adapter coordinated.
type BackgroundSync struct {
	cfg     config.Config
	manager *Manager

	mu           sync.Mutex
	triggerTimer *time.Timer
	lastTrigger  time.Time

	convergeMu sync.Mutex
	converging map[string]struct{}

	deferredWatcherMu    sync.Mutex
	deferredWatcherPaths map[string]map[string]struct{}

	queue   *EventQueue
	watcher *Watcher
}

// NewBackgroundSync constructs the daemon background sync coordinator.
func NewBackgroundSync(cfg config.Config, manager *Manager) *BackgroundSync {
	return &BackgroundSync{
		cfg:                  cfg,
		manager:              manager,
		mu:                   sync.Mutex{},
		triggerTimer:         nil,
		lastTrigger:          time.Time{},
		convergeMu:           sync.Mutex{},
		converging:           make(map[string]struct{}),
		deferredWatcherMu:    sync.Mutex{},
		deferredWatcherPaths: make(map[string]map[string]struct{}),
		queue:                nil,
		watcher:              nil,
	}
}

// Start launches the file watcher plus the periodic and trigger-driven sync
// loops.
func (syncer *BackgroundSync) Start(ctx context.Context) {
	if syncer.cfg.FileWatcherEnabled {
		syncer.queue = NewEventQueue(defaultTriggerDebounce, func(codebaseID string, relativePaths []string) {
			syncer.convergeViaWatcher(ctx, codebaseID, relativePaths)
		})
		syncer.watcher = NewWatcher(syncer.manager, syncer.queue)
		syncer.manager.SetCodebaseLifecycleHook(syncer)
		go func() {
			defer func() {
				if recovered := recover(); recovered != nil {
					slog.ErrorContext(ctx, "background sync loop panic", "loop", "watcher", "err", recovered)
				}
			}()
			syncer.watcher.Run(ctx)
		}()
	}
	if syncer.cfg.TriggerWatcherEnabled {
		go func() {
			defer func() {
				if recovered := recover(); recovered != nil {
					slog.ErrorContext(ctx, "background sync loop panic", "loop", "watchTrigger", "err", recovered)
				}
			}()
			syncer.watchTrigger(ctx)
		}()
	}
	if syncer.cfg.BackgroundSyncEnabled {
		go func() {
			defer func() {
				if recovered := recover(); recovered != nil {
					slog.ErrorContext(ctx, "background sync loop panic", "loop", "runPeriodicSync", "err", recovered)
				}
			}()
			syncer.runPeriodicSync(ctx)
		}()
	}
}

// AddCodebase forwards a newly registered codebase to the underlying watcher so
// its file events are tracked.
func (syncer *BackgroundSync) AddCodebase(ctx context.Context, codebase model.Codebase) {
	if syncer.watcher == nil {
		return
	}
	syncer.watcher.AddCodebase(ctx, codebase)
}

// RemoveCodebase drops any deferred first-build watcher paths for the codebase
// and stops the underlying watcher from tracking it.
func (syncer *BackgroundSync) RemoveCodebase(ctx context.Context, codebaseID string) {
	syncer.IndexStopped(ctx, codebaseID)
	if syncer.watcher == nil {
		return
	}
	syncer.watcher.RemoveCodebase(ctx, codebaseID)
}

// IndexReady flushes the watcher paths deferred during a codebase's first build
// once that build has promoted, so edits made while the live collection did not
// yet exist converge exactly once.
func (syncer *BackgroundSync) IndexReady(ctx context.Context, codebase model.Codebase) {
	relativePaths := syncer.takeDeferredWatcherPaths(codebase.ID)
	if len(relativePaths) == 0 {
		return
	}
	flushCtx := context.WithoutCancel(ctx)
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.ErrorContext(flushCtx, "background sync loop panic", "loop", "indexReadyFlush", "err", recovered)
			}
		}()
		syncer.convergeViaWatcher(flushCtx, codebase.ID, relativePaths)
	}()
}

// IndexStopped drops any watcher paths deferred for a codebase whose first build
// failed or was cancelled, so a failed build does not later replay churn.
func (syncer *BackgroundSync) IndexStopped(_ context.Context, codebaseID string) {
	syncer.deferredWatcherMu.Lock()
	defer syncer.deferredWatcherMu.Unlock()
	delete(syncer.deferredWatcherPaths, codebaseID)
}

func (syncer *BackgroundSync) runPeriodicSync(ctx context.Context) {
	initialTimer := time.NewTimer(defaultInitialSyncDelay)
	defer initialTimer.Stop()

	syncInterval := time.Duration(syncer.cfg.SyncIntervalMS) * time.Millisecond
	if syncer.cfg.SyncIntervalMS < minimumSyncIntervalMS {
		syncInterval = time.Duration(minimumSyncIntervalMS) * time.Millisecond
	}
	ticker := time.NewTicker(syncInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-initialTimer.C:
			syncer.ensureMmapEnabled(ctx)
			syncer.backfillConversationColumns(ctx)
			syncer.runSyncAll(ctx, "startup")
		case <-ticker.C:
			syncer.ensureMmapEnabled(ctx)
			syncer.backfillConversationColumns(ctx)
			syncer.runSyncAll(ctx, "interval")
		}
	}
}

// ensureMmapEnabled drives the idempotent dense-vector mmap migration across all
// collections once per periodic tick. It is a no-op when Milvus is unavailable
// and near-free after the first successful sweep (already-migrated collections
// are in-memory guard hits), so it is safe to run on every tick. Running it from
// the periodic loop gives the migration convergence and self-heal without putting
// migration policy in the semantic connection layer.
func (syncer *BackgroundSync) ensureMmapEnabled(ctx context.Context) {
	if syncer.manager == nil || syncer.manager.semantic == nil {
		return
	}
	syncer.manager.semantic.EnsureMmapEnabledAllCollections(ctx)
}

// backfillConversationColumns drives the metadata-only conversation scalar-column
// backfill once per conversation collection per process. It is a no-op when
// Milvus is unavailable and a guard hit after the first successful run per
// collection, so it is safe to run on every tick. It preserves each dense vector,
// so no chunk is re-embedded.
func (syncer *BackgroundSync) backfillConversationColumns(ctx context.Context) {
	if syncer.manager == nil || syncer.manager.semantic == nil {
		return
	}
	syncer.manager.semantic.BackfillConversationCollectionsOnce(ctx)
}

func (syncer *BackgroundSync) watchTrigger(ctx context.Context) {
	if err := store.EnsureDir(syncer.cfg.ContextRoot); err != nil {
		slog.ErrorContext(ctx, "ensure legacy context directory failed", "path", syncer.cfg.ContextRoot, "err", err)
		return
	}

	triggerPath := filepath.Join(syncer.cfg.ContextRoot, ".sync-trigger")
	if info, err := os.Stat(triggerPath); err == nil {
		syncer.lastTrigger = info.ModTime()
	}

	ticker := time.NewTicker(defaultTriggerPollPeriod)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			info, err := os.Stat(triggerPath)
			if err != nil {
				if !errors.Is(err, os.ErrNotExist) {
					slog.ErrorContext(ctx, "stat sync trigger failed", "path", triggerPath, "err", err)
				}
				continue
			}
			if !info.ModTime().After(syncer.lastTrigger) {
				continue
			}
			syncer.lastTrigger = info.ModTime()
			syncer.scheduleTriggerSync(ctx)
		}
	}
}

func (syncer *BackgroundSync) scheduleTriggerSync(ctx context.Context) {
	syncer.mu.Lock()
	defer syncer.mu.Unlock()

	if syncer.triggerTimer != nil {
		syncer.triggerTimer.Stop()
	}
	syncer.triggerTimer = time.AfterFunc(defaultTriggerDebounce, func() {
		syncer.runSyncAll(ctx, "trigger")
	})
}

func (syncer *BackgroundSync) runSyncAll(ctx context.Context, source string) {
	if ctx.Err() != nil {
		return
	}

	rootCorr := correlation.New("").WithIdentityAttributes(
		correlation.IdentityAttribute{Key: "origin", Value: "sync-" + source},
	)
	ctx = correlation.WithContext(ctx, rootCorr)

	defer func() {
		if recovered := recover(); recovered != nil {
			_, _ = adapterr.Respond(ctx, adapterr.NewInternal("background sync panic", fmt.Errorf("panic: %v", recovered)))
		}
	}()

	// Skip the sweep whenever the embedder is already recorded unreachable. That
	// one fact is recorded reactively by whatever first fails an embed against the
	// down endpoint (a search, an index job, or an earlier sweep that attempted
	// before the fact was set). Once it is set, every sweep reads it here and
	// enqueues nothing, so a sustained outage stops re-recording itself as a fresh
	// failed and superseded job per codebase per interval. The fact clears through
	// the existing embed-success signals (a search query embed or a real embed),
	// after which the next sweep resumes. This only reads the shared health owner;
	// it adds no probe and no second source of truth.
	if syncer.manager.DependencyHealth().Mode == dependencyEmbedderUnreachable {
		return
	}

	syncer.manager.RepairMissingCollections(ctx)

	codebases := syncer.manager.ListIndexes(ctx)
	for _, codebase := range codebases {
		if codebase.Kind == model.CodebaseKindDocument {
			continue
		}
		if codebase.Status == model.CodebaseStatusQuarantined {
			syncer.handleQuarantinedCodebase(ctx, codebase)
			continue
		}
		if _, err := os.Stat(codebase.CanonicalPath); errors.Is(err, os.ErrNotExist) {
			continue
		}
		hasActiveJob := syncer.hasActiveJob(codebase)
		if shouldSkipForActiveFirstBuildStaging(codebase, hasActiveJob) {
			continue
		}
		// A discovered worktree whose deferred build never ran (for example the
		// daemon restarted before the short timer fired) is built here as the
		// backstop. StartIndex deduplicates, so this never double-starts a build
		// that the timer already kicked off.
		if codebase.Status == model.CodebaseStatusDiscovered {
			discoverCtx := correlation.WithContext(ctx, correlation.FromContext(ctx).Child().WithIdentityAttributes(
				correlation.IdentityAttribute{Key: "codebase_id", Value: codebase.ID},
			))
			syncer.manager.startDeferredBuild(discoverCtx, codebase.CanonicalPath)
			continue
		}
		if codebase.Status == model.CodebaseStatusFailed {
			retryCtx := correlation.WithContext(ctx, correlation.FromContext(ctx).Child().WithIdentityAttributes(
				correlation.IdentityAttribute{Key: "codebase_id", Value: codebase.ID},
			))
			syncer.manager.retryFailedBuild(retryCtx, codebase)
			continue
		}
		if codebase.Status != model.CodebaseStatusIndexed {
			continue
		}

		syntheticJobID := fmt.Sprintf("sync-%s-%d", codebase.ID, clock.Now().Unix())
		iterCtx := correlation.WithContext(ctx, correlation.FromContext(ctx).Child().WithIdentityAttributes(
			correlation.IdentityAttribute{Key: "codebase_id", Value: codebase.ID},
			correlation.IdentityAttribute{Key: "job_id", Value: syntheticJobID},
		))

		// Keep this codebase's ignore rules fresh independent of the file watcher.
		// CheckSources stats the codebase's ignore sources and invalidates the
		// resolver when any changed, so an edit to a non-indexed source or any edit
		// made while the watcher is disabled is caught on this sweep. It runs before
		// and independent of the change detection below.
		syncer.manager.observer.CheckSources(iterCtx, codebase.ID, codebase.CanonicalPath)

		changed, err := syncer.codebaseChanged(iterCtx, codebase)
		if err != nil {
			slog.ErrorContext(iterCtx, "check sync state failed", "path", codebase.CanonicalPath, "err", err)
			continue
		}
		metrics.SweepRan(changed)
		if !changed {
			continue
		}

		_, _, _, err = syncer.manager.SyncIndex(
			iterCtx,
			codebase.CanonicalPath,
			model.ClientInfo{Name: "daemon-sync", PID: 0},
		)
		if err != nil {
			if syncConflictError(err) {
				continue
			}
			slog.ErrorContext(iterCtx, "start sync job failed", "path", codebase.CanonicalPath, "err", err)
		}
	}
}

func (syncer *BackgroundSync) handleQuarantinedCodebase(ctx context.Context, codebase model.Codebase) {
	if sourceDirMissing(codebase.CanonicalPath) {
		syncer.manager.markCodebaseMissing(ctx, codebase.ID)
		return
	}

	// Never advance toward destructive sync or clear quarantine while a git
	// operation is mid-flight: tracked files legitimately vanish during a
	// checkout, rebase, or merge and reappear when it finishes. Hold the
	// quarantine and re-evaluate on a later sweep once the tree settles.
	if vcsOperationInProgress(codebase.CanonicalPath) {
		slog.WarnContext(ctx, "quarantine held during vcs operation", "codebase_id", codebase.ID, "path", codebase.CanonicalPath)
		return
	}

	snapshotPath := syncer.manager.snapshotPathForCodebase(codebase)
	snapshot := merkle.LoadSnapshotForConfig(snapshotPath, codebase.EffectiveConfig.IgnoreDigest, syncer.manager.legacyDigestForCodebase(codebase.ID))
	currentSnapshot, err := merkle.Capture(ctx, syncer.manager.indexability, codebase.ID, codebase.CanonicalPath, codebase.EffectiveConfig)
	if err != nil {
		slog.ErrorContext(ctx, "quarantine capture failed", "codebase_id", codebase.ID, "path", codebase.CanonicalPath, "err", err)
		return
	}
	diff := merkle.DiffSnapshots(snapshot, currentSnapshot)
	signal, suspicious := assessDeltaDeleteWave(codebase, diff, snapshot, codebase.CanonicalPath)
	if !suspicious {
		syncer.manager.clearCodebaseQuarantine(ctx, codebase.ID, model.CodebaseStatusIndexed)
		if diff.Empty() {
			return
		}
		_, _, _, err = syncer.manager.SyncIndex(
			ctx,
			codebase.CanonicalPath,
			model.ClientInfo{Name: "daemon-quarantine-release", PID: 0},
		)
		if err != nil && !syncConflictError(err) {
			slog.ErrorContext(ctx, "start sync job after clearing quarantine failed", "codebase_id", codebase.ID, "path", codebase.CanonicalPath, "err", err)
		}
		return
	}

	observations := syncer.manager.quarantineCodebase(ctx, codebase.ID, signal)
	if observations < quarantineConfirmationObservations {
		slog.WarnContext(ctx, "quarantine held after corroborating full scan", "codebase_id", codebase.ID, "missing_count", signal.missingCount, "total_count", signal.totalCount, "observations", observations)
		return
	}

	job, codebase, deduplicated, err := syncer.manager.SyncIndex(
		ctx,
		codebase.CanonicalPath,
		model.ClientInfo{Name: "daemon-quarantine-release", PID: 0},
	)
	_ = job
	_ = codebase
	_ = deduplicated
	if err != nil && !syncConflictError(err) {
		slog.ErrorContext(ctx, "start destructive sync after quarantine confirmation failed", "codebase_id", codebase.ID, "path", codebase.CanonicalPath, "err", err)
	}
}

// convergeViaWatcher runs the debounced path set for one codebase through the
// manager's index-slot semaphore, so several codebases converge at once up to
// the cap and a heavily-edited repository never blocks the others. A second
// converge of the same codebase, or one that finds the shared lock held by the
// external tool, requeues its paths so no change is lost.
func (syncer *BackgroundSync) convergeViaWatcher(ctx context.Context, codebaseID string, relativePaths []string) {
	if ctx.Err() != nil {
		return
	}

	corr := correlation.New("").WithIdentityAttributes(
		correlation.IdentityAttribute{Key: "origin", Value: "watcher"},
		correlation.IdentityAttribute{Key: "codebase_id", Value: codebaseID},
		correlation.IdentityAttribute{Key: "job_id", Value: fmt.Sprintf("watch-%s-%d", codebaseID, clock.Now().Unix())},
	)
	ctx = correlation.WithContext(ctx, corr)

	if codebase, found := syncer.watcherCodebase(codebaseID); found && shouldDeferWatcherConvergeForFirstBuild(codebase) {
		syncer.deferWatcherPaths(codebaseID, relativePaths)
		return
	}

	// Serialize converges of the same codebase so two never race on its
	// snapshot; a concurrent one requeues rather than waits.
	if !syncer.beginConverge(codebaseID) {
		metrics.SyncSkippedInflight()
		syncer.requeuePaths(codebaseID, relativePaths)
		return
	}
	defer syncer.endConverge(codebaseID)

	// Bound concurrency across codebases through the shared index-slot
	// semaphore that user index jobs also use.
	select {
	case syncer.manager.indexSlots <- struct{}{}:
		defer func() { <-syncer.manager.indexSlots }()
	case <-ctx.Done():
		return
	}

	// Hold the shared advisory lock for the embed window. A zero-refcount lock
	// held on disk means the external TS tool owns it, so defer and requeue.
	if !syncer.manager.syncLock.acquire(ctx) {
		metrics.SyncSkippedInflight()
		syncer.requeuePaths(codebaseID, relativePaths)
		return
	}
	defer syncer.manager.syncLock.release(ctx)

	if err := syncer.manager.ConvergePaths(ctx, codebaseID, relativePaths); err != nil {
		slog.ErrorContext(ctx, "watcher.converge_failed", "component", "daemon", "subcomponent", "watcher", "codebase_id", codebaseID, "err", err)
	}
}

func (syncer *BackgroundSync) watcherCodebase(codebaseID string) (model.Codebase, bool) {
	syncer.manager.mu.Lock()
	defer syncer.manager.mu.Unlock()
	codebase, found := syncer.manager.codebases[codebaseID]
	return codebase, found
}

func (syncer *BackgroundSync) hasActiveJob(codebase model.Codebase) bool {
	syncer.manager.mu.Lock()
	defer syncer.manager.mu.Unlock()
	current, found := syncer.manager.codebases[codebase.ID]
	if !found {
		return false
	}
	return syncer.manager.activeJobSnapshotLocked(current) != nil
}

func (syncer *BackgroundSync) deferWatcherPaths(codebaseID string, relativePaths []string) {
	if len(relativePaths) == 0 {
		return
	}
	syncer.deferredWatcherMu.Lock()
	defer syncer.deferredWatcherMu.Unlock()
	buffered, found := syncer.deferredWatcherPaths[codebaseID]
	if !found {
		buffered = make(map[string]struct{}, len(relativePaths))
		syncer.deferredWatcherPaths[codebaseID] = buffered
	}
	for _, relativePath := range relativePaths {
		buffered[relativePath] = struct{}{}
	}
}

func (syncer *BackgroundSync) takeDeferredWatcherPaths(codebaseID string) []string {
	syncer.deferredWatcherMu.Lock()
	defer syncer.deferredWatcherMu.Unlock()
	buffered := syncer.deferredWatcherPaths[codebaseID]
	if len(buffered) == 0 {
		return nil
	}
	relativePaths := make([]string, 0, len(buffered))
	for relativePath := range buffered {
		relativePaths = append(relativePaths, relativePath)
	}
	sort.Strings(relativePaths)
	delete(syncer.deferredWatcherPaths, codebaseID)
	return relativePaths
}

// beginConverge claims the per-codebase converge slot, returning false when a
// converge for that codebase is already running.
func (syncer *BackgroundSync) beginConverge(codebaseID string) bool {
	syncer.convergeMu.Lock()
	defer syncer.convergeMu.Unlock()
	if _, running := syncer.converging[codebaseID]; running {
		return false
	}
	syncer.converging[codebaseID] = struct{}{}
	return true
}

// endConverge releases the per-codebase converge slot.
func (syncer *BackgroundSync) endConverge(codebaseID string) {
	syncer.convergeMu.Lock()
	defer syncer.convergeMu.Unlock()
	delete(syncer.converging, codebaseID)
}

// requeuePaths re-enqueues a deferred converge's paths so the change is picked
// up on the next debounce rather than dropped.
func (syncer *BackgroundSync) requeuePaths(codebaseID string, relativePaths []string) {
	for _, relativePath := range relativePaths {
		syncer.queue.Enqueue(codebaseID, relativePath)
	}
}

func (syncer *BackgroundSync) codebaseChanged(ctx context.Context, codebase model.Codebase) (bool, error) {
	if codebase.Kind == model.CodebaseKindDocument {
		return false, nil
	}

	snapshotPath := syncer.manager.snapshotPathForCodebase(codebase)

	existingSnapshot, err := merkle.ReadSnapshot(snapshotPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return true, nil
		}
		slog.ErrorContext(ctx, "read Merkle snapshot failed", "path", snapshotPath, "err", err)
		return false, fmt.Errorf("read Merkle snapshot %s: %w", snapshotPath, err)
	}

	currentSnapshot, err := merkle.Capture(
		ctx,
		syncer.manager.indexability,
		codebase.ID,
		codebase.CanonicalPath,
		codebase.EffectiveConfig,
	)
	if err != nil {
		slog.ErrorContext(ctx, "capture Merkle snapshot failed", "path", codebase.CanonicalPath, "err", err)
		return false, fmt.Errorf("capture Merkle snapshot for %s: %w", codebase.CanonicalPath, err)
	}
	if !merkle.Equal(existingSnapshot, currentSnapshot) {
		return true, nil
	}
	presence := syncer.manager.probeCollectionPresence(ctx, codebase.CanonicalPath, "backgroundSync")
	currentSnapshotHash := snapshotHashForGraph(currentSnapshot, codebase.EffectiveConfig.IgnoreDigest)
	return syncer.manager.shouldReconcileGraph(codebase.ID, currentSnapshotHash, presence), nil
}

func syncConflictError(err error) bool {
	if err == nil {
		return false
	}
	message := err.Error()
	return strings.Contains(message, "conflicting active job") ||
		strings.Contains(message, "codebase not tracked")
}

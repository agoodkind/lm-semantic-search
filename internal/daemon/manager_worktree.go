package daemon

import (
	"context"
	"log/slog"
	"time"

	"goodkind.io/gklog/correlation"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/gitworktree"
	"goodkind.io/lm-semantic-search/internal/model"
)

var worktreeDeferredBuildClient = model.ClientInfo{Name: "worktree-deferred-build", PID: 0}

var worktreeHeldReleaseClient = model.ClientInfo{Name: "worktree-held-release", PID: 0}

// defaultDeferredBuildDelay is how long after discovering a worktree on a read
// the daemon waits before starting its build. It is short enough to be far
// faster than the periodic sweep, yet keeps the build off the read path so the
// status or search call that discovered the worktree returns without embedding.
const defaultDeferredBuildDelay = 3 * time.Second

func (manager *Manager) resolveWorktreeIndex(ctx context.Context, canonicalPath string) (model.Codebase, bool) {
	var empty model.Codebase
	info, isWorktree := gitworktree.Resolve(canonicalPath)
	if !isWorktree {
		return empty, false
	}

	manager.mu.Lock()
	if _, exists := manager.findCodebaseByExactRoot(info.WorktreeRoot); exists {
		// The worktree already has its own codebase; longest-prefix coverage
		// resolves to it without intervention.
		manager.mu.Unlock()
		return empty, false
	}
	hasIndexedSibling := manager.hasIndexedSiblingWorktreeLocked(info.WorktreeRoot, info.CommonDir)
	siblingFirstBuildRunning := false
	if !hasIndexedSibling {
		siblingFirstBuildRunning = manager.siblingFirstBuildInProgressLocked(info.WorktreeRoot, info.CommonDir)
	}
	manager.mu.Unlock()
	if !hasIndexedSibling && !siblingFirstBuildRunning {
		return empty, false
	}

	record, ok := manager.discoverWorktree(ctx, info)
	if !ok {
		return empty, false
	}
	manager.scheduleDeferredBuild(ctx, record.CanonicalPath)
	return record, true
}

func (manager *Manager) discoverWorktree(ctx context.Context, info gitworktree.Info) (model.Codebase, bool) {
	indexConfig := manager.enrichIndexConfig(emptyAutoIndexConfig())
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)

	collectionName := ""
	collectionAbsent := true
	if manager.semantic != nil {
		collectionName = manager.semantic.CollectionName(info.WorktreeRoot)
		if manager.semantic.Available() {
			hasCollection, collectionErr := manager.semantic.HasCollectionForPath(ctx, info.WorktreeRoot)
			if collectionErr != nil {
				slog.WarnContext(ctx, "discover worktree: check collection failed", "path", info.WorktreeRoot, "err", collectionErr)
			} else {
				collectionAbsent = !hasCollection
			}
		}
	}

	sourceOverlaps := manager.discoveredSourceOverlaps(ctx, info, indexConfig)

	manager.policyMutationMutex.Lock()
	defer manager.policyMutationMutex.Unlock()
	manager.mu.Lock()
	if existing, found := manager.findCodebaseByExactRoot(info.WorktreeRoot); found {
		manager.mu.Unlock()
		return existing, true
	}
	record := newCodebaseRecord(info.WorktreeRoot)
	record.Status = model.CodebaseStatusDiscovered
	record.PolicyPendingInitialization = collectionAbsent
	record.EffectiveConfig = indexConfig
	record.CollectionName = collectionName
	record.WorktreeCommonDir = info.CommonDir
	source, sourceSelected := manager.selectDiscoveredWorktreeSourceLocked(info, indexConfig, sourceOverlaps)
	if sourceSelected {
		record.WorktreeSourcePath = source.codebase.CanonicalPath
		record.WorktreeSourceID = source.codebase.ID
	}
	record.InodeTrackingDisabled = detectInodeTrackingDisabled(ctx, info.WorktreeRoot)
	record.MerkleSnapshotPath = manager.merklePath(record.ID)
	record.UpdatedAt = clock.Now()
	manager.codebases[record.ID] = record
	if err := manager.saveLocked(); err != nil {
		delete(manager.codebases, record.ID)
		manager.mu.Unlock()
		slog.ErrorContext(ctx, "discover worktree: persist registry failed", "path", info.WorktreeRoot, "err", err)
		var empty model.Codebase
		return empty, false
	}
	// A discovered worktree persists a fresh EffectiveConfig, so signal the
	// observer to invalidate rather than relying on the id being new; the next
	// decision rebuilds from the registry source of truth.
	manager.observer.Invalidate(record.ID)
	manager.mu.Unlock()

	notifyCtx := correlation.WithContext(context.WithoutCancel(ctx), correlation.FromContext(ctx).Child())
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.ErrorContext(notifyCtx, "notify codebase added panic", "codebase_id", record.ID, "err", recovered)
			}
		}()
		manager.notifyCodebaseAdded(notifyCtx, record)
	}()
	slog.InfoContext(ctx, "discovered worktree on read; build deferred", "codebase_id", record.ID, "path", info.WorktreeRoot, "common_dir", info.CommonDir, "source_path", record.WorktreeSourcePath, "source_id", record.WorktreeSourceID, "source_tier", string(source.tier), "source_overlap_matches", source.overlap.matches, "source_overlap_sampled", source.overlap.sampled)
	return record, true
}

// A detached timer starts the discovered worktree's build after the deferred build delay.
// Index admission prevents repeated reads or watcher events from starting a second build while a job is in flight.
func (manager *Manager) scheduleDeferredBuild(ctx context.Context, canonicalPath string) {
	manager.afterDeferredBuildDelay(ctx, canonicalPath, func(detached context.Context) {
		manager.startDeferredBuild(detached, canonicalPath)
	})
}

// afterDeferredBuildDelay runs start in a detached timer after the deferred
// build delay, recovering a panic so it cannot take the daemon down.
func (manager *Manager) afterDeferredBuildDelay(ctx context.Context, canonicalPath string, start func(context.Context)) {
	detached := correlation.WithContext(context.WithoutCancel(ctx), correlation.FromContext(ctx).Child())
	delay := manager.deferredBuildDelay
	if delay <= 0 {
		delay = defaultDeferredBuildDelay
	}
	time.AfterFunc(delay, func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.ErrorContext(detached, "deferred worktree build panic", "path", canonicalPath, "err", recovered)
			}
		}()
		start(detached)
	})
}

// It is the body scheduleDeferredBuild fires on its timer, split out so a test
// can drive it synchronously. Shared index admission deduplicates, so calling
// it for a worktree that already has an in-flight job is a no-op.
func (manager *Manager) startDeferredBuild(ctx context.Context, canonicalPath string) {
	if _, err := manager.startAutomaticIndex(ctx, canonicalPath, worktreeDeferredBuildClient, emptyAutoIndexConfig(), indexPolicyIntent{
		Patch:      model.SchedulingPolicyPatch{Priority: nil, Quiet: nil, IdleAfterSeconds: nil},
		Initialize: false,
	}); err != nil {
		slog.WarnContext(ctx, "deferred worktree build failed to start", "path", canonicalPath, "err", err)
	}
}

// automaticStart is what one startAutomaticIndex call did.
type automaticStart struct {
	job          model.Job
	codebase     model.Codebase
	deduplicated bool
	held         bool
	// paused reports that nothing started because the daemon is in
	// maintenance mode. The sweep after the mode ends starts it again.
	paused bool
}

func (manager *Manager) startAutomaticIndex(
	ctx context.Context,
	canonicalPath string,
	client model.ClientInfo,
	indexConfig model.IndexConfig,
	policyIntent indexPolicyIntent,
) (automaticStart, error) {
	var noJob model.Job
	var noCodebase model.Codebase
	if manager.skipForMaintenance(ctx, "automatic-index:"+client.Name) {
		return automaticStart{job: noJob, codebase: noCodebase, deduplicated: false, held: false, paused: true}, nil
	}
	if manager.holdForSourceFirstBuild(canonicalPath) {
		slog.InfoContext(ctx, "automatic build waits for the first build of the recorded source", "path", canonicalPath, "client", client.Name)
		return automaticStart{job: noJob, codebase: noCodebase, deduplicated: false, held: true, paused: false}, nil
	}
	job, codebase, deduplicated, _, err := manager.startIndexWithIntent(ctx, canonicalPath, client, indexConfig, false, emptyAdmissionBudget, policyIntent)
	if err != nil {
		return automaticStart{job: noJob, codebase: noCodebase, deduplicated: false, held: false, paused: false}, err
	}
	manager.clearHeldWorktreeBuild(codebase.ID)
	return automaticStart{job: job, codebase: codebase, deduplicated: deduplicated, held: false, paused: false}, nil
}

func (manager *Manager) startAutomaticSync(ctx context.Context, codebase model.Codebase, client model.ClientInfo) {
	if manager.skipForMaintenance(ctx, "automatic-sync:"+client.Name) {
		return
	}
	if manager.holdForSourceFirstBuild(codebase.CanonicalPath) {
		slog.InfoContext(ctx, "automatic sync waits for the first build of the recorded source", "codebase_id", codebase.ID, "path", codebase.CanonicalPath, "client", client.Name)
		return
	}
	job, _, deduplicated, err := manager.SyncIndex(ctx, codebase.CanonicalPath, client)
	if err != nil {
		if !syncConflictError(err) {
			slog.ErrorContext(ctx, "start sync job failed", "codebase_id", codebase.ID, "path", codebase.CanonicalPath, "client", client.Name, "err", err)
		}
		return
	}
	manager.clearHeldWorktreeBuild(codebase.ID)
	slog.DebugContext(ctx, "automatic sync started", "codebase_id", codebase.ID, "job_id", job.ID, "client", client.Name, "deduplicated", deduplicated)
}

func (manager *Manager) noteHeldWorktreeBuild(codebaseID string) {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	manager.noteHeldWorktreeBuildLocked(codebaseID)
}

// noteHeldWorktreeBuildLocked is noteHeldWorktreeBuild for a caller that holds
// manager.mu.
func (manager *Manager) noteHeldWorktreeBuildLocked(codebaseID string) {
	if manager.heldWorktreeBuilds == nil {
		manager.heldWorktreeBuilds = map[string]struct{}{}
	}
	manager.heldWorktreeBuilds[codebaseID] = struct{}{}
}

// clearHeldWorktreeBuild forgets a held build once one has started for it.
func (manager *Manager) clearHeldWorktreeBuild(codebaseID string) {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	delete(manager.heldWorktreeBuilds, codebaseID)
}

// hasIndexedSiblingWorktreeLocked reports whether any worktree of the same repo
// group (other than worktreeRoot) is a tracked codebase holding embedded
// content, which is the condition that turns a worktree into "a worktree of an
// indexed repo" for the auto-create trigger. A sibling without embedded content does not count.
// Caller must hold manager.mu.
func (manager *Manager) hasIndexedSiblingWorktreeLocked(worktreeRoot string, commonDir string) bool {
	if commonDir == "" {
		return false
	}
	siblingRoots := gitworktree.SiblingWorktreeRoots(commonDir)
	siblings := make(map[string]struct{}, len(siblingRoots))
	for _, root := range siblingRoots {
		if root != worktreeRoot {
			siblings[root] = struct{}{}
		}
	}
	if len(siblings) == 0 {
		return false
	}
	for _, codebase := range manager.codebases {
		if _, ok := siblings[codebase.CanonicalPath]; !ok {
			continue
		}
		if ownsLiveCollection(codebase) {
			return true
		}
	}
	return false
}

// siblingFirstBuildInProgressLocked reports whether another worktree in the same repo group is building without embedded content. The caller must hold manager.mu.
func (manager *Manager) siblingFirstBuildInProgressLocked(worktreeRoot string, commonDir string) bool {
	if commonDir == "" {
		return false
	}
	siblings := make(map[string]struct{})
	for _, root := range gitworktree.SiblingWorktreeRoots(commonDir) {
		if root != worktreeRoot {
			siblings[root] = struct{}{}
		}
	}
	for _, codebase := range manager.codebases {
		if _, ok := siblings[codebase.CanonicalPath]; !ok {
			continue
		}
		if manager.activeJobSnapshotLocked(codebase) != nil && !ownsLiveCollection(codebase) {
			return true
		}
	}
	return false
}

func (manager *Manager) waitsForSourceFirstBuild(canonicalPath string) bool {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	_, held := manager.sourceFirstBuildHoldLocked(canonicalPath)
	return held
}

func (manager *Manager) holdForSourceFirstBuild(canonicalPath string) bool {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	ownID, held := manager.sourceFirstBuildHoldLocked(canonicalPath)
	if held {
		manager.noteHeldWorktreeBuildLocked(ownID)
	}
	return held
}

func (manager *Manager) sourceFirstBuildHoldLocked(canonicalPath string) (string, bool) {
	own, found := manager.findCodebaseByExactRoot(canonicalPath)
	if !found {
		return "", false
	}
	if own.WorktreeSourcePath == "" || ownsLiveCollection(own) {
		return own.ID, false
	}
	source, found := manager.findCodebaseByExactRoot(own.WorktreeSourcePath)
	if !found || source.Kind == model.CodebaseKindDocument || ownsLiveCollection(source) {
		return own.ID, false
	}
	return own.ID, manager.activeJobSnapshotLocked(source) != nil
}

func (manager *Manager) startBuildsHeldForSource(ctx context.Context, codebaseID string) {
	manager.mu.Lock()
	ended, found := manager.codebases[codebaseID]
	if !found || ended.Kind == model.CodebaseKindDocument {
		manager.mu.Unlock()
		return
	}
	held := make([]model.Codebase, 0)
	heldPaths := make([]string, 0)
	for _, codebase := range manager.codebases {
		if codebase.WorktreeSourcePath != ended.CanonicalPath {
			continue
		}
		if _, wasHeld := manager.heldWorktreeBuilds[codebase.ID]; !wasHeld {
			continue
		}
		delete(manager.heldWorktreeBuilds, codebase.ID)
		held = append(held, codebase)
		heldPaths = append(heldPaths, codebase.CanonicalPath)
	}
	manager.mu.Unlock()
	if len(held) == 0 {
		return
	}

	slog.InfoContext(ctx, "daemon scheduled waiting builds after the source build ended", "codebase_id", codebaseID, "paths", heldPaths)
	for _, codebase := range held {
		releasedID := codebase.ID
		manager.afterDeferredBuildDelay(ctx, codebase.CanonicalPath, func(detached context.Context) {
			manager.startReleasedBuild(detached, releasedID)
		})
	}
}

func (manager *Manager) startReleasedBuild(ctx context.Context, codebaseID string) {
	if manager.DependencyHealth().Mode == dependencyEmbedderUnreachable {
		slog.InfoContext(ctx, "held worktree build left to the sweep while the embedder is unreachable", "codebase_id", codebaseID)
		manager.noteHeldWorktreeBuild(codebaseID)
		return
	}
	manager.mu.Lock()
	codebase, found := manager.codebases[codebaseID]
	busy := found && manager.activeJobSnapshotLocked(codebase) != nil
	_, coalesced := manager.pendingCodeJobs[codebaseID]
	manager.mu.Unlock()
	if !found || busy || coalesced {
		return
	}

	switch codebase.Status {
	case model.CodebaseStatusDiscovered:
		manager.startDeferredBuild(ctx, codebase.CanonicalPath)
	case model.CodebaseStatusFailed:
		manager.retryFailedBuild(ctx, codebase)
	case model.CodebaseStatusIndexing, model.CodebaseStatusPending, model.CodebaseStatusNotIndexed:
		if _, err := manager.startAutomaticIndex(ctx, codebase.CanonicalPath, worktreeHeldReleaseClient, codebase.EffectiveConfig, indexPolicyIntent{
			Patch:      model.SchedulingPolicyPatch{Priority: nil, Quiet: nil, IdleAfterSeconds: nil},
			Initialize: true,
		}); err != nil {
			slog.WarnContext(ctx, "held worktree build failed to start", "codebase_id", codebaseID, "path", codebase.CanonicalPath, "err", err)
		}
	case model.CodebaseStatusIndexed:
		// Only an empty last run leaves an indexed worktree held; one that has
		// content of its own since then needs no release.
		if ranWithoutCreatingACollection(codebase.LastSuccessfulRun) {
			manager.startAutomaticSync(ctx, codebase, worktreeHeldReleaseClient)
		}
	case model.CodebaseStatusStale, model.CodebaseStatusMissing, model.CodebaseStatusQuarantined:
		return
	default:
		return
	}
}

// isWorktreeBoundary reports whether canonicalPath is the root of a git worktree
// whose repo group matches the covering ancestor, which makes the two distinct
// worktrees of the same repository that must stay separate codebases rather than
// merge. The merge-up redirect consults it so a worktree root is never folded
// into a sibling worktree's index.
func (manager *Manager) isWorktreeBoundary(canonicalPath string, ancestor model.Codebase) bool {
	info, ok := gitworktree.Resolve(canonicalPath)
	if !ok || info.WorktreeRoot != canonicalPath {
		return false
	}
	ancestorCommon, ok := gitworktree.CommonDirAt(ancestor.CanonicalPath)
	if !ok {
		return false
	}
	return ancestorCommon == info.CommonDir
}

// isSameRepoSiblingWorktree reports whether child is a worktree of the same repo
// group as parentPath but rooted at a different worktree. Such a child is a
// sibling worktree, not a nested part of the parent, so the parent's build must
// not reuse its vectors or absorb its registration.
func isSameRepoSiblingWorktree(parentPath string, childPath string) bool {
	parentCommon, ok := gitworktree.CommonDirAt(parentPath)
	if !ok {
		return false
	}
	childCommon, ok := gitworktree.CommonDirAt(childPath)
	if !ok {
		return false
	}
	return parentCommon == childCommon && parentPath != childPath
}

// emptyAutoIndexConfig is the zero index config handed to an auto-created
// worktree build; enrichIndexConfig fills in the daemon defaults.
func emptyAutoIndexConfig() model.IndexConfig {
	return model.IndexConfig{
		SplitterType: "", SplitterChunkSize: 0, SplitterOverlap: 0,
		IgnorePatterns: nil, IncludeSubmodules: nil, IgnoreDigest: "",
		EmbeddingProvider: "", EmbeddingModel: "", EmbeddingDimension: 0,
		VectorBackend: "", Hybrid: false,
	}
}

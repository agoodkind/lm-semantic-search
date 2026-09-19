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

// worktreeDeferredBuildClient labels the reuse-seeded index jobs the daemon
// starts for a worktree it discovered on a read, so the job's origin is legible
// in status and logs and is distinct from an operator-driven index_codebase.
var worktreeDeferredBuildClient = model.ClientInfo{Name: "worktree-deferred-build", PID: 0}

// worktreeHeldReleaseClient labels the build that resumes an interrupted
// worktree build once the sibling first build that held it has ended.
var worktreeHeldReleaseClient = model.ClientInfo{Name: "worktree-held-release", PID: 0}

// defaultDeferredBuildDelay is how long after discovering a worktree on a read
// the daemon waits before starting its build. It is short enough to be far
// faster than the periodic sweep, yet keeps the build off the read path so the
// status or search call that discovered the worktree returns without embedding.
const defaultDeferredBuildDelay = 3 * time.Second

// resolveWorktreeIndex implements worktree-bounded resolution as a read that
// discovers but never embeds. When canonicalPath lives inside a worktree whose
// root is not yet tracked, and at least one sibling worktree of the same
// repository holds embedded content, the daemon registers the worktree as a
// discovered codebase, starts watching it, and schedules a reuse-seeded build in
// the background; the read itself launches no embed job. When no sibling holds
// content yet but one is running its first build, the worktree is registered
// the same way and its build is held, so it does not embed what that sibling is
// about to hold: startDeferredBuild skips a held worktree, and
// startHeldSiblingWorktreeBuilds schedules it again when that build ends. The
// returned bool is false when canonicalPath is not such a worktree,
// leaving the caller's normal coverage resolution untouched.
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

// discoverWorktree persists a first-class registry record for a worktree the
// daemon learned about on a read, in the discovered state, and starts watching
// it. It mirrors adoptUnregisteredCodebase but creates no job and starts no
// embed: the reuse-seeded build is deferred to scheduleDeferredBuild. It returns
// the persisted record and true, or false when the registry write fails so the
// caller falls back to normal resolution.
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
	slog.InfoContext(ctx, "discovered worktree on read; build deferred", "codebase_id", record.ID, "path", info.WorktreeRoot, "common_dir", info.CommonDir)
	return record, true
}

// scheduleDeferredBuild starts the reuse-seeded build for a discovered worktree
// after a short delay, off the read path, in a detached timer. The delay makes
// the build far faster than the periodic sweep while keeping the read that
// discovered the worktree free of any embed. The build deduplicates against any
// job already in flight, so a repeat read or a watcher event cannot double-start.
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

// startDeferredBuild starts the reuse-seeded bootstrap for a discovered worktree.
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
	// held reports that nothing started because the codebase waits for a
	// sibling worktree's first build.
	held bool
}

// startAutomaticIndex is the one entry for every build the daemon starts on its
// own: the deferred worktree build, the repair pass's resume of an interrupted
// build, and the failed build retry. It starts nothing while the codebase at
// canonicalPath is a worktree that waits for a sibling's first build, so none of
// those paths embeds what that build is about to hold. The hold is read when
// the build would start, not when it was scheduled. startHeldSiblingWorktreeBuilds
// starts a held codebase again when that sibling's build ends. An operator's
// index request goes straight to startIndexWithIntent and is never held.
func (manager *Manager) startAutomaticIndex(
	ctx context.Context,
	canonicalPath string,
	client model.ClientInfo,
	indexConfig model.IndexConfig,
	policyIntent indexPolicyIntent,
) (automaticStart, error) {
	if manager.waitsForSiblingFirstBuild(canonicalPath) {
		slog.InfoContext(ctx, "automatic build held for sibling first build", "path", canonicalPath, "client", client.Name)
		return automaticStart{job: model.Job{}, codebase: model.Codebase{}, deduplicated: false, held: true}, nil
	}
	job, codebase, deduplicated, _, err := manager.startIndexWithIntent(ctx, canonicalPath, client, indexConfig, false, emptyAdmissionBudget, policyIntent)
	if err != nil {
		return automaticStart{job: model.Job{}, codebase: model.Codebase{}, deduplicated: false, held: false}, err
	}
	return automaticStart{job: job, codebase: codebase, deduplicated: deduplicated, held: false}, nil
}

// worktreeReuseForecast reports how many indexed sibling worktree collections a
// worktree at codebase.CanonicalPath would reuse on its build, for the discovered
// status and the list. It is cheap: git topology plus an in-memory registry scan,
// no vector-store call, so a status read that shows the forecast stays cheap.
func (manager *Manager) worktreeReuseForecast(codebase model.Codebase) int32 {
	return safeInt32(len(manager.worktreeSiblingReuseCollections(codebase.CanonicalPath, codebase.EffectiveConfig)))
}

// hasIndexedSiblingWorktreeLocked reports whether any worktree of the same repo
// group (other than worktreeRoot) is a tracked codebase holding embedded
// content, which is the condition that turns a worktree into "a worktree of an
// indexed repo" for the auto-create trigger. A sibling whose last run indexed no
// file does not count, because it has no vectors for the worktree to reuse.
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

// siblingFirstBuildInProgressLocked reports whether any worktree of the same
// repo group (other than worktreeRoot) is running a build while it holds no
// embedded content yet, which is a first build whose vectors a worktree at the
// same commit would otherwise embed a second time. Caller must hold manager.mu.
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

// waitsForSiblingFirstBuild reports whether the worktree at canonicalPath must
// hold its build: it holds no embedded content of its own, no sibling holds any
// to reuse yet, and one sibling is running the first build that will produce it.
// A worktree with its own content builds as a delta against it, which a
// sibling's vectors would not shorten.
func (manager *Manager) waitsForSiblingFirstBuild(canonicalPath string) bool {
	info, ok := gitworktree.Resolve(canonicalPath)
	if !ok {
		return false
	}
	manager.mu.Lock()
	defer manager.mu.Unlock()
	if own, found := manager.findCodebaseByExactRoot(canonicalPath); found && ownsLiveCollection(own) {
		return false
	}
	if manager.hasIndexedSiblingWorktreeLocked(info.WorktreeRoot, info.CommonDir) {
		return false
	}
	return manager.siblingFirstBuildInProgressLocked(info.WorktreeRoot, info.CommonDir)
}

// startHeldSiblingWorktreeBuilds schedules a build for every sibling worktree
// of a codebase whose job just ended that has no live job and no content of its
// own, whichever automatic path held it: a discovered worktree, an interrupted
// build the repair pass would resume, or a failed build the retry would rerun.
// Such a worktree would otherwise wait for the periodic sweep. It runs on
// success, failure, and cancellation alike: after a success the worktree reuses
// the new content, and after a failure or cancellation it builds without reuse
// rather than staying stranded. startReleasedBuild re-reads the codebase and the
// hold when its timer fires, and admission deduplicates, so scheduling one that
// is not held, or one already building, is a no-op.
func (manager *Manager) startHeldSiblingWorktreeBuilds(ctx context.Context, codebaseID string) {
	manager.mu.Lock()
	ended, found := manager.codebases[codebaseID]
	manager.mu.Unlock()
	if !found || ended.Kind == model.CodebaseKindDocument {
		return
	}
	info, ok := gitworktree.Resolve(ended.CanonicalPath)
	if !ok || info.CommonDir == "" {
		return
	}
	siblings := make(map[string]struct{})
	for _, root := range gitworktree.SiblingWorktreeRoots(info.CommonDir) {
		if root != info.WorktreeRoot {
			siblings[root] = struct{}{}
		}
	}

	manager.mu.Lock()
	held := make([]model.Codebase, 0)
	heldPaths := make([]string, 0)
	for _, codebase := range manager.codebases {
		if _, ok := siblings[codebase.CanonicalPath]; !ok {
			continue
		}
		if manager.activeJobSnapshotLocked(codebase) != nil || ownsLiveCollection(codebase) {
			continue
		}
		if releasesHeldBuild(codebase.Status) {
			held = append(held, codebase)
			heldPaths = append(heldPaths, codebase.CanonicalPath)
		}
	}
	manager.mu.Unlock()
	if len(held) == 0 {
		return
	}

	slog.InfoContext(ctx, "sibling build ended; scheduling held worktree builds", "codebase_id", codebaseID, "paths", heldPaths)
	for _, codebase := range held {
		releasedID := codebase.ID
		manager.afterDeferredBuildDelay(ctx, codebase.CanonicalPath, func(detached context.Context) {
			manager.startReleasedBuild(detached, releasedID)
		})
	}
}

// releasesHeldBuild reports whether a codebase in status is one an automatic
// path would build and so may have been held: discovered and awaiting its first
// build, interrupted before one finished, or failed and awaiting a retry.
func releasesHeldBuild(status model.CodebaseStatus) bool {
	switch status {
	case model.CodebaseStatusDiscovered, model.CodebaseStatusIndexing, model.CodebaseStatusPending,
		model.CodebaseStatusNotIndexed, model.CodebaseStatusFailed:
		return true
	case model.CodebaseStatusIndexed, model.CodebaseStatusStale, model.CodebaseStatusMissing,
		model.CodebaseStatusQuarantined:
		return false
	default:
		return false
	}
}

// startReleasedBuild starts the build the automatic path for a codebase's
// current status would start, once the sibling build that held it has ended. It
// goes through the same path so that path's own rules still apply, such as the
// failed build retry's attempt cap.
func (manager *Manager) startReleasedBuild(ctx context.Context, codebaseID string) {
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
	case model.CodebaseStatusIndexed, model.CodebaseStatusStale, model.CodebaseStatusMissing,
		model.CodebaseStatusQuarantined:
		return
	default:
		return
	}
}

// worktreeSiblingReuseCollections returns the collection names of indexed
// sibling worktrees of the same repo group as canonicalPath whose embedding
// model matches indexConfig. A worktree build preloads reuse vectors from these
// so files unchanged from a sibling reuse their embeddings and only the branch
// diff is embedded. It returns nil when canonicalPath is not a worktree or has
// no eligible sibling.
func (manager *Manager) worktreeSiblingReuseCollections(canonicalPath string, indexConfig model.IndexConfig) []string {
	info, ok := gitworktree.Resolve(canonicalPath)
	if !ok {
		return nil
	}
	siblingRoots := gitworktree.SiblingWorktreeRoots(info.CommonDir)
	siblings := make(map[string]struct{}, len(siblingRoots))
	for _, root := range siblingRoots {
		if root != info.WorktreeRoot {
			siblings[root] = struct{}{}
		}
	}
	if len(siblings) == 0 {
		return nil
	}

	manager.mu.Lock()
	defer manager.mu.Unlock()
	collections := make([]string, 0)
	for _, codebase := range manager.codebases {
		if _, member := siblings[codebase.CanonicalPath]; !member {
			continue
		}
		if codebase.Kind == model.CodebaseKindDocument {
			continue
		}
		if codebase.CollectionName == "" {
			continue
		}
		// Reuse keys on durable facts, not the transient ActiveJobID: a sibling
		// whose last completed run indexed files, or an adopted sibling, owns a
		// live collection. A run that indexed no file created no collection, so it
		// has nothing to reuse. An in-flight sync does not drop the live
		// collection, and reuse is content-hash keyed, so reading a mid-sync
		// sibling is safe. This mirrors the auto-create trigger's eligibility so
		// the two agree.
		if !ownsLiveCollection(codebase) {
			continue
		}
		if !reuseModelMatches(codebase.EffectiveConfig, indexConfig) {
			continue
		}
		collections = append(collections, codebase.CollectionName)
	}
	return collections
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

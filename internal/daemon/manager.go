// Package daemon owns persisted daemon state and request coordination.
package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"goodkind.io/gklog/correlation"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/discovery"
	"goodkind.io/lm-semantic-search/internal/gitworktree"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/spans"
	"goodkind.io/lm-semantic-search/internal/store"
)

// jobOperation tags one daemon job so runJob can route it to the right
// execution path. The model.Job.Operation field is a plain string for wire
// compatibility, but the daemon's internal switch uses this named type so
// staticcheck can verify the dispatch covers every case.
type jobOperation string

const (
	// jobOperationIndex runs a full Replace against an empty or
	// previously-cleared collection.
	jobOperationIndex jobOperation = "index"
	// jobOperationSync runs an incremental delta against the existing
	// merkle snapshot and falls back to full Replace when no snapshot exists.
	jobOperationSync jobOperation = "sync"
	// jobOperationStreamingReindex re-walks the entire codebase and
	// replaces chunks file by file through semantic.Reindex, so the existing
	// Milvus collection stays searchable across the upgrade.
	jobOperationStreamingReindex jobOperation = "streaming_reindex"
	// jobOperationConversationIngest upserts or deletes virtual conversation
	// documents in a document collection.
	jobOperationConversationIngest jobOperation = "conversation_ingest"
)

// CodebaseLifecycleHook is the watcher-side interface the manager calls so
// new codebases start receiving filesystem events without a restart. The
// hook is plugged in via SetCodebaseLifecycleHook; until then the manager
// is a no-op for these callbacks.
type CodebaseLifecycleHook interface {
	AddCodebase(ctx context.Context, codebase model.Codebase)
	RemoveCodebase(ctx context.Context, codebaseID string)
}

// Manager coordinates persisted codebase and job state for the daemon.
type Manager struct {
	config           config.Config
	mu               sync.Mutex
	codebases        map[string]model.Codebase
	jobs             map[string]model.Job
	conversationJobs map[string]conversationJobPayload
	cancels          map[string]context.CancelFunc
	done             map[string]chan struct{}
	runner           indexingRunner
	semantic         semanticIndex
	lifecycleHook    CodebaseLifecycleHook
	lifecycleMutex   sync.Mutex
	// indexSlots caps concurrently running index jobs. Each runJob holds one
	// buffered slot for its duration; jobs that cannot acquire a slot stay
	// queued until one frees.
	indexSlots chan struct{}
	// syncLock is the process-wide refcounted hold of the shared advisory lock
	// that coordinates embedding with the upstream TS adapter. Index jobs and
	// background converges all take a reference for the duration of their
	// embed, so the external tool backs off while any daemon embed runs.
	syncLock *syncLock
	// health is the daemon's view of shared-infrastructure health (the embedding
	// pipeline and the vector store). It is global, not per-codebase, observed
	// from job outcomes, and drives the status banner. Guarded by mu.
	health dependencyHealth
	// lastDepProbeAt debounces refreshDependencyHealth's backend probe. Guarded by mu.
	lastDepProbeAt time.Time
	// deferredBuildDelay is the post-discovery wait before a worktree build starts; settable so a test can keep the timer from firing mid-test.
	deferredBuildDelay time.Duration
}

// SearchOutcome carries search results plus current indexing context.
// StateNote adds read-only repair or availability context to the rendered
// response when results alone are not enough to explain the current state.
type SearchOutcome struct {
	Codebase  model.Codebase
	ActiveJob *model.Job
	Results   []model.StoredChunk
	StateNote string
}

type indexingRunner interface {
	Index(context.Context, string, model.IndexConfig, func(indexer.Progress)) (indexer.Result, error)
	IndexFiles(context.Context, string, []string, model.IndexConfig, func(indexer.Progress)) (indexer.Result, error)
	IndexOne(context.Context, string, string, model.IndexConfig) (indexer.OneFileResult, error)
}

// NewManager loads persisted daemon state from disk.
func NewManager(ctx context.Context, cfg config.Config) (*Manager, error) {
	manager := &Manager{
		config:             cfg,
		mu:                 sync.Mutex{},
		codebases:          map[string]model.Codebase{},
		jobs:               map[string]model.Job{},
		conversationJobs:   map[string]conversationJobPayload{},
		cancels:            map[string]context.CancelFunc{},
		done:               map[string]chan struct{}{},
		runner:             indexer.NewRunner(),
		semantic:           nil,
		lifecycleHook:      nil,
		lifecycleMutex:     sync.Mutex{},
		indexSlots:         make(chan struct{}, max(1, cfg.MaxConcurrentIndexJobs)),
		syncLock:           newSyncLock(filepath.Join(cfg.ContextRoot, "mcp-sync.lock"), cfg.ContextRoot, cfg.SyncLockStaleMS),
		health:             dependencyHealth{Mode: dependencyHealthy, Since: time.Time{}, LastHealthyAt: time.Time{}},
		lastDepProbeAt:     time.Time{},
		deferredBuildDelay: defaultDeferredBuildDelay,
	}
	semanticService, err := semantic.NewService(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("create semantic service: %w", err)
	}
	manager.semantic = semanticService
	if semanticService.Degraded() {
		manager.health = dependencyHealth{Mode: dependencyStoreUnavailable, Since: clock.Now(), LastHealthyAt: time.Time{}}
	}
	if err := manager.load(ctx); err != nil {
		slog.ErrorContext(ctx, "load daemon state failed", "state_root", cfg.StateRoot, "err", err)
		return nil, fmt.Errorf("load daemon state: %w", err)
	}
	return manager, nil
}

func (manager *Manager) load(ctx context.Context) error {
	registry, err := store.ReadRegistry(manager.config.RegistryPath)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		slog.ErrorContext(ctx, "read registry failed", "path", manager.config.RegistryPath, "err", err)
		return fmt.Errorf("read registry: %w", err)
	}
	for _, codebase := range registry.Codebases {
		if codebase.Kind == "" {
			codebase.Kind = model.CodebaseKindCode
		}
		manager.codebases[codebase.ID] = codebase
	}
	dropGhostURICodebases(manager.codebases)

	jobs, err := store.ReadJobEvents(manager.config.JobsPath)
	if err != nil {
		slog.ErrorContext(ctx, "read jobs failed", "path", manager.config.JobsPath, "err", err)
		return fmt.Errorf("read jobs: %w", err)
	}
	maps.Copy(manager.jobs, jobs)
	manager.reconcileJournalOnStartLocked()
	return nil
}

// dropGhostURICodebases removes code-kind records whose canonical path is a
// filesystem-mangled URI, which the previous boot resume path could create by
// running [filepath.Abs] on a chat URI. A legitimate conversation codebase keeps
// its scheme intact and its kind set to document, so it never matches.
func dropGhostURICodebases(codebases map[string]model.Codebase) {
	for id, codebase := range codebases {
		if codebase.Kind == model.CodebaseKindDocument {
			continue
		}
		segments := strings.SplitN(strings.TrimPrefix(codebase.CanonicalPath, "/"), "/", 2)
		if len(segments) > 0 && strings.HasSuffix(segments[0], ":") {
			slog.Warn("dropping ghost URI codebase record", "codebase_id", id, "path", codebase.CanonicalPath)
			delete(codebases, id)
		}
	}
}

// reconcileJournalOnStartLocked sanitizes the job journal after the previous
// daemon process exited. Any queued, running, or cancelling job becomes
// cancelled in the journal because its goroutine is gone. Codebase records
// keep Status=Indexing when they were mid-flight so ResumeOrphanedJobs can
// pick them back up on boot; the registry already holds the canonical path
// and effective config that resume needs.
func (manager *Manager) reconcileJournalOnStartLocked() {
	now := clock.Now()
	documentCodebaseChanged := false
	for id, job := range manager.jobs {
		switch job.State {
		case model.JobStateQueued, model.JobStateRunning, model.JobStateCancelling:
		case model.JobStateCompleted, model.JobStateFailed, model.JobStateCancelled:
			continue
		default:
			continue
		}
		job.State = model.JobStateCancelled
		job.UpdatedAt = now
		completedAt := now
		job.CompletedAt = &completedAt
		job.Progress.Phase = "cancelled"
		job.Progress.LastEventAt = now
		job.Progress.HeartbeatAt = now
		manager.jobs[id] = job
		if err := store.AppendJobEvent(manager.config.JobsPath, model.JobEvent{
			Event:      "job_orphan_recovered",
			OccurredAt: now,
			Job:        job,
		}); err != nil {
			slog.Error("append orphan recovery event failed", "job_id", id, "err", err)
		}
		codebase, found := manager.codebases[job.CodebaseID]
		if found && codebase.Kind == model.CodebaseKindDocument && codebase.ActiveJobID == id {
			codebase.Status = model.CodebaseStatusIndexed
			codebase.ActiveJobID = ""
			codebase.UpdatedAt = now
			manager.codebases[codebase.ID] = codebase
			documentCodebaseChanged = true
		}
		slog.Warn("orphan job sanitized in journal after restart", "job_id", id, "codebase_id", job.CodebaseID)
	}
	if documentCodebaseChanged {
		if err := manager.saveLocked(); err != nil {
			slog.Error("write registry after document orphan recovery failed", "err", err)
		}
	}
}

func (manager *Manager) saveLocked() error {
	codebases := make([]model.Codebase, 0, len(manager.codebases))
	for _, codebase := range manager.codebases {
		codebases = append(codebases, codebase)
	}
	sort.Slice(codebases, func(i int, j int) bool {
		return codebases[i].CanonicalPath < codebases[j].CanonicalPath
	})
	registry := model.RegistryFile{
		Codebases: codebases,
		UpdatedAt: clock.Now(),
	}
	if err := store.WriteRegistry(manager.config.RegistryPath, registry); err != nil {
		slog.Error("write registry failed", "path", manager.config.RegistryPath, "err", err)
		return fmt.Errorf("write registry %s: %w", manager.config.RegistryPath, err)
	}
	return nil
}

func (manager *Manager) appendJobLocked(event string, job model.Job) error {
	manager.jobs[job.ID] = job
	jobEvent := model.JobEvent{
		Event:      event,
		OccurredAt: clock.Now(),
		Job:        job,
	}
	if err := store.AppendJobEvent(manager.config.JobsPath, jobEvent); err != nil {
		slog.Error("append jobs journal failed", "path", manager.config.JobsPath, "err", err)
		return fmt.Errorf("append jobs journal %s: %w", manager.config.JobsPath, err)
	}
	return nil
}

// Version returns daemon runtime path details.
func (manager *Manager) Version() map[string]string {
	return map[string]string{
		"state_root":  manager.config.StateRoot,
		"socket_path": manager.config.SocketPath,
	}
}

func newCodebaseRecord(canonicalPath string) model.Codebase {
	return model.Codebase{
		ID:                newID("cb"),
		Kind:              model.CodebaseKindCode,
		CanonicalPath:     canonicalPath,
		Status:            model.CodebaseStatusNotIndexed,
		ActiveJobID:       "",
		LastSuccessfulRun: nil,
		LastFailedRun:     nil,
		LiveFileTotal:     0,
		LiveChunkTotal:    0,
		EffectiveConfig: model.IndexConfig{
			SplitterType:       "",
			SplitterChunkSize:  0,
			SplitterOverlap:    0,
			Extensions:         nil,
			IgnorePatterns:     nil,
			IgnoreDigest:       "",
			EmbeddingProvider:  "",
			EmbeddingModel:     "",
			EmbeddingDimension: 0,
			VectorBackend:      "",
			Hybrid:             false,
		},
		CollectionName:        "",
		LegacyCollectionNames: nil,
		MerkleSnapshotPath:    "",
		WorktreeCommonDir:     "",
		InodeTrackingDisabled: false,
		ResolvedIgnoreRules:   discovery.IgnoreRules{Nodes: nil},
		UpdatedAt:             clock.Now(),
	}
}

func newQueuedJob(
	codebaseID string,
	requestedPath string,
	canonicalPath string,
	client model.ClientInfo,
	operation string,
	forced bool,
	indexConfig model.IndexConfig,
	now time.Time,
) model.Job {
	return model.Job{
		ID:            newID("job"),
		CodebaseID:    codebaseID,
		RequestedPath: requestedPath,
		CanonicalPath: canonicalPath,
		Client:        client,
		Operation:     operation,
		State:         model.JobStateQueued,
		Forced:        forced,
		Progress: model.Progress{
			Phase:                     "queued",
			PhasePercent:              0,
			OverallPercent:            0,
			Unit:                      "",
			RunMode:                   "",
			ScopeUnit:                 "",
			FilesTotal:                0,
			FilesProcessed:            0,
			FilesAdded:                0,
			FilesModified:             0,
			FilesRemoved:              0,
			FilesInCodebase:           0,
			FilesEmbedded:             0,
			FilesSkippedOversize:      0,
			FilesSkippedUnreadable:    0,
			FilesPending:              0,
			ChunksTotal:               0,
			ChunksProcessed:           0,
			ChunksReused:              0,
			ChunksEmbedded:            0,
			ChunksGenerated:           0,
			ReuseVectorsLoaded:        0,
			EmbeddingBatchesTotal:     0,
			EmbeddingBatchesCompleted: 0,
			CollectionRowsWritten:     0,
			LastEventAt:               now,
			HeartbeatAt:               now,
		},
		Config:      indexConfig,
		StartedAt:   now,
		UpdatedAt:   now,
		CompletedAt: nil,
		Error:       nil,
	}
}

// startIndexDecision captures one StartIndex call's resolved codebase plus
// the routing decision derived from the current registry state.
type startIndexDecision struct {
	codebase  model.Codebase
	activeJob model.Job
	dedup     bool
	mode      startIndexMode
}

// decideStartIndexLocked resolves the codebase record and routing mode from
// the registry plus the caller-provided collection presence. The shared
// collection-state policy decides whether this call is already indexed, should
// queue an incremental run, or must bootstrap. Caller must hold manager.mu.
func (manager *Manager) decideStartIndexLocked(canonicalPath string, indexConfig model.IndexConfig, force bool, presence collectionPresence) (startIndexDecision, error) {
	var emptyJob model.Job
	codebase, found := manager.findCodebaseByExactRoot(canonicalPath)
	if !found {
		fresh := newCodebaseRecord(canonicalPath)
		if presence == collectionPresencePresent {
			fresh.Status = model.CodebaseStatusIndexed
		}
		return startIndexDecision{
			codebase:  fresh,
			activeJob: emptyJob,
			dedup:     false,
			mode:      decideStartIndexMode(false, fresh.Status, false, force, presence),
		}, nil
	}
	activeJob, deduplicated, err := manager.activeJobLocked(codebase, canonicalPath, indexConfig)
	if err != nil {
		return startIndexDecision{}, err
	}
	if deduplicated {
		return startIndexDecision{
			codebase:  codebase,
			activeJob: activeJob,
			dedup:     true,
			mode:      startIndexModeBootstrap,
		}, nil
	}
	return startIndexDecision{
		codebase:  codebase,
		activeJob: emptyJob,
		dedup:     false,
		mode: decideStartIndexMode(
			true,
			codebase.Status,
			codebase.EffectiveConfig.IgnoreDigest == indexConfig.IgnoreDigest,
			force,
			presence,
		),
	}, nil
}

// StartIndex registers a new indexing job or deduplicates an existing one.
//
// Returns the queued or in-flight job, the resolved codebase record, a
// deduplicated flag (true when the call matched an in-flight job), and the
// id of any existing codebase whose canonical path strictly prefix-covers
// the new registration (empty when no overlap).
func (manager *Manager) StartIndex(ctx context.Context, requestedPath string, client model.ClientInfo, indexConfig model.IndexConfig, force bool) (model.Job, model.Codebase, bool, string, error) {
	var emptyJob model.Job
	var emptyCodebase model.Codebase

	canonicalPath, err := manager.resolveCanonicalPath(requestedPath)
	if err != nil {
		slog.ErrorContext(ctx, "canonicalize path failed", "path", requestedPath, "err", err)
		return emptyJob, emptyCodebase, false, "", fmt.Errorf("canonicalize path %s: %w", requestedPath, err)
	}

	if err := manager.guardStateRoot(canonicalPath); err != nil {
		return emptyJob, emptyCodebase, false, "", err
	}
	if err := manager.guardFilesystemRoot(canonicalPath); err != nil {
		return emptyJob, emptyCodebase, false, "", err
	}
	if err := manager.guardDirectory(canonicalPath); err != nil {
		return emptyJob, emptyCodebase, false, "", err
	}

	// Merge-up: a nested path already covered by an indexed parent does not get
	// its own redundant index. Resolve to the covering parent and sync it so the
	// requested subtree is current, rather than building a second overlapping
	// collection over the shared files. A git worktree root is exempt: it shares
	// the parent's repo group but holds a different branch, so it stays its own
	// codebase rather than merging into a sibling worktree.
	if ancestor, found := manager.mergeUpTarget(canonicalPath); found && !manager.isWorktreeBoundary(canonicalPath, ancestor) {
		return manager.redirectIndexToAncestor(ctx, requestedPath, ancestor, client)
	}

	indexConfig = manager.enrichIndexConfig(indexConfig)
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)

	if dedupedJob, dedupedCodebase, deduped := manager.dedupAgainstActiveJob(canonicalPath, indexConfig); deduped {
		return dedupedJob, dedupedCodebase, true, "", nil
	}

	if force {
		if err := manager.cancelActiveJobForPath(ctx, canonicalPath); err != nil {
			return emptyJob, emptyCodebase, false, "", err
		}
	}

	presence := manager.probeCollectionPresence(ctx, canonicalPath, "StartIndex")

	job, codebase, deduped, overlapsCodebaseID, err := manager.commitStartIndexLocked(ctx, canonicalPath, requestedPath, client, indexConfig, force, presence)
	if err != nil || deduped {
		return job, codebase, deduped, overlapsCodebaseID, err
	}
	if job.ID == "" {
		return emptyJob, codebase, false, overlapsCodebaseID, nil
	}
	notifyCtx := correlation.WithContext(context.WithoutCancel(ctx), correlation.FromContext(ctx).Child())
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.ErrorContext(notifyCtx, "notify codebase added panic", "codebase_id", codebase.ID, "err", recovered)
			}
		}()
		manager.notifyCodebaseAdded(notifyCtx, codebase)
	}()
	ctx = spans.Attach(ctx, correlation.IdentityAttribute{Key: "job_id", Value: job.ID}, correlation.IdentityAttribute{Key: "codebase_id", Value: codebase.ID})
	manager.runJobAsync(ctx, job.ID)
	return job, codebase, false, overlapsCodebaseID, nil
}

// probeCollectionPresence asks Milvus whether canonicalPath already has a live
// collection and preserves the distinction between missing and unknown
// backend state. Unknown must not be treated as definite collection loss.
func (manager *Manager) probeCollectionPresence(ctx context.Context, canonicalPath string, caller string) collectionPresence {
	if manager.semantic == nil || !manager.semantic.Available() {
		return collectionPresenceUnknown
	}
	present, hasErr := manager.semantic.HasCollectionForPath(ctx, canonicalPath)
	if hasErr != nil {
		slog.WarnContext(ctx, "Milvus HasCollection failed", "caller", caller, "path", canonicalPath, "err", hasErr)
		return collectionPresenceUnknown
	}
	if present {
		return collectionPresencePresent
	}
	return collectionPresenceMissing
}

// commitStartIndexLocked acquires the registry lock, runs the decision
// table, applies the resulting codebase mutation, persists the registry,
// and queues the job event. The returned job has an empty ID when the
// decision resolved as already-indexed; the caller treats that as a no-op
// success.
func (manager *Manager) commitStartIndexLocked(ctx context.Context, canonicalPath string, requestedPath string, client model.ClientInfo, indexConfig model.IndexConfig, force bool, presence collectionPresence) (model.Job, model.Codebase, bool, string, error) {
	var emptyJob model.Job
	var emptyCodebase model.Codebase
	manager.mu.Lock()
	defer manager.mu.Unlock()

	decision, err := manager.decideStartIndexLocked(canonicalPath, indexConfig, force, presence)
	if err != nil {
		slog.ErrorContext(ctx, "resolve active job failed", "canonical_path", canonicalPath, "err", err)
		return emptyJob, emptyCodebase, false, "", err
	}
	if decision.dedup {
		return decision.activeJob, decision.codebase, true, "", nil
	}
	overlapsCodebaseID := ""
	if ancestor, found := manager.findStrictAncestor(canonicalPath); found {
		overlapsCodebaseID = ancestor.ID
	}
	if decision.mode == startIndexModeAlreadyIndexed {
		return emptyJob, decision.codebase, false, overlapsCodebaseID, nil
	}

	codebase := decision.codebase
	codebase.Status = model.CodebaseStatusIndexing
	codebase.EffectiveConfig = indexConfig
	codebase.InodeTrackingDisabled = detectInodeTrackingDisabled(ctx, canonicalPath)
	if manager.semantic != nil && manager.semantic.Available() {
		codebase.CollectionName = manager.semantic.CollectionName(canonicalPath)
	}
	if info, ok := gitworktree.Resolve(canonicalPath); ok && info.Linked {
		codebase.WorktreeCommonDir = info.CommonDir
	}
	codebase.UpdatedAt = clock.Now()

	operation := jobOperationIndex
	if decision.mode == startIndexModeIncremental {
		operation = jobOperationStreamingReindex
	}
	job := newQueuedJob(codebase.ID, requestedPath, canonicalPath, client, string(operation), force, indexConfig, clock.Now())
	codebase.ActiveJobID = job.ID
	manager.codebases[codebase.ID] = codebase
	if err := manager.saveLocked(); err != nil {
		return emptyJob, emptyCodebase, false, "", err
	}
	if err := manager.appendJobLocked("start_index", job); err != nil {
		return emptyJob, emptyCodebase, false, "", err
	}
	return job, codebase, false, overlapsCodebaseID, nil
}

// SyncIndex registers a new sync job for an existing tracked codebase.
func (manager *Manager) SyncIndex(ctx context.Context, requestedPath string, client model.ClientInfo) (model.Job, model.Codebase, bool, error) {
	canonicalPath, err := manager.resolveCanonicalPath(requestedPath)
	if err != nil {
		slog.ErrorContext(ctx, "canonicalize path failed", "path", requestedPath, "err", err)
		var emptyJob model.Job
		var emptyCodebase model.Codebase
		return emptyJob, emptyCodebase, false, fmt.Errorf("canonicalize path %s: %w", requestedPath, err)
	}

	manager.mu.Lock()

	matches := manager.findCodebasesByCoverage(canonicalPath)
	if len(matches) == 0 {
		manager.mu.Unlock()
		var emptyJob model.Job
		var emptyCodebase model.Codebase
		return emptyJob, emptyCodebase, false, errors.New("codebase not tracked: " + requestedPath)
	}
	codebase := matches[0]

	indexConfig := manager.enrichIndexConfig(codebase.EffectiveConfig)
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)

	activeJob, deduplicated, err := manager.activeJobLocked(codebase, canonicalPath, indexConfig)
	if err != nil {
		slog.ErrorContext(ctx, "resolve active sync job failed", "canonical_path", canonicalPath, "err", err)
		manager.mu.Unlock()
		var emptyJob model.Job
		var emptyCodebase model.Codebase
		return emptyJob, emptyCodebase, false, err
	}
	if deduplicated {
		manager.mu.Unlock()
		return activeJob, codebase, true, nil
	}

	codebase.Status = model.CodebaseStatusIndexing
	codebase.EffectiveConfig = indexConfig
	if manager.semantic != nil && manager.semantic.Available() {
		codebase.CollectionName = manager.semantic.CollectionName(canonicalPath)
	}
	if info, ok := gitworktree.Resolve(canonicalPath); ok && info.Linked {
		codebase.WorktreeCommonDir = info.CommonDir
	}
	codebase.UpdatedAt = clock.Now()

	now := clock.Now()
	job := newQueuedJob(codebase.ID, requestedPath, canonicalPath, client, string(jobOperationSync), false, indexConfig, now)

	codebase.ActiveJobID = job.ID
	manager.codebases[codebase.ID] = codebase
	if err := manager.saveLocked(); err != nil {
		manager.mu.Unlock()
		var emptyJob model.Job
		var emptyCodebase model.Codebase
		return emptyJob, emptyCodebase, false, err
	}
	if err := manager.appendJobLocked("start_sync", job); err != nil {
		manager.mu.Unlock()
		var emptyJob model.Job
		var emptyCodebase model.Codebase
		return emptyJob, emptyCodebase, false, err
	}
	manager.mu.Unlock()
	ctx = spans.Attach(ctx, correlation.IdentityAttribute{Key: "job_id", Value: job.ID}, correlation.IdentityAttribute{Key: "codebase_id", Value: codebase.ID})
	manager.runJobAsync(ctx, job.ID)
	return job, codebase, false, nil
}

// ClearIndex removes a tracked codebase from daemon state.
func (manager *Manager) ClearIndex(ctx context.Context, requestedPath string, client model.ClientInfo) (model.Codebase, error) {
	_ = client

	canonicalPath, err := manager.resolveCanonicalPath(requestedPath)
	if err != nil {
		slog.ErrorContext(ctx, "canonicalize path failed", "path", requestedPath, "err", err)
		return model.Codebase{}, fmt.Errorf("canonicalize path %s: %w", requestedPath, err)
	}

	manager.mu.Lock()
	matches := manager.findCodebasesByCoverage(canonicalPath)
	if len(matches) == 0 {
		manager.mu.Unlock()
		return model.Codebase{}, errors.New("codebase not tracked: " + requestedPath)
	}
	codebase := matches[0]
	jobDone, cancel := manager.beginActiveJobCancellationLocked(codebase)
	manager.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	if err := waitForJobDone(ctx, jobDone); err != nil {
		return model.Codebase{}, err
	}

	if err := store.RemoveFile(manager.chunkPath(codebase.ID)); err != nil {
		return model.Codebase{}, fmt.Errorf("remove chunk cache for %s: %w", codebase.ID, err)
	}
	if err := store.RemoveFile(manager.merklePath(codebase.ID)); err != nil {
		return model.Codebase{}, fmt.Errorf("remove Merkle snapshot for %s: %w", codebase.ID, err)
	}
	if manager.semantic != nil {
		if err := manager.semantic.Drop(ctx, codebase.CanonicalPath); err != nil && !errors.Is(err, semantic.ErrUnavailable) {
			return model.Codebase{}, fmt.Errorf("drop semantic index for %s: %w", codebase.CanonicalPath, err)
		}
		if err := manager.semantic.DropStaging(ctx, codebase.CanonicalPath); err != nil && !errors.Is(err, semantic.ErrUnavailable) {
			return model.Codebase{}, fmt.Errorf("drop semantic staging for %s: %w", codebase.CanonicalPath, err)
		}
	}

	manager.mu.Lock()

	clearedCodebase := codebase
	current, found := manager.codebases[codebase.ID]
	if !found {
		manager.mu.Unlock()
		manager.notifyCodebaseRemoved(ctx, codebase.ID)
		return clearedCodebase, nil
	}
	delete(manager.codebases, current.ID)
	if err := manager.saveLocked(); err != nil {
		manager.mu.Unlock()
		return model.Codebase{}, err
	}
	manager.mu.Unlock()
	manager.notifyCodebaseRemoved(ctx, current.ID)
	return current, nil
}

// CancelJob marks a tracked job as cancelled.
func (manager *Manager) CancelJob(jobID string) (model.Job, error) {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	job, found := manager.jobs[jobID]
	if !found {
		return model.Job{}, fmt.Errorf("job not found: %s", jobID)
	}
	if job.State == model.JobStateCompleted || job.State == model.JobStateFailed || job.State == model.JobStateCancelled {
		return job, nil
	}
	delete(manager.conversationJobs, jobID)

	cancel, found := manager.cancels[jobID]
	if found {
		cancel()
		delete(manager.cancels, jobID)
	}

	now := clock.Now()
	job.State = model.JobStateCancelled
	job.UpdatedAt = now
	job.CompletedAt = &now
	job.Progress.Phase = "cancelled"
	job.Progress.LastEventAt = now
	job.Progress.HeartbeatAt = now
	if err := manager.appendJobLocked("cancel_job", job); err != nil {
		return model.Job{}, err
	}

	codebase, found := manager.codebases[job.CodebaseID]
	if found && codebase.ActiveJobID == job.ID {
		// A cancellation is not a failure: leave the codebase at its last-good
		// state so a status check reflects the current usable state.
		codebase.ActiveJobID = ""
		codebase.UpdatedAt = now
		manager.codebases[codebase.ID] = codebase
		if err := manager.saveLocked(); err != nil {
			return model.Job{}, err
		}
	}

	return job, nil
}

// Codebase lifecycle hook plumbing lives in manager_lifecycle.go.

// GetIndex / classification / synthesis helpers live in manager_status.go.

// ListIndexes returns every tracked codebase in canonical path order.
func (manager *Manager) ListIndexes(ctx context.Context) []model.Codebase {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	codebases := make([]model.Codebase, 0, len(manager.codebases))
	for _, codebase := range manager.codebases {
		codebases = append(codebases, codebase)
	}
	sort.Slice(codebases, func(i int, j int) bool {
		return codebases[i].CanonicalPath < codebases[j].CanonicalPath
	})
	return codebases
}

// CodebaseView pairs a codebase with its daemon-computed display status, so the
// presentation fold (live job phase) is decided once, under the lock, rather
// than recomputed at each rendering callsite.
type CodebaseView struct {
	Codebase model.Codebase
	Display  displayStatus
}

// ListIndexesView returns every tracked codebase in canonical path order, each
// paired with its single-source-of-truth display status. It folds the active
// job in under the lock so the list and detail surfaces agree by construction.
func (manager *Manager) ListIndexesView() []CodebaseView {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	degraded := manager.health.Degraded()
	views := make([]CodebaseView, 0, len(manager.codebases))
	for _, codebase := range manager.codebases {
		activeJob := manager.activeJobSnapshotLocked(codebase)
		views = append(views, CodebaseView{
			Codebase: codebase,
			Display:  computeDisplayStatus(codebase, activeJob, degraded),
		})
	}
	sort.Slice(views, func(i int, j int) bool {
		return views[i].Codebase.CanonicalPath < views[j].Codebase.CanonicalPath
	})
	return views
}

// GetJob resolves one tracked job by id.
func (manager *Manager) GetJob(jobID string) (model.Job, bool) {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	job, found := manager.jobs[jobID]
	return job, found
}

// ListJobs returns tracked jobs, optionally filtered by codebase id.
func (manager *Manager) ListJobs(codebaseID string) []model.Job {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	jobs := make([]model.Job, 0, len(manager.jobs))
	for _, job := range manager.jobs {
		if codebaseID == "" || job.CodebaseID == codebaseID {
			jobs = append(jobs, job)
		}
	}
	sort.Slice(jobs, func(i int, j int) bool {
		return jobs[i].StartedAt.After(jobs[j].StartedAt)
	})
	return jobs
}

// JobSuccessorID returns the id of the immediate next terminal job for job's
// codebase, or empty when job is the latest terminal job. The single-job views
// use it since they do not hold the full job set the list view does.
func (manager *Manager) JobSuccessorID(job model.Job) string {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	codebaseJobs := make([]model.Job, 0)
	for _, candidate := range manager.jobs {
		if candidate.CodebaseID == job.CodebaseID {
			codebaseJobs = append(codebaseJobs, candidate)
		}
	}
	return buildJobSuccessors(codebaseJobs)[job.ID]
}

// Doctor reports basic local state-path diagnostics.
func (manager *Manager) Doctor() []string {
	diagnostics := []string{}
	for _, path := range []string{
		manager.config.StateRoot,
		manager.config.SocketsDir,
		manager.config.LogsDir,
	} {
		if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
			diagnostics = append(diagnostics, "missing path: "+path)
		}
	}

	manager.mu.Lock()
	codebases := make([]model.Codebase, 0, len(manager.codebases))
	for _, codebase := range manager.codebases {
		codebases = append(codebases, codebase)
	}
	manager.mu.Unlock()
	sort.Slice(codebases, func(i int, j int) bool {
		return codebases[i].CanonicalPath < codebases[j].CanonicalPath
	})
	for _, codebase := range codebases {
		if codebase.LastSuccessfulRun == nil {
			continue
		}
		skipped := len(codebase.LastSuccessfulRun.SkippedFiles)
		if skipped == 0 {
			continue
		}
		diagnostics = append(diagnostics, fmt.Sprintf(
			"%s: %d non-UTF-8 %s skipped during last indexing run",
			codebase.CanonicalPath,
			skipped,
			plural("file", skipped),
		))
	}
	return diagnostics
}

// dedupAgainstActiveJob returns an existing in-flight job that matches the
// caller's effective config so concurrent MCP requests (including
// force-reindex requests) collapse into a single embedding pass.
func (manager *Manager) dedupAgainstActiveJob(canonicalPath string, indexConfig model.IndexConfig) (model.Job, model.Codebase, bool) {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	existingCodebase, codebaseFound := manager.findCodebaseByExactRoot(canonicalPath)
	if !codebaseFound {
		var emptyJob model.Job
		var emptyCodebase model.Codebase
		return emptyJob, emptyCodebase, false
	}
	activeJob, deduplicated, err := manager.activeJobLocked(existingCodebase, canonicalPath, indexConfig)
	if err != nil || !deduplicated {
		var emptyJob model.Job
		var emptyCodebase model.Codebase
		return emptyJob, emptyCodebase, false
	}
	return activeJob, existingCodebase, true
}

func (manager *Manager) activeJobLocked(codebase model.Codebase, canonicalPath string, indexConfig model.IndexConfig) (model.Job, bool, error) {
	if codebase.ActiveJobID == "" {
		var emptyJob model.Job
		return emptyJob, false, nil
	}

	activeJob, found := manager.jobs[codebase.ActiveJobID]
	if !found {
		var emptyJob model.Job
		return emptyJob, false, nil
	}

	switch activeJob.State {
	case model.JobStateCompleted, model.JobStateFailed, model.JobStateCancelled:
		var emptyJob model.Job
		return emptyJob, false, nil
	case model.JobStateQueued, model.JobStateRunning, model.JobStateCancelling:
	default:
		var emptyJob model.Job
		return emptyJob, false, fmt.Errorf("unknown job state %s for active job %s", activeJob.State, activeJob.ID)
	}

	if activeJob.Config.IgnoreDigest == indexConfig.IgnoreDigest && activeJob.Config.SplitterType == indexConfig.SplitterType {
		return activeJob, true, nil
	}

	var emptyJob model.Job
	return emptyJob, false, fmt.Errorf("conflicting active job %s for canonical path %s", activeJob.ID, canonicalPath)
}

func (manager *Manager) activeJobSnapshotLocked(codebase model.Codebase) *model.Job {
	if codebase.ActiveJobID == "" {
		return nil
	}

	job, found := manager.jobs[codebase.ActiveJobID]
	if !found {
		return nil
	}
	switch job.State {
	case model.JobStateQueued, model.JobStateRunning, model.JobStateCancelling:
		jobCopy := job
		return &jobCopy
	case model.JobStateCompleted, model.JobStateFailed, model.JobStateCancelled:
		return nil
	default:
		return nil
	}
}

func (manager *Manager) cancelActiveJobForPath(ctx context.Context, canonicalPath string) error {
	manager.mu.Lock()
	codebase, found := manager.findCodebaseByExactRoot(canonicalPath)
	if !found {
		manager.mu.Unlock()
		return nil
	}
	jobDone, cancel := manager.beginActiveJobCancellationLocked(codebase)
	manager.mu.Unlock()

	if cancel == nil {
		return nil
	}

	cancel()
	if err := waitForJobDone(ctx, jobDone); err != nil {
		return err
	}
	return nil
}

func (manager *Manager) beginActiveJobCancellationLocked(codebase model.Codebase) (chan struct{}, context.CancelFunc) {
	if codebase.ActiveJobID == "" {
		return nil, nil
	}

	job, found := manager.jobs[codebase.ActiveJobID]
	if !found {
		return nil, nil
	}
	if job.State == model.JobStateCompleted || job.State == model.JobStateFailed || job.State == model.JobStateCancelled {
		return nil, nil
	}

	now := clock.Now()
	job.State = model.JobStateCancelling
	job.UpdatedAt = now
	job.Progress.Phase = "cancelling"
	job.Progress.LastEventAt = now
	job.Progress.HeartbeatAt = now
	manager.jobs[job.ID] = job
	cancel := manager.cancels[job.ID]
	jobDone := manager.done[job.ID]
	return jobDone, cancel
}

// Delta sync helpers live in manager_delta.go.
// Job state mutators live in manager_jobs_state.go.
// SearchCode and rankChunks live in manager_search.go.
// Path helpers live in manager_paths.go.
// Config helpers and id helpers live in manager_config.go.
// Boundary guards (StateRoot, directory, inode-stability) live in
// manager_guards.go.

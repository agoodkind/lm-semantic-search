package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"math"
	"os"
	"sort"

	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/spans"
	"goodkind.io/lm-semantic-search/internal/store"
)

// deltaPlan packages the file-set decision for one runDeltaSync invocation.
// fallback=true signals "no usable previous snapshot, route through full
// Replace instead". handled=true signals the helper already terminated the
// job (cancellation, snapshot-capture failure, or a no-op completion). The
// seedSnapshot is the previous on-disk checkpoint loaded under the
// requested ConfigDigest so the per-file loop can skip files already
// embedded by a prior crashed run.
type deltaPlan struct {
	diff            merkle.Diff
	currentSnapshot merkle.Snapshot
	seedSnapshot    merkle.Snapshot
	configDigest    string
	fallback        bool
	handled         bool
}

// deltaOutcome reports what happened inside a runDeltaSync step.
// fallback=true tells the caller to drop to full Replace. handled=true
// means the step terminated the job (failed, cancelled, or progressed
// normally and the caller should not run later steps). progressed=true means
// the step changed the working set (an item embedded or removed), which is
// what makes a checkpoint write worthwhile; a skipped item leaves it false so
// the per-item loop does not rewrite an unchanged snapshot.
type deltaOutcome struct {
	fallback   bool
	handled    bool
	progressed bool
}

type deltaState struct {
	plan         deltaPlan
	snapshotPath string
	working      map[string]string
	// source lists items and produces one item's chunks. It is the only
	// kind-specific part of the routine: a code source walks the filesystem, a
	// conversation source reads the manifest and documents handed over the wire.
	source   itemSource
	semantic bool
	// staging routes per-file embeds into the staging collection that a
	// from-scratch build promotes onto the live name at the end, instead of
	// the live collection an incremental sync writes to directly.
	staging bool
	// reuse maps a chunk's content hash to an already-embedded dense vector,
	// populated for a merge-down build from the collections of the indexed
	// child codebases the new parent absorbs. The embed step takes a reused
	// vector instead of calling the embedder, so the shared subtree is never
	// re-embedded. Nil for an ordinary build, which embeds every chunk.
	reuse map[string][]float32
	// chunkCounts accumulates the reused and embedded chunk counts across every
	// per-file reindex in this run, so the job progress can show total = reused +
	// embedded. It is a pointer because deltaState is copied by value through the
	// per-file loop, and the totals must survive each copy.
	chunkCounts *chunkCounters
	// seededReuse is the size of the reuse-vector pool loaded at bootstrap seed
	// time. The building view shows it immediately as the reuse total, so the
	// displayed reuse does not climb file-by-file from zero; the per-file accrual
	// can only rise to meet it as matching chunks are served from the pool.
	seededReuse int32
	admission   *admissionState
}

// chunkCounters accumulates the reuse-vs-embed split across one run's per-file
// reindex calls.
type chunkCounters struct {
	processed          int32
	reused             int32
	embedded           int32
	reuseVectorsLoaded int32
}

// applyReindexForState runs one per-file delta against the live collection, or
// against the staging collection when the job is a from-scratch build, and
// classifies any backend error into a deltaOutcome. Both targets go through
// the same delete-then-insert flow, so a re-embedded file is idempotent either
// way. A zero-value outcome means the embed succeeded and the caller should
// continue; a non-zero outcome means the step already resolved the job
// (failed, cancelled) or signalled a fallback to a full build.
func (manager *Manager) applyReindexForState(ctx context.Context, job model.Job, state deltaState, chunks []model.StoredChunk, removal semantic.Removal, phase string) deltaOutcome {
	if state.admission != nil {
		if err := state.admission.Admit(chunks); err != nil {
			return manager.handleAdmissionHalt(ctx, job, state, err)
		}
	}
	// Capture the last semantic progress for this one reindex call. The semantic
	// callback reports cumulative reused/embedded counts within the call, so the
	// final value is this file's split, which we fold into the run totals.
	var fileProcessed, fileReused, fileEmbedded int32
	progressFn := func(progress semantic.Progress) {
		fileProcessed = progress.ChunksProcessed
		fileReused = progress.ChunksReused
		fileEmbedded = progress.ChunksEmbedded
	}
	var err error
	if state.staging {
		err = manager.semantic.StageReindex(ctx, job.CanonicalPath, chunks, removal, progressFn, state.reuse)
	} else {
		err = manager.semantic.Reindex(ctx, job.CanonicalPath, chunks, removal, progressFn, state.reuse)
	}
	if err != nil {
		return manager.classifyReindexErr(ctx, job, err, phase)
	}
	if state.chunkCounts != nil {
		if fileProcessed == 0 {
			fileProcessed = fileReused + fileEmbedded
		}
		state.chunkCounts.processed += fileProcessed
		state.chunkCounts.reused += fileReused
		state.chunkCounts.embedded += fileEmbedded
	}
	return deltaOutcome{fallback: false, handled: false, progressed: false}
}

func (manager *Manager) handleAdmissionHalt(ctx context.Context, job model.Job, state deltaState, err error) deltaOutcome {
	if state.staging {
		manager.cleanupHaltedStaging(ctx, job, state)
	}
	manager.updateJobFailed(ctx, job.ID, err)
	return deltaOutcome{fallback: false, handled: true, progressed: false}
}

// cleanupHaltedStaging drops the staging collection and removes the staging
// merkle checkpoint after an admission halt, so a halted bootstrap or forced
// rebuild leaves no partial staging state and never promotes over the live
// collection.
func (manager *Manager) cleanupHaltedStaging(ctx context.Context, job model.Job, state deltaState) {
	if manager.semantic != nil && manager.semantic.Available() {
		if dropErr := manager.semantic.DropStaging(ctx, job.CanonicalPath); dropErr != nil {
			slog.WarnContext(ctx, "drop staging after admission halt failed", "path", job.CanonicalPath, "err", dropErr)
		}
	}
	if state.snapshotPath != "" {
		if removeErr := store.RemoveFile(state.snapshotPath); removeErr != nil {
			slog.WarnContext(ctx, "remove staging checkpoint after admission halt failed", "path", state.snapshotPath, "err", removeErr)
		}
	}
}

// planSyncDiff loads the previous snapshot under the requested config
// digest, captures the current one, and returns the diff. An empty diff
// completes the job as a no-op only when the shared collection-state policy
// confirms the live collection is still present or otherwise not definitively
// missing. A missing snapshot produces an empty seed whose diff classifies
// every file as Added, which the per-file loop handles uniformly.
func (manager *Manager) planSyncDiff(ctx context.Context, job model.Job, codebaseID string, source itemSource) deltaPlan {
	configDigest := job.Config.IgnoreDigest
	snapshotPath := manager.merklePath(codebaseID)
	legacyDigest := manager.legacyDigestForCodebase(codebaseID)
	seed := merkle.LoadSnapshotForConfig(snapshotPath, configDigest, legacyDigest)
	captured, err := source.capture(ctx)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			manager.updateJobCancelled(ctx, job.ID)
		} else {
			manager.updateJobFailed(ctx, job.ID, fmt.Errorf("capture sync snapshot: %w", err))
		}
		return deltaPlan{
			diff:            merkle.Diff{Added: nil, Modified: nil, Removed: nil},
			currentSnapshot: merkle.Snapshot{ConfigDigest: "", Files: nil, Inodes: nil},
			seedSnapshot:    seed,
			configDigest:    configDigest,
			fallback:        false,
			handled:         true,
		}
	}
	diff := merkle.DiffSnapshots(seed, captured)
	if diff.Empty() {
		evidence := manager.probeCollectionEvidence(ctx, job.CanonicalPath, "planSyncDiff")
		if decideEmptyDiffMode(evidence, len(seed.Files)) == emptyDiffModeFallbackBootstrap {
			manager.routeToBootstrap(ctx, job.ID, bootstrapReasonForEmptyDiffFallback(evidence))
			return deltaPlan{
				diff:            diff,
				currentSnapshot: captured,
				seedSnapshot:    seed,
				configDigest:    configDigest,
				fallback:        true,
				handled:         false,
			}
		}
		fileCount, chunkCount := manager.codebaseTotals(ctx, job.CanonicalPath, captured.Files, 0)
		var totalBytes int64
		manager.mu.Lock()
		if codebase, found := manager.codebases[codebaseID]; found && codebase.LastSuccessfulRun != nil {
			totalBytes = codebase.LastSuccessfulRun.TotalBytes
		}
		manager.mu.Unlock()
		manager.updateJobCompleted(ctx, job.ID, indexer.Result{
			IndexedFiles:      fileCount,
			TotalChunks:       chunkCount,
			TotalBytes:        totalBytes,
			Chunks:            nil,
			FileHashes:        captured.Files,
			SkippedFiles:      nil,
			SkippedOversize:   0,
			SkippedUnreadable: 0,
			SkippedPending:    0,
		})
		return deltaPlan{
			diff:            diff,
			currentSnapshot: captured,
			seedSnapshot:    seed,
			configDigest:    configDigest,
			fallback:        false,
			handled:         true,
		}
	}
	return deltaPlan{
		diff:            diff,
		currentSnapshot: captured,
		seedSnapshot:    seed,
		configDigest:    configDigest,
		fallback:        false,
		handled:         false,
	}
}

// runDeltaSync attempts the incremental sync path and returns true when it
// fully handled the job (success, failure, no-op, or cancellation). It
// returns false to fall back to the full Replace path when there is no
// previous snapshot or the semantic collection is gone.
//
// Both "sync" and "streaming_reindex" route here and share one plan: the
// merkle diff against the previous snapshot, processing only added and
// modified files. The streaming operation differs only in a post-pass prune
// that drops rows for files no longer present, which covers the splitter
// upgrade where the empty seed re-embeds everything.
func (manager *Manager) runDeltaSync(ctx context.Context, job model.Job, source itemSource) bool {
	ctx, done := spans.Open(ctx, "daemon.runDeltaSync")
	defer done(nil)

	manager.mu.Lock()
	codebase, codebaseFound := manager.codebases[job.CodebaseID]
	manager.mu.Unlock()
	if !codebaseFound {
		manager.routeToBootstrap(ctx, job.ID, bootstrapReasonDeltaCodebaseMissing)
		return false
	}

	streamingReindex := jobOperation(job.Operation) == jobOperationStreamingReindex
	plan := manager.planSyncDiff(ctx, job, codebase.ID, source)
	if plan.fallback {
		return false
	}
	if plan.handled {
		return true
	}
	// The source's absence policy decides what a large disappearance means here.
	// A code source deletes the missing files, guarded by the large-delete
	// quarantine. A conversation source retains them, because a transcript
	// missing from a push is almost always a transient disappearance, so the
	// removals are dropped from this run and the rows stay.
	switch source.absencePolicy() {
	case absenceDeleteGuarded:
		if signal, suspicious := assessDeltaDeleteWave(codebase, plan.diff, plan.seedSnapshot, job.CanonicalPath); suspicious {
			manager.updateJobQuarantined(ctx, job.ID, signal)
			return true
		}
	case absenceRetain:
		if retained := len(plan.diff.Removed); retained > 0 {
			slog.InfoContext(ctx, "converge.retain_absent", "component", "daemon", "subcomponent", "delta", "collection", codebase.CollectionName, "retained", retained)
		}
		plan.diff.Removed = nil
	}
	manager.setJobRunMode(job.ID, model.RunModeChanged)

	// Record the change breakdown so the status and job views can report the
	// magnitude of this reconcile while it runs. The per-file progress updates
	// touch only the embed counters, so these counts persist for the run.
	manager.setJobDeltaCounts(job.ID, len(plan.diff.Added), len(plan.diff.Modified), len(plan.diff.Removed), len(plan.currentSnapshot.Files))

	state := deltaState{
		plan:         plan,
		snapshotPath: manager.merklePath(codebase.ID),
		working:      make(map[string]string, len(plan.seedSnapshot.Files)),
		source:       source,
		semantic:     manager.semantic != nil && manager.semantic.Available(),
		staging:      false,
		reuse:        nil,
		chunkCounts:  &chunkCounters{processed: 0, reused: 0, embedded: 0, reuseVectorsLoaded: 0},
		seededReuse:  0,
		admission:    manager.admissionForJob(job),
	}
	maps.Copy(state.working, plan.seedSnapshot.Files)

	if outcome := manager.applyDeltaRemovals(ctx, job, state); outcome.fallback {
		return false
	} else if outcome.handled {
		return true
	}

	if codebase.Kind == model.CodebaseKindCode && len(plan.diff.Added) > 0 && state.semantic {
		reuse, seeded, _ := manager.resolveReuseSeed(ctx, job)
		state.reuse = reuse
		state.seededReuse = seeded
		state.chunkCounts.reuseVectorsLoaded = seeded
	}

	result, outcome := manager.applyDeltaChanges(ctx, job, state)
	if outcome.fallback {
		return false
	}
	if outcome.handled {
		return true
	}

	if streamingReindex && state.semantic {
		if outcome := manager.pruneAfterStreaming(ctx, job, plan.currentSnapshot); outcome.handled {
			return true
		}
	}

	result.FileHashes = state.working
	fileCount, chunkCount := manager.codebaseTotals(ctx, job.CanonicalPath, state.working, result.TotalChunks)
	result.IndexedFiles = fileCount
	result.TotalChunks = chunkCount
	var normalizeErr error
	result, normalizeErr = manager.normalizeDeltaTotalBytes(ctx, codebase, state, result)
	if normalizeErr != nil {
		manager.updateJobFailed(ctx, job.ID, normalizeErr)
		return true
	}
	manager.updateJobCompleted(ctx, job.ID, result)
	return true
}

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
	if codebase.Kind == model.CodebaseKindCode {
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

// runBootstrap builds a codebase index from scratch into a staging collection
// and swaps it onto the live name only once every file is embedded, so a
// search never observes a half-built first index. It embeds file by file and
// checkpoints after each, so an interrupted build resumes by skipping the
// files already embedded rather than restarting from the first one. Every
// operation that lacks a usable incremental delta routes here: a true first
// index, a forced rebuild, and a sync or streaming reindex whose live
// collection has gone missing.
func (manager *Manager) runBootstrap(ctx context.Context, job model.Job, source itemSource) {
	ctx, done := spans.Open(ctx, "daemon.runBootstrap")
	defer done(nil)

	manager.mu.Lock()
	codebase, codebaseFound := manager.codebases[job.CodebaseID]
	manager.mu.Unlock()
	if !codebaseFound {
		return
	}

	plan := manager.planBootstrap(ctx, job, codebase.ID, source)
	if plan.handled {
		return
	}
	runMode := model.RunModeFirstBuild
	if len(plan.seedSnapshot.Files) > 0 {
		runMode = model.RunModeResuming
	}
	if job.Forced && codebase.LastSuccessfulRun != nil {
		runMode = model.RunModeForcedReindex
	}
	manager.setJobRunMode(job.ID, runMode)

	state := deltaState{
		plan:         plan,
		snapshotPath: manager.stagingMerklePath(codebase.ID),
		working:      make(map[string]string, len(plan.currentSnapshot.Files)),
		source:       source,
		semantic:     manager.semantic != nil && manager.semantic.Available(),
		staging:      true,
		reuse:        nil,
		chunkCounts:  &chunkCounters{processed: 0, reused: 0, embedded: 0, reuseVectorsLoaded: 0},
		seededReuse:  0,
		admission:    manager.admissionForJob(job),
	}
	maps.Copy(state.working, plan.seedSnapshot.Files)

	reuse, seeded, descendants := manager.resolveReuseSeed(ctx, job)
	state.reuse = reuse
	state.seededReuse = seeded
	state.chunkCounts.reuseVectorsLoaded = seeded

	result, outcome := manager.applyDeltaChanges(ctx, job, state)
	if outcome.handled {
		return
	}
	if outcome.fallback {
		// A from-scratch build has no further fallback; treat one as a failure
		// so the job reaches a terminal state instead of stalling in running.
		manager.updateJobFailed(ctx, job.ID, errors.New("bootstrap build could not complete against the semantic backend"))
		return
	}

	if promote := manager.promoteBootstrap(ctx, job, state); promote.handled {
		return
	}
	if promote := manager.promoteStagingMerkle(ctx, job, state); promote.handled {
		return
	}

	manager.absorbDescendants(ctx, descendants)

	result.FileHashes = state.working
	fileCount, chunkCount := manager.codebaseTotals(ctx, job.CanonicalPath, state.working, result.TotalChunks)
	result.IndexedFiles = fileCount
	result.TotalChunks = chunkCount
	result.FileHashes = nil
	manager.updateJobCompleted(ctx, job.ID, result)
}

// resolveReuseSeed loads build-wide reuse vectors from indexed descendants and
// sibling worktrees that share the requested embedding model. It also returns
// the descendant candidates it scanned so a from-scratch build can absorb them
// without re-scanning the registry, keeping the reuse seed and the absorb list
// derived from one consistent snapshot.
func (manager *Manager) resolveReuseSeed(ctx context.Context, job model.Job) (map[string][]float32, int32, []model.Codebase) {
	descendants := manager.descendantReuseCandidates(job.CanonicalPath, job.Config)
	reuse := map[string][]float32{}
	if manager.semantic == nil || !manager.semantic.Available() {
		return reuse, 0, descendants
	}
	reuseCollections := collectionNamesOf(descendants)
	reuseCollections = append(reuseCollections, manager.worktreeSiblingReuseCollections(job.CanonicalPath, job.Config)...)
	if len(reuseCollections) == 0 {
		return reuse, 0, descendants
	}
	loadedReuse, err := manager.semantic.LoadReuseVectors(ctx, reuseCollections)
	if err != nil {
		slog.WarnContext(ctx, "load reuse vectors failed; building without the reuse seed", "job_id", job.ID, "err", err)
		return reuse, 0, descendants
	}
	seeded := safeInt32(len(loadedReuse))
	slog.InfoContext(ctx, "build.reuse_seeded", "job_id", job.ID, "reuse_collections", len(reuseCollections), "reuse_vectors", len(loadedReuse))
	return loadedReuse, seeded, descendants
}

// planBootstrap captures a fresh snapshot for a from-scratch build and decides
// whether a persisted checkpoint can seed a resume. Every discovered file is
// classified Added so the per-file loop embeds it unless the checkpoint
// already records its current hash. The checkpoint seeds a resume only when a
// staging collection still backs it; otherwise the build restarts from the
// first file and any stale staging is dropped first.
func (manager *Manager) planBootstrap(ctx context.Context, job model.Job, codebaseID string, source itemSource) deltaPlan {
	configDigest := job.Config.IgnoreDigest
	legacyDigest := manager.legacyDigestForCodebase(codebaseID)
	stagingSnapshotPath := manager.stagingMerklePath(codebaseID)
	seed := merkle.LoadSnapshotForConfig(stagingSnapshotPath, configDigest, legacyDigest)
	semanticReady := manager.semantic != nil && manager.semantic.Available()

	captured, err := source.capture(ctx)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			manager.updateJobCancelled(ctx, job.ID)
		} else {
			manager.updateJobFailed(ctx, job.ID, fmt.Errorf("capture bootstrap snapshot: %w", err))
		}
		return deltaPlan{
			diff:            merkle.Diff{Added: nil, Modified: nil, Removed: nil},
			currentSnapshot: merkle.Snapshot{ConfigDigest: "", Files: nil, Inodes: nil},
			seedSnapshot:    seed,
			configDigest:    configDigest,
			fallback:        false,
			handled:         true,
		}
	}

	if !manager.canResumeStaging(ctx, job.CanonicalPath, seed, semanticReady) {
		if semanticReady {
			if dropErr := manager.semantic.DropStaging(ctx, job.CanonicalPath); dropErr != nil {
				slog.WarnContext(ctx, "drop stale staging before bootstrap failed", "path", job.CanonicalPath, "err", dropErr)
			}
		}
		seed = merkle.Snapshot{ConfigDigest: configDigest, Files: nil, Inodes: nil}
		if removeErr := store.RemoveFile(stagingSnapshotPath); removeErr != nil {
			slog.WarnContext(ctx, "remove stale bootstrap checkpoint failed", "path", stagingSnapshotPath, "err", removeErr)
		}
	}

	addedFiles := make([]string, 0, len(captured.Files))
	for relativePath := range captured.Files {
		addedFiles = append(addedFiles, relativePath)
	}
	sort.Strings(addedFiles)
	return deltaPlan{
		diff:            merkle.Diff{Added: addedFiles, Modified: nil, Removed: nil},
		currentSnapshot: captured,
		seedSnapshot:    seed,
		configDigest:    configDigest,
		fallback:        false,
		handled:         false,
	}
}

func (manager *Manager) promoteStagingMerkle(ctx context.Context, job model.Job, state deltaState) deltaOutcome {
	if !state.staging {
		return deltaOutcome{fallback: false, handled: false, progressed: false}
	}
	if state.snapshotPath == "" {
		return deltaOutcome{fallback: false, handled: false, progressed: false}
	}
	livePath := manager.merklePath(job.CodebaseID)
	if err := os.Rename(state.snapshotPath, livePath); err != nil {
		if errors.Is(err, os.ErrNotExist) && len(state.working) == 0 {
			return deltaOutcome{fallback: false, handled: false, progressed: false}
		}
		manager.cleanupHaltedStaging(ctx, job, state)
		manager.updateJobFailed(ctx, job.ID, fmt.Errorf("promote staging Merkle snapshot: %w", err))
		return deltaOutcome{fallback: false, handled: true, progressed: false}
	}
	slog.InfoContext(ctx, "promoted staging Merkle snapshot to live", "component", "daemon", "subcomponent", "delta", "codebase_id", job.CodebaseID, "path", livePath)
	return deltaOutcome{fallback: false, handled: false, progressed: false}
}

// canResumeStaging reports whether a persisted checkpoint can seed a resumed
// build. A checkpoint with no files cannot. When the semantic backend is
// configured the staging collection must still exist, because that is where
// the embedded vectors for the checkpointed files live; without it the
// checkpoint describes work whose vectors were lost, so the build restarts.
// When the backend is unavailable the checkpoint is the only state and is
// trusted on its own.
func (manager *Manager) canResumeStaging(ctx context.Context, canonicalPath string, seed merkle.Snapshot, semanticReady bool) bool {
	if len(seed.Files) == 0 {
		return false
	}
	if !semanticReady {
		return true
	}
	hasStaging, err := manager.semantic.HasStaging(ctx, canonicalPath)
	if err != nil {
		slog.WarnContext(ctx, "check staging for resume failed; restarting build", "path", canonicalPath, "err", err)
		return false
	}
	return hasStaging
}

// promoteBootstrap swaps the freshly built staging collection onto the live
// name. When no file produced chunks there is no staging collection to
// promote, which is a successful empty index rather than an error. A handled
// outcome means promoteBootstrap already set a terminal job state.
func (manager *Manager) promoteBootstrap(ctx context.Context, job model.Job, state deltaState) deltaOutcome {
	if !state.semantic {
		return deltaOutcome{fallback: false, handled: false, progressed: false}
	}
	hasStaging, err := manager.semantic.HasStaging(ctx, job.CanonicalPath)
	if err != nil {
		manager.updateJobFailed(ctx, job.ID, err)
		return deltaOutcome{fallback: false, handled: true, progressed: false}
	}
	if !hasStaging {
		return deltaOutcome{fallback: false, handled: false, progressed: false}
	}
	if err := manager.semantic.PromoteStaging(ctx, job.CanonicalPath); err != nil {
		manager.cleanupHaltedStaging(ctx, job, state)
		if errors.Is(err, context.Canceled) {
			manager.updateJobCancelled(ctx, job.ID)
		} else {
			manager.updateJobFailed(ctx, job.ID, fmt.Errorf("promote staging collection: %w", err))
		}
		return deltaOutcome{fallback: false, handled: true, progressed: false}
	}
	return deltaOutcome{fallback: false, handled: false, progressed: false}
}

func (manager *Manager) applyDeltaRemovals(ctx context.Context, job model.Job, state deltaState) deltaOutcome {
	removed := state.plan.diff.Removed
	if len(removed) == 0 || !state.semantic {
		return deltaOutcome{fallback: false, handled: false, progressed: false}
	}
	if outcome := manager.applyReindexForState(ctx, job, state, nil, state.source.removalFor(removed), "delta removal"); outcome.fallback || outcome.handled {
		return outcome
	}
	for _, path := range removed {
		delete(state.working, path)
	}
	manager.writeCheckpoint(ctx, state, "removals")
	return deltaOutcome{fallback: false, handled: false, progressed: false}
}

func (manager *Manager) applyDeltaChanges(ctx context.Context, job model.Job, state deltaState) (indexer.Result, deltaOutcome) {
	changed := make([]string, 0, len(state.plan.diff.Added)+len(state.plan.diff.Modified))
	changed = append(changed, state.plan.diff.Added...)
	changed = append(changed, state.plan.diff.Modified...)

	totalChanged := len(changed)
	totalFiles := safeInt32(totalChanged)
	result := indexer.Result{
		IndexedFiles:      0,
		TotalChunks:       0,
		TotalBytes:        0,
		Chunks:            make([]model.StoredChunk, 0),
		FileHashes:        nil,
		SkippedFiles:      []string{},
		SkippedOversize:   0,
		SkippedUnreadable: 0,
		SkippedPending:    0,
	}
	for index, relativePath := range changed {
		if err := ctx.Err(); err != nil {
			manager.updateJobCancelled(ctx, job.ID)
			return result, deltaOutcome{fallback: false, handled: true, progressed: false}
		}
		if seedHash, present := state.plan.seedSnapshot.Files[relativePath]; present && seedHash == state.plan.currentSnapshot.Files[relativePath] {
			state.working[relativePath] = seedHash
			processed, reused, embedded, loaded := state.chunkSplit()
			manager.reportDeltaProgress(job.ID, safeInt32(index+1), totalChanged, totalFiles, result, processed, reused, embedded, loaded, state.source.unit())
			continue
		}
		outcome := manager.handleChangedFile(ctx, job, state, relativePath, &result)
		if outcome.fallback || outcome.handled {
			return result, outcome
		}
		// A skipped item changes nothing in the working set, so rewriting the
		// snapshot for it would be one full-file disk write per skipped item; a
		// job that skips a thousand undelivered conversations checkpoints only
		// after the items that actually embedded or removed.
		if outcome.progressed {
			manager.writeCheckpoint(ctx, state, relativePath)
		}
		processed, reused, embedded, loaded := state.chunkSplit()
		manager.reportDeltaProgress(job.ID, safeInt32(index+1), totalChanged, totalFiles, result, processed, reused, embedded, loaded, state.source.unit())
	}
	return result, deltaOutcome{fallback: false, handled: false, progressed: false}
}

func (manager *Manager) handleChangedFile(ctx context.Context, job model.Job, state deltaState, relativePath string, result *indexer.Result) deltaOutcome {
	fileResult, err := state.source.indexOne(ctx, relativePath)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			manager.updateJobCancelled(ctx, job.ID)
		} else {
			manager.updateJobFailed(ctx, job.ID, err)
		}
		return deltaOutcome{fallback: false, handled: true, progressed: false}
	}
	if fileResult.Removed {
		slog.InfoContext(ctx, "converge.remove", "component", "daemon", "subcomponent", "delta", "path", relativePath, "semantic", state.semantic)
		if state.semantic {
			if outcome := manager.applyReindexForState(ctx, job, state, nil, state.source.removalFor([]string{relativePath}), "per-file removal"); outcome.fallback || outcome.handled {
				return outcome
			}
		}
		delete(state.working, relativePath)
		return deltaOutcome{fallback: false, handled: false, progressed: true}
	}
	if fileResult.Skipped {
		switch fileResult.SkipReason {
		case indexer.SkipOversize:
			result.SkippedFiles = append(result.SkippedFiles, relativePath)
			result.SkippedOversize++
		case indexer.SkipPending:
			// Transient, not a permanent skip, so it stays out of SkippedFiles (the
			// ready-view's non-UTF-8 summary) and counts only as pending.
			result.SkippedPending++
		case indexer.SkipUnreadable, indexer.SkipNone:
			result.SkippedFiles = append(result.SkippedFiles, relativePath)
			result.SkippedUnreadable++
		}
		return deltaOutcome{fallback: false, handled: false, progressed: false}
	}
	if state.semantic {
		reuse, loaded := manager.itemReuse(ctx, state, relativePath)
		state.reuse = reuse
		if state.chunkCounts != nil {
			state.chunkCounts.reuseVectorsLoaded += loaded
		}
		if outcome := manager.applyReindexForState(ctx, job, state, fileResult.Chunks, state.source.removalFor([]string{relativePath}), "per-file reindex"); outcome.fallback || outcome.handled {
			return outcome
		}
	}
	state.working[relativePath] = fileResult.FileHash
	result.Chunks = append(result.Chunks, fileResult.Chunks...)
	result.TotalChunks += safeInt32(len(fileResult.Chunks))
	result.TotalBytes += storedChunkBytes(fileResult.Chunks)
	result.IndexedFiles++
	return deltaOutcome{fallback: false, handled: false, progressed: true}
}

// itemReuse loads one item's own stored vectors before the reindex deletes
// them, so chunks whose content is unchanged take their vector from the store
// instead of the embedder. It returns the build-wide reuse map unchanged when
// the source has no per-item reuse, and on a failed load it logs and falls
// back to that map so the item embeds every chunk rather than failing.
func (manager *Manager) itemReuse(ctx context.Context, state deltaState, relativePath string) (map[string][]float32, int32) {
	// Per-item same-collection reuse is an incremental live-collection feature.
	// A staging/bootstrap build already has its own build-wide reuse sources, and
	// probing the live collection file-by-file on a true first build would be
	// pure overhead when that collection does not exist yet.
	if state.staging {
		return state.reuse, 0
	}
	source := state.source.reuseSource(relativePath)
	if source.Scope == itemReuseScopeNone {
		return state.reuse, 0
	}
	var itemReuse map[string][]float32
	var err error
	switch source.Scope {
	case itemReuseScopeNone:
		return state.reuse, 0
	case itemReuseScopePath:
		itemReuse, err = manager.semantic.LoadReuseVectorsForPath(ctx, source.CollectionName, source.RelativePath)
	case itemReuseScopePrefix:
		itemReuse, err = manager.semantic.LoadReuseVectorsForPrefix(ctx, source.CollectionName, source.RelativePath)
	default:
		return state.reuse, 0
	}
	if err != nil {
		slog.WarnContext(ctx, "load item reuse vectors failed; embedding every chunk", "path", relativePath, "collection", source.CollectionName, "scope", source.Scope, "err", err)
		return state.reuse, 0
	}
	return mergedReuse(state.reuse, itemReuse), safeInt32(len(itemReuse))
}

// mergedReuse overlays an item's own reuse vectors on any build-wide reuse map
// without mutating either input. With no build-wide map the item map is used
// as-is, which is the conversation delta case.
func mergedReuse(base map[string][]float32, item map[string][]float32) map[string][]float32 {
	if len(base) == 0 {
		return item
	}
	merged := make(map[string][]float32, len(base)+len(item))
	maps.Copy(merged, base)
	maps.Copy(merged, item)
	return merged
}

func (manager *Manager) classifyReindexErr(ctx context.Context, job model.Job, err error, phase string) deltaOutcome {
	switch {
	case errors.Is(err, semantic.ErrCollectionMissing):
		manager.routeToBootstrap(ctx, job.ID, bootstrapReasonDeltaCollectionMissing)
		slog.WarnContext(ctx, "semantic collection missing; falling back to full reindex", "job_id", job.ID, "phase", phase)
		return deltaOutcome{fallback: true, handled: false, progressed: false}
	case errors.Is(err, context.Canceled):
		manager.updateJobCancelled(ctx, job.ID)
		return deltaOutcome{fallback: false, handled: true, progressed: false}
	default:
		manager.updateJobFailed(ctx, job.ID, err)
		return deltaOutcome{fallback: false, handled: true, progressed: false}
	}
}

func bootstrapReasonForEmptyDiffFallback(evidence collectionEvidence) bootstrapReason {
	if evidence.presence == collectionPresencePresent && evidence.rowsKnown && evidence.rows == 0 {
		return bootstrapReasonEmptyDiffCollectionEmpty
	}
	return bootstrapReasonEmptyDiffCollectionMissing
}

func (manager *Manager) writeCheckpoint(ctx context.Context, state deltaState, label string) {
	snapshot := merkle.Snapshot{ConfigDigest: state.plan.configDigest, Files: state.working, Inodes: nil}
	if err := merkle.WriteSnapshot(state.snapshotPath, snapshot); err != nil {
		slog.ErrorContext(ctx, "checkpoint write failed", "path", state.snapshotPath, "label", label, "err", err)
	}
}

// phaseReindexingChanged is the job phase while the per-file embed loop runs,
// shared by every progress update from that loop.
const phaseReindexingChanged = "Reindexing changed files..."

// chunkSplit returns the run's processed, reused, embedded, and
// reuse-vectors-loaded counts for progress reporting. reused is the chunks
// actually served from the reuse pool this run, so it never exceeds processed.
// The seeded pool size is reported separately as the reuse-vectors-loaded
// figure, which the building view can show immediately for context without
// inflating the reused count.
func (state deltaState) chunkSplit() (int32, int32, int32, int32) {
	if state.chunkCounts == nil {
		return 0, 0, 0, state.seededReuse
	}
	loaded := max(state.chunkCounts.reuseVectorsLoaded, state.seededReuse)
	return state.chunkCounts.processed, state.chunkCounts.reused, state.chunkCounts.embedded, loaded
}

// reportDeltaProgress publishes one progress update from the per-file embed
// loop. processed is the number of files the loop has finished. An initial call
// with processed=0 establishes the embedding phase and the to-process total, so
// the status shows the indexing view before the first (possibly slow) embed
// rather than a stalled "Preparing". reused and embedded are the run's
// accumulated chunk split; ChunksGenerated carries the embedded-this-run total
// so a surface can show total = reused + embedded.
func (manager *Manager) reportDeltaProgress(jobID string, processed int32, totalChanged int, totalFiles int32, result indexer.Result, chunksProcessed int32, reused int32, embedded int32, reuseVectorsLoaded int32, unit string) {
	manager.updateJobProgress(jobID, indexer.Progress{
		Phase:                  phaseReindexingChanged,
		OverallPercent:         float64(processed) / float64(maxInt(totalChanged, 1)) * 100,
		FilesTotal:             totalFiles,
		FilesProcessed:         processed,
		FilesEmbedded:          result.IndexedFiles,
		FilesSkippedOversize:   result.SkippedOversize,
		FilesSkippedUnreadable: result.SkippedUnreadable,
		FilesPending:           result.SkippedPending,
		ChunksProcessed:        chunksProcessed,
		ChunksReused:           reused,
		ChunksEmbedded:         embedded,
		ChunksGenerated:        embedded,
		ReuseVectorsLoaded:     reuseVectorsLoaded,
	}, unit)
}

func (manager *Manager) pruneAfterStreaming(ctx context.Context, job model.Job, currentSnapshot merkle.Snapshot) deltaOutcome {
	currentPaths := make([]string, 0, len(currentSnapshot.Files))
	for relativePath := range currentSnapshot.Files {
		currentPaths = append(currentPaths, relativePath)
	}
	sort.Strings(currentPaths)
	if err := manager.semantic.PruneToCurrent(ctx, job.CanonicalPath, currentPaths); err != nil {
		switch {
		case errors.Is(err, context.Canceled):
			manager.updateJobCancelled(ctx, job.ID)
			return deltaOutcome{fallback: false, handled: true, progressed: false}
		case errors.Is(err, semantic.ErrCollectionMissing):
			slog.WarnContext(ctx, "semantic collection missing during streaming prune", "job_id", job.ID)
		default:
			manager.updateJobFailed(ctx, job.ID, err)
			return deltaOutcome{fallback: false, handled: true, progressed: false}
		}
	}
	return deltaOutcome{fallback: false, handled: false, progressed: false}
}

func maxInt(a int, b int) int {
	if a > b {
		return a
	}
	return b
}

// safeInt32 clamps int to int32 for protobuf-bound progress fields.
func safeInt32(value int) int32 {
	if value > math.MaxInt32 {
		return math.MaxInt32
	}
	if value < math.MinInt32 {
		return math.MinInt32
	}
	return int32(value)
}

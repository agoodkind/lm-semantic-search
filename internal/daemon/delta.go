package daemon

import (
	"context"
	"log/slog"
	"maps"
	"math"

	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func effectiveRemoval(
	source itemSource,
	fileResult indexer.OneFileResult,
	relativePath string,
) semantic.Removal {
	if fileResult.RemovalOverride {
		return semantic.Removal{Paths: fileResult.RemovalPaths, ItemColumn: "", ItemIDs: nil}
	}

	return source.removalFor([]string{relativePath})
}

// mergedReuse overlays an item's own reuse vectors on any build-wide reuse map
// without mutating either input. With no build-wide map the item map is used
// as-is, which is the conversation delta case.
func mergedReuse(
	base map[string]model.ReuseVector,
	item map[string]model.ReuseVector,
) map[string]model.ReuseVector {
	if len(base) == 0 {
		return item
	}

	merged := make(map[string]model.ReuseVector, len(base)+len(item))
	maps.Copy(merged, base)
	maps.Copy(merged, item)

	return merged
}

func (manager *Manager) writeCheckpoint(ctx context.Context, state deltaState, label string) {
	snapshot := checkpointSnapshot(
		state.plan.configDigest,
		state.working,
		state.plan.currentSnapshot.Inodes,
		state.inodeTrackingDisabled,
	)
	if err := merkle.WriteSnapshot(state.snapshotPath, snapshot); err != nil {
		slog.ErrorContext(
			ctx,
			"checkpoint write failed",
			"path",
			state.snapshotPath,
			"label",
			label,
			"err",
			err,
		)

		return
	}
}

// checkpointSnapshot keeps captured inode entries in full build and delta sync
// snapshots. The converge operation matches those entries in the Merkle
// snapshot to find the source path of a renamed file.
func checkpointSnapshot(
	configDigest string,
	files map[string]string,
	capturedInodes map[string]merkle.InodeRef,
	inodeTrackingDisabled bool,
) merkle.Snapshot {
	snapshot := merkle.Snapshot{ConfigDigest: configDigest, Files: files, Inodes: nil}
	if inodeTrackingDisabled {
		return snapshot
	}

	for relativePath := range files {
		if ref, found := capturedInodes[relativePath]; found {
			snapshot.RecordInode(relativePath, ref)
		}
	}

	return snapshot
}

// chunkSplit returns the run's processed, reused, embedded, dropped, and
// reuse-vectors-loaded counts for progress reporting. reused is the chunks
// actually served from the reuse pool this run, so it never exceeds processed.
// The seeded pool size is reported separately as the reuse-vectors-loaded
// figure, which the building view can show immediately for context without
// inflating the reused count.
func (state deltaState) chunkSplit() (int32, int32, int32, int32, int32) {
	if state.chunkCounts == nil {
		return 0, 0, 0, 0, state.seededReuse
	}

	loaded := max(state.chunkCounts.reuseVectorsLoaded, state.seededReuse)

	return state.chunkCounts.processed,
		state.chunkCounts.reused,
		state.chunkCounts.embedded,
		state.chunkCounts.dropped,
		loaded
}

// reportDeltaProgress publishes one progress update from the per-file embed
// loop. processed is the number of files the loop has finished. An initial call
// with processed=0 establishes the embedding phase and the to-process total, so
// the status shows the indexing view before the first (possibly slow) embed
// rather than a stalled "Preparing". reused and embedded are the run's
// accumulated chunk split; ChunksGenerated carries the embedded-this-run total
// so a surface can show total = reused + embedded.
func (manager *Manager) reportDeltaProgress(
	jobID string,
	processed int32,
	totalChanged int,
	totalFiles int32,
	result indexer.Result,
	chunksProcessed int32,
	reused int32,
	embedded int32,
	dropped int32,
	reuseVectorsLoaded int32,
	unit string,
) {
	const percentScale = 100

	overallPercent := float64(processed) / float64(maxInt(totalChanged, 1)) * percentScale
	manager.updateJobProgress(
		jobID,
		indexer.Progress{
			Phase:                  phaseReindexingChanged,
			OverallPercent:         overallPercent,
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
			ChunksDropped:          dropped,
			ReuseVectorsLoaded:     reuseVectorsLoaded,
		},
		unit,
	)
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

func (manager *Manager) completeEmptySync(
	ctx context.Context,
	job model.Job,
	codebaseID string,
	captured merkle.Snapshot,
	configDigest string,
	evidence collectionEvidence,
) *graphIndexTask {
	snapshotHash := snapshotHashForGraph(captured, configDigest)
	fileCount, chunkCount := manager.codebaseTotals(ctx, job.CanonicalPath, captured.Files, 0)
	var totalBytes int64
	manager.mu.Lock()
	if codebase, found := manager.codebases[codebaseID]; found &&
		codebase.LastSuccessfulRun != nil {
		totalBytes = codebase.LastSuccessfulRun.TotalBytes
	}

	manager.mu.Unlock()
	emptyDiffResult := indexer.Result{
		IndexedFiles:      fileCount,
		TotalChunks:       chunkCount,
		TotalBytes:        totalBytes,
		Chunks:            nil,
		FileHashes:        captured.Files,
		FileInodes:        captured.Inodes,
		SkippedFiles:      nil,
		SkippedOversize:   0,
		SkippedUnreadable: 0,
		SkippedPending:    0,
	}
	var graphTask *graphIndexTask
	if manager.shouldReconcileGraph(codebaseID, snapshotHash, evidence.presence) {
		graphTask = newGraphIndexTask(
			codebaseID,
			job.CanonicalPath,
			snapshotHash,
			func(completeCtx context.Context) {
				manager.updateJobCompleted(completeCtx, job.ID, emptyDiffResult)
			},
		)
	}

	if graphTask == nil {
		manager.updateJobCompleted(ctx, job.ID, emptyDiffResult)
	}

	return graphTask
}

func (manager *Manager) newDeltaSyncState(
	ctx context.Context,
	job model.Job,
	codebase model.Codebase,
	source itemSource,
	plan deltaPlan,
	semanticReady bool,
) deltaState {
	state := deltaState{
		plan:             plan,
		snapshotPath:     manager.merklePath(codebase.ID),
		working:          make(map[string]string, len(plan.seedSnapshot.Files)),
		source:           source,
		semantic:         semanticReady,
		itemReuseEnabled: manager.resolveItemReusePolicy(ctx, job, false, semanticReady),
		staging:          false,
		reuse:            nil,
		chunkCounts: &chunkCounters{
			processed:          0,
			reused:             0,
			embedded:           0,
			dropped:            0,
			reuseVectorsLoaded: 0,
		},
		seededReuse:           0,
		admission:             manager.admissionForJob(job),
		forced:                forcedItemsSet(plan.forced),
		inodeTrackingDisabled: codebase.InodeTrackingDisabled,
		renamedPaths:          make(map[string]string),
	}
	maps.Copy(state.working, plan.seedSnapshot.Files)

	return state
}

func (manager *Manager) newBootstrapState(
	ctx context.Context,
	job model.Job,
	codebase model.Codebase,
	source itemSource,
	plan deltaPlan,
	semanticReady bool,
) deltaState {
	state := deltaState{
		plan:             plan,
		snapshotPath:     manager.stagingMerklePath(codebase.ID),
		working:          make(map[string]string, len(plan.currentSnapshot.Files)),
		source:           source,
		semantic:         semanticReady,
		itemReuseEnabled: manager.resolveItemReusePolicy(ctx, job, true, semanticReady),
		staging:          true,
		reuse:            nil,
		chunkCounts: &chunkCounters{
			processed:          0,
			reused:             0,
			embedded:           0,
			dropped:            0,
			reuseVectorsLoaded: 0,
		},
		seededReuse: 0,
		admission:   manager.admissionForJob(job),
		// A bootstrap re-embeds every item, so nothing needs per-item forcing.
		forced:                nil,
		inodeTrackingDisabled: codebase.InodeTrackingDisabled,
		renamedPaths:          nil,
	}
	maps.Copy(state.working, plan.seedSnapshot.Files)

	return state
}

func (manager *Manager) completeDeltaSync(
	ctx context.Context,
	job model.Job,
	codebase model.Codebase,
	source itemSource,
	plan deltaPlan,
	state deltaState,
	result indexer.Result,
) (bool, *graphIndexTask) {
	result.FileHashes = state.working
	result.FileInodes = plan.currentSnapshot.Inodes
	fileCount, chunkCount := manager.codebaseTotals(
		ctx,
		job.CanonicalPath,
		state.working,
		result.TotalChunks,
	)
	result.IndexedFiles = fileCount
	result.TotalChunks = chunkCount
	var normalizeErr error
	result, normalizeErr = manager.normalizeDeltaTotalBytes(ctx, codebase, state, result)
	if normalizeErr != nil {
		manager.updateJobFailed(ctx, job.ID, normalizeErr)

		return true, nil
	}

	var graphTask *graphIndexTask
	if source.producesGraph() {
		// The graph task is stamped with the hash of the snapshot this run
		// committed (the working set), not the captured one: skipped, pending,
		// or unreadable paths can leave the checkpoint behind the capture, and
		// reconciliation compares against the committed checkpoint.
		committedSnapshot := merkle.Snapshot{
			ConfigDigest: plan.configDigest,
			Files:        state.working,
			Inodes:       nil,
		}
		graphTask = newGraphIndexTask(
			codebase.ID,
			job.CanonicalPath,
			snapshotHashForGraph(committedSnapshot, plan.configDigest),
			func(completeCtx context.Context) {
				manager.updateJobCompleted(completeCtx, job.ID, result)
			},
		)
	}

	if graphTask == nil {
		manager.updateJobCompleted(ctx, job.ID, result)
	}

	return true, graphTask
}

func (manager *Manager) completeBootstrap(
	ctx context.Context,
	job model.Job,
	source itemSource,
	plan deltaPlan,
	state deltaState,
	result indexer.Result,
) *graphIndexTask {
	var graphTask *graphIndexTask
	if source.producesGraph() {
		graphTask = newGraphIndexTask(
			job.CodebaseID,
			job.CanonicalPath,
			snapshotHashForGraph(plan.currentSnapshot, plan.configDigest),
			func(completeCtx context.Context) {
				manager.updateJobCompleted(completeCtx, job.ID, result)
			},
		)
	}

	result.FileHashes = state.working
	fileCount, chunkCount := manager.codebaseTotals(
		ctx,
		job.CanonicalPath,
		state.working,
		result.TotalChunks,
	)
	result.IndexedFiles = fileCount
	result.TotalChunks = chunkCount
	result.FileHashes = nil
	if graphTask == nil {
		manager.updateJobCompleted(ctx, job.ID, result)
	}

	return graphTask
}

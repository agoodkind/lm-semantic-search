package daemon

import (
	"context"
	"fmt"
	"log/slog"

	"goodkind.io/gklog/correlation"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/metrics"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/store"
)

func (manager *Manager) updateJobRunning(job model.Job) {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	currentJob, found := manager.jobs[job.ID]
	if !found {
		return
	}
	now := clock.Now()
	currentJob.State = model.JobStateRunning
	currentJob.UpdatedAt = now
	currentJob.Progress.Phase = "Preparing and scanning files..."
	currentJob.Progress.LastEventAt = now
	currentJob.Progress.HeartbeatAt = now
	currentJob.Progress.OverallPercent = 0
	manager.jobs[currentJob.ID] = currentJob
	// A first build was pending while its job sat queued; now that the job is
	// running, the codebase is actively indexing. A rebuild was already indexing.
	// The flip is in-memory so live status reads see it at once; the next registry
	// save on completion persists it, and an interrupted run re-queues on resume.
	if codebase, ok := manager.codebases[currentJob.CodebaseID]; ok && codebase.Status == model.CodebaseStatusPending {
		codebase.Status = model.CodebaseStatusIndexing
		codebase.UpdatedAt = now
		manager.codebases[codebase.ID] = codebase
	}
	_ = manager.appendJobLocked("job_running", currentJob)
}

func (manager *Manager) updateJobProgress(jobID string, progress indexer.Progress, unit string) {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	job, found := manager.jobs[jobID]
	if !found {
		return
	}
	delete(manager.conversationJobs, jobID)
	if job.State != model.JobStateQueued && job.State != model.JobStateRunning && job.State != model.JobStateCancelling {
		return
	}

	now := clock.Now()
	job.State = model.JobStateRunning
	job.UpdatedAt = now
	job.Progress.Phase = progress.Phase
	job.Progress.OverallPercent = progress.OverallPercent
	if unit != "" {
		job.Progress.Unit = unit
		if unit == "document" {
			job.Progress.ScopeUnit = "conversation"
		}
	}
	job.Progress.FilesTotal = progress.FilesTotal
	job.Progress.FilesProcessed = progress.FilesProcessed
	job.Progress.FilesEmbedded = progress.FilesEmbedded
	job.Progress.FilesSkippedOversize = progress.FilesSkippedOversize
	job.Progress.FilesSkippedUnreadable = progress.FilesSkippedUnreadable
	job.Progress.FilesPending = progress.FilesPending
	job.Progress.ChunksProcessed = progress.ChunksProcessed
	job.Progress.ChunksReused = progress.ChunksReused
	job.Progress.ChunksEmbedded = progress.ChunksEmbedded
	job.Progress.ChunksGenerated = progress.ChunksGenerated
	job.Progress.ReuseVectorsLoaded = progress.ReuseVectorsLoaded
	job.Progress.LastEventAt = now
	job.Progress.HeartbeatAt = now
	manager.jobs[jobID] = job
	manager.updateCodebaseLiveTotalsLocked(job)

	// A progress update that embedded a file proves the embedding pipeline is
	// reachable right now, so clear a degraded banner as soon as embedding
	// resumes rather than waiting for the job to complete. Without this a long
	// recovering build would keep showing a stale "paused" banner while it is
	// visibly making progress. A reuse-only or no-op update embeds nothing, so it
	// leaves the banner untouched.
	if progress.FilesEmbedded > 0 {
		manager.noteDependencyHealthyLocked()
	}
}

func (manager *Manager) updateCodebaseLiveTotalsLocked(job model.Job) {
	codebase, found := manager.codebases[job.CodebaseID]
	if !found {
		return
	}
	changed := false
	lastRun := codebase.LastSuccessfulRun
	if lastRun != nil && codebase.LiveFileTotal == 0 {
		codebase.LiveFileTotal = lastRun.IndexedFiles
		changed = true
	}
	if lastRun != nil && codebase.LiveChunkTotal == 0 {
		codebase.LiveChunkTotal = lastRun.TotalChunks
		changed = true
	}
	liveFiles := job.Progress.FilesInCodebase
	if liveFiles == 0 {
		liveFiles = job.Progress.FilesTotal
	}
	if liveFiles > 0 && codebase.LiveFileTotal != liveFiles {
		codebase.LiveFileTotal = liveFiles
		changed = true
	}
	liveChunks := max(job.Progress.ChunksTotal, max(job.Progress.ChunksProcessed, job.Progress.ChunksReused+job.Progress.ChunksEmbedded))
	if liveChunks > codebase.LiveChunkTotal {
		codebase.LiveChunkTotal = liveChunks
		changed = true
	}
	if !changed {
		return
	}
	manager.codebases[job.CodebaseID] = codebase
}

// setJobDeltaCounts records how many files a delta sync will add, modify, and
// remove, plus the codebase file total, so the status and job views can report
// the magnitude of a reconcile (for example after a large merge). The counts
// are set once when the diff is known; updateJobProgress sets only the embed
// counters, so these survive the per-file progress updates.
func (manager *Manager) setJobDeltaCounts(jobID string, added int, modified int, removed int, filesInCodebase int) {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	job, found := manager.jobs[jobID]
	if !found {
		return
	}
	job.Progress.FilesAdded = safeInt32(added)
	job.Progress.FilesModified = safeInt32(modified)
	job.Progress.FilesRemoved = safeInt32(removed)
	job.Progress.FilesInCodebase = safeInt32(filesInCodebase)
	manager.jobs[jobID] = job
}

// setJobRunMode records the kind of pass a run is making, decided once when
// the plan is chosen, so surfaces can label denominators and name a resume.
func (manager *Manager) setJobRunMode(jobID string, runMode string) {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	job, found := manager.jobs[jobID]
	if !found {
		return
	}
	job.Progress.RunMode = runMode
	manager.jobs[jobID] = job
}

func (manager *Manager) updateJobCompleted(ctx context.Context, jobID string, result indexer.Result) {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	job, found := manager.jobs[jobID]
	if !found {
		return
	}

	now := clock.Now()
	metrics.JobCompleted()
	// Clear the degraded banner only when this run actually embedded a file this
	// run, which proves the dependency is reachable, matching the embed-progress
	// clear path. FilesEmbedded is the per-run embedded count the embed loop
	// recorded, zero for an empty-diff no-op or a skipped-only sync. Gating on
	// result.IndexedFiles instead would clear the banner for a no-op completion,
	// because that path reports the whole codebase file count without touching the
	// store, and could wipe a banner raised by a real outage on another codebase.
	if job.Progress.FilesEmbedded > 0 {
		manager.noteDependencyHealthyLocked()
	}
	job.State = model.JobStateCompleted
	job.UpdatedAt = now
	job.CompletedAt = &now
	job.Progress.Phase = "completed"
	job.Progress.OverallPercent = 100
	job.Progress.FilesProcessed = result.IndexedFiles
	job.Progress.FilesTotal = result.IndexedFiles
	// ChunksTotal is the codebase's collection size. ChunksProcessed, ChunksReused,
	// and ChunksEmbedded stay at the real per-run values the embed loop recorded,
	// which are zero for a completion that embedded nothing (an empty-diff no-op or
	// a skipped-only sync). Reporting the collection total as embedded here would
	// make a no-embed run look like a full re-embed.
	job.Progress.ChunksTotal = result.TotalChunks
	job.Progress.ChunksGenerated = job.Progress.ChunksEmbedded
	job.Progress.LastEventAt = now
	job.Progress.HeartbeatAt = now
	if err := manager.appendJobLocked("job_completed", job); err != nil {
		slog.ErrorContext(ctx, "append completed job event failed", "job_id", jobID, "err", err)
	}

	codebase, found := manager.codebases[job.CodebaseID]
	if !found {
		return
	}
	delete(manager.failedBuildRetries, codebase.ID)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.ActiveJobID = ""
	codebase.Quarantine = nil
	codebase.LastSuccessfulRun = &model.IndexRunSummary{
		IndexedFiles: result.IndexedFiles,
		TotalChunks:  result.TotalChunks,
		Status:       "completed",
		CompletedAt:  now,
		SkippedFiles: result.SkippedFiles,
	}
	codebase.LiveFileTotal = result.IndexedFiles
	codebase.LiveChunkTotal = result.TotalChunks
	codebase.MerkleSnapshotPath = manager.merklePath(codebase.ID)
	codebase.UpdatedAt = now
	manager.codebases[codebase.ID] = codebase
	// A conversation collection's literal-fallback cache must stay complete across
	// an incremental ingest, so a delta merges into the prior cache rather than
	// overwriting it with only the changed conversations. A code codebase keeps
	// the existing whole-result write.
	if codebase.Kind == model.CodebaseKindDocument {
		if err := manager.mergeConversationChunkCache(ctx, codebase.ID, result.Chunks, result.FileHashes); err != nil {
			slog.ErrorContext(ctx, "merge conversation chunk cache failed", "job_id", jobID, "err", err)
		}
	} else {
		chunkPath := manager.chunkPath(codebase.ID)
		if err := store.WriteChunks(chunkPath, result.Chunks); err != nil {
			slog.ErrorContext(ctx, "write chunk cache failed", "job_id", jobID, "err", err)
		}
	}
	if len(result.FileHashes) != 0 {
		snapshot := merkle.Snapshot{ConfigDigest: codebase.EffectiveConfig.IgnoreDigest, Files: result.FileHashes, Inodes: nil}
		if err := merkle.WriteSnapshot(codebase.MerkleSnapshotPath, snapshot); err != nil {
			slog.ErrorContext(ctx, "write Merkle snapshot failed", "job_id", jobID, "err", err)
		}
	}
	if err := manager.saveLocked(); err != nil {
		slog.ErrorContext(ctx, "write registry after completed job failed", "job_id", jobID, "err", err)
	}
}

func (manager *Manager) updateJobFailed(ctx context.Context, jobID string, runErr error) {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	job, found := manager.jobs[jobID]
	if !found {
		return
	}
	delete(manager.conversationJobs, jobID)

	traceID := string(correlation.FromContext(ctx).TraceID)
	now := clock.Now()
	metrics.JobFailed()
	// A self-healing failure marks the job retryable. A shared-infrastructure
	// failure (the embedding pipeline or the vector store) never changes the
	// codebase's durable state, because it affects every codebase the same way
	// and is surfaced once by the daemon health banner; only a fault local to
	// this codebase becomes terminal. The persisted message is the safe class
	// message, never the wrapped cause, which stays in the log below correlated
	// by trace_id.
	transient := adapterr.IsTransient(runErr)
	infra := adapterr.IsInfraFailure(runErr)
	safeMessage := adapterr.SafeMessage(runErr)
	job.State = model.JobStateFailed
	job.UpdatedAt = now
	job.CompletedAt = &now
	job.Progress.Phase = "failed"
	job.Progress.LastEventAt = now
	job.Progress.HeartbeatAt = now
	job.Error = &model.JobError{
		Message:   safeMessage,
		Retryable: transient,
		TraceID:   traceID,
		JobID:     jobID,
	}
	slog.ErrorContext(ctx, "job.failed", "component", "daemon", "subcomponent", "jobs", "job_id", jobID, "trace_id", traceID, "transient", transient, "err", runErr)
	if err := manager.appendJobLocked("job_failed", job); err != nil {
		slog.ErrorContext(ctx, "append failed job event failed", "job_id", jobID, "err", err)
	}

	codebase, found := manager.codebases[job.CodebaseID]
	if !found {
		return
	}
	codebase.ActiveJobID = ""
	switch {
	case infra:
		// A shared-infrastructure failure is not the codebase's fault and never
		// terminal; keep the codebase at its resumable last-good state. The repair
		// pass re-attempts it once the dependency recovers, and the health banner
		// carries the cause.
		manager.noteDependencyFailureLocked(runErr)
	case codebase.Kind != model.CodebaseKindDocument && sourceDirMissing(codebase.CanonicalPath):
		// The source directory vanished mid-run. This is not a build failure, so
		// present it as missing and keep the index in case the directory returns.
		codebase.Status = model.CodebaseStatusMissing
		codebase.LastFailedRun = nil
	default:
		codebase.Status = model.CodebaseStatusFailed
		codebase.LastFailedRun = &model.IndexRunFailure{
			Message:                 safeMessage,
			LastAttemptedPercentage: 0,
			FailedAt:                now,
			TraceID:                 traceID,
			JobID:                   jobID,
		}
	}
	codebase.UpdatedAt = now
	manager.codebases[codebase.ID] = codebase
	if err := manager.saveLocked(); err != nil {
		slog.ErrorContext(ctx, "write registry after failed job failed", "job_id", jobID, "err", err)
	}
}

func (manager *Manager) updateJobCancelled(ctx context.Context, jobID string) {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	job, found := manager.jobs[jobID]
	if !found {
		return
	}
	delete(manager.conversationJobs, jobID)

	now := clock.Now()
	metrics.JobCancelled()
	job.State = model.JobStateCancelled
	job.UpdatedAt = now
	job.CompletedAt = &now
	job.Progress.Phase = "cancelled"
	job.Progress.LastEventAt = now
	job.Progress.HeartbeatAt = now
	if err := manager.appendJobLocked("job_cancelled", job); err != nil {
		slog.ErrorContext(ctx, "append cancelled job event failed", "job_id", jobID, "err", err)
	}

	codebase, found := manager.codebases[job.CodebaseID]
	if !found {
		return
	}
	// A cancellation is not a failure: leave the codebase at its last-good state
	// so a status check reflects the current usable state, not a stale failure.
	codebase.ActiveJobID = ""
	codebase.UpdatedAt = now
	manager.codebases[codebase.ID] = codebase
	if err := manager.saveLocked(); err != nil {
		slog.ErrorContext(ctx, "write registry after cancelled job failed", "job_id", jobID, "err", err)
	}
}

func waitForJobDone(ctx context.Context, jobDone chan struct{}) error {
	if jobDone == nil {
		return nil
	}

	select {
	case <-jobDone:
		return nil
	case <-ctx.Done():
		slog.ErrorContext(ctx, "wait for active job cancellation failed", "err", ctx.Err())
		return fmt.Errorf("wait for active job cancellation: %w", ctx.Err())
	}
}

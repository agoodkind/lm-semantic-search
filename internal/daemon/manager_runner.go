package daemon

import (
	"context"
	"fmt"
	"log/slog"

	"goodkind.io/gklog/correlation"
	"goodkind.io/lm-semantic-search/internal/metrics"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/spans"
)

func (manager *Manager) runJobAsync(ctx context.Context, jobID string) {
	detachedCorr := correlation.FromContext(ctx).Child()
	backgroundContext, cancel := context.WithCancel(
		correlation.WithContext(context.WithoutCancel(ctx), detachedCorr),
	)
	done := make(chan struct{})

	manager.mu.Lock()
	manager.cancels[jobID] = cancel
	manager.done[jobID] = done
	manager.mu.Unlock()

	go func() {
		defer func() {
			cancel()
			if recovered := recover(); recovered != nil {
				slog.ErrorContext(backgroundContext, "indexing goroutine panic", "err", fmt.Errorf("panic: %v", recovered), "job_id", jobID)
			}
			manager.mu.Lock()
			delete(manager.cancels, jobID)
			delete(manager.done, jobID)
			manager.mu.Unlock()
			close(done)
		}()
		// The slot is acquired inside the goroutine so callers never block on
		// the cap; the job stays JobStateQueued until runJob calls
		// updateJobRunning, so a queued-behind-the-cap job reports queued.
		select {
		case manager.indexSlots <- struct{}{}:
			slotReleased := false
			defer func() {
				if !slotReleased {
					<-manager.indexSlots
				}
			}()
			graphTask := manager.runJob(backgroundContext, jobID)
			<-manager.indexSlots
			slotReleased = true
			manager.runGraphIndexTask(backgroundContext, graphTask)
		case <-backgroundContext.Done():
			manager.updateJobCancelled(backgroundContext, jobID)
			return
		}
	}()
}

func (manager *Manager) runJob(ctx context.Context, jobID string) *graphIndexTask {
	ctx, done := spans.Open(ctx, "daemon.runJob")
	defer done(nil)

	metrics.JobStarted()
	defer metrics.JobFinished()

	manager.mu.Lock()
	job, found := manager.jobs[jobID]
	manager.mu.Unlock()
	if !found {
		return nil
	}

	manager.updateJobRunning(job)

	// Hold the shared advisory lock for the embed so the upstream TS adapter
	// backs off while this job writes the collection. Skip it when there is no
	// semantic backend, since then the job performs no embedding to coordinate.
	if manager.semantic != nil && manager.semantic.Available() {
		if !manager.syncLock.acquireBlocking(ctx) {
			manager.updateJobCancelled(ctx, job.ID)
			return nil
		}
		defer manager.syncLock.release(ctx)
	}

	// Every operation reaches a terminal job state below. An incremental sync
	// or streaming reindex that finds no usable delta (no prior snapshot, or a
	// live collection that has gone missing) falls through to the from-scratch
	// staging build, which is also the path a true first index and a forced
	// rebuild take. A code job walks the filesystem through the code source; a
	// conversation ingest feeds the manifest and documents through its own source
	// in runConversationIngest, then shares the same delta-then-bootstrap routine.
	codeSource := newCodeItemSource(manager.runner, manager.indexability, job.CodebaseID, job.CanonicalPath, job.Config)
	if manager.semantic != nil && manager.semantic.Available() {
		codeSource = codeSource.withCollectionName(manager.semantic.CollectionName(job.CanonicalPath))
	}
	switch jobOperation(job.Operation) {
	case jobOperationSync:
		handled, graphTask := manager.runDeltaSync(ctx, job, codeSource)
		if handled {
			return graphTask
		}
		return manager.runBootstrap(ctx, job, codeSource)
	case jobOperationStreamingReindex:
		handled, graphTask := manager.runDeltaSync(ctx, job, codeSource)
		if handled {
			return graphTask
		}
		return manager.runBootstrap(ctx, job, codeSource)
	case jobOperationIndex:
		return manager.runBootstrap(ctx, job, codeSource)
	case jobOperationConversationIngest:
		manager.runConversationIngest(ctx, job)
	}
	return nil
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

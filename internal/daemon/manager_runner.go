package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

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
			capacity := &jobCapacity{
				manager:       manager,
				mu:            sync.Mutex{},
				slotHeld:      true,
				syncLockHeld:  false,
				syncLockLease: syncLockLease{lock: nil, once: nil},
			}
			defer capacity.release(backgroundContext)
			runContext := withJobCapacity(backgroundContext, capacity)
			graphTask := manager.runJob(runContext, jobID)
			capacity.release(backgroundContext)
			manager.runGraphIndexTask(backgroundContext, graphTask)
		case <-backgroundContext.Done():
			manager.updateJobCancelled(backgroundContext, jobID)
			return
		}
	}()
}

// acquireJobSyncLock takes the sync lock for one job's embed and reports the
// outcome, so the caller can separate a cancelled request from a machine that
// cannot grant the lock at all.
//
// A job running under a capacity hold takes the lock through that hold, because
// the stall watchdog gives up the lock from its own goroutine while the job
// waits inside a read and a resume takes it back. Such a job gets an empty
// lease: the capacity owns the reference and releasing it twice would drop the
// one the resume took. A job with no capacity hold gets its own lease and
// releases it when the job ends.
func (manager *Manager) acquireJobSyncLock(
	ctx context.Context,
	capacity *jobCapacity,
) (syncLockLease, syncLockOutcome, error) {
	if capacity != nil {
		outcome, err := capacity.acquireSyncLock(ctx)
		return syncLockLease{lock: nil, once: nil}, outcome, err
	}
	return manager.syncLock.acquireBlocking(ctx)
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

	if err := manager.updateJobRunning(job); err != nil {
		slog.ErrorContext(ctx, "start job persistence failed", "job_id", job.ID, "err", err)
		return nil
	}

	// Hold the sync lock for the embed so every other holder of that lock file
	// backs off while this job writes the collection. Skip it when there is no
	// semantic backend, since then the job performs no embedding to coordinate.
	// A permanent lock failure fails the job rather than waiting, because the
	// wait would never end and the job would hold its index slot the whole time.
	if manager.semantic != nil && manager.semantic.Available() {
		// A job that runs under a capacity hold takes the lock through it, so the
		// stall watchdog can give the lock back and a resume can take it again.
		// Without one the job holds a lease for its own duration.
		capacity := jobCapacityFromContext(ctx)
		lease, outcome, lockErr := manager.acquireJobSyncLock(ctx, capacity)
		switch outcome {
		case syncLockAcquired:
			// The lease is empty when the capacity hold owns the reference, and
			// releasing an empty lease does nothing, so this one line covers both
			// shapes without asking again which one is in play.
			defer lease.release(ctx)
		case syncLockCancelled:
			manager.updateJobCancelled(ctx, job.ID)
			return nil
		case syncLockFailed:
			manager.updateJobFailed(ctx, job.ID, lockErr)
			return nil
		case syncLockBusy:
			// acquireBlocking waits out ordinary contention, so it never returns
			// busy; the exhaustive switch check still requires the case. A busy
			// outcome carries no error of its own, so the job reports the lock as
			// unavailable rather than embedding with no lock held.
			manager.updateJobFailed(ctx, job.ID, errSyncLockUnavailable)
			return nil
		default:
			// Any outcome this switch does not name ends the job for the same
			// reason, which puts the safe direction on the fallback.
			manager.updateJobFailed(ctx, job.ID, errSyncLockUnavailable)
			return nil
		}
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
		reason := bootstrapReasonFirstIndex
		if job.Forced {
			reason = bootstrapReasonForcedReindex
		}
		manager.routeToBootstrap(ctx, job.ID, reason)
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

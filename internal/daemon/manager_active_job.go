package daemon

import (
	"context"

	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/model"
)

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

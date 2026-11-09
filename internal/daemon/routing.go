package daemon

import (
	"context"
	"log/slog"
	"runtime"
)

const bootstrapCallerDepth = 2

// routeToBootstrap records the machine-readable reason before a job enters the
// expensive bootstrap path.
func (manager *Manager) routeToBootstrap(
	ctx context.Context,
	jobID string,
	reason bootstrapReason,
) {
	caller := bootstrapRouteCaller()
	if !reason.known() {
		slog.WarnContext(
			ctx,
			"unknown bootstrap reason",
			"reason",
			reason,
			"caller",
			caller,
			"job_id",
			jobID,
		)
	}

	manager.transitionMutex.Lock()
	defer manager.transitionMutex.Unlock()
	manager.mu.Lock()
	job, found := manager.jobs[jobID]
	if !found {
		manager.mu.Unlock()

		return
	}

	job.Progress.BootstrapReason = string(reason)
	codebaseID := job.CodebaseID
	manager.jobs[jobID] = job
	manager.mu.Unlock()

	slog.InfoContext(
		ctx,
		"bootstrap.route",
		"reason",
		reason,
		"caller",
		caller,
		"codebase_id",
		codebaseID,
		"job_id",
		jobID,
	)
}

func (reason bootstrapReason) known() bool {
	switch reason {
	case bootstrapReasonFirstIndex,
		bootstrapReasonForcedReindex,
		bootstrapReasonStagingResume,
		bootstrapReasonEmptyDiffCollectionMissing,
		bootstrapReasonEmptyDiffCollectionEmpty,
		bootstrapReasonDeltaCollectionMissing,
		bootstrapReasonDeltaCodebaseMissing:
		return true
	default:
		return false
	}
}

func bootstrapRouteCaller() string {
	pc, _, _, ok := runtime.Caller(bootstrapCallerDepth)
	if !ok {
		return ""
	}

	function := runtime.FuncForPC(pc)
	if function == nil {
		return ""
	}

	return function.Name()
}

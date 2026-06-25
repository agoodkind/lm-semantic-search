package daemon

import (
	"context"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/model"
)

const maxFailedBuildRetries = 3

func (manager *Manager) retryFailedBuild(ctx context.Context, codebase model.Codebase) {
	manager.mu.Lock()
	if manager.failedBuildRetries == nil {
		manager.failedBuildRetries = map[string]int{}
	}
	if manager.failedBuildRetries[codebase.ID] >= maxFailedBuildRetries {
		manager.mu.Unlock()
		return
	}
	manager.mu.Unlock()

	startedJob, startedCodebase, deduplicated, _, err := manager.StartIndex(
		ctx,
		codebase.CanonicalPath,
		model.ClientInfo{Name: "daemon-failed-retry", PID: 0},
		codebase.EffectiveConfig,
		false,
	)
	if err != nil {
		slog.WarnContext(ctx, "failed build retry could not start", "codebase_id", codebase.ID, "path", codebase.CanonicalPath, "err", err)
		return
	}
	if deduplicated {
		// A retry for this codebase is already in flight (a concurrent sweep, or a
		// redirect to a covering parent), so this call did not start a new build
		// and must not consume an attempt.
		return
	}
	// Count the attempt only after a new build actually started, so a
	// deduplicated or failed StartIndex never burns a retry and the cap reflects
	// real attempts.
	manager.mu.Lock()
	manager.failedBuildRetries[codebase.ID]++
	attemptCount := manager.failedBuildRetries[codebase.ID]
	manager.mu.Unlock()
	slog.InfoContext(ctx, "retrying failed codebase build", "codebase_id", startedCodebase.ID, "path", codebase.CanonicalPath, "job_id", startedJob.ID, "attempt", attemptCount, "max_attempts", maxFailedBuildRetries)
}

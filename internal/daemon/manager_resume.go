package daemon

import (
	"context"
	"log/slog"

	"goodkind.io/claude-context-go/internal/metrics"
	"goodkind.io/claude-context-go/internal/model"
)

// ResumeOrphanedJobs re-queues a streaming reindex for every codebase whose
// previous indexing job was still running when the daemon exited. The
// streaming path's per-file delete-then-upsert keeps the run idempotent, so
// resuming is safe even though no mid-job state is persisted. Call this
// once after NewManager returns and before the daemon advertises ready.
func (manager *Manager) ResumeOrphanedJobs(ctx context.Context) {
	manager.mu.Lock()
	type resumePlan struct {
		canonicalPath string
		config        model.IndexConfig
		codebaseID    string
	}
	plans := make([]resumePlan, 0)
	for _, codebase := range manager.codebases {
		if codebase.Status != model.CodebaseStatusIndexing {
			continue
		}
		plans = append(plans, resumePlan{
			canonicalPath: codebase.CanonicalPath,
			config:        codebase.EffectiveConfig,
			codebaseID:    codebase.ID,
		})
	}
	manager.mu.Unlock()

	if !manager.config.ResumeIndexingOnBoot {
		for _, plan := range plans {
			manager.logResumeSkipped(ctx, plan.codebaseID, plan.canonicalPath)
		}
		return
	}

	if len(plans) > 0 {
		paths := make([]string, 0, len(plans))
		for _, plan := range plans {
			paths = append(paths, plan.canonicalPath)
		}
		slog.InfoContext(ctx, "resuming orphaned indexing jobs", "count", len(plans), "paths", paths)
	}
	for _, plan := range plans {
		client := model.ClientInfo{Name: "daemon-resume", PID: 0}
		_, _, _, _, err := manager.StartIndex(ctx, plan.canonicalPath, client, plan.config, false)
		if err != nil {
			slog.ErrorContext(ctx, "resume orphaned job failed", "codebase_id", plan.codebaseID, "path", plan.canonicalPath, "err", err)
			continue
		}
		metrics.JobResumed()
		manager.logResumeLaunched(ctx, plan.codebaseID, plan.canonicalPath)
	}
}

// logResumeSkipped records that boot resume is disabled for one tracked
// codebase. It exists as a method so the per-codebase line is not emitted
// lexically inside the ResumeOrphanedJobs loop.
func (manager *Manager) logResumeSkipped(ctx context.Context, codebaseID string, path string) {
	slog.InfoContext(ctx, "skipping orphaned indexing job resume", "codebase_id", codebaseID, "path", path, "reason", "resume_on_boot_disabled")
}

// logResumeLaunched records that boot resume re-queued one codebase. It
// exists as a method so the per-codebase line is not emitted lexically
// inside the ResumeOrphanedJobs loop.
func (manager *Manager) logResumeLaunched(ctx context.Context, codebaseID string, path string) {
	slog.InfoContext(ctx, "resumed orphaned indexing job", "codebase_id", codebaseID, "path", path)
}

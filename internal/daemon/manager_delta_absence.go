package daemon

import (
	"context"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/model"
)

// applyDeltaAbsencePolicy decides whether missing source items are deleted or
// retained. It returns true only when a suspicious delete wave terminally
// quarantined the job.
func (manager *Manager) applyDeltaAbsencePolicy(
	ctx context.Context,
	job model.Job,
	codebase model.Codebase,
	source itemSource,
	plan *deltaPlan,
) bool {
	// A code source deletes missing files behind the large-delete quarantine. A
	// conversation source retains absent transcripts because a missing document
	// in one push is usually transient.
	switch source.absencePolicy() {
	case absenceDeleteGuarded:
		signal, suspicious := assessDeltaDeleteWave(
			codebase,
			plan.diff,
			plan.seedSnapshot,
			job.CanonicalPath,
		)
		if suspicious {
			manager.updateJobQuarantined(ctx, job.ID, signal)
			return true
		}
	case absenceRetain:
		if retained := len(plan.diff.Removed); retained > 0 {
			slog.InfoContext(
				ctx,
				"converge.retain_absent",
				"component", "daemon",
				"subcomponent", "delta",
				"collection", codebase.CollectionName,
				"retained", retained,
			)
		}
		plan.diff.Removed = nil
	}
	return false
}

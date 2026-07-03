package daemon

import (
	"context"

	"goodkind.io/lm-semantic-search/internal/model"
)

// resolveItemReusePolicy owns this invariant: A build whose live counterpart
// collection exists loads and reuses its per-item vectors, regardless of
// codebase kind. Forced jobs rebuild through the staging path, so the force
// gate lives there; the live path never carries forced jobs.
func (manager *Manager) resolveItemReusePolicy(ctx context.Context, job model.Job, staging bool, semanticReady bool) bool {
	if !semanticReady {
		return false
	}
	if !staging {
		return true
	}
	if job.Forced {
		return false
	}
	evidence := manager.probeCollectionEvidence(ctx, job.CanonicalPath, "resolveItemReusePolicy")
	return evidence.presence == collectionPresencePresent
}

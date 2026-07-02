package daemon

import (
	"context"

	"goodkind.io/lm-semantic-search/internal/model"
)

// resolveItemReusePolicy owns this invariant: A build whose live counterpart
// collection exists loads and reuses its per-item vectors, regardless of
// staging target or codebase kind. The force escape hatch disables that reuse.
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

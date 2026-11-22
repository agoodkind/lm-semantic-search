package daemon

import (
	"context"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
)

// SyncIndexWithPolicy starts a sync with a per-run scheduling policy patch.
//
// The daemon refuses sync when ingestion is disabled.
//
// Example:
//
//	manager.SyncIndexWithPolicy(ctx, path, client, patch)
func (manager *Manager) SyncIndexWithPolicy(
	ctx context.Context,
	requestedPath string,
	client model.ClientInfo,
	policyPatch model.SchedulingPolicyPatch,
) (model.Job, model.Codebase, bool, error) {
	if manager.config.IngestionDisabled {
		var job model.Job
		var codebase model.Codebase

		return job, codebase, false, adapterr.NewIngestionDisabled()
	}

	return manager.syncIndexWithPolicyAndRun(ctx, requestedPath, client, policyPatch)
}

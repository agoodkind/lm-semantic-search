package daemon

import (
	"context"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
)

// SyncIndexWithPolicy returns the job registered for a sync request.
//
// The policy patch supplies scheduling overrides for that request.
//
// Results:
//   - An active job can satisfy the request without creating another job.
//   - Disabled ingestion returns the ingestion_disabled error.
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

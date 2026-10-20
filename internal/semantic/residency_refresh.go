package semantic

import (
	"context"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

const residencyRefreshInterval = 5 * time.Second

// refreshLoadingResidency observes externally started loads until they finish.
// The reconciliation context cancels this work during reconnect or shutdown.
func (service *Service) refreshLoadingResidency(
	ctx context.Context,
	client *milvusclient.Client,
	generation uint64,
	loading []string,
) {
	if len(loading) == 0 {
		return
	}
	ticker := time.NewTicker(residencyRefreshInterval)
	defer ticker.Stop()
	for len(loading) != 0 {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
		pending := loading[:0]
		for _, name := range loading {
			observe, keep := service.residency.prepareLoadingReconciliation(generation, name)
			if !observe {
				if keep {
					pending = append(pending, name)
				}
				continue
			}
			state := collectionResidencyUnknown
			remote, err := client.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(name))
			if err != nil {
				service.handleReconciliationError(ctx, err, "refresh load state for "+name)
			} else {
				switch remote.State {
				case entity.LoadStateLoaded:
					state = collectionResidencyReady
				case entity.LoadStateLoading:
					state = collectionResidencyLoading
				case entity.LoadStateNotLoad:
					state = collectionResidencyCold
				case entity.LoadStateUnloading:
				}
			}
			service.residency.applyReconciliation(ctx, generation, name, state)
			if state == collectionResidencyLoading || state == collectionResidencyUnknown {
				pending = append(pending, name)
			}
		}
		loading = pending
	}
}

func (controller *collectionResidencyController) prepareLoadingReconciliation(
	generation uint64,
	name string,
) (observe bool, pending bool) {
	controller.mutex.Lock()
	defer controller.mutex.Unlock()
	if controller.closed || generation != controller.reconciliationGeneration ||
		controller.reconciliationActivity[name] >= generation {
		return false, false
	}
	entry := controller.entries[name]
	if entry == nil || (entry.state != collectionResidencyLoading && entry.state != collectionResidencyUnknown) {
		return false, false
	}
	if controller.entryHasActivityLocked(entry) {
		return false, true
	}
	entry.reconciliation = generation
	return true, true
}

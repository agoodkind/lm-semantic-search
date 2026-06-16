package semantic

import (
	"context"
	"fmt"
	"strings"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/adapterr"
)

// storeProbeMaxCollections bounds how many collections the data-plane probe
// inspects before concluding. One loaded collection proves the query plane
// serves, so a healthy store returns on the first loaded collection it finds and
// the cap only bounds the unhealthy case where many collections read unloaded.
const storeProbeMaxCollections = 16

// ProbeHealth actively verifies that search can serve a query right now, so a
// caller can report searchability instead of echoing the last job outcome. It
// checks the store's query plane and the embedding endpoint. The returned error
// is adapterr-classified so the daemon maps it to the right degraded banner; nil
// means search is usable now.
//
// A service with no Milvus address is not configured for semantic search rather
// than degraded, so it returns nil (no outage to surface). A configured service
// whose client is not connected returns a store-unavailable outage.
func (service *Service) ProbeHealth(ctx context.Context) error {
	if service == nil || strings.TrimSpace(service.cfg.MilvusAddress) == "" {
		return nil
	}
	if !service.Available() || service.milvus == nil || service.embedder == nil {
		return adapterr.NewMilvusUnavailable(ErrUnavailable)
	}
	if err := service.probeStoreQueryPlane(ctx); err != nil {
		return err
	}
	if err := service.embedder.Health(ctx); err != nil {
		return adapterr.NewEmbedderUnreachable(fmt.Errorf("probe embedder: %w", err))
	}
	return nil
}

// probeStoreQueryPlane verifies the vector store can serve a search now, not just
// that its metadata endpoint answers. ListCollections proves the store is
// reachable; GetLoadState then proves the query plane actually holds a collection
// in memory, which is what a real search needs. A store that restarted answers
// ListCollections while every collection reads NotLoad until it reloads, so this
// catches the metadata-up / query-down window that a ListCollections-only probe
// missed and that forced the agent onto a failing search path.
//
// One loaded collection is enough to prove the query plane serves, so the scan
// returns on the first. A reachable store whose collections all read unloaded is
// reported unavailable so search gating fails open. Staging collections are
// skipped because a build leaves them unloaded by design. With no real collection
// to inspect there is nothing to gate, so the store reads healthy.
func (service *Service) probeStoreQueryPlane(ctx context.Context) error {
	collections, err := service.milvus.ListCollections(ctx, milvusclient.NewListCollectionOption())
	if err != nil {
		return adapterr.NewMilvusUnavailable(fmt.Errorf("probe Milvus list: %w", err))
	}

	examined := 0
	for _, collectionName := range collections {
		if strings.HasSuffix(collectionName, stagingCollectionSuffix) {
			continue
		}
		if examined >= storeProbeMaxCollections {
			break
		}
		examined++

		loadState, stateErr := service.milvus.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(collectionName))
		if stateErr != nil {
			if storeUnavailable(stateErr) {
				return adapterr.NewMilvusUnavailable(fmt.Errorf("probe Milvus load state %s: %w", collectionName, stateErr))
			}
			continue
		}
		if loadState.State == entity.LoadStateLoaded {
			return nil
		}
	}

	if examined > 0 {
		return adapterr.NewMilvusUnavailable(fmt.Errorf("probe Milvus: no searchable collection loaded (%d examined)", examined))
	}
	return nil
}

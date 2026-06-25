package semantic

import (
	"context"
	"fmt"
	"strings"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/adapterr"
)

// ProbeHealth verifies the vector store is reachable right now. It is the global
// shared-dependency probe for surfaces that have no single codebase path (the
// status banner, list views). It checks store reachability only: per-path
// searchability is decided by CollectionState against the specific
// collection's load state, and embedder health is observed from real embed
// outcomes on the search and index paths rather than a synthetic probe. That
// keeps a slow real embed (seconds) from blocking a status read and stops a
// shallow liveness route from reading healthy while embeds actually fail. The
// returned error is adapterr-classified; nil means the store answers.
//
// A service with no Milvus address is not configured for semantic search rather
// than degraded, so it returns nil. A configured service whose client is not
// connected returns a store-unavailable outage.
func (service *Service) ProbeHealth(ctx context.Context) error {
	if service == nil || strings.TrimSpace(service.cfg.MilvusAddress) == "" {
		return nil
	}
	if !service.Available() || service.milvus == nil {
		return adapterr.NewMilvusUnavailable(ErrUnavailable)
	}
	if _, err := service.milvus.ListCollections(ctx, milvusclient.NewListCollectionOption()); err != nil {
		return adapterr.NewMilvusUnavailable(fmt.Errorf("probe Milvus list: %w", err))
	}
	return nil
}

// CollectionState reports the per-path facts the daemon maps to a collection
// readiness: whether the collection that serves codebasePath exists and whether
// it is loaded into Milvus query nodes right now. Distinguishing absent from
// not-loaded is why this checks HasCollection before GetLoadState: a first build
// that has not created the collection reads (false, false), a built-but-not-loaded
// collection reads (true, false), and a loaded collection reads (true, true). A
// store that cannot answer returns a classified outage so the caller treats the
// readiness as unknown; the global store banner is left to ProbeHealth. A service
// not configured for or not connected to Milvus reports (false, false, nil).
func (service *Service) CollectionState(ctx context.Context, codebasePath string) (bool, bool, error) {
	if service == nil || !service.Available() || service.milvus == nil {
		return false, false, nil
	}
	collectionName := service.CollectionName(codebasePath)
	has, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		return false, false, adapterr.NewMilvusUnavailable(fmt.Errorf("check collection %s: %w", collectionName, err))
	}
	if !has {
		return false, false, nil
	}
	loadState, err := service.milvus.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(collectionName))
	if err != nil {
		return true, false, adapterr.NewMilvusUnavailable(fmt.Errorf("load state %s: %w", collectionName, err))
	}
	return true, loadState.State == entity.LoadStateLoaded, nil
}

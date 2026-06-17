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
// searchability is decided by CollectionSearchable against the specific
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

// CollectionSearchable reports whether the collection that serves codebasePath is
// loaded into Milvus query nodes right now, the deterministic per-path
// precondition for a real search. It reads GetLoadState, which returns a typed
// load-state enum, so there is no error-string guessing: a collection that exists
// but is not loaded (a store that restarted and has not reloaded it) reads false,
// and a store that cannot answer the load state reads false with a classified
// outage so the caller degrades and fails open. A service not configured for or
// not connected to Milvus reports false with no error, leaving the gate to the
// store-reachability probe.
func (service *Service) CollectionSearchable(ctx context.Context, codebasePath string) (bool, error) {
	if service == nil || !service.Available() || service.milvus == nil {
		return false, nil
	}
	collectionName := service.CollectionName(codebasePath)
	loadState, err := service.milvus.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(collectionName))
	if err != nil {
		return false, adapterr.NewMilvusUnavailable(fmt.Errorf("load state %s: %w", collectionName, err))
	}
	return loadState.State == entity.LoadStateLoaded, nil
}

package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

const defaultCollectionSearchLimit = 10

// CollectionSearchRequest is one typed search of a registered document
// collection. Filter is nil to match every row.
type CollectionSearchRequest struct {
	CollectionID  string
	Query         string
	Limit         int32
	MinScore      float64
	Filter        *semantic.CollectionFilter
	GroupBy       string
	PerGroupLimit int32
}

// SearchCollection searches a registered document collection. The daemon's
// maintenance refusal applies before any registry, store, or collection work.
// The request is validated against the saved declaration before the store
// loads the collection or runs the query. An unregistered collection fails
// with [adapterr.NewCollectionNotRegistered] and registers nothing.
func (manager *Manager) SearchCollection(ctx context.Context, request CollectionSearchRequest) ([]semantic.CollectionHit, error) {
	if refusal := manager.maintenanceRefusal(); refusal != nil {
		return nil, refusal
	}
	collectionID := strings.TrimSpace(request.CollectionID)
	if collectionID == "" {
		return nil, adapterr.NewMissingArgument("collection_id")
	}
	manager.mu.Lock()
	codebase, found := manager.findDocumentCollectionLocked(collectionID)
	manager.mu.Unlock()
	if !found {
		return nil, adapterr.NewCollectionNotRegistered(collectionID)
	}
	return manager.searchRegisteredCollection(ctx, collectionID, codebase, request)
}

// CollectionItemState returns the content fingerprint the collection's Merkle
// checkpoint records for itemID. An unregistered collection and an item the
// checkpoint does not list both return the empty fingerprint. The read never
// registers a collection and never calls the vector store.
func (manager *Manager) CollectionItemState(ctx context.Context, collectionID string, itemID string) (string, error) {
	trimmedCollectionID := strings.TrimSpace(collectionID)
	if trimmedCollectionID == "" {
		return "", adapterr.NewMissingArgument("collection_id")
	}
	trimmedItemID := strings.TrimSpace(itemID)
	if trimmedItemID == "" {
		return "", adapterr.NewMissingArgument("item_id")
	}
	manager.mu.Lock()
	codebase, found := manager.findDocumentCollectionLocked(trimmedCollectionID)
	manager.mu.Unlock()
	if !found {
		return "", nil
	}
	checkpoint := manager.loadLiveCheckpoint(ctx, codebase, codebase.EffectiveConfig.IgnoreDigest)
	return checkpoint.snapshot.Files[trimmedItemID], nil
}

func collectionDeclaration(codebase model.Codebase) model.CollectionDeclaration {
	return savedCollectionDeclaration(codebase)
}

func (manager *Manager) searchRegisteredCollection(ctx context.Context, collectionID string, codebase model.Codebase, request CollectionSearchRequest) ([]semantic.CollectionHit, error) {
	declaration := collectionDeclaration(codebase)
	if err := validateCollectionSearch(collectionID, declaration, request.Filter, request.GroupBy, request.PerGroupLimit); err != nil {
		return nil, err
	}
	limit := request.Limit
	if limit <= 0 {
		limit = defaultCollectionSearchLimit
	}

	if manager.semantic == nil || !manager.semantic.Available() {
		manager.noteDependencyFailure(semantic.ErrUnavailable)
		return nil, semantic.ErrUnavailable
	}

	lease, leaseErr := manager.semantic.AcquireCollection(ctx, codebase.CollectionName)
	if leaseErr != nil {
		manager.noteDependencyFailure(leaseErr)
		return nil, fmt.Errorf("acquire collection %s: %w", codebase.CollectionName, leaseErr)
	}
	defer lease.Release()
	hits, err := manager.semantic.SearchCollection(ctx, semantic.CollectionSearch{
		CollectionName: codebase.CollectionName,
		Query:          request.Query,
		Limit:          limit,
		MinScore:       request.MinScore,
		Filter:         request.Filter,
		GroupBy:        request.GroupBy,
		PerGroupLimit:  request.PerGroupLimit,
		Declaration:    declaration,
	})
	if err != nil {
		manager.noteDependencyFailure(err)
		slog.ErrorContext(ctx, "search collection failed", "collection_id", collectionID, "collection", codebase.CollectionName, "err", err)
		return nil, fmt.Errorf("search collection %s: %w", codebase.CollectionName, err)
	}
	manager.noteDependencyHealthy()
	return hits, nil
}

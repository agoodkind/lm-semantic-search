package semantic

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"goodkind.io/lm-semantic-search/collection"
	"goodkind.io/lm-semantic-search/internal/model"
	"google.golang.org/grpc/peer"
)

// defaultCollectionSearchLimit is the hit count a search returns when the
// caller sets no positive limit.
const defaultCollectionSearchLimit = 10

// CollectionHit is one ranked search hit. Chunk is the stored row decoded
// through the legacy metadata JSON, and Chunk.RelativePath is the logical row
// key. Scalars lists every declared scalar column in declaration order.
type CollectionHit struct {
	Chunk   model.StoredChunk
	Scalars []collection.ScalarCell
}

// Scalar returns the hit's cell for column. It reports false when column is
// not declared.
func (hit CollectionHit) Scalar(column string) (collection.ScalarCell, bool) {
	for _, cell := range hit.Scalars {
		if cell.Column == column {
			return cell, true
		}
	}
	return collection.AbsentCell(column), false
}

// CollectionSearch is one validated typed search of a registered collection.
// Filter is nil to match every row. PerGroupLimit caps the hits that share one
// GroupBy value, and zero means uncapped. Declaration is the collection's saved
// declaration.
type CollectionSearch struct {
	CollectionName string
	Query          string
	Limit          int32
	MinScore       float64
	Filter         *collection.Filter
	GroupBy        string
	PerGroupLimit  int32
	Declaration    collection.Declaration
}

// SearchCollection runs a typed search and returns at most Limit hits, at most
// PerGroupLimit per GroupBy value, none scoring below MinScore. On an unchanged
// collection, repeating the same query with the same filter returns the same
// rows in the same order, and a smaller limit returns a prefix of a larger
// one. The filter restricts one fixed-depth ranking natively, and every
// membership set binds as one template parameter.
func (service *Service) SearchCollection(ctx context.Context, search CollectionSearch) ([]CollectionHit, error) {
	peerInfo, _ := peer.FromContext(ctx)
	request, err := service.prepareCollectionSearch(ctx, search)
	if err != nil {
		return nil, err
	}
	hits, err := service.collectionStore().Search(ctx, request)
	if err != nil {
		slog.ErrorContext(ctx, "search collection failed", "collection", request.Collection, "peer", peerInfo.String(), "err", err)
		return nil, fmt.Errorf("search %s: %w", request.Collection, err)
	}
	return collectionHits(hits, search.Declaration), nil
}

// prepareCollectionSearch validates a typed search, checks that the collection
// exists, runs the split part migration, and embeds the query. It returns the
// store request with the default limit applied.
func (service *Service) prepareCollectionSearch(ctx context.Context, search CollectionSearch) (collection.SearchRequest, error) {
	peerInfo, _ := peer.FromContext(ctx)
	var request collection.SearchRequest
	if !service.Available() {
		return request, ErrUnavailable
	}
	collectionName := strings.TrimSpace(search.CollectionName)
	if collectionName == "" {
		return request, errors.New("collection name is required")
	}
	limit := search.Limit
	if limit <= 0 {
		limit = defaultCollectionSearchLimit
	}
	if _, err := collection.Compile(search.Filter); err != nil {
		slog.ErrorContext(ctx, "compile collection filter failed", "collection", collectionName, "err", err)
		return request, fmt.Errorf("compile filter for %s: %w", collectionName, err)
	}
	hasCollection, err := service.hasCollection(ctx, collectionName, "check Milvus collection "+collectionName)
	if err != nil {
		return request, err
	}
	if !hasCollection {
		return request, ErrCollectionMissing
	}
	if err := service.ensureSplitPartColumnOnce(ctx, collectionName); err != nil {
		return request, err
	}
	queryVector, err := service.embedder.Embed(ctx, service.queryTextForEmbedding(search.Query))
	if err != nil {
		slog.ErrorContext(ctx, "embed query failed", "peer", peerInfo.String(), "err", err)
		return request, fmt.Errorf("embed query: %w", err)
	}
	return collection.SearchRequest{
		Collection:    collectionName,
		Query:         search.Query,
		Vector:        queryVector,
		Limit:         limit,
		MinScore:      search.MinScore,
		Filter:        search.Filter,
		GroupBy:       search.GroupBy,
		PerGroupLimit: search.PerGroupLimit,
		Declaration:   search.Declaration,
	}, nil
}

// collectionHits converts store hits to daemon hits. Each hit lists a cell for
// every declared scalar column in declaration order.
func collectionHits(hits []collection.Hit, declaration collection.Declaration) []CollectionHit {
	converted := make([]CollectionHit, 0, len(hits))
	for _, hit := range hits {
		cells := make([]collection.ScalarCell, 0, len(declaration.Scalars))
		for _, declared := range declaration.Scalars {
			cell, found := hit.Scalars[declared.Name]
			if !found {
				cell = collection.AbsentCell(declared.Name)
			}
			cells = append(cells, cell)
		}
		converted = append(converted, CollectionHit{Chunk: chunkFromHit(hit), Scalars: cells})
	}
	return converted
}

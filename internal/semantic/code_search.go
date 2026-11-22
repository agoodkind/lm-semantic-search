package semantic

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/milvus-io/milvus/client/v3/milvusclient"
	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"
	"goodkind.io/lm-semantic-search/internal/model"
	"google.golang.org/grpc/peer"
)

// Search runs semantic or hybrid search against a codebase collection.
//
// A nonempty relativePathPrefix restricts the results to that subtree.
//
// Example:
//
//	service.Search(ctx, path, "parse config", 5, nil, "internal")
func (service *Service) Search(
	ctx context.Context,
	codebasePath string,
	query string,
	limit int32,
	extensionFilter []string,
	relativePathPrefix string,
) ([]model.StoredChunk, error) {
	if !service.Available() {
		return nil, ErrUnavailable
	}

	collectionName := service.CollectionName(codebasePath)

	return service.searchCollection(
		ctx,
		collectionName,
		query,
		limit,
		buildSearchFilter(extensionFilter, []string{relativePathPrefix}),
	)
}

// queryTextForEmbedding adds the configured instruction to the dense query.
// BM25 searches the raw query text.
func (service *Service) queryTextForEmbedding(query string) string {
	prefix := service.cfg.QueryInstructionPrefix
	if prefix == "" {
		return query
	}

	return prefix + query
}

func (service *Service) searchCollection(
	ctx context.Context,
	collectionName string,
	query string,
	limit int32,
	filterExpr string,
) ([]model.StoredChunk, error) {
	peerInfo, _ := peer.FromContext(ctx)
	hasCollection, err := service.hasCollection(
		ctx,
		collectionName,
		"check Milvus collection "+collectionName,
	)
	if err != nil {
		return nil, err
	}

	if !hasCollection {
		return nil, ErrCollectionMissing
	}

	queryVector, err := service.embedder.Embed(ctx, service.queryTextForEmbedding(query))
	if err != nil {
		slog.ErrorContext(ctx, "embed query failed", "peer", peerInfo.String(), "err", err)

		return nil, fmt.Errorf("embed query: %w", err)
	}

	return service.searchCollectionWithVector(
		ctx,
		collectionName,
		queryVector,
		query,
		int(limit),
		filterExpr,
	)
}

// searchCollectionWithVector searches with a precomputed query vector.
// The caller checks that the collection exists before calling this method.
func (service *Service) searchCollectionWithVector(
	ctx context.Context,
	collectionName string,
	queryVector []float32,
	rawQuery string,
	limit int,
	filterExpr string,
) ([]model.StoredChunk, error) {
	peerInfo, _ := peer.FromContext(ctx)
	omitSplitPart, err := service.omitSplitPartForSearch(ctx, collectionName)
	if err != nil {
		return nil, err
	}

	hits, err := service.collectionStore().SearchExpression(
		ctx,
		milvusstore.ExpressionSearch{
			Collection:    collectionName,
			Vector:        queryVector,
			Query:         rawQuery,
			Limit:         limit,
			Expression:    filterExpr,
			OmitSplitPart: omitSplitPart,
		},
	)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"search collection rows failed",
			"collection",
			collectionName,
			"peer",
			peerInfo.String(),
			"err",
			err,
		)

		return nil, fmt.Errorf("search %s: %w", collectionName, err)
	}

	return chunksFromHits(hits), nil
}

func (service *Service) omitSplitPartForSearch(
	ctx context.Context,
	collectionName string,
) (bool, error) {
	if !service.cfg.IngestionDisabled {
		return false, service.ensureSplitPartColumnOnce(ctx, collectionName)
	}

	described, err := service.milvus.DescribeCollection(
		ctx,
		milvusclient.NewDescribeCollectionOption(collectionName),
	)
	if err != nil {
		return false, wrapStoreError(ctx, err, "describe Milvus search collection "+collectionName)
	}

	return len(splitPartFieldsToAdd(described.Schema)) != 0, nil
}

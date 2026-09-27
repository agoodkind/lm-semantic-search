package semantic

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

// CollectionItemRows is the stored row identity of one item in a generic
// document collection. UsablePaths is the set of relativePath values of the
// item's rows with content a search can return.
type CollectionItemRows struct {
	UsablePaths map[string]struct{}
}

// CollectionItemBatchState is one batched read of a generic document
// collection for a set of item ids. Rows maps each requested item id with at
// least one stored row to its row identities. Reuse maps a content hash to a
// dense vector from any row the read selected, restricted to rows with no
// recorded embedding model or the current one.
type CollectionItemBatchState struct {
	Rows  map[string]CollectionItemRows
	Reuse map[string][]float32
}

// LoadCollectionItemBatch reads the stored rows of itemIDs from a generic
// document collection. It selects rows by the declared item id scalar column
// itemColumn, one Milvus query per id batch.
func (service *Service) LoadCollectionItemBatch(
	ctx context.Context,
	collectionName string,
	itemColumn string,
	itemIDs []string,
) (CollectionItemBatchState, error) {
	state := CollectionItemBatchState{Rows: map[string]CollectionItemRows{}, Reuse: map[string][]float32{}}
	uniqueIDs := dedupeConversationIDs(itemIDs)
	if !service.Available() || collectionName == "" || itemColumn == "" || len(uniqueIDs) == 0 {
		return state, nil
	}
	hasCollection, err := service.hasCollection(ctx, collectionName, "check Milvus collection "+collectionName)
	if err != nil {
		return CollectionItemBatchState{}, err
	}
	if !hasCollection {
		return state, nil
	}
	if err := service.PrepareCollection(ctx, collectionName); err != nil {
		return CollectionItemBatchState{}, err
	}
	lease, err := service.AcquireCollection(ctx, collectionName)
	if err != nil {
		return CollectionItemBatchState{}, err
	}
	defer lease.Release()

	for _, idBatch := range batchConversationIDs(uniqueIDs, conversationBatchIDFilterSize) {
		if err := service.loadCollectionItemGroup(ctx, collectionName, itemColumn, idBatch, state); err != nil {
			return CollectionItemBatchState{}, err
		}
	}
	return state, nil
}

func (service *Service) loadCollectionItemGroup(
	ctx context.Context,
	collectionName string,
	itemColumn string,
	itemIDs []string,
	state CollectionItemBatchState,
) error {
	iterator, err := service.milvus.QueryIterator(ctx, milvusclient.NewQueryIteratorOption(collectionName).
		WithBatchSize(reuseVectorBatchSize).
		WithFilter(inStringClause(itemColumn, itemIDs)).
		WithOutputFields(itemColumn, relativePathFieldName, contentFieldName, embeddingModelFieldName, denseVectorFieldName))
	if err != nil {
		slog.ErrorContext(ctx, "open collection item query iterator failed", "collection", collectionName, "err", err)
		return fmt.Errorf("open collection item iterator for %s: %w", collectionName, err)
	}
	for {
		resultSet, nextErr := iterator.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			return nil
		}
		if nextErr != nil {
			slog.ErrorContext(ctx, "collection item query iterator next failed", "collection", collectionName, "err", nextErr)
			return fmt.Errorf("iterate %s for collection items: %w", collectionName, nextErr)
		}
		if err := service.appendCollectionItemRows(resultSet, itemColumn, state); err != nil {
			return err
		}
	}
}

func (service *Service) appendCollectionItemRows(
	resultSet milvusclient.ResultSet,
	itemColumn string,
	state CollectionItemBatchState,
) error {
	contentColumn := resultSet.GetColumn(contentFieldName)
	vectorColumn := resultSet.GetColumn(denseVectorFieldName)
	itemIDColumn := resultSet.GetColumn(itemColumn)
	relativePathColumn := resultSet.GetColumn(relativePathFieldName)
	if contentColumn == nil || vectorColumn == nil || itemIDColumn == nil || relativePathColumn == nil {
		return ErrSearchResultIncomplete
	}
	embeddingModelColumn := resultSet.GetColumn(embeddingModelFieldName)
	for rowIndex := range resultSet.ResultCount {
		contentValue, vector, contentErr := conversationContentVectorAt(contentColumn, vectorColumn, rowIndex)
		if contentErr != nil {
			return contentErr
		}
		embeddingModel, modelErr := nullableStringAt(embeddingModelColumn, rowIndex)
		if modelErr != nil {
			return fmt.Errorf("read collection item embedding model at %d: %w", rowIndex, modelErr)
		}
		if embeddingModelsCompatible(embeddingModel, service.cfg.EmbeddingModel) {
			state.Reuse[contentVectorKey(contentValue)] = vector
		}
		itemID, present, idErr := readOptionalStringAt(itemIDColumn, rowIndex)
		if idErr != nil {
			return idErr
		}
		if !present || itemID == "" {
			continue
		}
		relativePath, pathErr := relativePathColumn.GetAsString(rowIndex)
		if pathErr != nil {
			slog.Error("read collection item relative path failed", "index", rowIndex, "err", pathErr)
			return fmt.Errorf("read relative path column at %d: %w", rowIndex, pathErr)
		}
		stored, found := state.Rows[itemID]
		if !found {
			stored = CollectionItemRows{UsablePaths: map[string]struct{}{}}
			state.Rows[itemID] = stored
		}
		if strings.TrimSpace(contentValue) != "" {
			stored.UsablePaths[relativePath] = struct{}{}
		}
	}
	return nil
}

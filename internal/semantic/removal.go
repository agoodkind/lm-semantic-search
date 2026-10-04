package semantic

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"goodkind.io/lm-semantic-search/collection"
	"goodkind.io/lm-semantic-search/internal/spans"
)

// Removal names the stored rows one delta step drops before inserting the
// item's fresh chunks. Paths match a row's relativePath exactly, which a code
// file uses because all its chunks share one relativePath.
//
// ItemColumn and ItemIDs select rows by a declared item id scalar column. A
// document collection removes an item's rows by the item id stored in that
// column.
type Removal struct {
	Paths      []string
	ItemColumn string
	ItemIDs    []string
}

// Empty reports whether the removal would delete nothing.
func (removal Removal) Empty() bool {
	return len(removal.Paths) == 0 && len(removal.ItemIDs) == 0
}

// RemoveItems builds a removal that drops every row with an itemColumn value in
// itemIDs.
func RemoveItems(itemColumn string, itemIDs []string) Removal {
	return Removal{Paths: nil, ItemColumn: itemColumn, ItemIDs: itemIDs}
}

// RemovePaths builds a removal that drops rows by exact relativePath, the code
// file shape.
func RemovePaths(paths []string) Removal {
	return Removal{Paths: paths, ItemColumn: "", ItemIDs: nil}
}

// DeleteItemRows deletes the rows removal selects from a document collection.
// It serves an explicit item delete, and a missing collection deletes nothing.
// DeleteItemRows prepares the collection before an item removal.
func (service *Service) DeleteItemRows(ctx context.Context, collectionName string, removal Removal) (err error) {
	ctx, done := spans.Open(ctx, "semantic.deleteItemRows")
	defer done(&err)

	if !service.Available() {
		return ErrUnavailable
	}
	trimmedCollectionName := strings.TrimSpace(collectionName)
	if trimmedCollectionName == "" {
		return errors.New("document collection name is required")
	}
	if removal.Empty() {
		return errors.New("item removal selects no rows")
	}
	hasCollection, err := service.hasCollection(ctx, trimmedCollectionName, "check Milvus collection "+trimmedCollectionName)
	if err != nil {
		return err
	}
	if !hasCollection {
		return nil
	}
	if len(removal.ItemIDs) > 0 {
		if err := service.PrepareCollection(ctx, trimmedCollectionName); err != nil {
			return err
		}
	}
	lease, err := service.AcquireCollection(ctx, trimmedCollectionName)
	if err != nil {
		return err
	}
	defer lease.Release()
	return service.deleteByRemoval(ctx, trimmedCollectionName, removal)
}

// deleteByRemoval drops an item's prior rows by exact relativePath, by item id
// column, or both. The caller holds the collection lease because
// Milvus serves an expression-filtered Delete only on a loaded collection.
//
// The span separates the delete from the embed and insert phases of the same
// reindex. An expression-filtered Delete matches an unbounded row count and a
// cold collection pays a load first, so this phase can dominate a slow reindex
// without any other line saying so. semantic.removal_completed reports the
// rows the store removed after every delete succeeds.
func (service *Service) deleteByRemoval(ctx context.Context, collectionName string, removal Removal) (err error) {
	ctx, done := spans.Open(ctx, "semantic.deleteByRemoval")
	defer done(&err)

	var pathRowsRemoved int64
	if len(removal.Paths) > 0 {
		pathRowsRemoved, err = service.deleteByRelativePaths(
			ctx,
			collectionName,
			removal.Paths,
		)
		if err != nil {
			return err
		}
	}
	store := service.collectionStore()
	declaration := collection.Declaration{ItemIDColumn: removal.ItemColumn, Scalars: nil}
	var itemRowsRemoved int64
	if len(removal.ItemIDs) > 0 {
		itemRowsRemoved, err = store.DeleteItems(ctx, collection.DeleteItemsRequest{
			Collection:   collectionName,
			Declaration:  declaration,
			ItemIDs:      removal.ItemIDs,
			PathPrefixes: nil,
		})
		if err != nil {
			slog.ErrorContext(ctx, "delete item rows failed", "collection", collectionName, "err", err)
			return fmt.Errorf("delete items from %s: %w", collectionName, err)
		}
	}
	slog.InfoContext(
		ctx,
		"semantic.removal_completed",
		"collection",
		collectionName,
		"path_rows_removed",
		pathRowsRemoved,
		"item_rows_removed",
		itemRowsRemoved,
		"rows_removed",
		pathRowsRemoved+itemRowsRemoved,
	)
	return nil
}

// relativePathPrefixExpression renders the Milvus filter expression matching
// every row whose relativePath begins with prefix. The prefix delete and the
// prefix-scoped reuse read share it so both name the same row set.
func relativePathPrefixExpression(prefix string) string {
	return fmt.Sprintf(`%s like "%s%%"`, relativePathFieldName, escapeMilvusLikePattern(prefix))
}

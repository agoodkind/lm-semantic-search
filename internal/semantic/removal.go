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
// file uses because all its chunks share one relativePath. Prefixes match every
// row whose relativePath begins with the prefix, which a conversation uses
// because its messages span many relativePaths under one conv/<id>/ prefix.
//
// ItemColumn and ItemIDs select rows by a declared item id scalar column. A
// document collection removes an item's rows by the item id stored in that
// column, and a conversation collection adds its legacy relativePath prefixes
// for rows written before the conversationId column existed.
type Removal struct {
	Paths      []string
	Prefixes   []string
	ItemColumn string
	ItemIDs    []string
}

// Empty reports whether the removal would delete nothing.
func (removal Removal) Empty() bool {
	return len(removal.Paths) == 0 && len(removal.Prefixes) == 0 && len(removal.ItemIDs) == 0
}

// RemoveItems builds a removal that drops every row with an itemColumn value in
// itemIDs, plus every row under legacyPrefixes.
func RemoveItems(itemColumn string, itemIDs []string, legacyPrefixes []string) Removal {
	return Removal{Paths: nil, Prefixes: legacyPrefixes, ItemColumn: itemColumn, ItemIDs: itemIDs}
}

// RemovePaths builds a removal that drops rows by exact relativePath, the code
// file shape.
func RemovePaths(paths []string) Removal {
	return Removal{Paths: paths, Prefixes: nil, ItemColumn: "", ItemIDs: nil}
}

// DeleteItemRows deletes the rows removal selects from a document collection.
// It serves an explicit item delete, and a missing collection deletes nothing.
// An item removal filters on the item id column. The conversation scalar
// migration adds that column to a legacy conversation collection, and
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

// deleteByRemoval drops an item's prior rows by exact relativePath, by
// relativePath prefix, or both. The caller holds the collection lease because
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
	var prefixRowsRemoved int64
	for _, prefix := range removal.Prefixes {
		if prefix == "" {
			continue
		}
		removed, deleteErr := store.DeleteItems(ctx, collection.DeleteItemsRequest{
			Collection:   collectionName,
			Declaration:  declaration,
			ItemIDs:      nil,
			PathPrefixes: []string{prefix},
		})
		if deleteErr != nil {
			slog.ErrorContext(ctx, "delete prefix rows failed", "collection", collectionName, "prefix", prefix, "err", deleteErr)
			return fmt.Errorf("delete prefix %s from %s: %w", prefix, collectionName, deleteErr)
		}
		prefixRowsRemoved += removed
	}
	slog.InfoContext(
		ctx,
		"semantic.removal_completed",
		"collection",
		collectionName,
		"path_rows_removed",
		pathRowsRemoved,
		"prefix_rows_removed",
		prefixRowsRemoved,
		"item_rows_removed",
		itemRowsRemoved,
		"rows_removed",
		pathRowsRemoved+prefixRowsRemoved+itemRowsRemoved,
	)
	return nil
}

// relativePathPrefixExpression renders the Milvus filter expression matching
// every row whose relativePath begins with prefix. The prefix delete and the
// prefix-scoped reuse read share it so both name the same row set.
func relativePathPrefixExpression(prefix string) string {
	return fmt.Sprintf(`%s like "%s%%"`, relativePathFieldName, escapeMilvusLikePattern(prefix))
}

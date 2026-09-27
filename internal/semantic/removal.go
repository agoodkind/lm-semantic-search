package semantic

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
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
	var itemRowsRemoved int64
	if len(removal.ItemIDs) > 0 {
		itemRowsRemoved, err = service.deleteByItemIDs(ctx, collectionName, removal.ItemColumn, removal.ItemIDs)
		if err != nil {
			return err
		}
	}
	var prefixRowsRemoved int64
	if len(removal.Prefixes) > 0 {
		for _, prefix := range removal.Prefixes {
			removed, deleteErr := service.deleteByRelativePathPrefix(
				ctx,
				collectionName,
				prefix,
			)
			if deleteErr != nil {
				err = deleteErr
				return err
			}
			prefixRowsRemoved += removed
		}
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

// deleteByItemIDs removes every row with an itemColumn value in itemIDs.
func (service *Service) deleteByItemIDs(
	ctx context.Context,
	collectionName string,
	itemColumn string,
	itemIDs []string,
) (int64, error) {
	if itemColumn == "" {
		return 0, fmt.Errorf("delete items from %s: item column is required", collectionName)
	}
	var removed int64
	for _, idBatch := range batchConversationIDs(itemIDs, conversationFilterIDBatchSize) {
		result, err := service.milvus.Delete(
			ctx,
			milvusclient.NewDeleteOption(collectionName).WithExpr(inStringClause(itemColumn, idBatch)),
		)
		if err != nil {
			return removed, wrapStoreError(
				ctx,
				err,
				"delete from "+collectionName+" by item id column "+itemColumn,
			)
		}
		removed += result.DeleteCount
	}
	return removed, nil
}

// deleteByRelativePathPrefix removes every row whose relativePath begins with
// prefix. A conversation uses it to drop all of one conversation's message rows
// in a single expression delete.
func (service *Service) deleteByRelativePathPrefix(
	ctx context.Context,
	collectionName string,
	prefix string,
) (int64, error) {
	if prefix == "" {
		return 0, nil
	}
	expression := relativePathPrefixExpression(prefix)
	result, err := service.milvus.Delete(
		ctx,
		milvusclient.NewDeleteOption(collectionName).WithExpr(expression),
	)
	if err != nil {
		return 0, wrapStoreError(
			ctx,
			err,
			"delete from "+collectionName+" by relative path prefix "+prefix,
		)
	}
	return result.DeleteCount, nil
}

// relativePathPrefixExpression renders the Milvus filter expression matching
// every row whose relativePath begins with prefix. The prefix delete and the
// prefix-scoped reuse read share it so both name the same row set.
func relativePathPrefixExpression(prefix string) string {
	return fmt.Sprintf(`%s like "%s%%"`, relativePathFieldName, escapeMilvusLikePattern(prefix))
}

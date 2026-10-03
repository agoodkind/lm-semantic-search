package milvus

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
)

// deleteItemIDBatchSize bounds the item IDs in one delete expression.
const deleteItemIDBatchSize = 256

// DeleteItems removes every row with an item ID column value in ItemIDs and
// every row with a relativePath that starts with one of PathPrefixes. It deletes
// no other row and returns the deleted count. Milvus serves an expression delete
// on a loaded collection only. The caller loads the collection before the call.
func (store *Store) DeleteItems(ctx context.Context, request collection.DeleteItemsRequest) (int64, error) {
	collectionName := strings.TrimSpace(request.Collection)
	if collectionName == "" {
		return 0, errors.New("collection name is required")
	}
	if len(request.ItemIDs) == 0 && len(request.PathPrefixes) == 0 {
		return 0, errors.New("delete items selects no item and no path prefix")
	}
	if len(request.ItemIDs) > 0 && request.Declaration.ItemIDColumn == "" {
		return 0, fmt.Errorf("delete items from %s: declaration has no item ID column", collectionName)
	}
	var removed int64
	for start := 0; start < len(request.ItemIDs); start += deleteItemIDBatchSize {
		end := min(start+deleteItemIDBatchSize, len(request.ItemIDs))
		expression := itemIDsExpression(request.Declaration.ItemIDColumn, request.ItemIDs[start:end])
		count, err := store.deleteByExpression(ctx, collectionName, expression)
		if err != nil {
			return removed, err
		}
		removed += count
	}
	for _, prefix := range request.PathPrefixes {
		if prefix == "" {
			continue
		}
		expression := fmt.Sprintf(`%s like "%s%%"`, RelativePathField, escapeLikePattern(prefix))
		count, err := store.deleteByExpression(ctx, collectionName, expression)
		if err != nil {
			return removed, err
		}
		removed += count
	}
	return removed, nil
}

func itemIDsExpression(itemColumn string, itemIDs []string) string {
	quoted := make([]string, 0, len(itemIDs))
	for _, itemID := range itemIDs {
		quoted = append(quoted, `"`+collection.EscapeString(itemID)+`"`)
	}
	return itemColumn + " in [" + strings.Join(quoted, ", ") + "]"
}

func (store *Store) deleteByExpression(ctx context.Context, collectionName string, expression string) (int64, error) {
	result, err := store.client.Delete(ctx, milvusclient.NewDeleteOption(collectionName).WithExpr(expression))
	if err != nil {
		return 0, WrapError(ctx, err, "delete items from "+collectionName)
	}
	return result.DeleteCount, nil
}

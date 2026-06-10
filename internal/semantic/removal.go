package semantic

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

// Removal names the stored rows one delta step drops before inserting the
// item's fresh chunks. Paths match a row's relativePath exactly, which a code
// file uses because all its chunks share one relativePath. Prefixes match every
// row whose relativePath begins with the prefix, which a conversation uses
// because its messages span many relativePaths under one conv/<id>/ prefix.
type Removal struct {
	Paths    []string
	Prefixes []string
}

// Empty reports whether the removal would delete nothing.
func (removal Removal) Empty() bool {
	return len(removal.Paths) == 0 && len(removal.Prefixes) == 0
}

// RemovePaths builds a removal that drops rows by exact relativePath, the code
// file shape.
func RemovePaths(paths []string) Removal {
	return Removal{Paths: paths, Prefixes: nil}
}

// RemovePrefixes builds a removal that drops rows by relativePath prefix, the
// conversation shape.
func RemovePrefixes(prefixes []string) Removal {
	return Removal{Paths: nil, Prefixes: prefixes}
}

// deleteByRemoval drops an item's prior rows by exact relativePath, by
// relativePath prefix, or both. The prefix branch loads the collection first
// because Milvus serves an expression-filtered Delete only on a loaded
// collection, and a daemon that did not create this collection never loaded it.
func (service *Service) deleteByRemoval(ctx context.Context, collectionName string, removal Removal) error {
	if len(removal.Paths) > 0 {
		if err := service.deleteByRelativePaths(ctx, collectionName, removal.Paths); err != nil {
			return err
		}
	}
	if len(removal.Prefixes) == 0 {
		return nil
	}
	if err := service.loadCollection(ctx, collectionName); err != nil {
		return err
	}
	for _, prefix := range removal.Prefixes {
		if err := service.deleteByRelativePathPrefix(ctx, collectionName, prefix); err != nil {
			return err
		}
	}
	return nil
}

// deleteByRelativePathPrefix removes every row whose relativePath begins with
// prefix. A conversation uses it to drop all of one conversation's message rows
// in a single expression delete.
func (service *Service) deleteByRelativePathPrefix(ctx context.Context, collectionName string, prefix string) error {
	if prefix == "" {
		return nil
	}
	expression := relativePathPrefixExpression(prefix)
	if _, err := service.milvus.Delete(ctx, milvusclient.NewDeleteOption(collectionName).WithExpr(expression)); err != nil {
		slog.ErrorContext(ctx, "delete by relative path prefix failed", "collection", collectionName, "prefix", prefix, "err", err)
		return fmt.Errorf("delete from %s by relative path prefix: %w", collectionName, err)
	}
	return nil
}

// relativePathPrefixExpression renders the Milvus filter expression matching
// every row whose relativePath begins with prefix. The prefix delete and the
// prefix-scoped reuse read share it so both name the same row set.
func relativePathPrefixExpression(prefix string) string {
	return fmt.Sprintf(`%s like "%s%%"`, relativePathFieldName, escapeMilvusString(prefix))
}

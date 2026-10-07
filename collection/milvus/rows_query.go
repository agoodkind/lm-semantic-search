package milvus

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
)

const rowsPageSize = 1000

// QueryRows returns every stored row of the requested items. A query iterator
// pages through the result, and the call returns more rows than the
// single-query ceiling. A row matches by item ID column value or by a
// relativePath prefix.
func (store *Store) QueryRows(ctx context.Context, request collection.RowsRequest) ([]collection.StoredRow, error) {
	collectionName := strings.TrimSpace(request.Collection)
	if collectionName == "" {
		return nil, errors.New("collection name is required")
	}
	expression, err := rowsFilterExpression(request)
	if err != nil {
		slog.ErrorContext(ctx, "build stored row filter failed", "collection", collectionName, "err", err)
		return nil, fmt.Errorf("build stored row filter for %s: %w", collectionName, err)
	}
	outputFields := []string{IDField, ContentField, RelativePathField, SplitPartField, EmbeddingModelField, ContentHashField}
	if request.IncludeVector {
		outputFields = append(outputFields, DenseVectorField)
	}
	for _, declared := range request.Declaration.Scalars {
		outputFields = append(outputFields, declared.Name)
	}
	iterator, err := store.client.QueryIterator(ctx, milvusclient.NewQueryIteratorOption(collectionName).
		WithBatchSize(rowsPageSize).
		WithFilter(expression).
		WithOutputFields(outputFields...))
	if err != nil {
		return nil, SearchError(ctx, "open stored row iterator for", collectionName, err)
	}
	rows := make([]collection.StoredRow, 0)
	for {
		resultSet, nextErr := iterator.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			return rows, nil
		}
		if nextErr != nil {
			return nil, SearchError(ctx, "iterate stored rows of", collectionName, nextErr)
		}
		page, pageErr := storedRowsFromResultSet(resultSet, request)
		if pageErr != nil {
			return nil, pageErr
		}
		rows = append(rows, page...)
	}
}

func rowsFilterExpression(request collection.RowsRequest) (string, error) {
	clauses := make([]string, 0, 1+len(request.PathPrefixes))
	if len(request.ItemIDs) > 0 {
		if request.Declaration.ItemIDColumn == "" {
			return "", errors.New("declaration has no item ID column")
		}
		quoted := make([]string, 0, len(request.ItemIDs))
		for _, itemID := range request.ItemIDs {
			quoted = append(quoted, `"`+collection.EscapeString(itemID)+`"`)
		}
		clauses = append(clauses, request.Declaration.ItemIDColumn+" in ["+strings.Join(quoted, ", ")+"]")
	}
	for _, prefix := range request.PathPrefixes {
		clauses = append(clauses, fmt.Sprintf(`%s like "%s%%"`, RelativePathField, escapeLikePattern(prefix)))
	}
	if len(clauses) == 0 {
		return "", errors.New("request selects no item and no path prefix")
	}
	return "(" + strings.Join(clauses, " or ") + ")", nil
}

// escapeLikePattern escapes the like wildcards of a prefix and then the string
// literal characters.
func escapeLikePattern(value string) string {
	value = strings.ReplaceAll(value, "%", `\%`)
	value = strings.ReplaceAll(value, "_", `\_`)
	return collection.EscapeString(value)
}

// storedRowsFromResultSet decodes one page. The id, content, and relativePath
// columns are required. The reuse identity columns, the split position, and the
// declared scalars are optional, and a missing or null value reads empty.
func storedRowsFromResultSet(resultSet milvusclient.ResultSet, request collection.RowsRequest) ([]collection.StoredRow, error) {
	if resultSet.ResultCount == 0 {
		return nil, nil
	}
	idColumn := resultSet.GetColumn(IDField)
	contentColumn := resultSet.GetColumn(ContentField)
	pathColumn := resultSet.GetColumn(RelativePathField)
	if idColumn == nil || contentColumn == nil || pathColumn == nil {
		return nil, collection.ErrSearchResultIncomplete
	}
	var vectorColumn column.Column
	if request.IncludeVector {
		vectorColumn = resultSet.GetColumn(DenseVectorField)
		if vectorColumn == nil {
			return nil, collection.ErrSearchResultIncomplete
		}
	}
	modelColumn := resultSet.GetColumn(EmbeddingModelField)
	hashColumn := resultSet.GetColumn(ContentHashField)
	splitPartColumn := resultSet.GetColumn(SplitPartField)
	rows := make([]collection.StoredRow, 0, resultSet.ResultCount)
	for i := range resultSet.ResultCount {
		row := collection.StoredRow{Scalars: make(map[string]collection.ScalarCell, len(request.Declaration.Scalars))}
		var err error
		if row.ID, err = idColumn.GetAsString(i); err != nil {
			return nil, rowReadError("id", i, err)
		}
		if row.Content, err = contentColumn.GetAsString(i); err != nil {
			return nil, rowReadError("content", i, err)
		}
		if row.RelativePath, err = pathColumn.GetAsString(i); err != nil {
			return nil, rowReadError("relative path", i, err)
		}
		if row.EmbeddingModel, err = optionalStringAt(modelColumn, i); err != nil {
			return nil, rowReadError("embedding model", i, err)
		}
		if row.ContentHash, err = optionalStringAt(hashColumn, i); err != nil {
			return nil, rowReadError("content hash", i, err)
		}
		if row.SplitPart, row.SplitPartRecorded, err = SplitPartAt(splitPartColumn, i); err != nil {
			return nil, err
		}
		if vectorColumn != nil {
			if row.Vector, err = VectorAt(vectorColumn, i); err != nil {
				return nil, err
			}
		}
		for _, declared := range request.Declaration.Scalars {
			cell, cellErr := ScalarCellAt(resultSet.GetColumn(declared.Name), declared, i)
			if cellErr != nil {
				return nil, cellErr
			}
			row.Scalars[declared.Name] = cell
		}
		rows = append(rows, row)
	}
	return rows, nil
}

// optionalStringAt reads a nullable string column. A missing column and a null
// value both read empty.
func optionalStringAt(valueColumn column.Column, rowIndex int) (string, error) {
	if valueColumn == nil {
		return "", nil
	}
	isNull, err := valueColumn.IsNull(rowIndex)
	if err != nil {
		slog.Error("read stored row null state failed", "index", rowIndex, "err", err)
		return "", fmt.Errorf("read null state: %w", err)
	}
	if isNull {
		return "", nil
	}
	value, err := valueColumn.GetAsString(rowIndex)
	if err != nil {
		slog.Error("read stored row string failed", "index", rowIndex, "err", err)
		return "", fmt.Errorf("read string: %w", err)
	}
	return value, nil
}

func rowReadError(field string, rowIndex int, err error) error {
	slog.Error("read stored row column failed", "column", field, "index", rowIndex, "err", err)
	return fmt.Errorf("read %s column at %d: %w", field, rowIndex, err)
}

package milvus

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
)

// Query returns the rows a filter matches without ranking. A nil filter needs a
// positive Limit, because Milvus refuses an unbounded query.
func (store *Store) Query(ctx context.Context, request collection.QueryRequest) ([]collection.Hit, error) {
	collectionName := strings.TrimSpace(request.Collection)
	if collectionName == "" {
		return nil, errors.New("collection name is required")
	}
	compiled, err := collection.Compile(request.Filter)
	if err != nil {
		slog.ErrorContext(ctx, "compile collection filter failed", "collection", collectionName, "err", err)
		return nil, fmt.Errorf("compile filter for %s: %w", collectionName, err)
	}
	if compiled.Expression == "" && request.Limit <= 0 {
		return nil, fmt.Errorf("query %s: a query without a filter needs a positive limit", collectionName)
	}
	outputFields := []string{
		IDField,
		ContentField,
		RelativePathField,
		StartLineField,
		EndLineField,
		FileExtensionField,
		MetadataField,
		SplitPartField,
	}
	for _, declared := range request.Declaration.Scalars {
		outputFields = append(outputFields, declared.Name)
	}
	option := milvusclient.NewQueryOption(collectionName).WithOutputFields(outputFields...)
	if compiled.Expression != "" {
		option = option.WithFilter(compiled.Expression)
	}
	for _, param := range compiled.Params {
		switch param.Type {
		case collection.ScalarTypeBool:
			option = option.WithTemplateParam(param.Name, param.Bools)
		case collection.ScalarTypeInt64:
			option = option.WithTemplateParam(param.Name, param.Int64s)
		case collection.ScalarTypeString:
			option = option.WithTemplateParam(param.Name, param.Strings)
		default:
			option = option.WithTemplateParam(param.Name, param.Strings)
		}
	}
	if request.Limit > 0 {
		option = option.WithLimit(request.Limit)
	}
	resultSet, err := store.client.Query(ctx, option)
	if err != nil {
		return nil, SearchError(ctx, "query rows", collectionName, err)
	}
	return HitsFromResultSet(resultSet, request.Declaration.Scalars)
}

// Delete removes the rows a filter matches and returns the deleted count. The
// delete request cannot bind template parameters, so membership sets are
// written inline.
func (store *Store) Delete(ctx context.Context, collectionName string, filter collection.Filter) (int64, error) {
	expression, err := collection.CompileInline(&filter)
	if err != nil {
		slog.ErrorContext(ctx, "compile collection filter failed", "collection", collectionName, "err", err)
		return 0, fmt.Errorf("compile filter for %s: %w", collectionName, err)
	}
	result, err := store.client.Delete(ctx, milvusclient.NewDeleteOption(collectionName).WithExpr(expression))
	if err != nil {
		return 0, WrapError(ctx, err, "delete from "+collectionName)
	}
	return result.DeleteCount, nil
}

package milvus

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
)

const (
	bm25FunctionName   = "bm25"
	bm25DropRatioBuild = 0.2
)

// ScalarFieldSchemas builds the Milvus field definitions for declared scalar
// columns, in declaration order.
func ScalarFieldSchemas(columns []collection.ScalarColumn) []*entity.Field {
	fields := make([]*entity.Field, 0, len(columns))
	for _, declared := range columns {
		fields = append(fields, ScalarFieldSchema(declared))
	}
	return fields
}

// ScalarFieldSchema builds the Milvus field definition for one declared scalar
// column. A string column becomes a VarChar with the declared maximum length.
func ScalarFieldSchema(declared collection.ScalarColumn) *entity.Field {
	field := entity.NewField().WithName(declared.Name)
	switch declared.Type {
	case collection.ScalarTypeString:
		field = field.WithDataType(entity.FieldTypeVarChar).WithMaxLength(int64(declared.MaxLength))
	case collection.ScalarTypeBool:
		field = field.WithDataType(entity.FieldTypeBool)
	case collection.ScalarTypeInt64:
		field = field.WithDataType(entity.FieldTypeInt64)
	default:
		field = field.WithDataType(entity.FieldTypeNone)
	}
	return field.WithNullable(declared.Nullable)
}

// SplitPartFieldSchema is the nullable splitPart field definition.
func SplitPartFieldSchema() *entity.Field {
	return entity.NewField().
		WithName(SplitPartField).
		WithDataType(entity.FieldTypeInt64).
		WithNullable(true)
}

// ContentHashFieldSchema is the nullable contentHash field definition.
func ContentHashFieldSchema() *entity.Field {
	return entity.NewField().
		WithName(ContentHashField).
		WithDataType(entity.FieldTypeVarChar).
		WithMaxLength(contentHashFieldMaxLength).
		WithNullable(true)
}

// EmbeddingModelFieldSchema is the nullable embeddingModel field definition.
func EmbeddingModelFieldSchema() *entity.Field {
	return entity.NewField().
		WithName(EmbeddingModelField).
		WithDataType(entity.FieldTypeVarChar).
		WithMaxLength(embeddingModelMaxLength).
		WithNullable(true)
}

// CreateCollectionOption builds the request that creates a collection with the
// built-in columns, the declared scalar columns, a dense vector of width
// dimension, and the content hash index. A hybrid store also adds the sparse
// vector, its BM25 function, and its index. Milvus 2.6 rejects mmap.enabled on
// AUTOINDEX creation. The caller applies the mmap policy through field and
// index property changes after every required index exists.
func (store *Store) CreateCollectionOption(collectionName string, dimension int, scalars []collection.ScalarColumn) milvusclient.CreateCollectionOption {
	schema := entity.NewSchema().
		WithField(entity.NewField().WithName(IDField).WithDataType(entity.FieldTypeVarChar).WithMaxLength(idFieldMaxLength).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName(ContentField).WithDataType(entity.FieldTypeVarChar).WithMaxLength(contentFieldMaxLength).WithEnableAnalyzer(true).WithEnableMatch(true)).
		WithField(entity.NewField().WithName(RelativePathField).WithDataType(entity.FieldTypeVarChar).WithMaxLength(relativePathFieldMaxLength)).
		WithField(entity.NewField().WithName(StartLineField).WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName(EndLineField).WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName(FileExtensionField).WithDataType(entity.FieldTypeVarChar).WithMaxLength(fileExtensionFieldMaxLength)).
		WithField(entity.NewField().WithName(MetadataField).WithDataType(entity.FieldTypeVarChar).WithMaxLength(metadataFieldMaxLength)).
		WithField(ContentHashFieldSchema()).
		WithField(EmbeddingModelFieldSchema()).
		WithField(SplitPartFieldSchema()).
		WithField(entity.NewField().WithName(DenseVectorField).WithDataType(entity.FieldTypeFloatVector).WithDim(int64(dimension)))

	for _, field := range ScalarFieldSchemas(scalars) {
		schema = schema.WithField(field)
	}

	indexOptions := []milvusclient.CreateIndexOption{
		milvusclient.NewCreateIndexOption(collectionName, DenseVectorField, index.NewAutoIndex(entity.COSINE)),
		milvusclient.NewCreateIndexOption(collectionName, ContentHashField, index.NewInvertedIndex()),
	}

	if store.options.Hybrid {
		schema = schema.
			WithField(entity.NewField().WithName(SparseVectorField).WithDataType(entity.FieldTypeSparseVector)).
			WithFunction(entity.NewFunction().WithName(bm25FunctionName).WithType(entity.FunctionTypeBM25).WithInputFields(ContentField).WithOutputFields(SparseVectorField))
		indexOptions = append(indexOptions, milvusclient.NewCreateIndexOption(collectionName, SparseVectorField, index.NewSparseInvertedIndex(entity.BM25, bm25DropRatioBuild)))
	}

	return milvusclient.NewCreateCollectionOption(collectionName, schema).WithIndexOptions(indexOptions...)
}

// CreateCollection creates a collection from [Store.CreateCollectionOption].
func (store *Store) CreateCollection(ctx context.Context, collectionName string, dimension int, scalars []collection.ScalarColumn) error {
	if err := store.client.CreateCollection(ctx, store.CreateCollectionOption(collectionName, dimension, scalars)); err != nil {
		return WrapError(ctx, err, "create Milvus collection "+collectionName)
	}
	return nil
}

// AddMissingScalarColumns reads the collection schema and adds, through the
// Milvus AddCollectionField API, any column in columns the collection lacks.
// It returns the names it added. It backfills no value. Every added column must
// be nullable, which AddCollectionField requires for a collection that already
// stores rows.
func (store *Store) AddMissingScalarColumns(ctx context.Context, collectionName string, columns []collection.ScalarColumn) ([]string, error) {
	described, err := store.client.DescribeCollection(ctx, milvusclient.NewDescribeCollectionOption(collectionName))
	if err != nil {
		slog.ErrorContext(ctx, "describe collection for scalar migration failed", "collection", collectionName, "err", err)
		return nil, fmt.Errorf("describe collection %s: %w", collectionName, err)
	}
	existing := make(map[string]struct{})
	if described.Schema != nil {
		for _, field := range described.Schema.Fields {
			existing[field.Name] = struct{}{}
		}
	}
	added := make([]string, 0, len(columns))
	for _, field := range ScalarFieldSchemas(columns) {
		if _, found := existing[field.Name]; found {
			continue
		}
		if err := store.client.AddCollectionField(ctx, milvusclient.NewAddCollectionFieldOption(collectionName, field)); err != nil {
			slog.ErrorContext(ctx, "add scalar column failed", "collection", collectionName, "field", field.Name, "err", err)
			return added, fmt.Errorf("add scalar column %s to %s: %w", field.Name, collectionName, err)
		}
		added = append(added, field.Name)
	}
	return added, nil
}

// EnsureCollection creates the collection when it is absent. A collection that
// exists gains any declared scalar column it lacks.
func (store *Store) EnsureCollection(ctx context.Context, request collection.EnsureRequest) error {
	collectionName := strings.TrimSpace(request.Collection)
	if collectionName == "" {
		return errors.New("collection name is required")
	}
	exists, err := store.client.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		return WrapError(ctx, err, "check Milvus collection "+collectionName)
	}
	if !exists {
		if request.Dimension <= 0 {
			return fmt.Errorf("create Milvus collection %s: dimension %d is not positive", collectionName, request.Dimension)
		}
		return store.CreateCollection(ctx, collectionName, request.Dimension, request.Declaration.Scalars)
	}
	_, err = store.AddMissingScalarColumns(ctx, collectionName, request.Declaration.Scalars)
	return err
}

package semantic

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

func (service *Service) createCollection(ctx context.Context, collectionName string, dimension int) error {
	schema := entity.NewSchema().
		WithField(entity.NewField().WithName(idFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(512).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName(contentFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(65535).WithEnableAnalyzer(true).WithEnableMatch(true)).
		WithField(entity.NewField().WithName(relativePathFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(1024)).
		WithField(entity.NewField().WithName(startLineFieldName).WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName(endLineFieldName).WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName(fileExtensionFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(32)).
		WithField(entity.NewField().WithName(metadataFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(65535)).
		WithField(entity.NewField().WithName(denseVectorFieldName).WithDataType(entity.FieldTypeFloatVector).WithDim(int64(dimension)))

	indexOptions := []milvusclient.CreateIndexOption{
		milvusclient.NewCreateIndexOption(collectionName, denseVectorFieldName, index.NewAutoIndex(entity.COSINE)),
	}

	if service.cfg.HybridMode {
		schema = schema.
			WithField(entity.NewField().WithName(sparseVectorFieldName).WithDataType(entity.FieldTypeSparseVector)).
			WithFunction(entity.NewFunction().WithName("bm25").WithType(entity.FunctionTypeBM25).WithInputFields(contentFieldName).WithOutputFields(sparseVectorFieldName))
		indexOptions = append(indexOptions, milvusclient.NewCreateIndexOption(collectionName, sparseVectorFieldName, index.NewSparseInvertedIndex(entity.BM25, 0.2)))
	}

	if err := service.milvus.CreateCollection(ctx, milvusclient.NewCreateCollectionOption(collectionName, schema).WithIndexOptions(indexOptions...)); err != nil {
		slog.ErrorContext(ctx, "create Milvus collection failed", "collection", collectionName, "err", err)
		return fmt.Errorf("create Milvus collection %s: %w", collectionName, err)
	}
	return service.loadCollection(ctx, collectionName)
}

// loadCollection loads collectionName into memory and waits for the load to
// finish. A loaded collection is what makes an expression-filtered delete
// usable: Milvus answers a Delete(WithExpr(...)) on a non-primary field by
// first querying for the matching ids, which requires the collection to be
// loaded, and rejects the delete with "collection not loaded" otherwise.
// createCollection runs it once for a freshly built collection; the
// conversation upsert and delete paths run it against an already-existing
// collection before their prefix delete, since a daemon process that did not
// create the collection itself never loaded it.
func (service *Service) loadCollection(ctx context.Context, collectionName string) error {
	loadTask, err := service.milvus.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(collectionName))
	if err != nil {
		slog.ErrorContext(ctx, "load Milvus collection failed", "collection", collectionName, "err", err)
		return fmt.Errorf("load Milvus collection %s: %w", collectionName, err)
	}
	if err := loadTask.Await(ctx); err != nil {
		slog.ErrorContext(ctx, "await Milvus collection load failed", "collection", collectionName, "err", err)
		return fmt.Errorf("await Milvus collection load %s: %w", collectionName, err)
	}
	return nil
}

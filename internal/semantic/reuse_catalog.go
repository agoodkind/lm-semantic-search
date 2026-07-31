package semantic

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/config"
)

const (
	reuseCatalogCollectionPrefix = "content_vector_catalog_"
	reuseCatalogRowKeyFieldName  = "catalogKey"
)

type reuseCatalogVectors map[string][]float32

type reuseCatalogEntry struct {
	embeddingModel string
	vector         []float32
}

type reuseCatalogEntries map[string][]reuseCatalogEntry

// ReuseCatalogCollectionName returns the state-root-scoped catalog.
func ReuseCatalogCollectionName(cfg config.Config) string {
	stateRootSum := sha256.Sum256([]byte(cfg.StateRoot))
	stateRootIdentity := hex.EncodeToString(stateRootSum[:])
	return reuseCatalogCollectionPrefix + stateRootIdentity
}

func (service *Service) reuseCatalogAvailable(
	ctx context.Context,
	createDimension int,
) (bool, error) {
	if service.reuseCatalogReady.Load() {
		return true, nil
	}
	service.reuseCatalogMutex.Lock()
	defer service.reuseCatalogMutex.Unlock()
	if service.reuseCatalogReady.Load() {
		return true, nil
	}

	collectionName := ReuseCatalogCollectionName(service.cfg)
	exists, err := service.hasCollection(
		ctx,
		collectionName,
		"check content vector reuse catalog",
	)
	if err != nil {
		return false, err
	}
	if !exists {
		if createDimension <= 0 {
			return false, nil
		}
		if err := service.createReuseCatalog(ctx, collectionName, createDimension); err != nil {
			return false, err
		}
	} else if err := service.loadCollectionForRead(ctx, collectionName); err != nil {
		return false, err
	}
	service.reuseCatalogReady.Store(true)
	return true, nil
}

func (service *Service) createReuseCatalog(
	ctx context.Context,
	collectionName string,
	dimension int,
) error {
	schema := entity.NewSchema().
		WithField(entity.NewField().
			WithName(reuseCatalogRowKeyFieldName).
			WithDataType(entity.FieldTypeVarChar).
			WithMaxLength(64).
			WithIsPrimaryKey(true)).
		WithField(entity.NewField().
			WithName(contentHashFieldName).
			WithDataType(entity.FieldTypeVarChar).
			WithMaxLength(64)).
		WithField(embeddingModelField()).
		WithField(entity.NewField().
			WithName(denseVectorFieldName).
			WithDataType(entity.FieldTypeFloatVector).
			WithDim(int64(dimension)))
	option := milvusclient.NewCreateCollectionOption(collectionName, schema).
		WithIndexOptions(
			milvusclient.NewCreateIndexOption(
				collectionName,
				denseVectorFieldName,
				index.NewAutoIndex(entity.COSINE),
			),
			milvusclient.NewCreateIndexOption(
				collectionName,
				contentHashFieldName,
				index.NewInvertedIndex(),
			),
		)
	if err := service.milvus.CreateCollection(ctx, option); err != nil {
		exists, existsErr := service.hasCollection(
			ctx,
			collectionName,
			"recheck content vector reuse catalog after create failure",
		)
		if existsErr != nil {
			combinedErr := errors.Join(err, existsErr)
			slog.ErrorContext(
				ctx,
				"recheck content vector reuse catalog failed",
				"collection", collectionName,
				"err", combinedErr,
			)
			return combinedErr
		}
		if !exists {
			return wrapStoreError(ctx, err, "create content vector reuse catalog")
		}
	}
	service.invalidateCollectionCaches(collectionName)
	if err := service.loadCollection(ctx, collectionName); err != nil {
		return err
	}
	return nil
}

func (service *Service) loadReuseCatalogKeys(
	ctx context.Context,
	storageKeys []string,
) (reuseCatalogEntries, error) {
	// A completed catalog append must be visible to the next changed item or
	// corpus so that one embedding remains sufficient.
	return service.queryReuseCatalogKeys(ctx, storageKeys, entity.ClStrong)
}

func (service *Service) queryReuseCatalogKeys(
	ctx context.Context,
	storageKeys []string,
	consistency entity.ConsistencyLevel,
) (reuseCatalogEntries, error) {
	entries := make(reuseCatalogEntries, len(storageKeys))
	if len(storageKeys) == 0 {
		return entries, nil
	}
	collectionName := ReuseCatalogCollectionName(service.cfg)
	resultSet, err := service.milvus.Query(
		ctx,
		milvusclient.NewQueryOption(collectionName).
			WithFilter(inStringClause(contentHashFieldName, storageKeys)).
			WithOutputFields(contentHashFieldName, embeddingModelFieldName, denseVectorFieldName).
			WithConsistencyLevel(consistency),
	)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"query content vector reuse catalog failed",
			"collection", collectionName,
			"err", err,
		)
		return nil, fmt.Errorf("query content vector reuse catalog: %w", err)
	}
	if resultSet.ResultCount == 0 {
		return entries, nil
	}
	keyColumn := resultSet.GetColumn(contentHashFieldName)
	embeddingModelColumn := resultSet.GetColumn(embeddingModelFieldName)
	vectorColumn := resultSet.GetColumn(denseVectorFieldName)
	if keyColumn == nil || vectorColumn == nil {
		return nil, ErrSearchResultIncomplete
	}
	for rowIndex := range resultSet.ResultCount {
		storageKey, keyErr := keyColumn.GetAsString(rowIndex)
		if keyErr != nil {
			return nil, fmt.Errorf("read catalog content hash at %d: %w", rowIndex, keyErr)
		}
		embeddingModel, modelErr := nullableStringAt(embeddingModelColumn, rowIndex)
		if modelErr != nil {
			return nil, fmt.Errorf("read catalog embedding model at %d: %w", rowIndex, modelErr)
		}
		vector, vectorErr := vectorAt(vectorColumn, rowIndex)
		if vectorErr != nil {
			return nil, vectorErr
		}
		entries[storageKey] = append(entries[storageKey], reuseCatalogEntry{
			embeddingModel: embeddingModel,
			vector:         vector,
		})
	}
	return entries, nil
}

func reuseCatalogRowKey(contentHashValue string, embeddingModel string) string {
	normalizedModel, _ := sanitizeUTF8(embeddingModel)
	sum := sha256.Sum256([]byte(contentHashValue + "\x00" + normalizedModel))
	return hex.EncodeToString(sum[:])
}

func (service *Service) loadReuseCatalogRowKeys(
	ctx context.Context,
	rowKeys []string,
) (map[string]struct{}, error) {
	existing := make(map[string]struct{}, len(rowKeys))
	resultSet, err := service.milvus.Get(
		ctx,
		milvusclient.NewQueryOption(ReuseCatalogCollectionName(service.cfg)).
			WithIDs(column.NewColumnVarChar(reuseCatalogRowKeyFieldName, rowKeys)).
			WithOutputFields(reuseCatalogRowKeyFieldName).
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		wrappedErr := fmt.Errorf("query content vector catalog row keys: %w", err)
		slog.ErrorContext(ctx, "query content vector catalog row keys failed", "err", wrappedErr)
		return nil, wrappedErr
	}
	rowKeyColumn := resultSet.GetColumn(reuseCatalogRowKeyFieldName)
	if resultSet.ResultCount > 0 && rowKeyColumn == nil {
		return nil, ErrSearchResultIncomplete
	}
	for rowIndex := range resultSet.ResultCount {
		rowKey, keyErr := rowKeyColumn.GetAsString(rowIndex)
		if keyErr != nil {
			return nil, fmt.Errorf("read catalog row key at %d: %w", rowIndex, keyErr)
		}
		existing[rowKey] = struct{}{}
	}
	return existing, nil
}

func (service *Service) appendReuseCatalog(
	ctx context.Context,
	vectorsByContent reuseCatalogVectors,
) error {
	if !service.Available() || len(vectorsByContent) == 0 {
		return nil
	}
	vectorsByStorageKey := make(map[string][]float32, len(vectorsByContent))
	dimension := 0
	for rawContent, vector := range vectorsByContent {
		content, _ := sanitizeUTF8(rawContent)
		if len(vector) == 0 {
			continue
		}
		if dimension == 0 {
			dimension = len(vector)
		}
		if len(vector) != dimension {
			return fmt.Errorf(
				"append content vector reuse catalog: vector dimensions %d and %d differ",
				dimension,
				len(vector),
			)
		}
		storageKey := contentHash(content)
		vectorsByStorageKey[storageKey] = vector
	}
	if len(vectorsByStorageKey) == 0 {
		return nil
	}

	service.reuseCatalogAppendMutex.Lock()
	defer service.reuseCatalogAppendMutex.Unlock()
	if _, err := service.reuseCatalogAvailable(ctx, dimension); err != nil {
		return err
	}
	storageKeys := make([]string, 0, len(vectorsByStorageKey))
	rowKeysByStorageKey := make(map[string]string, len(vectorsByStorageKey))
	for storageKey := range vectorsByStorageKey {
		storageKeys = append(storageKeys, storageKey)
		rowKeysByStorageKey[storageKey] = reuseCatalogRowKey(
			storageKey,
			service.cfg.EmbeddingModel,
		)
	}
	rowKeys := make([]string, 0, len(storageKeys))
	for _, storageKey := range storageKeys {
		rowKeys = append(rowKeys, rowKeysByStorageKey[storageKey])
	}
	existing, err := service.loadReuseCatalogRowKeys(ctx, rowKeys)
	if err != nil {
		return err
	}
	missingKeys := make([]string, 0, len(storageKeys))
	missingRowKeys := make([]string, 0, len(storageKeys))
	missingVectors := make([][]float32, 0, len(storageKeys))
	for _, storageKey := range storageKeys {
		rowKey := rowKeysByStorageKey[storageKey]
		if _, found := existing[rowKey]; found {
			continue
		}
		missingKeys = append(missingKeys, storageKey)
		missingRowKeys = append(missingRowKeys, rowKey)
		missingVectors = append(missingVectors, vectorsByStorageKey[storageKey])
	}
	if len(missingKeys) == 0 {
		return nil
	}
	embeddingModelColumn, err := newEmbeddingModelColumn(
		ReuseCatalogCollectionName(service.cfg),
		service.cfg.EmbeddingModel,
		len(missingKeys),
	)
	if err != nil {
		return err
	}
	result, err := service.milvus.Insert(
		ctx,
		milvusclient.NewColumnBasedInsertOption(ReuseCatalogCollectionName(service.cfg)).
			WithVarcharColumn(reuseCatalogRowKeyFieldName, missingRowKeys).
			WithVarcharColumn(contentHashFieldName, missingKeys).
			WithColumns(embeddingModelColumn).
			WithFloatVectorColumn(denseVectorFieldName, dimension, missingVectors),
	)
	if err != nil {
		return wrapStoreError(ctx, err, "append content vector reuse catalog")
	}
	if result.InsertCount != int64(len(missingKeys)) {
		return fmt.Errorf(
			"append content vector reuse catalog: Milvus acknowledged %d of %d rows",
			result.InsertCount,
			len(missingKeys),
		)
	}
	slog.DebugContext(ctx, "semantic.reuse_catalog_appended", "count", len(missingKeys))
	return nil
}

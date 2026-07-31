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

const reuseCatalogCollectionPrefix = "content_vector_catalog_"

type reuseCatalogVectors map[string][]float32

// ReuseCatalogCollectionName returns the state-root and identity-scoped catalog.
func ReuseCatalogCollectionName(cfg config.Config) string {
	stateRootSum := sha256.Sum256([]byte(cfg.StateRoot))
	stateRootIdentity := hex.EncodeToString(stateRootSum[:])
	return reuseCatalogCollectionPrefix + stateRootIdentity + "_" + embeddingIdentity(cfg)
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
			WithName(contentVectorKeyFieldName).
			WithDataType(entity.FieldTypeVarChar).
			WithMaxLength(64).
			WithIsPrimaryKey(true)).
		WithField(entity.NewField().
			WithName(denseVectorFieldName).
			WithDataType(entity.FieldTypeFloatVector).
			WithDim(int64(dimension)))
	option := milvusclient.NewCreateCollectionOption(collectionName, schema).
		WithIndexOptions(milvusclient.NewCreateIndexOption(
			collectionName,
			denseVectorFieldName,
			index.NewAutoIndex(entity.COSINE),
		))
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
) (map[string][]float32, error) {
	// A completed catalog append must be visible to the next changed item or
	// corpus so that one embedding remains sufficient.
	return service.queryReuseCatalogKeys(ctx, storageKeys, entity.ClStrong)
}

func (service *Service) loadReuseCatalogKeysForAppend(
	ctx context.Context,
	storageKeys []string,
) (map[string][]float32, error) {
	return service.queryReuseCatalogKeys(ctx, storageKeys, entity.ClStrong)
}

func (service *Service) queryReuseCatalogKeys(
	ctx context.Context,
	storageKeys []string,
	consistency entity.ConsistencyLevel,
) (map[string][]float32, error) {
	vectors := make(map[string][]float32, len(storageKeys))
	if len(storageKeys) == 0 {
		return vectors, nil
	}
	queryKeys := append([]string(nil), storageKeys...)
	collectionName := ReuseCatalogCollectionName(service.cfg)
	resultSet, err := service.milvus.Get(
		ctx,
		milvusclient.NewQueryOption(collectionName).
			WithIDs(column.NewColumnVarChar(contentVectorKeyFieldName, queryKeys)).
			WithOutputFields(contentVectorKeyFieldName, denseVectorFieldName).
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
		return vectors, nil
	}
	keyColumn := resultSet.GetColumn(contentVectorKeyFieldName)
	vectorColumn := resultSet.GetColumn(denseVectorFieldName)
	if keyColumn == nil || vectorColumn == nil {
		return nil, ErrSearchResultIncomplete
	}
	for rowIndex := range resultSet.ResultCount {
		storageKey, keyErr := keyColumn.GetAsString(rowIndex)
		if keyErr != nil {
			return nil, fmt.Errorf("read catalog content vector key at %d: %w", rowIndex, keyErr)
		}
		vector, vectorErr := vectorAt(vectorColumn, rowIndex)
		if vectorErr != nil {
			return nil, vectorErr
		}
		vectors[storageKey] = vector
	}
	return vectors, nil
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
		storageKey := contentVectorStorageKey(service.cfg, content)
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
	for storageKey := range vectorsByStorageKey {
		storageKeys = append(storageKeys, storageKey)
	}
	existing, err := service.loadReuseCatalogKeysForAppend(ctx, storageKeys)
	if err != nil {
		return err
	}
	missingKeys := make([]string, 0, len(storageKeys))
	missingVectors := make([][]float32, 0, len(storageKeys))
	for _, storageKey := range storageKeys {
		if _, found := existing[storageKey]; found {
			continue
		}
		missingKeys = append(missingKeys, storageKey)
		missingVectors = append(missingVectors, vectorsByStorageKey[storageKey])
	}
	if len(missingKeys) == 0 {
		return nil
	}
	result, err := service.milvus.Insert(
		ctx,
		milvusclient.NewColumnBasedInsertOption(ReuseCatalogCollectionName(service.cfg)).
			WithVarcharColumn(contentVectorKeyFieldName, missingKeys).
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

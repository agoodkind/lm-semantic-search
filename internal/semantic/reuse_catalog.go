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

// ReuseCatalogCollectionName returns the catalog for the configured dimension.
func ReuseCatalogCollectionName(cfg config.Config) string {
	return formatReuseCatalogCollectionName(cfg.StateRoot, int(cfg.EmbeddingDimension))
}

func reuseCatalogCollectionName(cfg config.Config, dimension int) string {
	if int64(dimension) == int64(cfg.EmbeddingDimension) {
		return ReuseCatalogCollectionName(cfg)
	}
	return formatReuseCatalogCollectionName(cfg.StateRoot, dimension)
}

func formatReuseCatalogCollectionName(stateRoot string, dimension int) string {
	stateRootSum := sha256.Sum256([]byte(stateRoot))
	stateRootIdentity := hex.EncodeToString(stateRootSum[:])
	return fmt.Sprintf(
		"%s%s_%d",
		reuseCatalogCollectionPrefix,
		stateRootIdentity,
		dimension,
	)
}

func (service *Service) reuseCatalogAvailable(
	ctx context.Context,
	dimension int,
	create bool,
) (bool, error) {
	if dimension <= 0 {
		return false, nil
	}
	if _, ready := service.reuseCatalogReady.Load(dimension); ready {
		return true, nil
	}
	service.reuseCatalogMutex.Lock()
	defer service.reuseCatalogMutex.Unlock()
	if _, ready := service.reuseCatalogReady.Load(dimension); ready {
		return true, nil
	}

	collectionName := reuseCatalogCollectionName(service.cfg, dimension)
	exists, err := service.hasCollection(
		ctx,
		collectionName,
		"check content vector reuse catalog",
	)
	if err != nil {
		return false, err
	}
	if !exists {
		if !create {
			return false, nil
		}
		if err := service.createReuseCatalog(ctx, collectionName, dimension); err != nil {
			return false, err
		}
	}
	service.reuseCatalogReady.Store(dimension, struct{}{})
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
	maintenance, err := service.residency.Maintain(ctx, collectionName)
	if err != nil {
		return err
	}
	if err := service.milvus.CreateCollection(ctx, option); err != nil {
		exists, existsErr := service.hasCollection(
			ctx,
			collectionName,
			"recheck content vector reuse catalog after create failure",
		)
		if existsErr != nil {
			maintenance.ReleaseContext(ctx)
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
			maintenance.ReleaseContext(ctx)
			return wrapStoreError(ctx, err, "create content vector reuse catalog")
		}
	}
	service.invalidateCollectionCaches(collectionName)
	maintenance.ReleaseContext(ctx)
	return nil
}

func (service *Service) loadReuseCatalogKeys(
	ctx context.Context,
	storageKeys []string,
	dimension int,
) (reuseCatalogEntries, error) {
	// Catalog rows are immutable, so a bounded hit is valid. Retry misses with
	// strong consistency so a completed append is visible to the next item.
	if service.cfg.EmbeddingModel != "" {
		return service.getReuseCatalogKeys(ctx, storageKeys, dimension)
	}
	return service.queryReuseCatalogKeys(ctx, storageKeys, dimension, entity.ClStrong)
}

func (service *Service) getReuseCatalogKeys(
	ctx context.Context,
	storageKeys []string,
	dimension int,
) (reuseCatalogEntries, error) {
	entries, err := service.getReuseCatalogModelKeys(
		ctx,
		storageKeys,
		dimension,
		service.cfg.EmbeddingModel,
		entity.ClBounded,
	)
	if err != nil {
		return nil, err
	}
	missingStorageKeys := missingReuseCatalogKeys(storageKeys, entries)
	strongEntries, err := service.getReuseCatalogModelKeys(
		ctx,
		missingStorageKeys,
		dimension,
		service.cfg.EmbeddingModel,
		entity.ClStrong,
	)
	if err != nil {
		return nil, err
	}
	mergeReuseCatalogEntries(entries, strongEntries)
	missingStorageKeys = missingReuseCatalogKeys(storageKeys, entries)
	legacyEntries, err := service.getReuseCatalogModelKeys(
		ctx,
		missingStorageKeys,
		dimension,
		"",
		entity.ClBounded,
	)
	if err != nil {
		return nil, err
	}
	mergeReuseCatalogEntries(entries, legacyEntries)
	missingStorageKeys = missingReuseCatalogKeys(storageKeys, entries)
	strongLegacyEntries, err := service.getReuseCatalogModelKeys(
		ctx,
		missingStorageKeys,
		dimension,
		"",
		entity.ClStrong,
	)
	if err != nil {
		return nil, err
	}
	mergeReuseCatalogEntries(entries, strongLegacyEntries)
	return entries, nil
}

func missingReuseCatalogKeys(
	storageKeys []string,
	entries reuseCatalogEntries,
) []string {
	missingStorageKeys := make([]string, 0, len(storageKeys))
	for _, storageKey := range storageKeys {
		if len(entries[storageKey]) == 0 {
			missingStorageKeys = append(missingStorageKeys, storageKey)
		}
	}
	return missingStorageKeys
}

func mergeReuseCatalogEntries(entries reuseCatalogEntries, additions reuseCatalogEntries) {
	for storageKey, candidates := range additions {
		entries[storageKey] = append(entries[storageKey], candidates...)
	}
}

func (service *Service) getReuseCatalogModelKeys(
	ctx context.Context,
	storageKeys []string,
	dimension int,
	embeddingModel string,
	consistency entity.ConsistencyLevel,
) (reuseCatalogEntries, error) {
	entries := make(reuseCatalogEntries, len(storageKeys))
	if len(storageKeys) == 0 {
		return entries, nil
	}
	rowKeys := make([]string, 0, len(storageKeys))
	storageKeyByRowKey := make(map[string]string, len(storageKeys))
	for _, storageKey := range storageKeys {
		rowKey := reuseCatalogRowKey(storageKey, embeddingModel)
		rowKeys = append(rowKeys, rowKey)
		storageKeyByRowKey[rowKey] = storageKey
	}
	collectionName := reuseCatalogCollectionName(service.cfg, dimension)
	resultSet, err := service.milvus.Get(
		ctx,
		milvusclient.NewQueryOption(collectionName).
			WithIDs(column.NewColumnVarChar(reuseCatalogRowKeyFieldName, rowKeys)).
			WithOutputFields(reuseCatalogRowKeyFieldName, denseVectorFieldName).
			WithConsistencyLevel(consistency),
	)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"get content vector reuse catalog failed",
			"collection", collectionName,
			"err", err,
		)
		return nil, fmt.Errorf("get content vector reuse catalog: %w", err)
	}
	rowKeyColumn := resultSet.GetColumn(reuseCatalogRowKeyFieldName)
	vectorColumn := resultSet.GetColumn(denseVectorFieldName)
	if resultSet.ResultCount > 0 && (rowKeyColumn == nil || vectorColumn == nil) {
		return nil, ErrSearchResultIncomplete
	}
	for rowIndex := range resultSet.ResultCount {
		rowKey, keyErr := rowKeyColumn.GetAsString(rowIndex)
		if keyErr != nil {
			return nil, fmt.Errorf("read catalog row key at %d: %w", rowIndex, keyErr)
		}
		storageKey, found := storageKeyByRowKey[rowKey]
		if !found {
			return nil, fmt.Errorf("read unrequested catalog row key %q", rowKey)
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

func (service *Service) queryReuseCatalogKeys(
	ctx context.Context,
	storageKeys []string,
	dimension int,
	consistency entity.ConsistencyLevel,
) (reuseCatalogEntries, error) {
	entries := make(reuseCatalogEntries, len(storageKeys))
	if len(storageKeys) == 0 {
		return entries, nil
	}
	collectionName := reuseCatalogCollectionName(service.cfg, dimension)
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
	return readReuseCatalogEntries(ctx, resultSet, entries)
}

func readReuseCatalogEntries(
	ctx context.Context,
	resultSet milvusclient.ResultSet,
	entries reuseCatalogEntries,
) (reuseCatalogEntries, error) {
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
			wrappedErr := fmt.Errorf("read catalog content hash at %d: %w", rowIndex, keyErr)
			slog.ErrorContext(ctx, "read catalog content hash failed", "err", wrappedErr)
			return nil, wrappedErr
		}
		embeddingModel, modelErr := nullableStringAt(embeddingModelColumn, rowIndex)
		if modelErr != nil {
			wrappedErr := fmt.Errorf("read catalog embedding model at %d: %w", rowIndex, modelErr)
			slog.ErrorContext(ctx, "read catalog embedding model failed", "err", wrappedErr)
			return nil, wrappedErr
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
	dimension int,
) (map[string]struct{}, error) {
	existing := make(map[string]struct{}, len(rowKeys))
	resultSet, err := service.milvus.Get(
		ctx,
		milvusclient.NewQueryOption(reuseCatalogCollectionName(service.cfg, dimension)).
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
	vectorsByStorageKey, dimension, err := reuseCatalogStorageVectors(vectorsByContent)
	if err != nil {
		return err
	}
	if len(vectorsByStorageKey) == 0 {
		return nil
	}

	service.reuseCatalogAppendMutex.Lock()
	defer service.reuseCatalogAppendMutex.Unlock()
	if _, err := service.reuseCatalogAvailable(ctx, dimension, true); err != nil {
		return err
	}
	collectionName := reuseCatalogCollectionName(service.cfg, dimension)
	lease, err := service.AcquireCollection(ctx, collectionName)
	if err != nil {
		return err
	}
	defer lease.Release()
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
	existing, err := service.loadReuseCatalogRowKeys(ctx, rowKeys, dimension)
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
		reuseCatalogCollectionName(service.cfg, dimension),
		service.cfg.EmbeddingModel,
		len(missingKeys),
	)
	if err != nil {
		return err
	}
	result, err := service.milvus.Insert(
		ctx,
		milvusclient.NewColumnBasedInsertOption(reuseCatalogCollectionName(service.cfg, dimension)).
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

func reuseCatalogStorageVectors(
	vectorsByContent reuseCatalogVectors,
) (map[string][]float32, int, error) {
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
			return nil, 0, fmt.Errorf(
				"append content vector reuse catalog: vector dimensions %d and %d differ",
				dimension,
				len(vector),
			)
		}
		storageKey := contentHash(content)
		vectorsByStorageKey[storageKey] = vector
	}
	return vectorsByStorageKey, dimension, nil
}

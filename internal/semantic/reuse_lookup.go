package semantic

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"math"
	"slices"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/store"
	"google.golang.org/grpc/peer"
)

const (
	reuseLookupBatchSize        = 256
	reuseLookupMaxEscapedBytes  = 256 * 1024
	reuseVectorFetchBudgetBytes = 64 * 1024 * 1024
	reuseVectorRowFramingBytes  = 256
	reuseVectorFloatBytes       = 4
)

func contentHash(content string) string {
	normalized, _ := sanitizeUTF8(content)
	return contentVectorKey(normalized)
}

func contentHashes(contents []string) []string {
	hashes := make([]string, 0, len(contents))
	for _, content := range contents {
		hashes = append(hashes, contentHash(content))
	}
	return hashes
}

// LoadReuseVectorsForContents resolves only vectors needed by chunks. It reads
// the content catalog first, then the target collection, then resident fallback
// collections. Stored vectors qualify unless both stored and current model
// names are known and unequal. Every lookup is read-only.
func (service *Service) LoadReuseVectorsForContents(
	ctx context.Context,
	collectionName string,
	chunks []model.StoredChunk,
) (map[string][]float32, error) {
	reuse := make(map[string][]float32)
	if !service.Available() || len(chunks) == 0 {
		return reuse, nil
	}
	contentsByStorageKey, storageKeys := service.reuseCandidates(chunks)
	if err := service.loadCatalogReuse(ctx, storageKeys, contentsByStorageKey, reuse); err != nil {
		return nil, err
	}
	if len(missingReuseContents(contentsByStorageKey, reuse)) == 0 {
		return reuse, nil
	}
	if err := service.loadCollectionReuse(
		ctx,
		collectionName,
		contentsByStorageKey,
		reuse,
	); err != nil {
		return nil, err
	}
	return reuse, nil
}

func (service *Service) reuseCandidates(
	chunks []model.StoredChunk,
) (map[string]string, []string) {
	contentsByStorageKey := make(map[string]string, len(chunks))
	storageKeys := make([]string, 0, len(chunks))
	for _, chunk := range chunks {
		content, _ := sanitizeUTF8(chunk.Content)
		storageKey := contentHash(content)
		if _, found := contentsByStorageKey[storageKey]; found {
			continue
		}
		contentsByStorageKey[storageKey] = content
		storageKeys = append(storageKeys, storageKey)
	}
	return contentsByStorageKey, storageKeys
}

func (service *Service) loadCatalogReuse(
	ctx context.Context,
	storageKeys []string,
	contentsByStorageKey map[string]string,
	reuse map[string][]float32,
) error {
	dimension := int(service.cfg.EmbeddingDimension)
	if dimension <= 0 {
		return nil
	}
	catalogAvailable, err := service.reuseCatalogAvailable(ctx, dimension, false)
	if err != nil {
		return err
	}
	if !catalogAvailable {
		return nil
	}
	catalogName := reuseCatalogCollectionName(service.cfg, dimension)
	lease, err := service.AcquireCollection(ctx, catalogName)
	if err != nil {
		return err
	}
	defer lease.Release()
	for _, batch := range reuseLookupBatches(storageKeys) {
		catalogVectors, loadErr := service.loadReuseCatalogKeys(ctx, batch, dimension)
		if loadErr != nil {
			return loadErr
		}
		for storageKey, entries := range catalogVectors {
			content, found := contentsByStorageKey[storageKey]
			if !found {
				continue
			}
			for _, entry := range entries {
				if embeddingModelsCompatible(entry.embeddingModel, service.cfg.EmbeddingModel) {
					reuse[contentVectorKey(content)] = entry.vector
					break
				}
			}
		}
	}
	return nil
}

func missingReuseContents(
	contentsByStorageKey map[string]string,
	reuse map[string][]float32,
) []string {
	missingContents := make([]string, 0, len(contentsByStorageKey))
	for _, content := range contentsByStorageKey {
		if _, found := reuse[contentVectorKey(content)]; !found {
			missingContents = append(missingContents, content)
		}
	}
	return missingContents
}

func embeddingModelsCompatible(stored string, current string) bool {
	return stored == "" || current == "" || stored == current
}

func (service *Service) loadCollectionReuse(
	ctx context.Context,
	collectionName string,
	contentsByStorageKey map[string]string,
	reuse map[string][]float32,
) error {
	if len(contentsByStorageKey) == 0 || collectionName == "" {
		return nil
	}
	collectionNames, err := service.reuseSourceCollections(collectionName)
	if err != nil {
		return err
	}
	for sourceIndex, sourceCollectionName := range collectionNames {
		if loadErr := service.loadRegisteredReuseSource(
			ctx,
			sourceCollectionName,
			sourceIndex == 0,
			contentsByStorageKey,
			reuse,
		); loadErr != nil {
			return loadErr
		}
		if len(missingReuseContents(contentsByStorageKey, reuse)) == 0 {
			break
		}
	}
	return nil
}

func (service *Service) loadRegisteredReuseSource(
	ctx context.Context,
	collectionName string,
	primary bool,
	contentsByStorageKey map[string]string,
	reuse map[string][]float32,
) error {
	var residentProbe CollectionLease
	if !primary {
		lease, ready, err := service.acquireReuseSourceCollection(
			ctx,
			collectionName,
			false,
		)
		if err != nil || !ready {
			return err
		}
		residentProbe = lease
	}
	hasCollection, err := service.hasCollection(
		ctx,
		collectionName,
		"check collection before whole-store reuse",
	)
	if err != nil {
		if residentProbe != nil {
			residentProbe.Release()
		}
		return err
	}
	if !hasCollection {
		if residentProbe != nil {
			residentProbe.Release()
		}
		return nil
	}
	if residentProbe != nil {
		residentProbe.Release()
	}
	if err := service.ensureReuseIdentityColumnsOnce(
		ctx,
		collectionName,
	); err != nil {
		return err
	}
	lease, ready, err := service.acquireReuseSourceCollection(
		ctx,
		collectionName,
		primary,
	)
	if err != nil || !ready {
		return err
	}
	loadErr := service.loadCollectionReuseFromSource(
		ctx,
		collectionName,
		contentsByStorageKey,
		reuse,
	)
	lease.Release()
	return loadErr
}

func (service *Service) acquireReuseSourceCollection(
	ctx context.Context,
	collectionName string,
	primary bool,
) (CollectionLease, bool, error) {
	if primary {
		lease, err := service.AcquireCollection(ctx, collectionName)
		return lease, err == nil, err
	}
	return service.acquireResidentCollection(ctx, collectionName)
}

func (service *Service) reuseSourceCollections(collectionName string) ([]string, error) {
	collectionNames := []string{collectionName}
	seen := map[string]struct{}{collectionName: {}}
	if service.cfg.RegistryPath == "" {
		return collectionNames, nil
	}
	registry, err := store.ReadRegistry(service.cfg.RegistryPath)
	if err != nil {
		wrappedErr := fmt.Errorf("read registry for whole-store content reuse: %w", err)
		slog.Error("read registry for whole-store content reuse failed", "err", wrappedErr)
		return nil, wrappedErr
	}
	for _, codebase := range registry.Codebases {
		names := append([]string{codebase.CollectionName}, codebase.LegacyCollectionNames...)
		for _, name := range names {
			if name == "" {
				continue
			}
			if _, found := seen[name]; found {
				continue
			}
			seen[name] = struct{}{}
			collectionNames = append(collectionNames, name)
		}
	}
	return collectionNames, nil
}

func (service *Service) loadCollectionReuseFromSource(
	ctx context.Context,
	collectionName string,
	contentsByStorageKey map[string]string,
	reuse map[string][]float32,
) error {
	missingStorageKeys := make([]string, 0, len(contentsByStorageKey))
	for storageKey, content := range contentsByStorageKey {
		if _, found := reuse[contentVectorKey(content)]; !found {
			missingStorageKeys = append(missingStorageKeys, storageKey)
		}
	}
	for _, batch := range reuseLookupBatches(missingStorageKeys) {
		if err := service.loadStoredHashReuseBatch(
			ctx,
			collectionName,
			batch,
			contentsByStorageKey,
			reuse,
		); err != nil {
			return err
		}
	}

	legacyContents := missingReuseContents(contentsByStorageKey, reuse)
	for _, content := range legacyContents {
		if err := service.loadLegacyReuseContent(
			ctx,
			collectionName,
			content,
			reuse,
		); err != nil {
			return err
		}
	}
	return nil
}

func (service *Service) loadStoredHashReuseBatch(
	ctx context.Context,
	collectionName string,
	contentHashes []string,
	contentsByHash map[string]string,
	reuse map[string][]float32,
) error {
	candidates, err := service.discoverStoredHashReuseCandidates(
		ctx,
		collectionName,
		inStringClause(contentHashFieldName, contentHashes),
		contentsByHash,
	)
	if err != nil {
		return err
	}
	return service.loadReuseVectorCandidates(ctx, collectionName, candidates, reuse)
}

func reuseLookupBatches(values []string) [][]string {
	batches := make([][]string, 0)
	for len(values) > 0 {
		count := 0
		escapedBytes := 0
		for count < len(values) && count < reuseLookupBatchSize {
			valueBytes := len(escapeMilvusString(values[count])) + 4
			if count > 0 && escapedBytes+valueBytes > reuseLookupMaxEscapedBytes {
				break
			}
			escapedBytes += valueBytes
			count++
		}
		batches = append(batches, values[:count])
		values = values[count:]
	}
	return batches
}

type reuseVectorCandidate struct {
	id              string
	expectedContent string
}

type reuseVectorCandidateByID map[string]reuseVectorCandidate

func estimatedReuseVectorRowBytes(dimension int) (int64, error) {
	if dimension <= 0 {
		return 0, fmt.Errorf("reuse vector dimension must be positive: %d", dimension)
	}
	rowBytes := int64(0)
	for _, fieldBytes := range []int64{
		idFieldMaxLength,
		contentFieldMaxLength,
		embeddingModelFieldMaxLength,
		reuseVectorRowFramingBytes,
	} {
		if rowBytes > math.MaxInt64-fieldBytes {
			return 0, errors.New("reuse vector row byte estimate overflow")
		}
		rowBytes += fieldBytes
	}
	dimensionBytes := int64(dimension)
	if dimensionBytes > (math.MaxInt64-rowBytes)/reuseVectorFloatBytes {
		return 0, errors.New("reuse vector row byte estimate overflow")
	}
	rowBytes += dimensionBytes * reuseVectorFloatBytes
	return rowBytes, nil
}

func packReuseVectorCandidates(
	candidates reuseVectorCandidateByID,
	dimension int,
) ([][]string, error) {
	rowBytes, err := estimatedReuseVectorRowBytes(dimension)
	if err != nil {
		return nil, err
	}
	ids := make([]string, 0, len(candidates))
	for id := range candidates {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	return packReuseVectorCandidateIDs(ids, rowBytes, reuseVectorFetchBudgetBytes)
}

func packReuseVectorCandidateIDs(
	ids []string,
	rowBytes int64,
	budgetBytes int64,
) ([][]string, error) {
	if rowBytes <= 0 {
		return nil, fmt.Errorf("reuse vector row byte estimate must be positive: %d", rowBytes)
	}
	if budgetBytes <= 0 {
		return nil, fmt.Errorf("reuse vector fetch budget must be positive: %d", budgetBytes)
	}
	if rowBytes > budgetBytes {
		return nil, fmt.Errorf(
			"estimated reuse vector row size %d exceeds fetch budget %d",
			rowBytes,
			budgetBytes,
		)
	}
	batches := make([][]string, 0)
	batch := make([]string, 0)
	batchBytes := int64(0)
	for _, id := range ids {
		if batchBytes > budgetBytes-rowBytes {
			batches = append(batches, batch)
			batch = make([]string, 0)
			batchBytes = 0
		}
		batch = append(batch, id)
		batchBytes += rowBytes
	}
	if len(batch) > 0 {
		batches = append(batches, batch)
	}
	return batches, nil
}

func (service *Service) discoverStoredHashReuseCandidates(
	ctx context.Context,
	collectionName string,
	filter string,
	contentsByHash map[string]string,
) (reuseVectorCandidateByID, error) {
	iterator, err := service.milvus.QueryIterator(
		ctx,
		milvusclient.NewQueryIteratorOption(collectionName).
			WithBatchSize(reuseVectorBatchSize).
			WithFilter(filter).
			WithOutputFields(idFieldName, contentHashFieldName, embeddingModelFieldName),
	)
	if err != nil {
		peerInfo, _ := peer.FromContext(ctx)
		slog.ErrorContext(
			ctx,
			"open stored reuse candidate iterator failed",
			"collection", collectionName,
			"peer", peerInfo.String(),
			"err", err,
		)
		return nil, fmt.Errorf("open stored reuse candidate iterator for %s: %w", collectionName, err)
	}
	candidates := make(reuseVectorCandidateByID)
	selectedHashes := make(map[string]struct{}, len(contentsByHash))
	for {
		resultSet, nextErr := iterator.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			return candidates, nil
		}
		if nextErr != nil {
			return nil, fmt.Errorf("iterate stored reuse candidates for %s: %w", collectionName, nextErr)
		}
		idColumn := resultSet.GetColumn(idFieldName)
		contentHashColumn := resultSet.GetColumn(contentHashFieldName)
		embeddingModelColumn := resultSet.GetColumn(embeddingModelFieldName)
		if idColumn == nil || contentHashColumn == nil {
			return nil, ErrSearchResultIncomplete
		}
		for rowIndex := range resultSet.ResultCount {
			id, idErr := idColumn.GetAsString(rowIndex)
			if idErr != nil {
				return nil, fmt.Errorf("read reuse candidate ID at %d: %w", rowIndex, idErr)
			}
			contentHash, contentHashErr := contentHashColumn.GetAsString(rowIndex)
			if contentHashErr != nil {
				return nil, fmt.Errorf("read reuse content hash at %d: %w", rowIndex, contentHashErr)
			}
			expectedContent, wanted := contentsByHash[contentHash]
			if !wanted {
				continue
			}
			if _, selected := selectedHashes[contentHash]; selected {
				continue
			}
			embeddingModel, modelErr := nullableStringAt(embeddingModelColumn, rowIndex)
			if modelErr != nil {
				return nil, fmt.Errorf("read reuse embedding model at %d: %w", rowIndex, modelErr)
			}
			if !embeddingModelsCompatible(embeddingModel, service.cfg.EmbeddingModel) {
				continue
			}
			candidates[id] = reuseVectorCandidate{
				id:              id,
				expectedContent: expectedContent,
			}
			selectedHashes[contentHash] = struct{}{}
		}
	}
}

func (service *Service) loadLegacyReuseContent(
	ctx context.Context,
	collectionName string,
	content string,
	reuse map[string][]float32,
) error {
	iterator, err := service.milvus.QueryIterator(
		ctx,
		milvusclient.NewQueryIteratorOption(collectionName).
			WithBatchSize(reuseVectorBatchSize).
			WithFilter(contentHashFieldName+" is null and "+
				contentFieldName+" == \""+escapeMilvusString(content)+"\"").
			WithOutputFields(idFieldName, embeddingModelFieldName),
	)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"open legacy reuse candidate iterator failed",
			"collection", collectionName,
			"err", err,
		)
		return fmt.Errorf("open legacy reuse candidate iterator for %s: %w", collectionName, err)
	}
	for {
		resultSet, nextErr := iterator.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			return nil
		}
		if nextErr != nil {
			return fmt.Errorf("iterate legacy reuse candidates for %s: %w", collectionName, nextErr)
		}
		idColumn := resultSet.GetColumn(idFieldName)
		embeddingModelColumn := resultSet.GetColumn(embeddingModelFieldName)
		if idColumn == nil {
			return ErrSearchResultIncomplete
		}
		for rowIndex := range resultSet.ResultCount {
			id, idErr := idColumn.GetAsString(rowIndex)
			if idErr != nil {
				return fmt.Errorf("read legacy reuse candidate ID at %d: %w", rowIndex, idErr)
			}
			embeddingModel, modelErr := nullableStringAt(embeddingModelColumn, rowIndex)
			if modelErr != nil {
				return fmt.Errorf("read legacy reuse embedding model at %d: %w", rowIndex, modelErr)
			}
			if !embeddingModelsCompatible(embeddingModel, service.cfg.EmbeddingModel) {
				continue
			}
			candidate := reuseVectorCandidate{id: id, expectedContent: content}
			return service.loadReuseVectorCandidates(
				ctx,
				collectionName,
				reuseVectorCandidateByID{id: candidate},
				reuse,
			)
		}
	}
}

func (service *Service) loadReuseVectorCandidates(
	ctx context.Context,
	collectionName string,
	candidates reuseVectorCandidateByID,
	reuse map[string][]float32,
) error {
	if len(candidates) == 0 {
		return nil
	}
	dimension, err := service.resolveReuseVectorDimension(ctx, collectionName)
	if err != nil {
		return err
	}
	batches, err := packReuseVectorCandidates(candidates, dimension)
	if err != nil {
		return err
	}
	for _, ids := range batches {
		resultSet, getErr := service.milvus.Get(
			ctx,
			milvusclient.NewQueryOption(collectionName).
				WithIDs(column.NewColumnVarChar(idFieldName, slices.Clone(ids))).
				WithOutputFields(
					idFieldName,
					contentFieldName,
					embeddingModelFieldName,
					denseVectorFieldName,
				),
		)
		if getErr != nil {
			slog.ErrorContext(
				ctx,
				"get selected reuse vectors failed",
				"collection", collectionName,
				"err", getErr,
			)
			return fmt.Errorf("get selected reuse vectors for %s: %w", collectionName, getErr)
		}
		batchReuse, readErr := service.readReuseVectorBatch(
			resultSet,
			ids,
			candidates,
			dimension,
		)
		if readErr != nil {
			return readErr
		}
		maps.Copy(reuse, batchReuse)
	}
	return nil
}

func (service *Service) resolveReuseVectorDimension(
	ctx context.Context,
	collectionName string,
) (int, error) {
	dimension, generation, found, err := service.loadReuseVectorDimensionCache(collectionName)
	if err != nil {
		return 0, err
	}
	if found {
		return dimension, nil
	}

	collection, err := service.milvus.DescribeCollection(
		ctx,
		milvusclient.NewDescribeCollectionOption(collectionName),
	)
	if err != nil {
		peerInfo, _ := peer.FromContext(ctx)
		slog.ErrorContext(
			ctx,
			"describe collection for reuse vector dimension failed",
			"collection", collectionName,
			"peer", peerInfo.String(),
			"err", err,
		)
		return 0, fmt.Errorf(
			"describe collection %s for reuse vector dimension: %w",
			collectionName,
			err,
		)
	}
	if collection == nil || collection.Schema == nil {
		return 0, fmt.Errorf("reuse source collection %s is missing schema", collectionName)
	}
	schemaDimension, err := reuseVectorDimensionFromSchema(
		collectionName,
		collection.Schema,
	)
	if err != nil {
		return 0, err
	}
	configuredDimension := int64(service.cfg.EmbeddingDimension)
	if configuredDimension < 0 {
		return 0, fmt.Errorf(
			"configured reuse vector dimension must not be negative: %d",
			configuredDimension,
		)
	}
	if configuredDimension > 0 && configuredDimension != schemaDimension {
		return 0, fmt.Errorf(
			"reuse source dimension %d in %s does not match configured dimension %d",
			schemaDimension,
			collectionName,
			configuredDimension,
		)
	}
	dimension = int(schemaDimension)
	if int64(dimension) != schemaDimension {
		return 0, fmt.Errorf(
			"reuse vector dimension in %s exceeds local integer range: %d",
			collectionName,
			schemaDimension,
		)
	}
	service.storeReuseVectorDimensionIfCurrent(collectionName, generation, dimension)
	return dimension, nil
}

func (service *Service) loadReuseVectorDimensionCache(
	collectionName string,
) (int, uint64, bool, error) {
	service.reuseVectorDimensionMutex.Lock()
	defer service.reuseVectorDimensionMutex.Unlock()
	if service.reuseVectorDimensionGeneration == nil {
		service.reuseVectorDimensionGeneration = make(map[string]uint64)
	}
	generation := service.reuseVectorDimensionGeneration[collectionName]
	cached, found := service.reuseVectorDimensions.Load(collectionName)
	if !found {
		return 0, generation, false, nil
	}
	dimension, ok := cached.(int)
	if !ok || dimension <= 0 {
		cacheErr := fmt.Errorf(
			"cached reuse vector dimension for %s has unexpected value %v",
			collectionName,
			cached,
		)
		slog.Error(
			"reuse vector dimension cache is invalid",
			"collection", collectionName,
			"err", cacheErr,
		)
		return 0, generation, false, cacheErr
	}
	return dimension, generation, true, nil
}

func (service *Service) storeReuseVectorDimensionIfCurrent(
	collectionName string,
	generation uint64,
	dimension int,
) {
	service.reuseVectorDimensionMutex.Lock()
	defer service.reuseVectorDimensionMutex.Unlock()
	if service.reuseVectorDimensionGeneration[collectionName] != generation {
		return
	}
	service.reuseVectorDimensions.Store(collectionName, dimension)
}

func reuseVectorDimensionFromSchema(
	collectionName string,
	schema *entity.Schema,
) (int64, error) {
	var vectorField *entity.Field
	for _, field := range schema.Fields {
		if field.Name == denseVectorFieldName {
			vectorField = field
			break
		}
	}
	if vectorField == nil {
		return 0, fmt.Errorf(
			"reuse source collection %s is missing vector field %s",
			collectionName,
			denseVectorFieldName,
		)
	}
	if vectorField.DataType != entity.FieldTypeFloatVector {
		return 0, fmt.Errorf(
			"reuse source field %s in %s must be a float vector, got %s",
			denseVectorFieldName,
			collectionName,
			vectorField.DataType.Name(),
		)
	}
	dimension, err := vectorField.GetDim()
	if err != nil {
		wrappedErr := fmt.Errorf(
			"read reuse vector dimension from %s: %w",
			collectionName,
			err,
		)
		slog.Error(
			"read reuse vector dimension from schema failed",
			"collection", collectionName,
			"err", wrappedErr,
		)
		return 0, wrappedErr
	}
	if dimension <= 0 {
		return 0, fmt.Errorf(
			"reuse vector dimension in %s must be positive: %d",
			collectionName,
			dimension,
		)
	}
	return dimension, nil
}

func (service *Service) readReuseVectorBatch(
	resultSet milvusclient.ResultSet,
	requestedIDs []string,
	candidates reuseVectorCandidateByID,
	dimension int,
) (map[string][]float32, error) {
	idColumn := resultSet.GetColumn(idFieldName)
	contentColumn := resultSet.GetColumn(contentFieldName)
	embeddingModelColumn := resultSet.GetColumn(embeddingModelFieldName)
	vectorColumn := resultSet.GetColumn(denseVectorFieldName)
	if idColumn == nil || contentColumn == nil || embeddingModelColumn == nil || vectorColumn == nil {
		return nil, ErrSearchResultIncomplete
	}
	requested := make(map[string]reuseVectorCandidate, len(requestedIDs))
	for _, id := range requestedIDs {
		candidate, found := candidates[id]
		if !found {
			return nil, fmt.Errorf("selected reuse vector ID %q has no candidate", id)
		}
		requested[id] = candidate
	}
	seen := make(map[string]struct{}, resultSet.ResultCount)
	batchReuse := make(map[string][]float32, resultSet.ResultCount)
	for rowIndex := range resultSet.ResultCount {
		id, idErr := idColumn.GetAsString(rowIndex)
		if idErr != nil {
			wrappedErr := fmt.Errorf("read selected reuse ID at %d: %w", rowIndex, idErr)
			slog.Error("read selected reuse ID failed", "row_index", rowIndex, "err", wrappedErr)
			return nil, wrappedErr
		}
		candidate, found := requested[id]
		if !found {
			return nil, fmt.Errorf("selected reuse response returned unknown ID %q", id)
		}
		if _, duplicate := seen[id]; duplicate {
			return nil, fmt.Errorf("selected reuse response returned duplicate ID %q", id)
		}
		seen[id] = struct{}{}
		content, contentErr := contentColumn.GetAsString(rowIndex)
		if contentErr != nil {
			return nil, fmt.Errorf("read selected reuse content at %d: %w", rowIndex, contentErr)
		}
		if content != candidate.expectedContent {
			return nil, fmt.Errorf("selected reuse content mismatch for ID %q", id)
		}
		embeddingModel, modelErr := nullableStringAt(embeddingModelColumn, rowIndex)
		if modelErr != nil {
			return nil, fmt.Errorf("read selected reuse embedding model at %d: %w", rowIndex, modelErr)
		}
		if !embeddingModelsCompatible(embeddingModel, service.cfg.EmbeddingModel) {
			return nil, fmt.Errorf("selected reuse embedding model mismatch for ID %q", id)
		}
		vector, vectorErr := vectorAt(vectorColumn, rowIndex)
		if vectorErr != nil {
			return nil, vectorErr
		}
		if len(vector) != dimension {
			return nil, fmt.Errorf(
				"selected reuse vector dimension for ID %q is %d, want %d",
				id,
				len(vector),
				dimension,
			)
		}
		batchReuse[contentVectorKey(content)] = vector
	}
	for _, id := range requestedIDs {
		if _, found := seen[id]; !found {
			return nil, fmt.Errorf("selected reuse response omitted requested ID %q", id)
		}
	}
	return batchReuse, nil
}

func nullableStringAt(field column.Column, rowIndex int) (string, error) {
	if field == nil {
		return "", nil
	}
	isNull, err := field.IsNull(rowIndex)
	if err != nil {
		wrappedErr := fmt.Errorf("read nullable marker at %d: %w", rowIndex, err)
		slog.Error("read nullable string marker failed", "row_index", rowIndex, "err", wrappedErr)
		return "", wrappedErr
	}
	if isNull {
		return "", nil
	}
	value, err := field.GetAsString(rowIndex)
	if err != nil {
		wrappedErr := fmt.Errorf("read nullable string at %d: %w", rowIndex, err)
		slog.Error("read nullable string failed", "row_index", rowIndex, "err", wrappedErr)
		return "", wrappedErr
	}
	return value, nil
}

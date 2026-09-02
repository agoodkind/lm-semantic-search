package semantic

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/store"
)

const (
	reuseLookupBatchSize       = 256
	reuseLookupMaxEscapedBytes = 256 * 1024
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
		slog.ErrorContext(
			ctx,
			"open stored reuse candidate iterator failed",
			"collection", collectionName,
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
			return service.loadReuseVectorCandidate(ctx, collectionName, reuseVectorCandidate{
				id:              id,
				expectedContent: content,
			}, reuse)
		}
	}
}

func (service *Service) loadReuseVectorCandidates(
	ctx context.Context,
	collectionName string,
	candidates reuseVectorCandidateByID,
	reuse map[string][]float32,
) error {
	for _, candidate := range candidates {
		if err := service.loadReuseVectorCandidate(ctx, collectionName, candidate, reuse); err != nil {
			return err
		}
	}
	return nil
}

func (service *Service) loadReuseVectorCandidate(
	ctx context.Context,
	collectionName string,
	candidate reuseVectorCandidate,
	reuse map[string][]float32,
) error {
	iterator, err := service.milvus.QueryIterator(
		ctx,
		milvusclient.NewQueryIteratorOption(collectionName).
			WithBatchSize(reuseVectorBatchSize).
			WithFilter(inStringClause(idFieldName, []string{candidate.id})).
			WithOutputFields(contentFieldName, denseVectorFieldName),
	)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"open reuse vector iterator failed",
			"collection", collectionName,
			"id", candidate.id,
			"err", err,
		)
		return fmt.Errorf("open reuse vector iterator for %s: %w", collectionName, err)
	}
	for {
		resultSet, nextErr := iterator.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			return nil
		}
		if nextErr != nil {
			return fmt.Errorf("iterate reuse vector for %s: %w", collectionName, nextErr)
		}
		contentColumn := resultSet.GetColumn(contentFieldName)
		vectorColumn := resultSet.GetColumn(denseVectorFieldName)
		if contentColumn == nil || vectorColumn == nil {
			return ErrSearchResultIncomplete
		}
		for rowIndex := range resultSet.ResultCount {
			content, contentErr := contentColumn.GetAsString(rowIndex)
			if contentErr != nil {
				return fmt.Errorf("read reuse content at %d: %w", rowIndex, contentErr)
			}
			if content != candidate.expectedContent {
				continue
			}
			vector, vectorErr := vectorAt(vectorColumn, rowIndex)
			if vectorErr != nil {
				return vectorErr
			}
			reuse[contentVectorKey(content)] = vector
			return nil
		}
	}
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

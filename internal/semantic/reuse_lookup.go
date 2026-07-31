package semantic

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
)

const (
	reuseLookupBatchSize       = 256
	reuseLookupMaxEscapedBytes = 256 * 1024
)

func embeddingIdentity(cfg config.Config) string {
	value := fmt.Sprintf(
		"%s\x00%s\x00%s\x00%d\x00%s",
		cfg.EmbeddingProvider,
		cfg.EmbeddingModel,
		cfg.OfflineEmbeddingModel,
		cfg.EmbeddingDimension,
		cfg.OpenAIBaseURL,
	)
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func contentVectorStorageKey(cfg config.Config, content string) string {
	sum := sha256.Sum256([]byte(embeddingIdentity(cfg) + "\x00" + content))
	return hex.EncodeToString(sum[:])
}

func contentVectorStorageKeys(cfg config.Config, contents []string) []string {
	keys := make([]string, 0, len(contents))
	for _, content := range contents {
		keys = append(keys, contentVectorStorageKey(cfg, content))
	}
	return keys
}

// LoadReuseVectorsForContents resolves only vectors needed by chunks through the
// identity-scoped catalog. Catalog misses fall back only to legacy corpus rows
// whose indexed content key is null, and those rows remain unchanged.
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
	legacyContents := missingReuseContents(contentsByStorageKey, reuse)
	if err := service.loadLegacyReuse(ctx, collectionName, legacyContents, reuse); err != nil {
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
		storageKey := contentVectorStorageKey(service.cfg, content)
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
	catalogAvailable, err := service.reuseCatalogAvailable(ctx, 0)
	if err != nil {
		return err
	}
	if !catalogAvailable {
		return nil
	}
	for _, batch := range reuseLookupBatches(storageKeys) {
		catalogVectors, loadErr := service.loadReuseCatalogKeys(ctx, batch)
		if loadErr != nil {
			return loadErr
		}
		for storageKey, vector := range catalogVectors {
			content, found := contentsByStorageKey[storageKey]
			if found {
				reuse[contentVectorKey(content)] = vector
			}
		}
	}
	return nil
}

func missingReuseContents(
	contentsByStorageKey map[string]string,
	reuse map[string][]float32,
) []string {
	legacyContents := make([]string, 0, len(contentsByStorageKey))
	for _, content := range contentsByStorageKey {
		if _, found := reuse[contentVectorKey(content)]; !found {
			legacyContents = append(legacyContents, content)
		}
	}
	return legacyContents
}

func (service *Service) loadLegacyReuse(
	ctx context.Context,
	collectionName string,
	legacyContents []string,
	reuse map[string][]float32,
) error {
	if len(legacyContents) == 0 || collectionName == "" {
		return nil
	}
	hasCollection, err := service.hasCollection(
		ctx,
		collectionName,
		"check collection before legacy reuse",
	)
	if err != nil {
		return err
	}
	if !hasCollection {
		return nil
	}
	if err := service.ensureContentVectorKeyColumnOnce(ctx, collectionName); err != nil {
		return err
	}
	if err := service.loadCollectionForRead(ctx, collectionName); err != nil {
		return err
	}

	legacyCatalog := make(reuseCatalogVectors)
	for _, batch := range reuseLookupBatches(legacyContents) {
		if err := service.loadLegacyReuseBatch(
			ctx,
			collectionName,
			batch,
			reuse,
			legacyCatalog,
		); err != nil {
			return err
		}
	}
	if err := service.appendReuseCatalog(ctx, legacyCatalog); err != nil {
		return err
	}
	return nil
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

func (service *Service) loadLegacyReuseBatch(
	ctx context.Context,
	collectionName string,
	contents []string,
	reuse map[string][]float32,
	legacyCatalog reuseCatalogVectors,
) error {
	iterator, err := service.milvus.QueryIterator(
		ctx,
		milvusclient.NewQueryIteratorOption(collectionName).
			WithBatchSize(reuseVectorBatchSize).
			WithFilter(
				contentVectorKeyFieldName+" is null and ("+
					inStringClause(contentFieldName, contents)+")",
			).
			WithOutputFields(contentFieldName, denseVectorFieldName),
	)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"open legacy content reuse iterator failed",
			"collection", collectionName,
			"err", err,
		)
		return fmt.Errorf("open legacy content reuse iterator for %s: %w", collectionName, err)
	}
	for {
		resultSet, nextErr := iterator.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			return nil
		}
		if nextErr != nil {
			return fmt.Errorf("iterate legacy content reuse for %s: %w", collectionName, nextErr)
		}
		contentColumn := resultSet.GetColumn(contentFieldName)
		vectorColumn := resultSet.GetColumn(denseVectorFieldName)
		if contentColumn == nil || vectorColumn == nil {
			return ErrSearchResultIncomplete
		}
		for rowIndex := range resultSet.ResultCount {
			content, contentErr := contentColumn.GetAsString(rowIndex)
			if contentErr != nil {
				return fmt.Errorf("read legacy reuse content at %d: %w", rowIndex, contentErr)
			}
			vector, vectorErr := vectorAt(vectorColumn, rowIndex)
			if vectorErr != nil {
				return vectorErr
			}
			reuse[contentVectorKey(content)] = vector
			if service.cfg.EmbeddingDimension > 0 &&
				len(vector) == int(service.cfg.EmbeddingDimension) {
				legacyCatalog[content] = vector
			}
		}
	}
}

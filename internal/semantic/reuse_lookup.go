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

// LoadReuseVectorsForContents resolves only vectors needed by chunks. New rows
// use the indexed identity-aware key. Rows written before that field existed use
// exact-content fallback and remain unchanged.
func (service *Service) LoadReuseVectorsForContents(
	ctx context.Context,
	collectionName string,
	chunks []model.StoredChunk,
) (map[string][]float32, error) {
	reuse := make(map[string][]float32)
	if !service.Available() || collectionName == "" || len(chunks) == 0 {
		return reuse, nil
	}
	hasCollection, err := service.hasCollection(
		ctx,
		collectionName,
		"check collection before corpus reuse",
	)
	if err != nil {
		return nil, err
	}
	if !hasCollection {
		return reuse, nil
	}
	if err := service.ensureContentVectorKeyColumnOnce(ctx, collectionName); err != nil {
		return nil, err
	}
	if err := service.loadCollectionForRead(ctx, collectionName); err != nil {
		return nil, err
	}

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
	for _, batch := range reuseLookupBatches(storageKeys) {
		if err := service.loadReuseKeyBatch(
			ctx,
			collectionName,
			batch,
			contentsByStorageKey,
			reuse,
		); err != nil {
			return nil, err
		}
	}

	legacyContents := make([]string, 0, len(contentsByStorageKey))
	for _, content := range contentsByStorageKey {
		if _, found := reuse[contentVectorKey(content)]; !found {
			legacyContents = append(legacyContents, content)
		}
	}
	for _, batch := range reuseLookupBatches(legacyContents) {
		if err := service.loadLegacyReuseBatch(
			ctx,
			collectionName,
			batch,
			reuse,
		); err != nil {
			return nil, err
		}
	}
	return reuse, nil
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

func (service *Service) loadReuseKeyBatch(
	ctx context.Context,
	collectionName string,
	storageKeys []string,
	contentsByStorageKey map[string]string,
	reuse map[string][]float32,
) error {
	iterator, err := service.milvus.QueryIterator(
		ctx,
		milvusclient.NewQueryIteratorOption(collectionName).
			WithBatchSize(reuseVectorBatchSize).
			WithFilter(inStringClause(contentVectorKeyFieldName, storageKeys)).
			WithOutputFields(contentVectorKeyFieldName, denseVectorFieldName),
	)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"open content-key reuse iterator failed",
			"collection", collectionName,
			"err", err,
		)
		return fmt.Errorf("open content-key reuse iterator for %s: %w", collectionName, err)
	}
	for {
		resultSet, nextErr := iterator.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			return nil
		}
		if nextErr != nil {
			return fmt.Errorf("iterate content-key reuse for %s: %w", collectionName, nextErr)
		}
		keyColumn := resultSet.GetColumn(contentVectorKeyFieldName)
		vectorColumn := resultSet.GetColumn(denseVectorFieldName)
		if keyColumn == nil || vectorColumn == nil {
			return ErrSearchResultIncomplete
		}
		for rowIndex := range resultSet.ResultCount {
			storageKey, keyErr := keyColumn.GetAsString(rowIndex)
			if keyErr != nil {
				return fmt.Errorf("read content vector key at %d: %w", rowIndex, keyErr)
			}
			content, found := contentsByStorageKey[storageKey]
			if !found {
				continue
			}
			vector, vectorErr := vectorAt(vectorColumn, rowIndex)
			if vectorErr != nil {
				return vectorErr
			}
			reuse[contentVectorKey(content)] = vector
		}
	}
}

func (service *Service) loadLegacyReuseBatch(
	ctx context.Context,
	collectionName string,
	contents []string,
	reuse map[string][]float32,
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
		}
	}
}

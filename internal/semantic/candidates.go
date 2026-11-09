package semantic

import (
	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"

	"goodkind.io/lm-semantic-search/collection"
	"goodkind.io/lm-semantic-search/internal/model"
)

const escapedReuseValueFramingBytes = 4

func contentHash(content string) string {
	normalized, _ := milvusstore.SanitizeUTF8(content)

	return contentVectorKey(normalized)
}

func (service *Service) reuseCandidates(
	chunks []model.StoredChunk,
) (map[string]string, []string) {
	contentsByStorageKey := make(map[string]string, len(chunks))

	storageKeys := make([]string, 0, len(chunks))
	for _, chunk := range chunks {
		content, _ := milvusstore.SanitizeUTF8(chunk.Content)
		storageKey := contentHash(content)
		if _, found := contentsByStorageKey[storageKey]; found {
			continue
		}

		contentsByStorageKey[storageKey] = content
		storageKeys = append(storageKeys, storageKey)
	}

	return contentsByStorageKey, storageKeys
}

func missingReuseContents(
	contentsByStorageKey map[string]string,
	reuse map[string]model.ReuseVector,
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

func reuseLookupBatches(values []string) [][]string {
	batches := make([][]string, 0)

	for len(values) > 0 {
		count := 0
		escapedBytes := 0

		for count < len(values) && count < reuseLookupBatchSize {
			valueBytes := len(
				collection.EscapeString(values[count]),
			) + escapedReuseValueFramingBytes
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

package semantic

import (
	"goodkind.io/lm-semantic-search/collection"
	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"
	"goodkind.io/lm-semantic-search/internal/model"
)

// collectionStore returns the Milvus collection store over the service's
// current client and configuration. The store is a value over the client. The
// reconnector replaces the client, and each call reads the current one.
func (service *Service) collectionStore() *milvusstore.Store {
	return milvusstore.New(service.milvus, milvusstore.Options{
		Hybrid:         service.cfg.HybridMode,
		EmbeddingModel: service.cfg.EmbeddingModel,
	})
}

// chunkFromHit converts one store hit to a stored chunk. The chunk decodes the
// language from the metadata JSON.
func chunkFromHit(hit collection.Hit) model.StoredChunk {
	metadata := decodeMetadata(hit.Metadata)
	return model.StoredChunk{
		Content:           hit.Content,
		RelativePath:      hit.RelativePath,
		StartLine:         hit.StartLine,
		EndLine:           hit.EndLine,
		Language:          metadata.Language,
		FileExtension:     hit.FileExtension,
		SplitPart:         hit.SplitPart,
		SplitPartRecorded: hit.SplitPartRecorded,
		Scalars:           nil,
		Score:             hit.Score,
	}
}

// chunksFromHits converts store hits to stored chunks in order.
func chunksFromHits(hits []collection.Hit) []model.StoredChunk {
	chunks := make([]model.StoredChunk, 0, len(hits))
	for _, hit := range hits {
		chunks = append(chunks, chunkFromHit(hit))
	}
	return chunks
}

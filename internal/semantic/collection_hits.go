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
// language and the conversation attributes from the metadata JSON. The
// workspaceRoot and loadRules columns come from the hit's scalar cells and read
// empty when the hit lacks them.
func chunkFromHit(hit collection.Hit) model.StoredChunk {
	metadata := decodeMetadata(hit.Metadata)
	return model.StoredChunk{
		Content:              hit.Content,
		RelativePath:         hit.RelativePath,
		StartLine:            hit.StartLine,
		EndLine:              hit.EndLine,
		Language:             metadata.Language,
		FileExtension:        hit.FileExtension,
		ConversationID:       metadata.ConversationID,
		ParentConversationID: metadata.ParentConversationID,
		MessageIndex:         metadata.messageIndex(),
		Role:                 metadata.Role,
		TimestampUnix:        metadata.timestampUnix(),
		WorkspaceRoot:        hitScalarString(hit, workspaceRootFieldName),
		Archived:             false,
		SplitPart:            hit.SplitPart,
		SplitPartRecorded:    hit.SplitPartRecorded,
		LoadRules:            hitScalarString(hit, loadRulesFieldName),
		Scalars:              nil,
		Score:                hit.Score,
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

// hitScalarString returns the string value of a hit's scalar cell, or empty
// when the hit lacks the cell, the cell is null, or the value is not a string.
func hitScalarString(hit collection.Hit, columnName string) string {
	cell, found := hit.Scalars[columnName]
	if !found || cell.State != collection.ScalarCellValue {
		return ""
	}
	return cell.Value.String
}

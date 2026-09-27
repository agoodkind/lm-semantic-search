package daemon

import (
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// newConversationItemSource builds the production document ingest source of a
// conversation collection from delivered conversation documents through
// newDocumentItemSource with the conversation declaration. A nil rowReader
// leaves the source without a stored-row reader.
func newConversationItemSource(collectionName string, manifest map[string]string, documents []model.ConversationDocument, rowReader conversationRowReader, absence absencePolicy, backfill bool, force bool, chunkByteBudget ...int) collectionItemSource {
	var stored collectionStoredReader
	if rowReader != nil {
		stored = conversationStoredReader{rowReader: rowReader}
	}
	return newDocumentItemSource(collectionName, semantic.ConversationDeclaration(), stored, documentDelivery{
		manifest:        manifest,
		documents:       documents,
		rows:            nil,
		absence:         absence,
		backfill:        backfill,
		force:           force,
		chunkByteBudget: resolveConversationChunkBudget(chunkByteBudget),
	})
}

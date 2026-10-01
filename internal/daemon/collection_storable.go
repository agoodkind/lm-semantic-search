package daemon

import (
	"strings"

	"goodkind.io/lm-semantic-search/internal/model"
)

func collectionTextIsStorable(text string) bool {
	return strings.TrimSpace(text) != ""
}

// Check the complete field before splitting. Removing whitespace-only interior
// pieces would change the reconstructed text.
func appendStorableCollectionField(
	chunks []model.StoredChunk,
	content string,
	budget int,
	buildChunk func(piece string, partIndex int, multipart bool) model.StoredChunk,
) []model.StoredChunk {
	if !collectionTextIsStorable(content) {
		return chunks
	}
	pieces := splitCollectionText(content, budget)
	multipart := len(pieces) > 1
	for partIndex, piece := range pieces {
		chunks = append(chunks, buildChunk(piece, partIndex, multipart))
	}
	return chunks
}

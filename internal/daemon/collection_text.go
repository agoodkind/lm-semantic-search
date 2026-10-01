package daemon

import (
	"unicode/utf8"

	"goodkind.io/lm-semantic-search/internal/model"
)

func splitCollectionText(text string, chunkByteBudget ...int) []string {
	return splitTextByBytes(text, resolveCollectionChunkBudget(chunkByteBudget))
}

func appendContinuedStorableField(
	chunks []model.StoredChunk,
	content string,
	budget int,
	continuationPrefix string,
	buildChunk func(piece string, partIndex int, multipart bool) model.StoredChunk,
) []model.StoredChunk {
	if continuationPrefix != "" && budget > len(continuationPrefix)+1 {
		budget -= len(continuationPrefix) + 1
	}
	return appendStorableCollectionField(
		chunks,
		content,
		budget,
		func(piece string, partIndex int, multipart bool) model.StoredChunk {
			if partIndex > 0 && continuationPrefix != "" {
				piece = continuationPrefix + "\n" + piece
			}
			return buildChunk(piece, partIndex, multipart)
		},
	)
}

// splitTextByBytes cuts text into UTF-8-aligned pieces of at most maxBytes each.
// A non-positive maxBytes disables splitting and returns the text unchanged.
func splitTextByBytes(text string, maxBytes int) []string {
	if maxBytes <= 0 || len(text) <= maxBytes {
		return []string{text}
	}
	pieces := make([]string, 0, (len(text)+maxBytes-1)/maxBytes)
	start := 0
	for start < len(text) {
		end := start + maxBytes
		if end >= len(text) {
			pieces = append(pieces, text[start:])
			break
		}
		for end > start && !utf8.RuneStart(text[end]) {
			end--
		}
		if end == start {
			_, size := utf8.DecodeRuneInString(text[start:])
			end = start + size
		}
		pieces = append(pieces, text[start:end])
		start = end
	}
	return pieces
}

package daemon

import (
	"fmt"
	"unicode/utf8"

	"goodkind.io/lm-semantic-search/internal/model"
)

// This file owns where a conversation message's text rows live and how a long
// text is cut into several. Whether a piece is worth storing at all belongs to
// its sibling manager_conversation_storable, which owns that rule for every
// field; manager_conversation_tools owns a message's tool payloads.

func conversationRelativePath(conversationID string, messageIndex int32, partIndex int, multipart bool) string {
	basePath := fmt.Sprintf("conv/%s/%d", conversationID, messageIndex)
	if !multipart {
		return basePath
	}
	return fmt.Sprintf("%s/%d", basePath, partIndex)
}

func conversationRelativePathPrefix(conversationID string) string {
	return "conv/" + conversationID + "/"
}

func splitConversationText(text string, chunkByteBudget ...int) []string {
	return splitTextByBytes(text, resolveConversationChunkBudget(chunkByteBudget))
}

// appendContinuedStorableField adds the rows of one field through
// appendStorableConversationField and starts every part after the first with
// continuationPrefix and a newline. A non-empty prefix shorter than the budget
// lowers the budget by its length plus one, and the field splits at that
// lowered budget. An empty prefix changes nothing. A conversation tool call row
// splits with its trimmed tool name as the prefix, and a client row of a
// document collection splits with the prefix its client sends.
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
	return appendStorableConversationField(
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

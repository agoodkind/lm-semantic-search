package daemon

import (
	"fmt"
	"unicode/utf8"
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

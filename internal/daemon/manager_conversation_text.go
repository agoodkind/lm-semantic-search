package daemon

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"goodkind.io/lm-semantic-search/internal/model"
)

// This file owns how a conversation message's own text becomes stored rows:
// where those rows live, whether the text is worth a row at all, and how a long
// text is cut into several. Its sibling manager_conversation_tools does the same
// for a message's tool payloads.

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

// conversationTextIsStorable reports whether a message's text holds something a
// search could return. Text that is empty or only spacing does not, so no row is
// written for it.
//
// Two rules depend on this one answer and have to agree: whether a text row is
// written, and whether a delivered text matches what was stored. If they
// disagreed, a message whose text was declined would compare unequal to the
// empty text the store holds, so it would be re-sent and its derived rows
// removed on every sync for as long as the conversation existed.
func conversationTextIsStorable(text string) bool {
	return strings.TrimSpace(text) != ""
}

// conversationStorableText is the text a message actually stores: itself when it
// holds something, and empty when it does not. It is what a delivered text must
// be reduced to before comparing against a stored one, since the store never
// holds text this function would call unstorable.
func conversationStorableText(text string) string {
	if conversationTextIsStorable(text) {
		return text
	}
	return ""
}

// appendStorableConversationChunk adds a row only when its content holds
// something a search could return.
//
// Every conversation row, whatever kind it is, is appended through here, so the
// rule lives in one place rather than as a condition repeated at each producer.
// A message's text, a tool's distilled summary, its command, its input, its
// output, and a turn's reasoning all reach the store through this function, and
// each one used to decide for itself. They disagreed: the text row tested
// whether its content held a non-whitespace character while the other five
// tested only whether the field was set, so a field holding a space produced a
// row that no search could ever return.
//
// It also drops an individual empty piece of a split, which a per-field
// condition cannot see because the split happens after the field is checked.
func appendStorableConversationChunk(chunks []model.StoredChunk, chunk model.StoredChunk) []model.StoredChunk {
	if !conversationTextIsStorable(chunk.Content) {
		return chunks
	}
	return append(chunks, chunk)
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

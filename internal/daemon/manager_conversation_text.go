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

// appendStorableConversationField splits one of a message's fields and adds a
// row for every piece, or adds nothing at all when the field holds nothing a
// search could return.
//
// The decision is about the whole field, and it is taken here before the split,
// so every producer answers it the same way. A message's text, a tool's
// distilled summary, its command, its input, its output, and a turn's reasoning
// all reach the store through this decision, and each one used to take it alone.
// They disagreed: the text row tested whether its content held a non-whitespace
// character while the other five tested only whether the field was set, so a
// field holding a single space produced a row no search could ever return.
//
// The decision cannot move to the individual piece. A message's stored text is
// rebuilt by concatenating its pieces in order, so a declined interior piece
// would leave the rebuilt text shorter than the delivered one, and the message
// would compare unequal to itself on every later sync forever. A field that is
// worth storing stores all of itself, spacing included.
func appendStorableConversationField(
	chunks []model.StoredChunk,
	content string,
	budget int,
	buildChunk func(piece string, partIndex int, multipart bool) model.StoredChunk,
) []model.StoredChunk {
	if !conversationTextIsStorable(content) {
		return chunks
	}
	pieces := splitConversationText(content, budget)
	multipart := len(pieces) > 1
	for partIndex, piece := range pieces {
		chunks = append(chunks, buildChunk(piece, partIndex, multipart))
	}
	return chunks
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

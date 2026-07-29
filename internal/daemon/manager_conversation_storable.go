package daemon

import (
	"strings"

	"goodkind.io/lm-semantic-search/internal/model"
)

// This file owns one rule: whether a piece of a conversation message is worth
// storing. Six producers write rows, and everything that asks whether a row
// exists or is expected has to answer the same way they do, so the rule and
// every question derived from it live together here.

// conversationTextIsStorable reports whether content holds something a search
// could return. Content that is empty or only spacing does not, so no row is
// written for it.
//
// Two rules depend on this one answer and have to agree: whether a row is
// written, and whether a delivered message matches what was stored. If they
// disagreed, a message whose content was declined would compare unequal to what
// the store holds, so it would be re-sent and its derived rows removed on every
// sync for as long as the conversation existed.
func conversationTextIsStorable(text string) bool {
	return strings.TrimSpace(text) != ""
}

// conversationStorableText is the text a message actually stores: itself when it
// holds something, and empty when it does not.
//
// Both sides of the comparison are reduced through this. The delivered side
// because the store never holds text this rule calls unstorable. The stored side
// because rows written before this rule did hold such text, and reducing only
// one side would call those messages changed and rewrite rows that are meant to
// stay exactly as they are.
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

// conversationDocumentStoresNothing reports whether a delivered message would
// write no row at all, because every field it carries holds nothing a search
// could return.
//
// Such a message has no stored identity, so the delta comparison would read it
// as new on every sync, send it, write nothing, and find it absent again the
// next time. It names the same fields conversationDocumentsToStoredChunks reads,
// in the same order, and TestStoresNothingAgreesWithTheGenerator holds the two
// together.
func conversationDocumentStoresNothing(document model.ConversationDocument) bool {
	if conversationTextIsStorable(document.Text) {
		return false
	}
	if conversationTextIsStorable(document.Thinking) {
		return false
	}
	for _, toolCall := range document.Tools {
		if !conversationToolCallStoresNothing(toolCall) {
			return false
		}
	}
	return true
}

// conversationToolCallStoresNothing reports whether one tool call writes no row,
// because its distilled summary, its command, its input, and its output all hold
// nothing a search could return.
//
// Anything that asks whether a tool call is expected to have a stored row must
// ask this rather than whether the call is present, because a call whose every
// field holds only spacing is present and stores nothing.
func conversationToolCallStoresNothing(toolCall model.ConversationToolCall) bool {
	if conversationTextIsStorable(conversationToolTokenContent(toolCall)) {
		return false
	}
	if conversationTextIsStorable(toolCall.Command) {
		return false
	}
	if conversationTextIsStorable(toolCall.InputJSON) {
		return false
	}
	return !conversationTextIsStorable(toolCall.Output)
}

// conversationDocumentExpectsToolRows reports whether any of a message's tool
// calls writes at least one row.
func conversationDocumentExpectsToolRows(document model.ConversationDocument) bool {
	for _, toolCall := range document.Tools {
		if !conversationToolCallStoresNothing(toolCall) {
			return true
		}
	}
	return false
}

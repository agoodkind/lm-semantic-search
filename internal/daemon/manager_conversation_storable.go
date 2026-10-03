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
// Row generation and stored-family presence use this same rule. Content that
// generation declines cannot suppress a later usable family insert.
func conversationTextIsStorable(text string) bool {
	return strings.TrimSpace(text) != ""
}

// conversationStorableText is the text a message actually stores: itself when it
// holds something, and empty when it does not.
//
// Stored base-family checks reduce historical blank rows through this helper so
// they do not suppress a later usable base insert.
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
// would make the assembled base-family content incomplete. A field that is worth
// storing stores all of itself, spacing included.
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

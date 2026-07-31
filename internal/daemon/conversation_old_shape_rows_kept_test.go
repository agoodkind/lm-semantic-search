package daemon

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// TestOldShapeDerivedRowsReadAsUnchanged pins the rule that a message keeps the
// derived rows it already has.
//
// The store holds millions of rows written when one tool call produced four rows,
// at /tok, /cmd, /in and /out. This build writes one row per call at a different
// path. Asking whether the stored set equals the set this build would write makes
// every one of those messages read as changed forever, which deletes rows that
// still hold what the person saw and re-embeds content the store already has.
// A transcript is append-only, so a recorded tool call never changes afterward.
func TestOldShapeDerivedRowsReadAsUnchanged(t *testing.T) {
	t.Parallel()

	const conversationID = "claude:a"
	document := model.ConversationDocument{
		ConversationID: conversationID,
		MessageIndex:   3,
		Role:           "assistant",
		Text:           "here is the result",
		Tools:          []model.ConversationToolCall{{Name: "Bash", Display: "ls -la", LangHint: "bash"}},
	}
	// The four-row shape an earlier rule wrote for one tool call.
	storedDerived := map[string]string{
		"convtool/claude:a/3/0/tok": "hash-tok",
		"convtool/claude:a/3/0/cmd": "hash-cmd",
		"convtool/claude:a/3/0/in":  "hash-in",
		"convtool/claude:a/3/0/out": "hash-out",
	}
	stored := semantic.StoredMessageState{
		Role:              "assistant",
		Text:              "here is the result",
		HasDerivedContent: true,
	}

	if !conversationDocumentMatchesStored(conversationID, document, stored, storedDerived) {
		t.Fatal("a message whose derived rows predate the current shape read as changed, so its rows would be deleted and re-embedted")
	}
}

// TestReasoningRowsAloneReadAsUnchanged covers the message that carries stored
// reasoning rows while this build offers no reasoning at all, because reasoning
// is not an offered content kind. Its rows stay.
func TestReasoningRowsAloneReadAsUnchanged(t *testing.T) {
	t.Parallel()

	const conversationID = "claude:b"
	document := model.ConversationDocument{
		ConversationID: conversationID,
		MessageIndex:   7,
		Role:           "assistant",
		Text:           "answer",
	}
	storedDerived := map[string]string{"convthink/claude:b/7": "hash-think"}
	stored := semantic.StoredMessageState{Role: "assistant", Text: "answer", HasDerivedContent: true}

	if !conversationDocumentMatchesStored(conversationID, document, stored, storedDerived) {
		t.Fatal("a message with stored reasoning rows read as changed, so those rows would be deleted")
	}
}

// TestChangedTextStillReadsAsChanged proves the presence rule did not blind the
// comparison to a real edit: message text is still compared.
func TestChangedTextStillReadsAsChanged(t *testing.T) {
	t.Parallel()

	const conversationID = "claude:c"
	document := model.ConversationDocument{
		ConversationID: conversationID,
		MessageIndex:   1,
		Role:           "assistant",
		Text:           "the new text",
		Tools:          []model.ConversationToolCall{{Name: "Bash", Display: "ls", LangHint: "bash"}},
	}
	storedDerived := map[string]string{"convtool/claude:c/1/0": "hash-tool"}
	stored := semantic.StoredMessageState{Role: "assistant", Text: "the old text", HasDerivedContent: true}

	if conversationDocumentMatchesStored(conversationID, document, stored, storedDerived) {
		t.Fatal("a message whose text changed read as unchanged, so the edit would never reach the store")
	}
}

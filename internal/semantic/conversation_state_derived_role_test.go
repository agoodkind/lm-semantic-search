package semantic

import (
	"testing"
)

// TestPerConversationLoaderCarriesTheDerivedRole is the third implementation's
// half of the guarantee the two batched loaders already make: a message whose
// only stored rows are derived is registered with the role those rows carry.
//
// Three separate implementations read stored conversation rows back, and each
// one has to answer the same way. The delta comparison rejects a message whose
// stored role differs from the delivered one, so a loader that registered such a
// message with an empty role would report it as changed on every sync, re-send
// it, and remove its derived rows as orphans, for as long as the conversation
// existed.
func TestPerConversationLoaderCarriesTheDerivedRole(t *testing.T) {
	rows := []conversationBatchTestRow{
		{conversationID: "claude:a", relativePath: "conv/claude:a/0", role: "user", content: "ask", messageIndex: 0, hasMessageIndex: true, vector: []float32{1}},
		{conversationID: "claude:a", relativePath: "convtool/claude:a/1/0/tok", role: "assistant", content: "Bash", messageIndex: 1, hasMessageIndex: true, vector: []float32{2}},
		{conversationID: "claude:a", relativePath: "convthink/claude:a/2", role: "assistant", content: "considering", messageIndex: 2, hasMessageIndex: true, vector: []float32{3}},
	}

	assemblies := map[int32]*storedMessageAssembly{}
	legacyRows, err := appendConversationMessageStateRows(
		conversationBatchResultSet(t, rows),
		"conv/claude:a/",
		assemblies,
		map[string][]float32{},
	)
	if err != nil {
		t.Fatalf("appendConversationMessageStateRows returned error: %v", err)
	}
	if legacyRows != 0 {
		t.Fatalf("legacyRows = %d, want 0 because every row carries a message index", legacyRows)
	}
	state := assembleStoredMessageState(assemblies)

	toolOnly, found := state[1]
	if !found {
		t.Fatalf("tool-only message 1 is absent: %#v", state)
	}
	if toolOnly.Role != "assistant" {
		t.Fatalf("tool-only message 1 role = %q, want assistant from its derived row", toolOnly.Role)
	}
	if toolOnly.Text != "" {
		t.Fatalf("tool-only message 1 text = %q, want empty because it stored no text row", toolOnly.Text)
	}
	if !toolOnly.HasDerivedContent {
		t.Fatal("tool-only message 1 does not report derived content")
	}

	reasoningOnly, found := state[2]
	if !found {
		t.Fatalf("reasoning-only message 2 is absent: %#v", state)
	}
	if reasoningOnly.Role != "assistant" {
		t.Fatalf("reasoning-only message 2 role = %q, want assistant", reasoningOnly.Role)
	}

	if state[0].Text != "ask" || state[0].Role != "user" {
		t.Fatalf("message 0 = %#v, want its unchanged base row", state[0])
	}
}

// TestPerConversationLoaderKeepsTheBaseRole pins that a base row's role wins
// however late it arrives, matching the two batched loaders. Rows come back in
// no guaranteed order, and the comparison is against the base row.
func TestPerConversationLoaderKeepsTheBaseRole(t *testing.T) {
	rows := []conversationBatchTestRow{
		{conversationID: "claude:a", relativePath: "convtool/claude:a/0/0/tok", role: "assistant", content: "Bash", messageIndex: 0, hasMessageIndex: true, vector: []float32{1}},
		{conversationID: "claude:a", relativePath: "conv/claude:a/0", role: "user", content: "text", messageIndex: 0, hasMessageIndex: true, vector: []float32{2}},
	}

	assemblies := map[int32]*storedMessageAssembly{}
	if _, err := appendConversationMessageStateRows(
		conversationBatchResultSet(t, rows),
		"conv/claude:a/",
		assemblies,
		map[string][]float32{},
	); err != nil {
		t.Fatalf("appendConversationMessageStateRows returned error: %v", err)
	}
	state := assembleStoredMessageState(assemblies)

	if state[0].Role != "user" {
		t.Fatalf(
			"message 0 role = %q, want user from the base row even though the derived row arrived first",
			state[0].Role,
		)
	}
	if state[0].Text != "text" {
		t.Fatalf("message 0 text = %q, want text", state[0].Text)
	}
}

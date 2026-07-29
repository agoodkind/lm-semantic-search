package localvec

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/semantic"
)

// TestOfflineDerivedRowsRegisterTheirMessage is the offline backend's half of
// the guarantee the Milvus-backed one makes: a message whose only stored rows
// are derived is still found when the stored state is read back.
//
// A turn carrying just a tool call or just reasoning stores no text row, so its
// derived rows are all the store holds for it. If reading them back does not
// register the message, the examination path treats it as new, re-sends it, and
// reads those same derived rows as orphans to remove, on every sync for as long
// as the conversation exists.
//
// The role comes with it because the comparison rejects a message whose stored
// role differs from the delivered one, so an empty role would mismatch every
// time and churn exactly as if the message were missing.
func TestOfflineDerivedRowsRegisterTheirMessage(t *testing.T) {
	t.Parallel()

	rows := []row{
		{
			ConversationID: "claude:a",
			RelativePath:   "conv/claude:a/0",
			MessageIndex:   0,
			Role:           "user",
			Content:        "ask",
			Vector:         []float32{1},
		},
		{
			ConversationID: "claude:a",
			RelativePath:   "convtool/claude:a/1/0/tok",
			MessageIndex:   1,
			Role:           "assistant",
			Content:        "Bash",
			Vector:         []float32{2},
		},
		{
			ConversationID: "claude:a",
			RelativePath:   "convthink/claude:a/2",
			MessageIndex:   2,
			Role:           "assistant",
			Content:        "considering",
			Vector:         []float32{3},
		},
	}

	state, err := buildConversationBatchState(rows, map[string]struct{}{"claude:a": {}})
	if err != nil {
		t.Fatalf("buildConversationBatchState returned error: %v", err)
	}
	stored := state.Rows["claude:a"]

	toolOnly, found := stored.Messages[1]
	if !found {
		t.Fatalf("tool-only message 1 is absent from Messages: %#v", stored.Messages)
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

	reasoningOnly, found := stored.Messages[2]
	if !found {
		t.Fatalf("reasoning-only message 2 is absent from Messages: %#v", stored.Messages)
	}
	if reasoningOnly.Role != "assistant" {
		t.Fatalf("reasoning-only message 2 role = %q, want assistant", reasoningOnly.Role)
	}

	if stored.Messages[0].Text != "ask" || stored.Messages[0].Role != "user" {
		t.Fatalf("message 0 = %#v, want the unchanged base row", stored.Messages[0])
	}
}

// TestOfflineBaseRoleWinsOverDerivedRole pins that a base row's role is the one
// the comparison sees, whatever order the rows arrive in. Both rows carry the
// same role in practice, so a disagreement means one is wrong, and the base row
// is the one the delivered document is compared against.
func TestOfflineBaseRoleWinsOverDerivedRole(t *testing.T) {
	t.Parallel()

	rows := []row{
		{
			ConversationID: "claude:a",
			RelativePath:   "convtool/claude:a/0/0/tok",
			MessageIndex:   0,
			Role:           "assistant",
			Content:        "Bash",
			Vector:         []float32{1},
		},
		{
			ConversationID: "claude:a",
			RelativePath:   "conv/claude:a/0",
			MessageIndex:   0,
			Role:           "user",
			Content:        "text",
			Vector:         []float32{2},
		},
	}

	state, err := buildConversationBatchState(rows, map[string]struct{}{"claude:a": {}})
	if err != nil {
		t.Fatalf("buildConversationBatchState returned error: %v", err)
	}
	stored := state.Rows["claude:a"]

	if stored.Messages[0].Role != "user" {
		t.Fatalf(
			"message 0 role = %q, want user from the base row even though the derived row arrived first",
			stored.Messages[0].Role,
		)
	}
	if stored.Messages[0].Text != "text" {
		t.Fatalf("message 0 text = %q, want text", stored.Messages[0].Text)
	}
}

// TestOfflineDerivedContentIsReported proves a message with both a text row and
// derived rows reports that it has derived content, which the comparison needs
// in order to tell a message that lost its tool call from one that never had
// one.
func TestOfflineDerivedContentIsReported(t *testing.T) {
	t.Parallel()

	rows := []row{
		{
			ConversationID: "claude:a",
			RelativePath:   "conv/claude:a/0",
			MessageIndex:   0,
			Role:           "assistant",
			Content:        "here is what I ran",
			Vector:         []float32{1},
		},
		{
			ConversationID: "claude:a",
			RelativePath:   "convtool/claude:a/0/0/cmd",
			MessageIndex:   0,
			Role:           "assistant",
			Content:        "ls -la",
			Vector:         []float32{2},
		},
	}

	state, err := buildConversationBatchState(rows, map[string]struct{}{"claude:a": {}})
	if err != nil {
		t.Fatalf("buildConversationBatchState returned error: %v", err)
	}
	stored := state.Rows["claude:a"]

	if !stored.Messages[0].HasDerivedContent {
		t.Fatalf("message 0 = %#v, want it to report the derived row it stored", stored.Messages[0])
	}
	if stored.Messages[0].Text != "here is what I ran" {
		t.Fatalf("message 0 text = %q, want its base row unchanged", stored.Messages[0].Text)
	}
	if _, found := stored.DerivedPaths["convtool/claude:a/0/0/cmd"]; !found {
		t.Fatalf("derived paths = %v, want the command row", stored.DerivedPaths)
	}
}

var _ = semantic.StoredMessageState{}

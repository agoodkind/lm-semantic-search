package localvec

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/semantic"
)

// TestOfflineDerivedRowsRegisterTheirMessage is the offline backend's half of
// the guarantee the Milvus-backed one makes: a message whose only stored rows
// are derived is still found when the stored state is read back.
//
// A turn carrying just a tool call or reasoning stores no text row. Registering
// derived-only messages preserves the stored message index and role.
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

// TestOfflineBaseRoleWinsOverDerivedRole pins that a base row owns the assembled
// role, whatever order the rows arrive in.
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
// derived rows reports that it has usable derived content.
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

func TestOfflineBatchMarksOnlyUsableDerivedPaths(t *testing.T) {
	t.Parallel()

	rows := []row{
		{
			ConversationID: "claude:a",
			RelativePath:   "convtool/claude:a/2/0/tok",
			MessageIndex:   2,
			Role:           "assistant",
			Content:        " \n",
			Vector:         []float32{1},
		},
		{
			ConversationID: "claude:a",
			RelativePath:   "convthink/claude:a/2",
			MessageIndex:   2,
			Role:           "assistant",
			Content:        "usable reasoning",
			Vector:         []float32{2},
		},
	}

	state, err := buildConversationBatchState(rows, map[string]struct{}{"claude:a": {}})
	if err != nil {
		t.Fatalf("buildConversationBatchState returned error: %v", err)
	}
	stored := state.Rows["claude:a"]

	if len(stored.DerivedPaths) != 2 {
		t.Fatalf("derived paths = %v, want both stored identities", stored.DerivedPaths)
	}
	if _, found := stored.UsableDerivedPaths["convtool/claude:a/2/0/tok"]; found {
		t.Fatalf("usable derived paths = %v, blank tool row must not satisfy a family", stored.UsableDerivedPaths)
	}
	if _, found := stored.UsableDerivedPaths["convthink/claude:a/2"]; !found {
		t.Fatalf("usable derived paths = %v, want usable thinking row", stored.UsableDerivedPaths)
	}
}

func TestOfflineBatchResolvesScalarLessLegacyPaths(t *testing.T) {
	t.Parallel()

	rows := []row{
		{
			RelativePath: "conv/claude:legacy/5/0",
			Content:      "stored answer",
			Vector:       []float32{1},
		},
		{
			RelativePath: "convthink/claude:legacy/5/old",
			Content:      "stored reasoning",
			Vector:       []float32{2},
		},
	}

	state, err := buildConversationBatchState(
		rows,
		map[string]struct{}{"claude:legacy": {}},
	)
	if err != nil {
		t.Fatalf("buildConversationBatchState returned error: %v", err)
	}
	stored := state.Rows["claude:legacy"]

	if got := stored.Messages[5].Text; got != "stored answer" {
		t.Fatalf("legacy message text = %q, want stored answer", got)
	}
	if _, found := stored.UsableDerivedPaths["convthink/claude:legacy/5/old"]; !found {
		t.Fatalf("usable derived paths = %v, want scalar-less thinking row", stored.UsableDerivedPaths)
	}
}

var _ = semantic.StoredMessageState{}

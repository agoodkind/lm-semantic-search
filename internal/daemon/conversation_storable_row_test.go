package daemon

import (
	"context"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

// TestNoRowKindStoresContentASearchCannotReturn covers every kind of row a
// conversation message produces, not only its text.
//
// Six producers write rows: the message text, a tool's distilled summary, its
// command, its input, its output, and a turn's reasoning. Each one used to
// decide for itself whether its content was worth storing, and they disagreed:
// the text row tested whether its content held a non-whitespace character while
// the other five tested only whether the field was set. A field holding a
// single space therefore produced a row no search could ever return.
//
// The decision now lives at the one point a row is appended, so this test fails
// if any producer regains its own condition and gets it wrong.
func TestNoRowKindStoresContentASearchCannotReturn(t *testing.T) {
	t.Parallel()

	for _, spacing := range []string{" ", "   ", "\n", "\t\n  "} {
		documents := []model.ConversationDocument{{
			ConversationID: "claude:a",
			MessageIndex:   0,
			Role:           "assistant",
			Text:           spacing,
			Thinking:       spacing,
			Tools: []model.ConversationToolCall{{
				Name:      "Bash",
				Command:   spacing,
				InputJSON: spacing,
				Output:    spacing,
			}},
		}}

		chunks, err := conversationDocumentsToStoredChunks(context.Background(), documents)
		if err != nil {
			t.Fatalf("conversationDocumentsToStoredChunks(%q) returned error: %v", spacing, err)
		}
		for _, chunk := range chunks {
			if strings.TrimSpace(chunk.Content) == "" {
				t.Fatalf(
					"spacing %q stored a row with nothing to retrieve at %q",
					spacing,
					chunk.RelativePath,
				)
			}
		}
	}
}

// TestEveryRowKindStillStoresRealContent is the other half: the refusal must not
// take content with it. A message carrying real values in all six places stores
// a row for each.
func TestEveryRowKindStillStoresRealContent(t *testing.T) {
	t.Parallel()

	documents := []model.ConversationDocument{{
		ConversationID: "claude:a",
		MessageIndex:   0,
		Role:           "assistant",
		Text:           "here is what I ran",
		Thinking:       "weighing the options",
		Tools: []model.ConversationToolCall{{
			Name:      "Bash",
			Command:   "ls -la",
			InputJSON: `{"command":"ls -la"}`,
			Output:    "total 0",
		}},
	}}

	chunks, err := conversationDocumentsToStoredChunks(context.Background(), documents)
	if err != nil {
		t.Fatalf("conversationDocumentsToStoredChunks returned error: %v", err)
	}

	wantSuffixes := map[string]string{
		"conv/claude:a/0":           "the message text",
		"convtool/claude:a/0/0/tok": "the tool summary",
		"convtool/claude:a/0/0/cmd": "the tool command",
		"convthink/claude:a/0":      "the reasoning",
	}
	seen := map[string]bool{}
	for _, chunk := range chunks {
		for path := range wantSuffixes {
			if chunk.RelativePath == path || strings.HasPrefix(chunk.RelativePath, path+"/") {
				seen[path] = true
			}
		}
	}
	for path, description := range wantSuffixes {
		if !seen[path] {
			t.Fatalf("%s is missing: no row at or under %q in %d chunks", description, path, len(chunks))
		}
	}

	inputStored := false
	outputStored := false
	for _, chunk := range chunks {
		if strings.HasPrefix(chunk.RelativePath, "convtool/claude:a/0/0/in") {
			inputStored = true
		}
		if strings.HasPrefix(chunk.RelativePath, "convtool/claude:a/0/0/out") {
			outputStored = true
		}
	}
	if !inputStored {
		t.Fatal("the tool input is missing")
	}
	if !outputStored {
		t.Fatal("the tool output is missing")
	}
}

// TestASplitMessageStoresEveryPieceOfItsText is the reason the decision cannot
// move from the field to the individual piece.
//
// A message's stored text is rebuilt by concatenating its pieces in order. A
// piece that is declined for holding only spacing leaves the rebuilt text
// shorter than the delivered one, so the message compares unequal to itself on
// every later sync, is re-sent, and has its derived rows removed as orphans, for
// as long as the conversation exists. A field worth storing stores all of
// itself.
func TestASplitMessageStoresEveryPieceOfItsText(t *testing.T) {
	t.Parallel()

	text := "alpha" + strings.Repeat(" ", 64) + "omega"
	documents := []model.ConversationDocument{{
		ConversationID: "claude:a",
		MessageIndex:   0,
		Role:           "assistant",
		Text:           text,
	}}

	chunks, err := conversationDocumentsToStoredChunks(context.Background(), documents, 8)
	if err != nil {
		t.Fatalf("conversationDocumentsToStoredChunks returned error: %v", err)
	}

	var rebuilt strings.Builder
	pieces := 0
	for _, chunk := range chunks {
		if strings.HasPrefix(chunk.RelativePath, "conv/claude:a/0") {
			rebuilt.WriteString(chunk.Content)
			pieces++
		}
	}
	if pieces < 2 {
		t.Fatalf("expected the text to split into several pieces, got %d", pieces)
	}
	if rebuilt.String() != text {
		t.Fatalf(
			"the stored pieces rebuild to %q, want the delivered text %q",
			rebuilt.String(),
			text,
		)
	}
}

// TestAFieldOfOnlySpacingStoresNothing is the other half of the same rule,
// taken at the field rather than the piece.
func TestAFieldOfOnlySpacingStoresNothing(t *testing.T) {
	t.Parallel()

	documents := []model.ConversationDocument{{
		ConversationID: "claude:a",
		MessageIndex:   0,
		Role:           "assistant",
		Text:           strings.Repeat(" ", 64),
	}}

	chunks, err := conversationDocumentsToStoredChunks(context.Background(), documents, 8)
	if err != nil {
		t.Fatalf("conversationDocumentsToStoredChunks returned error: %v", err)
	}
	for _, chunk := range chunks {
		if strings.HasPrefix(chunk.RelativePath, "conv/claude:a/0") {
			t.Fatalf("a text of only spacing stored a row at %q", chunk.RelativePath)
		}
	}
}

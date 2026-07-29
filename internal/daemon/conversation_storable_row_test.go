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

// TestAnEmptyPieceOfASplitIsDropped covers what a per-field condition cannot
// see. A field is checked before it is split, so a split that yields an empty
// piece stored that piece regardless of how the field was checked.
func TestAnEmptyPieceOfASplitIsDropped(t *testing.T) {
	t.Parallel()

	documents := []model.ConversationDocument{{
		ConversationID: "claude:a",
		MessageIndex:   0,
		Role:           "assistant",
		Text:           "alpha" + strings.Repeat(" ", 64) + "omega",
	}}

	chunks, err := conversationDocumentsToStoredChunks(context.Background(), documents, 8)
	if err != nil {
		t.Fatalf("conversationDocumentsToStoredChunks returned error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("a text carrying real words stored nothing")
	}
	for _, chunk := range chunks {
		if strings.TrimSpace(chunk.Content) == "" {
			t.Fatalf("a split piece of pure spacing was stored at %q", chunk.RelativePath)
		}
	}
}

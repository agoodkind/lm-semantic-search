package daemon

import (
	"context"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

// TestNoTextWritesNoTextRow proves a delivered message carrying no text stores
// no row for that text.
//
// Splitting an empty string yields one empty piece, so the text loop used to
// store exactly one row holding nothing. Such a row can never be returned by a
// search, because there is nothing in it to match, and it occupies a vector for
// as long as it exists. They were 33.6% of the live conversation collection.
//
// Every sibling producer on this path already declines to write an absent field:
// the tool token, command, input, output, and reasoning rows each write only
// when their content is present. Only the text row wrote regardless.
func TestNoTextWritesNoTextRow(t *testing.T) {
	t.Parallel()

	documents := []model.ConversationDocument{
		{
			ConversationID: "claude:a",
			MessageIndex:   0,
			Role:           "assistant",
			Text:           "",
			Tools: []model.ConversationToolCall{
				{Name: "Bash", Display: "ls -la", LangHint: "bash"},
			},
		},
		{
			ConversationID: "claude:a",
			MessageIndex:   1,
			Role:           "assistant",
			Text:           "",
			Thinking:       "weighing the options",
		},
		{
			ConversationID: "claude:a",
			MessageIndex:   2,
			Role:           "user",
			Text:           "what does this do",
		},
	}

	chunks, err := conversationDocumentsToStoredChunks(context.Background(), documents)
	if err != nil {
		t.Fatalf("conversationDocumentsToStoredChunks returned error: %v", err)
	}

	for _, chunk := range chunks {
		if chunk.Content == "" {
			t.Fatalf("stored a chunk with no content at %q", chunk.RelativePath)
		}
	}

	basePaths := map[string]bool{}
	for _, chunk := range chunks {
		if !isDerivedConversationChunk(chunk) {
			basePaths[chunk.RelativePath] = true
		}
	}
	if len(basePaths) != 1 {
		t.Fatalf("base text rows = %v, want only the message that carried text", basePaths)
	}
	if !basePaths["conv/claude:a/2"] {
		t.Fatalf("base text rows = %v, want conv/claude:a/2", basePaths)
	}

	// The derived rows of the text-free messages are still written, because the
	// tool call and the reasoning are content a person saw.
	derivedCount := 0
	for _, chunk := range chunks {
		if isDerivedConversationChunk(chunk) {
			derivedCount++
		}
	}
	if derivedCount == 0 {
		t.Fatal("no derived rows were written for the tool call and the reasoning")
	}
}

// TestWhitespaceOnlyTextWritesNoTextRow covers text that holds only spacing.
//
// A sender is expected to normalize such text away before delivering it, but
// proto3 cannot express absence for a string field, so an unset field and an
// empty one are identical bytes while a single space is content on the wire.
// Storing a row that a search can never return is the same waste whether the
// spacing arrived deliberately or not.
func TestWhitespaceOnlyTextWritesNoTextRow(t *testing.T) {
	t.Parallel()

	for _, text := range []string{" ", "   ", "\n", "\t\n  "} {
		documents := []model.ConversationDocument{{
			ConversationID: "claude:a",
			MessageIndex:   0,
			Role:           "assistant",
			Text:           text,
			Tools:          []model.ConversationToolCall{{Name: "Bash", Display: "ls", LangHint: "bash"}},
		}}
		chunks, err := conversationDocumentsToStoredChunks(context.Background(), documents)
		if err != nil {
			t.Fatalf("conversationDocumentsToStoredChunks(%q) returned error: %v", text, err)
		}
		for _, chunk := range chunks {
			if strings.TrimSpace(chunk.Content) == "" {
				t.Fatalf("text %q stored a chunk with nothing to retrieve at %q", text, chunk.RelativePath)
			}
		}
	}
}

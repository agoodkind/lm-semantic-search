package daemon

import (
	"context"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// TestWhitespaceOnlyTextSettlesAgainstItsStoredRows proves a delivered message
// whose text holds only spacing converges rather than being re-sent forever.
//
// Two rules decide what happens to such a message, and they have to agree. The
// generator declines to write a text row for it, so the store holds an empty
// text. The comparison then asks whether the delivered text equals the stored
// one. Comparing the raw delivered spacing against that empty stored text finds
// a difference every time, so the message would be re-sent, its derived rows
// removed, and everything embedded again on every sync for as long as the
// conversation exists.
//
// A sender is expected to normalize such text away before delivering it, but the
// receiver cannot rely on that: the two rules live here and must agree here.
func TestWhitespaceOnlyTextSettlesAgainstItsStoredRows(t *testing.T) {
	t.Parallel()

	for _, spacing := range []string{" ", "   ", "\n", "\t\n  "} {
		document := model.ConversationDocument{
			ConversationID: "claude:a",
			MessageIndex:   3,
			Role:           "assistant",
			Text:           spacing,
			Tools: []model.ConversationToolCall{
				{Name: "Bash", Command: "ls -la", Output: "total 0"},
			},
		}

		chunks, err := conversationDocumentsToStoredChunks(
			context.Background(),
			[]model.ConversationDocument{document},
		)
		if err != nil {
			t.Fatalf("conversationDocumentsToStoredChunks(%q) returned error: %v", spacing, err)
		}

		storedDerived := map[string]string{}
		storedRole := ""
		for _, chunk := range chunks {
			if !isDerivedConversationChunk(chunk) {
				t.Fatalf("text %q wrote a base row at %q, want none", spacing, chunk.RelativePath)
			}
			storedDerived[chunk.RelativePath] = semantic.ContentVectorKey(chunk.Content)
			if storedRole == "" {
				storedRole = chunk.Role
			}
		}

		matches, err := conversationDocumentMatchesStored(
			context.Background(),
			"claude:a",
			document,
			semantic.StoredMessageState{
				Role:              storedRole,
				Text:              "",
				HasDerivedContent: true,
			},
			storedDerived,
		)
		if err != nil {
			t.Fatalf("conversationDocumentMatchesStored(%q) returned error: %v", spacing, err)
		}
		if !matches {
			t.Fatalf(
				"text %q did not match its own stored rows, so every sync would re-send it and delete its derived rows",
				spacing,
			)
		}
	}
}

// TestTextWithContentStillComparesExactly pins that only spacing-only text is
// normalized for the comparison.
//
// Leading and trailing spacing on text that has content is stored verbatim, so
// the comparison must keep seeing it. Treating the two as equal would make a
// message that genuinely gained or lost surrounding spacing read as unchanged,
// and the stored row would keep the old text forever.
func TestTextWithContentStillComparesExactly(t *testing.T) {
	t.Parallel()

	document := model.ConversationDocument{
		ConversationID: "claude:a",
		MessageIndex:   0,
		Role:           "assistant",
		Text:           "  answer  ",
	}

	matches, err := conversationDocumentMatchesStored(
		context.Background(),
		"claude:a",
		document,
		semantic.StoredMessageState{Role: "assistant", Text: "answer", HasDerivedContent: false},
		map[string]string{},
	)
	if err != nil {
		t.Fatalf("conversationDocumentMatchesStored returned error: %v", err)
	}
	if matches {
		t.Fatal("text differing only in surrounding spacing matched, so the stored row would never be corrected")
	}
}

package daemon

import (
	"context"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// storesNothingCases covers the shapes a delivered message takes: real content
// in each field on its own, spacing in each field on its own, and the mixtures
// that decide whether the predicate and the generator can disagree.
func storesNothingCases() []model.ConversationDocument {
	spacing := "  \n\t "
	return []model.ConversationDocument{
		{ConversationID: "claude:a", MessageIndex: 0, Role: "assistant"},
		{ConversationID: "claude:a", MessageIndex: 1, Role: "assistant", Text: "real text"},
		{ConversationID: "claude:a", MessageIndex: 2, Role: "assistant", Text: spacing},
		{ConversationID: "claude:a", MessageIndex: 3, Role: "assistant", Thinking: "real reasoning"},
		{ConversationID: "claude:a", MessageIndex: 4, Role: "assistant", Thinking: spacing},
		{
			ConversationID: "claude:a", MessageIndex: 5, Role: "assistant",
			Tools: []model.ConversationToolCall{{Name: "Bash", Command: "ls"}},
		},
		{
			ConversationID: "claude:a", MessageIndex: 6, Role: "assistant",
			Tools: []model.ConversationToolCall{{Name: spacing, Command: spacing, InputJSON: spacing, Output: spacing}},
		},
		{
			ConversationID: "claude:a", MessageIndex: 7, Role: "assistant",
			Tools: []model.ConversationToolCall{{Name: "", Command: "", InputJSON: "", Output: "real output"}},
		},
		{
			ConversationID: "claude:a", MessageIndex: 8, Role: "assistant",
			Text:  spacing,
			Tools: []model.ConversationToolCall{{Name: spacing, Output: spacing}},
		},
		{
			ConversationID: "claude:a", MessageIndex: 9, Role: "assistant",
			Text: spacing, Thinking: spacing,
			Tools: []model.ConversationToolCall{{Name: spacing, Command: spacing}},
		},
	}
}

// TestStoresNothingAgreesWithTheGenerator holds the predicate to the code it
// describes. conversationDocumentStoresNothing names the fields by hand, so a
// producer added or changed without updating it would let a message that does
// store rows be skipped by the delta, or one that stores none be re-sent on
// every sync. Generating the chunks is the only authority on which happens.
func TestStoresNothingAgreesWithTheGenerator(t *testing.T) {
	t.Parallel()

	for _, document := range storesNothingCases() {
		chunks, err := conversationDocumentsToStoredChunks(
			context.Background(),
			[]model.ConversationDocument{document},
		)
		if err != nil {
			t.Fatalf("message %d: generator returned error: %v", document.MessageIndex, err)
		}
		generatorStoresNothing := len(chunks) == 0
		predicateStoresNothing := conversationDocumentStoresNothing(document)
		if generatorStoresNothing != predicateStoresNothing {
			t.Fatalf(
				"message %d: predicate says stores-nothing=%v but the generator produced %d chunks",
				document.MessageIndex,
				predicateStoresNothing,
				len(chunks),
			)
		}
	}
}

// TestAMessageStoringNothingIsNotResentForever is the churn this predicate
// exists to stop.
//
// A message whose every field holds only spacing writes no row, so it never
// appears in the stored state. Without the predicate the delta reads it as new
// on every pass, sends it, writes nothing, and finds it absent again, for as
// long as the conversation exists.
func TestAMessageStoringNothingIsNotResentForever(t *testing.T) {
	t.Parallel()

	spacing := "   "
	documents := []model.ConversationDocument{
		{ConversationID: "claude:a", MessageIndex: 0, Role: "user", Text: "a real question"},
		{ConversationID: "claude:a", MessageIndex: 1, Role: "assistant", Text: spacing, Thinking: spacing},
	}

	// The store holds the first message and nothing for the second, which is
	// what the generator would produce for this delivery.
	stored := semantic.ConversationStoredRows{
		Messages: map[int32]semantic.StoredMessageState{
			0: {Role: "user", Text: "a real question"},
		},
		DerivedPaths: map[string]string{},
	}

	diff, err := diffConversationMessages(context.Background(), "claude:a", documents, stored)
	if err != nil {
		t.Fatalf("diffConversationMessages returned error: %v", err)
	}
	if len(diff.documents) != 0 {
		t.Fatalf(
			"the delta wants to send %d documents, want none: %#v",
			len(diff.documents),
			diff.documents,
		)
	}
	if len(diff.removalPaths) != 0 || len(diff.removalPrefixes) != 0 {
		t.Fatalf(
			"the delta wants removals (%d paths, %d prefixes), want none",
			len(diff.removalPaths),
			len(diff.removalPrefixes),
		)
	}
}

// TestAMessageStoringNothingStillRepairsItsOrphans pins that the skip does not
// swallow a repair. A message with nothing storable but surviving derived rows
// from an earlier version must still have those rows purged.
func TestAMessageStoringNothingStillRepairsItsOrphans(t *testing.T) {
	t.Parallel()

	documents := []model.ConversationDocument{
		{ConversationID: "claude:a", MessageIndex: 0, Role: "assistant", Text: "   "},
	}
	stored := semantic.ConversationStoredRows{
		Messages: map[int32]semantic.StoredMessageState{},
		DerivedPaths: map[string]string{
			"convtool/claude:a/0/0/tok": "somehash",
		},
	}

	diff, err := diffConversationMessages(context.Background(), "claude:a", documents, stored)
	if err != nil {
		t.Fatalf("diffConversationMessages returned error: %v", err)
	}
	removals := len(diff.removalPaths) + len(diff.removalPrefixes)
	if removals == 0 {
		t.Fatal("a surviving orphaned derived row was not purged")
	}
	if !strings.Contains(strings.Join(append(diff.removalPaths, diff.removalPrefixes...), " "), "claude:a") {
		t.Fatalf("the removal does not name the conversation: %#v", diff)
	}
}

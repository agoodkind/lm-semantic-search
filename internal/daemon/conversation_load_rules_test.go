package daemon

import (
	"context"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/model"
)

// TestConversationDocumentsCarryLoadRulesOntoEveryDerivedChunk proves the
// caller's loading-rules tag reaches each stored row a document produces, so a
// search hit on any row family can report which rules produced its message
// index.
func TestConversationDocumentsCarryLoadRulesOntoEveryDerivedChunk(t *testing.T) {
	t.Parallel()

	chunks, err := conversationDocumentsToStoredChunks(context.Background(), []model.ConversationDocument{{
		ConversationID: "conv-rules",
		MessageIndex:   2,
		Role:           "assistant",
		TimestampUnix:  1712345600,
		Text:           "answer text",
		Thinking:       "reasoning text",
		LoadRules:      "v1;injected",
		Tools: []model.ConversationToolCall{{
			Name:    "Bash",
			Display: "true",
		}},
	}})
	if err != nil {
		t.Fatalf("conversationDocumentsToStoredChunks returned error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("no chunks produced")
	}
	for _, chunk := range chunks {
		if chunk.LoadRules != "v1;injected" {
			t.Fatalf("chunk %s LoadRules = %q, want v1;injected", chunk.RelativePath, chunk.LoadRules)
		}
	}
}

// TestConversationSearchResultsCarryLoadRules proves the stored tag returns on
// the wire hit, which is how the caller learns the rules a row was written
// under.
func TestConversationSearchResultsCarryLoadRules(t *testing.T) {
	t.Parallel()

	results := conversationSearchResults([]model.StoredChunk{{
		ConversationID: "conv-rules",
		MessageIndex:   2,
		Role:           "assistant",
		Content:        "answer text",
		LoadRules:      "v1;injected",
	}})
	if len(results) != 1 {
		t.Fatalf("results = %d, want 1", len(results))
	}
	if results[0].GetLoadRules() != "v1;injected" {
		t.Fatalf("hit LoadRules = %q, want v1;injected", results[0].GetLoadRules())
	}
}

// TestPBConversationDocumentsDecodeLoadRules proves the wire field survives the
// proto-to-model conversion at the upsert boundary.
func TestPBConversationDocumentsDecodeLoadRules(t *testing.T) {
	t.Parallel()

	documents := pbConversationDocuments([]*pb.ConversationDocument{{
		ConversationId: "conv-rules",
		MessageIndex:   2,
		Role:           "user",
		Text:           "typed prompt",
		LoadRules:      "v1;",
	}})
	if len(documents) != 1 {
		t.Fatalf("documents = %d, want 1", len(documents))
	}
	if documents[0].LoadRules != "v1;" {
		t.Fatalf("document LoadRules = %q, want v1;", documents[0].LoadRules)
	}
}

package daemon

import (
	"context"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

func TestSplitTextByBytesRespectsBudget(t *testing.T) {
	t.Parallel()
	if got := splitTextByBytes("short", 100); len(got) != 1 || got[0] != "short" {
		t.Fatalf("small input should be one piece, got %v", got)
	}
	if got := splitTextByBytes("anything", 0); len(got) != 1 {
		t.Fatalf("non-positive budget disables splitting, got %d pieces", len(got))
	}

	text := strings.Repeat("x", 250)
	pieces := splitTextByBytes(text, 100)
	if len(pieces) != 3 {
		t.Fatalf("expected 3 pieces at budget 100, got %d", len(pieces))
	}
	var joined strings.Builder
	for _, piece := range pieces {
		if len(piece) > 100 {
			t.Fatalf("piece over budget: %d bytes", len(piece))
		}
		joined.WriteString(piece)
	}
	if joined.String() != text {
		t.Fatal("pieces did not round-trip to the original text")
	}
}

func TestConversationChunkBudgetLowersSplit(t *testing.T) {
	t.Parallel()

	text := strings.Repeat("a", 2500)
	chunks, err := conversationDocumentsToStoredChunks(context.Background(), []model.ConversationDocument{{
		ConversationID: "thread-cap",
		MessageIndex:   3,
		Role:           "user",
		Text:           text,
	}}, 1000)
	if err != nil {
		t.Fatalf("conversationDocumentsToStoredChunks returned error: %v", err)
	}

	var textRows []model.StoredChunk
	for _, chunk := range chunks {
		if chunk.RelativePath == "conv/thread-cap/3" || strings.HasPrefix(chunk.RelativePath, "conv/thread-cap/3/") {
			textRows = append(textRows, chunk)
		}
	}
	if len(textRows) < 3 {
		t.Fatalf("2500-byte message at budget 1000 should split into at least 3 rows, got %d", len(textRows))
	}
	for _, chunk := range textRows {
		if len(chunk.Content) > 1000 {
			t.Fatalf("row over budget: %d bytes", len(chunk.Content))
		}
	}
}

func TestConversationChunkBudgetDefaultsToVarcharCap(t *testing.T) {
	t.Parallel()

	// No budget argument resolves to the varchar-safe default, so a message just
	// over conversationChunkMaxBytes splits into two rows (prior behavior).
	text := strings.Repeat("a", conversationChunkMaxBytes+5)
	chunks, err := conversationDocumentsToStoredChunks(context.Background(), []model.ConversationDocument{{
		ConversationID: "thread-default",
		MessageIndex:   1,
		Role:           "user",
		Text:           text,
	}})
	if err != nil {
		t.Fatalf("conversationDocumentsToStoredChunks returned error: %v", err)
	}
	textRows := 0
	for _, chunk := range chunks {
		if strings.HasPrefix(chunk.RelativePath, "conv/thread-default/1") {
			textRows++
			if len(chunk.Content) > conversationChunkMaxBytes {
				t.Fatalf("row over varchar cap: %d bytes", len(chunk.Content))
			}
		}
	}
	if textRows != 2 {
		t.Fatalf("expected 2 rows at the default cap, got %d", textRows)
	}
}

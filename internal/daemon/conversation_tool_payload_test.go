package daemon

import (
	"context"
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
)

// realToolPayload is one tool payload captured from the live conversation
// collection, together with the row shape production had stored for it. The
// payloads are real provider output rather than invented fixtures, because the
// behaviour under test only shows up on what Claude, Codex, and Zed actually
// emit: content-block envelopes, bare JSON objects, and plain command output.
type realToolPayload struct {
	SourceGroup      string `json:"source_group"`
	Leaf             string `json:"leaf"`
	Shape            string `json:"shape"`
	StoredPieceCount int    `json:"storedPieceCount"`
	// StoredShortestPieceChars is the shortest row production stored for this
	// payload. Every captured payload has a value of 1 to 3, because the rows it
	// was stored as included pieces holding only the punctuation between two
	// serialized values.
	StoredShortestPieceChars int `json:"storedShortestPieceChars"`
	// ProductionLangHint is the hint the payload was stored under. It selected a
	// tree-sitter grammar and a synthesized file name, and reproducing it is what
	// makes these tests fail against the splitter that stored these rows. Splitting
	// no longer consults it, so it changes nothing about the behaviour under test.
	ProductionLangHint string `json:"productionLangHint"`
	Payload            string `json:"payload"`
}

func loadRealToolPayloads(t *testing.T) []realToolPayload {
	t.Helper()
	raw, err := os.ReadFile("testdata/conversation_tool_payloads.json")
	if err != nil {
		t.Fatalf("read real tool payloads: %v", err)
	}
	var payloads []realToolPayload
	if err := json.Unmarshal(raw, &payloads); err != nil {
		t.Fatalf("decode real tool payloads: %v", err)
	}
	if len(payloads) == 0 {
		t.Fatal("real tool payload testdata is empty")
	}
	return payloads
}

// storedRowsForPayload runs one real payload through the single derived-chunk
// chokepoint at the production byte budget and returns the rows a searcher
// would see for it, in part order.
func storedRowsForPayload(t *testing.T, payload realToolPayload) []model.StoredChunk {
	t.Helper()
	toolCall := model.ConversationToolCall{Name: "Bash", LangHint: payload.ProductionLangHint}
	if payload.Leaf == "in" {
		toolCall.InputJSON = payload.Payload
	} else {
		toolCall.Output = payload.Payload
	}
	document := model.ConversationDocument{
		ConversationID: "conv-real",
		MessageIndex:   1,
		Role:           "assistant",
		Text:           "ran a command",
		Tools:          []model.ConversationToolCall{toolCall},
	}
	chunks, err := conversationDocumentsToStoredChunks(
		context.Background(),
		[]model.ConversationDocument{document},
		config.EmbedChunkByteBudget(0),
	)
	if err != nil {
		t.Fatalf("%s: conversationDocumentsToStoredChunks returned error: %v", payload.SourceGroup, err)
	}
	prefix := "convtool/conv-real/1/0/" + payload.Leaf + "/"
	rows := make([]model.StoredChunk, 0, len(chunks))
	for _, chunk := range chunks {
		if strings.HasPrefix(chunk.RelativePath, prefix) {
			rows = append(rows, chunk)
		}
	}
	if len(rows) == 0 {
		t.Fatalf("%s: no rows stored under %s", payload.SourceGroup, prefix)
	}
	return rows
}

// TestRealToolPayloadRowsReconstructThePayload pins the contract a reader of the
// collection depends on: the rows stored for one tool payload are that payload,
// split and nothing else. Joining them in part order returns the exact bytes the
// feeder delivered, so no part of a command or its result is dropped at a
// boundary and none is stored twice.
func TestRealToolPayloadRowsReconstructThePayload(t *testing.T) {
	t.Parallel()

	for _, payload := range loadRealToolPayloads(t) {
		rows := storedRowsForPayload(t, payload)
		var rebuilt strings.Builder
		for _, row := range rows {
			rebuilt.WriteString(row.Content)
		}
		if rebuilt.String() != payload.Payload {
			t.Errorf(
				"%s (%s): joined rows do not reproduce the payload: got %d bytes, want %d",
				payload.SourceGroup, payload.Shape, rebuilt.Len(), len(payload.Payload),
			)
		}
	}
}

// TestRealToolPayloadRowsStayWithinEmbedBudget pins that a tool payload obeys
// the same embedder-derived cap as the message text beside it, so raising or
// lowering the configured cap moves tool rows with it.
func TestRealToolPayloadRowsStayWithinEmbedBudget(t *testing.T) {
	t.Parallel()

	budget := config.EmbedChunkByteBudget(0)
	for _, payload := range loadRealToolPayloads(t) {
		for _, row := range storedRowsForPayload(t, payload) {
			if len(row.Content) > budget {
				t.Errorf(
					"%s: row %s holds %d bytes, want at most the %d-byte budget",
					payload.SourceGroup, row.RelativePath, len(row.Content), budget,
				)
			}
		}
	}
}

// TestRealToolPayloadRowsCarryRetrievableContent pins that every row stored for
// a real payload is something a search can return usefully: it carries a word a
// person could have meant, rather than only the punctuation that separated two
// values in the provider's serialization.
func TestRealToolPayloadRowsCarryRetrievableContent(t *testing.T) {
	t.Parallel()

	const shortestUsefulWord = 3
	for _, payload := range loadRealToolPayloads(t) {
		for _, row := range storedRowsForPayload(t, payload) {
			if longestWordRun(row.Content) < shortestUsefulWord {
				t.Errorf(
					"%s: row %s carries no word of %d or more characters: %q",
					payload.SourceGroup, row.RelativePath, shortestUsefulWord, row.Content,
				)
			}
		}
	}
}

// TestRealToolPayloadKeepsWholePayloadsInOneRow pins that a payload within the
// budget is stored as exactly one row, so the common tool result stays whole and
// a reader sees the command's output as it was produced.
func TestRealToolPayloadKeepsWholePayloadsInOneRow(t *testing.T) {
	t.Parallel()

	budget := config.EmbedChunkByteBudget(0)
	checked := 0
	for _, payload := range loadRealToolPayloads(t) {
		if len(payload.Payload) > budget {
			continue
		}
		checked++
		rows := storedRowsForPayload(t, payload)
		if len(rows) != 1 {
			t.Errorf(
				"%s: %d-byte payload stored as %d rows, want 1",
				payload.SourceGroup, len(payload.Payload), len(rows),
			)
		}
	}
	if checked == 0 {
		t.Log("no captured payload is within the budget; whole-payload storage is unexercised here")
	}
}

// TestRealToolPayloadRowsUseIndexedPathScheme pins the stored path scheme for
// tool payload rows. It is deliberately unchanged from what the collection
// already holds, so old and new rows address the same way and the per-message
// removal prefixes keep matching both.
func TestRealToolPayloadRowsUseIndexedPathScheme(t *testing.T) {
	t.Parallel()

	for _, payload := range loadRealToolPayloads(t) {
		rows := storedRowsForPayload(t, payload)
		for partIndex, row := range rows {
			want := "convtool/conv-real/1/0/" + payload.Leaf + "/" + strconv.Itoa(partIndex)
			if row.RelativePath != want {
				t.Errorf("%s: row %d has path %q, want %q", payload.SourceGroup, partIndex, row.RelativePath, want)
			}
		}
	}
}

// longestWordRun returns the longest run of letters and digits in content.
func longestWordRun(content string) int {
	longest := 0
	current := 0
	for _, character := range content {
		isWord := (character >= 'a' && character <= 'z') ||
			(character >= 'A' && character <= 'Z') ||
			(character >= '0' && character <= '9')
		if isWord {
			current++
			if current > longest {
				longest = current
			}
			continue
		}
		current = 0
	}
	return longest
}

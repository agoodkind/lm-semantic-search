//go:build live

package live

import (
	"strings"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

// sharedPreamble stands in for the agent-instruction preamble that opens a large
// number of otherwise unrelated sessions. It is the content that dominates the
// corpus duplication: the same bytes at the same early position in many separate
// conversations, each of which used to pay its own embedding call. It is long
// enough that the chunker splits it, so the reuse under test is reuse of several
// chunks rather than of one convenient string.
var sharedPreamble = strings.Repeat(
	"You are an agent. Follow the repository conventions, prefer small changes, and explain your reasoning. ",
	40,
)

// preambleConversation is a two-message conversation that opens with the shared
// preamble and then says something unique to this conversation, so two such
// conversations overlap on the preamble and differ everywhere else.
func preambleConversation(id string) []*pb.ConversationDocument {
	return []*pb.ConversationDocument{
		{
			ConversationId: id,
			MessageIndex:   0,
			Role:           "user",
			TimestampUnix:  1712345000,
			Text:           sharedPreamble,
		},
		{
			ConversationId: id,
			MessageIndex:   1,
			Role:           "assistant",
			TimestampUnix:  1712345001,
			Text:           "acknowledged, working on the request for " + id,
		},
	}
}

func upsertConversation(t *testing.T, h *harness, ids ...string) (int32, int32) {
	t.Helper()
	convs := make(map[string][]*pb.ConversationDocument, len(ids))
	for _, id := range ids {
		convs[id] = preambleConversation(id)
	}
	job := h.upsert(convs, pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN, false, false)
	requireCompleted(t, job, "ingest of "+strings.Join(ids, ","))
	return job.Progress.ChunksReused, job.Progress.ChunksEmbedded
}

// TestIdenticalContentEmbedsOnceAcrossConversations proves the engine embeds a
// piece of content once for the whole collection rather than once per
// conversation that contains it.
//
// The two conversations are ingested in separate upserts, so the second runs
// with its own job, its own batch, and its own stored-row read. Nothing about it
// overlaps the first except the content itself. Before the content-addressed
// vector store, reuse was resolved only from the rows of the conversations being
// delivered, so the second ingest re-embedded the shared preamble and the
// no-repeat assertion below read 2.
func TestIdenticalContentEmbedsOnceAcrossConversations(t *testing.T) {
	h := newHarness(t)

	_, firstEmbedded := upsertConversation(t, h, "conv-preamble-first")
	if firstEmbedded <= 0 {
		t.Fatalf("first ingest embedded %d chunks, want more than 0; nothing was indexed", firstEmbedded)
	}

	secondReused, secondEmbedded := upsertConversation(t, h, "conv-preamble-second")

	if repeat := h.embedRecorder.maxRepeat(); repeat != 1 {
		t.Fatalf(
			"some content was embedded %d times across two separate conversations, want 1; the second ingest re-embedded content the collection already held (reused=%d embedded=%d)",
			repeat, secondReused, secondEmbedded,
		)
	}
	if secondReused <= 0 {
		t.Fatalf("second ingest reported ChunksReused = %d, want more than 0; the shared preamble should have come from the store", secondReused)
	}
	if secondEmbedded >= firstEmbedded {
		t.Fatalf(
			"second ingest embedded %d chunks and the first embedded %d; the second should embed strictly fewer because only its unique turn is new",
			secondEmbedded, firstEmbedded,
		)
	}

	// Reuse skips the embedding call, never the row, so both conversations must
	// still own their own rows at their own paths.
	for _, id := range []string{"conv-preamble-first", "conv-preamble-second"} {
		if rows := h.countRowsWithPrefix(convBasePrefix(id)); rows == 0 {
			t.Fatalf("conversation %s stored %d base rows, want more than 0; reuse must not skip rows", id, rows)
		}
	}
}

// TestRepeatedContentWithinOneIngestEmbedsOnce proves the same rule holds inside
// a single delivery: a preamble shared by several conversations in one upsert
// costs one embedding call per distinct chunk, not one per conversation. This is
// the intra-run half of the same defect, and it holds even though the packer
// splits those conversations across several embedding requests.
func TestRepeatedContentWithinOneIngestEmbedsOnce(t *testing.T) {
	h := newHarness(t)

	reused, embedded := upsertConversation(t, h, "conv-batch-a", "conv-batch-b", "conv-batch-c", "conv-batch-d")

	if repeat := h.embedRecorder.maxRepeat(); repeat != 1 {
		t.Fatalf("some content was embedded %d times within one four-conversation ingest, want 1 (reused=%d embedded=%d)", repeat, reused, embedded)
	}
	if reused <= 0 {
		t.Fatalf("four conversations sharing a preamble reported ChunksReused = %d, want more than 0", reused)
	}
	if h.embedRecorder.distinct() == 0 {
		t.Fatal("the embedder was never called, so this test proves nothing about reuse")
	}
	// Each conversation contributes one unique assistant turn, and they share
	// every preamble chunk, so the embedder must see strictly fewer inputs than
	// the run stored chunks for.
	if int32(h.embedRecorder.distinct()) >= reused+embedded {
		t.Fatalf("embedder saw %d distinct inputs for %d stored chunks, want strictly fewer", h.embedRecorder.distinct(), reused+embedded)
	}
}

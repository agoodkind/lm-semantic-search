//go:build live

package live

import (
	"fmt"
	"strings"
	"testing"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

// contentField mirrors the collection's content column, so a test can count rows
// holding nothing without importing an unexported constant.
const contentField = "content"

// TestNoRowHoldsNothingEndToEnd is the whole point of this change, proven
// against a real Milvus rather than a fake.
//
// A third of the operator's conversation collection, 879,957 rows of 2,623,170,
// holds no content. Such a row consumed an embedding call, occupies a vector,
// and can never be returned by a search, because there is nothing in it to
// match. They exist because splitting an empty string yields one empty piece, so
// a turn carrying only a tool call or only reasoning stored exactly one row
// holding nothing.
//
// This ingests a conversation covering every shape that produced them and reads
// the rows straight back out of Milvus.
func TestNoRowHoldsNothingEndToEnd(t *testing.T) {
	harness := newHarness(t)

	conversationID := "blankrow-shapes"
	documents := map[string][]*pb.ConversationDocument{
		conversationID: blankRowShapes(conversationID),
	}

	job := harness.upsert(
		documents,
		pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN,
		true,
		false,
	)
	requireCompleted(t, job, "ingest")
	if job.Progress.ChunksEmbedded <= 0 {
		t.Fatalf(
			"nothing embedded, so the assertions below would pass vacuously\n%s",
			progressString(job),
		)
	}

	// The rule: no stored row holds nothing.
	if empty := harness.countRowsHoldingNothing(); empty != 0 {
		t.Fatalf(
			"%d stored rows hold nothing, want 0; this is the class the change exists to stop",
			empty,
		)
	}

	// The other half of the rule: content a person saw is still stored. Without
	// these the test above would pass by storing nothing at all.
	for prefix, description := range map[string]string{
		convBasePrefix(conversationID):  "the messages that carried text",
		convToolPrefix(conversationID):  "the tool call",
		convThinkPrefix(conversationID): "the reasoning",
	} {
		if count := harness.countRowsWithPrefix(prefix); count <= 0 {
			t.Fatalf("%s stored no rows under %q", description, prefix)
		}
	}

	// A turn carrying only a tool call, and one carrying only reasoning, each
	// store their derived rows and no base row. That absent base row is what
	// used to be a row holding nothing.
	toolOnlyBase := fmt.Sprintf("conv/%s/2", conversationID)
	if count := harness.countRowsWithPrefix(toolOnlyBase); count != 0 {
		t.Fatalf(
			"the tool-only turn stored %d base rows under %q, want 0",
			count,
			toolOnlyBase,
		)
	}
	reasoningOnlyBase := fmt.Sprintf("conv/%s/3", conversationID)
	if count := harness.countRowsWithPrefix(reasoningOnlyBase); count != 0 {
		t.Fatalf(
			"the reasoning-only turn stored %d base rows under %q, want 0",
			count,
			reasoningOnlyBase,
		)
	}
}

// TestSecondSyncHasNothingToDoEndToEnd is the convergence half, and it is what
// makes the change safe to deploy.
//
// A turn carrying only a tool call stores no base row, so the store holds only
// its derived rows. If reading those back did not register the message, the
// examination path would read it as new, re-send it, and remove those same rows
// as orphans, on every sync for as long as the conversation existed. That is
// worse than the blank rows it replaced.
//
// Delivering the identical conversation twice must therefore embed nothing the
// second time, and must leave the stored row count unchanged.
func TestSecondSyncHasNothingToDoEndToEnd(t *testing.T) {
	harness := newHarness(t)

	conversationID := "blankrow-converge"
	documents := map[string][]*pb.ConversationDocument{
		conversationID: blankRowShapes(conversationID),
	}

	first := harness.upsert(
		documents,
		pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN,
		true,
		false,
	)
	requireCompleted(t, first, "first sync")
	if first.Progress.ChunksEmbedded <= 0 {
		t.Fatalf("first sync embedded nothing\n%s", progressString(first))
	}
	rowsAfterFirst := harness.countRowsWithPrefix("conv")

	second := harness.upsert(
		documents,
		pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN,
		true,
		false,
	)
	requireCompleted(t, second, "second sync")

	if second.Progress.ChunksEmbedded != 0 {
		t.Fatalf(
			"second sync embedded %d chunks, want 0; an unchanged conversation is being re-embedded on every pass\n%s",
			second.Progress.ChunksEmbedded,
			progressString(second),
		)
	}
	if second.Progress.FilesEmbedded != 0 {
		t.Fatalf(
			"second sync re-embedded %d conversations, want 0\n%s",
			second.Progress.FilesEmbedded,
			progressString(second),
		)
	}

	rowsAfterSecond := harness.countRowsWithPrefix("conv")
	if rowsAfterSecond != rowsAfterFirst {
		t.Fatalf(
			"stored rows moved from %d to %d across an identical re-delivery, so rows are being removed and rewritten",
			rowsAfterFirst,
			rowsAfterSecond,
		)
	}
	if rowsAfterSecond == 0 {
		t.Fatal("no rows stored at all, so the comparison above proves nothing")
	}
}

// TestAppendedTurnStillEmbedsEndToEnd guards the reciprocal failure: a change
// that made every conversation look unchanged would pass both tests above while
// silently indexing nothing new.
func TestAppendedTurnStillEmbedsEndToEnd(t *testing.T) {
	harness := newHarness(t)

	conversationID := "blankrow-append"
	documents := map[string][]*pb.ConversationDocument{
		conversationID: blankRowShapes(conversationID),
	}

	first := harness.upsert(
		documents,
		pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN,
		true,
		false,
	)
	requireCompleted(t, first, "first sync")
	rowsAfterFirst := harness.countRowsWithPrefix("conv")

	documents[conversationID] = appendMessage(documents[conversationID], conversationID)
	appended := harness.upsert(
		documents,
		pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN,
		true,
		false,
	)
	requireCompleted(t, appended, "appended sync")

	if appended.Progress.ChunksEmbedded <= 0 {
		t.Fatalf(
			"an appended turn embedded nothing, so new content is not reaching the store\n%s",
			progressString(appended),
		)
	}
	if rowsAfter := harness.countRowsWithPrefix("conv"); rowsAfter <= rowsAfterFirst {
		t.Fatalf(
			"stored rows went from %d to %d after appending a turn, want more",
			rowsAfterFirst,
			rowsAfter,
		)
	}
	if empty := harness.countRowsHoldingNothing(); empty != 0 {
		t.Fatalf("%d rows hold nothing after the append, want 0", empty)
	}
}

// blankRowShapes is one conversation covering every message shape that used to
// produce a row holding nothing, plus the shapes that must keep storing rows.
//
// Index 0 and 1 carry text. Index 2 carries only a tool call. Index 3 carries
// only reasoning. Index 4 carries text that is only spacing alongside a tool
// call, which the sender is expected to normalize away but which proto3 cannot
// express as absence: an unset string and an empty one are identical bytes on
// the wire while a single space is content.
func blankRowShapes(id string) []*pb.ConversationDocument {
	return []*pb.ConversationDocument{
		{
			ConversationId: id,
			MessageIndex:   0,
			Role:           "user",
			TimestampUnix:  1712345000,
			Text:           "what does the indexer do with a turn that has no text",
		},
		{
			ConversationId: id,
			MessageIndex:   1,
			Role:           "assistant",
			TimestampUnix:  1712345001,
			Text:           "it used to store one row holding nothing for that turn",
			Thinking:       "the user is asking about the blank row class",
			Tools: []*pb.ConversationToolCall{
				{
						Name:     "run_shell",
						Display:  "ls -la /work/" + id,
						LangHint: "bash",
						Output:   "total 0\ndrwxr-xr-x  2 user  staff   64 " + id,
				},
			},
		},
		{
			ConversationId: id,
			MessageIndex:   2,
			Role:           "assistant",
			TimestampUnix:  1712345002,
			Text:           "",
			Tools: []*pb.ConversationToolCall{
				{
						Name:    "read_file",
						Display: "/work/" + id + "/main.go",
						Output:  "package main\n\nfunc main() {}\n",
				},
			},
		},
		{
			ConversationId: id,
			MessageIndex:   3,
			Role:           "assistant",
			TimestampUnix:  1712345003,
			Text:           "",
			Thinking:       "weighing whether to read another file before answering " + id,
		},
		{
			ConversationId: id,
			MessageIndex:   4,
			Role:           "assistant",
			TimestampUnix:  1712345004,
			Text:           "   \n  ",
			Tools: []*pb.ConversationToolCall{
				{
						Name:     "run_shell",
						Display:  "echo done",
						LangHint: "bash",
						Output:   "done\n",
				},
			},
		},
	}
}

// countRowsHoldingNothing counts stored rows whose content is empty, which is
// the class this change exists to stop. It reads the collection directly at
// strong consistency rather than trusting the job's own counters.
func (h *harness) countRowsHoldingNothing() int64 {
	h.t.Helper()
	expression := fmt.Sprintf(`%s == ""`, contentField)
	resultSet, err := h.milvus.Query(
		correlatedContext(),
		milvusclient.NewQueryOption(h.collectionName).
			WithFilter(expression).
			WithOutputFields(countOutputField).
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		h.t.Fatalf("Milvus empty-content count returned error: %v", err)
	}
	column := resultSet.GetColumn(countOutputField)
	if column == nil {
		h.t.Fatal("Milvus empty-content count returned no count column")
	}
	total, err := column.GetAsInt64(0)
	if err != nil {
		h.t.Fatalf("read empty-content count column returned error: %v", err)
	}
	return total
}

func (h *harness) countRowsContaining(content string) int64 {
	h.t.Helper()
	escapedContent := strings.ReplaceAll(content, "\\", "\\\\")
	escapedContent = strings.ReplaceAll(escapedContent, "\"", "\\\"")
	expression := fmt.Sprintf(`%s like "%%%s%%"`, contentField, escapedContent)
	resultSet, err := h.milvus.Query(
		correlatedContext(),
		milvusclient.NewQueryOption(h.collectionName).
			WithFilter(expression).
			WithOutputFields(countOutputField).
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		h.t.Fatalf("Milvus content count for %q returned error: %v", content, err)
	}
	column := resultSet.GetColumn(countOutputField)
	if column == nil {
		h.t.Fatalf("Milvus content count for %q returned no count column", content)
	}
	total, err := column.GetAsInt64(0)
	if err != nil {
		h.t.Fatalf("read content count column for %q returned error: %v", content, err)
	}
	return total
}

func (h *harness) contentForRelativePath(relativePath string) string {
	h.t.Helper()
	expression := fmt.Sprintf(`%s == %q`, relativePathField, relativePath)
	resultSet, err := h.milvus.Query(
		correlatedContext(),
		milvusclient.NewQueryOption(h.collectionName).
			WithFilter(expression).
			WithOutputFields(contentField).
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		h.t.Fatalf("Milvus content query for %q returned error: %v", relativePath, err)
	}
	column := resultSet.GetColumn(contentField)
	if column == nil {
		h.t.Fatalf("Milvus content query for %q returned no content column", relativePath)
	}
	content, err := column.GetAsString(0)
	if err != nil {
		h.t.Fatalf("read content column for %q returned error: %v", relativePath, err)
	}
	return content
}

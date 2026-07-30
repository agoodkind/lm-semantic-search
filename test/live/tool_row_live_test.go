//go:build live

package live

import (
	"strings"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

func TestToolCallsStoreOneUserVisibleRowEndToEnd(t *testing.T) {
	harness := newHarness(t)
	conversationID := "tool-row-display"
	documents := map[string][]*pb.ConversationDocument{
		conversationID: {
			{
				ConversationId: conversationID,
				MessageIndex:   0,
				Role:           "assistant",
				TimestampUnix:  1712345000,
				Tools: []*pb.ConversationToolCall{
					{Name: "Bash", Display: "ls -la /tmp", LangHint: "bash"},
					{Name: "Read", Display: "/tmp/readme.md"},
				},
			},
		},
	}

	job := harness.upsert(
		documents,
		pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN,
		true,
		false,
	)
	requireCompleted(t, job, "tool row ingest")
	if job.Progress.ChunksEmbedded <= 0 {
		t.Fatalf("tool row ingest embedded nothing\n%s", progressString(job))
	}

	for _, path := range []string{
		"convtool/" + conversationID + "/0/0",
		"convtool/" + conversationID + "/0/1",
	} {
		if count := harness.countRowsWithPrefix(path); count != 1 {
			t.Fatalf("rows under %q = %d, want 1", path, count)
		}
	}
	for _, content := range []string{"file_path", "\"command\"", "{\"", "input_text"} {
		if count := harness.countRowsContaining(content); count != 0 {
			t.Fatalf("rows containing %q = %d, want 0", content, count)
		}
	}
	bashContent := harness.contentForRelativePath("convtool/" + conversationID + "/0/0")
	if count := strings.Count(bashContent, "ls -la /tmp"); count != 1 {
		t.Fatalf("Bash display occurrences = %d, want 1 in %q", count, bashContent)
	}
	if count := harness.countRowsHoldingNothing(); count != 0 {
		t.Fatalf("rows holding nothing = %d, want 0", count)
	}
}

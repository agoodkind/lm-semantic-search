package semantic

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

func TestPartitionConversationEnrichmentFillsAndCountsOrphans(t *testing.T) {
	t.Parallel()
	ids := []string{"a", "b", "c"}
	chunks := []model.StoredChunk{
		{ConversationID: "claude:1"},
		{ConversationID: "codex:2"},
		{ConversationID: "claude:gone"},
	}
	vectors := [][]float32{{1}, {2}, {3}}
	enrichment := ConversationEnrichment{
		"claude:1":    {WorkspaceRoot: "/repo/one"},
		"codex:2":     {WorkspaceRoot: "/repo/two"},
		"claude:gone": {WorkspaceRoot: ""}, // present but empty counts as orphan
	}

	fillIDs, fillChunks, fillVectors, orphan := partitionConversationEnrichment(ids, chunks, vectors, enrichment)

	if orphan != 1 {
		t.Fatalf("orphan = %d, want 1", orphan)
	}
	if len(fillIDs) != 2 || len(fillChunks) != 2 || len(fillVectors) != 2 {
		t.Fatalf("fill lengths = %d/%d/%d, want 2/2/2", len(fillIDs), len(fillChunks), len(fillVectors))
	}
	if fillIDs[0] != "a" || fillIDs[1] != "b" {
		t.Fatalf("fill ids = %v, want [a b]", fillIDs)
	}
	if fillChunks[0].WorkspaceRoot != "/repo/one" || fillChunks[1].WorkspaceRoot != "/repo/two" {
		t.Fatalf("workspace roots = %q, %q, want /repo/one, /repo/two", fillChunks[0].WorkspaceRoot, fillChunks[1].WorkspaceRoot)
	}
}

func TestPartitionConversationEnrichmentMissingConversationIsOrphan(t *testing.T) {
	t.Parallel()
	ids := []string{"x"}
	chunks := []model.StoredChunk{{ConversationID: "claude:unknown"}}
	vectors := [][]float32{{1}}

	fillIDs, _, _, orphan := partitionConversationEnrichment(ids, chunks, vectors, ConversationEnrichment{})

	if orphan != 1 || len(fillIDs) != 0 {
		t.Fatalf("orphan=%d fill=%d, want 1/0", orphan, len(fillIDs))
	}
}

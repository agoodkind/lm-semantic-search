package semantic

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

func TestPartitionConversationEnrichmentFillsKnownAndOrphansUnknown(t *testing.T) {
	t.Parallel()
	ids := []string{"a", "b", "c", "d"}
	chunks := []model.StoredChunk{
		{ConversationID: "claude:1"},                           // empty workspace, filled from enrichment
		{ConversationID: "codex:2", WorkspaceRoot: "/already"}, // workspace already set, preserved
		{ConversationID: "claude:noworkspace"},                 // known but enrichment has no workspace
		{ConversationID: "claude:gone"},                        // unknown -> orphan
	}
	vectors := [][]float32{{1}, {2}, {3}, {4}}
	enrichment := ConversationEnrichment{
		"claude:1":           {WorkspaceRoot: "/repo/one", Archived: true},
		"codex:2":            {WorkspaceRoot: "/repo/two", Archived: true},
		"claude:noworkspace": {WorkspaceRoot: "", Archived: true},
	}

	fillIDs, fillChunks, fillVectors, orphan := partitionConversationEnrichment(ids, chunks, vectors, enrichment)

	if orphan != 1 {
		t.Fatalf("orphan = %d, want 1 (claude:gone is unknown)", orphan)
	}
	if len(fillIDs) != 3 || len(fillChunks) != 3 || len(fillVectors) != 3 {
		t.Fatalf("fill lengths = %d/%d/%d, want 3/3/3", len(fillIDs), len(fillChunks), len(fillVectors))
	}
	// Empty workspace is filled from the enrichment, and archived is set.
	if fillChunks[0].WorkspaceRoot != "/repo/one" || !fillChunks[0].Archived {
		t.Fatalf("fillChunks[0] = %+v, want workspace /repo/one archived true", fillChunks[0])
	}
	// A row that already carries a workspace keeps it; archived is still filled.
	if fillChunks[1].WorkspaceRoot != "/already" || !fillChunks[1].Archived {
		t.Fatalf("fillChunks[1] = %+v, want workspace /already preserved archived true", fillChunks[1])
	}
	// A known conversation with no workspace stays empty but still gets archived.
	if fillChunks[2].WorkspaceRoot != "" || !fillChunks[2].Archived {
		t.Fatalf("fillChunks[2] = %+v, want empty workspace archived true", fillChunks[2])
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

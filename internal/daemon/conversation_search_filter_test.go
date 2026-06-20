package daemon

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

func filterTestChunk(conversationID string, messageIndex int32, role string, timestampUnix int64, score float64) model.StoredChunk {
	return model.StoredChunk{
		Content:              "chunk",
		RelativePath:         "conv/" + conversationID + "/0",
		StartLine:            0,
		EndLine:              0,
		Language:             "",
		FileExtension:        "",
		ConversationID:       conversationID,
		ParentConversationID: "",
		MessageIndex:         messageIndex,
		Role:                 role,
		TimestampUnix:        timestampUnix,
		Score:                score,
	}
}

func emptyConversationSearchFilter() conversationSearchFilter {
	return conversationSearchFilter{
		Roles:                nil,
		FromUnix:             0,
		UntilUnix:            0,
		ConversationIDs:      nil,
		ParentConversationID: "",
		MinScore:             0,
		MessageIndexFrom:     0,
		MessageIndexUntil:    0,
	}
}

// TestConversationSearchFilterMatches proves each row condition: roles match
// case-insensitively, time and index bounds are inclusive-from and
// exclusive-until, the id set restricts membership, and min_score floors the
// retrieval score.
func TestConversationSearchFilterMatches(t *testing.T) {
	t.Parallel()

	chunk := filterTestChunk("conv-a", 5, "User", 1000, 0.8)

	if !emptyConversationSearchFilter().matches(chunk) {
		t.Fatal("empty filter rejected a chunk, want match-everything")
	}

	roleFilter := emptyConversationSearchFilter()
	roleFilter.Roles = []string{"user"}
	if !roleFilter.matches(chunk) {
		t.Fatal("role filter rejected case-insensitive match")
	}
	roleFilter.Roles = []string{"assistant"}
	if roleFilter.matches(chunk) {
		t.Fatal("role filter matched a non-member role")
	}

	timeFilter := emptyConversationSearchFilter()
	timeFilter.FromUnix = 1000
	if !timeFilter.matches(chunk) {
		t.Fatal("from bound rejected an equal timestamp, want inclusive")
	}
	timeFilter.UntilUnix = 1000
	if timeFilter.matches(chunk) {
		t.Fatal("until bound matched an equal timestamp, want exclusive")
	}

	idFilter := emptyConversationSearchFilter()
	idFilter.ConversationIDs = []string{"conv-b"}
	if idFilter.matches(chunk) {
		t.Fatal("id-set filter matched a conversation outside the set")
	}

	scoreFilter := emptyConversationSearchFilter()
	scoreFilter.MinScore = 0.9
	if scoreFilter.matches(chunk) {
		t.Fatal("min-score filter matched a chunk below the floor")
	}

	indexFilter := emptyConversationSearchFilter()
	indexFilter.MessageIndexFrom = 5
	indexFilter.MessageIndexUntil = 6
	if !indexFilter.matches(chunk) {
		t.Fatal("index range rejected an in-range message index")
	}
	indexFilter.MessageIndexUntil = 5
	if indexFilter.matches(chunk) {
		t.Fatal("index until bound matched an equal index, want exclusive")
	}
}

// TestConversationSearchFilterProviders proves the Providers branch matches a
// chunk by the provider prefix parsed from its conversation id, and rejects a
// chunk whose provider is outside the set or whose id carries no provider
// separator.
func TestConversationSearchFilterProviders(t *testing.T) {
	t.Parallel()

	claudeChunk := filterTestChunk("claude:thread-a", 0, "user", 1000, 0.8)
	codexChunk := filterTestChunk("codex:thread-b", 0, "user", 1000, 0.8)
	prefixlessChunk := filterTestChunk("thread-c", 0, "user", 1000, 0.8)

	providerFilter := emptyConversationSearchFilter()
	providerFilter.Providers = []string{"claude"}
	if !providerFilter.matchesScope(claudeChunk) {
		t.Fatal("provider filter rejected a claude: chunk for provider claude")
	}
	if providerFilter.matchesScope(codexChunk) {
		t.Fatal("provider filter matched a codex: chunk for provider claude")
	}
	if providerFilter.matchesScope(prefixlessChunk) {
		t.Fatal("provider filter matched a chunk whose id has no provider separator")
	}

	multiFilter := emptyConversationSearchFilter()
	multiFilter.Providers = []string{"claude", "codex"}
	if !multiFilter.matchesScope(claudeChunk) || !multiFilter.matchesScope(codexChunk) {
		t.Fatal("multi-provider filter rejected a member provider")
	}
}

// TestConversationSearchFilterWorkspaceRoots proves the WorkspaceRoots branch
// matches a chunk by its WorkspaceRoot value, which the semantic path now
// populates from the Milvus scalar column, and rejects a chunk whose workspace
// is outside the set or empty.
func TestConversationSearchFilterWorkspaceRoots(t *testing.T) {
	t.Parallel()

	inWorkspace := filterTestChunk("claude:thread-a", 0, "user", 1000, 0.8)
	inWorkspace.WorkspaceRoot = "/work/alpha"
	otherWorkspace := filterTestChunk("claude:thread-b", 0, "user", 1000, 0.8)
	otherWorkspace.WorkspaceRoot = "/work/beta"
	noWorkspace := filterTestChunk("claude:thread-c", 0, "user", 1000, 0.8)

	workspaceFilter := emptyConversationSearchFilter()
	workspaceFilter.WorkspaceRoots = []string{"/work/alpha"}
	if !workspaceFilter.matchesScope(inWorkspace) {
		t.Fatal("workspace filter rejected a chunk in the requested workspace")
	}
	if workspaceFilter.matchesScope(otherWorkspace) {
		t.Fatal("workspace filter matched a chunk in a different workspace")
	}
	if workspaceFilter.matchesScope(noWorkspace) {
		t.Fatal("workspace filter matched a chunk with an empty workspace root")
	}
}

// TestConversationSearchFilterArchived proves the Archived branch matches a
// chunk by its archived value when the filter is set, and ignores archived
// entirely when the filter is nil.
func TestConversationSearchFilterArchived(t *testing.T) {
	t.Parallel()

	archivedChunk := filterTestChunk("claude:thread-a", 0, "user", 1000, 0.8)
	archivedChunk.Archived = true
	liveChunk := filterTestChunk("claude:thread-b", 0, "user", 1000, 0.8)
	liveChunk.Archived = false

	nilFilter := emptyConversationSearchFilter()
	if !nilFilter.matchesScope(archivedChunk) || !nilFilter.matchesScope(liveChunk) {
		t.Fatal("nil archived filter rejected a chunk, want match-everything")
	}

	wantLive := false
	liveFilter := emptyConversationSearchFilter()
	liveFilter.Archived = &wantLive
	if !liveFilter.matchesScope(liveChunk) {
		t.Fatal("archived=false filter rejected a non-archived chunk")
	}
	if liveFilter.matchesScope(archivedChunk) {
		t.Fatal("archived=false filter matched an archived chunk")
	}

	wantArchived := true
	archivedFilter := emptyConversationSearchFilter()
	archivedFilter.Archived = &wantArchived
	if !archivedFilter.matchesScope(archivedChunk) {
		t.Fatal("archived=true filter rejected an archived chunk")
	}
	if archivedFilter.matchesScope(liveChunk) {
		t.Fatal("archived=true filter matched a non-archived chunk")
	}
}

// TestApplyConversationSearchFilterCapsPerConversation proves the
// per-conversation cap keeps each conversation's earliest (highest-ranked)
// hits and the overall limit truncates the final list.
func TestApplyConversationSearchFilterCapsPerConversation(t *testing.T) {
	t.Parallel()

	chunks := []model.StoredChunk{
		filterTestChunk("conv-a", 0, "user", 1000, 0.9),
		filterTestChunk("conv-a", 1, "user", 1001, 0.8),
		filterTestChunk("conv-a", 2, "user", 1002, 0.7),
		filterTestChunk("conv-b", 0, "user", 1003, 0.6),
		filterTestChunk("conv-c", 0, "user", 1004, 0.5),
	}

	capped := applyConversationSearchFilter(chunks, emptyConversationSearchFilter(), 1, 0)
	if len(capped) != 3 {
		t.Fatalf("per-conversation cap kept %d chunks, want 3 (one per conversation)", len(capped))
	}
	if capped[0].ConversationID != "conv-a" || capped[0].MessageIndex != 0 {
		t.Fatalf("cap kept %+v first, want conv-a's top-ranked hit", capped[0])
	}

	limited := applyConversationSearchFilter(chunks, emptyConversationSearchFilter(), 0, 2)
	if len(limited) != 2 {
		t.Fatalf("limit kept %d chunks, want 2", len(limited))
	}
}

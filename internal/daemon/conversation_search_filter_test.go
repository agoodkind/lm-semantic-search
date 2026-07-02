package daemon

import "testing"

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

func TestConversationSearchFilterMapsToSemanticFilter(t *testing.T) {
	t.Parallel()

	wantArchived := true
	filter := conversationSearchFilter{
		Providers:            []string{"claude"},
		WorkspaceRoots:       []string{"/work/repo"},
		Roles:                []string{"user"},
		FromUnix:             1000,
		UntilUnix:            2000,
		ConversationIDs:      []string{"conv-a"},
		ParentConversationID: "parent-a",
		MinScore:             0.8,
		MessageIndexFrom:     3,
		MessageIndexUntil:    9,
		Archived:             &wantArchived,
	}

	got := filter.toSemanticFilter()
	if got.Providers[0] != "claude" {
		t.Fatalf("Providers = %v, want [claude]", got.Providers)
	}
	if got.WorkspaceRoots[0] != "/work/repo" {
		t.Fatalf("WorkspaceRoots = %v, want [/work/repo]", got.WorkspaceRoots)
	}
	if got.Roles[0] != "user" {
		t.Fatalf("Roles = %v, want [user]", got.Roles)
	}
	if got.ConversationIDs[0] != "conv-a" {
		t.Fatalf("ConversationIDs = %v, want [conv-a]", got.ConversationIDs)
	}
	if got.ParentConversationID != "parent-a" {
		t.Fatalf("ParentConversationID = %q, want parent-a", got.ParentConversationID)
	}
	if got.FromUnix != 1000 {
		t.Fatalf("FromUnix = %d, want 1000", got.FromUnix)
	}
	if got.UntilUnix != 2000 {
		t.Fatalf("UntilUnix = %d, want 2000", got.UntilUnix)
	}
	if got.MessageIndexFrom != 3 {
		t.Fatalf("MessageIndexFrom = %d, want 3", got.MessageIndexFrom)
	}
	if got.MessageIndexUntil != 9 {
		t.Fatalf("MessageIndexUntil = %d, want 9", got.MessageIndexUntil)
	}
	if got.Archived == nil || *got.Archived != wantArchived {
		t.Fatalf("Archived = %v, want true pointer", got.Archived)
	}
}

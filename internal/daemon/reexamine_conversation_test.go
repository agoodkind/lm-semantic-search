package daemon

import (
	"slices"
	"testing"

	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
)

// TestConversationItemSourceForcedItemsReexamine proves a reexamine-flagged
// conversation source forces every delivered conversation, and an unflagged one
// forces nothing, so the normal sync stays byte-for-byte unchanged.
func TestConversationItemSourceForcedItemsReexamine(t *testing.T) {
	docs := []model.ConversationDocument{
		{ConversationID: "claude:a", MessageIndex: 0, Role: "user", Text: "hi"},
		{ConversationID: "claude:a", MessageIndex: 1, Role: "assistant", Text: "yo"},
		{ConversationID: "codex:b", MessageIndex: 0, Role: "user", Text: "sup"},
	}
	manifest := map[string]string{"claude:a": "fp1", "codex:b": "fp2"}

	forced := newConversationItemSource("coll", manifest, docs, nil, absenceRetain, true)
	got := forced.forcedItems()
	slices.Sort(got)
	if !slices.Equal(got, []string{"claude:a", "codex:b"}) {
		t.Fatalf("forcedItems with reexamine = %v, want [claude:a codex:b]", got)
	}

	quiet := newConversationItemSource("coll", manifest, docs, nil, absenceRetain, false)
	if got := quiet.forcedItems(); got != nil {
		t.Fatalf("forcedItems without reexamine = %v, want nil", got)
	}
}

// TestCodeItemSourceForcedItemsNil proves a code source never forces an item, so
// unionForcedItems is a no-op for filesystem syncs.
func TestCodeItemSourceForcedItemsNil(t *testing.T) {
	if got := (codeItemSource{}).forcedItems(); got != nil {
		t.Fatalf("code forcedItems = %v, want nil", got)
	}
}

// TestUnionForcedItemsAddsUnchangedPresentItem proves a forced id that the
// merkle diff classified as unchanged is added to Modified, an already-changed
// id is not duplicated, and an id absent from the current capture is ignored.
func TestUnionForcedItemsAddsUnchangedPresentItem(t *testing.T) {
	captured := merkle.Snapshot{ConfigDigest: "", Files: map[string]string{"a": "1", "b": "2", "c": "3"}, Inodes: nil}
	diff := merkle.Diff{Added: []string{"a"}, Modified: []string{"b"}, Removed: nil}

	got := unionForcedItems(diff, []string{"b", "c", "z"}, captured)

	if !slices.Equal(got.Modified, []string{"b", "c"}) {
		t.Fatalf("Modified = %v, want [b c] (c forced, b not duplicated, z ignored)", got.Modified)
	}
	if !slices.Equal(got.Added, []string{"a"}) {
		t.Fatalf("Added = %v, want [a] unchanged", got.Added)
	}
}

// TestUnionForcedItemsNilIsNoOp proves the normal sync path, which forces
// nothing, leaves the diff untouched.
func TestUnionForcedItemsNilIsNoOp(t *testing.T) {
	captured := merkle.Snapshot{ConfigDigest: "", Files: map[string]string{"a": "1"}, Inodes: nil}
	diff := merkle.Diff{Added: nil, Modified: []string{"m"}, Removed: []string{"r"}}

	got := unionForcedItems(diff, nil, captured)

	if !slices.Equal(got.Modified, []string{"m"}) || !slices.Equal(got.Removed, []string{"r"}) {
		t.Fatalf("nil forced changed the diff: %+v", got)
	}
}

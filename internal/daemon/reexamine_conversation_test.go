package daemon

import (
	"context"
	"errors"
	"slices"
	"testing"

	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// TestConversationItemSourceForcedWorkSetBackfill proves the cheap store-presence
// classifier: a backfill forces only the delivered conversation whose expected
// derived rows are missing, prunes a conversation whose derived rows are all
// present, prunes a conversation that expects no derived rows at all, and forces
// nothing when neither backfill nor force is set.
func TestConversationItemSourceForcedWorkSetBackfill(t *testing.T) {
	docs := []model.ConversationDocument{
		{ConversationID: "claude:a", MessageIndex: 0, Role: "assistant", Text: "a", Thinking: "reasoning-a"},
		{ConversationID: "codex:b", MessageIndex: 0, Role: "assistant", Text: "b", Thinking: "reasoning-b"},
		{ConversationID: "plain:c", MessageIndex: 0, Role: "user", Text: "c"},
	}
	manifest := map[string]string{"claude:a": "fp1", "codex:b": "fp2", "plain:c": "fp3"}
	batch := semantic.ConversationBatchState{
		Rows: map[string]semantic.ConversationStoredRows{
			"claude:a": {Messages: map[int32]semantic.StoredMessageState{}, DerivedPaths: map[string]string{"convthink/claude:a/0": "h"}},
			"codex:b":  {Messages: map[int32]semantic.StoredMessageState{}, DerivedPaths: map[string]string{}},
			"plain:c":  {Messages: map[int32]semantic.StoredMessageState{}, DerivedPaths: map[string]string{}},
		},
		Reuse: map[string][]float32{},
	}
	reader := &testConversationRowReader{batch: &batch}

	forced := newConversationItemSource("coll", manifest, docs, reader, absenceRetain, true, false)
	got, err := forced.forcedWorkSet(context.Background())
	if err != nil {
		t.Fatalf("forcedWorkSet returned error: %v", err)
	}
	if !slices.Equal(got, []string{"codex:b"}) {
		t.Fatalf("forcedWorkSet = %v, want [codex:b] (only the missing-derived conversation)", got)
	}

	quiet := newConversationItemSource("coll", manifest, docs, reader, absenceRetain, false, false)
	got, err = quiet.forcedWorkSet(context.Background())
	if err != nil {
		t.Fatalf("forcedWorkSet with neither flag returned error: %v", err)
	}
	if got != nil {
		t.Fatalf("forcedWorkSet with neither flag = %v, want nil", got)
	}
}

// TestConversationItemSourceForceForcesAllAndDisablesReuse proves the force path:
// force returns every delivered id from forcedWorkSet even for a fully-present
// conversation (no presence prune), reuseSource reports the no-reuse scope so
// present chunks re-embed, and indexOne regenerates the whole conversation's
// chunks with no reuse map. It contrasts with backfill, which prunes the same
// fully-present conversation.
func TestConversationItemSourceForceForcesAllAndDisablesReuse(t *testing.T) {
	t.Parallel()

	conversationID := "claude:present"
	documents := []model.ConversationDocument{{
		ConversationID: conversationID,
		MessageIndex:   0,
		Role:           "assistant",
		Text:           "answer",
		Thinking:       "private reasoning",
	}}
	manifest := map[string]string{conversationID: "fp-present"}
	// Every expected derived row is already present, so a backfill would prune it.
	batch := conversationBatchStateForDocuments(t, map[string][]model.ConversationDocument{
		conversationID: documents,
	})
	reader := &testConversationRowReader{batch: &batch}

	// Backfill prunes the fully-present conversation.
	backfillSource := newConversationItemSource("conv_chunks_live", manifest, documents, reader, absenceRetain, true, false)
	backfillForced, err := backfillSource.forcedWorkSet(context.Background())
	if err != nil {
		t.Fatalf("backfill forcedWorkSet returned error: %v", err)
	}
	if len(backfillForced) != 0 {
		t.Fatalf("backfill forcedWorkSet = %v, want empty (present conversation pruned)", backfillForced)
	}

	// Force returns the delivered id despite full presence, and disables reuse.
	forceSource := newConversationItemSource("conv_chunks_live", manifest, documents, reader, absenceRetain, false, true)
	forceForced, err := forceSource.forcedWorkSet(context.Background())
	if err != nil {
		t.Fatalf("force forcedWorkSet returned error: %v", err)
	}
	if !slices.Equal(forceForced, []string{conversationID}) {
		t.Fatalf("force forcedWorkSet = %v, want [%s] (force forces all delivered, no prune)", forceForced, conversationID)
	}
	if scope := forceSource.reuseSource(conversationID).Scope; scope != itemReuseScopeNone {
		t.Fatalf("force reuseSource scope = %q, want none (reuse disabled under force)", scope)
	}
	// indexOne under force regenerates the whole conversation and carries no reuse
	// map, so every present chunk re-embeds.
	result, err := forceSource.indexOne(context.Background(), conversationID)
	if err != nil {
		t.Fatalf("force indexOne returned error: %v", err)
	}
	if len(result.Chunks) == 0 {
		t.Fatalf("force indexOne produced no chunks, want the full present conversation regenerated")
	}
	if result.ReuseVectors != nil {
		t.Fatalf("force indexOne ReuseVectors = %v, want nil (reuse disabled so chunks re-embed)", result.ReuseVectors)
	}
}

// TestForcedWorkSetPrunesFullyPresentConversation proves the cheap up-front
// classification: a delivered conversation whose expected derived rows are all
// present in the store is pruned from the forced work set after exactly one
// batch read, with no per-item regeneration. No marker is involved; the
// classifier judges from store presence directly.
func TestForcedWorkSetPrunesFullyPresentConversation(t *testing.T) {
	t.Parallel()

	reader := &testConversationRowReader{
		derivedPaths: map[string]string{"convthink/claude:current/0": "hash"},
	}
	documents := []model.ConversationDocument{{
		ConversationID: "claude:current",
		MessageIndex:   0,
		Role:           "assistant",
		Text:           "unchanged",
		Thinking:       "private reasoning",
	}}
	source := newConversationItemSource(
		"conversation_collection",
		map[string]string{"claude:current": "fp-current"},
		documents,
		reader,
		absenceRetain,
		true,
		false,
	)
	captured, err := source.capture(context.Background())
	if err != nil {
		t.Fatalf("capture returned error: %v", err)
	}
	forced, forcedErr := source.forcedWorkSet(context.Background())
	if forcedErr != nil {
		t.Fatalf("forcedWorkSet returned error: %v", forcedErr)
	}
	diff := unionForcedItems(merkle.Diff{}, forced, captured)
	if !diff.Empty() {
		t.Fatalf("fully present conversation produced changed diff: %+v", diff)
	}
	if calls := reader.callsSnapshot(); len(calls) != 1 {
		t.Fatalf("derived batch loads = %v, want exactly one cheap read", calls)
	}
}

// TestConversationForcedWorkSetFailsSafeOnBatchError proves a store-read failure
// forces every delivered id rather than under-embedding.
func TestConversationForcedWorkSetFailsSafeOnBatchError(t *testing.T) {
	docs := []model.ConversationDocument{
		{ConversationID: "claude:a", MessageIndex: 0, Role: "assistant", Text: "a", Thinking: "ra"},
		{ConversationID: "codex:b", MessageIndex: 0, Role: "assistant", Text: "b", Thinking: "rb"},
	}
	manifest := map[string]string{"claude:a": "fp1", "codex:b": "fp2"}
	reader := &testConversationRowReader{err: errors.New("batch read failed")}

	source := newConversationItemSource("coll", manifest, docs, reader, absenceRetain, true, false)
	got, err := source.forcedWorkSet(context.Background())
	if err != nil {
		t.Fatalf("forcedWorkSet fail-safe returned error: %v", err)
	}
	if !slices.Equal(got, []string{"claude:a", "codex:b"}) {
		t.Fatalf("forcedWorkSet on batch error = %v, want all delivered ids", got)
	}
}

// TestCodeItemSourceForcedWorkSetNil proves a code source never forces an item, so
// unionForcedItems is a no-op for filesystem syncs.
func TestCodeItemSourceForcedWorkSetNil(t *testing.T) {
	got, err := (codeItemSource{}).forcedWorkSet(context.Background())
	if err != nil || got != nil {
		t.Fatalf("code forcedWorkSet = (%v, %v), want (nil, nil)", got, err)
	}
}

// TestItemSourceColumnSet proves each source names its own store column family,
// so the store write is told the row shape rather than inferring it.
func TestItemSourceColumnSet(t *testing.T) {
	if got := (codeItemSource{}).columnSet(); got != semantic.StoreColumnSetCode {
		t.Fatalf("code columnSet = %v, want StoreColumnSetCode", got)
	}
	if got := (conversationItemSource{}).columnSet(); got != semantic.StoreColumnSetConversation {
		t.Fatalf("conversation columnSet = %v, want StoreColumnSetConversation", got)
	}
}

// TestItemSourceCapabilities proves each source reports the spine capabilities
// that replaced the codebase.Kind branches: a code source produces a code graph
// and tracks whole-codebase byte totals, while a conversation source does
// neither.
func TestItemSourceCapabilities(t *testing.T) {
	if !(codeItemSource{}).producesGraph() {
		t.Fatal("code producesGraph = false, want true")
	}
	if !(codeItemSource{}).tracksByteTotals() {
		t.Fatal("code tracksByteTotals = false, want true")
	}
	if (conversationItemSource{}).producesGraph() {
		t.Fatal("conversation producesGraph = true, want false")
	}
	if (conversationItemSource{}).tracksByteTotals() {
		t.Fatal("conversation tracksByteTotals = true, want false")
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

package daemon

import (
	"context"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// TestConversationIndexOneInsertsMissingTargetWithSharedVector proves the
// identity fix: a derived target row that is absent for this conversation is
// re-emitted even when an identical content vector is present elsewhere in the
// batch, and the reused vector still reaches the reindex so the row is inserted
// without a fresh embed.
func TestConversationIndexOneInsertsMissingTargetWithSharedVector(t *testing.T) {
	t.Parallel()

	conversationID := "conv-b"
	sharedThinking := "shared private reasoning"
	document := model.ConversationDocument{
		ConversationID: conversationID,
		MessageIndex:   0,
		Role:           "assistant",
		Text:           "answer b",
		Thinking:       sharedThinking,
	}
	// The base message matches, but the derived target row (convthink/conv-b/0) is
	// absent for this conversation. The shared thinking vector is present in the
	// batch reuse from a different conversation.
	batch := semantic.ConversationBatchState{
		Rows: map[string]semantic.ConversationStoredRows{
			conversationID: {
				Messages:     map[int32]semantic.StoredMessageState{0: {Role: "assistant", Text: "answer b"}},
				DerivedPaths: map[string]string{},
			},
		},
		Reuse: map[string][]float32{
			semantic.ContentVectorKey("answer b"):     {8},
			semantic.ContentVectorKey(sharedThinking): {9},
		},
	}
	reader := &testConversationRowReader{batch: &batch}
	source := newConversationItemSource(
		"conv_chunks_live",
		map[string]string{conversationID: "fp-b"},
		[]model.ConversationDocument{document},
		reader,
		absenceRetain,
		false,
	)

	result, err := source.indexOne(context.Background(), conversationID)
	if err != nil {
		t.Fatalf("indexOne returned error: %v", err)
	}

	thinkingPath := conversationThinkingPath(conversationID, 0)
	foundTarget := false
	for _, chunk := range result.Chunks {
		if chunk.RelativePath == thinkingPath && chunk.Content == sharedThinking {
			foundTarget = true
		}
	}
	if !foundTarget {
		t.Fatalf("missing target row %q not re-emitted: %+v", thinkingPath, result.Chunks)
	}
	if result.ReuseVectors[semantic.ContentVectorKey(sharedThinking)] == nil {
		t.Fatalf("shared thinking vector not offered for reuse: %+v", result.ReuseVectors)
	}
}

// TestConversationIndexOnePresentDerivedRowsSkip proves a conversation whose
// stored rows exactly match the expected identities re-emits nothing and removes
// nothing, so a second pass over unchanged content thrashes zero rows.
func TestConversationIndexOnePresentDerivedRowsSkip(t *testing.T) {
	t.Parallel()

	conversationID := "conv-present"
	document := model.ConversationDocument{
		ConversationID: conversationID,
		MessageIndex:   1,
		Role:           "assistant",
		Text:           "answer",
		Thinking:       "private reasoning",
	}
	batch := conversationBatchStateForDocuments(t, map[string][]model.ConversationDocument{
		conversationID: {document},
	})
	reader := &testConversationRowReader{batch: &batch}
	source := newConversationItemSource(
		"conv_chunks_live",
		map[string]string{conversationID: "fp-present"},
		[]model.ConversationDocument{document},
		reader,
		absenceRetain,
		false,
	)

	result, err := source.indexOne(context.Background(), conversationID)
	if err != nil {
		t.Fatalf("indexOne returned error: %v", err)
	}
	if len(result.Chunks) != 0 {
		t.Fatalf("Chunks = %+v, want none for a fully present conversation", result.Chunks)
	}
	if len(result.RemovalPaths) != 0 || len(result.RemovalPrefixes) != 0 {
		t.Fatalf("removals = %v / %v, want none for a fully present conversation", result.RemovalPaths, result.RemovalPrefixes)
	}
}

// TestConversationIndexOneReconcilesDerivedPathShapeChange proves an obsolete
// derived row deletes on a path-shape change: a message whose thinking shrank
// from multipart to single re-emits and removes the message's tool and thinking
// prefixes, so the stale multipart rows are dropped.
func TestConversationIndexOneReconcilesDerivedPathShapeChange(t *testing.T) {
	t.Parallel()

	conversationID := "conv-shape-derived"
	document := model.ConversationDocument{
		ConversationID: conversationID,
		MessageIndex:   2,
		Role:           "assistant",
		Text:           "answer",
		Thinking:       "short reasoning",
	}
	thinkingPath := conversationThinkingPath(conversationID, 2)
	// The stored thinking was multipart; the new thinking is a single row, so the
	// stored derived path set differs from the expected set.
	batch := semantic.ConversationBatchState{
		Rows: map[string]semantic.ConversationStoredRows{
			conversationID: {
				Messages: map[int32]semantic.StoredMessageState{2: {Role: "assistant", Text: "answer"}},
				DerivedPaths: map[string]string{
					thinkingPath + "/0": semantic.ContentVectorKey("stale part zero"),
					thinkingPath + "/1": semantic.ContentVectorKey("stale part one"),
				},
			},
		},
		Reuse: map[string][]float32{},
	}
	reader := &testConversationRowReader{batch: &batch}
	source := newConversationItemSource(
		"conv_chunks_live",
		map[string]string{conversationID: "fp-shape"},
		[]model.ConversationDocument{document},
		reader,
		absenceRetain,
		false,
	)

	result, err := source.indexOne(context.Background(), conversationID)
	if err != nil {
		t.Fatalf("indexOne returned error: %v", err)
	}
	assertStringSliceEqual(t, result.RemovalPaths, conversationRemovalPathsForTest(conversationID, 2))
	assertStringSliceEqual(t, result.RemovalPrefixes, conversationRemovalPrefixesForTest(conversationID, 2))
	foundThinking := false
	for _, chunk := range result.Chunks {
		if chunk.RelativePath == thinkingPath && chunk.Content == "short reasoning" {
			foundThinking = true
		}
	}
	if !foundThinking {
		t.Fatalf("reshaped thinking row %q not re-emitted: %+v", thinkingPath, result.Chunks)
	}
}

// TestStampFullyEmbeddedConversations proves the bootstrap stamps only a
// conversation whose expected rows are all present, never a partially embedded
// one, and never embeds.
func TestStampFullyEmbeddedConversations(t *testing.T) {
	t.Parallel()

	fullID := "conv-full"
	partialID := "conv-partial"
	fullDocuments := []model.ConversationDocument{
		{ConversationID: fullID, MessageIndex: 0, Role: "user", Text: "hi"},
		{ConversationID: fullID, MessageIndex: 1, Role: "assistant", Text: "answer", Thinking: "reasoning"},
	}
	partialDocuments := []model.ConversationDocument{
		{ConversationID: partialID, MessageIndex: 0, Role: "user", Text: "hi"},
		{ConversationID: partialID, MessageIndex: 1, Role: "assistant", Text: "answer", Thinking: "reasoning"},
	}

	batch := conversationBatchStateForDocuments(t, map[string][]model.ConversationDocument{
		fullID:    fullDocuments,
		partialID: partialDocuments,
	})
	// Break the partial conversation: drop its derived rows so message 1 is missing
	// its thinking row and the conversation is not fully embedded.
	partialRows := batch.Rows[partialID]
	partialRows.DerivedPaths = map[string]string{}
	batch.Rows[partialID] = partialRows

	manager, _, _ := newTestManager(t)
	fake := &fakeSemantic{
		loadDerivedBatch: func(_ context.Context, _ string, _ []string) (semantic.ConversationBatchState, error) {
			return batch, nil
		},
	}
	manager.semantic = fake
	ctx := context.Background()
	collectionID := "thread-bootstrap"
	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}

	documents := append(append([]model.ConversationDocument{}, fullDocuments...), partialDocuments...)
	examined, stamped, err := manager.StampFullyEmbeddedConversations(ctx, collectionID, documents)
	if err != nil {
		t.Fatalf("StampFullyEmbeddedConversations returned error: %v", err)
	}
	if examined != 2 {
		t.Fatalf("examined = %d, want 2", examined)
	}
	if stamped != 1 {
		t.Fatalf("stamped = %d, want 1 (only the fully embedded conversation)", stamped)
	}

	markers := loadConversationDerivedMarkers(conversationDerivedMarkerPath(manager.merklePath(codebase.ID)))
	if markers[fullID] != derivedPipelineVersion {
		t.Fatalf("fully embedded marker = %q, want %q", markers[fullID], derivedPipelineVersion)
	}
	if _, found := markers[partialID]; found {
		t.Fatalf("partial conversation stamped %q, want it left unstamped", markers[partialID])
	}
	if calls := fake.reindexCallsSnapshot(); len(calls) != 0 {
		t.Fatalf("bootstrap issued %d reindex calls, want 0 (never re-embeds)", len(calls))
	}
}

// TestStampFullyEmbeddedConversationsFreshInstallStampsNothing proves the
// migration guard is safe on a fresh install: an empty batched read means no
// conversation is fully embedded, so nothing is stamped.
func TestStampFullyEmbeddedConversationsFreshInstallStampsNothing(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	fake := &fakeSemantic{
		loadDerivedBatch: func(_ context.Context, _ string, _ []string) (semantic.ConversationBatchState, error) {
			return semantic.ConversationBatchState{Rows: map[string]semantic.ConversationStoredRows{}, Reuse: map[string][]float32{}}, nil
		},
	}
	manager.semantic = fake
	ctx := context.Background()
	collectionID := "thread-bootstrap-fresh"
	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}

	documents := []model.ConversationDocument{
		{ConversationID: "conv-fresh", MessageIndex: 0, Role: "user", Text: "hi"},
	}
	examined, stamped, err := manager.StampFullyEmbeddedConversations(ctx, collectionID, documents)
	if err != nil {
		t.Fatalf("StampFullyEmbeddedConversations returned error: %v", err)
	}
	if examined != 1 || stamped != 0 {
		t.Fatalf("examined = %d, stamped = %d, want 1 examined and 0 stamped on a fresh install", examined, stamped)
	}
	markers := loadConversationDerivedMarkers(conversationDerivedMarkerPath(manager.merklePath(codebase.ID)))
	if len(markers) != 0 {
		t.Fatalf("markers = %v, want none on a fresh install", markers)
	}
}

// TestConversationReexamineNeedsBootstrap proves the auto-trigger fires only for
// a reexamine backfill whose marker store is empty.
func TestConversationReexamineNeedsBootstrap(t *testing.T) {
	t.Parallel()

	snapshotPath := filepath.Join(t.TempDir(), "conversation.json")
	if !conversationReexamineNeedsBootstrap(true, snapshotPath) {
		t.Fatal("reexamine with empty markers should trigger bootstrap")
	}
	if conversationReexamineNeedsBootstrap(false, snapshotPath) {
		t.Fatal("a non-reexamine upsert should never trigger bootstrap")
	}
	markerPath := conversationDerivedMarkerPath(snapshotPath)
	if err := writeConversationDerivedMarkers(markerPath, map[string]string{"conv-x": derivedPipelineVersion}); err != nil {
		t.Fatalf("writeConversationDerivedMarkers returned error: %v", err)
	}
	if conversationReexamineNeedsBootstrap(true, snapshotPath) {
		t.Fatal("reexamine with existing markers should not re-trigger bootstrap")
	}
}

// TestReexamineBackfillAutoStampsFullyEmbedded proves the migration path: a
// reexamine upsert over a corpus with no markers stamps the already fully
// embedded conversation before planning, so it is skipped rather than
// re-examined every run, and it is never re-embedded.
func TestReexamineBackfillAutoStampsFullyEmbedded(t *testing.T) {
	t.Parallel()

	conversationID := "conv-done"
	documents := []model.ConversationDocument{
		{ConversationID: conversationID, MessageIndex: 0, Role: "user", Text: "hi"},
		{ConversationID: conversationID, MessageIndex: 1, Role: "assistant", Text: "answer", Thinking: "reasoning"},
	}
	batch := conversationBatchStateForDocuments(t, map[string][]model.ConversationDocument{
		conversationID: documents,
	})

	manager, _, _ := newTestManager(t)
	fake := &fakeSemantic{
		loadDerivedBatch: func(_ context.Context, _ string, _ []string) (semantic.ConversationBatchState, error) {
			return batch, nil
		},
	}
	manager.semantic = fake
	ctx := context.Background()
	collectionID := "thread-reexamine-migration"
	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	if err := merkle.WriteSnapshot(manager.merklePath(codebase.ID), merkle.Snapshot{
		ConfigDigest: codebase.EffectiveConfig.IgnoreDigest,
		Files:        map[string]string{conversationID: "fp-done"},
		Inodes:       nil,
	}); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	job, err := manager.upsertConversationDocuments(ctx, collectionID, documents, map[string]string{conversationID: "fp-done"}, testClientInfo(), absenceRetain, true)
	if err != nil {
		t.Fatalf("upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, job.ID, model.JobStateCompleted)

	markers := loadConversationDerivedMarkers(conversationDerivedMarkerPath(manager.merklePath(codebase.ID)))
	if markers[conversationID] != derivedPipelineVersion {
		t.Fatalf("marker = %q, want %q stamped by the auto-bootstrap", markers[conversationID], derivedPipelineVersion)
	}
	if calls := fake.reindexCallsSnapshot(); len(calls) != 0 {
		t.Fatalf("reindex calls = %d, want 0 (fully embedded conversation skipped)", len(calls))
	}
}

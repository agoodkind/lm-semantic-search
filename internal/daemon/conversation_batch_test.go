package daemon

import (
	"context"
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

// TestReexamineBackfillFullyPresentEmbedsZero proves the store-presence no-op:
// a reexamine upsert over a corpus whose delivered conversation already has all
// its expected derived rows present classifies it as done via forcedWorkSet and
// embeds nothing, with no marker involved. The unchanged checkpoint keeps the
// merkle diff empty, and the presence classifier prunes the delivered id, so the
// per-item loop never runs and no reindex is issued.
func TestReexamineBackfillFullyPresentEmbedsZero(t *testing.T) {
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
	collectionID := "thread-reexamine-present"
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

	job, err := manager.upsertConversationDocuments(ctx, collectionID, documents, map[string]string{conversationID: "fp-done"}, testClientInfo(), absenceRetain, true, false)
	if err != nil {
		t.Fatalf("upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, job.ID, model.JobStateCompleted)

	if calls := fake.reindexCallsSnapshot(); len(calls) != 0 {
		t.Fatalf("reindex calls = %d, want 0 (fully present conversation pruned by store presence)", len(calls))
	}
}

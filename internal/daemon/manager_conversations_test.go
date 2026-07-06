package daemon

import (
	"context"
	"errors"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/store"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestRegisterConversationCollectionIsIdempotent(t *testing.T) {
	t.Parallel()

	manager, cfg, _ := newTestManager(t)

	first, err := manager.RegisterConversationCollection(context.Background(), "thread-alpha")
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	second, err := manager.RegisterConversationCollection(context.Background(), "thread-alpha")
	if err != nil {
		t.Fatalf("second RegisterConversationCollection returned error: %v", err)
	}

	if first.ID != second.ID {
		t.Fatalf("repeat registration ID = %q, want %q", second.ID, first.ID)
	}
	if first.Kind != model.CodebaseKindDocument {
		t.Fatalf("Kind = %q, want %q", first.Kind, model.CodebaseKindDocument)
	}
	if first.CanonicalPath != "chat:///thread-alpha" {
		t.Fatalf("CanonicalPath = %q, want chat:///thread-alpha", first.CanonicalPath)
	}
	if !strings.HasPrefix(first.CollectionName, "conv_chunks_") {
		t.Fatalf("CollectionName = %q, want conv_chunks_ prefix", first.CollectionName)
	}

	indexes := manager.ListIndexes(context.Background())
	if len(indexes) != 1 {
		t.Fatalf("ListIndexes returned %d codebases, want 1", len(indexes))
	}
	registry, err := store.ReadRegistry(cfg.RegistryPath)
	if err != nil {
		t.Fatalf("ReadRegistry returned error: %v", err)
	}
	if len(registry.Codebases) != 1 {
		t.Fatalf("registry contains %d codebases, want 1", len(registry.Codebases))
	}
	if registry.Codebases[0].Kind != model.CodebaseKindDocument {
		t.Fatalf("persisted Kind = %q, want %q", registry.Codebases[0].Kind, model.CodebaseKindDocument)
	}
}

// TestConversationManifestSyncReturnsNeededIDs proves the manifest diff: a first
// sync needs every conversation, a re-sync with one changed fingerprint needs
// only that conversation, and an unchanged re-sync needs none. This is the
// engine owning drift so clyde holds no change-tracking state.
func TestConversationManifestSyncReturnsNeededIDs(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{}
	ctx := context.Background()
	collectionID := "thread-manifest"

	needed, err := manager.SyncConversationManifest(ctx, collectionID, map[string]string{"conv-a": "fp-a-1", "conv-b": "fp-b-1"})
	if err != nil {
		t.Fatalf("first SyncConversationManifest returned error: %v", err)
	}
	if len(needed) != 2 {
		t.Fatalf("first sync needed %d ids, want 2", len(needed))
	}

	job, err := manager.upsertConversationDocuments(ctx, collectionID, []model.ConversationDocument{
		{ConversationID: "conv-a", MessageIndex: 0, Role: "user", Text: "alpha"},
		{ConversationID: "conv-b", MessageIndex: 0, Role: "user", Text: "beta"},
	}, map[string]string{"conv-a": "fp-a-1", "conv-b": "fp-b-1"}, testClientInfo(), absenceRetain)
	if err != nil {
		t.Fatalf("upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, job.ID, model.JobStateCompleted)

	needed, err = manager.SyncConversationManifest(ctx, collectionID, map[string]string{"conv-a": "fp-a-2", "conv-b": "fp-b-1"})
	if err != nil {
		t.Fatalf("second SyncConversationManifest returned error: %v", err)
	}
	if len(needed) != 1 || needed[0] != "conv-a" {
		t.Fatalf("second sync needed %v, want [conv-a]", needed)
	}

	needed, err = manager.SyncConversationManifest(ctx, collectionID, map[string]string{"conv-a": "fp-a-1", "conv-b": "fp-b-1"})
	if err != nil {
		t.Fatalf("third SyncConversationManifest returned error: %v", err)
	}
	if len(needed) != 0 {
		t.Fatalf("unchanged re-sync needed %v, want none", needed)
	}
}

func TestConversationEmptyDiffStoredNamePresentCompletesNoop(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	fake := &fakeSemantic{}
	manager.semantic = fake
	ctx := context.Background()
	collectionID := "thread-stored-name"
	manifest := map[string]string{"conv-stored": "fp-stored-1"}
	documents := []model.ConversationDocument{{
		ConversationID: "conv-stored",
		MessageIndex:   0,
		Role:           "user",
		TimestampUnix:  1712346000,
		Text:           "stored collection regression",
	}}

	firstJob, err := manager.upsertConversationDocuments(ctx, collectionID, documents, manifest, testClientInfo(), absenceRetain)
	if err != nil {
		t.Fatalf("first upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, firstJob.ID, model.JobStateCompleted)

	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	storedCollection := codebase.CollectionName
	derivedCollection := fake.CollectionName(codebase.CanonicalPath)
	if storedCollection == derivedCollection {
		t.Fatalf("stored collection unexpectedly equals derived collection %q", storedCollection)
	}

	fake.mu.Lock()
	fake.reindexCalls = nil
	fake.stageCalls = nil
	fake.promoted = nil
	fake.droppedStaging = nil
	fake.inspectCollection = func(_ context.Context, collectionName string) (semantic.CollectionFacts, error) {
		if collectionName == storedCollection {
			return semantic.CollectionFacts{Exists: true, Rows: 3, RowsKnown: true}, nil
		}
		return semantic.CollectionFacts{Exists: false, Rows: 0, RowsKnown: false}, nil
	}
	fake.mu.Unlock()

	secondJob, err := manager.upsertConversationDocuments(ctx, collectionID, nil, manifest, testClientInfo(), absenceRetain)
	if err != nil {
		t.Fatalf("second upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, secondJob.ID, model.JobStateCompleted)

	if got := fake.reindexCallsSnapshot(); len(got) != 0 {
		t.Fatalf("reindex calls after empty diff = %d, want 0: %+v", len(got), got)
	}
	if got := fake.stageCallsSnapshot(); len(got) != 0 {
		t.Fatalf("stage calls after empty diff = %d, want 0: %+v", len(got), got)
	}
	if got := fake.promotedSnapshot(); len(got) != 0 {
		t.Fatalf("promoted staging collections after empty diff = %d, want 0: %v", len(got), got)
	}
	if got := fake.droppedStagingSnapshot(); len(got) != 0 {
		t.Fatalf("dropped staging collections after empty diff = %d, want 0: %v", len(got), got)
	}
}

func TestSyncConversationManifestCapsNeededReply(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{}
	manager.config.MaxConversationsPerIngest = 3
	ctx := context.Background()
	collectionID := "thread-manifest-cap"
	manifest := testConversationManifest(
		"conv-01",
		"conv-02",
		"conv-03",
		"conv-04",
		"conv-05",
		"conv-06",
		"conv-07",
		"conv-08",
	)

	needed, err := manager.SyncConversationManifest(ctx, collectionID, manifest)
	if err != nil {
		t.Fatalf("first SyncConversationManifest returned error: %v", err)
	}
	assertStringSliceEqual(t, needed, []string{"conv-01", "conv-02", "conv-03"})

	upsertConversationsForManifest(t, manager, ctx, collectionID, manifest, needed)

	needed, err = manager.SyncConversationManifest(ctx, collectionID, manifest)
	if err != nil {
		t.Fatalf("second SyncConversationManifest returned error: %v", err)
	}
	assertStringSliceEqual(t, needed, []string{"conv-04", "conv-05", "conv-06"})

	upsertConversationsForManifest(t, manager, ctx, collectionID, manifest, needed)

	needed, err = manager.SyncConversationManifest(ctx, collectionID, manifest)
	if err != nil {
		t.Fatalf("third SyncConversationManifest returned error: %v", err)
	}
	assertStringSliceEqual(t, needed, []string{"conv-07", "conv-08"})

	upsertConversationsForManifest(t, manager, ctx, collectionID, manifest, needed)

	needed, err = manager.SyncConversationManifest(ctx, collectionID, manifest)
	if err != nil {
		t.Fatalf("fourth SyncConversationManifest returned error: %v", err)
	}
	assertStringSliceEqual(t, needed, nil)
}

func TestSyncConversationManifestPrefersModified(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{}
	manager.config.MaxConversationsPerIngest = 3
	ctx := context.Background()
	collectionID := "thread-manifest-modified"
	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	checkpoint := map[string]string{
		"conv-mod-a": "fp-mod-a-old",
		"conv-mod-b": "fp-mod-b-old",
	}
	if err := merkle.WriteSnapshot(manager.merklePath(codebase.ID), merkle.Snapshot{
		ConfigDigest: codebase.EffectiveConfig.IgnoreDigest,
		Files:        checkpoint,
		Inodes:       nil,
	}); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}
	manifest := testConversationManifest(
		"conv-added-01",
		"conv-added-02",
		"conv-added-03",
		"conv-added-04",
		"conv-added-05",
		"conv-mod-a",
		"conv-mod-b",
	)

	needed, err := manager.SyncConversationManifest(ctx, collectionID, manifest)
	if err != nil {
		t.Fatalf("SyncConversationManifest returned error: %v", err)
	}

	if len(needed) != 3 {
		t.Fatalf("needed = %v, want 3 ids", needed)
	}
	assertStringSliceContains(t, needed, "conv-mod-a")
	assertStringSliceContains(t, needed, "conv-mod-b")
}

func TestSyncConversationManifestRotatesModifiedOverflow(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{}
	manager.config.MaxConversationsPerIngest = 3
	ctx := context.Background()
	collectionID := "thread-manifest-modified-overflow"
	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	modifiedIDs := []string{
		"conv-mod-01",
		"conv-mod-02",
		"conv-mod-03",
		"conv-mod-04",
		"conv-mod-05",
	}
	checkpoint := make(map[string]string, len(modifiedIDs))
	for _, conversationID := range modifiedIDs {
		checkpoint[conversationID] = "old-" + conversationID
	}
	if err := merkle.WriteSnapshot(manager.merklePath(codebase.ID), merkle.Snapshot{
		ConfigDigest: codebase.EffectiveConfig.IgnoreDigest,
		Files:        checkpoint,
		Inodes:       nil,
	}); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}
	manifest := testConversationManifest(modifiedIDs...)

	first, err := manager.SyncConversationManifest(ctx, collectionID, manifest)
	if err != nil {
		t.Fatalf("first SyncConversationManifest returned error: %v", err)
	}
	second, err := manager.SyncConversationManifest(ctx, collectionID, manifest)
	if err != nil {
		t.Fatalf("second SyncConversationManifest returned error: %v", err)
	}
	third, err := manager.SyncConversationManifest(ctx, collectionID, manifest)
	if err != nil {
		t.Fatalf("third SyncConversationManifest returned error: %v", err)
	}
	fourth, err := manager.SyncConversationManifest(ctx, collectionID, manifest)
	if err != nil {
		t.Fatalf("fourth SyncConversationManifest returned error: %v", err)
	}
	expectedReplies := [][]string{
		{"conv-mod-01", "conv-mod-02", "conv-mod-03"},
		{"conv-mod-01", "conv-mod-04", "conv-mod-05"},
		{"conv-mod-02", "conv-mod-03", "conv-mod-04"},
		{"conv-mod-01", "conv-mod-02", "conv-mod-05"},
	}
	for i, expectedReply := range expectedReplies {
		reply := [][]string{first, second, third, fourth}[i]
		assertStringSliceEqual(t, reply, expectedReply)
	}

	replies := [][]string{first, second, third, fourth}
	for i := 0; i < len(modifiedIDs); i++ {
		reply, err := manager.SyncConversationManifest(ctx, collectionID, manifest)
		if err != nil {
			t.Fatalf("repeat SyncConversationManifest returned error: %v", err)
		}
		replies = append(replies, reply)
	}

	expected := make(map[string]bool, len(modifiedIDs))
	seen := make(map[string]bool, len(modifiedIDs))
	for _, conversationID := range modifiedIDs {
		expected[conversationID] = true
	}
	for _, reply := range replies {
		if len(reply) != 3 {
			t.Fatalf("reply = %v, want exactly 3 ids", reply)
		}
		for _, conversationID := range reply {
			if !expected[conversationID] {
				t.Fatalf("reply = %v, contains unexpected id %q", reply, conversationID)
			}
			seen[conversationID] = true
		}
	}
	for _, conversationID := range modifiedIDs {
		if !seen[conversationID] {
			t.Fatalf("modified id %q never appeared across replies %v", conversationID, replies)
		}
	}
}

func TestSyncConversationManifestRotatesAddedWindow(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{}
	manager.config.MaxConversationsPerIngest = 3
	ctx := context.Background()
	collectionID := "thread-manifest-rotate"
	manifest := testConversationManifest(
		"conv-01",
		"conv-02",
		"conv-03",
		"conv-04",
		"conv-05",
		"conv-06",
		"conv-07",
		"conv-08",
	)

	first, err := manager.SyncConversationManifest(ctx, collectionID, manifest)
	if err != nil {
		t.Fatalf("first SyncConversationManifest returned error: %v", err)
	}
	assertStringSliceEqual(t, first, []string{"conv-01", "conv-02", "conv-03"})

	second, err := manager.SyncConversationManifest(ctx, collectionID, manifest)
	if err != nil {
		t.Fatalf("second SyncConversationManifest returned error: %v", err)
	}
	assertStringSliceEqual(t, second, []string{"conv-04", "conv-05", "conv-06"})

	third, err := manager.SyncConversationManifest(ctx, collectionID, manifest)
	if err != nil {
		t.Fatalf("third SyncConversationManifest returned error: %v", err)
	}
	assertStringSliceEqual(t, third, []string{"conv-01", "conv-07", "conv-08"})

	fourth, err := manager.SyncConversationManifest(ctx, collectionID, manifest)
	if err != nil {
		t.Fatalf("fourth SyncConversationManifest returned error: %v", err)
	}
	assertStringSliceEqual(t, fourth, []string{"conv-02", "conv-03", "conv-04"})

	replies := [][]string{first, second, third, fourth}
	seen := make(map[string]bool, len(manifest))
	for _, reply := range replies {
		for _, conversationID := range reply {
			seen[conversationID] = true
		}
	}
	for conversationID := range manifest {
		if !seen[conversationID] {
			t.Fatalf("conversation id %q never appeared across replies %v", conversationID, replies)
		}
	}
}

func TestSyncConversationManifestUncappedWhenZero(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{}
	manager.config.MaxConversationsPerIngest = 0
	ctx := context.Background()
	collectionID := "thread-manifest-uncapped"
	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	checkpoint := map[string]string{
		"conv-mod": "fp-mod-old",
	}
	if err := merkle.WriteSnapshot(manager.merklePath(codebase.ID), merkle.Snapshot{
		ConfigDigest: codebase.EffectiveConfig.IgnoreDigest,
		Files:        checkpoint,
		Inodes:       nil,
	}); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}
	manifest := testConversationManifest("conv-added-a", "conv-added-b", "conv-added-c", "conv-mod")

	needed, err := manager.SyncConversationManifest(ctx, collectionID, manifest)
	if err != nil {
		t.Fatalf("SyncConversationManifest returned error: %v", err)
	}
	assertStringSliceEqual(t, needed, []string{"conv-added-a", "conv-added-b", "conv-added-c", "conv-mod"})
}

func TestRegisterConversationCollectionRPC(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	server := NewGRPCServer(manager, nil)

	response, err := server.RegisterConversationCollection(context.Background(), &pb.RegisterConversationCollectionRequest{
		CollectionId: "thread-rpc",
	})
	if err != nil {
		t.Fatalf("RegisterConversationCollection RPC returned error: %v", err)
	}
	if response.GetCodebaseId() == "" {
		t.Fatal("response CodebaseId is empty")
	}
	if !strings.HasPrefix(response.GetCollectionName(), "conv_chunks_") {
		t.Fatalf("response CollectionName = %q, want conv_chunks_ prefix", response.GetCollectionName())
	}
	if !strings.Contains(response.GetDisplayText(), "Registered conversation collection") {
		t.Fatalf("response DisplayText missing registration text: %q", response.GetDisplayText())
	}
}

func TestRunSyncAllLeavesDocumentCollectionUntouched(t *testing.T) {
	t.Parallel()

	manager, cfg, _ := newTestManager(t)
	codebase, err := manager.RegisterConversationCollection(context.Background(), "thread-sync")
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	manager.semantic = &fakeSemantic{
		listCollections: func(context.Context) ([]string, error) { return []string{}, nil },
	}

	syncer := NewBackgroundSync(cfg, manager)
	syncer.runSyncAll(context.Background(), "test")

	manager.mu.Lock()
	got, found := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if !found {
		t.Fatal("document codebase was removed by background sync")
	}
	if got.Kind != model.CodebaseKindDocument {
		t.Fatalf("Kind after sync = %q, want %q", got.Kind, model.CodebaseKindDocument)
	}
	if got.Status != model.CodebaseStatusIndexed {
		t.Fatalf("Status after sync = %q, want %q", got.Status, model.CodebaseStatusIndexed)
	}
	if got.CanonicalPath != codebase.CanonicalPath {
		t.Fatalf("CanonicalPath after sync = %q, want %q", got.CanonicalPath, codebase.CanonicalPath)
	}
	if got.CollectionName != codebase.CollectionName {
		t.Fatalf("CollectionName after sync = %q, want %q", got.CollectionName, codebase.CollectionName)
	}
	if jobs := manager.ListJobs(""); len(jobs) != 0 {
		t.Fatalf("background sync queued %d jobs for a document collection, want 0", len(jobs))
	}
}

func TestSearchConversationsReturnsConversationMetadata(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	collectionID := "thread-search"
	codebase, err := manager.RegisterConversationCollection(context.Background(), collectionID)
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}

	expectedChunks := []model.StoredChunk{{
		Content:              "Needle conversation content",
		RelativePath:         "conv/conversation-alpha/12",
		StartLine:            0,
		EndLine:              0,
		Language:             "",
		FileExtension:        "",
		ConversationID:       "conversation-alpha",
		ParentConversationID: "conversation-root",
		MessageIndex:         12,
		Role:                 "assistant",
		TimestampUnix:        1712345678,
	}}
	searchCalls := 0
	manager.semantic = &fakeSemantic{
		conversationSearch: func(ctx context.Context, collectionName string, query string, limit int32) ([]model.StoredChunk, error) {
			_ = ctx
			searchCalls++
			if collectionName != codebase.CollectionName {
				t.Fatalf("collectionName = %q, want %q", collectionName, codebase.CollectionName)
			}
			if query != "needle" {
				t.Fatalf("query = %q, want needle", query)
			}
			if limit != 5 {
				t.Fatalf("limit = %d, want 5", limit)
			}
			return append([]model.StoredChunk{}, expectedChunks...), nil
		},
	}

	chunks, err := manager.SearchConversations(context.Background(), collectionID, "needle", 5, conversationSearchFilter{Roles: nil, FromUnix: 0, UntilUnix: 0, ConversationIDs: nil, ParentConversationID: "", MinScore: 0, MessageIndexFrom: 0, MessageIndexUntil: 0}, 0)
	if err != nil {
		t.Fatalf("SearchConversations returned error: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("SearchConversations returned %d chunks, want 1", len(chunks))
	}
	if chunks[0].ConversationID != expectedChunks[0].ConversationID {
		t.Fatalf("ConversationID = %q, want %q", chunks[0].ConversationID, expectedChunks[0].ConversationID)
	}
	if chunks[0].ParentConversationID != expectedChunks[0].ParentConversationID {
		t.Fatalf("ParentConversationID = %q, want %q", chunks[0].ParentConversationID, expectedChunks[0].ParentConversationID)
	}
	if chunks[0].MessageIndex != expectedChunks[0].MessageIndex {
		t.Fatalf("MessageIndex = %d, want %d", chunks[0].MessageIndex, expectedChunks[0].MessageIndex)
	}
	if chunks[0].Role != expectedChunks[0].Role {
		t.Fatalf("Role = %q, want %q", chunks[0].Role, expectedChunks[0].Role)
	}
	if chunks[0].TimestampUnix != expectedChunks[0].TimestampUnix {
		t.Fatalf("TimestampUnix = %d, want %d", chunks[0].TimestampUnix, expectedChunks[0].TimestampUnix)
	}

	server := NewGRPCServer(manager, nil)
	response, err := server.SearchConversations(context.Background(), &pb.SearchConversationsRequest{
		CollectionId: collectionID,
		Query:        "needle",
		Limit:        5,
	})
	if err != nil {
		t.Fatalf("SearchConversations RPC returned error: %v", err)
	}
	if len(response.GetResults()) != 1 {
		t.Fatalf("SearchConversations RPC returned %d results, want 1", len(response.GetResults()))
	}
	result := response.GetResults()[0]
	if result.GetConversationId() != expectedChunks[0].ConversationID {
		t.Fatalf("RPC ConversationId = %q, want %q", result.GetConversationId(), expectedChunks[0].ConversationID)
	}
	if result.GetParentConversationId() != expectedChunks[0].ParentConversationID {
		t.Fatalf("RPC ParentConversationId = %q, want %q", result.GetParentConversationId(), expectedChunks[0].ParentConversationID)
	}
	if result.GetMessageIndex() != expectedChunks[0].MessageIndex {
		t.Fatalf("RPC MessageIndex = %d, want %d", result.GetMessageIndex(), expectedChunks[0].MessageIndex)
	}
	if result.GetRole() != expectedChunks[0].Role {
		t.Fatalf("RPC Role = %q, want %q", result.GetRole(), expectedChunks[0].Role)
	}
	if result.GetTimestampUnix() != expectedChunks[0].TimestampUnix {
		t.Fatalf("RPC TimestampUnix = %d, want %d", result.GetTimestampUnix(), expectedChunks[0].TimestampUnix)
	}
	if result.GetContent() != expectedChunks[0].Content {
		t.Fatalf("RPC Content = %q, want %q", result.GetContent(), expectedChunks[0].Content)
	}
	if response.GetDependencyHealth() == nil {
		t.Fatal("SearchConversations RPC returned nil DependencyHealth")
	}
	if !strings.Contains(response.GetDisplayText(), "Found 1 conversation results") {
		t.Fatalf("SearchConversations RPC DisplayText = %q, want result summary", response.GetDisplayText())
	}

	callsBeforeUnregistered := searchCalls
	unregisteredChunks, err := manager.SearchConversations(context.Background(), "missing-thread", "needle", 5, conversationSearchFilter{Roles: nil, FromUnix: 0, UntilUnix: 0, ConversationIDs: nil, ParentConversationID: "", MinScore: 0, MessageIndexFrom: 0, MessageIndexUntil: 0}, 0)
	if err != nil {
		t.Fatalf("SearchConversations for unregistered collection returned error: %v", err)
	}
	if len(unregisteredChunks) != 0 {
		t.Fatalf("SearchConversations for unregistered collection returned %d chunks, want 0", len(unregisteredChunks))
	}
	if searchCalls != callsBeforeUnregistered {
		t.Fatalf("SearchConversations called semantic backend for unregistered collection")
	}

	unregisteredResponse, err := server.SearchConversations(context.Background(), &pb.SearchConversationsRequest{
		CollectionId: "missing-thread",
		Query:        "needle",
		Limit:        5,
	})
	if err != nil {
		t.Fatalf("SearchConversations RPC for unregistered collection returned error: %v", err)
	}
	if len(unregisteredResponse.GetResults()) != 0 {
		t.Fatalf("SearchConversations RPC for unregistered collection returned %d results, want 0", len(unregisteredResponse.GetResults()))
	}
	if searchCalls != callsBeforeUnregistered {
		t.Fatalf("SearchConversations RPC called semantic backend for unregistered collection")
	}
}

func TestConversationIngestDoesNotWriteChunkCache(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{}
	ctx := context.Background()
	collectionID := "thread-cache"
	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}

	firstManifest := map[string]string{"conv-alpha": "alpha-1", "conv-beta": "beta-1"}
	firstJob, err := manager.upsertConversationDocuments(ctx, collectionID, []model.ConversationDocument{
		{
			ConversationID: "conv-alpha",
			MessageIndex:   0,
			Role:           "user",
			TimestampUnix:  1712345000,
			Text:           "old needle cache entry",
		},
		{
			ConversationID: "conv-beta",
			MessageIndex:   0,
			Role:           "assistant",
			TimestampUnix:  1712345001,
			Text:           "beta cache entry",
		},
	}, firstManifest, testClientInfo(), absenceRetain)
	if err != nil {
		t.Fatalf("upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, firstJob.ID, model.JobStateCompleted)

	if _, err := os.Stat(manager.chunkPath(codebase.ID)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("document ingest chunk cache stat error = %v, want not exist", err)
	}

	// conv-beta is unchanged, so its fingerprint stays and clyde sends no
	// document for it. conv-alpha changed, so only its document is delivered.
	secondManifest := map[string]string{"conv-alpha": "alpha-2", "conv-beta": "beta-1"}
	secondJob, err := manager.upsertConversationDocuments(ctx, collectionID, []model.ConversationDocument{{
		ConversationID: "conv-alpha",
		MessageIndex:   1,
		Role:           "assistant",
		TimestampUnix:  1712345002,
		Text:           "fresh needle cache entry",
	}}, secondManifest, testClientInfo(), absenceRetain)
	if err != nil {
		t.Fatalf("second upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, secondJob.ID, model.JobStateCompleted)

	if _, err := os.Stat(manager.chunkPath(codebase.ID)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("document re-ingest chunk cache stat error = %v, want not exist", err)
	}
}

func TestSearchConversationsReturnsUnavailableWhenSemanticUnavailable(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	collectionID := "thread-cache-error"
	codebase, err := manager.RegisterConversationCollection(context.Background(), collectionID)
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	manager.semantic = &fakeSemantic{unavailable: true}
	if err := store.WriteChunks(manager.chunkPath(codebase.ID), []model.StoredChunk{
		{
			Content:        "needle cache fallback result",
			RelativePath:   "conv/conv-cache/0",
			ConversationID: "conv-cache",
			MessageIndex:   0,
			Role:           "assistant",
			TimestampUnix:  1712345000,
		},
		{
			Content:        "unrelated cache result",
			RelativePath:   "conv/conv-other/0",
			ConversationID: "conv-other",
			MessageIndex:   0,
			Role:           "user",
			TimestampUnix:  1712345001,
		},
	}); err != nil {
		t.Fatalf("WriteChunks returned error: %v", err)
	}

	results, err := manager.SearchConversations(context.Background(), collectionID, "needle", 5, conversationSearchFilter{Roles: nil, FromUnix: 0, UntilUnix: 0, ConversationIDs: nil, ParentConversationID: "", MinScore: 0, MessageIndexFrom: 0, MessageIndexUntil: 0}, 0)
	if !errors.Is(err, semantic.ErrUnavailable) {
		t.Fatalf("SearchConversations error = %v, want semantic.ErrUnavailable", err)
	}
	if len(results) != 0 {
		t.Fatalf("SearchConversations returned %d chunks, want none", len(results))
	}
}

func TestSearchConversationsRPCReturnsUnavailableWhenSemanticUnavailable(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{unavailable: true}
	server := NewGRPCServer(manager, nil)
	ctx := context.Background()

	if _, err := manager.RegisterConversationCollection(ctx, "thread-rpc-unavailable"); err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}

	_, err := server.SearchConversations(ctx, &pb.SearchConversationsRequest{
		CollectionId: "thread-rpc-unavailable",
		Query:        "needle",
		Limit:        5,
	})
	if status.Code(err) != codes.Unavailable {
		t.Fatalf("SearchConversations RPC status = %v, want %v (err=%v)", status.Code(err), codes.Unavailable, err)
	}
}

func TestConversationDeleteFailsWhenSemanticUnavailableWithoutReadingChunkCache(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{unavailable: true}
	ctx := context.Background()
	collectionID := "thread-delete-unavailable"
	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	if err := store.WriteChunks(manager.chunkPath(codebase.ID), []model.StoredChunk{{
		Content:        "cached chunk must stay untouched",
		RelativePath:   "conv/conv-delete/0",
		ConversationID: "conv-delete",
	}}); err != nil {
		t.Fatalf("WriteChunks returned error: %v", err)
	}

	deleteJob, err := manager.DeleteConversation(ctx, collectionID, "conv-delete")
	if err != nil {
		t.Fatalf("DeleteConversation returned error: %v", err)
	}
	waitForConversationJobState(t, manager, deleteJob.ID, model.JobStateFailed)

	job, found := manager.GetJob(deleteJob.ID)
	if !found {
		t.Fatal("delete job not found")
	}
	if job.Error == nil || !strings.Contains(job.Error.Message, "semantic backend is unavailable") {
		t.Fatalf("delete job error = %+v, want semantic backend unavailable", job.Error)
	}

	chunks := readConversationChunkCache(t, manager, codebase.ID)
	if len(chunks) != 1 || chunks[0].ConversationID != "conv-delete" {
		t.Fatalf("delete touched cache chunks = %+v, want original conv-delete chunk", chunks)
	}
}

func TestCancelledConversationIngestReportsCancelledWhenSemanticUnavailable(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{}
	ctx := context.Background()
	codebase, err := manager.RegisterConversationCollection(ctx, "thread-cancel-ingest")
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	manager.semantic = &fakeSemantic{unavailable: true}
	job := stageConversationJob(t, manager, codebase, conversationJobPayload{
		Kind:           conversationJobKindUpsert,
		CollectionName: codebase.CollectionName,
		Manifest:       map[string]string{"conv-cancel": "fp-cancel"},
		Documents: []model.ConversationDocument{{
			ConversationID: "conv-cancel",
			MessageIndex:   0,
			Role:           "user",
			Text:           "cancelled ingest",
		}},
	})

	cancelledContext, cancel := context.WithCancel(ctx)
	cancel()
	manager.runConversationIngest(cancelledContext, job)

	assertConversationJobCancelled(t, manager, job.ID)
}

func TestCancelledConversationDeleteReportsCancelledWhenSemanticUnavailable(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{}
	ctx := context.Background()
	codebase, err := manager.RegisterConversationCollection(ctx, "thread-cancel-delete")
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	manager.semantic = &fakeSemantic{unavailable: true}
	job := stageConversationJob(t, manager, codebase, conversationJobPayload{
		Kind:           conversationJobKindDelete,
		CollectionName: codebase.CollectionName,
		ConversationID: "conv-cancel-delete",
	})

	cancelledContext, cancel := context.WithCancel(ctx)
	cancel()
	manager.runConversationDelete(cancelledContext, job, conversationJobPayload{
		Kind:           conversationJobKindDelete,
		CollectionName: codebase.CollectionName,
		ConversationID: "conv-cancel-delete",
	})

	assertConversationJobCancelled(t, manager, job.ID)
}

func TestConversationDocumentsToStoredChunksSplitsOversizedMessage(t *testing.T) {
	t.Parallel()

	text := strings.Repeat("a", conversationChunkMaxBytes+5)
	chunks, err := conversationDocumentsToStoredChunks(context.Background(), []model.ConversationDocument{{
		ConversationID:       "thread-large",
		ParentConversationID: "thread-large-parent",
		MessageIndex:         7,
		Role:                 "assistant",
		TimestampUnix:        1712345678,
		Text:                 text,
	}})
	if err != nil {
		t.Fatalf("conversationDocumentsToStoredChunks returned error: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("conversationDocumentsToStoredChunks returned %d chunks, want 2", len(chunks))
	}
	if chunks[0].RelativePath != "conv/thread-large/7/0" {
		t.Fatalf("first RelativePath = %q, want conv/thread-large/7/0", chunks[0].RelativePath)
	}
	if chunks[1].RelativePath != "conv/thread-large/7/1" {
		t.Fatalf("second RelativePath = %q, want conv/thread-large/7/1", chunks[1].RelativePath)
	}
	for index, chunk := range chunks {
		if len(chunk.Content) > conversationChunkMaxBytes {
			t.Fatalf("chunk %d has %d bytes, want at most %d", index, len(chunk.Content), conversationChunkMaxBytes)
		}
		if chunk.ConversationID != "thread-large" {
			t.Fatalf("chunk %d ConversationID = %q, want thread-large", index, chunk.ConversationID)
		}
		if chunk.ParentConversationID != "thread-large-parent" {
			t.Fatalf("chunk %d ParentConversationID = %q, want thread-large-parent", index, chunk.ParentConversationID)
		}
		if chunk.MessageIndex != 7 {
			t.Fatalf("chunk %d MessageIndex = %d, want 7", index, chunk.MessageIndex)
		}
		if chunk.Role != "assistant" {
			t.Fatalf("chunk %d Role = %q, want assistant", index, chunk.Role)
		}
		if chunk.TimestampUnix != 1712345678 {
			t.Fatalf("chunk %d TimestampUnix = %d, want 1712345678", index, chunk.TimestampUnix)
		}
	}
}

func TestConversationIndexOneEmbedsOnlyAppendedMessage(t *testing.T) {
	t.Parallel()

	reader := &testConversationRowReader{
		state: map[int32]semantic.StoredMessageState{
			0: {Role: "user", Text: "hello"},
			1: {Role: "assistant", Text: "old answer"},
		},
		reuse: map[string][]float32{"reuse-alpha": {1, 2}},
	}
	source := newConversationItemSource(
		"conv_chunks_live",
		map[string]string{"conv-append": "fp-append"},
		[]model.ConversationDocument{
			{ConversationID: "conv-append", MessageIndex: 0, Role: "user", Text: "hello"},
			{ConversationID: "conv-append", MessageIndex: 1, Role: "assistant", Text: "old answer"},
			{ConversationID: "conv-append", MessageIndex: 2, Role: "user", Text: "new question"},
		},
		reader,
		absenceRetain,
	)

	result, err := source.indexOne(context.Background(), "conv-append")
	if err != nil {
		t.Fatalf("indexOne returned error: %v", err)
	}

	assertConversationDeltaChunks(t, result.Chunks, []conversationDeltaChunkWant{
		{relativePath: "conv/conv-append/2", messageIndex: 2, role: "user", content: "new question"},
	})
	if result.FileHash != "fp-append" {
		t.Fatalf("FileHash = %q, want fp-append", result.FileHash)
	}
	if !result.RemovalOverride {
		t.Fatal("RemovalOverride = false, want true")
	}
	assertStringSliceEqual(t, result.RemovalPaths, conversationRemovalPathsForTest("conv-append", 2))
	assertStringSliceEqual(t, result.RemovalPrefixes, conversationRemovalPrefixesForTest("conv-append", 2))
	assertReuseVector(t, result.ReuseVectors, "reuse-alpha", []float32{1, 2})
	assertMessageStateCalls(t, reader.callsSnapshot(), []messageStateCall{{
		Collection: "conv_chunks_live",
		Prefix:     "conv/conv-append/",
	}})
}

func TestConversationIndexOneReindexesOnlyEditedMessage(t *testing.T) {
	t.Parallel()

	reader := &testConversationRowReader{
		state: map[int32]semantic.StoredMessageState{
			0: {Role: "user", Text: "hello"},
			1: {Role: "assistant", Text: "old answer"},
			2: {Role: "user", Text: "follow up"},
		},
		reuse: map[string][]float32{},
	}
	source := newConversationItemSource(
		"conv_chunks_live",
		map[string]string{"conv-edit": "fp-edit"},
		[]model.ConversationDocument{
			{ConversationID: "conv-edit", MessageIndex: 0, Role: "user", Text: "hello"},
			{ConversationID: "conv-edit", MessageIndex: 1, Role: "assistant", Text: "new answer"},
			{ConversationID: "conv-edit", MessageIndex: 2, Role: "user", Text: "follow up"},
		},
		reader,
		absenceRetain,
	)

	result, err := source.indexOne(context.Background(), "conv-edit")
	if err != nil {
		t.Fatalf("indexOne returned error: %v", err)
	}

	assertConversationDeltaChunks(t, result.Chunks, []conversationDeltaChunkWant{
		{relativePath: "conv/conv-edit/1", messageIndex: 1, role: "assistant", content: "new answer"},
	})
	assertStringSliceEqual(t, result.RemovalPaths, conversationRemovalPathsForTest("conv-edit", 1))
	assertStringSliceEqual(t, result.RemovalPrefixes, conversationRemovalPrefixesForTest("conv-edit", 1))
}

func TestConversationIndexOneDeletesStaleMessages(t *testing.T) {
	t.Parallel()

	reader := &testConversationRowReader{
		state: map[int32]semantic.StoredMessageState{
			0: {Role: "user", Text: "hello"},
			1: {Role: "assistant", Text: "answer"},
			2: {Role: "user", Text: "stale"},
		},
		reuse: map[string][]float32{"reuse-stale": {3}},
	}
	source := newConversationItemSource(
		"conv_chunks_live",
		map[string]string{"conv-stale": "fp-stale"},
		[]model.ConversationDocument{
			{ConversationID: "conv-stale", MessageIndex: 0, Role: "user", Text: "hello"},
			{ConversationID: "conv-stale", MessageIndex: 1, Role: "assistant", Text: "answer"},
		},
		reader,
		absenceRetain,
	)

	result, err := source.indexOne(context.Background(), "conv-stale")
	if err != nil {
		t.Fatalf("indexOne returned error: %v", err)
	}

	if len(result.Chunks) != 0 {
		t.Fatalf("Chunks = %+v, want none for stale-only delta", result.Chunks)
	}
	assertStringSliceEqual(t, result.RemovalPaths, conversationRemovalPathsForTest("conv-stale", 2))
	assertStringSliceEqual(t, result.RemovalPrefixes, conversationRemovalPrefixesForTest("conv-stale", 2))
	assertReuseVector(t, result.ReuseVectors, "reuse-stale", []float32{3})

	manager, _, _ := newTestManager(t)
	fake := &fakeSemantic{
		loadMessageState: func(context.Context, string, string) (map[int32]semantic.StoredMessageState, map[string][]float32, error) {
			return cloneStoredMessageState(reader.state), map[string][]float32{}, nil
		},
	}
	manager.semantic = fake
	ctx := context.Background()
	collectionID := "thread-stale-checkpoint"
	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	if err := merkle.WriteSnapshot(manager.merklePath(codebase.ID), merkle.Snapshot{
		ConfigDigest: codebase.EffectiveConfig.IgnoreDigest,
		Files:        map[string]string{"conv-stale": "fp-stale-old"},
		Inodes:       nil,
	}); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	job, err := manager.upsertConversationDocuments(ctx, collectionID, source.documents["conv-stale"], source.manifest, testClientInfo(), absenceRetain)
	if err != nil {
		t.Fatalf("upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, job.ID, model.JobStateCompleted)

	snapshot, err := merkle.ReadSnapshot(manager.merklePath(codebase.ID))
	if err != nil {
		t.Fatalf("ReadSnapshot returned error: %v", err)
	}
	if snapshot.Files["conv-stale"] != "fp-stale" {
		t.Fatalf("checkpoint fingerprint = %q, want fp-stale", snapshot.Files["conv-stale"])
	}
	calls := fake.reindexCallsSnapshot()
	if len(calls) != 1 {
		t.Fatalf("Reindex calls = %+v, want one stale removal call", calls)
	}
	if calls[0].Chunks != 0 {
		t.Fatalf("stale removal chunks = %d, want 0", calls[0].Chunks)
	}
	assertRemovalEqual(t, calls[0].Removal, semantic.Removal{
		Paths:    conversationRemovalPathsForTest("conv-stale", 2),
		Prefixes: conversationRemovalPrefixesForTest("conv-stale", 2),
	})
}

func TestConversationIndexOneMultipartTransitions(t *testing.T) {
	t.Parallel()

	t.Run("multipart stored row becomes single part", func(t *testing.T) {
		t.Parallel()

		reader := &testConversationRowReader{
			state: map[int32]semantic.StoredMessageState{
				7: {Role: "assistant", Text: strings.Repeat("a", conversationChunkMaxBytes+5)},
			},
			reuse: map[string][]float32{},
		}
		source := newConversationItemSource(
			"conv_chunks_live",
			map[string]string{"conv-shape": "fp-shape-single"},
			[]model.ConversationDocument{
				{ConversationID: "conv-shape", MessageIndex: 7, Role: "assistant", Text: "short"},
			},
			reader,
			absenceRetain,
		)

		result, err := source.indexOne(context.Background(), "conv-shape")
		if err != nil {
			t.Fatalf("indexOne returned error: %v", err)
		}

		assertConversationDeltaChunks(t, result.Chunks, []conversationDeltaChunkWant{
			{relativePath: "conv/conv-shape/7", messageIndex: 7, role: "assistant", content: "short"},
		})
		assertStringSliceEqual(t, result.RemovalPaths, conversationRemovalPathsForTest("conv-shape", 7))
		assertStringSliceEqual(t, result.RemovalPrefixes, conversationRemovalPrefixesForTest("conv-shape", 7))
	})

	t.Run("single stored row becomes multipart", func(t *testing.T) {
		t.Parallel()

		text := strings.Repeat("b", conversationChunkMaxBytes+5)
		reader := &testConversationRowReader{
			state: map[int32]semantic.StoredMessageState{
				8: {Role: "assistant", Text: "short"},
			},
			reuse: map[string][]float32{},
		}
		source := newConversationItemSource(
			"conv_chunks_live",
			map[string]string{"conv-shape": "fp-shape-multipart"},
			[]model.ConversationDocument{
				{ConversationID: "conv-shape", MessageIndex: 8, Role: "assistant", Text: text},
			},
			reader,
			absenceRetain,
		)

		result, err := source.indexOne(context.Background(), "conv-shape")
		if err != nil {
			t.Fatalf("indexOne returned error: %v", err)
		}

		if len(result.Chunks) != 2 {
			t.Fatalf("chunks = %d, want 2 multipart chunks", len(result.Chunks))
		}
		assertConversationDeltaChunks(t, result.Chunks, []conversationDeltaChunkWant{
			{relativePath: "conv/conv-shape/8/0", messageIndex: 8, role: "assistant", content: result.Chunks[0].Content},
			{relativePath: "conv/conv-shape/8/1", messageIndex: 8, role: "assistant", content: result.Chunks[1].Content},
		})
		if result.Chunks[0].Content+result.Chunks[1].Content != text {
			t.Fatalf("multipart content did not round trip to delivered text")
		}
		assertStringSliceEqual(t, result.RemovalPaths, conversationRemovalPathsForTest("conv-shape", 8))
		assertStringSliceEqual(t, result.RemovalPrefixes, conversationRemovalPrefixesForTest("conv-shape", 8))
	})
}

func TestConversationIndexOneSiblingIndexSafety(t *testing.T) {
	t.Parallel()

	reader := &testConversationRowReader{
		state: map[int32]semantic.StoredMessageState{
			12:  {Role: "user", Text: "old twelve"},
			120: {Role: "user", Text: "one twenty"},
		},
		reuse: map[string][]float32{},
	}
	source := newConversationItemSource(
		"conv_chunks_live",
		map[string]string{"conv-sibling": "fp-sibling"},
		[]model.ConversationDocument{
			{ConversationID: "conv-sibling", MessageIndex: 12, Role: "user", Text: "new twelve"},
			{ConversationID: "conv-sibling", MessageIndex: 120, Role: "user", Text: "one twenty"},
		},
		reader,
		absenceRetain,
	)

	result, err := source.indexOne(context.Background(), "conv-sibling")
	if err != nil {
		t.Fatalf("indexOne returned error: %v", err)
	}

	assertConversationDeltaChunks(t, result.Chunks, []conversationDeltaChunkWant{
		{relativePath: "conv/conv-sibling/12", messageIndex: 12, role: "user", content: "new twelve"},
	})
	assertStringSliceEqual(t, result.RemovalPaths, conversationRemovalPathsForTest("conv-sibling", 12))
	assertStringSliceEqual(t, result.RemovalPrefixes, conversationRemovalPrefixesForTest("conv-sibling", 12))
}

func TestConversationIndexOneStateLoadFailureFallsBackToFullReindex(t *testing.T) {
	t.Parallel()

	reader := &testConversationRowReader{err: errors.New("state load failed")}
	documents := []model.ConversationDocument{
		{ConversationID: "conv-load-failure", MessageIndex: 0, Role: "user", Text: "hello"},
		{ConversationID: "conv-load-failure", MessageIndex: 1, Role: "assistant", Text: "answer"},
	}
	source := newConversationItemSource(
		"conv_chunks_live",
		map[string]string{"conv-load-failure": "fp-load-failure"},
		documents,
		reader,
		absenceRetain,
	)

	result, err := source.indexOne(context.Background(), "conv-load-failure")
	if err != nil {
		t.Fatalf("indexOne returned error: %v", err)
	}

	assertConversationDeltaChunks(t, result.Chunks, []conversationDeltaChunkWant{
		{relativePath: "conv/conv-load-failure/0", messageIndex: 0, role: "user", content: "hello"},
		{relativePath: "conv/conv-load-failure/1", messageIndex: 1, role: "assistant", content: "answer"},
	})
	assertStringSliceEqual(t, result.RemovalPaths, nil)
	assertStringSliceEqual(t, result.RemovalPrefixes, conversationFullRemovalPrefixes("conv-load-failure"))
	if result.ReuseVectors != nil {
		t.Fatalf("ReuseVectors = %v, want nil on loader-error fallback", result.ReuseVectors)
	}

	manager, _, _ := newTestManager(t)
	fake := &fakeSemantic{
		loadMessageState: func(context.Context, string, string) (map[int32]semantic.StoredMessageState, map[string][]float32, error) {
			return nil, nil, errors.New("state load failed")
		},
		loadReuseForPrefix: func(context.Context, string, string) (map[string][]float32, error) {
			return map[string][]float32{"fallback-reuse": {5}}, nil
		},
	}
	manager.semantic = fake
	ctx := context.Background()
	collectionID := "thread-load-failure"
	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	if err := merkle.WriteSnapshot(manager.merklePath(codebase.ID), merkle.Snapshot{
		ConfigDigest: codebase.EffectiveConfig.IgnoreDigest,
		Files:        map[string]string{"conv-load-failure": "fp-load-failure-old"},
		Inodes:       nil,
	}); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	job, err := manager.upsertConversationDocuments(ctx, collectionID, documents, map[string]string{"conv-load-failure": "fp-load-failure"}, testClientInfo(), absenceRetain)
	if err != nil {
		t.Fatalf("upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, job.ID, model.JobStateCompleted)

	prefixCalls := fake.reusePrefixCallsSnapshot()
	if len(prefixCalls) != 1 || prefixCalls[0].Prefix != "conv/conv-load-failure/" {
		t.Fatalf("reuse prefix calls = %+v, want one fallback prefix load", prefixCalls)
	}
	reuseByConversation := fake.reindexReuseSnapshot()
	assertReuseVector(t, reuseByConversation["conv-load-failure"], "fallback-reuse", []float32{5})
}

func TestConversationIndexOneHealsMissingRows(t *testing.T) {
	t.Parallel()

	reader := &testConversationRowReader{
		state: map[int32]semantic.StoredMessageState{
			0: {Role: "user", Text: "hello"},
		},
		reuse: map[string][]float32{},
	}
	source := newConversationItemSource(
		"conv_chunks_live",
		map[string]string{"conv-heal": "fp-heal"},
		[]model.ConversationDocument{
			{ConversationID: "conv-heal", MessageIndex: 0, Role: "user", Text: "hello"},
			{ConversationID: "conv-heal", MessageIndex: 1, Role: "assistant", Text: "restored"},
		},
		reader,
		absenceRetain,
	)

	result, err := source.indexOne(context.Background(), "conv-heal")
	if err != nil {
		t.Fatalf("indexOne returned error: %v", err)
	}

	assertConversationDeltaChunks(t, result.Chunks, []conversationDeltaChunkWant{
		{relativePath: "conv/conv-heal/1", messageIndex: 1, role: "assistant", content: "restored"},
	})
	assertStringSliceEqual(t, result.RemovalPaths, conversationRemovalPathsForTest("conv-heal", 1))
	assertStringSliceEqual(t, result.RemovalPrefixes, conversationRemovalPrefixesForTest("conv-heal", 1))
}

func TestConversationIndexOneHealsLegacyRowsWithoutDuplicates(t *testing.T) {
	t.Parallel()

	reader := &testConversationRowReader{
		state: map[int32]semantic.StoredMessageState{},
		reuse: map[string][]float32{
			"legacy-zero": {1},
			"legacy-one":  {2},
		},
	}
	source := newConversationItemSource(
		"conv_chunks_live",
		map[string]string{"conv-legacy": "fp-legacy"},
		[]model.ConversationDocument{
			{ConversationID: "conv-legacy", MessageIndex: 0, Role: "user", Text: "hello"},
			{ConversationID: "conv-legacy", MessageIndex: 1, Role: "assistant", Text: "answer"},
		},
		reader,
		absenceRetain,
	)

	result, err := source.indexOne(context.Background(), "conv-legacy")
	if err != nil {
		t.Fatalf("indexOne returned error: %v", err)
	}

	assertConversationDeltaChunks(t, result.Chunks, []conversationDeltaChunkWant{
		{relativePath: "conv/conv-legacy/0", messageIndex: 0, role: "user", content: "hello"},
		{relativePath: "conv/conv-legacy/1", messageIndex: 1, role: "assistant", content: "answer"},
	})
	assertStringSliceEqual(t, result.RemovalPaths, append(
		conversationRemovalPathsForTest("conv-legacy", 0),
		conversationRemovalPathsForTest("conv-legacy", 1)...,
	))
	assertStringSliceEqual(t, result.RemovalPrefixes, append(
		conversationRemovalPrefixesForTest("conv-legacy", 0),
		conversationRemovalPrefixesForTest("conv-legacy", 1)...,
	))
	assertReuseVector(t, result.ReuseVectors, "legacy-zero", []float32{1})
	assertReuseVector(t, result.ReuseVectors, "legacy-one", []float32{2})
}

func TestConversationIngestWritesOnlyMessageDeltas(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	stateStore := newConversationStateStore()
	fake := &fakeSemantic{
		loadMessageState: func(_ context.Context, _ string, prefix string) (map[int32]semantic.StoredMessageState, map[string][]float32, error) {
			return stateStore.state(prefix), map[string][]float32{}, nil
		},
	}
	manager.semantic = fake
	ctx := context.Background()
	collectionID := "thread-message-delta"
	conversationID := "conv-e2e"
	prefix := conversationRelativePathPrefix(conversationID)

	coldDocuments := []model.ConversationDocument{
		{ConversationID: conversationID, MessageIndex: 0, Role: "user", Text: "hello"},
		{ConversationID: conversationID, MessageIndex: 1, Role: "assistant", Text: "answer"},
	}
	runConversationDeltaIngest(t, manager, ctx, collectionID, coldDocuments, map[string]string{conversationID: "fp-cold"})
	assertReindexCallCount(t, fake, 1)
	assertReindexCall(t, fake.reindexCallsSnapshot()[0], 2, semantic.Removal{
		Paths: append(
			conversationRemovalPathsForTest("conv-e2e", 0),
			conversationRemovalPathsForTest("conv-e2e", 1)...,
		),
		Prefixes: append(
			conversationRemovalPrefixesForTest("conv-e2e", 0),
			conversationRemovalPrefixesForTest("conv-e2e", 1)...,
		),
	})
	stateStore.setFromDocuments(prefix, coldDocuments)

	runConversationDeltaIngest(t, manager, ctx, collectionID, coldDocuments, map[string]string{conversationID: "fp-unchanged"})
	assertReindexCallCount(t, fake, 1)
	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	snapshot, err := merkle.ReadSnapshot(manager.merklePath(codebase.ID))
	if err != nil {
		t.Fatalf("ReadSnapshot returned error: %v", err)
	}
	if snapshot.Files[conversationID] != "fp-unchanged" {
		t.Fatalf("unchanged-message checkpoint = %q, want fp-unchanged", snapshot.Files[conversationID])
	}

	appendedDocuments := append([]model.ConversationDocument{}, coldDocuments...)
	appendedDocuments = append(appendedDocuments, model.ConversationDocument{
		ConversationID: conversationID,
		MessageIndex:   2,
		Role:           "user",
		Text:           "next question",
	})
	runConversationDeltaIngest(t, manager, ctx, collectionID, appendedDocuments, map[string]string{conversationID: "fp-appended"})
	assertReindexCallCount(t, fake, 2)
	assertReindexCall(t, fake.reindexCallsSnapshot()[1], 1, semantic.Removal{
		Paths:    conversationRemovalPathsForTest("conv-e2e", 2),
		Prefixes: conversationRemovalPrefixesForTest("conv-e2e", 2),
	})
	stateStore.setFromDocuments(prefix, appendedDocuments)

	editedDocuments := append([]model.ConversationDocument{}, appendedDocuments...)
	editedDocuments[1].Text = "edited answer"
	runConversationDeltaIngest(t, manager, ctx, collectionID, editedDocuments, map[string]string{conversationID: "fp-edited"})
	assertReindexCallCount(t, fake, 3)
	assertReindexCall(t, fake.reindexCallsSnapshot()[2], 1, semantic.Removal{
		Paths:    conversationRemovalPathsForTest("conv-e2e", 1),
		Prefixes: conversationRemovalPrefixesForTest("conv-e2e", 1),
	})
	stateStore.setFromDocuments(prefix, editedDocuments)

	staleDocuments := []model.ConversationDocument{editedDocuments[0], editedDocuments[1]}
	runConversationDeltaIngest(t, manager, ctx, collectionID, staleDocuments, map[string]string{conversationID: "fp-stale"})
	assertReindexCallCount(t, fake, 4)
	assertReindexCall(t, fake.reindexCallsSnapshot()[3], 0, semantic.Removal{
		Paths:    conversationRemovalPathsForTest("conv-e2e", 2),
		Prefixes: conversationRemovalPrefixesForTest("conv-e2e", 2),
	})
}

func TestConversationRPCsQueueJournaledJobs(t *testing.T) {
	t.Parallel()

	manager, cfg, _ := newTestManager(t)
	upsertedChunks := make(chan []model.StoredChunk, 1)
	deletedConversation := make(chan string, 1)
	manager.semantic = &fakeSemantic{
		reindex: func(ctx context.Context, codebasePath string, chunks []model.StoredChunk, removed []string) error {
			_ = ctx
			_ = codebasePath
			_ = removed
			if len(chunks) > 0 {
				select {
				case upsertedChunks <- append([]model.StoredChunk{}, chunks...):
				default:
				}
			}
			return nil
		},
		deleteConversation: func(ctx context.Context, collectionName string, conversationID string) error {
			_ = ctx
			_ = collectionName
			deletedConversation <- conversationID
			return nil
		},
	}
	server := NewGRPCServer(manager, nil)

	upsertStream := &fakeUpsertStreamServer{
		ClientStreamingServer: nil,
		chunks: []*pb.UpsertConversationDocumentsChunk{
			{Chunk: &pb.UpsertConversationDocumentsChunk_Header{Header: &pb.UpsertConversationDocumentsHeader{
				CollectionId: "thread-rpc-jobs",
				Client:       nil,
			}}},
			{Chunk: &pb.UpsertConversationDocumentsChunk_Documents{Documents: &pb.UpsertConversationDocumentsDocuments{
				Documents: []*pb.ConversationDocument{{
					ConversationId: "conv-rpc",
					MessageIndex:   0,
					Role:           "user",
					TimestampUnix:  1712345678,
					Text:           "hello",
				}},
			}}},
		},
		cursor:   0,
		response: nil,
	}
	if err := server.UpsertConversationDocumentsStream(upsertStream); err != nil {
		t.Fatalf("UpsertConversationDocumentsStream returned error: %v", err)
	}
	upsertResponse := upsertStream.response
	if upsertResponse == nil {
		t.Fatal("UpsertConversationDocumentsStream sent no response")
	}
	if upsertResponse.GetJobId() == "" {
		t.Fatal("UpsertConversationDocumentsStream returned an empty job id")
	}
	if !strings.Contains(upsertResponse.GetDisplayText(), "Started conversation ingest job") {
		t.Fatalf("upsert DisplayText = %q, want ingest start text", upsertResponse.GetDisplayText())
	}

	select {
	case chunks := <-upsertedChunks:
		if len(chunks) != 1 {
			t.Fatalf("upsert passed %d chunks, want 1", len(chunks))
		}
		if chunks[0].RelativePath != "conv/conv-rpc/0" {
			t.Fatalf("upsert chunk RelativePath = %q, want conv/conv-rpc/0", chunks[0].RelativePath)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("UpsertConversationDocumentsStream did not call semantic upsert")
	}
	waitForCondition(t, func() bool {
		job, found := manager.GetJob(upsertResponse.GetJobId())
		return found && job.State == model.JobStateCompleted
	})

	deleteResponse, err := server.DeleteConversation(context.Background(), &pb.DeleteConversationRequest{
		CollectionId:   "thread-rpc-jobs",
		ConversationId: "conv-rpc",
	})
	if err != nil {
		t.Fatalf("DeleteConversation returned error: %v", err)
	}
	if deleteResponse.GetJobId() == "" {
		t.Fatal("DeleteConversation returned an empty job id")
	}
	if !strings.Contains(deleteResponse.GetDisplayText(), "Started conversation delete job") {
		t.Fatalf("delete DisplayText = %q, want delete start text", deleteResponse.GetDisplayText())
	}

	select {
	case conversationID := <-deletedConversation:
		if conversationID != "conv-rpc" {
			t.Fatalf("deleted conversation id = %q, want conv-rpc", conversationID)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("DeleteConversation did not call semantic delete")
	}
	waitForCondition(t, func() bool {
		job, found := manager.GetJob(deleteResponse.GetJobId())
		return found && job.State == model.JobStateCompleted
	})

	jobs, err := store.ReadJobEvents(cfg.JobsPath)
	if err != nil {
		t.Fatalf("ReadJobEvents returned error: %v", err)
	}
	if jobs[upsertResponse.GetJobId()].Operation != string(jobOperationConversationIngest) {
		t.Fatalf("upsert job operation = %q, want %q", jobs[upsertResponse.GetJobId()].Operation, jobOperationConversationIngest)
	}
	if jobs[deleteResponse.GetJobId()].Operation != string(jobOperationConversationIngest) {
		t.Fatalf("delete job operation = %q, want %q", jobs[deleteResponse.GetJobId()].Operation, jobOperationConversationIngest)
	}
	if jobs[upsertResponse.GetJobId()].State != model.JobStateCompleted {
		t.Fatalf("upsert journal state = %q, want completed", jobs[upsertResponse.GetJobId()].State)
	}
	if jobs[deleteResponse.GetJobId()].State != model.JobStateCompleted {
		t.Fatalf("delete journal state = %q, want completed", jobs[deleteResponse.GetJobId()].State)
	}
}

func readConversationChunkCache(t *testing.T, manager *Manager, codebaseID string) []model.StoredChunk {
	t.Helper()

	chunks, err := store.ReadChunks(manager.chunkPath(codebaseID))
	if err != nil {
		t.Fatalf("ReadChunks returned error: %v", err)
	}
	return chunks
}

func testConversationManifest(conversationIDs ...string) map[string]string {
	manifest := make(map[string]string, len(conversationIDs))
	for _, conversationID := range conversationIDs {
		manifest[conversationID] = "fp-" + conversationID
	}
	return manifest
}

func upsertConversationsForManifest(t *testing.T, manager *Manager, ctx context.Context, collectionID string, manifest map[string]string, conversationIDs []string) {
	t.Helper()

	documents := make([]model.ConversationDocument, 0, len(conversationIDs))
	for _, conversationID := range conversationIDs {
		documents = append(documents, model.ConversationDocument{
			ConversationID: conversationID,
			MessageIndex:   0,
			Role:           "user",
			Text:           "body " + conversationID,
		})
	}
	job, err := manager.upsertConversationDocuments(ctx, collectionID, documents, manifest, testClientInfo(), absenceRetain)
	if err != nil {
		t.Fatalf("upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, job.ID, model.JobStateCompleted)
}

func assertStringSliceEqual(t *testing.T, got []string, want []string) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for index, gotValue := range got {
		if gotValue != want[index] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

func assertStringSliceContains(t *testing.T, values []string, want string) {
	t.Helper()

	for _, value := range values {
		if value == want {
			return
		}
	}
	t.Fatalf("values %v missing %q", values, want)
}

type testConversationRowReader struct {
	state map[int32]semantic.StoredMessageState
	reuse map[string][]float32
	err   error
	calls []messageStateCall
}

func (reader *testConversationRowReader) LoadConversationMessageState(_ context.Context, collectionName string, conversationPrefix string) (map[int32]semantic.StoredMessageState, map[string][]float32, error) {
	reader.calls = append(reader.calls, messageStateCall{Collection: collectionName, Prefix: conversationPrefix})
	if reader.err != nil {
		return nil, nil, reader.err
	}
	return cloneStoredMessageState(reader.state), cloneReuseVectors(reader.reuse), nil
}

func (reader *testConversationRowReader) callsSnapshot() []messageStateCall {
	return append([]messageStateCall(nil), reader.calls...)
}

type conversationDeltaChunkWant struct {
	relativePath string
	messageIndex int32
	role         string
	content      string
}

func assertConversationDeltaChunks(t *testing.T, got []model.StoredChunk, want []conversationDeltaChunkWant) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("chunks = %d, want %d: %+v", len(got), len(want), got)
	}
	for index, wantChunk := range want {
		gotChunk := got[index]
		if gotChunk.RelativePath != wantChunk.relativePath {
			t.Fatalf("chunk %d RelativePath = %q, want %q", index, gotChunk.RelativePath, wantChunk.relativePath)
		}
		if gotChunk.MessageIndex != wantChunk.messageIndex {
			t.Fatalf("chunk %d MessageIndex = %d, want %d", index, gotChunk.MessageIndex, wantChunk.messageIndex)
		}
		if gotChunk.Role != wantChunk.role {
			t.Fatalf("chunk %d Role = %q, want %q", index, gotChunk.Role, wantChunk.role)
		}
		if gotChunk.Content != wantChunk.content {
			t.Fatalf("chunk %d Content = %q, want %q", index, gotChunk.Content, wantChunk.content)
		}
	}
}

func assertMessageStateCalls(t *testing.T, got []messageStateCall, want []messageStateCall) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("message state calls = %+v, want %+v", got, want)
	}
	for index, wantCall := range want {
		gotCall := got[index]
		if gotCall != wantCall {
			t.Fatalf("message state calls = %+v, want %+v", got, want)
		}
	}
}

func cloneStoredMessageState(state map[int32]semantic.StoredMessageState) map[int32]semantic.StoredMessageState {
	copied := make(map[int32]semantic.StoredMessageState, len(state))
	for messageIndex, stored := range state {
		copied[messageIndex] = stored
	}
	return copied
}

type conversationStateStore struct {
	mu     sync.Mutex
	states map[string]map[int32]semantic.StoredMessageState
}

func newConversationStateStore() *conversationStateStore {
	return &conversationStateStore{states: map[string]map[int32]semantic.StoredMessageState{}}
}

func (store *conversationStateStore) state(prefix string) map[int32]semantic.StoredMessageState {
	store.mu.Lock()
	defer store.mu.Unlock()
	return cloneStoredMessageState(store.states[prefix])
}

func (store *conversationStateStore) setFromDocuments(prefix string, documents []model.ConversationDocument) {
	next := make(map[int32]semantic.StoredMessageState, len(documents))
	for _, document := range documents {
		next[document.MessageIndex] = semantic.StoredMessageState{Role: document.Role, Text: document.Text}
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	store.states[prefix] = next
}

func runConversationDeltaIngest(t *testing.T, manager *Manager, ctx context.Context, collectionID string, documents []model.ConversationDocument, manifest map[string]string) {
	t.Helper()

	job, err := manager.upsertConversationDocuments(ctx, collectionID, documents, manifest, testClientInfo(), absenceRetain)
	if err != nil {
		t.Fatalf("upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, job.ID, model.JobStateCompleted)
}

func assertReindexCallCount(t *testing.T, fake *fakeSemantic, want int) {
	t.Helper()

	calls := fake.reindexCallsSnapshot()
	if len(calls) != want {
		t.Fatalf("Reindex calls = %+v, want %d call(s)", calls, want)
	}
}

func assertReindexCall(t *testing.T, call reindexCall, wantChunks int, wantRemoval semantic.Removal) {
	t.Helper()

	if call.Chunks != wantChunks {
		t.Fatalf("Reindex chunks = %d, want %d", call.Chunks, wantChunks)
	}
	assertRemovalEqual(t, call.Removal, wantRemoval)
}

func waitForConversationJobState(t *testing.T, manager *Manager, jobID string, state model.JobState) {
	t.Helper()

	waitForCondition(t, func() bool {
		job, found := manager.GetJob(jobID)
		return found && job.State == state
	})
}

func stageConversationJob(t *testing.T, manager *Manager, codebase model.Codebase, payload conversationJobPayload) model.Job {
	t.Helper()

	job := model.Job{
		ID:            "job-" + string(payload.Kind) + "-" + codebase.ID,
		CodebaseID:    codebase.ID,
		CanonicalPath: codebase.CanonicalPath,
		RequestedPath: codebase.CanonicalPath,
		Operation:     string(jobOperationConversationIngest),
		State:         model.JobStateRunning,
		Config:        codebase.EffectiveConfig,
	}
	codebase.Status = model.CodebaseStatusIndexing
	codebase.ActiveJobID = job.ID

	manager.mu.Lock()
	defer manager.mu.Unlock()
	manager.jobs[job.ID] = job
	manager.codebases[codebase.ID] = codebase
	manager.conversationJobs[job.ID] = payload
	return job
}

func assertConversationJobCancelled(t *testing.T, manager *Manager, jobID string) {
	t.Helper()

	job, found := manager.GetJob(jobID)
	if !found {
		t.Fatal("conversation job not found")
	}
	if job.State != model.JobStateCancelled {
		t.Fatalf("conversation job state = %q, want %q", job.State, model.JobStateCancelled)
	}
	if job.Error != nil {
		t.Fatalf("conversation job error = %+v, want nil", job.Error)
	}
}

// TestSearchWithinConversationScopesAndReportsFingerprint proves the within
// search returns only the target conversation's rows and the checkpointed
// fingerprint for it, while an unknown conversation returns empty hits with an
// empty fingerprint, which is the typed not-indexed answer.
func TestSearchWithinConversationScopesAndReportsFingerprint(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	searchCalls := 0
	manager.semantic = &fakeSemantic{
		conversationSearch: func(_ context.Context, _ string, _ string, _ int32) ([]model.StoredChunk, error) {
			searchCalls++
			if searchCalls > 1 {
				return []model.StoredChunk{}, nil
			}
			return []model.StoredChunk{{
				Content:        "needle in alpha",
				RelativePath:   "conv/conv-a/0",
				ConversationID: "conv-a",
				MessageIndex:   0,
				Role:           "user",
				TimestampUnix:  1712345000,
				Score:          0.7,
			}}, nil
		},
	}
	ctx := context.Background()
	collectionID := "thread-within"

	job, err := manager.upsertConversationDocuments(ctx, collectionID, []model.ConversationDocument{
		{ConversationID: "conv-a", MessageIndex: 0, Role: "user", TimestampUnix: 1712345000, Text: "needle in alpha"},
		{ConversationID: "conv-b", MessageIndex: 0, Role: "user", TimestampUnix: 1712345001, Text: "needle in beta"},
	}, map[string]string{"conv-a": "fp-a-1", "conv-b": "fp-b-1"}, testClientInfo(), absenceRetain)
	if err != nil {
		t.Fatalf("upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, job.ID, model.JobStateCompleted)

	hits, indexedFingerprint, err := manager.SearchWithinConversation(ctx, collectionID, "conv-a", "needle", 5, emptyConversationSearchFilter())
	if err != nil {
		t.Fatalf("SearchWithinConversation returned error: %v", err)
	}
	if len(hits) != 1 || hits[0].ConversationID != "conv-a" {
		t.Fatalf("within hits = %+v, want only conv-a", hits)
	}
	if hits[0].Score <= 0 {
		t.Fatalf("within hit score = %v, want a positive semantic score", hits[0].Score)
	}
	if indexedFingerprint != "fp-a-1" {
		t.Fatalf("indexed fingerprint = %q, want fp-a-1", indexedFingerprint)
	}

	missingHits, missingFingerprint, err := manager.SearchWithinConversation(ctx, collectionID, "conv-unknown", "needle", 5, emptyConversationSearchFilter())
	if err != nil {
		t.Fatalf("SearchWithinConversation for unknown conversation returned error: %v", err)
	}
	if len(missingHits) != 0 {
		t.Fatalf("unknown conversation hits = %+v, want none", missingHits)
	}
	if missingFingerprint != "" {
		t.Fatalf("unknown conversation fingerprint = %q, want empty", missingFingerprint)
	}
}

// TestSearchWithinConversationPushesPrefixScope proves the within search hands
// the engine the conversation's conv/<id>/ prefix, so scoping happens in the
// vector store rather than by post-filtering an unscoped result list.
func TestSearchWithinConversationPushesNativeScope(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	fake := &fakeSemantic{
		conversationSearch: func(_ context.Context, _ string, _ string, _ int32) ([]model.StoredChunk, error) {
			return []model.StoredChunk{}, nil
		},
	}
	manager.semantic = fake
	ctx := context.Background()

	if _, _, err := manager.SearchWithinConversation(ctx, "thread-scope", "conv-scoped", "needle", 5, emptyConversationSearchFilter()); err != nil {
		t.Fatalf("SearchWithinConversation returned error: %v", err)
	}

	fake.mu.Lock()
	scopeCalls := append([][]string(nil), fake.conversationSearchScopes...)
	fake.mu.Unlock()
	if len(scopeCalls) != 1 {
		t.Fatalf("conversation searches = %d, want 1", len(scopeCalls))
	}
	if len(scopeCalls[0]) != 1 || scopeCalls[0][0] != "conv-scoped" {
		t.Fatalf("scope conversation ids = %v, want [conv-scoped]", scopeCalls[0])
	}
}

// TestSearchWithinConversationRPCBoundary proves the wire handler round-trips
// hits, scores, and the indexed fingerprint, and rejects a missing
// conversation id.
func TestSearchWithinConversationRPCBoundary(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{
		conversationSearch: func(_ context.Context, _ string, _ string, _ int32) ([]model.StoredChunk, error) {
			return []model.StoredChunk{{
				Content:        "needle on the wire",
				RelativePath:   "conv/conv-rpc/3",
				ConversationID: "conv-rpc",
				MessageIndex:   3,
				Role:           "assistant",
				TimestampUnix:  1712345002,
				Score:          0.9,
			}}, nil
		},
	}
	server := NewGRPCServer(manager, nil)
	ctx := context.Background()
	collectionID := "thread-within-rpc"

	job, err := manager.upsertConversationDocuments(ctx, collectionID, []model.ConversationDocument{
		{ConversationID: "conv-rpc", MessageIndex: 3, Role: "assistant", TimestampUnix: 1712345002, Text: "needle on the wire"},
	}, map[string]string{"conv-rpc": "fp-rpc-1"}, testClientInfo(), absenceRetain)
	if err != nil {
		t.Fatalf("upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, job.ID, model.JobStateCompleted)

	response, err := server.SearchWithinConversation(ctx, &pb.SearchWithinConversationRequest{
		CollectionId:   collectionID,
		ConversationId: "conv-rpc",
		Query:          "needle",
		Limit:          5,
	})
	if err != nil {
		t.Fatalf("SearchWithinConversation RPC returned error: %v", err)
	}
	if response.GetIndexedFingerprint() != "fp-rpc-1" {
		t.Fatalf("RPC indexed fingerprint = %q, want fp-rpc-1", response.GetIndexedFingerprint())
	}
	results := response.GetResults()
	if len(results) != 1 {
		t.Fatalf("RPC results = %d, want 1", len(results))
	}
	if results[0].GetConversationId() != "conv-rpc" || results[0].GetMessageIndex() != 3 || results[0].GetRole() != "assistant" {
		t.Fatalf("RPC result = %+v, want conv-rpc message 3 assistant", results[0])
	}
	if results[0].GetScore() <= 0 {
		t.Fatalf("RPC result score = %v, want positive", results[0].GetScore())
	}

	if _, err := server.SearchWithinConversation(ctx, &pb.SearchWithinConversationRequest{
		CollectionId: collectionID,
		Query:        "needle",
	}); err == nil {
		t.Fatal("RPC accepted a missing conversation_id, want an argument error")
	}
}

// TestConversationIngestLoadsReuseVectorsPerConversation proves the upsert path
// loads each delivered conversation's existing message state and vectors from
// the live collection, scoped to its conv/<id>/ prefix, and hands that exact
// reuse map to the reindex.
func TestConversationIngestLoadsReuseVectorsPerConversation(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	reuseByPrefix := map[string]map[string][]float32{
		"conv/conv-alpha/": {"hash-alpha": {0.1}},
		"conv/conv-beta/":  {"hash-beta": {0.2}},
	}
	fake := &fakeSemantic{
		loadMessageState: func(_ context.Context, _ string, prefix string) (map[int32]semantic.StoredMessageState, map[string][]float32, error) {
			return map[int32]semantic.StoredMessageState{}, reuseByPrefix[prefix], nil
		},
	}
	manager.semantic = fake
	ctx := context.Background()
	collectionID := "thread-reuse"
	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}

	job, err := manager.upsertConversationDocuments(ctx, collectionID, []model.ConversationDocument{
		{ConversationID: "conv-alpha", MessageIndex: 0, Role: "user", TimestampUnix: 1712345000, Text: "alpha"},
		{ConversationID: "conv-beta", MessageIndex: 0, Role: "user", TimestampUnix: 1712345001, Text: "beta"},
	}, map[string]string{"conv-alpha": "fp-a-1", "conv-beta": "fp-b-1"}, testClientInfo(), absenceRetain)
	if err != nil {
		t.Fatalf("upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, job.ID, model.JobStateCompleted)

	stateCalls := fake.messageStateCallsSnapshot()
	if len(stateCalls) != 2 {
		t.Fatalf("message state loads = %d, want 2 (one per delivered conversation): %+v", len(stateCalls), stateCalls)
	}
	seenPrefixes := map[string]bool{}
	for _, call := range stateCalls {
		if call.Collection != codebase.CollectionName {
			t.Fatalf("message state load collection = %q, want live collection %q", call.Collection, codebase.CollectionName)
		}
		seenPrefixes[call.Prefix] = true
	}
	if !seenPrefixes["conv/conv-alpha/"] || !seenPrefixes["conv/conv-beta/"] {
		t.Fatalf("message state load prefixes = %v, want conv/conv-alpha/ and conv/conv-beta/", seenPrefixes)
	}
	if prefixCalls := fake.reusePrefixCallsSnapshot(); len(prefixCalls) != 0 {
		t.Fatalf("separate reuse prefix loads = %+v, want none on the combined loader path", prefixCalls)
	}

	reuseByConversation := fake.reindexReuseSnapshot()
	alphaReuse, alphaSeen := reuseByConversation["conv-alpha"]
	if !alphaSeen || len(alphaReuse) != 1 || alphaReuse["hash-alpha"] == nil {
		t.Fatalf("conv-alpha reindex reuse = %v, want the conv/conv-alpha/ map", alphaReuse)
	}
	betaReuse, betaSeen := reuseByConversation["conv-beta"]
	if !betaSeen || len(betaReuse) != 1 || betaReuse["hash-beta"] == nil {
		t.Fatalf("conv-beta reindex reuse = %v, want the conv/conv-beta/ map", betaReuse)
	}
}

// TestHandleChangedFileProgressReflectsRealWork proves the per-item outcome
// reports progressed only when the item changed the working set: an embedded
// conversation progresses, while one whose documents were not delivered is
// skipped without progress, so the loop writes the checkpoint only after real
// work instead of once per skipped item.
func TestHandleChangedFileProgressReflectsRealWork(t *testing.T) {
	t.Parallel()

	manager, cfg, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{}
	source := newConversationItemSource(
		"conv_chunks_test",
		map[string]string{"conv-delivered": "fp-d-1", "conv-missing": "fp-m-1"},
		[]model.ConversationDocument{
			{ConversationID: "conv-delivered", MessageIndex: 0, Role: "user", TimestampUnix: 1712345003, Text: "delivered"},
		},
		nil,
		absenceRetain,
	)
	state := deltaState{
		plan:         deltaPlan{},
		snapshotPath: cfg.MerkleDir + "/checkpoint-test.json",
		working:      map[string]string{},
		source:       source,
		semantic:     true,
		staging:      false,
		reuse:        nil,
		chunkCounts:  &chunkCounters{reused: 0, embedded: 0},
	}
	result := indexer.Result{
		IndexedFiles:      0,
		TotalChunks:       0,
		Chunks:            []model.StoredChunk{},
		FileHashes:        nil,
		SkippedFiles:      []string{},
		SkippedOversize:   0,
		SkippedUnreadable: 0,
	}
	job := model.Job{ID: "job-checkpoint-test"}

	delivered := manager.handleChangedFile(context.Background(), job, state, "conv-delivered", &result)
	if delivered.fallback || delivered.handled {
		t.Fatalf("delivered outcome = %+v, want neither fallback nor handled", delivered)
	}
	if !delivered.progressed {
		t.Fatal("delivered conversation outcome progressed = false, want true")
	}

	skipped := manager.handleChangedFile(context.Background(), job, state, "conv-missing", &result)
	if skipped.fallback || skipped.handled {
		t.Fatalf("skipped outcome = %+v, want neither fallback nor handled", skipped)
	}
	if skipped.progressed {
		t.Fatal("undelivered conversation outcome progressed = true, want false (no checkpoint write)")
	}
}

func TestHandleChangedFileHonorsOneFileResultOverrides(t *testing.T) {
	t.Parallel()

	t.Run("reuse override skips item reuse load", func(t *testing.T) {
		t.Parallel()

		manager, cfg, _ := newTestManager(t)
		var reindexReuse map[string][]float32
		fake := &fakeSemantic{
			loadReuseForPrefix: func(context.Context, string, string) (map[string][]float32, error) {
				t.Fatal("LoadReuseVectorsForPrefix was called despite a OneFileResult reuse override")
				return nil, nil
			},
			reindexWithReuse: func(_ context.Context, _ string, chunks []model.StoredChunk, _ []string, progress func(semantic.Progress), reuse map[string][]float32) error {
				reindexReuse = cloneReuseVectors(reuse)
				if progress != nil {
					progress(semantic.Progress{ChunksProcessed: safeInt32(len(chunks)), ChunksReused: 1, ChunksEmbedded: 0})
				}
				return nil
			},
		}
		manager.semantic = fake
		state := overrideDeltaState(cfg.MerkleDir, oneFileResultOverrideSource{
			result: indexer.OneFileResult{
				Chunks: []model.StoredChunk{{
					Content:        "reuse override content",
					RelativePath:   "conv/conv-reuse/0",
					ConversationID: "conv-reuse",
					MessageIndex:   0,
				}},
				FileHash:     "fp-reuse",
				ReuseVectors: map[string][]float32{"override-hash": {2, 3}},
			},
			fallbackRemoval: semantic.RemovePrefixes([]string{"conv/conv-reuse/"}),
			reuse: itemReuseSource{
				CollectionName: "conv_chunks_test",
				RelativePath:   "conv/conv-reuse/",
				Scope:          itemReuseScopePrefix,
			},
		})
		state.reuse = map[string][]float32{"base-hash": {1}}
		state.itemReuseEnabled = true
		result := emptyOverrideResult()

		outcome := manager.handleChangedFile(context.Background(), model.Job{ID: "job-reuse-override"}, state, "conv-reuse", &result)

		if outcome.fallback || outcome.handled || !outcome.progressed {
			t.Fatalf("outcome = %+v, want progressed without fallback or handled", outcome)
		}
		if calls := fake.reusePrefixCallsSnapshot(); len(calls) != 0 {
			t.Fatalf("reuse prefix calls = %v, want none", calls)
		}
		assertReuseVector(t, reindexReuse, "base-hash", []float32{1})
		assertReuseVector(t, reindexReuse, "override-hash", []float32{2, 3})
		if state.chunkCounts.reuseVectorsLoaded != 1 {
			t.Fatalf("reuseVectorsLoaded = %d, want 1", state.chunkCounts.reuseVectorsLoaded)
		}
	})

	t.Run("removal override preserves paths and prefixes", func(t *testing.T) {
		t.Parallel()

		manager, cfg, _ := newTestManager(t)
		fake := &fakeSemantic{}
		manager.semantic = fake
		state := overrideDeltaState(cfg.MerkleDir, oneFileResultOverrideSource{
			result: indexer.OneFileResult{
				Chunks: []model.StoredChunk{{
					Content:        "removal override content",
					RelativePath:   "conv/conv-remove/0",
					ConversationID: "conv-remove",
					MessageIndex:   0,
				}},
				FileHash:        "fp-remove",
				RemovalOverride: true,
				RemovalPaths:    []string{"conv/conv-remove/legacy"},
				RemovalPrefixes: []string{"conv/conv-remove/messages/"},
			},
			fallbackRemoval: semantic.RemovePrefixes([]string{"conv/fallback/"}),
		})
		result := emptyOverrideResult()

		outcome := manager.handleChangedFile(context.Background(), model.Job{ID: "job-removal-override"}, state, "conv-remove", &result)

		if outcome.fallback || outcome.handled || !outcome.progressed {
			t.Fatalf("outcome = %+v, want progressed without fallback or handled", outcome)
		}
		calls := fake.reindexCallsSnapshot()
		if len(calls) != 1 {
			t.Fatalf("Reindex calls = %v, want exactly one", calls)
		}
		assertRemovalEqual(t, calls[0].Removal, semantic.Removal{
			Paths:    []string{"conv/conv-remove/legacy"},
			Prefixes: []string{"conv/conv-remove/messages/"},
		})
	})

	t.Run("removal override can reindex chunks with empty removal", func(t *testing.T) {
		t.Parallel()

		manager, cfg, _ := newTestManager(t)
		fake := &fakeSemantic{}
		manager.semantic = fake
		state := overrideDeltaState(cfg.MerkleDir, oneFileResultOverrideSource{
			result: indexer.OneFileResult{
				Chunks: []model.StoredChunk{{
					Content:        "empty removal content",
					RelativePath:   "conv/conv-empty-removal/0",
					ConversationID: "conv-empty-removal",
					MessageIndex:   0,
				}},
				FileHash:        "fp-empty-removal",
				RemovalOverride: true,
				RemovalPaths:    nil,
				RemovalPrefixes: nil,
			},
			fallbackRemoval: semantic.RemovePrefixes([]string{"conv/conv-empty-removal/"}),
		})
		result := emptyOverrideResult()

		outcome := manager.handleChangedFile(context.Background(), model.Job{ID: "job-empty-removal"}, state, "conv-empty-removal", &result)

		if outcome.fallback || outcome.handled || !outcome.progressed {
			t.Fatalf("outcome = %+v, want progressed without fallback or handled", outcome)
		}
		calls := fake.reindexCallsSnapshot()
		if len(calls) != 1 {
			t.Fatalf("Reindex calls = %v, want exactly one", calls)
		}
		if calls[0].Chunks != 1 {
			t.Fatalf("Reindex chunks = %d, want 1", calls[0].Chunks)
		}
		assertRemovalEqual(t, calls[0].Removal, semantic.Removal{})
	})

	t.Run("empty reuse override skips item reuse load", func(t *testing.T) {
		t.Parallel()

		manager, cfg, _ := newTestManager(t)
		fake := &fakeSemantic{
			loadReuseForPrefix: func(context.Context, string, string) (map[string][]float32, error) {
				t.Fatal("LoadReuseVectorsForPrefix was called despite an empty OneFileResult reuse override")
				return nil, nil
			},
		}
		manager.semantic = fake
		state := overrideDeltaState(cfg.MerkleDir, oneFileResultOverrideSource{
			result: indexer.OneFileResult{
				Chunks: []model.StoredChunk{{
					Content:        "empty reuse override content",
					RelativePath:   "conv/conv-empty-reuse/0",
					ConversationID: "conv-empty-reuse",
					MessageIndex:   0,
				}},
				FileHash:     "fp-empty-reuse",
				ReuseVectors: map[string][]float32{},
			},
			fallbackRemoval: semantic.RemovePrefixes([]string{"conv/conv-empty-reuse/"}),
			reuse: itemReuseSource{
				CollectionName: "conv_chunks_test",
				RelativePath:   "conv/conv-empty-reuse/",
				Scope:          itemReuseScopePrefix,
			},
		})
		state.itemReuseEnabled = true
		result := emptyOverrideResult()

		outcome := manager.handleChangedFile(context.Background(), model.Job{ID: "job-empty-reuse"}, state, "conv-empty-reuse", &result)

		if outcome.fallback || outcome.handled || !outcome.progressed {
			t.Fatalf("outcome = %+v, want progressed without fallback or handled", outcome)
		}
		if calls := fake.reusePrefixCallsSnapshot(); len(calls) != 0 {
			t.Fatalf("reuse prefix calls = %v, want none", calls)
		}
	})

	t.Run("empty chunks and empty removal progress without reindex", func(t *testing.T) {
		t.Parallel()

		manager, cfg, _ := newTestManager(t)
		fake := &fakeSemantic{}
		manager.semantic = fake
		state := overrideDeltaState(cfg.MerkleDir, oneFileResultOverrideSource{
			result: indexer.OneFileResult{
				Chunks:          nil,
				FileHash:        "fp-zero",
				RemovalOverride: true,
				RemovalPaths:    nil,
				RemovalPrefixes: nil,
			},
			fallbackRemoval: semantic.RemovePrefixes([]string{"conv/conv-zero/"}),
		})
		result := emptyOverrideResult()

		outcome := manager.handleChangedFile(context.Background(), model.Job{ID: "job-zero-override"}, state, "conv-zero", &result)

		if outcome.fallback || outcome.handled || !outcome.progressed {
			t.Fatalf("outcome = %+v, want progressed without fallback or handled", outcome)
		}
		if calls := fake.reindexCallsSnapshot(); len(calls) != 0 {
			t.Fatalf("Reindex calls = %v, want none", calls)
		}
		if state.working["conv-zero"] != "fp-zero" {
			t.Fatalf("working fingerprint = %q, want fp-zero", state.working["conv-zero"])
		}
		if result.IndexedFiles != 1 {
			t.Fatalf("IndexedFiles = %d, want 1", result.IndexedFiles)
		}
	})
}

func TestHandleRemovedFileSkipsSemanticForEmptyRemoval(t *testing.T) {
	t.Parallel()

	manager, cfg, _ := newTestManager(t)
	fake := &fakeSemantic{}
	manager.semantic = fake
	state := overrideDeltaState(cfg.MerkleDir, oneFileResultOverrideSource{
		fallbackRemoval: semantic.Removal{},
	})
	state.working["conv-empty-delete"] = "fp-empty-delete"

	outcome := manager.handleRemovedFile(context.Background(), model.Job{ID: "job-empty-delete"}, state, "conv-empty-delete", indexer.OneFileResult{Removed: true})

	if outcome.fallback || outcome.handled || !outcome.progressed {
		t.Fatalf("outcome = %+v, want progressed without fallback or handled", outcome)
	}
	if calls := fake.reindexCallsSnapshot(); len(calls) != 0 {
		t.Fatalf("Reindex calls = %v, want none for empty removal", calls)
	}
	if _, present := state.working["conv-empty-delete"]; present {
		t.Fatalf("working still contains removed item")
	}
}

type oneFileResultOverrideSource struct {
	result          indexer.OneFileResult
	fallbackRemoval semantic.Removal
	reuse           itemReuseSource
}

func (source oneFileResultOverrideSource) capture(context.Context) (merkle.Snapshot, error) {
	return merkle.Snapshot{}, nil
}

func (source oneFileResultOverrideSource) indexOne(context.Context, string) (indexer.OneFileResult, error) {
	return source.result, nil
}

func (source oneFileResultOverrideSource) removalFor([]string) semantic.Removal {
	return source.fallbackRemoval
}

func (source oneFileResultOverrideSource) absencePolicy() absencePolicy {
	return absenceRetain
}

func (source oneFileResultOverrideSource) reuseSource(string) itemReuseSource {
	return source.reuse
}

func (source oneFileResultOverrideSource) unit() string {
	return "document"
}

func overrideDeltaState(merkleDir string, source oneFileResultOverrideSource) deltaState {
	return deltaState{
		plan:         deltaPlan{},
		snapshotPath: merkleDir + "/override-checkpoint-test.json",
		working:      map[string]string{},
		source:       source,
		semantic:     true,
		staging:      false,
		reuse:        nil,
		chunkCounts:  &chunkCounters{},
	}
}

func emptyOverrideResult() indexer.Result {
	return indexer.Result{
		IndexedFiles:      0,
		TotalChunks:       0,
		Chunks:            []model.StoredChunk{},
		FileHashes:        nil,
		SkippedFiles:      []string{},
		SkippedOversize:   0,
		SkippedUnreadable: 0,
	}
}

func assertReuseVector(t *testing.T, reuse map[string][]float32, key string, want []float32) {
	t.Helper()

	got, present := reuse[key]
	if !present {
		t.Fatalf("reuse missing key %q in %v", key, reuse)
	}
	if len(got) != len(want) {
		t.Fatalf("reuse[%q] = %v, want %v", key, got, want)
	}
	for index, gotValue := range got {
		if gotValue != want[index] {
			t.Fatalf("reuse[%q] = %v, want %v", key, got, want)
		}
	}
}

func assertRemovalEqual(t *testing.T, got semantic.Removal, want semantic.Removal) {
	t.Helper()

	assertStringSliceEqual(t, got.Paths, want.Paths)
	assertStringSliceEqual(t, got.Prefixes, want.Prefixes)
}

func conversationRemovalPathsForTest(conversationID string, messageIndex int32) []string {
	messageSegment := strconv.Itoa(int(messageIndex))
	return []string{
		conversationRelativePath(conversationID, messageIndex, 0, false),
		"convtool/" + conversationID + "/" + messageSegment,
		"convthink/" + conversationID + "/" + messageSegment,
	}
}

func conversationRemovalPrefixesForTest(conversationID string, messageIndex int32) []string {
	messageSegment := strconv.Itoa(int(messageIndex))
	return []string{
		conversationRelativePath(conversationID, messageIndex, 0, false) + "/",
		"convtool/" + conversationID + "/" + messageSegment + "/",
		"convthink/" + conversationID + "/" + messageSegment + "/",
	}
}

func findConversationChunkForTest(t *testing.T, chunks []model.StoredChunk, relativePath string) model.StoredChunk {
	t.Helper()
	for _, chunk := range chunks {
		if chunk.RelativePath == relativePath {
			return chunk
		}
	}
	t.Fatalf("chunk %q not found in %+v", relativePath, chunks)
	return model.StoredChunk{}
}

func findConversationChunkWithPrefixForTest(t *testing.T, chunks []model.StoredChunk, prefix string) model.StoredChunk {
	t.Helper()
	for _, chunk := range chunks {
		if strings.HasPrefix(chunk.RelativePath, prefix) {
			return chunk
		}
	}
	t.Fatalf("chunk prefix %q not found in %+v", prefix, chunks)
	return model.StoredChunk{}
}

func reconstructConversationTextRowsForTest(chunks []model.StoredChunk, conversationID string, messageIndex int32) string {
	var builder strings.Builder
	for _, chunk := range conversationTextRowsForTest(chunks, conversationID, messageIndex) {
		builder.WriteString(chunk.Content)
	}
	return builder.String()
}

func conversationTextRowSignaturesForTest(chunks []model.StoredChunk, conversationID string, messageIndex int32) []string {
	textRows := conversationTextRowsForTest(chunks, conversationID, messageIndex)
	signatures := make([]string, 0, len(textRows))
	for _, chunk := range textRows {
		signatures = append(signatures, chunk.RelativePath+"\x00"+chunk.Content)
	}
	return signatures
}

func conversationTextRowsForTest(chunks []model.StoredChunk, conversationID string, messageIndex int32) []model.StoredChunk {
	exactPath := conversationRelativePath(conversationID, messageIndex, 0, false)
	partPrefix := exactPath + "/"
	textRows := make([]model.StoredChunk, 0)
	for _, chunk := range chunks {
		if chunk.RelativePath == exactPath || strings.HasPrefix(chunk.RelativePath, partPrefix) {
			textRows = append(textRows, chunk)
		}
	}
	return textRows
}

// TestConversationIngestReuseLoadFailureFallsBackToFullEmbed proves a failed
// message state and reuse load does not fail the job: the conversation falls
// back to a full prefix reindex with nil override reuse, and the per-item reuse
// load can fail independently without failing the job.
func TestConversationIngestReuseLoadFailureFallsBackToFullEmbed(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	fake := &fakeSemantic{
		loadMessageState: func(_ context.Context, _ string, _ string) (map[int32]semantic.StoredMessageState, map[string][]float32, error) {
			return nil, nil, errors.New("message state read failed")
		},
		loadReuseForPrefix: func(_ context.Context, _ string, _ string) (map[string][]float32, error) {
			return nil, errors.New("milvus read failed")
		},
	}
	manager.semantic = fake
	ctx := context.Background()
	collectionID := "thread-reuse-fallback"

	job, err := manager.upsertConversationDocuments(ctx, collectionID, []model.ConversationDocument{
		{ConversationID: "conv-solo", MessageIndex: 0, Role: "user", TimestampUnix: 1712345002, Text: "solo"},
	}, map[string]string{"conv-solo": "fp-s-1"}, testClientInfo(), absenceRetain)
	if err != nil {
		t.Fatalf("upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, job.ID, model.JobStateCompleted)

	reuseByConversation := fake.reindexReuseSnapshot()
	soloReuse, soloSeen := reuseByConversation["conv-solo"]
	if !soloSeen {
		t.Fatal("conv-solo was not reindexed after reuse load failure")
	}
	if len(soloReuse) != 0 {
		t.Fatalf("conv-solo reindex reuse = %v, want empty after load failure", soloReuse)
	}
	stateCalls := fake.messageStateCallsSnapshot()
	if len(stateCalls) != 1 || stateCalls[0].Prefix != "conv/conv-solo/" {
		t.Fatalf("message state loads = %+v, want one conv/conv-solo/ load", stateCalls)
	}
	prefixCalls := fake.reusePrefixCallsSnapshot()
	if len(prefixCalls) != 1 || prefixCalls[0].Prefix != "conv/conv-solo/" {
		t.Fatalf("fallback reuse prefix loads = %+v, want one conv/conv-solo/ load", prefixCalls)
	}
}

// TestItemSourceAbsencePolicy proves a code source always deletes a file the
// walk no longer finds, while a conversation source honors the caller-declared
// policy: it retains by default and deletes only when built AUTHORITATIVE.
func TestItemSourceAbsencePolicy(t *testing.T) {
	t.Parallel()

	code := newCodeItemSource(fakeRunner{}, nil, "cb", "/repo", defaultIndexConfig())
	if code.absencePolicy() != absenceDeleteGuarded {
		t.Fatalf("code absencePolicy = %v, want absenceDeleteGuarded", code.absencePolicy())
	}
	retain := newConversationItemSource("conv_chunks_test", map[string]string{}, nil, nil, absenceRetain)
	if retain.absencePolicy() != absenceRetain {
		t.Fatalf("retain conversation absencePolicy = %v, want absenceRetain", retain.absencePolicy())
	}
	authoritative := newConversationItemSource("conv_chunks_test", map[string]string{}, nil, nil, absenceDeleteGuarded)
	if authoritative.absencePolicy() != absenceDeleteGuarded {
		t.Fatalf("authoritative conversation absencePolicy = %v, want absenceDeleteGuarded", authoritative.absencePolicy())
	}
}

func TestConversationDocumentsToStoredChunksEmbedsBashToolTokens(t *testing.T) {
	t.Parallel()

	chunks, err := conversationDocumentsToStoredChunks(context.Background(), []model.ConversationDocument{{
		ConversationID:       "conv-tool",
		ParentConversationID: "parent-tool",
		MessageIndex:         3,
		Role:                 "assistant",
		TimestampUnix:        1712345700,
		Text:                 "ran a command",
		WorkspaceRoot:        "/workspace",
		Archived:             true,
		Tools: []model.ConversationToolCall{{
			Name:     "Bash",
			Command:  "cat /tmp/input.txt > /tmp/output.txt",
			LangHint: "bash",
		}},
	}})
	if err != nil {
		t.Fatalf("conversationDocumentsToStoredChunks returned error: %v", err)
	}

	tokenChunk := findConversationChunkForTest(t, chunks, "convtool/conv-tool/3/0/tok")
	for _, expected := range []string{"Bash", "cat", "/tmp/input.txt", "/tmp/output.txt"} {
		if !strings.Contains(tokenChunk.Content, expected) {
			t.Fatalf("token chunk content = %q, want to contain %q", tokenChunk.Content, expected)
		}
	}
	if tokenChunk.ParentConversationID != "parent-tool" {
		t.Fatalf("ParentConversationID = %q, want parent-tool", tokenChunk.ParentConversationID)
	}
	if tokenChunk.WorkspaceRoot != "/workspace" {
		t.Fatalf("WorkspaceRoot = %q, want /workspace", tokenChunk.WorkspaceRoot)
	}
	if !tokenChunk.Archived {
		t.Fatal("Archived = false, want true")
	}

	commandChunk := findConversationChunkForTest(t, chunks, "convtool/conv-tool/3/0/cmd")
	if commandChunk.Content != "cat /tmp/input.txt > /tmp/output.txt" {
		t.Fatalf("command chunk content = %q, want raw command", commandChunk.Content)
	}
}

func TestConversationDocumentsToStoredChunksSplitsJSONToolInput(t *testing.T) {
	t.Parallel()

	chunks, err := conversationDocumentsToStoredChunks(context.Background(), []model.ConversationDocument{{
		ConversationID: "conv-json",
		MessageIndex:   4,
		Role:           "assistant",
		Text:           "read input",
		Tools: []model.ConversationToolCall{{
			Name:      "Read",
			InputJSON: `{"path":"/tmp/input.json","limit":5}`,
			LangHint:  "json",
		}},
	}})
	if err != nil {
		t.Fatalf("conversationDocumentsToStoredChunks returned error: %v", err)
	}

	inputChunk := findConversationChunkWithPrefixForTest(t, chunks, "convtool/conv-json/4/0/in/")
	if !strings.Contains(inputChunk.Content, "/tmp/input.json") {
		t.Fatalf("input chunk content = %q, want JSON input", inputChunk.Content)
	}
}

func TestConversationDocumentsToStoredChunksKeepsTextDeltaStable(t *testing.T) {
	t.Parallel()

	document := model.ConversationDocument{
		ConversationID: "conv-stable",
		MessageIndex:   7,
		Role:           "assistant",
		Text:           "visible transcript text",
		Tools: []model.ConversationToolCall{{
			Name:      "Read",
			InputJSON: `{"file":"/tmp/private.json"}`,
			LangHint:  "json",
		}},
		Thinking: "private reasoning",
	}
	firstChunks, err := conversationDocumentsToStoredChunks(context.Background(), []model.ConversationDocument{document})
	if err != nil {
		t.Fatalf("first conversationDocumentsToStoredChunks returned error: %v", err)
	}
	secondChunks, err := conversationDocumentsToStoredChunks(context.Background(), []model.ConversationDocument{document})
	if err != nil {
		t.Fatalf("second conversationDocumentsToStoredChunks returned error: %v", err)
	}

	reconstructed := reconstructConversationTextRowsForTest(firstChunks, document.ConversationID, document.MessageIndex)
	if reconstructed != document.Text {
		t.Fatalf("reconstructed text = %q, want %q", reconstructed, document.Text)
	}
	assertStringSliceEqual(
		t,
		conversationTextRowSignaturesForTest(firstChunks, document.ConversationID, document.MessageIndex),
		conversationTextRowSignaturesForTest(secondChunks, document.ConversationID, document.MessageIndex),
	)
}

func TestConversationDocumentsToStoredChunksEmbedsThinking(t *testing.T) {
	t.Parallel()

	chunks, err := conversationDocumentsToStoredChunks(context.Background(), []model.ConversationDocument{{
		ConversationID: "conv-think",
		MessageIndex:   2,
		Role:           "assistant",
		Text:           "answer",
		Thinking:       "private reasoning",
	}})
	if err != nil {
		t.Fatalf("conversationDocumentsToStoredChunks returned error: %v", err)
	}

	thinkingChunk := findConversationChunkForTest(t, chunks, "convthink/conv-think/2")
	if thinkingChunk.Content != "private reasoning" {
		t.Fatalf("thinking chunk content = %q, want private reasoning", thinkingChunk.Content)
	}
}

func TestConversationDocumentsToStoredChunksEmbedsToolOnlyTokenChunk(t *testing.T) {
	t.Parallel()

	chunks, err := conversationDocumentsToStoredChunks(context.Background(), []model.ConversationDocument{{
		ConversationID: "conv-tool-only",
		MessageIndex:   0,
		Role:           "assistant",
		Text:           "",
		Tools: []model.ConversationToolCall{{
			Name:      "Read",
			InputJSON: `{"file":"/tmp/tool-only.txt"}`,
			LangHint:  "json",
		}},
	}})
	if err != nil {
		t.Fatalf("conversationDocumentsToStoredChunks returned error: %v", err)
	}

	tokenChunk := findConversationChunkForTest(t, chunks, "convtool/conv-tool-only/0/0/tok")
	if !strings.Contains(tokenChunk.Content, "Read") {
		t.Fatalf("token chunk content = %q, want tool name", tokenChunk.Content)
	}
}

// TestConversationIngestRetainsConversationsAbsentFromManifest proves the
// retain-on-absence policy: a conversation missing from a later push is kept,
// not deleted. Its id stays in the snapshot, a restoring push is a no-op, and
// an explicit delete completes through the vector store. This guards the index
// against a transient mass disappearance of transcript files.
func TestConversationIngestRetainsConversationsAbsentFromManifest(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{}
	ctx := context.Background()
	collectionID := "thread-retain"

	fullManifest := map[string]string{
		"conv-0": "fp-0",
		"conv-1": "fp-1",
		"conv-2": "fp-2",
		"conv-3": "fp-3",
		"conv-4": "fp-4",
	}
	fullDocuments := []model.ConversationDocument{
		{ConversationID: "conv-0", MessageIndex: 0, Role: "user", TimestampUnix: 1712345000, Text: "zero"},
		{ConversationID: "conv-1", MessageIndex: 0, Role: "user", TimestampUnix: 1712345001, Text: "one"},
		{ConversationID: "conv-2", MessageIndex: 0, Role: "user", TimestampUnix: 1712345002, Text: "two"},
		{ConversationID: "conv-3", MessageIndex: 0, Role: "user", TimestampUnix: 1712345003, Text: "three"},
		{ConversationID: "conv-4", MessageIndex: 0, Role: "user", TimestampUnix: 1712345004, Text: "four"},
	}
	firstJob, err := manager.upsertConversationDocuments(ctx, collectionID, fullDocuments, fullManifest, testClientInfo(), absenceRetain)
	if err != nil {
		t.Fatalf("first upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, firstJob.ID, model.JobStateCompleted)

	// The second push omits conv-2, conv-3, and conv-4 and delivers no documents.
	// Retain-on-absence keeps them: no removal runs and the snapshot still lists
	// the omitted ids.
	reducedManifest := map[string]string{"conv-0": "fp-0", "conv-1": "fp-1"}
	secondJob, err := manager.upsertConversationDocuments(ctx, collectionID, nil, reducedManifest, testClientInfo(), absenceRetain)
	if err != nil {
		t.Fatalf("second upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, secondJob.ID, model.JobStateCompleted)

	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	for _, conversationID := range []string{"conv-2", "conv-3", "conv-4"} {
		snapshot, err := merkle.ReadSnapshot(manager.merklePath(codebase.ID))
		if err != nil {
			t.Fatalf("ReadSnapshot returned error: %v", err)
		}
		if _, present := snapshot.Files[conversationID]; !present {
			t.Fatalf("snapshot dropped %s on absence; have %v", conversationID, snapshot.Files)
		}
	}

	// A restoring push sends the full manifest again with no documents. The ids and
	// fingerprints already match, so nothing re-embeds and the cache is unchanged.
	thirdJob, err := manager.upsertConversationDocuments(ctx, collectionID, nil, fullManifest, testClientInfo(), absenceRetain)
	if err != nil {
		t.Fatalf("third upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, thirdJob.ID, model.JobStateCompleted)

	// An explicit single-conversation delete still completes through semantic
	// storage; the manifest checkpoint is converged by later manifest syncs.
	deleteJob, err := manager.DeleteConversation(ctx, collectionID, "conv-2")
	if err != nil {
		t.Fatalf("DeleteConversation returned error: %v", err)
	}
	waitForConversationJobState(t, manager, deleteJob.ID, model.JobStateCompleted)
}

// TestConversationIngestDeletesConversationsAbsentUnderAuthoritative proves the
// wire reconcile mode drives behavior: an upsert built AUTHORITATIVE removes a
// conversation the manifest omits, the mirror of the retain test. This is the
// path a caller opts into by sending CONVERSATION_RECONCILE_MODE_AUTHORITATIVE.
func TestConversationIngestDeletesConversationsAbsentUnderAuthoritative(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{}
	ctx := context.Background()
	collectionID := "thread-authoritative"

	fullManifest := map[string]string{
		"conv-0": "fp-0",
		"conv-1": "fp-1",
		"conv-2": "fp-2",
	}
	fullDocuments := []model.ConversationDocument{
		{ConversationID: "conv-0", MessageIndex: 0, Role: "user", TimestampUnix: 1712345000, Text: "zero"},
		{ConversationID: "conv-1", MessageIndex: 0, Role: "user", TimestampUnix: 1712345001, Text: "one"},
		{ConversationID: "conv-2", MessageIndex: 0, Role: "user", TimestampUnix: 1712345002, Text: "two"},
	}
	firstJob, err := manager.upsertConversationDocuments(ctx, collectionID, fullDocuments, fullManifest, testClientInfo(), absenceDeleteGuarded)
	if err != nil {
		t.Fatalf("first upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, firstJob.ID, model.JobStateCompleted)

	// The second push omits conv-2 and delivers no documents. AUTHORITATIVE
	// deletes it, so the checkpoint snapshot no longer lists it.
	reducedManifest := map[string]string{"conv-0": "fp-0", "conv-1": "fp-1"}
	secondJob, err := manager.upsertConversationDocuments(ctx, collectionID, nil, reducedManifest, testClientInfo(), absenceDeleteGuarded)
	if err != nil {
		t.Fatalf("second upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, secondJob.ID, model.JobStateCompleted)

	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	snapshot, err := merkle.ReadSnapshot(manager.merklePath(codebase.ID))
	if err != nil {
		t.Fatalf("ReadSnapshot returned error: %v", err)
	}
	if _, present := snapshot.Files["conv-2"]; present {
		t.Fatalf("AUTHORITATIVE upsert retained conv-2 on absence; snapshot = %v", snapshot.Files)
	}
	for _, conversationID := range []string{"conv-0", "conv-1"} {
		if _, present := snapshot.Files[conversationID]; !present {
			t.Fatalf("AUTHORITATIVE upsert dropped present %s; snapshot = %v", conversationID, snapshot.Files)
		}
	}
}

// TestUpsertConversationDocumentsRejectsAuthoritativeWithoutManifest proves the
// missing-manifest guard: an authoritative upsert with a nil manifest is rejected
// rather than deriving the manifest from only the delivered documents, which would
// treat every other indexed conversation as absent and delete it. The same
// nil-manifest upsert under retain is still allowed and derives the manifest.
func TestUpsertConversationDocumentsRejectsAuthoritativeWithoutManifest(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{}
	ctx := context.Background()

	documents := []model.ConversationDocument{
		{ConversationID: "conv-x", MessageIndex: 0, Role: "user", TimestampUnix: 1712345000, Text: "x"},
	}

	if _, err := manager.upsertConversationDocuments(ctx, "authoritative-no-manifest", documents, nil, testClientInfo(), absenceDeleteGuarded); err == nil {
		t.Fatal("authoritative upsert with nil manifest was accepted; want rejection to avoid a derived-manifest mass delete")
	}

	retainJob, err := manager.upsertConversationDocuments(ctx, "retain-no-manifest", documents, nil, testClientInfo(), absenceRetain)
	if err != nil {
		t.Fatalf("retain upsert with nil manifest was rejected: %v", err)
	}
	waitForConversationJobState(t, manager, retainJob.ID, model.JobStateCompleted)
}

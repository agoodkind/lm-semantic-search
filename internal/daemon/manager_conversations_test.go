package daemon

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/store"
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
	manager.semantic = &fakeSemantic{unavailable: true}
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
	}, map[string]string{"conv-a": "fp-a-1", "conv-b": "fp-b-1"}, testClientInfo())
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

	chunks, err := manager.SearchConversations(context.Background(), collectionID, "needle", 5)
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
	unregisteredChunks, err := manager.SearchConversations(context.Background(), "missing-thread", "needle", 5)
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

func TestConversationIngestMaintainsChunkCache(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{unavailable: true}
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
	}, firstManifest, testClientInfo())
	if err != nil {
		t.Fatalf("upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, firstJob.ID, model.JobStateCompleted)

	chunks := readConversationChunkCache(t, manager, codebase.ID)
	if len(chunks) != 2 {
		t.Fatalf("first cache write stored %d chunks, want 2", len(chunks))
	}

	// conv-beta is unchanged, so its fingerprint stays and clyde sends no
	// document for it. conv-alpha changed, so only its document is delivered. The
	// merge must keep conv-beta's prior chunk and replace conv-alpha's.
	secondManifest := map[string]string{"conv-alpha": "alpha-2", "conv-beta": "beta-1"}
	secondJob, err := manager.upsertConversationDocuments(ctx, collectionID, []model.ConversationDocument{{
		ConversationID: "conv-alpha",
		MessageIndex:   1,
		Role:           "assistant",
		TimestampUnix:  1712345002,
		Text:           "fresh needle cache entry",
	}}, secondManifest, testClientInfo())
	if err != nil {
		t.Fatalf("second upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, secondJob.ID, model.JobStateCompleted)

	chunks = readConversationChunkCache(t, manager, codebase.ID)
	if len(chunks) != 2 {
		t.Fatalf("re-upsert cache stored %d chunks, want 2", len(chunks))
	}
	alphaCount := 0
	for _, chunk := range chunks {
		if strings.Contains(chunk.Content, "old needle") {
			t.Fatalf("re-upsert left stale chunk content %q", chunk.Content)
		}
		if strings.HasPrefix(chunk.RelativePath, "conv/conv-alpha/") {
			alphaCount++
			if chunk.Content != "fresh needle cache entry" {
				t.Fatalf("conv-alpha cached content = %q, want fresh needle cache entry", chunk.Content)
			}
		}
	}
	if alphaCount != 1 {
		t.Fatalf("re-upsert stored %d conv-alpha chunks, want 1", alphaCount)
	}

	results, err := manager.SearchConversations(ctx, collectionID, "fresh needle", 5)
	if err != nil {
		t.Fatalf("SearchConversations returned error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("fallback SearchConversations returned %d chunks, want 1", len(results))
	}
	if results[0].ConversationID != "conv-alpha" {
		t.Fatalf("fallback result ConversationID = %q, want conv-alpha", results[0].ConversationID)
	}

	deleteJob, err := manager.DeleteConversation(ctx, collectionID, "conv-alpha")
	if err != nil {
		t.Fatalf("DeleteConversation returned error: %v", err)
	}
	waitForConversationJobState(t, manager, deleteJob.ID, model.JobStateCompleted)

	chunks = readConversationChunkCache(t, manager, codebase.ID)
	if len(chunks) != 1 {
		t.Fatalf("delete cache stored %d chunks, want 1", len(chunks))
	}
	if chunks[0].ConversationID != "conv-beta" {
		t.Fatalf("remaining cached ConversationID = %q, want conv-beta", chunks[0].ConversationID)
	}
}

func TestSearchConversationsFallsBackToChunkCacheAfterSemanticError(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	collectionID := "thread-cache-error"
	codebase, err := manager.RegisterConversationCollection(context.Background(), collectionID)
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	searchCalls := 0
	manager.semantic = &fakeSemantic{
		conversationSearch: func(ctx context.Context, collectionName string, query string, limit int32) ([]model.StoredChunk, error) {
			_ = ctx
			_ = collectionName
			_ = query
			_ = limit
			searchCalls++
			return nil, errors.New("semantic search unavailable")
		},
	}
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

	results, err := manager.SearchConversations(context.Background(), collectionID, "needle", 5)
	if err != nil {
		t.Fatalf("SearchConversations returned error: %v", err)
	}
	if searchCalls != 1 {
		t.Fatalf("semantic search calls = %d, want 1", searchCalls)
	}
	if len(results) != 1 {
		t.Fatalf("fallback SearchConversations returned %d chunks, want 1", len(results))
	}
	if results[0].ConversationID != "conv-cache" {
		t.Fatalf("fallback result ConversationID = %q, want conv-cache", results[0].ConversationID)
	}
}

func TestConversationDocumentsToStoredChunksSplitsOversizedMessage(t *testing.T) {
	t.Parallel()

	text := strings.Repeat("a", conversationChunkMaxBytes+5)
	chunks, err := conversationDocumentsToStoredChunks([]model.ConversationDocument{{
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

	upsertResponse, err := server.UpsertConversationDocuments(context.Background(), &pb.UpsertConversationDocumentsRequest{
		CollectionId: "thread-rpc-jobs",
		Documents: []*pb.ConversationDocument{{
			ConversationId: "conv-rpc",
			MessageIndex:   0,
			Role:           "user",
			TimestampUnix:  1712345678,
			Text:           "hello",
		}},
	})
	if err != nil {
		t.Fatalf("UpsertConversationDocuments returned error: %v", err)
	}
	if upsertResponse.GetJobId() == "" {
		t.Fatal("UpsertConversationDocuments returned an empty job id")
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
		t.Fatal("UpsertConversationDocuments did not call semantic upsert")
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

func waitForConversationJobState(t *testing.T, manager *Manager, jobID string, state model.JobState) {
	t.Helper()

	waitForCondition(t, func() bool {
		job, found := manager.GetJob(jobID)
		return found && job.State == state
	})
}

// TestConversationIngestLoadsReuseVectorsPerConversation proves the upsert path
// loads each delivered conversation's existing vectors from the live collection,
// scoped to its conv/<id>/ prefix, and hands that exact map to the reindex, so
// unchanged chunks take their stored vector instead of the embedder.
func TestConversationIngestLoadsReuseVectorsPerConversation(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	reuseByPrefix := map[string]map[string][]float32{
		"conv/conv-alpha/": {"hash-alpha": {0.1}},
		"conv/conv-beta/":  {"hash-beta": {0.2}},
	}
	fake := &fakeSemantic{
		loadReuseForPrefix: func(_ context.Context, _ string, prefix string) (map[string][]float32, error) {
			return reuseByPrefix[prefix], nil
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
	}, map[string]string{"conv-alpha": "fp-a-1", "conv-beta": "fp-b-1"}, testClientInfo())
	if err != nil {
		t.Fatalf("upsertConversationDocuments returned error: %v", err)
	}
	waitForConversationJobState(t, manager, job.ID, model.JobStateCompleted)

	prefixCalls := fake.reusePrefixCallsSnapshot()
	if len(prefixCalls) != 2 {
		t.Fatalf("reuse prefix loads = %d, want 2 (one per delivered conversation): %+v", len(prefixCalls), prefixCalls)
	}
	seenPrefixes := map[string]bool{}
	for _, call := range prefixCalls {
		if call.Collection != codebase.CollectionName {
			t.Fatalf("reuse load collection = %q, want live collection %q", call.Collection, codebase.CollectionName)
		}
		seenPrefixes[call.Prefix] = true
	}
	if !seenPrefixes["conv/conv-alpha/"] || !seenPrefixes["conv/conv-beta/"] {
		t.Fatalf("reuse load prefixes = %v, want conv/conv-alpha/ and conv/conv-beta/", seenPrefixes)
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

// TestConversationIngestReuseLoadFailureFallsBackToFullEmbed proves a failed
// reuse load does not fail the job: the conversation reindexes with a nil reuse
// map, so every chunk embeds fresh, and the job still completes.
func TestConversationIngestReuseLoadFailureFallsBackToFullEmbed(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	fake := &fakeSemantic{
		loadReuseForPrefix: func(_ context.Context, _ string, _ string) (map[string][]float32, error) {
			return nil, errors.New("milvus read failed")
		},
	}
	manager.semantic = fake
	ctx := context.Background()
	collectionID := "thread-reuse-fallback"

	job, err := manager.upsertConversationDocuments(ctx, collectionID, []model.ConversationDocument{
		{ConversationID: "conv-solo", MessageIndex: 0, Role: "user", TimestampUnix: 1712345002, Text: "solo"},
	}, map[string]string{"conv-solo": "fp-s-1"}, testClientInfo())
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
}

package daemon

import (
	"context"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
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
		Content:        "Needle conversation content",
		RelativePath:   "conv/conversation-alpha/12",
		StartLine:      0,
		EndLine:        0,
		Language:       "",
		FileExtension:  "",
		ConversationID: "conversation-alpha",
		MessageIndex:   12,
		Role:           "assistant",
		TimestampUnix:  1712345678,
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

func TestConversationDocumentsToStoredChunksSplitsOversizedMessage(t *testing.T) {
	t.Parallel()

	text := strings.Repeat("a", conversationChunkMaxBytes+5)
	chunks, err := conversationDocumentsToStoredChunks([]model.ConversationDocument{{
		ConversationID: "thread-large",
		MessageIndex:   7,
		Role:           "assistant",
		TimestampUnix:  1712345678,
		Text:           text,
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
		upsertConversation: func(ctx context.Context, collectionName string, chunks []model.StoredChunk) error {
			_ = ctx
			_ = collectionName
			upsertedChunks <- append([]model.StoredChunk{}, chunks...)
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

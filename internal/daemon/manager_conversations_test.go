package daemon

import (
	"context"
	"strings"
	"testing"

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

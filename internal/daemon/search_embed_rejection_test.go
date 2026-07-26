package daemon

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/grpc/codes"
	grpcstatus "google.golang.org/grpc/status"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/embedding"
	"goodkind.io/lm-semantic-search/internal/model"
)

// TestSearchCodeTellsTheClientWhyTheEmbedderRefusedTheQuery drives a real
// embedding provider against an endpoint answering the way the live endpoint
// answers an over-long query, then asserts the gRPC client learns the reason and
// the model's limit. A failure the user cannot act on is only half an
// improvement over the silent truncation this branch removed: the query is
// fixable by shortening it, and the response has to say so.
func TestSearchCodeTellsTheClientWhyTheEmbedderRefusedTheQuery(t *testing.T) {
	endpoint := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusBadRequest)
		_, _ = writer.Write([]byte(`{"error":{"code":"context_length_exceeded","type":"invalid_request_error","message":"This model's maximum context length is 8192 tokens, however the input at index 0 resolved to 10000 tokens. Reduce the input length."}}`))
	}))
	defer endpoint.Close()

	provider, err := embedding.NewProvider(context.Background(), config.Config{
		EmbeddingProvider: "OpenAI",
		OpenAIAPIKey:      "test-key",
		OpenAIBaseURL:     endpoint.URL,
		EmbeddingModel:    "model",
	})
	if err != nil {
		t.Fatalf("NewProvider returned error: %v", err)
	}
	_, embedErr := provider.Embed(context.Background(), strings.Repeat("a", 40000))
	if embedErr == nil {
		t.Fatal("the provider accepted a query the endpoint rejected as too long")
	}

	manager, _, repoPath := newTestManager(t)
	canonical, symlinkErr := filepath.EvalSymlinks(repoPath)
	if symlinkErr != nil {
		t.Fatalf("EvalSymlinks returned error: %v", symlinkErr)
	}
	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	// The semantic layer wraps whatever the provider returned, exactly as
	// searchCollection does on the real path.
	manager.semantic = &fakeSemantic{
		search: func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error) {
			return nil, fmt.Errorf("embed query: %w", embedErr)
		},
	}

	server := NewGRPCServer(manager, nil)
	_, searchErr := server.SearchCode(context.Background(), &pb.SearchCodeRequest{
		Path:  repoPath,
		Query: "anything",
		Limit: 5,
	})
	if searchErr == nil {
		t.Fatal("SearchCode succeeded for a query the embedder refused")
	}
	responded, ok := grpcstatus.FromError(searchErr)
	if !ok {
		t.Fatalf("SearchCode returned a non-status error: %v", searchErr)
	}
	if responded.Code() != codes.InvalidArgument {
		t.Fatalf("gRPC code = %v, want %v so the caller knows the request itself is fixable", responded.Code(), codes.InvalidArgument)
	}
	message := responded.Message()
	if strings.Contains(message, "internal error") {
		t.Fatalf("the client saw an opaque internal error with no way to act on it: %q", message)
	}
	for _, want := range []string{"context_length_exceeded", "8192", "10000", "shorten the input"} {
		if !strings.Contains(message, want) {
			t.Fatalf("client message %q does not carry %q", message, want)
		}
	}
}

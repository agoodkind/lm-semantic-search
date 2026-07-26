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

// embedRefusalCase is one endpoint wording for a context_length_exceeded
// refusal, paired with the fragments a client should read back from it. The
// first names the offending index, which is what the live endpoint sends; the
// second carries the same machine-readable refusal with different prose, which
// is the case that used to fall through to an opaque internal error and a
// degraded embedder banner.
type embedRefusalCase struct {
	name          string
	body          string
	wantFragments []string
}

// searchQueryRefusalCases enumerates the endpoint wordings under test. The HTTP
// status and the error code are identical across them, so the client outcome has
// to be identical too.
func searchQueryRefusalCases() []embedRefusalCase {
	return []embedRefusalCase{
		{
			name:          "endpoint names the offending index",
			body:          `{"error":{"code":"context_length_exceeded","type":"invalid_request_error","message":"This model's maximum context length is 8192 tokens, however the input at index 0 resolved to 10000 tokens. Reduce the input length."}}`,
			wantFragments: []string{"context_length_exceeded", "8192", "10000", "shorten the input"},
		},
		{
			name:          "endpoint words the refusal without an index",
			body:          `{"error":{"code":"context_length_exceeded","type":"invalid_request_error","message":"The input is too long for this model."}}`,
			wantFragments: []string{"context_length_exceeded", "no size figures", "shorten the input"},
		},
	}
}

// TestSearchCodeTellsTheClientWhyTheEmbedderRefusedTheQuery drives a real
// embedding provider against an endpoint refusing an over-long query, then
// asserts the gRPC client learns the reason and can act on it. A failure the
// user cannot act on is only half an improvement over the silent truncation this
// branch removed. The endpoint's wording must change none of that, and a refused
// query must leave embedder health alone: the endpoint answered correctly and is
// healthy, it simply refused one oversized input.
func TestSearchCodeTellsTheClientWhyTheEmbedderRefusedTheQuery(t *testing.T) {
	for _, refusal := range searchQueryRefusalCases() {
		t.Run(refusal.name, func(t *testing.T) {
			endpoint := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				writer.Header().Set("Content-Type", "application/json")
				writer.WriteHeader(http.StatusBadRequest)
				_, _ = writer.Write([]byte(refusal.body))
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
				t.Fatal("the provider accepted a query the endpoint refused as too long")
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
			for _, want := range refusal.wantFragments {
				if !strings.Contains(message, want) {
					t.Fatalf("client message %q does not carry %q", message, want)
				}
			}
			if health := manager.DependencyHealth(); health.Degraded() {
				t.Fatalf("a refused query degraded embedder health to %q; the endpoint answered correctly and is healthy", health.Mode)
			}
		})
	}
}

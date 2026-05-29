package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"goodkind.io/claude-context-go/internal/config"
	"goodkind.io/claude-context-go/internal/metrics"
)

func TestOpenAICompatibleProviderEmbedBatch(t *testing.T) {
	t.Parallel()

	var requestPath string
	var requestHeader string
	var requestBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		requestPath = request.URL.Path
		requestHeader = request.Header.Get("Authorization")
		if err := json.NewDecoder(request.Body).Decode(&requestBody); err != nil {
			t.Fatalf("Decode returned error: %v", err)
		}
		_ = json.NewEncoder(writer).Encode(map[string]any{
			"data": []map[string]any{
				{"embedding": []float64{1.0, 2.0}},
				{"embedding": []float64{3.0, 4.0}},
			},
		})
	}))
	defer server.Close()

	provider, err := newOpenAICompatibleProvider("OpenAI", "test-key", server.URL, "text-embedding-3-small", 2)
	if err != nil {
		t.Fatalf("newOpenAICompatibleProvider returned error: %v", err)
	}

	vectors, err := provider.EmbedBatch(context.Background(), []string{"alpha", "beta"})
	if err != nil {
		t.Fatalf("EmbedBatch returned error: %v", err)
	}
	if requestPath != "/embeddings" {
		t.Fatalf("request path = %q", requestPath)
	}
	if requestHeader != "Bearer test-key" {
		t.Fatalf("authorization header = %q", requestHeader)
	}
	if requestBody["model"] != "text-embedding-3-small" {
		t.Fatalf("request model = %#v", requestBody["model"])
	}
	if len(vectors) != 2 || len(vectors[0]) != 2 || vectors[0][0] != 1 {
		t.Fatalf("vectors = %#v", vectors)
	}
}

func TestEmbedBatchRecordsMetrics(t *testing.T) {
	// Touches package-global metrics counters, so it cannot run in parallel
	// with other tests that read the same state.
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(writer).Encode(map[string]any{
			"data": []map[string]any{
				{"embedding": []float64{1.0, 2.0}},
				{"embedding": []float64{3.0, 4.0}},
			},
		})
	}))
	defer server.Close()

	provider, err := newOpenAICompatibleProvider("OpenAI", "test-key", server.URL, "text-embedding-3-small", 2)
	if err != nil {
		t.Fatalf("newOpenAICompatibleProvider returned error: %v", err)
	}

	before := metrics.Read()
	if _, err := provider.EmbedBatch(context.Background(), []string{"alpha", "beta"}); err != nil {
		t.Fatalf("EmbedBatch returned error: %v", err)
	}
	after := metrics.Read()

	if after.EmbedBatchesTotal-before.EmbedBatchesTotal != 1 {
		t.Fatalf("EmbedBatchesTotal delta = %d, want 1", after.EmbedBatchesTotal-before.EmbedBatchesTotal)
	}
	if after.EmbedVectorsTotal-before.EmbedVectorsTotal != 2 {
		t.Fatalf("EmbedVectorsTotal delta = %d, want 2", after.EmbedVectorsTotal-before.EmbedVectorsTotal)
	}
	if after.EmbedBatchesFailed-before.EmbedBatchesFailed != 0 {
		t.Fatalf("EmbedBatchesFailed delta = %d, want 0", after.EmbedBatchesFailed-before.EmbedBatchesFailed)
	}
	if after.EmbedInflight != 0 {
		t.Fatalf("EmbedInflight = %d, want 0", after.EmbedInflight)
	}
}

func TestNewProviderRejectsNonOpenAI(t *testing.T) {
	t.Parallel()

	_, err := NewProvider(config.Config{
		EmbeddingProvider: "VoyageAI",
		OpenAIAPIKey:      "test-key",
		EmbeddingModel:    "voyage-code-3",
	})
	if err == nil {
		t.Fatal("NewProvider returned nil error for unsupported provider")
	}
}

func TestNewProviderAcceptsOpenAIWithBaseURL(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider(config.Config{
		EmbeddingProvider: "OpenAI",
		OpenAIAPIKey:      "test-key",
		OpenAIBaseURL:     "https://example.invalid/v1",
		EmbeddingModel:    "text-embedding-3-small",
	})
	if err != nil {
		t.Fatalf("NewProvider returned error: %v", err)
	}
	if provider.ProviderName() != "OpenAI" {
		t.Fatalf("provider name = %q", provider.ProviderName())
	}
}

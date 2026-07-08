package embedding

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/openai/openai-go/v2"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/metrics"
)

// testEmbedTimeout bounds one request in the happy-path tests. It is generous
// relative to the instant httptest responses, so it exercises the bound without
// firing.
const testEmbedTimeout = 2 * time.Second

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

	provider, err := newOpenAICompatibleProvider("test-key", server.URL, "text-embedding-3-small", 2, testEmbedTimeout)
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

	provider, err := newOpenAICompatibleProvider("test-key", server.URL, "text-embedding-3-small", 2, testEmbedTimeout)
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

func TestEmbedBatchRetriesTransientThenSucceeds(t *testing.T) {
	t.Parallel()

	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		if calls.Add(1) == 1 {
			writer.WriteHeader(http.StatusTooManyRequests)
			_, _ = writer.Write([]byte(`{"error":{"message":"busy","type":"rate_limit_exceeded","code":"rate_limited"}}`))
			return
		}
		_ = json.NewEncoder(writer).Encode(map[string]any{
			"data": []map[string]any{{"embedding": []float64{1.0, 2.0}}},
		})
	}))
	defer server.Close()

	provider, err := newOpenAICompatibleProvider("test-key", server.URL, "text-embedding-3-small", 0, testEmbedTimeout)
	if err != nil {
		t.Fatalf("newOpenAICompatibleProvider returned error: %v", err)
	}

	vectors, err := provider.EmbedBatch(context.Background(), []string{"alpha"})
	if err != nil {
		t.Fatalf("EmbedBatch returned error: %v", err)
	}
	if len(vectors) != 1 {
		t.Fatalf("vectors = %#v", vectors)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("server calls = %d, want 2 (one 429 then a successful retry)", got)
	}
}

func TestEmbedBatchPersistentBusyReturnsErrEmbedderBusy(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusTooManyRequests)
		_, _ = writer.Write([]byte(`{"error":{"message":"busy","type":"rate_limit_exceeded"}}`))
	}))
	defer server.Close()

	provider, err := newOpenAICompatibleProvider("test-key", server.URL, "model", 0, testEmbedTimeout)
	if err != nil {
		t.Fatalf("newOpenAICompatibleProvider returned error: %v", err)
	}

	_, err = provider.EmbedBatch(context.Background(), []string{"alpha"})
	if err == nil {
		t.Fatal("EmbedBatch returned nil error for a persistent 429")
	}
	if !errors.Is(err, ErrEmbedderBusy) {
		t.Fatalf("error is not classified ErrEmbedderBusy: %v", err)
	}
}

func TestEmbedBatchNon429NotBusy(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusBadRequest)
		_, _ = writer.Write([]byte(`{"error":{"message":"bad request","type":"invalid_request_error"}}`))
	}))
	defer server.Close()

	provider, err := newOpenAICompatibleProvider("test-key", server.URL, "model", 0, testEmbedTimeout)
	if err != nil {
		t.Fatalf("newOpenAICompatibleProvider returned error: %v", err)
	}

	_, err = provider.EmbedBatch(context.Background(), []string{"alpha"})
	if err == nil {
		t.Fatal("EmbedBatch returned nil error for a 400")
	}
	if errors.Is(err, ErrEmbedderBusy) {
		t.Fatalf("a 400 was wrongly classified as ErrEmbedderBusy: %v", err)
	}
}

func TestEmbedBatchNon429ReturnsRejected(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusBadRequest)
		_, _ = writer.Write([]byte(`{"error":{"message":"bad","type":"invalid_request_error"}}`))
	}))
	defer server.Close()

	provider, err := newOpenAICompatibleProvider("test-key", server.URL, "model", 0, testEmbedTimeout)
	if err != nil {
		t.Fatalf("newOpenAICompatibleProvider returned error: %v", err)
	}

	_, err = provider.EmbedBatch(context.Background(), []string{"alpha"})
	if !errors.Is(err, ErrEmbedderRejected) {
		t.Fatalf("a 400 should classify as ErrEmbedderRejected: %v", err)
	}
	if errors.Is(err, ErrEmbedderBusy) {
		t.Fatalf("a 400 must not classify as ErrEmbedderBusy: %v", err)
	}
}

func TestEmbedBatchTimesOutOnUnresponsiveEndpoint(t *testing.T) {
	t.Parallel()

	// A listener that never accepts leaves the kernel to complete the TCP
	// handshake, so the client connects and sends its request, then blocks
	// forever waiting for a response. This mimics a wedged embedder that holds
	// the socket open without answering. Without a request bound the embed call
	// hangs indefinitely; the bound must fail it instead of stranding the caller.
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen returned error: %v", err)
	}
	defer func() { _ = listener.Close() }()

	baseURL := "http://" + listener.Addr().String() + "/v1"
	provider, err := newOpenAICompatibleProvider("test-key", baseURL, "model", 0, 200*time.Millisecond)
	if err != nil {
		t.Fatalf("newOpenAICompatibleProvider returned error: %v", err)
	}

	start := time.Now()
	_, err = provider.EmbedBatch(context.Background(), []string{"alpha"})
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("EmbedBatch returned nil error against an unresponsive endpoint")
	}
	if elapsed > 5*time.Second {
		t.Fatalf("EmbedBatch took %v against an unresponsive endpoint; the request bound did not fire", elapsed)
	}
	if errors.Is(err, ErrEmbedderBusy) || errors.Is(err, ErrEmbedderRejected) {
		t.Fatalf("request timeout misclassified as busy or rejected: %v", err)
	}
	if !strings.Contains(err.Error(), "did not respond within") {
		t.Fatalf("error does not indicate a request timeout: %v", err)
	}
}

func TestTransientEmbedStatus(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		err       error
		wantCode  int
		transient bool
	}{
		{"too many requests", &openai.Error{StatusCode: http.StatusTooManyRequests}, http.StatusTooManyRequests, true},
		{"service unavailable", &openai.Error{StatusCode: http.StatusServiceUnavailable}, http.StatusServiceUnavailable, true},
		{"bad request", &openai.Error{StatusCode: http.StatusBadRequest}, http.StatusBadRequest, false},
		{"wrapped 429", fmt.Errorf("context: %w", &openai.Error{StatusCode: http.StatusTooManyRequests}), http.StatusTooManyRequests, true},
		{"non-api error", errors.New("connection refused"), 0, false},
	}
	for _, testCase := range cases {
		code, transient := transientEmbedStatus(testCase.err)
		if code != testCase.wantCode || transient != testCase.transient {
			t.Fatalf("%s: got (%d, %v), want (%d, %v)", testCase.name, code, transient, testCase.wantCode, testCase.transient)
		}
	}
}

func TestEmbedBackoffDoubles(t *testing.T) {
	t.Parallel()

	if embedBackoff(1) != embedBackoffBase {
		t.Fatalf("attempt 1 backoff = %v, want %v", embedBackoff(1), embedBackoffBase)
	}
	if embedBackoff(2) != 2*embedBackoffBase {
		t.Fatalf("attempt 2 backoff = %v, want %v", embedBackoff(2), 2*embedBackoffBase)
	}
	if embedBackoff(3) != 4*embedBackoffBase {
		t.Fatalf("attempt 3 backoff = %v, want %v", embedBackoff(3), 4*embedBackoffBase)
	}
}

func TestNewProviderClampsTimeout(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name        string
		timeoutMS   int
		wantTimeout time.Duration
	}{
		{name: "negative clamps to unbounded", timeoutMS: -1, wantTimeout: 0},
		{name: "zero stays unbounded", timeoutMS: 0, wantTimeout: 0},
		{name: "positive is honored", timeoutMS: 1500, wantTimeout: 1500 * time.Millisecond},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			provider, err := NewProvider(config.Config{
				EmbeddingProvider:         "OpenAI",
				OpenAIAPIKey:              "test-key",
				EmbeddingModel:            "text-embedding-3-small",
				EmbeddingRequestTimeoutMS: testCase.timeoutMS,
			})
			if err != nil {
				t.Fatalf("NewProvider returned error: %v", err)
			}
			concrete, ok := provider.(*openAICompatibleProvider)
			if !ok {
				t.Fatalf("provider type = %T, want *openAICompatibleProvider", provider)
			}
			if concrete.requestTimeout != testCase.wantTimeout {
				t.Fatalf("requestTimeout = %v, want %v", concrete.requestTimeout, testCase.wantTimeout)
			}
		})
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

package embedding_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/embedding"
)

// embeddingsServer serves the OpenAI embeddings API. respond writes the reply
// for one decoded request.
func embeddingsServer(
	t *testing.T,
	respond func(writer http.ResponseWriter, request *http.Request, inputs []string),
) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var body struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Errorf("decode embeddings request: %v", err)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		respond(writer, request, body.Input)
	}))
	t.Cleanup(server.Close)
	return server
}

func writeVectors(t *testing.T, writer http.ResponseWriter, count int, dimension int) {
	t.Helper()
	data := make([]map[string][]float64, 0, count)
	for index := range count {
		vector := make([]float64, dimension)
		vector[0] = float64(index + 1)
		data = append(data, map[string][]float64{"embedding": vector})
	}
	if err := json.NewEncoder(writer).Encode(map[string]any{"data": data}); err != nil {
		t.Errorf("encode embeddings response: %v", err)
	}
}

func writeError(t *testing.T, writer http.ResponseWriter, status int, body map[string]string) {
	t.Helper()
	writer.WriteHeader(status)
	if err := json.NewEncoder(writer).Encode(map[string]any{"error": body}); err != nil {
		t.Errorf("encode error response: %v", err)
	}
}

func newAdapter(t *testing.T, server *httptest.Server, apiKey string) library.Embedder {
	t.Helper()
	embedder, err := embedding.NewOpenAI(context.Background(), embedding.OpenAIConfig{
		BaseURL:        server.URL,
		APIKey:         apiKey,
		Model:          "test-model",
		Dimension:      4,
		RequestTimeout: 5 * time.Second,
		MaxAttempts:    2,
		BackoffBase:    time.Millisecond,
	})
	if err != nil {
		t.Fatalf("NewOpenAI: %v", err)
	}
	return embedder
}

func TestOpenAIAdapterSendsOnlyTheCallerCredential(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "ambient-key")
	var authorization atomic.Value
	server := embeddingsServer(t, func(writer http.ResponseWriter, request *http.Request, inputs []string) {
		authorization.Store(request.Header.Get("Authorization"))
		writeVectors(t, writer, len(inputs), 4)
	})

	for _, testCase := range []struct {
		name   string
		apiKey string
		want   string
	}{
		{name: "no credential", apiKey: "", want: ""},
		{name: "caller credential", apiKey: "caller-key", want: "Bearer caller-key"},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			vectors, err := newAdapter(t, server, testCase.apiKey).EmbedBatch(context.Background(), []string{"alpha", "beta"})
			if err != nil {
				t.Fatalf("EmbedBatch: %v", err)
			}
			if len(vectors) != 2 || vectors[1][0] != 2 {
				t.Fatalf("EmbedBatch returned %v, want two vectors in input order", vectors)
			}
			if got := authorization.Load(); got != testCase.want {
				t.Fatalf("Authorization header = %q, want %q", got, testCase.want)
			}
		})
	}
}

func TestOpenAIAdapterFailsTheWholeBatchWithTypedErrors(t *testing.T) {
	t.Parallel()
	for _, testCase := range []struct {
		name    string
		respond func(t *testing.T, writer http.ResponseWriter, inputs []string)
		want    error
	}{
		{
			name: "one input over the context window",
			respond: func(t *testing.T, writer http.ResponseWriter, inputs []string) {
				if len(inputs) > 1 {
					writeError(t, writer, http.StatusBadRequest, map[string]string{
						"code":    "context_length_exceeded",
						"message": "input at index 1 resolved to 9000 tokens; maximum context length is 4096 tokens",
					})
					return
				}
				writeVectors(t, writer, len(inputs), 4)
			},
			want: embedding.ErrEmbedderRejected,
		},
		{
			name: "wrong dimension",
			respond: func(t *testing.T, writer http.ResponseWriter, inputs []string) {
				writeVectors(t, writer, len(inputs), 3)
			},
			want: embedding.ErrEmbedderRejected,
		},
		{
			name: "busy on every attempt",
			respond: func(t *testing.T, writer http.ResponseWriter, _ []string) {
				writeError(t, writer, http.StatusTooManyRequests, map[string]string{"message": "slow down"})
			},
			want: embedding.ErrEmbedderBusy,
		},
		{
			name: "service paused",
			respond: func(t *testing.T, writer http.ResponseWriter, _ []string) {
				writeError(t, writer, http.StatusServiceUnavailable, map[string]string{
					"type":    "service_paused",
					"message": "low power mode",
				})
			},
			want: embedding.ErrEmbedderPaused,
		},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			server := embeddingsServer(t, func(writer http.ResponseWriter, _ *http.Request, inputs []string) {
				testCase.respond(t, writer, inputs)
			})
			vectors, err := newAdapter(t, server, "caller-key").EmbedBatch(context.Background(), []string{"alpha", "beta", "gamma"})
			if !errors.Is(err, testCase.want) {
				t.Fatalf("EmbedBatch error = %v, want %v", err, testCase.want)
			}
			if vectors != nil {
				t.Fatalf("EmbedBatch returned %d vectors with an error, want none", len(vectors))
			}
		})
	}
}

func TestOpenAIAdapterCancellationIsTyped(t *testing.T) {
	t.Parallel()
	server := embeddingsServer(t, func(writer http.ResponseWriter, _ *http.Request, inputs []string) {
		writeVectors(t, writer, len(inputs), 4)
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := newAdapter(t, server, "").EmbedBatch(ctx, []string{"alpha"})
	if !errors.Is(err, embedding.ErrEmbedCancelled) {
		t.Fatalf("EmbedBatch error = %v, want ErrEmbedCancelled", err)
	}
}

func TestNewOpenAIRejectsInvalidConfiguration(t *testing.T) {
	t.Parallel()
	valid := embedding.OpenAIConfig{BaseURL: "http://localhost:1/v1", Model: "m", Dimension: 4}
	for _, testCase := range []struct {
		name   string
		mutate func(*embedding.OpenAIConfig)
	}{
		{name: "empty base URL", mutate: func(c *embedding.OpenAIConfig) { c.BaseURL = "" }},
		{name: "empty model", mutate: func(c *embedding.OpenAIConfig) { c.Model = " " }},
		{name: "zero dimension", mutate: func(c *embedding.OpenAIConfig) { c.Dimension = 0 }},
		{name: "negative timeout", mutate: func(c *embedding.OpenAIConfig) { c.RequestTimeout = -time.Second }},
		{name: "negative attempts", mutate: func(c *embedding.OpenAIConfig) { c.MaxAttempts = -1 }},
		{name: "negative backoff", mutate: func(c *embedding.OpenAIConfig) { c.BackoffBase = -time.Millisecond }},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			config := valid
			testCase.mutate(&config)
			if _, err := embedding.NewOpenAI(context.Background(), config); !errors.Is(err, library.ErrInvalidRequest) {
				t.Fatalf("NewOpenAI error = %v, want ErrInvalidRequest", err)
			}
		})
	}
}

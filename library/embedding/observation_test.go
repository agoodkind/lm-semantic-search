package embedding_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/library/embedding"
	"goodkind.io/lm-semantic-search/library/observation"
)

type embeddingEvents struct {
	mutex  sync.Mutex
	events []observation.Event
}

func (events *embeddingEvents) Observe(event observation.Event) {
	events.mutex.Lock()
	defer events.mutex.Unlock()
	events.events = append(events.events, event)
}

type embeddingHTTPRequest struct {
	Input []string `json:"input"`
}

type embeddingHTTPVector struct {
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

type embeddingHTTPResponse struct {
	Data []embeddingHTTPVector `json:"data"`
}

// characterEmbeddings computes four character-frequency dimensions for each input.
func characterEmbeddings(inputs []string) embeddingHTTPResponse {
	response := embeddingHTTPResponse{Data: make([]embeddingHTTPVector, 0, len(inputs))}
	for index, input := range inputs {
		vector := make([]float64, 4)
		for _, character := range input {
			vector[int(character)%len(vector)]++
		}
		response.Data = append(response.Data, embeddingHTTPVector{Embedding: vector, Index: index})
	}
	return response
}

func TestOpenAIObservationCountsActualHTTPRetryAndValidatedVectors(t *testing.T) {
	collector := &embeddingEvents{}
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var input embeddingHTTPRequest
		if err := json.NewDecoder(request.Body).Decode(&input); err != nil {
			t.Errorf("decode actual embedding request: %v", err)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		if requests.Add(1) == 1 {
			writer.WriteHeader(http.StatusTooManyRequests)
			return
		}
		if err := json.NewEncoder(writer).Encode(characterEmbeddings(input.Input)); err != nil {
			t.Errorf("encode computed embeddings: %v", err)
		}
	}))
	t.Cleanup(server.Close)
	ctx := observation.WithScope(t.Context(), observation.Scope{RunID: "http-retry", Generation: 1, ProcessID: os.Getpid(), Purpose: observation.Ingestion})
	embedder, err := embedding.NewOpenAI(ctx, embedding.OpenAIConfig{Observer: collector, BaseURL: server.URL, Model: "character-frequency", Dimension: 4, MaxAttempts: 2, BackoffBase: time.Millisecond})
	if err != nil {
		t.Fatalf("construct observed public embedder: %v", err)
	}
	vectors, err := embedder.EmbedBatch(ctx, []string{"alpha", "beta"})
	if err != nil || len(vectors) != 2 {
		t.Fatalf("real HTTP embedding result = %v, %v", vectors, err)
	}
	if requests.Load() != 2 {
		t.Fatalf("actual HTTP requests = %d, want 2", requests.Load())
	}
	requireHTTPRetryObservations(t, collector)
}

func requireHTTPRetryObservations(t *testing.T, collector *embeddingEvents) {
	t.Helper()
	collector.mutex.Lock()
	defer collector.mutex.Unlock()
	attempts := make([]observation.Event, 0)
	validated := 0
	for _, event := range collector.events {
		if event.Phase != observation.Completed {
			continue
		}
		if event.Scope.RunID != "http-retry" || event.Scope.Purpose != observation.Ingestion {
			t.Fatalf("incorrect HTTP request scope: %+v", event)
		}
		switch event.Operation {
		case observation.EmbeddingAttempt:
			attempts = append(attempts, event)
		case observation.EmbeddingValidation:
			validated += event.Data.Embedding.Validated
		}
	}
	if len(attempts) != 2 || validated != 2 {
		t.Fatalf("observed %d SDK attempts and %d validated vectors", len(attempts), validated)
	}
	first, second := attempts[0], attempts[1]
	if first.Outcome != observation.Failure || first.Data.Embedding != (observation.EmbeddingData{Attempt: 1, Requested: 2}) {
		t.Fatalf("first real HTTP attempt = %+v", first)
	}
	if second.Outcome != observation.Success || second.Data.Embedding != (observation.EmbeddingData{Attempt: 2, Requested: 2, Returned: 2}) {
		t.Fatalf("second real HTTP attempt = %+v", second)
	}
}

func TestOpenAIObservationClassifiesAnActualCancelledHTTPRequest(t *testing.T) {
	collector := &embeddingEvents{}
	requestStarted := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, request *http.Request) {
		var input embeddingHTTPRequest
		if err := json.NewDecoder(request.Body).Decode(&input); err != nil {
			t.Errorf("decode cancellable embedding request: %v", err)
			return
		}
		close(requestStarted)
		<-request.Context().Done()
	}))
	t.Cleanup(server.Close)
	ctx, cancel := context.WithCancel(observation.WithScope(t.Context(), observation.Scope{RunID: "http-cancel", Purpose: observation.Ingestion}))
	defer cancel()
	embedder, err := embedding.NewOpenAI(ctx, embedding.OpenAIConfig{Observer: collector, BaseURL: server.URL, Model: "character-frequency", Dimension: 4, MaxAttempts: 1})
	if err != nil {
		t.Fatalf("construct observed public embedder: %v", err)
	}
	done := make(chan error, 1)
	go func() { _, err := embedder.EmbedBatch(ctx, []string{"alpha"}); done <- err }()
	select {
	case <-requestStarted:
		cancel()
	case <-t.Context().Done():
		t.Fatal("actual embedding request did not begin")
	}
	if err := <-done; err == nil {
		t.Fatal("cancelled real HTTP request succeeded")
	}
	collector.mutex.Lock()
	defer collector.mutex.Unlock()
	completed := 0
	for _, event := range collector.events {
		if event.Operation == observation.EmbeddingAttempt && event.Phase == observation.Completed {
			completed++
			if event.Outcome != observation.Cancelled {
				t.Fatalf("actual cancelled SDK outcome = %s", event.Outcome)
			}
		}
	}
	if completed != 1 {
		t.Fatalf("cancelled SDK completions = %d, want 1", completed)
	}
}

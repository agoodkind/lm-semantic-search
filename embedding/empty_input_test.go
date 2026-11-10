package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"goodkind.io/lm-semantic-search/internal/adapterr"
)

// TestEmbedBatchRefusesEmptyContentWithoutCallingTheEndpoint proves the engine
// never spends an embedding call on content that has nothing to embed. An empty
// input and a whitespace-only input both carry no content, so each must come
// back as a skipped input with a nil vector, and neither may reach the endpoint.
// The surviving input must still be embedded, so one empty sibling never fails
// the batch.
func TestEmbedBatchRefusesEmptyContentWithoutCallingTheEndpoint(t *testing.T) {
	t.Parallel()

	var receivedInputs []string
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		var body struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Errorf("Decode returned error: %v", err)
			return
		}
		receivedInputs = append(receivedInputs, body.Input...)
		data := make([]map[string][]float64, 0, len(body.Input))
		for range body.Input {
			data = append(data, map[string][]float64{"embedding": {1.0, 2.0}})
		}
		_ = json.NewEncoder(writer).Encode(map[string]any{"data": data})
	}))
	defer server.Close()

	provider, err := newOpenAICompatibleProvider("test-key", server.URL, "model", 2, testEmbedTimeout)
	if err != nil {
		t.Fatalf("newOpenAICompatibleProvider returned error: %v", err)
	}

	result, err := provider.EmbedBatch(context.Background(), []string{"", "real content", "   \n\t "})
	if err != nil {
		t.Fatalf("EmbedBatch returned error: %v", err)
	}

	if len(receivedInputs) != 1 || receivedInputs[0] != "real content" {
		t.Fatalf("endpoint received %#v, want only the one input that carries content; an embedding call was spent on nothing", receivedInputs)
	}

	if len(result.Vectors) != 3 {
		t.Fatalf("len(Vectors) = %d, want 3 index-aligned with the inputs", len(result.Vectors))
	}
	if result.Vectors[0] != nil {
		t.Fatalf("Vectors[0] = %#v, want nil for empty content", result.Vectors[0])
	}
	if result.Vectors[2] != nil {
		t.Fatalf("Vectors[2] = %#v, want nil for whitespace-only content", result.Vectors[2])
	}
	if len(result.Vectors[1]) != 2 {
		t.Fatalf("Vectors[1] = %#v, want the embedded survivor", result.Vectors[1])
	}

	if len(result.Skipped) != 2 {
		t.Fatalf("len(Skipped) = %d, want 2; a refusal must be reported, never silent", len(result.Skipped))
	}
	for _, skip := range result.Skipped {
		if skip.Reason != adapterr.EmbedRejectionEmptyContent {
			t.Fatalf("Skipped[%d].Reason = %q, want %q", skip.Index, skip.Reason, adapterr.EmbedRejectionEmptyContent)
		}
		if skip.Index != 0 && skip.Index != 2 {
			t.Fatalf("Skipped carries index %d, want the empty inputs at 0 and 2", skip.Index)
		}
		// No size limit refused this input, so quoting one would send the reader
		// after a shorter input when the input was already as short as it gets.
		if skip.ReportedTokens.Reported || skip.MaxTokens.Reported {
			t.Fatalf("Skipped[%d] carries token figures %+v/%+v, want none", skip.Index, skip.ReportedTokens, skip.MaxTokens)
		}
	}
}

// TestEmbedBatchEmbedsOrdinaryContentUntouched is the other half of the guard:
// it must refuse nothing that carries content. Content that merely looks
// marginal, a single character, a lone digit, and text padded with surrounding
// whitespace, all still carry something a vector can describe, so every one of
// them must reach the endpoint exactly as the caller supplied it.
func TestEmbedBatchEmbedsOrdinaryContentUntouched(t *testing.T) {
	t.Parallel()

	ordinary := []string{"a", "0", "  padded  ", "\tleading tab", "ordinary sentence.", "}"}

	var receivedInputs []string
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		var body struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Errorf("Decode returned error: %v", err)
			return
		}
		receivedInputs = append(receivedInputs, body.Input...)
		data := make([]map[string][]float64, 0, len(body.Input))
		for range body.Input {
			data = append(data, map[string][]float64{"embedding": {1.0, 2.0}})
		}
		_ = json.NewEncoder(writer).Encode(map[string]any{"data": data})
	}))
	defer server.Close()

	provider, err := newOpenAICompatibleProvider("test-key", server.URL, "model", 2, testEmbedTimeout)
	if err != nil {
		t.Fatalf("newOpenAICompatibleProvider returned error: %v", err)
	}

	result, err := provider.EmbedBatch(context.Background(), ordinary)
	if err != nil {
		t.Fatalf("EmbedBatch returned error: %v", err)
	}
	if len(result.Skipped) != 0 {
		t.Fatalf("Skipped = %#v, want none; the guard refused content that carries something", result.Skipped)
	}
	if !slices.Equal(receivedInputs, ordinary) {
		t.Fatalf("endpoint received %#v, want every input unchanged %#v", receivedInputs, ordinary)
	}
	for index, vector := range result.Vectors {
		if len(vector) != 2 {
			t.Fatalf("Vectors[%d] = %#v, want an embedded vector for %q", index, vector, ordinary[index])
		}
	}
}

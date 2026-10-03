package embedding

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/openai/openai-go/v2"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/metrics"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
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

	result, err := provider.EmbedBatch(context.Background(), []string{"alpha", "beta"})
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
	if len(result.Vectors) != 2 || len(result.Vectors[0]) != 2 || result.Vectors[0][0] != 1 {
		t.Fatalf("vectors = %#v", result.Vectors)
	}
	if len(result.Skipped) != 0 {
		t.Fatalf("Skipped = %#v, want none", result.Skipped)
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

	result, err := provider.EmbedBatch(context.Background(), []string{"alpha"})
	if err != nil {
		t.Fatalf("EmbedBatch returned error: %v", err)
	}
	if len(result.Vectors) != 1 {
		t.Fatalf("vectors = %#v", result.Vectors)
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

func TestEmbedBatchServicePausedReportsEndpointMessageWithoutCapacityClaim(t *testing.T) {
	t.Parallel()

	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusServiceUnavailable)
		_, _ = writer.Write([]byte(`{"error":{"message":"service paused to preserve battery (low_power_mode)","type":"service_paused"}}`))
	}))
	defer server.Close()

	provider, err := newOpenAICompatibleProvider("test-key", server.URL, "model", 0, testEmbedTimeout)
	if err != nil {
		t.Fatalf("newOpenAICompatibleProvider returned error: %v", err)
	}

	_, err = provider.EmbedBatch(context.Background(), []string{"alpha"})
	if err == nil {
		t.Fatal("EmbedBatch returned nil error for a paused service")
	}
	message := adapterr.SafeMessage(err)
	want := "embedding endpoint reported: service paused to preserve battery (low_power_mode); leave low power mode or resume the embedding service"
	if message != want {
		t.Fatalf("paused service message = %q, want %q", message, want)
	}
	var adapterErr *adapterr.AdapterError
	if !errors.As(err, &adapterErr) || adapterErr.Class != adapterr.ClassEmbedderPaused {
		t.Fatalf("paused service class = %v, want %q", adapterErr, adapterr.ClassEmbedderPaused)
	}
	if errors.Is(err, ErrEmbedderBusy) {
		t.Fatalf("paused service was classified as capacity: %v", err)
	}
	if errors.Is(err, ErrEmbedderRejected) {
		t.Fatalf("paused service was classified as a configuration rejection: %v", err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("paused service received %d requests, want 1 without capacity retries", got)
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

func TestEmbedBatchSkipsOversizedInputAndEmbedsRest(t *testing.T) {
	t.Parallel()

	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		var body struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Fatalf("Decode returned error: %v", err)
		}
		// The first request carries both inputs; the endpoint rejects the whole
		// request naming the oversized input at index 0. After that input is
		// dropped, the retry carries only the survivor and succeeds.
		if calls.Add(1) == 1 {
			if len(body.Input) != 2 {
				t.Fatalf("first request input count = %d, want 2", len(body.Input))
			}
			writer.WriteHeader(http.StatusBadRequest)
			_, _ = writer.Write([]byte(`{"error":{"code":"context_length_exceeded","type":"invalid_request_error","message":"This model's maximum context length is 4096 tokens, however the input at index 0 resolved to 4472 tokens. Reduce the input length."}}`))
			return
		}
		if len(body.Input) != 1 {
			t.Fatalf("retry request input count = %d, want 1 (oversized input dropped)", len(body.Input))
		}
		_ = json.NewEncoder(writer).Encode(map[string]any{
			"data": []map[string]any{{"embedding": []float64{3.0, 4.0}}},
		})
	}))
	defer server.Close()

	provider, err := newOpenAICompatibleProvider("test-key", server.URL, "model", 2, testEmbedTimeout)
	if err != nil {
		t.Fatalf("newOpenAICompatibleProvider returned error: %v", err)
	}

	result, err := provider.EmbedBatch(context.Background(), []string{"oversized", "small"})
	// The whole batch must succeed: a per-input rejection is not a job failure, so
	// no error propagates and nothing marks the embedder unhealthy.
	if err != nil {
		t.Fatalf("EmbedBatch returned error for a per-input rejection: %v", err)
	}
	if errors.Is(err, ErrEmbedderRejected) || errors.Is(err, ErrEmbedderBusy) {
		t.Fatalf("a per-input rejection must not classify as a server error: %v", err)
	}
	if len(result.Vectors) != 2 {
		t.Fatalf("len(Vectors) = %d, want 2", len(result.Vectors))
	}
	if result.Vectors[0] != nil {
		t.Fatalf("Vectors[0] = %#v, want nil (input skipped)", result.Vectors[0])
	}
	if len(result.Vectors[1]) != 2 || result.Vectors[1][0] != 3 {
		t.Fatalf("Vectors[1] = %#v, want the embedded survivor", result.Vectors[1])
	}
	if len(result.Skipped) != 1 {
		t.Fatalf("len(Skipped) = %d, want 1", len(result.Skipped))
	}
	skip := result.Skipped[0]
	if skip.Index != 0 {
		t.Fatalf("Skipped[0].Index = %d, want 0", skip.Index)
	}
	if skip.Reason != "context_length_exceeded" {
		t.Fatalf("Skipped[0].Reason = %q, want context_length_exceeded", skip.Reason)
	}
	if skip.MaxTokens != adapterr.ReportedFigure(4096) {
		t.Fatalf("Skipped[0].MaxTokens = %+v, want a reported 4096", skip.MaxTokens)
	}
	if skip.ReportedTokens != adapterr.ReportedFigure(4472) {
		t.Fatalf("Skipped[0].ReportedTokens = %+v, want a reported 4472", skip.ReportedTokens)
	}
}

// oversizedHostedInputBytes is past the 32768-byte ceiling the provider used to
// cut every hosted input down to, so an input this long proves the cut is gone.
const oversizedHostedInputBytes = 40000

func TestEmbedBatchSendsOversizedInputWithoutShorteningIt(t *testing.T) {
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
		receivedInputs = body.Input
		_ = json.NewEncoder(writer).Encode(map[string]any{
			"data": []map[string]any{{"embedding": []float64{1.0, 2.0}}},
		})
	}))
	defer server.Close()

	provider, err := newOpenAICompatibleProvider("test-key", server.URL, "model", 2, testEmbedTimeout)
	if err != nil {
		t.Fatalf("newOpenAICompatibleProvider returned error: %v", err)
	}

	oversized := strings.Repeat("a", oversizedHostedInputBytes)
	result, err := provider.EmbedBatch(context.Background(), []string{oversized})
	if err != nil {
		t.Fatalf("EmbedBatch returned error: %v", err)
	}
	if len(receivedInputs) != 1 {
		t.Fatalf("endpoint received %d inputs, want 1", len(receivedInputs))
	}
	if len(receivedInputs[0]) != oversizedHostedInputBytes {
		t.Fatalf("endpoint received %d bytes, want the whole %d-byte input; the provider shortened it behind the caller's back", len(receivedInputs[0]), oversizedHostedInputBytes)
	}
	if receivedInputs[0] != oversized {
		t.Fatal("endpoint received input that differs from the caller's content")
	}
	if len(result.Vectors) != 1 || result.Vectors[0] == nil {
		t.Fatalf("vectors = %#v, want one vector for the accepted input", result.Vectors)
	}
	if len(result.Skipped) != 0 {
		t.Fatalf("Skipped = %#v, want none for an input the endpoint accepted", result.Skipped)
	}
}

func TestEmbedReportsOversizedQueryRejectionInsteadOfShorteningIt(t *testing.T) {
	t.Parallel()

	// A search query goes through single-input Embed, which no index path splits
	// beforehand. The endpoint sees the whole query and rejects it as too long, and
	// that rejection reaches the caller instead of a vector over a shortened copy.
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
		receivedInputs = body.Input
		writer.WriteHeader(http.StatusBadRequest)
		_, _ = writer.Write([]byte(`{"error":{"code":"context_length_exceeded","type":"invalid_request_error","message":"This model's maximum context length is 8192 tokens, however the input at index 0 resolved to 10000 tokens. Reduce the input length."}}`))
	}))
	defer server.Close()

	provider, err := newOpenAICompatibleProvider("test-key", server.URL, "model", 2, testEmbedTimeout)
	if err != nil {
		t.Fatalf("newOpenAICompatibleProvider returned error: %v", err)
	}

	oversized := strings.Repeat("b", oversizedHostedInputBytes)
	vector, embedErr := provider.Embed(context.Background(), oversized)
	if embedErr == nil {
		t.Fatal("Embed returned a vector for a query the endpoint rejected as too long")
	}
	if vector != nil {
		t.Fatalf("Embed returned %d values alongside the rejection", len(vector))
	}
	if len(receivedInputs) != 1 {
		t.Fatalf("endpoint received %d inputs, want 1", len(receivedInputs))
	}
	if len(receivedInputs[0]) != oversizedHostedInputBytes {
		t.Fatalf("endpoint received %d bytes, want the whole %d-byte query; the provider shortened it behind the caller's back", len(receivedInputs[0]), oversizedHostedInputBytes)
	}
}

// TestBothProvidersRefuseAnInputAsClientSafeInvalidArgument pins the shape both
// providers give a refused single input: a typed invalid-argument error whose
// client-safe message names the reason and the model's limit, and which never
// names the provider or the model. A caller can act on the reason and still
// cannot tell which provider refused the input.
func TestBothProvidersRefuseAnInputAsClientSafeInvalidArgument(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusBadRequest)
		_, _ = writer.Write([]byte(`{"error":{"code":"context_length_exceeded","type":"invalid_request_error","message":"This model's maximum context length is 8192 tokens, however the input at index 0 resolved to 10000 tokens. Reduce the input length."}}`))
	}))
	defer server.Close()

	hostedProvider, err := newOpenAICompatibleProvider("test-key", server.URL, "text-embedding-3-small", 2, testEmbedTimeout)
	if err != nil {
		t.Fatalf("newOpenAICompatibleProvider returned error: %v", err)
	}
	_, hostedErr := hostedProvider.Embed(context.Background(), strings.Repeat("a", oversizedHostedInputBytes))
	if hostedErr == nil {
		t.Fatal("the hosted provider accepted an input its endpoint rejected")
	}

	onnxProviderUnderTest := newUnloadedONNXProvider(t, offlinemodel.BGESmall)
	onnxOversized := strings.Repeat("a", onnxProviderUnderTest.runtime.tokenizer.maximumInputBytes()+1)
	_, onnxErr := onnxProviderUnderTest.Embed(context.Background(), onnxOversized)
	if onnxErr == nil {
		t.Fatal("the in-process provider accepted an input past its tokenizer bound")
	}

	// The in-process input was refused by the byte ceiling, which is 64 bytes per
	// allowed token, so the limit the caller must act on is that byte figure and
	// never the model's 512-token window.
	onnxByteBound := strconv.Itoa(onnxProviderUnderTest.runtime.tokenizer.maximumInputBytes())
	onnxTokenLimit := strconv.Itoa(onnxProviderUnderTest.runtime.tokenizer.maximumTokens)

	cases := []struct {
		name          string
		err           error
		wantReason    adapterr.EmbedRejectionReason
		wantFigures   []string
		forbidFigures []string
	}{
		{
			name:          "hosted endpoint rejection",
			err:           hostedErr,
			wantReason:    adapterr.EmbedRejectionContextLengthExceeded,
			wantFigures:   []string{"10000", "8192", "tokens"},
			forbidFigures: nil,
		},
		{
			name:          "in-process rejection",
			err:           onnxErr,
			wantReason:    adapterr.EmbedRejectionInputBytesExceeded,
			wantFigures:   []string{onnxByteBound, "bytes"},
			forbidFigures: []string{onnxTokenLimit + " tokens", onnxTokenLimit + "-token"},
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			var adapterErr *adapterr.AdapterError
			if !errors.As(testCase.err, &adapterErr) {
				t.Fatalf("error stayed untyped, so the boundary sanitizes it into an internal error: %v", testCase.err)
			}
			if adapterErr.Class != adapterr.ClassInvalidArgument {
				t.Fatalf("class = %q, want %q", adapterErr.Class, adapterr.ClassInvalidArgument)
			}
			if !adapterErr.SafeForClient {
				t.Fatal("a refused input must be safe to show the caller; otherwise the reason never leaves the daemon log")
			}
			if adapterErr.Code != string(testCase.wantReason) {
				t.Fatalf("code = %q, want the reason %q", adapterErr.Code, testCase.wantReason)
			}
			message := adapterr.SafeMessage(testCase.err)
			if !strings.Contains(message, string(testCase.wantReason)) {
				t.Fatalf("client message %q does not name the reason %q", message, testCase.wantReason)
			}
			for _, figure := range testCase.wantFigures {
				if !strings.Contains(message, figure) {
					t.Fatalf("client message %q does not carry the figure %q", message, figure)
				}
			}
			for _, wrongFigure := range testCase.forbidFigures {
				if strings.Contains(message, wrongFigure) {
					t.Fatalf("client message %q quotes %q, which is not the limit that refused the input", message, wrongFigure)
				}
			}
			for _, leak := range []string{"OpenAI", "ONNX", "onnx", "bge-small", "endpoint"} {
				if strings.Contains(message, leak) {
					t.Fatalf("client message %q names %q, so a caller can tell the providers apart", message, leak)
				}
			}
		})
	}
}

// TestHostedRefusalWithoutIndexProseStaysAnInvalidArgument covers the endpoint
// that answers context_length_exceeded without the "input at index N" wording
// the live endpoint happens to use. Classification comes from the HTTP status
// and the error code, so the caller still gets an actionable invalid argument
// rather than an embedder-rejected error that also degrades embedder health.
func TestHostedRefusalWithoutIndexProseStaysAnInvalidArgument(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusBadRequest)
		_, _ = writer.Write([]byte(`{"error":{"code":"context_length_exceeded","type":"invalid_request_error","message":"The input is too long for this model."}}`))
	}))
	defer server.Close()

	provider, err := newOpenAICompatibleProvider("test-key", server.URL, "model", 2, testEmbedTimeout)
	if err != nil {
		t.Fatalf("newOpenAICompatibleProvider returned error: %v", err)
	}

	vector, embedErr := provider.Embed(context.Background(), strings.Repeat("a", oversizedHostedInputBytes))
	if embedErr == nil {
		t.Fatal("Embed returned a vector for a query the endpoint refused")
	}
	if vector != nil {
		t.Fatalf("Embed returned %d values alongside the rejection", len(vector))
	}

	var adapterErr *adapterr.AdapterError
	if !errors.As(embedErr, &adapterErr) {
		t.Fatalf("a refusal worded without an index stayed untyped: %v", embedErr)
	}
	if adapterErr.Class != adapterr.ClassInvalidArgument {
		t.Fatalf("class = %q, want %q; the endpoint refused one input and is otherwise healthy", adapterErr.Class, adapterr.ClassInvalidArgument)
	}
	if adapterr.IsInfraFailure(embedErr) {
		t.Fatal("a refused input read as a shared-infrastructure failure, which degrades the embedder banner")
	}
	if errors.Is(embedErr, ErrEmbedderRejected) {
		t.Fatal("a refused input read as a rejecting embedder")
	}
	message := adapterr.SafeMessage(embedErr)
	if !strings.Contains(message, string(adapterr.EmbedRejectionContextLengthExceeded)) {
		t.Fatalf("client message %q does not name the reason", message)
	}
	// No figures could be read from that wording, so the message has to say so
	// rather than quote a zero the caller would act on.
	if !strings.Contains(message, "no size figures") {
		t.Fatalf("client message %q does not say the figures are missing", message)
	}
	if strings.Contains(message, " 0 ") || strings.Contains(message, "0-token") {
		t.Fatalf("client message %q quotes a zero figure", message)
	}
}

// TestHostedRefusalReportsAMeasuredCountTheEndpointGaveAsZero covers an endpoint
// that measures the input at zero tokens and says so. Zero is a count it did
// report, so the caller has to read it as the measurement rather than as a
// measurement that never arrived.
func TestHostedRefusalReportsAMeasuredCountTheEndpointGaveAsZero(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusBadRequest)
		_, _ = writer.Write([]byte(`{"error":{"code":"context_length_exceeded","type":"invalid_request_error","message":"This model's maximum context length is 8192 tokens, however the input resolved to 0 tokens."}}`))
	}))
	defer server.Close()

	provider, err := newOpenAICompatibleProvider("test-key", server.URL, "model", 2, testEmbedTimeout)
	if err != nil {
		t.Fatalf("newOpenAICompatibleProvider returned error: %v", err)
	}

	vector, embedErr := provider.Embed(context.Background(), "a query the endpoint measured at zero")
	if embedErr == nil {
		t.Fatal("Embed returned a vector for a query the endpoint refused")
	}
	if vector != nil {
		t.Fatalf("Embed returned %d values alongside the rejection", len(vector))
	}

	message := adapterr.SafeMessage(embedErr)
	if !strings.Contains(message, "0 tokens") {
		t.Fatalf("client message %q does not carry the zero the endpoint measured", message)
	}
	if !strings.Contains(message, "8192-token limit") {
		t.Fatalf("client message %q does not carry the maximum the endpoint reported", message)
	}
	if strings.Contains(message, "not reported") || strings.Contains(message, "no size figures") {
		t.Fatalf("client message %q calls a figure the endpoint reported missing", message)
	}
}

func TestOversizedInputRejectionClassification(t *testing.T) {
	t.Parallel()

	contextLength := &openai.Error{
		StatusCode: http.StatusBadRequest,
		Code:       "context_length_exceeded",
		Message:    "This model's maximum context length is 4096 tokens, however the input at index 2 resolved to 5000 tokens. Reduce the input length.",
	}
	rejection, ok := oversizedInputRejection(contextLength)
	if !ok {
		t.Fatal("context_length_exceeded 400 was not classified as a per-input rejection")
	}
	if rejection.index != 2 ||
		rejection.maxTokens != adapterr.ReportedFigure(4096) ||
		rejection.reportedTokens != adapterr.ReportedFigure(5000) {
		t.Fatalf("parsed rejection = %+v, want index 2, max 4096, reported 5000", rejection)
	}

	// A generic 400 (bad model, bad dimensions) is a server-side rejection, not a
	// per-input skip, so it must not be classified as droppable.
	genericBadRequest := &openai.Error{StatusCode: http.StatusBadRequest, Code: "invalid_request_error"}
	if _, ok := oversizedInputRejection(genericBadRequest); ok {
		t.Fatal("a generic 400 was wrongly classified as a per-input rejection")
	}

	// A 503 is a genuine transient server outage, never a per-input skip.
	serviceUnavailable := &openai.Error{StatusCode: http.StatusServiceUnavailable, Code: "context_length_exceeded"}
	if _, ok := oversizedInputRejection(serviceUnavailable); ok {
		t.Fatal("a 503 was wrongly classified as a per-input rejection")
	}

	// A non-API (transport) error is never a per-input rejection.
	if _, ok := oversizedInputRejection(errors.New("connection refused")); ok {
		t.Fatal("a transport error was wrongly classified as a per-input rejection")
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
			provider, err := NewProvider(context.Background(), config.Config{
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

	_, err := NewProvider(context.Background(), config.Config{
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

	provider, err := NewProvider(context.Background(), config.Config{
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

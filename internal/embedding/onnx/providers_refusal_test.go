package onnx

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/embedding"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

const (
	refusalTestTimeout       = 2 * time.Second
	oversizedHostedInputSize = 40000
)

// TestBothProvidersRefuseAnInputAsClientSafeInvalidArgument checks that both
// providers return a typed invalid-argument error for a refused single input.
// The client-safe message states the reason and the limit that refused the
// input, and never states the provider or model.
func TestBothProvidersRefuseAnInputAsClientSafeInvalidArgument(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusBadRequest)
		_, _ = writer.Write([]byte(`{"error":{"code":"context_length_exceeded","type":"invalid_request_error","message":"This model's maximum context length is 8192 tokens, however the input at index 0 resolved to 10000 tokens. Reduce the input length."}}`))
	}))
	defer server.Close()

	hostedProvider, err := embedding.NewOpenAICompatible(embedding.OpenAIOptions{
		APIKey:         "test-key",
		BaseURL:        server.URL,
		Model:          "text-embedding-3-small",
		Dimensions:     2,
		RequestTimeout: refusalTestTimeout,
	})
	if err != nil {
		t.Fatalf("NewOpenAICompatible returned error: %v", err)
	}
	_, hostedErr := hostedProvider.Embed(context.Background(), strings.Repeat("a", oversizedHostedInputSize))
	if hostedErr == nil {
		t.Fatal("the hosted provider accepted an input its endpoint rejected")
	}

	onnxProviderUnderTest := newUnloadedONNXProvider(t, offlinemodel.BGESmall)
	onnxOversized := strings.Repeat("a", onnxProviderUnderTest.runtime.tokenizer.maximumInputBytes()+1)
	_, onnxErr := onnxProviderUnderTest.Embed(context.Background(), onnxOversized)
	if onnxErr == nil {
		t.Fatal("the in-process provider accepted an input past its tokenizer bound")
	}

	// The byte ceiling refused the in-process input. The message states that byte
	// figure and not the model's token window.
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
				t.Fatalf("error is untyped: %v", testCase.err)
			}
			if adapterErr.Class != adapterr.ClassInvalidArgument {
				t.Fatalf("class = %q, want %q", adapterErr.Class, adapterr.ClassInvalidArgument)
			}
			if !adapterErr.SafeForClient {
				t.Fatal("a refused input must be safe to show the caller")
			}
			if adapterErr.Code != string(testCase.wantReason) {
				t.Fatalf("code = %q, want the reason %q", adapterErr.Code, testCase.wantReason)
			}
			message := adapterr.SafeMessage(testCase.err)
			if !strings.Contains(message, string(testCase.wantReason)) {
				t.Fatalf("client message %q does not state the reason %q", message, testCase.wantReason)
			}
			for _, figure := range testCase.wantFigures {
				if !strings.Contains(message, figure) {
					t.Fatalf("client message %q does not state the figure %q", message, figure)
				}
			}
			for _, wrongFigure := range testCase.forbidFigures {
				if strings.Contains(message, wrongFigure) {
					t.Fatalf("client message %q states %q, which did not refuse the input", message, wrongFigure)
				}
			}
			for _, leak := range []string{"OpenAI", "ONNX", "onnx", "bge-small", "endpoint"} {
				if strings.Contains(message, leak) {
					t.Fatalf("client message %q states %q, which identifies the provider", message, leak)
				}
			}
		})
	}
}

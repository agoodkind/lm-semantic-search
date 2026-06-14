// Package embedding implements text embedding providers for semantic indexing.
package embedding

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/metrics"
)

const maxEmbeddingTokens = 8192

// openAIProviderName is the only supported provider label.
const openAIProviderName = "OpenAI"

// Embedding retry policy for transient contention (HTTP 429/503). The endpoint
// is reachable but rate limiting or briefly unavailable, so the batch is retried
// with exponential backoff rather than failing the indexing job outright.
const (
	embedMaxAttempts = 4
	embedBackoffBase = 200 * time.Millisecond
)

// ErrEmbedderBusy marks a transient embedding failure: the endpoint answered but
// is at capacity (rate limited or temporarily unavailable). Callers branch on it
// with [errors.Is] to treat the failure as retryable rather than as the endpoint
// being unreachable.
var ErrEmbedderBusy = errors.New("embedding endpoint is at capacity")

// ErrEmbedderRejected marks a non-429 HTTP error from the endpoint: it is
// reachable but rejected the request (for example 400/401/500), distinct from a
// network failure that means the endpoint is unreachable.
var ErrEmbedderRejected = errors.New("embedding endpoint rejected the request")

// Provider generates dense embedding vectors.
type Provider interface {
	Embed(context.Context, string) ([]float32, error)
	EmbedBatch(context.Context, []string) ([][]float32, error)
	ProviderName() string
	// Health verifies the endpoint is reachable right now without performing an
	// embedding, so a caller can decide whether search can serve a query. A
	// non-nil result means the endpoint is unreachable or rejecting.
	Health(context.Context) error
}

// NewProvider constructs the configured embedding provider.
//
// Only the OpenAI-compatible HTTP adapter is supported. Users point it at any
// upstream that speaks the OpenAI embeddings API by setting OPENAI_BASE_URL
// (for example a self-hosted Ollama with `/v1/embeddings`, an OpenRouter
// account, or the OpenAI service itself).
func NewProvider(cfg config.Config) (Provider, error) {
	provider := strings.TrimSpace(cfg.EmbeddingProvider)
	if provider != "" && !strings.EqualFold(provider, "OpenAI") {
		slog.Error("embedding provider is not supported", "provider", provider, "err", errors.New("only OpenAI-compatible adapter is supported"))
		return nil, fmt.Errorf("embedding provider %q is not supported; only the OpenAI-compatible adapter is available", provider)
	}
	return newOpenAICompatibleProvider(cfg.OpenAIAPIKey, cfg.OpenAIBaseURL, cfg.EmbeddingModel, cfg.EmbeddingDimension)
}

type openAICompatibleProvider struct {
	name       string
	model      string
	dimensions int32
	client     openai.Client
}

func newOpenAICompatibleProvider(apiKey string, baseURL string, model string, dimensions int32) (Provider, error) {
	if strings.TrimSpace(apiKey) == "" {
		return nil, fmt.Errorf("%s embedding provider requires an API key", openAIProviderName)
	}
	if strings.TrimSpace(model) == "" {
		return nil, fmt.Errorf("%s embedding provider requires a model", openAIProviderName)
	}

	// Own the retry policy explicitly in embedWithRetry rather than letting the
	// SDK retry transparently, so transient 429/503 backoff is single-layered and
	// classified consistently instead of compounding with the SDK's own retries.
	requestOptions := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithMaxRetries(0),
	}
	if strings.TrimSpace(baseURL) != "" {
		requestOptions = append(requestOptions, option.WithBaseURL(baseURL))
	}

	return &openAICompatibleProvider{
		name:       openAIProviderName,
		model:      model,
		dimensions: dimensions,
		client:     openai.NewClient(requestOptions...),
	}, nil
}

func (provider *openAICompatibleProvider) ProviderName() string {
	return provider.name
}

// healthProbeTimeout bounds one liveness probe so a hung endpoint cannot stall
// the caller (a status read) waiting on the embedder.
const healthProbeTimeout = 2 * time.Second

// Health lists the endpoint's models (GET /v1/models), a metadata call that
// performs no embedding and so consumes no model capacity. Any error means the
// endpoint is unreachable or rejecting, which the caller treats as the embedder
// being unavailable for search. The caller debounces this probe, so a failing
// endpoint logs at most once per probe interval rather than on every request.
func (provider *openAICompatibleProvider) Health(ctx context.Context) error {
	probeCtx, cancel := context.WithTimeout(ctx, healthProbeTimeout)
	defer cancel()
	if _, err := provider.client.Models.List(probeCtx); err != nil {
		slog.WarnContext(ctx, "embedding endpoint health probe failed", "provider", provider.name, "model", provider.model, "err", err)
		return fmt.Errorf("%s embedding endpoint health probe: %w", provider.name, err)
	}
	return nil
}

func (provider *openAICompatibleProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	embeddings, err := provider.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("%s embedding provider returned no vectors", provider.name)
	}
	return embeddings[0], nil
}

func (provider *openAICompatibleProvider) EmbedBatch(ctx context.Context, texts []string) (vectors [][]float32, err error) {
	if len(texts) == 0 {
		return nil, nil
	}

	preprocessedTexts := make([]string, 0, len(texts))
	for _, text := range texts {
		preprocessedTexts = append(preprocessedTexts, preprocessText(text))
	}

	params := openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: preprocessedTexts,
		},
		Model:          provider.model,
		EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
	}
	if provider.dimensions > 0 {
		params.Dimensions = openai.Int(int64(provider.dimensions))
	}

	// Single choke point for every embedding call, so all per-batch latency and
	// counters flow through one defer regardless of which return fires.
	start := clock.Now()
	metrics.EmbedBatchStarted()
	defer func() {
		metrics.EmbedBatchDone(len(texts), clock.Now().Sub(start), err != nil)
	}()

	response, err := provider.embedWithRetry(ctx, params)
	if err != nil {
		return nil, err
	}
	if len(response.Data) != len(preprocessedTexts) {
		slog.ErrorContext(ctx, "embedding provider returned unexpected vector count", "provider", provider.name, "want", len(preprocessedTexts), "got", len(response.Data), "err", errors.New("vector count mismatch"))
		return nil, fmt.Errorf("%s embedding provider returned %d vectors for %d texts", provider.name, len(response.Data), len(preprocessedTexts))
	}

	vectors = make([][]float32, 0, len(response.Data))
	for _, item := range response.Data {
		vector := make([]float32, 0, len(item.Embedding))
		for _, value := range item.Embedding {
			vector = append(vector, float32(value))
		}
		vectors = append(vectors, vector)
	}
	return vectors, nil
}

// embedWithRetry issues the embeddings request, retrying transient contention
// (HTTP 429/503) with exponential backoff. Every error return is a typed
// [adapterr.AdapterError] so the daemon boundary classifies an embedding failure
// the same way regardless of which path (index or search) made the call. The
// embedding sentinels stay wrapped inside the cause so callers that branch with
// [errors.Is] keep working. Order matters: a cancellation is checked before the
// unreachable default so a cancelled request never reads as a down endpoint.
func (provider *openAICompatibleProvider) embedWithRetry(ctx context.Context, params openai.EmbeddingNewParams) (*openai.CreateEmbeddingResponse, error) {
	var lastErr error
	for attempt := 1; attempt <= embedMaxAttempts; attempt++ {
		response, err := provider.client.Embeddings.New(ctx, params)
		if err == nil {
			return response, nil
		}
		lastErr = err

		statusCode, transient := transientEmbedStatus(err)
		if !transient {
			slog.ErrorContext(ctx, "generate embeddings failed", "provider", provider.name, "model", provider.model, "status", statusCode, "err", err)
			var apiErr *openai.Error
			switch {
			case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
				return nil, adapterr.NewEmbedCancelled(fmt.Errorf("generate %s embeddings: %w", provider.name, err))
			case errors.As(err, &apiErr):
				// Reachable endpoint that answered with a non-429 HTTP error.
				return nil, adapterr.NewEmbedderRejected(fmt.Errorf("generate %s embeddings: %w: %w", provider.name, ErrEmbedderRejected, err))
			default:
				// Network failure: the endpoint is unreachable.
				return nil, adapterr.NewEmbedderUnreachable(fmt.Errorf("generate %s embeddings: %w", provider.name, err))
			}
		}
		if attempt == embedMaxAttempts {
			break
		}

		backoff := embedBackoff(attempt)
		slog.WarnContext(ctx, "embedding endpoint busy, retrying", "provider", provider.name, "model", provider.model, "status", statusCode, "attempt", attempt, "backoff", backoff)
		timer := time.NewTimer(backoff)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, adapterr.NewEmbedCancelled(fmt.Errorf("generate %s embeddings: %w", provider.name, ctx.Err()))
		case <-timer.C:
		}
	}

	statusCode, _ := transientEmbedStatus(lastErr)
	slog.ErrorContext(ctx, "embedding endpoint still busy after retries", "provider", provider.name, "model", provider.model, "status", statusCode, "attempts", embedMaxAttempts, "err", lastErr)
	return nil, adapterr.NewEmbedderBusy(fmt.Errorf("generate %s embeddings: %w: %w", provider.name, ErrEmbedderBusy, lastErr))
}

// transientEmbedStatus reports the HTTP status of an OpenAI API error and whether
// it indicates transient contention worth retrying (429 Too Many Requests or 503
// Service Unavailable). Non-API errors and other statuses are not transient.
func transientEmbedStatus(err error) (int, bool) {
	var apiErr *openai.Error
	if !errors.As(err, &apiErr) {
		return 0, false
	}
	switch apiErr.StatusCode {
	case http.StatusTooManyRequests, http.StatusServiceUnavailable:
		return apiErr.StatusCode, true
	default:
		return apiErr.StatusCode, false
	}
}

// embedBackoff returns the wait before the next attempt, doubling from the base
// (attempt 1 waits the base, attempt 2 twice the base, and so on).
func embedBackoff(attempt int) time.Duration {
	multiplier := 1 << (attempt - 1)
	return embedBackoffBase * time.Duration(multiplier)
}

func preprocessText(text string) string {
	if text == "" {
		return " "
	}

	maxCharacters := maxEmbeddingTokens * 4
	if len(text) > maxCharacters {
		return text[:maxCharacters]
	}
	return text
}

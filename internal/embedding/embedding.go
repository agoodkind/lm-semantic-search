// Package embedding implements text embedding providers for semantic indexing.
package embedding

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/metrics"
)

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

// embedCodeContextLengthExceeded is the OpenAI error code an embedding endpoint
// returns when a single input exceeds the model's context window. It is a
// permanent, per-input condition rather than a server outage, so the offending
// input is dropped and the rest of the batch is still embedded.
const embedCodeContextLengthExceeded = "context_length_exceeded"

// embedCodeInputContainsNUL is the reason code the in-process provider reports
// for an input carrying a NUL byte. The tokenizer binding is NUL-terminated, so
// it can only measure such an input up to its first NUL; the input is rejected
// whole rather than embedded as the prefix before that byte.
const embedCodeInputContainsNUL = "input_contains_nul_byte"

// embedCodeInputBytesExceeded is the reason code the in-process provider reports
// for an input larger than the byte ceiling it is willing to tokenize. Measuring
// such an input would materialize a complete encoding only to discard it as
// oversized, so it is rejected before the model runtime is touched.
const embedCodeInputBytesExceeded = "input_bytes_exceeded"

// SkippedInput identifies one input the embedding endpoint rejected as
// individually un-embeddable, for example a chunk whose token count exceeds the
// model's context window. The endpoint's own reported figures travel with it so
// a caller that holds the source chunk metadata can log the skip with full
// context.
type SkippedInput struct {
	// Index is the position of the skipped input in the EmbedBatch texts slice.
	Index int
	// Reason is the error code for the rejection, for example
	// "context_length_exceeded". A hosted endpoint supplies its own code; the
	// in-process provider supplies the code for the condition it detected.
	Reason string
	// ReportedTokens is the token count the endpoint measured for the input, or
	// zero when the endpoint did not report one.
	ReportedTokens int
	// MaxTokens is the model's maximum context length the endpoint reported, or
	// zero when the endpoint did not report one.
	MaxTokens int
}

// BatchResult is the outcome of an EmbedBatch call. Vectors holds one entry per
// input text, in input order, so it stays index-aligned with the caller's
// inputs; a skipped input has a nil Vectors entry. Skipped lists the inputs the
// endpoint rejected as individually un-embeddable, so a caller can log and drop
// those inputs without failing the whole batch or the indexing job.
type BatchResult struct {
	Vectors [][]float32
	Skipped []SkippedInput
}

// Provider generates dense embedding vectors.
type Provider interface {
	Embed(context.Context, string) ([]float32, error)
	EmbedBatch(context.Context, []string) (BatchResult, error)
	ProviderName() string
	// Health verifies the endpoint is reachable right now without performing an
	// embedding, so a caller can decide whether search can serve a query. A
	// non-nil result means the endpoint is unreachable or rejecting.
	Health(context.Context) error
}

// NewProvider constructs the configured embedding provider.
//
// The ONNX provider runs the embedded offline model in process. The default
// OpenAI-compatible adapter sends requests to the configured embeddings API.
func NewProvider(ctx context.Context, cfg config.Config) (Provider, error) {
	provider := strings.TrimSpace(cfg.EmbeddingProvider)
	if strings.EqualFold(provider, config.EmbeddingProviderONNX) {
		return newONNXProvider(ctx, cfg)
	}
	if provider != "" && !strings.EqualFold(provider, "OpenAI") {
		slog.ErrorContext(
			ctx,
			"embedding provider is not supported",
			"provider",
			provider,
			"err",
			errors.New("only ONNX and OpenAI-compatible adapters are supported"),
		)
		return nil, fmt.Errorf(
			"embedding provider %q is not supported; use %q or the OpenAI-compatible adapter",
			provider,
			config.EmbeddingProviderONNX,
		)
	}
	// A negative configured value would build a negative duration, which makes
	// context.WithTimeout expire immediately and fail every embed. Treat it as
	// disabled (unbounded) instead, matching the zero-disables semantics.
	requestTimeoutMS := max(cfg.EmbeddingRequestTimeoutMS, 0)
	requestTimeout := time.Duration(requestTimeoutMS) * time.Millisecond
	return newOpenAICompatibleProvider(cfg.OpenAIAPIKey, cfg.OpenAIBaseURL, cfg.EmbeddingModel, cfg.EmbeddingDimension, requestTimeout)
}

type openAICompatibleProvider struct {
	name       string
	model      string
	dimensions int32
	client     openai.Client
	// requestTimeout bounds one embedding HTTP request so an unresponsive endpoint
	// fails the call instead of hanging the goroutine forever. Zero leaves the
	// request unbounded (governed only by the caller's context).
	requestTimeout time.Duration
}

func newOpenAICompatibleProvider(apiKey string, baseURL string, model string, dimensions int32, requestTimeout time.Duration) (Provider, error) {
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
		name:           openAIProviderName,
		model:          model,
		dimensions:     dimensions,
		client:         openai.NewClient(requestOptions...),
		requestTimeout: requestTimeout,
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
	result, err := provider.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(result.Vectors) > 0 && result.Vectors[0] != nil {
		return result.Vectors[0], nil
	}
	if len(result.Skipped) == 0 {
		return nil, fmt.Errorf("%s embedding provider returned no vector for the input", provider.name)
	}
	// The endpoint rejected this single input as un-embeddable, for example a query
	// longer than the model's context window. Its own reported figures travel to
	// the caller so a person can shorten the input, rather than being sanitized
	// into an internal error nobody can act on.
	skipped := result.Skipped[0]
	slog.WarnContext(
		ctx,
		"embedding endpoint refused the input",
		"provider", provider.name,
		"model", provider.model,
		"reason", skipped.Reason,
		"reported_tokens", skipped.ReportedTokens,
		"model_max_tokens", skipped.MaxTokens,
	)
	return nil, rejectedInputError(skipped, fmt.Errorf(
		"%s embedding endpoint refused the input as %s",
		provider.name,
		skipped.Reason,
	))
}

// rejectedInputError renders one refused input as the typed, client-safe error
// both providers return from Embed. Embed's callers are the search paths, where
// a person is waiting on an answer, so the reason and the model's figures reach
// the client instead of a sanitized internal error. Both providers build the
// error here from the same SkippedInput, so a client cannot tell which provider
// refused the input; the provider-specific detail stays in the cause, which the
// boundary keeps in the daemon log.
func rejectedInputError(skipped SkippedInput, cause error) error {
	return adapterr.NewEmbedInputRejected(
		skipped.Reason,
		skipped.ReportedTokens,
		skipped.MaxTokens,
		cause,
	)
}

func (provider *openAICompatibleProvider) EmbedBatch(ctx context.Context, texts []string) (result BatchResult, err error) {
	if len(texts) == 0 {
		return BatchResult{Vectors: nil, Skipped: nil}, nil
	}

	preprocessedTexts := make([]string, 0, len(texts))
	for _, text := range texts {
		preprocessedTexts = append(preprocessedTexts, normalizeEmbeddingInput(text))
	}

	// Single choke point for every embedding call, so all per-batch latency and
	// counters flow through one defer regardless of which return fires.
	start := clock.Now()
	metrics.EmbedBatchStarted()
	defer func() {
		metrics.EmbedBatchDone(len(texts), clock.Now().Sub(start), err != nil)
	}()

	vectors := make([][]float32, len(texts))
	// surviving maps each position in the current request's input array back to
	// its original index in texts. An oversized input is removed from surviving
	// and its vectors slot stays nil, so the remaining inputs are re-requested
	// until the endpoint accepts them all or nothing is left to send.
	surviving := make([]int, 0, len(texts))
	for index := range preprocessedTexts {
		surviving = append(surviving, index)
	}
	var skipped []SkippedInput

	for len(surviving) > 0 {
		inputs := make([]string, 0, len(surviving))
		for _, originalIndex := range surviving {
			inputs = append(inputs, preprocessedTexts[originalIndex])
		}

		response, embedErr := provider.embedWithRetry(ctx, provider.embeddingParams(inputs))
		if embedErr == nil {
			if len(response.Data) != len(inputs) {
				slog.ErrorContext(ctx, "embedding provider returned unexpected vector count", "provider", provider.name, "want", len(inputs), "got", len(response.Data), "err", errors.New("vector count mismatch"))
				return BatchResult{}, fmt.Errorf("%s embedding provider returned %d vectors for %d texts", provider.name, len(response.Data), len(inputs))
			}
			for position, originalIndex := range surviving {
				vectors[originalIndex] = toFloat32Vector(response.Data[position].Embedding)
			}
			return BatchResult{Vectors: vectors, Skipped: skipped}, nil
		}

		rejection, isPerInput := oversizedInputRejection(embedErr)
		if !isPerInput || rejection.index < 0 || rejection.index >= len(surviving) {
			// Either a genuine server/transport failure, or a per-input rejection
			// whose offending input the endpoint did not identify. Neither can be
			// resolved by dropping one known input, so the typed error propagates
			// and the daemon classifies it (and may mark the embedder unhealthy).
			return BatchResult{}, embedErr
		}

		originalIndex := surviving[rejection.index]
		skipped = append(skipped, SkippedInput{
			Index:          originalIndex,
			Reason:         rejection.code,
			ReportedTokens: rejection.reportedTokens,
			MaxTokens:      rejection.maxTokens,
		})
		// vectors[originalIndex] stays nil to mark the input as skipped.
		surviving = append(surviving[:rejection.index], surviving[rejection.index+1:]...)
	}

	// Every input was skipped as un-embeddable; the batch still succeeds so the
	// job continues, and the caller drops the nil-vector inputs.
	return BatchResult{Vectors: vectors, Skipped: skipped}, nil
}

// embeddingParams builds the embeddings request for one input array.
func (provider *openAICompatibleProvider) embeddingParams(inputs []string) openai.EmbeddingNewParams {
	params := openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: inputs,
		},
		Model:          provider.model,
		EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
	}
	if provider.dimensions > 0 {
		params.Dimensions = openai.Int(int64(provider.dimensions))
	}
	return params
}

// toFloat32Vector narrows one endpoint embedding to the float32 storage form.
func toFloat32Vector(embedding []float64) []float32 {
	vector := make([]float32, 0, len(embedding))
	for _, value := range embedding {
		vector = append(vector, float32(value))
	}
	return vector
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
		// Bound the request when a timeout is configured so an unresponsive
		// endpoint fails the call instead of hanging the goroutine forever. The
		// external error is classified and wrapped below, never returned bare.
		requestCtx := ctx
		var cancel context.CancelFunc
		if provider.requestTimeout > 0 {
			requestCtx, cancel = context.WithTimeout(ctx, provider.requestTimeout)
		}
		response, err := provider.client.Embeddings.New(requestCtx, params)
		if cancel != nil {
			cancel()
		}
		if err == nil {
			return response, nil
		}
		lastErr = err

		statusCode, transient := transientEmbedStatus(err)
		if !transient {
			slog.ErrorContext(ctx, "generate embeddings failed", "provider", provider.name, "model", provider.model, "status", statusCode, "err", err)
			var apiErr *openai.Error
			switch {
			case ctx.Err() != nil:
				// The caller's context ended (a job cancel or daemon shutdown), so
				// report a cancellation, never a down endpoint.
				return nil, adapterr.NewEmbedCancelled(fmt.Errorf("generate %s embeddings: %w", provider.name, ctx.Err()))
			case errors.Is(err, context.DeadlineExceeded):
				// The per-request bound fired while the caller's context is still
				// live: the endpoint accepted the request but did not answer within
				// requestTimeout. Fail as unreachable so the job fails and retries
				// later instead of the goroutine hanging forever on a wedged endpoint.
				return nil, adapterr.NewEmbedderUnreachable(fmt.Errorf("generate %s embeddings: endpoint did not respond within %s: %w", provider.name, provider.requestTimeout, err))
			case errors.Is(err, context.Canceled):
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

// perInputRejection describes a per-input embedding rejection the endpoint
// blamed on one identified input. index is the offending input's position in the
// request's input array; the token figures are parsed from the endpoint message
// for logging and are zero when the message does not carry them.
type perInputRejection struct {
	code           string
	index          int
	reportedTokens int
	maxTokens      int
}

var (
	embedInputIndexPattern     = regexp.MustCompile(`input at index (\d+)`)
	embedMaxTokensPattern      = regexp.MustCompile(`maximum context length is (\d+) tokens`)
	embedResolvedTokensPattern = regexp.MustCompile(`resolved to (\d+) tokens`)
)

// oversizedInputRejection reports whether err is a per-input embedding rejection
// that names a single offending input, so that input can be dropped and the rest
// of the batch embedded. It matches an HTTP 400 whose OpenAI error code is
// context_length_exceeded: a permanent per-input condition (the input exceeds the
// model's context window), never a server outage. The offending index is relative
// to the request's input array, and the token figures come from the endpoint
// message; a message that omits the index yields index -1, which the caller
// treats as un-droppable and surfaces as an error.
func oversizedInputRejection(err error) (perInputRejection, bool) {
	var apiErr *openai.Error
	if !errors.As(err, &apiErr) {
		return perInputRejection{code: "", index: -1, reportedTokens: 0, maxTokens: 0}, false
	}
	if apiErr.StatusCode != http.StatusBadRequest || apiErr.Code != embedCodeContextLengthExceeded {
		return perInputRejection{code: "", index: -1, reportedTokens: 0, maxTokens: 0}, false
	}
	message := apiErr.Message
	return perInputRejection{
		code:           apiErr.Code,
		index:          parseFirstSubmatchInt(embedInputIndexPattern, message, -1),
		reportedTokens: parseFirstSubmatchInt(embedResolvedTokensPattern, message, 0),
		maxTokens:      parseFirstSubmatchInt(embedMaxTokensPattern, message, 0),
	}, true
}

// parseFirstSubmatchInt returns the first capture group of pattern in text as an
// int, or fallback when the pattern does not match or the capture is not a valid
// integer.
func parseFirstSubmatchInt(pattern *regexp.Regexp, text string, fallback int) int {
	match := pattern.FindStringSubmatch(text)
	if len(match) < 2 {
		return fallback
	}
	value, convErr := strconv.Atoi(match[1])
	if convErr != nil {
		return fallback
	}
	return value
}

// embedBackoff returns the wait before the next attempt, doubling from the base
// (attempt 1 waits the base, attempt 2 twice the base, and so on).
func embedBackoff(attempt int) time.Duration {
	multiplier := 1 << (attempt - 1)
	return embedBackoffBase * time.Duration(multiplier)
}

// normalizeEmbeddingInput prepares one input for the embeddings request. An empty
// input carries no content and the endpoint rejects it outright, so it becomes a
// single space. Every other input is sent exactly as the caller supplied it.
// Shortening a long input here would hand back a vector covering only the head of
// the content while the caller stores that vector under the whole content's
// identity; an input the endpoint cannot fit comes back as a
// context_length_exceeded rejection instead, which EmbedBatch reports through
// BatchResult.Skipped.
func normalizeEmbeddingInput(text string) string {
	if text == "" {
		return " "
	}
	return text
}

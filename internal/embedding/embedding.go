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
	"goodkind.io/lm-semantic-search/internal/model"
)

// openAIProviderName is what this adapter calls itself, taken from the closed
// set so the name it reports is the same value the configuration parses to.
const openAIProviderName = model.EmbeddingProviderOpenAI

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

// ErrEmbedderPaused marks a deliberate service pause reported by the endpoint.
// It is distinct from capacity and configuration rejection.
var ErrEmbedderPaused = errors.New("embedding endpoint paused the service")

// embedCodeContextLengthExceeded is the OpenAI error code an embedding endpoint
// returns when a single input exceeds the model's context window. It is a
// permanent, per-input condition rather than a server outage, so the offending
// input is dropped and the rest of the batch is still embedded. This is the wire
// value compared against the endpoint's response; the reason reported onward is
// this package's own [adapterr.EmbedRejectionContextLengthExceeded], never text
// the endpoint supplied.
const embedCodeContextLengthExceeded = "context_length_exceeded"

// embeddingEndpointErrorType is the endpoint's error-envelope discriminator.
// service_paused is a non-standard extension used by some OpenAI-compatible
// endpoints, so it is interpreted only at this HTTP adapter boundary.
//
// The discriminator is read from the NESTED error object, which is where the
// OpenAI error contract puts it and where compliant endpoints send it. Do not
// add a top-level parser for it.
//
// Two separate readers have concluded the opposite and proposed exactly that
// change, because the daemon log prints the pause as
// {"message":...,"type":"service_paused"} with no wrapper. That text is the
// client library rendering an already-parsed error, not the response body. The
// body on the wire carries the wrapper. Confirm against the endpoint's own
// source before believing otherwise; a log line and an error string are both
// renderings and neither is evidence of the wire format.
//
// The SDK settles it. openai-go builds its typed error in
// internal/requestconfig.Client.Execute by extracting the top-level "error" key
// with gjson and unmarshalling only that object, so openai.Error.Type is
// populated from the nested object and from nowhere else, and the same strip is
// why the rendered error string shows no wrapper. A top-level type field would
// leave openai.Error.Type empty and this discriminator would never match.
type embeddingEndpointErrorType string

const embeddingEndpointErrorTypeServicePaused embeddingEndpointErrorType = "service_paused"

const embeddingEndpointServicePausedHint = "leave low power mode or resume the embedding service"

// SkippedInput identifies one input the embedding endpoint rejected as
// individually un-embeddable, for example a chunk whose token count exceeds the
// model's context window. The endpoint's own reported figures travel with it so
// a caller that holds the source chunk metadata can log the skip with full
// context.
type SkippedInput struct {
	// Index is the position of the skipped input in the EmbedBatch texts slice.
	Index int
	// Reason is the closed-set reason for the rejection, for example
	// [adapterr.EmbedRejectionContextLengthExceeded]. It is always a value this
	// repository declares, never text an endpoint returned, so it stays safe to
	// show a client and stable to route on.
	Reason adapterr.EmbedRejectionReason
	// ReportedTokens is the token count the endpoint measured for the input. It
	// carries whether the endpoint reported a count at all, so a count it
	// genuinely reported as zero stays distinct from one it never sent.
	ReportedTokens adapterr.EmbedFigure
	// MaxTokens is the model's maximum context length the endpoint reported, and
	// likewise carries whether the endpoint reported it.
	MaxTokens adapterr.EmbedFigure
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
	ProviderName() model.EmbeddingProvider
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
	switch cfg.EmbeddingProvider {
	case config.EmbeddingProviderONNX:
		return newONNXProvider(ctx, cfg)
	case model.EmbeddingProviderNone, config.EmbeddingProviderOpenAI:
		// Both build the OpenAI-compatible adapter: an unnamed provider is the
		// historical default rather than an error.
	default:
		slog.ErrorContext(
			ctx,
			"embedding provider is not supported",
			"provider",
			cfg.EmbeddingProvider,
			"err",
			errors.New("only ONNX and OpenAI-compatible adapters are supported"),
		)
		return nil, fmt.Errorf(
			"embedding provider %q is not supported; use %q or %q",
			cfg.EmbeddingProvider,
			config.EmbeddingProviderONNX,
			config.EmbeddingProviderOpenAI,
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
	name       model.EmbeddingProvider
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

func (provider *openAICompatibleProvider) ProviderName() model.EmbeddingProvider {
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
	// longer than the model's context window. The reason and whatever figures the
	// endpoint reported travel to the caller so a person can shorten the input,
	// rather than being sanitized into an internal error nobody can act on.
	skipped := result.Skipped[0]
	slog.WarnContext(
		ctx,
		"embedding endpoint refused the input",
		"provider", provider.name,
		"model", provider.model,
		"reason", string(skipped.Reason),
		"reported_tokens", skipped.ReportedTokens,
		"model_max_tokens", skipped.MaxTokens,
	)
	return nil, adapterr.NewEmbedInputRejected(
		hostedInputRejection(skipped.Reason, skipped.ReportedTokens, skipped.MaxTokens),
		fmt.Errorf(
			"%s embedding endpoint refused the input as %s",
			provider.name,
			string(skipped.Reason),
		),
	)
}

// hostedInputRejection renders one endpoint refusal for the client-visible
// error. The endpoint reports its figures only in prose, so they are optional:
// when the endpoint reported neither the limit nor the measured count, the
// rejection is still a size refusal and says the figures are missing. The test is
// whether a figure arrived, not what it was, so a count the endpoint reported as
// zero is still reported onward as a figure.
func hostedInputRejection(
	reason adapterr.EmbedRejectionReason,
	reportedTokens adapterr.EmbedFigure,
	maxTokens adapterr.EmbedFigure,
) adapterr.EmbedInputRejection {
	if reason == adapterr.EmbedRejectionEmptyContent {
		// Nothing was measured and no ceiling was reached, so naming a size limit
		// here would send the reader after a shorter input when the input was
		// already as short as an input can be.
		return adapterr.EmbedInputRejection{
			Reason:   reason,
			Limit:    adapterr.EmbedLimitNone,
			Measured: adapterr.UnreportedFigure(),
			Maximum:  adapterr.UnreportedFigure(),
		}
	}
	limit := adapterr.EmbedLimitTokens
	if !maxTokens.Reported && !reportedTokens.Reported {
		limit = adapterr.EmbedLimitUnreported
	}
	return adapterr.EmbedInputRejection{
		Reason:   reason,
		Limit:    limit,
		Measured: reportedTokens,
		Maximum:  maxTokens,
	}
}

func (provider *openAICompatibleProvider) EmbedBatch(ctx context.Context, texts []string) (result BatchResult, err error) {
	if len(texts) == 0 {
		return BatchResult{Vectors: nil, Skipped: nil}, nil
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
	//
	// An input with nothing to embed never enters surviving, so it costs no
	// endpoint call at all. It is reported as skipped exactly like an input the
	// endpoint refused, which keeps one promise for every nil vector: the caller
	// reads why, and it holds whether the refusal happened here or upstream.
	surviving := make([]int, 0, len(texts))
	var skipped []SkippedInput
	for index, text := range texts {
		if hasNothingToEmbed(text) {
			skipped = append(skipped, SkippedInput{
				Index:          index,
				Reason:         adapterr.EmbedRejectionEmptyContent,
				ReportedTokens: adapterr.UnreportedFigure(),
				MaxTokens:      adapterr.UnreportedFigure(),
			})
			continue
		}
		surviving = append(surviving, index)
	}
	if refused := len(skipped); refused > 0 {
		metrics.EmbedInputsRefusedEmpty(refused)
	}

	for len(surviving) > 0 {
		inputs := make([]string, 0, len(surviving))
		for _, originalIndex := range surviving {
			inputs = append(inputs, texts[originalIndex])
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
		if !isPerInput {
			// A genuine server or transport failure. The typed error propagates and
			// the daemon classifies it, which may mark the embedder unhealthy.
			return BatchResult{}, embedErr
		}

		position := rejection.index
		if position < 0 && len(surviving) == 1 {
			// The endpoint refused a single-input request, so the offending input is
			// unambiguous even though its message named no index.
			position = 0
		}
		if position < 0 || position >= len(surviving) {
			// The endpoint refused the request as carrying an over-long input but did
			// not say which one, and several are in flight, so no single input can be
			// dropped. The fault is still the request's and not the endpoint's, so it
			// must not read as a rejecting embedder and must not degrade its health.
			slog.WarnContext(ctx, "embedding endpoint refused a batch without naming the input", "provider", provider.name, "model", provider.model, "inputs", len(surviving), "err", embedErr)
			return BatchResult{}, adapterr.NewEmbedInputRejected(
				hostedInputRejection(
					adapterr.EmbedRejectionContextLengthExceeded,
					rejection.reportedTokens,
					rejection.maxTokens,
				),
				embedErr,
			)
		}

		originalIndex := surviving[position]
		skipped = append(skipped, SkippedInput{
			Index:          originalIndex,
			Reason:         adapterr.EmbedRejectionContextLengthExceeded,
			ReportedTokens: rejection.reportedTokens,
			MaxTokens:      rejection.maxTokens,
		})
		// vectors[originalIndex] stays nil to mark the input as skipped.
		surviving = append(surviving[:position], surviving[position+1:]...)
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

		if paused, servicePaused := embedServicePaused(err); servicePaused {
			slog.WarnContext(
				ctx,
				"embedding endpoint reported service paused",
				"provider", provider.name,
				"model", provider.model,
				"status", paused.statusCode,
				"message", paused.message,
			)
			cause := fmt.Errorf(
				"generate %s embeddings: %w: %w",
				provider.name,
				ErrEmbedderPaused,
				err,
			)
			return nil, adapterr.NewEmbedderPaused(
				"embedding endpoint reported: "+paused.message,
				embeddingEndpointServicePausedHint,
				cause,
			)
		}

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

// servicePausedReport carries what the endpoint actually said when it reported
// a deliberate pause: the status it answered with and its own message. The
// status is read from the response rather than assumed, so a log line never
// states a status the code did not observe.
type servicePausedReport struct {
	statusCode int
	message    string
}

func embedServicePaused(err error) (servicePausedReport, bool) {
	var apiErr *openai.Error
	if !errors.As(err, &apiErr) {
		return servicePausedReport{statusCode: 0, message: ""}, false
	}
	if embeddingEndpointErrorType(apiErr.Type) != embeddingEndpointErrorTypeServicePaused {
		return servicePausedReport{statusCode: 0, message: ""}, false
	}
	message := strings.TrimSpace(apiErr.Message)
	if message == "" {
		message = "the embedding service is paused"
	}
	return servicePausedReport{statusCode: apiErr.StatusCode, message: message}, true
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
// reported. Only the figures live here: the classification comes from the HTTP
// status and the error code field, which are machine-readable, while index and
// the token counts are parsed from the endpoint's prose, which is not. index is
// -1 and each token count is absent when the message does not carry it, and
// neither absence changes how the refusal is classified.
type perInputRejection struct {
	index          int
	reportedTokens adapterr.EmbedFigure
	maxTokens      adapterr.EmbedFigure
}

var (
	embedInputIndexPattern     = regexp.MustCompile(`input at index (\d+)`)
	embedMaxTokensPattern      = regexp.MustCompile(`maximum context length is (\d+) tokens`)
	embedResolvedTokensPattern = regexp.MustCompile(`resolved to (\d+) tokens`)
)

// oversizedInputRejection reports whether err is a per-input embedding rejection:
// an HTTP 400 whose OpenAI error code is context_length_exceeded, a permanent
// per-input condition (the input exceeds the model's context window) and never a
// server outage. Both of those are machine-readable fields, so the classification
// never depends on how the endpoint worded its message. The returned index and
// token counts are read from that prose as a convenience and are absent when the
// wording does not carry them; an absent index only limits which input can be
// dropped, it does not make the refusal any less per-input.
func oversizedInputRejection(err error) (perInputRejection, bool) {
	var apiErr *openai.Error
	if !errors.As(err, &apiErr) {
		return unparsedInputRejection(), false
	}
	if apiErr.StatusCode != http.StatusBadRequest || apiErr.Code != embedCodeContextLengthExceeded {
		return unparsedInputRejection(), false
	}
	message := apiErr.Message
	index, indexFound := parseFirstSubmatchInt(embedInputIndexPattern, message)
	if !indexFound {
		index = embedInputIndexUnnamed
	}
	return perInputRejection{
		index:          index,
		reportedTokens: parseReportedFigure(embedResolvedTokensPattern, message),
		maxTokens:      parseReportedFigure(embedMaxTokensPattern, message),
	}, true
}

// embedInputIndexUnnamed marks a refusal whose message named no input position,
// so no single input of a multi-input request can be dropped.
const embedInputIndexUnnamed = -1

// unparsedInputRejection is the empty rejection returned beside a false result,
// so a caller that ignores the boolean still sees no figures rather than zeros.
func unparsedInputRejection() perInputRejection {
	return perInputRejection{
		index:          embedInputIndexUnnamed,
		reportedTokens: adapterr.UnreportedFigure(),
		maxTokens:      adapterr.UnreportedFigure(),
	}
}

// parseReportedFigure reads one size figure out of the endpoint's prose. A figure
// the message does not carry, or carries in a form that is not a readable
// integer, comes back absent rather than as a zero, so a figure the endpoint
// genuinely reported as zero stays distinguishable from one it never reported.
func parseReportedFigure(pattern *regexp.Regexp, text string) adapterr.EmbedFigure {
	value, found := parseFirstSubmatchInt(pattern, text)
	if !found {
		return adapterr.UnreportedFigure()
	}
	return adapterr.ReportedFigure(value)
}

// parseFirstSubmatchInt returns the first capture group of pattern in text as an
// int, and whether the pattern matched with a capture that read as an integer.
func parseFirstSubmatchInt(pattern *regexp.Regexp, text string) (int, bool) {
	match := pattern.FindStringSubmatch(text)
	if len(match) < 2 {
		return 0, false
	}
	value, convErr := strconv.Atoi(match[1])
	if convErr != nil {
		return 0, false
	}
	return value, true
}

// embedBackoff returns the wait before the next attempt, doubling from the base
// (attempt 1 waits the base, attempt 2 twice the base, and so on).
func embedBackoff(attempt int) time.Duration {
	multiplier := 1 << (attempt - 1)
	return embedBackoffBase * time.Duration(multiplier)
}

// hasNothingToEmbed reports whether an input carries no character a vector could
// describe. It protects one invariant: a returned vector always covers the whole
// input, and an input with no non-whitespace character offers nothing to cover,
// so embedding it would spend a model call to store a vector that can only be
// noise in a later search.
//
// Whitespace-only counts as empty because it is the same degenerate case. The
// tokenizer reduces both to the model's special tokens alone, so they yield the
// same vector, and treating only the zero-length string as empty would let an
// input of one newline through a guard whose whole purpose is to stop content
// that cannot be embedded.
//
// The question is deliberately physical and is asked of the string alone. This
// is not the place to decide whether content is worth indexing. That is a
// preference, it belongs to whoever assembles the input, and it is settled before
// anything reaches a provider.
func hasNothingToEmbed(text string) bool {
	return strings.TrimSpace(text) == ""
}

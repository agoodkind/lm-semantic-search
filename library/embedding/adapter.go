// Package embedding exposes the production OpenAI-compatible embedding
// provider as a [library.Embedder]. It links no native code. Package
// library/embedding/onnx exposes the in-process ONNX provider.
package embedding

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"

	internalembedding "goodkind.io/lm-semantic-search/internal/embedding"
	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/internal/embedadapter"
)

// Every adapter error from this package and from library/embedding/onnx wraps
// exactly one of these sentinels. A caller classifies a failure with
// [errors.Is].
var (
	// ErrEmbedderPaused reports an endpoint that deliberately paused service.
	ErrEmbedderPaused = embedadapter.ErrEmbedderPaused
	// ErrEmbedderBusy reports an endpoint still at capacity after every
	// attempt.
	ErrEmbedderBusy = embedadapter.ErrEmbedderBusy
	// ErrEmbedderRejected reports a provider that refused the request or one
	// input, or answered without a valid vector for every input.
	ErrEmbedderRejected = embedadapter.ErrEmbedderRejected
	// ErrEmbedCancelled reports a request that ended with its context.
	ErrEmbedCancelled = embedadapter.ErrEmbedCancelled
	// ErrEmbedderUnreachable reports an endpoint that did not answer.
	ErrEmbedderUnreachable = embedadapter.ErrEmbedderUnreachable
)

// OpenAIConfig configures the OpenAI-compatible adapter. The caller resolves
// any credential reference and passes the credential value in APIKey.
type OpenAIConfig struct {
	// BaseURL is the endpoint root, for example "http://localhost:5400/v1".
	BaseURL string
	// APIKey is the bearer credential. Empty sends no Authorization header.
	APIKey string
	// Model is the embedding model name.
	Model string
	// Dimension is the vector dimension the endpoint returns. The adapter
	// requests it and rejects any vector of another length.
	Dimension int
	// RequestTimeout bounds one HTTP request. Zero leaves the request bounded
	// only by the caller's context.
	RequestTimeout time.Duration
	// MaxAttempts is the number of attempts for a busy endpoint. Zero selects
	// 4.
	MaxAttempts int
	// BackoffBase is the wait before the second attempt, doubled for each
	// later attempt. Zero selects 200 milliseconds.
	BackoffBase time.Duration
}

// NewOpenAI returns the production OpenAI-compatible adapter. A negative
// timeout, attempt count, or backoff, an empty base URL or model, or a
// nonpositive dimension returns an error that wraps
// [library.ErrInvalidRequest].
func NewOpenAI(ctx context.Context, config OpenAIConfig) (library.Embedder, error) {
	if message := openAIConfigViolation(config); message != "" {
		err := fmt.Errorf("%w: embedding: %s", library.ErrInvalidRequest, message)
		slog.WarnContext(ctx, "embedding adapter configuration rejected", "err", err)
		return nil, err
	}
	maxAttempts := config.MaxAttempts
	if maxAttempts == 0 {
		maxAttempts = internalembedding.DefaultEmbedMaxAttempts
	}
	backoffBase := config.BackoffBase
	if backoffBase == 0 {
		backoffBase = internalembedding.DefaultEmbedBackoffBase
	}
	provider, err := internalembedding.NewOpenAICompatibleProvider(internalembedding.OpenAICompatibleOptions{
		APIKey:         config.APIKey,
		BaseURL:        config.BaseURL,
		Model:          config.Model,
		Dimensions:     config.Dimension,
		RequestTimeout: config.RequestTimeout,
		MaxAttempts:    maxAttempts,
		BackoffBase:    backoffBase,
	})
	if err != nil {
		slog.ErrorContext(ctx, "construct OpenAI-compatible embedding adapter failed", "model", config.Model, "err", err)
		return nil, fmt.Errorf("%w: embedding: %w", library.ErrInvalidRequest, err)
	}
	return embedadapter.New(provider, config.Dimension), nil
}

func openAIConfigViolation(config OpenAIConfig) string {
	switch {
	case strings.TrimSpace(config.BaseURL) == "":
		return "OpenAIConfig BaseURL is empty"
	case strings.TrimSpace(config.Model) == "":
		return "OpenAIConfig Model is empty"
	case config.Dimension <= 0:
		return fmt.Sprintf("OpenAIConfig Dimension %d must be positive", config.Dimension)
	case config.RequestTimeout < 0:
		return fmt.Sprintf("OpenAIConfig RequestTimeout %s must not be negative", config.RequestTimeout)
	case config.MaxAttempts < 0:
		return fmt.Sprintf("OpenAIConfig MaxAttempts %d must not be negative", config.MaxAttempts)
	case config.BackoffBase < 0:
		return fmt.Sprintf("OpenAIConfig BackoffBase %s must not be negative", config.BackoffBase)
	default:
		return ""
	}
}

// Package embedadapter converts an internal embedding provider into the
// [library.Embedder] contract. The public embedding packages share it.
package embedadapter

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/library/observation"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/embedding"
	"goodkind.io/lm-semantic-search/library"
)

// Each adapter error wraps exactly one of these sentinels. The public
// embedding packages export them.
var (
	ErrEmbedderPaused      = errors.New("embedding: endpoint paused")
	ErrEmbedderBusy        = errors.New("embedding: endpoint busy")
	ErrEmbedderRejected    = errors.New("embedding: request rejected")
	ErrEmbedCancelled      = errors.New("embedding: request cancelled")
	ErrEmbedderUnreachable = errors.New("embedding: endpoint unreachable")
)

// New returns a [library.Embedder] for provider. A positive dimension is the
// required length of every vector. Zero leaves the length to the provider.
func New(provider embedding.Provider, dimension int) library.Embedder {
	return NewObserved(provider, dimension, nil)
}

// NewObserved constructs an adapter that reports final batch validation.
func NewObserved(provider embedding.Provider, dimension int, observer observation.Observer) library.Embedder {
	return &adapter{provider: provider, dimension: dimension, observer: observer}
}

type adapter struct {
	observer  observation.Observer
	provider  embedding.Provider
	dimension int
}

// EmbedBatch returns one vector for every text. A provider that skips an input,
// returns a nil vector, or returns a vector of the wrong length makes the whole
// batch fail with [ErrEmbedderRejected]. It never returns a shorter slice.
func (adapter *adapter) EmbedBatch(ctx context.Context, texts []string) (_ [][]float32, err error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}
	result, err := adapter.provider.EmbedBatch(ctx, texts)
	if err != nil {
		classified := fmt.Errorf("%w: %w", providerErrorSentinel(ctx, err), err)
		slog.WarnContext(ctx, "embedding batch failed", "inputs", len(texts), "err", classified)
		return nil, classified
	}
	ctx, span := observation.Start(ctx, adapter.observer, observation.EmbeddingValidation)
	counts := observation.EmbeddingData{Requested: len(texts), Returned: len(result.Vectors), Validation: observation.AdapterValidation}
	defer func() { span.End(ctx, err, observation.Data{Embedding: counts}) }()
	if len(result.Skipped) > 0 {
		skipped := result.Skipped[0]
		err := fmt.Errorf(
			"%w: provider refused input %d of %d as %s; %d inputs refused",
			ErrEmbedderRejected,
			skipped.Index,
			len(texts),
			skipped.Reason,
			len(result.Skipped),
		)
		slog.WarnContext(ctx, "embedding batch refused inputs", "inputs", len(texts), "refused", len(result.Skipped), "err", err)
		return nil, err
	}
	if len(result.Vectors) != len(texts) {
		err := fmt.Errorf("%w: provider returned %d vectors for %d inputs", ErrEmbedderRejected, len(result.Vectors), len(texts))
		slog.ErrorContext(ctx, "embedding batch vector count mismatch", "err", err)
		return nil, err
	}
	for index, vector := range result.Vectors {
		if vector == nil {
			err := fmt.Errorf("%w: provider returned no vector for input %d of %d", ErrEmbedderRejected, index, len(texts))
			slog.ErrorContext(ctx, "embedding batch missing vector", "err", err)
			return nil, err
		}
		if adapter.dimension > 0 && len(vector) != adapter.dimension {
			err := fmt.Errorf(
				"%w: provider returned %d values for input %d, want %d",
				ErrEmbedderRejected,
				len(vector),
				index,
				adapter.dimension,
			)
			slog.ErrorContext(ctx, "embedding batch dimension mismatch", "err", err)
			return nil, err
		}
	}
	counts.Validated = len(result.Vectors)
	return result.Vectors, nil
}

// providerClassSentinels maps each provider error class that has its own
// adapter sentinel. Every other class maps to [ErrEmbedderRejected].
var providerClassSentinels = map[adapterr.Class]error{
	adapterr.ClassEmbedderPaused:      ErrEmbedderPaused,
	adapterr.ClassEmbedderBusy:        ErrEmbedderBusy,
	adapterr.ClassEmbedCancelled:      ErrEmbedCancelled,
	adapterr.ClassEmbedderUnreachable: ErrEmbedderUnreachable,
}

// providerErrorSentinel returns the one adapter sentinel for a provider error.
// The provider's typed class selects the sentinel. Without a typed class, a
// context cancellation or deadline selects [ErrEmbedCancelled] and any other
// failure selects [ErrEmbedderRejected].
func providerErrorSentinel(ctx context.Context, err error) error {
	var adapterError *adapterr.AdapterError
	if errors.As(err, &adapterError) {
		if sentinel, found := providerClassSentinels[adapterError.Class]; found {
			return sentinel
		}
		return ErrEmbedderRejected
	}
	if ctx.Err() != nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return ErrEmbedCancelled
	}
	return ErrEmbedderRejected
}

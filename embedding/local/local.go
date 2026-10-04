// Package local builds the embedding provider that runs an ONNX model in the
// calling process. The provider sends no request to an embedding service. The
// first use of a model downloads its files into the cache root, and every later
// use reads the cached files.
package local

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"goodkind.io/lm-semantic-search/embedding"
	"goodkind.io/lm-semantic-search/internal/embedding/onnx"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

// ErrModelUnavailable reports that a model file is absent from the cache root
// and its download failed.
var ErrModelUnavailable = onnx.ErrArtifactUnavailable

// Options configures [New].
type Options struct {
	// Model is a name from [Models]. An empty value selects [DefaultModel].
	Model string
	// CacheRoot is the directory that stores the downloaded model files. It is
	// required.
	CacheRoot string
}

// Model is the fixed description of one supported model.
type Model struct {
	Name string
	// Dimension is the width of every vector the model returns.
	Dimension int
	// QueryPrefix is the text a caller prepends to a search query before
	// embedding it. A stored document takes no prefix.
	QueryPrefix string
	// MaximumTokens is the longest input the model embeds. The provider refuses
	// a longer input and does not embed a prefix of it.
	MaximumTokens int
}

// DefaultModel is the model an empty [Options.Model] selects.
const DefaultModel = offlinemodel.DefaultName

// Models returns the supported model names.
func Models() []string {
	return offlinemodel.Names()
}

// Describe returns the description of a supported model. An empty name selects
// [DefaultModel].
func Describe(name string) (Model, error) {
	preset, err := offlinemodel.Resolve(name)
	if err != nil {
		slog.Error("describe local embedding model failed", "model", name, "err", err)
		return Model{}, fmt.Errorf("describe local embedding model: %w", err)
	}
	return Model{
		Name:          preset.Name,
		Dimension:     int(preset.Dimension),
		QueryPrefix:   preset.QueryPrefix,
		MaximumTokens: int(preset.MaximumTokens),
	}, nil
}

// New returns the in-process provider for a model. Every returned vector has
// unit length. Providers for the same model and cache root share one loaded
// model for the life of the process.
func New(ctx context.Context, options Options) (embedding.Provider, error) {
	if strings.TrimSpace(options.CacheRoot) == "" {
		err := errors.New("local embedding provider requires a cache root")
		slog.ErrorContext(ctx, "create local embedding provider failed", "model", options.Model, "err", err)
		return nil, err
	}
	provider, err := onnx.NewProviderForModel(ctx, options.Model, options.CacheRoot)
	if err != nil {
		slog.ErrorContext(ctx, "create local embedding provider failed", "model", options.Model, "err", err)
		return nil, fmt.Errorf("create local embedding provider: %w", err)
	}
	return provider, nil
}

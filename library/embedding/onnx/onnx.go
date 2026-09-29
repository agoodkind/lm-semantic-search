// Package onnx exposes the production in-process ONNX embedding provider as a
// [library.Embedder] and its tokenizer as a [library.Tokenizer]. It links ONNX
// Runtime and the tokenizers library. A binary that imports it loads the ONNX
// Runtime shared library at launch.
//
// Errors from adapters that [New] returns wrap the sentinels exported by
// package library/embedding.
package onnx

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	internalonnx "goodkind.io/lm-semantic-search/internal/embedding/onnx"
	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/internal/embedadapter"
)

// Config selects an offline model preset and its artifact cache.
type Config struct {
	// ModelName is the preset name. Empty selects the default preset.
	ModelName string
	// ModelCacheRoot is the directory for downloaded, checksum-verified model
	// artifacts.
	ModelCacheRoot string
}

// New returns the production in-process ONNX adapter for one offline model
// preset. It downloads and checksum-verifies missing artifacts under
// ModelCacheRoot. The native session stays cached for the process lifetime.
func New(ctx context.Context, config Config) (library.Embedder, error) {
	if err := validateConfig(ctx, config); err != nil {
		return nil, err
	}
	provider, err := internalonnx.NewProvider(ctx, config.ModelName, config.ModelCacheRoot)
	if err != nil {
		slog.ErrorContext(ctx, "construct ONNX embedding adapter failed", "model", config.ModelName, "err", err)
		return nil, fmt.Errorf("embedding: %w", err)
	}
	return embedadapter.New(provider, 0), nil
}

// Tokenizer counts tokens with the tokenizer of one offline model preset. It
// implements [library.Tokenizer]. A caller passes MaxTokens and MaxInputBytes
// as [library.PrepareRequest] limits.
type Tokenizer struct {
	counter *internalonnx.TokenCounter
}

// NewTokenizer returns the tokenizer of one offline model preset. It shares
// the cached native session with [New] for the same model.
func NewTokenizer(ctx context.Context, config Config) (*Tokenizer, error) {
	if err := validateConfig(ctx, config); err != nil {
		return nil, err
	}
	counter, err := internalonnx.NewTokenCounter(ctx, config.ModelName, config.ModelCacheRoot)
	if err != nil {
		slog.ErrorContext(ctx, "construct ONNX tokenizer failed", "model", config.ModelName, "err", err)
		return nil, fmt.Errorf("embedding: %w", err)
	}
	return &Tokenizer{counter: counter}, nil
}

// CountTokens returns the token count the ONNX provider measures for text,
// including the model's special tokens. Text with a NUL byte returns an error.
func (tokenizer *Tokenizer) CountTokens(ctx context.Context, text string) (int, error) {
	count, err := tokenizer.counter.CountTokens(ctx, text)
	if err != nil {
		slog.WarnContext(ctx, "count ONNX tokens failed", "input_bytes", len(text), "err", err)
		return 0, fmt.Errorf("embedding: %w", err)
	}
	return count, nil
}

// MaxTokens returns the model's maximum token count for one input.
func (tokenizer *Tokenizer) MaxTokens() int {
	return tokenizer.counter.MaxTokens()
}

// MaxInputBytes returns the byte ceiling the ONNX provider applies to one
// input before tokenizing it.
func (tokenizer *Tokenizer) MaxInputBytes() int {
	return tokenizer.counter.MaxInputBytes()
}

func validateConfig(ctx context.Context, config Config) error {
	if strings.TrimSpace(config.ModelCacheRoot) != "" {
		return nil
	}
	err := fmt.Errorf("%w: embedding: onnx.Config ModelCacheRoot is empty", library.ErrInvalidRequest)
	slog.WarnContext(ctx, "ONNX embedding configuration rejected", "err", err)
	return err
}

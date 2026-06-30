//go:build !(darwin && arm64)

package cbm

import (
	"context"
	"errors"
)

// Engine is unavailable off darwin/arm64.
type Engine struct{}

// Open reports that the cbm graph engine is not available on this platform.
func Open(project string) (*Engine, error) {
	_ = project
	return nil, errors.New("cbm graph engine is only supported on darwin/arm64")
}

// Index reports that the cbm graph engine is not available on this platform.
func (*Engine) Index(ctx context.Context, repoPath string, mode string) error {
	_ = ctx
	_ = repoPath
	_ = mode
	return errors.New("cbm graph engine is only supported on darwin/arm64")
}

// Tool reports that the cbm graph engine is not available on this platform.
func (*Engine) Tool(name string, args string) (string, error) {
	_ = name
	_ = args
	return "", errors.New("cbm graph engine is only supported on darwin/arm64")
}

// Close is a no-op off darwin/arm64.
func (*Engine) Close() {}

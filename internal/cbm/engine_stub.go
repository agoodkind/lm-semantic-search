//go:build !(darwin && arm64)

package cbm

import (
	"context"
)

// Engine is unavailable off darwin/arm64.
type Engine struct{}

// Open reports that the cbm graph engine is not available on this platform.
func Open(project string) (*Engine, error) {
	_ = project
	return nil, ErrUnsupportedPlatform
}

// Index reports that the cbm graph engine is not available on this platform.
func (*Engine) Index(ctx context.Context, repoPath string, mode string) error {
	_ = ctx
	_ = repoPath
	_ = mode
	return ErrUnsupportedPlatform
}

// Tool reports that the cbm graph engine is not available on this platform.
func (*Engine) Tool(name string, args string) (string, error) {
	_ = name
	_ = args
	return "", ErrUnsupportedPlatform
}

// Close is a no-op off darwin/arm64.
func (*Engine) Close() {}

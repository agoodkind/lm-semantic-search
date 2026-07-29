package model

import (
	"fmt"
	"strings"
)

// VectorBackend names a vector store implementation. It is a closed set, so a
// value that reaches a comparison, a stored record, or a config digest has
// exactly one spelling and cannot differ from another value that means the same
// store.
type VectorBackend string

const (
	// VectorBackendMilvus is the Milvus-backed store, the default.
	VectorBackendMilvus VectorBackend = "milvus"
	// VectorBackendLocal is the embedded store the offline profile uses.
	VectorBackendLocal VectorBackend = "local"
)

// EmbeddingProvider names an embedding implementation. It is a closed set for
// the same reason as [VectorBackend]. The zero value means no embedder, which a
// caller needs to be able to tell apart from any named one.
type EmbeddingProvider string

const (
	// EmbeddingProviderNone is the zero value: no embedder exists.
	EmbeddingProviderNone EmbeddingProvider = ""
	// EmbeddingProviderOpenAI is the OpenAI-compatible adapter.
	EmbeddingProviderOpenAI EmbeddingProvider = "OpenAI"
	// EmbeddingProviderONNX is the in-process ONNX runtime.
	EmbeddingProviderONNX EmbeddingProvider = "onnx"
)

// String returns the backend's canonical spelling.
func (backend VectorBackend) String() string {
	return string(backend)
}

// String returns the provider's canonical spelling, empty when none.
func (provider EmbeddingProvider) String() string {
	return string(provider)
}

// ParseEmbeddingProvider resolves a configured provider name to its canonical
// value, ignoring surrounding space and letter case. An empty name resolves to
// [EmbeddingProviderNone] rather than an error, because a config that names no
// provider is a legitimate state the caller decides what to do with.
//
// Parsing here rather than comparing raw strings at each use is what keeps a
// configured spelling and a constructed embedder's own name from disagreeing.
// The two reach the same config digest, and a digest that moves discards the
// checkpoint it gates, so every file would be embedded again.
//
// [VectorBackend] needs no equivalent: no environment variable and no persisted
// config field names a backend, so its value only ever comes from the constants
// above and no variant spelling can exist.
func ParseEmbeddingProvider(name string) (EmbeddingProvider, error) {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" {
		return EmbeddingProviderNone, nil
	}
	for _, candidate := range []EmbeddingProvider{
		EmbeddingProviderOpenAI,
		EmbeddingProviderONNX,
	} {
		if strings.EqualFold(trimmed, string(candidate)) {
			return candidate, nil
		}
	}
	return "", fmt.Errorf(
		"embedding provider %q is not supported; use %q or %q",
		trimmed,
		EmbeddingProviderOpenAI,
		EmbeddingProviderONNX,
	)
}

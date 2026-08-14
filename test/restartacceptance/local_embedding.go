//go:build restartacceptance

package restartacceptance

import "goodkind.io/lm-semantic-search/test/sandboxharness"

const (
	cloneEmbeddingDimension = 16
	cloneEmbeddingModel     = "restart-acceptance-local"
)

type localEmbeddingBackend = sandboxharness.EmbeddingProvider

func startLocalEmbeddingBackend() (*localEmbeddingBackend, error) {
	return sandboxharness.StartEmbeddingProvider(sandboxharness.EmbeddingProviderOptions{
		Model:     cloneEmbeddingModel,
		Dimension: cloneEmbeddingDimension,
	})
}

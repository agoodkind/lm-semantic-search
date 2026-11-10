package embeddingprovider

import (
	"context"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
)

func TestNewRejectsNonOpenAI(t *testing.T) {
	t.Parallel()

	_, err := New(context.Background(), config.Config{
		EmbeddingProvider: "VoyageAI",
		OpenAIAPIKey:      "test-key",
		EmbeddingModel:    "voyage-code-3",
	})
	if err == nil {
		t.Fatal("New returned nil error for unsupported provider")
	}
}

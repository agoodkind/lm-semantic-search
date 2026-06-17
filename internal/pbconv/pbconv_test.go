package pbconv

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

func TestToProgressBackfillsLegacyChunkCounters(t *testing.T) {
	t.Parallel()

	progress := ToProgress(model.Progress{
		ChunksGenerated: 7,
		ChunksReused:    3,
	})

	if got := progress.GetChunksEmbedded(); got != 7 {
		t.Fatalf("ChunksEmbedded = %d, want legacy ChunksGenerated value 7", got)
	}
	if got := progress.GetChunksProcessed(); got != 10 {
		t.Fatalf("ChunksProcessed = %d, want legacy generated + reused total 10", got)
	}
	if got := progress.GetChunksGenerated(); got != 7 {
		t.Fatalf("ChunksGenerated = %d, want legacy value 7", got)
	}
	if got := progress.GetChunksReused(); got != 3 {
		t.Fatalf("ChunksReused = %d, want 3", got)
	}
}

func TestToProgressPreservesExplicitChunkCounters(t *testing.T) {
	t.Parallel()

	progress := ToProgress(model.Progress{
		ChunksProcessed: 5,
		ChunksEmbedded:  2,
		ChunksGenerated: 7,
		ChunksReused:    3,
	})

	if got := progress.GetChunksEmbedded(); got != 2 {
		t.Fatalf("ChunksEmbedded = %d, want explicit value 2", got)
	}
	if got := progress.GetChunksProcessed(); got != 5 {
		t.Fatalf("ChunksProcessed = %d, want explicit value 5", got)
	}
	if got := progress.GetChunksGenerated(); got != 7 {
		t.Fatalf("ChunksGenerated = %d, want legacy value 7", got)
	}
	if got := progress.GetChunksReused(); got != 3 {
		t.Fatalf("ChunksReused = %d, want 3", got)
	}
}

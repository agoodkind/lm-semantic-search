package daemon

import (
	"context"
	"slices"
	"sync"

	"goodkind.io/lm-semantic-search/internal/indexer"
)

// indexOneSpySource wraps a collectionItemSource and counts indexOne calls so a
// test can prove chunk regeneration runs only for the pruned work set. It embeds
// the real source by value, sharing its single-flight batch pointer, so the cheap
// classifier and the per-item loop behave exactly as in production. A pointer
// receiver on indexOne and a pointer itemSource keep the call log shared across
// the value copies the delta routine makes of deltaState.
type indexOneSpySource struct {
	collectionItemSource
	mu    sync.Mutex
	calls []string
}

func (source *indexOneSpySource) indexOne(ctx context.Context, itemID string) (indexer.OneFileResult, error) {
	source.mu.Lock()
	source.calls = append(source.calls, itemID)
	source.mu.Unlock()
	return source.collectionItemSource.indexOne(ctx, itemID)
}

func (source *indexOneSpySource) indexOneCalls() []string {
	source.mu.Lock()
	defer source.mu.Unlock()
	sorted := append([]string(nil), source.calls...)
	slices.Sort(sorted)
	return sorted
}

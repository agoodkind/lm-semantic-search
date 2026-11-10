package localvec

import (
	"context"
	"os"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
)

// The daemon treats a nil ProbeHealth as proof the store answered and clears a
// recorded store outage on it, so the probe has to touch the store. A local store
// whose root has gone away must report that failure instead of answering nil,
// which would clear an outage with nothing having been reached.
func TestProbeHealthReportsAnUnreadableStoreRoot(t *testing.T) {
	t.Parallel()

	store, err := newStoreWithProvider(
		config.Config{StateRoot: t.TempDir()},
		&fakeEmbeddingProvider{vectors: map[string][]float32{}},
	)
	if err != nil {
		t.Fatalf("newStoreWithProvider returned error: %v", err)
	}
	if probeErr := store.ProbeHealth(context.Background()); probeErr != nil {
		t.Fatalf("ProbeHealth on a reachable store returned error: %v", probeErr)
	}

	if removeErr := os.RemoveAll(store.root); removeErr != nil {
		t.Fatalf("RemoveAll returned error: %v", removeErr)
	}
	if probeErr := store.ProbeHealth(context.Background()); probeErr == nil {
		t.Fatal("ProbeHealth answered nil for a store whose root is gone, so it supplies no evidence that anything was reached")
	}
}

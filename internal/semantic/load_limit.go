package semantic

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	"goodkind.io/lm-semantic-search/internal/clock"
)

// defaultMaxConcurrentCollectionLoads is the daemon-wide load cap a Service
// applies when its config contains no usable count. Config resolution returns
// the same value as its own fallback. Tests build a Service without config
// resolution, and that Service caps loads at this value rather than running
// unbounded.
const defaultMaxConcurrentCollectionLoads = 2

// collectionLoadLimiter caps how many distinct collections are in the load
// transition at once. The per-collection coordinator and the residency
// controller each collapse concurrent loads of one name into one flight, but
// nothing above this counts flights across names, and after a Milvus restore
// the daemon started well over a hundred of them within minutes and Milvus ran
// out of memory. Every load transition takes a slot here first, so Milvus
// never sees more than the cap in flight regardless of which path started the
// load.
type collectionLoadLimiter struct {
	limit int
	slots chan struct{}
}

func newCollectionLoadLimiter(limit int) *collectionLoadLimiter {
	if limit < 1 {
		limit = defaultMaxConcurrentCollectionLoads
	}
	return &collectionLoadLimiter{limit: limit, slots: make(chan struct{}, limit)}
}

// acquire takes one load slot, waits while every slot is held, and returns the
// release that frees it. A load queued behind the cap for longer than its own
// ceiling fails as not-ready rather than waiting forever. The wait ends when
// ctx ends. A residency load passes the detached load context, which the
// ceiling cancels. A coordinator load passes the ceiling-bounded flight
// context. The caller that asked for the load waits on the flight under its own
// context and deadline, never on this slot.
func (limiter *collectionLoadLimiter) acquire(
	ctx context.Context,
	collectionName string,
) (func(), error) {
	select {
	case limiter.slots <- struct{}{}:
		return limiter.releaseFunc(), nil
	default:
	}

	startedAt := clock.Now()
	slog.InfoContext(ctx, "semantic.collection_load_slot_wait",
		"component", "semantic",
		"subcomponent", "load",
		"collection", collectionName,
		"limit", limiter.limit,
	)
	select {
	case limiter.slots <- struct{}{}:
		slog.InfoContext(ctx, "semantic.collection_load_slot_acquired",
			"component", "semantic",
			"subcomponent", "load",
			"collection", collectionName,
			"limit", limiter.limit,
			"waited_ms", clock.Now().Sub(startedAt).Milliseconds(),
		)
		return limiter.releaseFunc(), nil
	case <-ctx.Done():
		err := fmt.Errorf("wait for a collection load slot for %s: %w", collectionName, ctx.Err())
		slog.WarnContext(ctx, "semantic.collection_load_slot_wait_cancelled",
			"component", "semantic",
			"subcomponent", "load",
			"collection", collectionName,
			"limit", limiter.limit,
			"waited_ms", clock.Now().Sub(startedAt).Milliseconds(),
			"err", err,
		)
		return nil, err
	}
}

// releaseFunc frees one slot exactly once, so a double release from a deferred
// call and an explicit call cannot hand out a slot that was never taken.
func (limiter *collectionLoadLimiter) releaseFunc() func() {
	var once sync.Once
	return func() {
		once.Do(func() {
			<-limiter.slots
		})
	}
}

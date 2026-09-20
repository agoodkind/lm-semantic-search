package semantic

import (
	"context"
	"errors"
	"testing"
	"time"
)

// A load parked on a full limiter ends with its own context error when that
// context ends, and the slot it never took stays free for the next load. The
// cap's effect on a real store is proved by test/milvusintegration.
func TestCollectionLoadLimiterHonorsContextWhileWaiting(t *testing.T) {
	t.Parallel()

	limiter := newCollectionLoadLimiter(1)
	release, err := limiter.acquire(context.Background(), "hybrid_code_chunks_first")
	if err != nil {
		t.Fatalf("first acquire returned error: %v", err)
	}

	waitCtx, cancelWait := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancelWait()
	if _, err := limiter.acquire(waitCtx, "hybrid_code_chunks_second"); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("second acquire error = %v, want the waiter's own deadline", err)
	}

	release()
	release()
	thirdRelease, err := limiter.acquire(context.Background(), "hybrid_code_chunks_third")
	if err != nil {
		t.Fatalf("acquire after release returned error: %v", err)
	}
	thirdRelease()
	if got := len(limiter.slots); got != 0 {
		t.Fatalf("slots held after every release = %d, want 0", got)
	}
}

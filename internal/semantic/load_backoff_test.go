package semantic

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

// backoffTestClock is a settable clock for the backoff, so a test moves time
// forward instead of sleeping through a pause.
type backoffTestClock struct {
	mutex sync.Mutex
	now   time.Time
}

func newBackoffTestClock() *backoffTestClock {
	return &backoffTestClock{mutex: sync.Mutex{}, now: time.Unix(1_000, 0).UTC()}
}

func (clock *backoffTestClock) Now() time.Time {
	clock.mutex.Lock()
	defer clock.mutex.Unlock()
	return clock.now
}

func (clock *backoffTestClock) Advance(delta time.Duration) {
	clock.mutex.Lock()
	defer clock.mutex.Unlock()
	clock.now = clock.now.Add(delta)
}

// The pause grows on every memory signal, is bounded, blocks admission with a
// not-ready error the callers already retry on, and resets once a load
// succeeds. Its effect against a real Milvus is proved by
// test/milvusintegration.
func TestCollectionLoadBackoffGrowsAndClears(t *testing.T) {
	t.Parallel()

	clock := newBackoffTestClock()
	backoff := newCollectionLoadBackoff()
	backoff.now = clock.Now
	memoryErr := errors.New("load Milvus collection x: service resource insufficient[resourceType=Memory]")

	if err := backoff.admit(context.Background(), "x"); err != nil {
		t.Fatalf("admit before any signal returned error: %v", err)
	}

	wantIntervals := []time.Duration{30 * time.Second, time.Minute, 2 * time.Minute, 4 * time.Minute, 5 * time.Minute, 5 * time.Minute}
	for i, want := range wantIntervals {
		backoff.noteLoadOutcome(context.Background(), "x", memoryErr)
		err := backoff.admit(context.Background(), "y")
		if !errors.Is(err, ErrCollectionLoadDeferred) || !errors.Is(err, ErrCollectionNotReady) {
			t.Fatalf("signal %d: admit error = %v, want the deferred not-ready sentinel", i, err)
		}
		clock.Advance(want - time.Second)
		if err := backoff.admit(context.Background(), "y"); err == nil {
			t.Fatalf("signal %d: admitted a load %s before the %s pause elapsed", i, want-time.Second, want)
		}
		clock.Advance(time.Second)
		if err := backoff.admit(context.Background(), "y"); err != nil {
			t.Fatalf("signal %d: admit after the %s pause elapsed returned error: %v", i, want, err)
		}
	}

	backoff.noteLoadOutcome(context.Background(), "x", nil)
	backoff.noteLoadOutcome(context.Background(), "x", memoryErr)
	clock.Advance(30 * time.Second)
	if err := backoff.admit(context.Background(), "y"); err != nil {
		t.Fatalf("pause after a successful load did not reset to the initial interval: %v", err)
	}
}

// Only a memory signal pauses loads. A cancelled load, a transport outage, a
// caller's wait timeout, and an ordinary store error all leave the backoff off,
// because pausing loads on them would stall a healthy store.
func TestCollectionLoadBackoffIgnoresOtherFailures(t *testing.T) {
	t.Parallel()

	backoff := newCollectionLoadBackoff()
	others := []error{
		context.Canceled,
		context.DeadlineExceeded,
		errors.New("load Milvus collection x: rpc error: code = Unavailable desc = connection refused"),
		errors.New("wait for collection x: " + ErrCollectionLoadWaitTimeout.Error()),
		errors.New("collection x did not become queryable within 90s (load progress 40%): collection_not_ready: semantic collection is not ready"),
		ErrCollectionNotReady,
	}
	for _, other := range others {
		backoff.noteLoadOutcome(context.Background(), "x", other)
		if err := backoff.admit(context.Background(), "y"); err != nil {
			t.Fatalf("%v started a pause: %v", other, err)
		}
	}

	unrecovered := errors.New("collection x: " + errCollectionLoadUnrecovered.Error())
	if collectionLoadMemorySignal(unrecovered) {
		t.Fatal("a message that merely mentions the unrecovered text is not the unrecovered outcome")
	}
}

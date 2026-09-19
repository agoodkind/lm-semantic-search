package semantic

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
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

// captureBackoffLogs routes the default logger into a buffer for the test and
// restores it afterwards, so the once-per-transition logging is observable.
func captureBackoffLogs(t *testing.T) *bytes.Buffer {
	t.Helper()
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, nil)))
	t.Cleanup(func() {
		slog.SetDefault(previous)
	})
	return &logs
}

// memoryExhaustedLoadStatus is the status a Milvus proxy answers a load with
// when the query nodes cannot fit the collection: merr.ErrServiceMemoryLimitExceeded
// (code 3) with the predicted and allowed sizes appended as merr fields.
func memoryExhaustedLoadStatus() *commonpb.Status {
	return &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_InsufficientMemoryToLoad,
		Reason:    "memory limit exceeded[predict(MB)=1024, limit(MB)=512]",
		Code:      3,
	}
}

// The pause grows on every memory signal, is bounded, blocks admission with a
// not-ready error the callers already retry on, and resets once a load
// succeeds.
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

// A load the proxy refuses for memory pauses every other collection's load
// for the initial interval: the next acquire fails fast with the deferred
// not-ready error and reaches Milvus with no LoadCollection request at all.
// Once the pause elapses loads resume, and the transitions into and out of the
// pause are each logged exactly once.
func TestMemoryExhaustedLoadPausesOtherCollectionLoads(t *testing.T) {
	logs := captureBackoffLogs(t)
	server := resetPromotionRecoveryServer()
	service := newLoadPathTestService(t, server)
	clock := newBackoffTestClock()
	service.loadBackoff().now = clock.Now

	const (
		refusedName = "hybrid_code_chunks_backoff_refused"
		queuedName  = "hybrid_code_chunks_backoff_queued"
	)
	server.setCollections(refusedName, queuedName)
	server.setLoadStates(commonpb.LoadState_LoadStateLoaded, refusedName, queuedName)
	server.setLoadFailure(memoryExhaustedLoadStatus())

	lease, err := service.AcquireCollection(context.Background(), refusedName)
	if lease != nil {
		lease.Release()
	}
	if err == nil || !strings.Contains(err.Error(), "memory limit exceeded") {
		t.Fatalf("refused AcquireCollection error = %v, want the Milvus memory error", err)
	}
	if calls := server.loadCallCount(); calls != 1 {
		t.Fatalf("LoadCollection calls after the refusal = %d, want 1", calls)
	}

	server.setLoadFailure(nil)
	lease, err = service.AcquireCollection(context.Background(), queuedName)
	if lease != nil {
		lease.Release()
	}
	if !errors.Is(err, ErrCollectionLoadDeferred) {
		t.Fatalf("AcquireCollection during the pause error = %v, want ErrCollectionLoadDeferred", err)
	}
	if !errors.Is(err, ErrCollectionNotReady) {
		t.Fatalf("deferred load error %v does not read as collection-not-ready", err)
	}
	if calls := server.loadCallCount(); calls != 1 {
		t.Fatalf("LoadCollection calls during the pause = %d, want 1: no new load may reach Milvus", calls)
	}

	clock.Advance(collectionLoadBackoffInitial)
	lease, err = service.AcquireCollection(context.Background(), queuedName)
	if err != nil {
		t.Fatalf("AcquireCollection after the pause returned error: %v", err)
	}
	lease.Release()
	if calls := server.loadCallCount(); calls != 2 {
		t.Fatalf("LoadCollection calls after the pause = %d, want 2", calls)
	}

	logged := logs.String()
	if got := strings.Count(logged, "semantic.collection_load_backoff_started"); got != 1 {
		t.Fatalf("backoff start logged %d times, want 1:\n%s", got, logged)
	}
	if got := strings.Count(logged, "semantic.collection_load_backoff_ended"); got != 1 {
		t.Fatalf("backoff end logged %d times, want 1:\n%s", got, logged)
	}
}

// Milvus reports memory exhaustion to this daemon mostly as a load that never
// finishes: the query node retries the failed segment loads itself and the
// client sees only a collection that stays loading past every bound. A load
// that exhausts both polls and the re-issued request therefore pauses other
// loads the same way a direct memory error does.
func TestUnrecoveredLoadPausesOtherCollectionLoads(t *testing.T) {
	server := resetPromotionRecoveryServer()
	service := newLoadPathTestService(t, server)
	service.cfg.MilvusCollectionLoadTimeoutMS = 40

	const (
		stuckName  = "hybrid_code_chunks_backoff_stuck"
		queuedName = "hybrid_code_chunks_backoff_after_stuck"
	)
	server.setCollections(stuckName, queuedName)
	server.setLoadStates(commonpb.LoadState_LoadStateLoading, stuckName)
	server.setLoadStates(commonpb.LoadState_LoadStateLoaded, queuedName)

	lease, err := service.AcquireCollection(context.Background(), stuckName)
	if lease != nil {
		lease.Release()
	}
	if !errors.Is(err, errCollectionLoadUnrecovered) || !errors.Is(err, ErrCollectionNotReady) {
		t.Fatalf("stuck AcquireCollection error = %v, want the unrecovered not-ready outcome", err)
	}
	if calls := server.loadCallCount(); calls != 2 {
		t.Fatalf("LoadCollection calls for the stuck load = %d, want the initial request plus one re-issue", calls)
	}

	lease, err = service.AcquireCollection(context.Background(), queuedName)
	if lease != nil {
		lease.Release()
	}
	if !errors.Is(err, ErrCollectionLoadDeferred) {
		t.Fatalf("AcquireCollection after the unrecovered load error = %v, want ErrCollectionLoadDeferred", err)
	}
	if calls := server.loadCallCount(); calls != 2 {
		t.Fatalf("LoadCollection calls during the pause = %d, want 2: no new load may reach Milvus", calls)
	}
}

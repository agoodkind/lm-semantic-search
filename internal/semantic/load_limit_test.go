package semantic

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"goodkind.io/lm-semantic-search/internal/config"
)

const loadCapSettle = 100 * time.Millisecond

// newLoadPathTestService builds a service against the fake Milvus with the
// asynchronous residency reconciliation stopped. The reconciliation can list
// the fake's collections after a test registers them and mark a loaded one
// ready without a load, which would let an acquire skip the load path these
// tests exist to exercise.
func newLoadPathTestService(t *testing.T, server *promotionRecoveryServer) *Service {
	t.Helper()
	service := newPromotionTestService(t, server)
	if err := service.stopResidencyReconciliation(context.Background()); err != nil {
		t.Fatalf("stop residency reconciliation: %v", err)
	}
	return service
}

// After a Milvus restore the daemon asked for over a hundred collection loads
// within minutes and Milvus ran out of memory. Loads of different collections
// must therefore share a daemon-wide cap: with five cold collections acquired at
// once under a cap of two, Milvus sees at most two LoadCollection requests in
// flight, and every caller still ends up with a lease once the loads drain.
func TestConcurrentLoadsOfDifferentCollectionsAreCapped(t *testing.T) {
	const (
		loadCap         = 2
		collectionCount = 5
	)
	server := resetPromotionRecoveryServer()
	service := newLoadPathTestService(t, server)
	service.cfg.MilvusMaxConcurrentCollectionLoads = loadCap

	names := make([]string, 0, collectionCount)
	for i := range collectionCount {
		names = append(names, fmt.Sprintf("hybrid_code_chunks_cap_%d", i))
	}
	server.setCollections(names...)
	server.setLoadStates(commonpb.LoadState_LoadStateLoaded, names...)
	arrived, resume := server.holdLoadCollections(collectionCount)
	defer resume()

	results := make(chan error, collectionCount)
	for _, name := range names {
		go func() {
			lease, err := service.AcquireCollection(context.Background(), name)
			if lease != nil {
				lease.Release()
			}
			results <- err
		}()
	}

	for range loadCap {
		select {
		case <-arrived:
		case <-time.After(5 * time.Second):
			t.Fatal("fewer loads reached Milvus than the cap allows")
		}
	}
	waitForLoadingCount(t, service.residency, collectionCount)
	// Every flight has started and the cap is full; a third request arriving
	// now is exactly the failure this cap prevents, so give one a chance to.
	time.Sleep(loadCapSettle)
	inFlight, peak := server.loadsInFlightNow()
	if inFlight != loadCap || peak != loadCap {
		t.Fatalf("loads in flight = %d (peak %d), want %d while the cap is full", inFlight, peak, loadCap)
	}

	resume()
	for range collectionCount {
		select {
		case err := <-results:
			if err != nil {
				t.Fatalf("AcquireCollection returned error after the loads drained: %v", err)
			}
		case <-time.After(10 * time.Second):
			t.Fatal("an acquire never completed after the loads were released")
		}
	}
	if calls := server.loadCallCount(); calls != collectionCount {
		t.Fatalf("LoadCollection calls = %d, want one per collection (%d)", calls, collectionCount)
	}
	if _, peak := server.loadsInFlightNow(); peak != loadCap {
		t.Fatalf("peak loads in flight = %d, want the cap %d", peak, loadCap)
	}
}

// A caller queued behind the cap still answers to its own deadline: its lease
// wait ends with its context error, and the load it would have started is never
// issued to Milvus while the slot is held.
func TestCallerWaitingBehindLoadCapHonorsItsOwnDeadline(t *testing.T) {
	server := resetPromotionRecoveryServer()
	service := newLoadPathTestService(t, server)
	service.cfg.MilvusMaxConcurrentCollectionLoads = 1

	const (
		heldName   = "hybrid_code_chunks_cap_held"
		queuedName = "hybrid_code_chunks_cap_queued"
	)
	server.setCollections(heldName, queuedName)
	server.setLoadStates(commonpb.LoadState_LoadStateLoaded, heldName, queuedName)
	arrived, resume := server.holdLoadCollections(2)
	defer resume()

	heldResult := make(chan error, 1)
	go func() {
		lease, err := service.AcquireCollection(context.Background(), heldName)
		if lease != nil {
			lease.Release()
		}
		heldResult <- err
	}()
	select {
	case name := <-arrived:
		if name != heldName {
			t.Fatalf("first load reached Milvus for %q, want %q", name, heldName)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the first load never reached Milvus")
	}

	queuedCtx, cancelQueued := context.WithTimeout(context.Background(), loadCapSettle)
	defer cancelQueued()
	lease, err := service.AcquireCollection(queuedCtx, queuedName)
	if lease != nil {
		lease.Release()
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("queued AcquireCollection error = %v, want the caller's own deadline", err)
	}
	if calls := server.loadCallCount(); calls != 1 {
		t.Fatalf("LoadCollection calls = %d, want 1: the queued load must not reach Milvus while the slot is held", calls)
	}

	resume()
	select {
	case err := <-heldResult:
		if err != nil {
			t.Fatalf("held AcquireCollection returned error after resume: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the held acquire never completed after resume")
	}
}

// A load parked on a full limiter ends with its own context error when that
// context ends, and the slot it never took stays free for the next load.
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

// A Service assembled without config resolution still caps loads at the
// built-in default, and a configured count replaces it.
func TestCollectionLoadSlotsReadConfig(t *testing.T) {
	t.Parallel()

	configured := &Service{cfg: config.Config{MilvusMaxConcurrentCollectionLoads: 3}}
	if got := configured.collectionLoadSlots().limit; got != 3 {
		t.Fatalf("configured load cap = %d, want 3", got)
	}

	unset := &Service{}
	if got := unset.collectionLoadSlots().limit; got != defaultMaxConcurrentCollectionLoads {
		t.Fatalf("unset load cap = %d, want the built-in %d", got, defaultMaxConcurrentCollectionLoads)
	}
}

// waitForLoadingCount waits until want residency entries have a load flight
// started, so a test knows every acquire has reached the load transition
// before it inspects how many of those loads Milvus actually received.
func waitForLoadingCount(t *testing.T, controller *collectionResidencyController, want int) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		loading := 0
		for _, collection := range controller.ResidencySnapshot().Collections {
			if collection.Loading {
				loading++
			}
		}
		if loading == want {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("loading residency entries did not reach %d", want)
}

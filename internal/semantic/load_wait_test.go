package semantic

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/config"
)

// stuckProbe reports a collection that is loading and never finishes, which is
// the condition the bound exists for.
func stuckProbe(context.Context) (collectionLoadState, error) {
	return collectionLoadState{Loaded: false, Progress: 40, ProgressKnown: true}, nil
}

// A collection that never finishes loading must fail within the bound rather
// than block forever, and it must report the collection, the wait, and the last
// progress Milvus exposed. Without the bound this call never returns, so the
// deadline below is the assertion that the wait is bounded at all.
func TestWaitForCollectionLoadBoundsAStuckLoad(t *testing.T) {
	t.Parallel()

	var reissues atomic.Int32
	reissue := func(context.Context) error {
		reissues.Add(1)
		return nil
	}

	done := make(chan error, 1)
	go func() {
		done <- waitForCollectionLoad(context.Background(), "hybrid_code_chunks_stuck", 20*time.Millisecond, stuckProbe, reissue)
	}()

	var err error
	select {
	case err = <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("waitForCollectionLoad did not return; the stuck load is unbounded")
	}

	if err == nil {
		t.Fatal("waitForCollectionLoad returned nil for a collection that never loaded")
	}
	if !errors.Is(err, ErrCollectionNotReady) {
		t.Fatalf("waitForCollectionLoad error = %v, want it to classify as ErrCollectionNotReady", err)
	}
	if got := reissues.Load(); got != 1 {
		t.Fatalf("re-issued loads = %d, want exactly 1 bounded recovery attempt", got)
	}
	if !strings.Contains(err.Error(), "hybrid_code_chunks_stuck") {
		t.Fatalf("error %q does not name the stuck collection", err)
	}
	if !strings.Contains(err.Error(), "40%") {
		t.Fatalf("error %q does not report the load progress Milvus exposed", err)
	}
}

// The single recovery attempt is what makes a lost or released load self-heal:
// when the re-issued load takes effect, the wait succeeds instead of failing.
func TestWaitForCollectionLoadRecoversAfterOneReissue(t *testing.T) {
	t.Parallel()

	var reissued atomic.Bool
	probe := func(context.Context) (collectionLoadState, error) {
		if reissued.Load() {
			return collectionLoadState{Loaded: true, Progress: 0, ProgressKnown: false}, nil
		}
		return collectionLoadState{Loaded: false, Progress: 0, ProgressKnown: false}, nil
	}
	reissue := func(context.Context) error {
		reissued.Store(true)
		return nil
	}

	if err := waitForCollectionLoad(context.Background(), "conv_chunks_recovered", 20*time.Millisecond, probe, reissue); err != nil {
		t.Fatalf("waitForCollectionLoad returned error after a successful re-issued load: %v", err)
	}
}

// A collection that is already loaded returns immediately and never re-issues
// the load, so the steady-state path costs one probe.
func TestWaitForCollectionLoadReturnsImmediatelyWhenLoaded(t *testing.T) {
	t.Parallel()

	var probes atomic.Int32
	probe := func(context.Context) (collectionLoadState, error) {
		probes.Add(1)
		return collectionLoadState{Loaded: true, Progress: 0, ProgressKnown: false}, nil
	}
	reissue := func(context.Context) error {
		t.Error("re-issued the load for an already-loaded collection")
		return nil
	}

	if err := waitForCollectionLoad(context.Background(), "hybrid_code_chunks_ready", time.Minute, probe, reissue); err != nil {
		t.Fatalf("waitForCollectionLoad returned error for a loaded collection: %v", err)
	}
	if got := probes.Load(); got != 1 {
		t.Fatalf("load-state probes = %d, want 1", got)
	}
}

// A probe that fails for a real reason (not the bound expiring) surfaces that
// classified store error instead of being reported as a stuck load, so a store
// outage keeps its own classification.
func TestWaitForCollectionLoadReturnsProbeFailure(t *testing.T) {
	t.Parallel()

	probeErr := errors.New("milvus refused the load-state read")
	probe := func(context.Context) (collectionLoadState, error) {
		return collectionLoadState{Loaded: false, Progress: 0, ProgressKnown: false}, probeErr
	}
	reissue := func(context.Context) error {
		t.Error("re-issued the load after a probe failure")
		return nil
	}

	err := waitForCollectionLoad(context.Background(), "hybrid_code_chunks_broken", time.Minute, probe, reissue)
	if !errors.Is(err, probeErr) {
		t.Fatalf("waitForCollectionLoad error = %v, want the probe failure", err)
	}
}

// A cancelled caller ends the wait with its own cancellation rather than a
// stuck-load verdict, so a shutdown never reports a false store fault.
func TestWaitForCollectionLoadHonorsCallerCancellation(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	reissue := func(context.Context) error {
		t.Error("re-issued the load after the caller cancelled")
		return nil
	}
	err := waitForCollectionLoad(ctx, "hybrid_code_chunks_cancelled", time.Minute, stuckProbe, reissue)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("waitForCollectionLoad error = %v, want context.Canceled", err)
	}
}

func TestCollectionLoadPollRecognizesElapsedBoundBeforeTimerCancellation(t *testing.T) {
	if !collectionLoadPollBoundExpired(
		context.Background(),
		context.Background(),
		0,
	) {
		t.Fatal("elapsed poll bound was not recognized before context cancellation became visible")
	}

	parent, cancelParent := context.WithCancel(context.Background())
	cancelParent()
	if collectionLoadPollBoundExpired(
		parent,
		context.Background(),
		0,
	) {
		t.Fatal("caller cancellation was mistaken for poll-bound exhaustion")
	}
}

// Concurrent callers for one collection share the whole load operation. The
// leader issues the initial load and the bounded recovery, while followers
// observe the same outcome without multiplying either request.
func TestCollectionLoadCoordinatorCollapsesConcurrentWaiters(t *testing.T) {
	const waiterCount = 8

	var coordinator collectionLoadCoordinator
	var operationCalls atomic.Int32
	var initialLoads atomic.Int32
	var recoveryLoads atomic.Int32
	start := make(chan struct{})
	release := make(chan struct{})
	results := make(chan error, waiterCount)
	var ready sync.WaitGroup
	ready.Add(waiterCount)

	operation := func(ctx context.Context) error {
		operationCalls.Add(1)
		initialLoads.Add(1)
		<-release
		reissue := func(context.Context) error {
			recoveryLoads.Add(1)
			return nil
		}
		return waitForCollectionLoad(
			ctx,
			"hybrid_code_chunks_shared",
			20*time.Millisecond,
			stuckProbe,
			reissue,
		)
	}
	for range waiterCount {
		go func() {
			ready.Done()
			<-start
			results <- coordinator.Do(
				context.Background(),
				"hybrid_code_chunks_shared",
				time.Minute,
				operation,
			)
		}()
	}
	ready.Wait()
	close(start)

	deadline := time.NewTimer(time.Second)
	defer deadline.Stop()
	for operationCalls.Load() == 0 {
		select {
		case <-deadline.C:
			t.Fatal("no collection load operation started")
		default:
			time.Sleep(time.Millisecond)
		}
	}
	time.Sleep(100 * time.Millisecond)
	close(release)

	for range waiterCount {
		err := <-results
		if !errors.Is(err, ErrCollectionNotReady) {
			t.Fatalf("shared load error = %v, want ErrCollectionNotReady", err)
		}
	}
	if got := operationCalls.Load(); got != 1 {
		t.Fatalf("collection load operations = %d, want 1 shared flight", got)
	}
	if got := initialLoads.Load(); got != 1 {
		t.Fatalf("initial collection loads = %d, want 1 shared request", got)
	}
	if got := recoveryLoads.Load(); got != 1 {
		t.Fatalf("recovery collection loads = %d, want 1 shared request", got)
	}
}

// Callers of one shared load do not share a fate. The caller that started the
// load cancels while the load is still running, and a caller whose own context
// is healthy still receives the load's real outcome rather than the other
// caller's cancellation.
func TestCollectionLoadOutlivesTheCallerThatStartedIt(t *testing.T) {
	const collectionName = "hybrid_code_chunks_detached"

	var coordinator collectionLoadCoordinator
	var operationCalls atomic.Int32
	loadOutcome := errors.New("the shared load reached its own verdict")
	operationRunning := make(chan struct{})
	release := make(chan struct{})
	operation := func(loadCtx context.Context) error {
		if operationCalls.Add(1) == 1 {
			close(operationRunning)
		}
		<-release
		if err := loadCtx.Err(); err != nil {
			return err
		}
		return loadOutcome
	}

	startingCtx, cancelStarting := context.WithCancel(context.Background())
	defer cancelStarting()
	startingResult := make(chan error, 1)
	go func() {
		startingResult <- coordinator.Do(startingCtx, collectionName, time.Minute, operation)
	}()
	<-operationRunning

	joinedResult := make(chan error, 1)
	joined := make(chan struct{})
	go func() {
		close(joined)
		joinedResult <- coordinator.Do(context.Background(), collectionName, time.Minute, operation)
	}()
	<-joined

	cancelStarting()
	startingErr := <-startingResult
	if !errors.Is(startingErr, context.Canceled) {
		t.Fatalf("starting caller error = %v, want its own cancellation", startingErr)
	}

	close(release)
	joinedErr := <-joinedResult
	if !errors.Is(joinedErr, loadOutcome) {
		t.Fatalf("joined caller error = %v, want the load's own outcome %v", joinedErr, loadOutcome)
	}
	if got := operationCalls.Load(); got != 1 {
		t.Fatalf("collection load operations = %d, want 1 shared load across both callers", got)
	}
}

// The bound is operator-configurable and falls back to the built-in value when
// the config carries nothing usable.
func TestCollectionLoadBoundReadsConfig(t *testing.T) {
	t.Parallel()

	configured := &Service{cfg: config.Config{MilvusCollectionLoadTimeoutMS: 2500}}
	if got := configured.collectionLoadBound(); got != 2500*time.Millisecond {
		t.Fatalf("collectionLoadBound = %s, want 2.5s from config", got)
	}

	unset := &Service{cfg: config.Config{MilvusCollectionLoadTimeoutMS: 0}}
	if got := unset.collectionLoadBound(); got != defaultCollectionLoadBound {
		t.Fatalf("collectionLoadBound = %s, want the built-in %s", got, defaultCollectionLoadBound)
	}

	rejected := &Service{cfg: config.Config{MilvusCollectionLoadTimeoutMS: -1}}
	if got := rejected.collectionLoadBound(); got != defaultCollectionLoadBound {
		t.Fatalf("collectionLoadBound = %s for a rejected count, want the built-in %s", got, defaultCollectionLoadBound)
	}
}

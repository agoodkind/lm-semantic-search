package semantic

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/config"
)

type testResidencyClock struct {
	mutex  sync.Mutex
	now    time.Time
	timers []*testResidencyTimer
}

type testResidencyTimer struct {
	clock    *testResidencyClock
	deadline time.Time
	callback func()
	stopped  bool
	fired    bool
}

func newTestResidencyClock() *testResidencyClock {
	return &testResidencyClock{now: time.Unix(1_000, 0)}
}

func TestResidencyControllerDefaultsDetachedLoadCeiling(t *testing.T) {
	controller := newCollectionResidencyController(residencyControllerConfig{})

	want := (&Service{}).sharedCollectionLoadCeiling()
	if controller.config.loadCeiling != want {
		t.Fatalf("load ceiling = %s, want %s", controller.config.loadCeiling, want)
	}
}

func (clock *testResidencyClock) Now() time.Time {
	clock.mutex.Lock()
	defer clock.mutex.Unlock()
	return clock.now
}

func (clock *testResidencyClock) AfterFunc(
	delay time.Duration,
	callback func(),
) residencyTimer {
	clock.mutex.Lock()
	defer clock.mutex.Unlock()
	timer := &testResidencyTimer{
		clock:    clock,
		deadline: clock.now.Add(delay),
		callback: callback,
	}
	clock.timers = append(clock.timers, timer)
	return timer
}

func (clock *testResidencyClock) Advance(duration time.Duration) {
	clock.mutex.Lock()
	clock.now = clock.now.Add(duration)
	callbacks := make([]func(), 0)
	for _, timer := range clock.timers {
		if timer.stopped || timer.fired || timer.deadline.After(clock.now) {
			continue
		}
		timer.fired = true
		callbacks = append(callbacks, timer.callback)
	}
	clock.mutex.Unlock()

	for _, callback := range callbacks {
		callback()
	}
}

func (clock *testResidencyClock) LastTimer() *testResidencyTimer {
	clock.mutex.Lock()
	defer clock.mutex.Unlock()
	return clock.timers[len(clock.timers)-1]
}

func (timer *testResidencyTimer) FireStale() {
	timer.callback()
}

func (timer *testResidencyTimer) Stop() bool {
	timer.clock.mutex.Lock()
	defer timer.clock.mutex.Unlock()
	if timer.stopped || timer.fired {
		return false
	}
	timer.stopped = true
	return true
}

func waitForMaintenance(
	t *testing.T,
	controller *collectionResidencyController,
	collectionName string,
) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		controller.mutex.Lock()
		entry := controller.entries[collectionName]
		active := entry != nil && entry.maintenance
		controller.mutex.Unlock()
		if active {
			return
		}
		runtime.Gosched()
	}
	t.Fatal("maintenance did not reserve the collection")
}

func waitForLeaseCount(
	t *testing.T,
	controller *collectionResidencyController,
	collectionName string,
	want int,
) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		controller.mutex.Lock()
		entry := controller.entries[collectionName]
		matches := entry != nil && entry.leases == want
		controller.mutex.Unlock()
		if matches {
			return
		}
		runtime.Gosched()
	}
	t.Fatalf("lease count did not reach %d", want)
}

func waitForResidencyState(
	t *testing.T,
	controller *collectionResidencyController,
	collectionName string,
	want collectionResidencyState,
) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		controller.mutex.Lock()
		entry := controller.entries[collectionName]
		matches := entry != nil && entry.state == want
		controller.mutex.Unlock()
		if matches {
			return
		}
		runtime.Gosched()
	}
	t.Fatalf("residency state did not reach %v", want)
}

func TestResidencyAcquireSharesLoadWithIndependentCallerLimits(t *testing.T) {
	clock := newTestResidencyClock()
	loadStarted := make(chan struct{})
	finishLoad := make(chan struct{})
	var loadCalls atomic.Int32
	load := func(ctx context.Context, _ string) error {
		if loadCalls.Add(1) == 1 {
			close(loadStarted)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-finishLoad:
			return nil
		}
	}
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: 15 * time.Second,
		loadCeiling: time.Minute,
		load:        load,
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	firstResult := make(chan error, 1)
	go func() {
		_, err := controller.Acquire(context.Background(), "collection")
		firstResult <- err
	}()
	<-loadStarted

	clock.Advance(5 * time.Second)
	secondResult := make(chan error, 1)
	go func() {
		lease, err := controller.Acquire(context.Background(), "collection")
		if lease != nil {
			lease.Release()
		}
		secondResult <- err
	}()

	clock.Advance(10 * time.Second)
	if err := <-firstResult; !errors.Is(err, ErrCollectionLoadWaitTimeout) {
		t.Fatalf("first Acquire error = %v, want ErrCollectionLoadWaitTimeout", err)
	}
	select {
	case err := <-secondResult:
		t.Fatalf("second Acquire returned before its independent limit: %v", err)
	default:
	}
	if got := loadCalls.Load(); got != 1 {
		t.Fatalf("load calls = %d, want one shared background load", got)
	}
	retryResult := make(chan error, 1)
	go func() {
		lease, err := controller.Acquire(context.Background(), "collection")
		if lease != nil {
			lease.Release()
		}
		retryResult <- err
	}()
	waitForLeaseCount(t, controller, "collection", 2)
	if got := loadCalls.Load(); got != 1 {
		t.Fatalf("load calls after retry = %d, want retry to join the active load", got)
	}

	close(finishLoad)
	if err := <-secondResult; err != nil {
		t.Fatalf("second Acquire returned error after shared load completed: %v", err)
	}
	if err := <-retryResult; err != nil {
		t.Fatalf("retry Acquire returned error after shared load completed: %v", err)
	}
}

func TestServiceResidencyUsesConfiguredIdleDelay(t *testing.T) {
	service := &Service{cfg: config.Config{
		MilvusCollectionLoadWaitTimeoutMS: 23000,
		MilvusCollectionIdleTimeoutMS:     42000,
	}}
	service.initializeResidencyControllerWithLoad(func(context.Context, string) error {
		return nil
	})
	t.Cleanup(func() {
		_ = service.Close(context.Background())
	})
	if got := service.residency.config.waitTimeout; got != 23*time.Second {
		t.Fatalf("wait timeout = %s, want 23s", got)
	}
	if got := service.residency.config.idleTimeout; got != 42*time.Second {
		t.Fatalf("idle timeout = %s, want 42s", got)
	}
}

func TestResidencyAcquireHonorsEarlierCancellationWithoutStartingLoad(t *testing.T) {
	var loadCalls atomic.Int32
	loadStarted := make(chan struct{}, 1)
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       newTestResidencyClock(),
		waitTimeout: 15 * time.Second,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			loadCalls.Add(1)
			loadStarted <- struct{}{}
			return nil
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	lease, err := controller.Acquire(ctx, "collection")
	if lease != nil {
		lease.Release()
		t.Fatal("Acquire returned a lease for a canceled caller")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Acquire error = %v, want context.Canceled", err)
	}
	select {
	case <-loadStarted:
		t.Fatal("an already-canceled caller started a detached load")
	case <-time.After(100 * time.Millisecond):
	}
	if got := loadCalls.Load(); got != 0 {
		t.Fatalf("load calls = %d, want 0 for an already-canceled caller", got)
	}
}

func TestResidencyFinalReleaseStartsFreshIdleDeadlineAndRejectsStaleTimer(
	t *testing.T,
) {
	clock := newTestResidencyClock()
	var loadCalls atomic.Int32
	unloaded := make(chan struct{}, 1)
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: 15 * time.Second,
		idleTimeout: 15 * time.Minute,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			loadCalls.Add(1)
			return nil
		},
		unload: func(context.Context, string) error {
			unloaded <- struct{}{}
			return nil
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	firstLease, err := controller.Acquire(context.Background(), "collection")
	if err != nil {
		t.Fatalf("first Acquire returned error: %v", err)
	}
	firstLease.Release()
	staleTimer := clock.LastTimer()

	clock.Advance(10 * time.Minute)
	secondLease, err := controller.Acquire(context.Background(), "collection")
	if err != nil {
		t.Fatalf("second Acquire returned error: %v", err)
	}
	if got := loadCalls.Load(); got != 1 {
		t.Fatalf("load calls after ready reacquisition = %d, want 1", got)
	}
	secondLease.Release()

	staleTimer.FireStale()
	select {
	case <-unloaded:
		t.Fatal("stale idle callback unloaded a reused collection")
	default:
	}
	clock.Advance(14*time.Minute + 59*time.Second)
	select {
	case <-unloaded:
		t.Fatal("collection unloaded before the fresh final-release deadline")
	default:
	}

	clock.Advance(time.Second)
	select {
	case <-unloaded:
	case <-time.After(time.Second):
		t.Fatal("collection did not unload at the fresh final-release deadline")
	}
}

func TestResidencyObservationDoesNotLoadOrPostponeIdleDeadline(t *testing.T) {
	clock := newTestResidencyClock()
	var loadCalls atomic.Int32
	unloaded := make(chan struct{}, 1)
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: 15 * time.Second,
		idleTimeout: 15 * time.Minute,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			loadCalls.Add(1)
			return nil
		},
		unload: func(context.Context, string) error {
			unloaded <- struct{}{}
			return nil
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	lease, err := controller.Acquire(context.Background(), "collection")
	if err != nil {
		t.Fatalf("Acquire returned error: %v", err)
	}
	lease.Release()
	clock.Advance(14 * time.Minute)

	state, observation, err := controller.Observe(context.Background(), "collection")
	if err != nil {
		t.Fatalf("Observe returned error: %v", err)
	}
	if state != collectionResidencyReady {
		t.Fatalf("observed state = %v, want ready", state)
	}
	clock.Advance(2 * time.Minute)
	select {
	case <-unloaded:
		t.Fatal("active observation did not protect the collection")
	default:
	}

	observation.Release()
	clock.Advance(0)
	select {
	case <-unloaded:
	case <-time.After(time.Second):
		t.Fatal("overdue collection did not unload after its observation ended")
	}

	state, coldObservation, err := controller.Observe(context.Background(), "collection")
	if err != nil {
		t.Fatalf("cold Observe returned error: %v", err)
	}
	defer coldObservation.Release()
	if state != collectionResidencyCold {
		t.Fatalf("cold observed state = %v, want cold", state)
	}
	if got := loadCalls.Load(); got != 1 {
		t.Fatalf("load calls after cold observation = %d, want 1", got)
	}
}

func TestResidencyObservationKeepsStartupReconciliationLive(t *testing.T) {
	clock := newTestResidencyClock()
	unloaded := make(chan struct{}, 1)
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: 15 * time.Second,
		idleTimeout: time.Minute,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			return nil
		},
		unload: func(context.Context, string) error {
			unloaded <- struct{}{}
			return nil
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	generation := controller.beginReconciliation()
	state, observation, err := controller.Observe(context.Background(), "collection")
	if err != nil {
		t.Fatalf("Observe returned error: %v", err)
	}
	if state != collectionResidencyUnknown {
		t.Fatalf("state before reconciliation = %v, want unknown", state)
	}
	controller.applyReconciliation(
		context.Background(),
		generation,
		"collection",
		collectionResidencyReady,
	)

	controller.mutex.Lock()
	reconciledState := controller.entries["collection"].state
	controller.mutex.Unlock()
	if reconciledState != collectionResidencyReady {
		t.Fatalf("state after reconciliation = %v, want ready", reconciledState)
	}
	clock.Advance(2 * time.Minute)
	select {
	case <-unloaded:
		t.Fatal("active observation did not protect the reconciled collection")
	default:
	}

	observation.Release()
	clock.Advance(0)
	select {
	case <-unloaded:
	case <-time.After(time.Second):
		t.Fatal("overdue reconciled collection did not unload after observation")
	}
}

func TestResidencyLoadCompletionPreservesDeadlineUnderObservation(t *testing.T) {
	clock := newTestResidencyClock()
	loadStarted := make(chan struct{})
	finishLoad := make(chan struct{})
	unloaded := make(chan struct{}, 1)
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: 15 * time.Second,
		idleTimeout: time.Minute,
		loadCeiling: 5 * time.Minute,
		load: func(ctx context.Context, _ string) error {
			close(loadStarted)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-finishLoad:
				return nil
			}
		},
		unload: func(context.Context, string) error {
			unloaded <- struct{}{}
			return nil
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	callerCtx, cancelCaller := context.WithCancel(context.Background())
	acquireResult := make(chan error, 1)
	go func() {
		_, err := controller.Acquire(callerCtx, "collection")
		acquireResult <- err
	}()
	<-loadStarted
	state, observation, err := controller.Observe(context.Background(), "collection")
	if err != nil {
		t.Fatalf("Observe returned error: %v", err)
	}
	if state != collectionResidencyLoading {
		t.Fatalf("observed state = %v, want loading", state)
	}
	cancelCaller()
	if err := <-acquireResult; !errors.Is(err, context.Canceled) {
		t.Fatalf("Acquire error = %v, want context.Canceled", err)
	}

	close(finishLoad)
	waitForResidencyState(t, controller, "collection", collectionResidencyReady)
	clock.Advance(2 * time.Minute)
	select {
	case <-unloaded:
		t.Fatal("active observation did not protect the completed load")
	default:
	}

	observation.Release()
	clock.Advance(0)
	select {
	case <-unloaded:
	case <-time.After(time.Second):
		t.Fatal("overdue load did not unload after its observation ended")
	}
}

func TestCollectionObservationPreservesControllerLoadingBeforeMilvusCatchesUp(t *testing.T) {
	loadStarted := make(chan struct{})
	finishLoad := make(chan struct{})
	var finishLoadOnce sync.Once
	finish := func() { finishLoadOnce.Do(func() { close(finishLoad) }) }
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       newTestResidencyClock(),
		waitTimeout: 15 * time.Second,
		loadCeiling: time.Minute,
		load: func(ctx context.Context, _ string) error {
			close(loadStarted)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-finishLoad:
				return nil
			}
		},
	})
	t.Cleanup(func() {
		finish()
		_ = controller.Close(context.Background())
	})

	acquireResult := make(chan error, 1)
	go func() {
		lease, err := controller.Acquire(context.Background(), "collection")
		if lease != nil {
			lease.Release()
		}
		acquireResult <- err
	}()
	<-loadStarted
	controllerState, observation, err := controller.Observe(
		context.Background(),
		"collection",
	)
	if err != nil {
		t.Fatalf("Observe returned error: %v", err)
	}
	observation.Release()
	if got := resolveObservedCollectionState(
		controllerState,
		entity.LoadStateNotLoad,
	); got != CollectionStateLoading {
		t.Fatalf("observed state = %q, want %q", got, CollectionStateLoading)
	}

	finish()
	if err := <-acquireResult; err != nil {
		t.Fatalf("Acquire returned error: %v", err)
	}
}

func TestResidencyPinDoesNotLoadAndReleaseStartsIdleDeadline(t *testing.T) {
	clock := newTestResidencyClock()
	var loadCalls atomic.Int32
	unloaded := make(chan struct{}, 1)
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: 15 * time.Second,
		idleTimeout: 15 * time.Minute,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			loadCalls.Add(1)
			return nil
		},
		unload: func(context.Context, string) error {
			unloaded <- struct{}{}
			return nil
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	pin, err := controller.Pin("collection")
	if err != nil {
		t.Fatalf("Pin returned error: %v", err)
	}
	if got := loadCalls.Load(); got != 0 {
		t.Fatalf("load calls after Pin = %d, want 0", got)
	}

	lease, err := controller.Acquire(context.Background(), "collection")
	if err != nil {
		t.Fatalf("Acquire returned error: %v", err)
	}
	lease.Release()
	clock.Advance(30 * time.Minute)
	select {
	case <-unloaded:
		t.Fatal("pinned collection unloaded")
	default:
	}

	pin.Release()
	clock.Advance(14*time.Minute + 59*time.Second)
	select {
	case <-unloaded:
		t.Fatal("collection unloaded before the post-pin idle deadline")
	default:
	}
	clock.Advance(time.Second)
	select {
	case <-unloaded:
	case <-time.After(time.Second):
		t.Fatal("collection did not unload after the post-pin idle deadline")
	}
}

func TestResidencyPinWaitsForActiveUnload(t *testing.T) {
	clock := newTestResidencyClock()
	unloadStarted := make(chan struct{})
	finishUnload := make(chan struct{})
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: 15 * time.Second,
		idleTimeout: time.Minute,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			return nil
		},
		unload: func(ctx context.Context, _ string) error {
			close(unloadStarted)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-finishUnload:
				return nil
			}
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	lease, err := controller.Acquire(context.Background(), "collection")
	if err != nil {
		t.Fatalf("Acquire returned error: %v", err)
	}
	lease.Release()
	clock.Advance(time.Minute)
	<-unloadStarted

	pinResult := make(chan *collectionPin, 1)
	pinErrors := make(chan error, 1)
	go func() {
		pin, pinErr := controller.Pin("collection")
		pinResult <- pin
		pinErrors <- pinErr
	}()
	select {
	case err := <-pinErrors:
		t.Fatalf("Pin returned during unload: %v", err)
	case <-time.After(50 * time.Millisecond):
	}

	close(finishUnload)
	pin := <-pinResult
	if err := <-pinErrors; err != nil {
		t.Fatalf("Pin returned error after unload: %v", err)
	}
	pin.Release()
}

func TestResidencyMaintenanceWaitsForHoldersAndBlocksNewReaders(t *testing.T) {
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       newTestResidencyClock(),
		waitTimeout: 15 * time.Second,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			return nil
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	activeLease, err := controller.Acquire(context.Background(), "collection")
	if err != nil {
		t.Fatalf("Acquire returned error: %v", err)
	}
	maintenanceResult := make(chan *collectionMaintenance, 1)
	maintenanceErrors := make(chan error, 1)
	go func() {
		maintenance, maintainErr := controller.Maintain(
			context.Background(),
			"collection",
		)
		maintenanceResult <- maintenance
		maintenanceErrors <- maintainErr
	}()
	waitForMaintenance(t, controller, "collection")

	leaseResult := make(chan *collectionLease, 1)
	leaseErrors := make(chan error, 1)
	go func() {
		lease, acquireErr := controller.Acquire(context.Background(), "collection")
		leaseResult <- lease
		leaseErrors <- acquireErr
	}()
	observationResult := make(chan *collectionObservation, 1)
	observationErrors := make(chan error, 1)
	go func() {
		_, observation, observeErr := controller.Observe(
			context.Background(),
			"collection",
		)
		observationResult <- observation
		observationErrors <- observeErr
	}()
	pinResult := make(chan *collectionPin, 1)
	pinErrors := make(chan error, 1)
	go func() {
		pin, pinErr := controller.Pin("collection")
		pinResult <- pin
		pinErrors <- pinErr
	}()
	select {
	case err := <-leaseErrors:
		t.Fatalf("Acquire returned during maintenance setup: %v", err)
	default:
	}
	select {
	case err := <-observationErrors:
		t.Fatalf("Observe returned during maintenance setup: %v", err)
	default:
	}
	select {
	case err := <-pinErrors:
		t.Fatalf("Pin returned during maintenance setup: %v", err)
	case <-time.After(50 * time.Millisecond):
	}

	activeLease.Release()
	maintenance := <-maintenanceResult
	if err := <-maintenanceErrors; err != nil {
		t.Fatalf("Maintain returned error: %v", err)
	}
	select {
	case err := <-leaseErrors:
		t.Fatalf("Acquire returned while maintenance remained active: %v", err)
	default:
	}
	select {
	case err := <-pinErrors:
		t.Fatalf("Pin returned while maintenance remained active: %v", err)
	default:
	}

	maintenance.Release()
	lease := <-leaseResult
	if err := <-leaseErrors; err != nil {
		t.Fatalf("Acquire returned error after maintenance: %v", err)
	}
	lease.Release()
	observation := <-observationResult
	if err := <-observationErrors; err != nil {
		t.Fatalf("Observe returned error after maintenance: %v", err)
	}
	observation.Release()
	pin := <-pinResult
	if err := <-pinErrors; err != nil {
		t.Fatalf("Pin returned error after maintenance: %v", err)
	}
	pin.Release()
}

func TestResidencyCanceledMaintenanceUnblocksEveryCollection(t *testing.T) {
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       newTestResidencyClock(),
		waitTimeout: 15 * time.Second,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			return nil
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	maintenance, err := controller.Maintain(ctx, "zeta", "alpha", "zeta")
	if maintenance != nil {
		maintenance.Release()
		t.Fatal("Maintain returned a hold for a canceled request")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Maintain error = %v, want context.Canceled", err)
	}

	for _, collectionName := range []string{"alpha", "zeta"} {
		lease, acquireErr := controller.Acquire(context.Background(), collectionName)
		if acquireErr != nil {
			t.Fatalf("Acquire(%s) after cancellation returned error: %v", collectionName, acquireErr)
		}
		lease.Release()
	}
}

func TestResidencyMaintenanceCancellationReleasesSortedPartialAcquisition(t *testing.T) {
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       newTestResidencyClock(),
		waitTimeout: 15 * time.Second,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			return nil
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	held, err := controller.Maintain(context.Background(), "beta")
	if err != nil {
		t.Fatalf("Maintain(beta) returned error: %v", err)
	}
	t.Cleanup(held.Release)

	ctx, cancel := context.WithCancel(context.Background())
	maintenanceResult := make(chan *collectionMaintenance, 1)
	maintenanceErrors := make(chan error, 1)
	go func() {
		maintenance, maintainErr := controller.Maintain(ctx, "zeta", "beta", "alpha")
		maintenanceResult <- maintenance
		maintenanceErrors <- maintainErr
	}()
	waitForMaintenance(t, controller, "alpha")

	controller.mutex.Lock()
	zetaHeld := controller.entries["zeta"] != nil && controller.entries["zeta"].maintenance
	controller.mutex.Unlock()
	if zetaHeld {
		t.Fatal("Maintain acquired zeta before blocked beta")
	}

	cancel()
	maintenance := <-maintenanceResult
	if maintenance != nil {
		maintenance.Release()
		t.Fatal("Maintain returned a hold after cancellation")
	}
	if err := <-maintenanceErrors; !errors.Is(err, context.Canceled) {
		t.Fatalf("Maintain error = %v, want context.Canceled", err)
	}

	controller.mutex.Lock()
	alphaHeld := controller.entries["alpha"] != nil && controller.entries["alpha"].maintenance
	betaHeld := controller.entries["beta"] != nil && controller.entries["beta"].maintenance
	controller.mutex.Unlock()
	if alphaHeld {
		t.Fatal("canceled Maintain retained partial alpha acquisition")
	}
	if !betaHeld {
		t.Fatal("canceled Maintain released another request's beta hold")
	}
}

func TestResidencyMaintenanceReleaseStartsIdleUnload(t *testing.T) {
	clock := newTestResidencyClock()
	unloads := make(chan string, 1)
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: 15 * time.Second,
		loadCeiling: time.Minute,
		idleTimeout: time.Minute,
		load: func(context.Context, string) error {
			return nil
		},
		unload: func(_ context.Context, collectionName string) error {
			unloads <- collectionName
			return nil
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	lease, err := controller.Acquire(context.Background(), "collection")
	if err != nil {
		t.Fatalf("Acquire returned error: %v", err)
	}
	lease.Release()
	maintenance, err := controller.Maintain(context.Background(), "collection")
	if err != nil {
		t.Fatalf("Maintain returned error: %v", err)
	}

	clock.Advance(2 * time.Minute)
	select {
	case collectionName := <-unloads:
		t.Fatalf("unloaded %q while maintenance remained active", collectionName)
	default:
	}

	maintenance.Release()
	clock.Advance(time.Minute - time.Nanosecond)
	select {
	case collectionName := <-unloads:
		t.Fatalf("unloaded %q before the post-maintenance idle interval", collectionName)
	default:
	}
	clock.Advance(time.Nanosecond)
	if collectionName := <-unloads; collectionName != "collection" {
		t.Fatalf("unloaded %q, want collection", collectionName)
	}
}

func TestResidencyRecoveryMaintenanceReleaseNeverStartsIdleUnload(t *testing.T) {
	clock := newTestResidencyClock()
	unloads := make(chan string, 1)
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: 15 * time.Second,
		loadCeiling: time.Minute,
		idleTimeout: time.Minute,
		load: func(context.Context, string) error {
			return nil
		},
		unload: func(_ context.Context, collectionName string) error {
			unloads <- collectionName
			return nil
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	recoveryName := "collection" + recoveryCollectionSuffix
	lease, err := controller.Acquire(context.Background(), recoveryName)
	if err != nil {
		t.Fatalf("Acquire returned error: %v", err)
	}
	lease.Release()
	maintenance, err := controller.Maintain(context.Background(), recoveryName)
	if err != nil {
		t.Fatalf("Maintain returned error: %v", err)
	}
	maintenance.Release()

	controller.mutex.Lock()
	entry := controller.entries[recoveryName]
	idleTimer := entry.idleTimer
	idleDeadline := entry.idleDeadline
	controller.mutex.Unlock()
	if idleTimer != nil || !idleDeadline.IsZero() {
		t.Fatal("recovery collection received an idle timer after maintenance")
	}
	clock.Advance(2 * time.Minute)
	select {
	case collectionName := <-unloads:
		t.Fatalf("unloaded recovery collection %q", collectionName)
	default:
	}
}

func TestResidencyUnloadTransitionBlocksNewReaders(t *testing.T) {
	clock := newTestResidencyClock()
	unloadStarted := make(chan struct{})
	finishUnload := make(chan struct{})
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: 15 * time.Second,
		idleTimeout: time.Minute,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			return nil
		},
		unload: func(ctx context.Context, _ string) error {
			close(unloadStarted)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-finishUnload:
				return nil
			}
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	lease, err := controller.Acquire(context.Background(), "collection")
	if err != nil {
		t.Fatalf("Acquire returned error: %v", err)
	}
	lease.Release()
	clock.Advance(time.Minute)
	<-unloadStarted

	leaseErrors := make(chan error, 1)
	go func() {
		newLease, acquireErr := controller.Acquire(context.Background(), "collection")
		if newLease != nil {
			newLease.Release()
		}
		leaseErrors <- acquireErr
	}()
	observationErrors := make(chan error, 1)
	go func() {
		_, observation, observeErr := controller.Observe(
			context.Background(),
			"collection",
		)
		if observation != nil {
			observation.Release()
		}
		observationErrors <- observeErr
	}()
	select {
	case err := <-leaseErrors:
		t.Fatalf("Acquire returned during unload: %v", err)
	case <-time.After(50 * time.Millisecond):
	}
	select {
	case err := <-observationErrors:
		t.Fatalf("Observe returned during unload: %v", err)
	case <-time.After(50 * time.Millisecond):
	}

	close(finishUnload)
	if err := <-leaseErrors; err != nil {
		t.Fatalf("Acquire after unload returned error: %v", err)
	}
	if err := <-observationErrors; err != nil {
		t.Fatalf("Observe after unload returned error: %v", err)
	}
}

func TestResidencyUnloadFailureForcesNextAcquireToReload(t *testing.T) {
	clock := newTestResidencyClock()
	var loadCount atomic.Int32
	unloadDone := make(chan struct{}, 1)
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: time.Minute,
		idleTimeout: time.Minute,
		loadCeiling: 5 * time.Minute,
		load: func(context.Context, string) error {
			loadCount.Add(1)
			return nil
		},
		unload: func(context.Context, string) error {
			unloadDone <- struct{}{}
			return errors.New("release state unavailable")
		},
	})
	t.Cleanup(func() { _ = controller.Close(context.Background()) })

	lease, err := controller.Acquire(context.Background(), "collection")
	if err != nil {
		t.Fatalf("initial Acquire returned error: %v", err)
	}
	lease.Release()
	clock.Advance(time.Minute)
	<-unloadDone

	lease, err = controller.Acquire(context.Background(), "collection")
	if err != nil {
		t.Fatalf("Acquire after failed unload returned error: %v", err)
	}
	lease.Release()
	if got := loadCount.Load(); got != 2 {
		t.Fatalf("load count = %d, want 2 after failed unload", got)
	}
}

func TestServiceCloseStopsActiveResidencyTransition(t *testing.T) {
	loadStarted := make(chan struct{})
	loadStopped := make(chan struct{})
	service := &Service{}
	service.initializeResidencyControllerWithLoad(func(ctx context.Context, _ string) error {
		close(loadStarted)
		<-ctx.Done()
		close(loadStopped)
		return ctx.Err()
	})
	controller := service.residency

	acquireResult := make(chan error, 1)
	go func() {
		_, err := controller.Acquire(context.Background(), "collection")
		acquireResult <- err
	}()
	<-loadStarted

	if err := service.Close(context.Background()); err != nil {
		t.Fatalf("Service.Close returned error: %v", err)
	}
	select {
	case <-loadStopped:
	default:
		t.Fatal("Service.Close returned before the active residency transition stopped")
	}
	if err := <-acquireResult; !errors.Is(err, ErrResidencyControllerClosed) {
		t.Fatalf("Acquire error after Service.Close = %v, want controller closed", err)
	}
}

func TestServiceCloseClosesMilvusAfterResidencyDeadline(t *testing.T) {
	loadStarted := make(chan struct{})
	loadStopped := make(chan struct{})
	service := &Service{milvus: &milvusclient.Client{}}
	service.available.Store(true)
	service.initializeResidencyControllerWithLoad(func(ctx context.Context, _ string) error {
		close(loadStarted)
		<-ctx.Done()
		close(loadStopped)
		return ctx.Err()
	})

	acquireResult := make(chan error, 1)
	go func() {
		_, err := service.residency.Acquire(context.Background(), "collection")
		acquireResult <- err
	}()
	<-loadStarted

	closeCtx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := service.Close(closeCtx); !errors.Is(err, context.Canceled) {
		t.Fatalf("Service.Close error = %v, want context canceled", err)
	}
	if service.Available() {
		t.Fatal("Service remained available after Milvus client close")
	}
	<-loadStopped
	if err := <-acquireResult; !errors.Is(err, ErrResidencyControllerClosed) {
		t.Fatalf("Acquire error after Service.Close = %v, want controller closed", err)
	}
}

func TestResidencyRenameTransfersPinToNewCollectionName(t *testing.T) {
	clock := newTestResidencyClock()
	unloads := make(chan string, 1)
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: 15 * time.Second,
		idleTimeout: time.Minute,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			return nil
		},
		unload: func(_ context.Context, collectionName string) error {
			unloads <- collectionName
			return nil
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	lease, err := controller.Acquire(context.Background(), "collection_stg")
	if err != nil {
		t.Fatalf("Acquire returned error: %v", err)
	}
	lease.Release()
	pin, err := controller.Pin("collection_stg")
	if err != nil {
		t.Fatalf("Pin returned error: %v", err)
	}
	maintenance, err := controller.Maintain(
		context.Background(),
		"collection",
		"collection_stg",
	)
	if err != nil {
		t.Fatalf("Maintain returned error: %v", err)
	}
	controller.Rename("collection_stg", "collection")
	maintenance.Release()

	clock.Advance(2 * time.Minute)
	select {
	case collectionName := <-unloads:
		t.Fatalf("unloaded %q while the transferred pin remained active", collectionName)
	default:
	}
	pin.Release()
	clock.Advance(time.Minute)
	if collectionName := <-unloads; collectionName != "collection" {
		t.Fatalf("unloaded %q, want collection", collectionName)
	}
}

func TestResidencyForgetPreservesActivePinEntry(t *testing.T) {
	clock := newTestResidencyClock()
	unloads := make(chan string, 1)
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: 15 * time.Second,
		idleTimeout: time.Minute,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			return nil
		},
		unload: func(_ context.Context, collectionName string) error {
			unloads <- collectionName
			return nil
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	const collectionName = "collection_stg"
	pin, err := controller.Pin(collectionName)
	if err != nil {
		t.Fatalf("Pin returned error: %v", err)
	}
	pinnedEntry := pin.entry
	controller.Forget(collectionName)

	controller.mutex.Lock()
	mappedEntry := controller.entries[collectionName]
	pinnedCount := pinnedEntry.pins
	mappedPinCount := 0
	if mappedEntry != nil {
		mappedPinCount = mappedEntry.pins
	}
	controller.mutex.Unlock()
	t.Logf(
		"pin lifecycle collection=%q pin_entry=%p pin_count=%d mapped_entry=%p mapped_pin_count=%d",
		collectionName,
		pinnedEntry,
		pinnedCount,
		mappedEntry,
		mappedPinCount,
	)
	if pinnedCount != 1 {
		t.Fatalf("active pin count = %d, want 1 before handle release", pinnedCount)
	}
	if mappedEntry != pinnedEntry {
		t.Fatalf(
			"Forget replaced the active pin entry: pin_entry=%p mapped_entry=%p",
			pinnedEntry,
			mappedEntry,
		)
	}

	lease, err := controller.Acquire(context.Background(), collectionName)
	if err != nil {
		t.Fatalf("Acquire after Forget returned error: %v", err)
	}
	lease.Release()
	clock.Advance(2 * time.Minute)
	select {
	case unloadedCollection := <-unloads:
		t.Fatalf("unloaded %q while the original pin remained active", unloadedCollection)
	default:
	}

	pin.Release()
	clock.Advance(time.Minute)
	if unloadedCollection := <-unloads; unloadedCollection != collectionName {
		t.Fatalf("unloaded %q, want %q", unloadedCollection, collectionName)
	}
}

func TestResidencyForgetPreservesActiveLeaseAndObservationEntry(t *testing.T) {
	clock := newTestResidencyClock()
	var loadCalls atomic.Int32
	unloads := make(chan string, 1)
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: 15 * time.Second,
		idleTimeout: time.Minute,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			loadCalls.Add(1)
			return nil
		},
		unload: func(_ context.Context, collectionName string) error {
			unloads <- collectionName
			return nil
		},
	})
	t.Cleanup(func() {
		_ = controller.Close(context.Background())
	})

	const collectionName = "collection"
	lease, err := controller.Acquire(context.Background(), collectionName)
	if err != nil {
		t.Fatalf("Acquire returned error: %v", err)
	}
	_, observation, err := controller.Observe(context.Background(), collectionName)
	if err != nil {
		t.Fatalf("Observe returned error: %v", err)
	}

	controller.mutex.Lock()
	protectedEntry := controller.entries[collectionName]
	controller.mutex.Unlock()
	controller.Forget(collectionName)

	controller.mutex.Lock()
	mappedEntry := controller.entries[collectionName]
	leaseCount := protectedEntry.leases
	observationCount := protectedEntry.observations
	controller.mutex.Unlock()
	if mappedEntry != protectedEntry {
		t.Fatalf(
			"Forget replaced an entry with active holders: protected_entry=%p mapped_entry=%p",
			protectedEntry,
			mappedEntry,
		)
	}
	if leaseCount != 1 || observationCount != 1 {
		t.Fatalf(
			"holder counts after Forget = leases %d observations %d, want 1 and 1",
			leaseCount,
			observationCount,
		)
	}

	observation.Release()
	lease.Release()
	reloadedLease, err := controller.Acquire(context.Background(), collectionName)
	if err != nil {
		t.Fatalf("Acquire after holder release returned error: %v", err)
	}
	reloadedLease.Release()
	if got := loadCalls.Load(); got != 2 {
		t.Fatalf("load calls after confirmed absence = %d, want 2", got)
	}
	clock.Advance(time.Minute)
	if unloadedCollection := <-unloads; unloadedCollection != collectionName {
		t.Fatalf("unloaded %q, want %q", unloadedCollection, collectionName)
	}
}

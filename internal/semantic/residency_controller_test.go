package semantic

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
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

func TestServiceCloseStopsActiveResidencyTransition(t *testing.T) {
	loadStarted := make(chan struct{})
	loadStopped := make(chan struct{})
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       newTestResidencyClock(),
		waitTimeout: 15 * time.Second,
		loadCeiling: time.Minute,
		load: func(ctx context.Context, _ string) error {
			close(loadStarted)
			<-ctx.Done()
			close(loadStopped)
			return ctx.Err()
		},
	})

	acquireResult := make(chan error, 1)
	go func() {
		_, err := controller.Acquire(context.Background(), "collection")
		acquireResult <- err
	}()
	<-loadStarted

	service := &Service{residency: controller}
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

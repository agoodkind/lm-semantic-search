package semantic

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"sync"
	"time"

	internalclock "goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/semantic/milvusgrpc"
)

const (
	defaultCollectionLoadWaitTimeout = 15 * time.Second
	defaultCollectionIdleTimeout     = 15 * time.Minute
)

var (
	// ErrCollectionLoadWaitTimeout reports that one caller exhausted its load wait.
	ErrCollectionLoadWaitTimeout = errors.New("collection load wait timed out")
	// ErrResidencyControllerClosed reports that shutdown rejects new protection.
	ErrResidencyControllerClosed = errors.New("collection residency controller closed")
)

type collectionTransition func(context.Context, string) error

type residencyTimer interface {
	Stop() bool
}

type residencyClock interface {
	Now() time.Time
	AfterFunc(time.Duration, func()) residencyTimer
}

type wallResidencyClock struct{}

func (wallResidencyClock) Now() time.Time {
	return internalclock.Now()
}

func (wallResidencyClock) AfterFunc(delay time.Duration, callback func()) residencyTimer {
	return time.AfterFunc(delay, callback)
}

type residencyControllerConfig struct {
	clock       residencyClock
	waitTimeout time.Duration
	idleTimeout time.Duration
	loadCeiling time.Duration
	load        collectionTransition
	unload      collectionTransition
}

type collectionResidencyState int

const (
	collectionResidencyUnknown collectionResidencyState = iota
	collectionResidencyCold
	collectionResidencyLoading
	collectionResidencyReady
)

type collectionResidencyLoad struct {
	done   chan struct{}
	cancel context.CancelFunc
	err    error
}

type collectionResidencyEntry struct {
	state            collectionResidencyState
	leases           int
	observations     int
	pins             int
	load             *collectionResidencyLoad
	idleTimer        residencyTimer
	idleGeneration   uint64
	idleDeadline     time.Time
	activeTransition context.CancelFunc
	maintenance      bool
	changed          chan struct{}
}

type collectionResidencyController struct {
	mutex       sync.Mutex
	config      residencyControllerConfig
	entries     map[string]*collectionResidencyEntry
	closed      bool
	closedCh    chan struct{}
	transitions sync.WaitGroup
}

type collectionLease struct {
	once       sync.Once
	controller *collectionResidencyController
	name       string
}

type collectionObservation struct {
	once       sync.Once
	controller *collectionResidencyController
	name       string
}

type collectionPin struct {
	once       sync.Once
	controller *collectionResidencyController
	name       string
}

type collectionMaintenance struct {
	once       sync.Once
	controller *collectionResidencyController
	names      []string
}

func (service *Service) initializeResidencyController() {
	service.initializeResidencyControllerWithLoad(service.loadCollectionTransition)
}

func (service *Service) initializeResidencyControllerWithLoad(load collectionTransition) {
	service.residency = newCollectionResidencyController(residencyControllerConfig{
		clock:       wallResidencyClock{},
		waitTimeout: defaultCollectionLoadWaitTimeout,
		// Task 4 enables this only after every collection operation holds protection.
		idleTimeout: 0,
		loadCeiling: service.sharedCollectionLoadCeiling(),
		load:        load,
		unload: func(ctx context.Context, collectionName string) error {
			if err := service.releaseCollection(ctx, collectionName); err != nil {
				return err
			}
			return service.awaitCollectionReleased(ctx, collectionName)
		},
	})
}

func newCollectionResidencyController(
	config residencyControllerConfig,
) *collectionResidencyController {
	if config.clock == nil {
		config.clock = wallResidencyClock{}
	}
	if config.waitTimeout <= 0 {
		config.waitTimeout = defaultCollectionLoadWaitTimeout
	}
	if config.loadCeiling <= 0 {
		config.loadCeiling = milvusgrpc.DefaultCallTimeouts().Metadata +
			3*defaultCollectionLoadBound
	}
	return &collectionResidencyController{
		mutex:       sync.Mutex{},
		config:      config,
		entries:     make(map[string]*collectionResidencyEntry),
		closed:      false,
		closedCh:    make(chan struct{}),
		transitions: sync.WaitGroup{},
	}
}

func (controller *collectionResidencyController) Acquire(
	ctx context.Context,
	collectionName string,
) (*collectionLease, error) {
	for {
		if err := ctx.Err(); err != nil {
			wrappedErr := fmt.Errorf("acquire collection residency: %w", err)
			slog.WarnContext(ctx, "collection residency acquire canceled", "err", wrappedErr)
			return nil, wrappedErr
		}
		controller.mutex.Lock()
		if controller.closed {
			controller.mutex.Unlock()
			return nil, ErrResidencyControllerClosed
		}
		entry := controller.entryLocked(collectionName)
		if entry.maintenance ||
			(entry.activeTransition != nil && entry.load == nil) {
			changed := entry.changed
			controller.mutex.Unlock()
			if err := controller.waitForChange(ctx, changed); err != nil {
				return nil, err
			}
			continue
		}
		entry.leases++
		controller.cancelIdleTimerLocked(entry)
		if entry.state == collectionResidencyReady {
			controller.mutex.Unlock()
			return controller.newLease(collectionName), nil
		}
		flight := entry.load
		if flight == nil {
			flight = controller.startLoadLocked(ctx, collectionName, entry)
		}
		controller.mutex.Unlock()

		err := controller.waitForLoad(ctx, collectionName, flight)
		if err != nil {
			controller.releaseLease(ctx, collectionName)
			return nil, err
		}
		return controller.newLease(collectionName), nil
	}
}

func (controller *collectionResidencyController) Observe(
	ctx context.Context,
	collectionName string,
) (collectionResidencyState, *collectionObservation, error) {
	for {
		if err := ctx.Err(); err != nil {
			wrappedErr := fmt.Errorf(
				"observe collection residency: %w",
				err,
			)
			slog.WarnContext(ctx, "collection residency observation canceled", "err", wrappedErr)
			return collectionResidencyUnknown, nil, wrappedErr
		}
		controller.mutex.Lock()
		if controller.closed {
			controller.mutex.Unlock()
			return collectionResidencyUnknown, nil, ErrResidencyControllerClosed
		}
		entry := controller.entryLocked(collectionName)
		if entry.maintenance ||
			(entry.activeTransition != nil && entry.load == nil) {
			changed := entry.changed
			controller.mutex.Unlock()
			if err := controller.waitForChange(ctx, changed); err != nil {
				return collectionResidencyUnknown, nil, err
			}
			continue
		}
		entry.observations++
		controller.pauseIdleTimerLocked(entry)
		observation := &collectionObservation{
			once:       sync.Once{},
			controller: controller,
			name:       collectionName,
		}
		state := entry.state
		controller.mutex.Unlock()
		return state, observation, nil
	}
}

func (controller *collectionResidencyController) Pin(
	collectionName string,
) (*collectionPin, error) {
	for {
		controller.mutex.Lock()
		if controller.closed {
			controller.mutex.Unlock()
			return nil, ErrResidencyControllerClosed
		}
		entry := controller.entryLocked(collectionName)
		if entry.activeTransition != nil && entry.load == nil {
			changed := entry.changed
			controller.mutex.Unlock()
			select {
			case <-controller.closedCh:
				return nil, ErrResidencyControllerClosed
			case <-changed:
			}
			continue
		}
		entry.pins++
		controller.cancelIdleTimerLocked(entry)
		controller.mutex.Unlock()
		return &collectionPin{
			once:       sync.Once{},
			controller: controller,
			name:       collectionName,
		}, nil
	}
}

func (controller *collectionResidencyController) Maintain(
	ctx context.Context,
	collectionNames ...string,
) (*collectionMaintenance, error) {
	names := sortedUniqueCollectionNames(collectionNames)
	acquired := make([]string, 0, len(names))
	for _, collectionName := range names {
		for {
			if err := ctx.Err(); err != nil {
				controller.releaseMaintenance(ctx, acquired)
				return nil, collectionMaintenanceContextError(ctx, err)
			}
			controller.mutex.Lock()
			if controller.closed {
				controller.releaseMaintenanceLocked(ctx, acquired)
				controller.mutex.Unlock()
				return nil, ErrResidencyControllerClosed
			}
			entry := controller.entryLocked(collectionName)
			if !entry.maintenance {
				entry.maintenance = true
				controller.cancelIdleTimerLocked(entry)
				controller.notifyLocked(entry)
				acquired = append(acquired, collectionName)
				controller.mutex.Unlock()
				break
			}
			changed := entry.changed
			controller.mutex.Unlock()
			if err := controller.waitForChange(ctx, changed); err != nil {
				controller.releaseMaintenance(ctx, acquired)
				return nil, err
			}
		}
	}

	for _, collectionName := range acquired {
		for {
			if err := ctx.Err(); err != nil {
				controller.releaseMaintenance(ctx, acquired)
				return nil, collectionMaintenanceContextError(ctx, err)
			}
			controller.mutex.Lock()
			entry := controller.entries[collectionName]
			if entry.leases == 0 && entry.observations == 0 &&
				entry.activeTransition == nil {
				controller.mutex.Unlock()
				break
			}
			changed := entry.changed
			controller.mutex.Unlock()
			if err := controller.waitForChange(ctx, changed); err != nil {
				controller.releaseMaintenance(ctx, acquired)
				return nil, err
			}
		}
	}
	if err := ctx.Err(); err != nil {
		controller.releaseMaintenance(ctx, acquired)
		return nil, collectionMaintenanceContextError(ctx, err)
	}
	return &collectionMaintenance{
		once:       sync.Once{},
		controller: controller,
		names:      acquired,
	}, nil
}

func collectionMaintenanceContextError(ctx context.Context, err error) error {
	wrappedErr := fmt.Errorf("acquire collection maintenance: %w", err)
	slog.WarnContext(ctx, "collection maintenance canceled", "err", wrappedErr)
	return wrappedErr
}

func sortedUniqueCollectionNames(collectionNames []string) []string {
	names := append([]string(nil), collectionNames...)
	sort.Strings(names)
	unique := names[:0]
	for _, name := range names {
		if len(unique) == 0 || unique[len(unique)-1] != name {
			unique = append(unique, name)
		}
	}
	return unique
}

func (controller *collectionResidencyController) waitForChange(
	ctx context.Context,
	changed <-chan struct{},
) error {
	select {
	case <-ctx.Done():
		err := fmt.Errorf("wait for collection residency state: %w", ctx.Err())
		slog.WarnContext(ctx, "collection residency state wait canceled", "err", err)
		return err
	case <-controller.closedCh:
		return ErrResidencyControllerClosed
	case <-changed:
		return nil
	}
}

func (controller *collectionResidencyController) entryLocked(
	collectionName string,
) *collectionResidencyEntry {
	entry := controller.entries[collectionName]
	if entry == nil {
		entry = &collectionResidencyEntry{
			state:            collectionResidencyUnknown,
			leases:           0,
			observations:     0,
			pins:             0,
			load:             nil,
			idleTimer:        nil,
			idleGeneration:   0,
			idleDeadline:     time.Time{},
			activeTransition: nil,
			maintenance:      false,
			changed:          make(chan struct{}),
		}
		controller.entries[collectionName] = entry
	}
	return entry
}

func (controller *collectionResidencyController) startLoadLocked(
	ctx context.Context,
	collectionName string,
	entry *collectionResidencyEntry,
) *collectionResidencyLoad {
	loadCtx, cancelLoad := context.WithCancel(context.WithoutCancel(ctx))
	flight := &collectionResidencyLoad{
		done:   make(chan struct{}),
		cancel: cancelLoad,
		err:    nil,
	}
	entry.state = collectionResidencyLoading
	entry.load = flight
	entry.activeTransition = cancelLoad
	controller.transitions.Add(1)
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				err := fmt.Errorf("collection load panic: %v", recovered)
				slog.ErrorContext(loadCtx, "semantic.collection_load_panic", "err", err)
				controller.finishLoad(loadCtx, collectionName, entry, flight, err)
			}
		}()
		controller.runLoad(loadCtx, collectionName, entry, flight)
	}()
	return flight
}

func (controller *collectionResidencyController) runLoad(
	ctx context.Context,
	collectionName string,
	entry *collectionResidencyEntry,
	flight *collectionResidencyLoad,
) {
	defer controller.transitions.Done()
	ceilingTimer := controller.config.clock.AfterFunc(
		controller.config.loadCeiling,
		flight.cancel,
	)
	err := controller.config.load(ctx, collectionName)
	ceilingTimer.Stop()

	controller.finishLoad(ctx, collectionName, entry, flight, err)
}

func (controller *collectionResidencyController) finishLoad(
	ctx context.Context,
	collectionName string,
	entry *collectionResidencyEntry,
	flight *collectionResidencyLoad,
	err error,
) {
	controller.mutex.Lock()
	defer controller.mutex.Unlock()
	if entry.load != flight {
		return
	}
	flight.err = err
	entry.load = nil
	entry.activeTransition = nil
	if err == nil {
		entry.state = collectionResidencyReady
		if entry.leases == 0 {
			controller.armIdleTimerLocked(ctx, collectionName, entry)
		}
	} else {
		entry.state = collectionResidencyUnknown
	}
	close(flight.done)
	controller.notifyLocked(entry)
}

func (controller *collectionResidencyController) waitForLoad(
	ctx context.Context,
	collectionName string,
	flight *collectionResidencyLoad,
) error {
	timedOut := make(chan struct{})
	timer := controller.config.clock.AfterFunc(controller.config.waitTimeout, func() {
		close(timedOut)
	})
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return collectionLoadWaitContextError(ctx, collectionName, ctx.Err())
	case <-controller.closedCh:
		return ErrResidencyControllerClosed
	case <-timedOut:
		if err := ctx.Err(); err != nil {
			return collectionLoadWaitContextError(ctx, collectionName, err)
		}
		err := fmt.Errorf(
			"wait for collection %s: %w",
			collectionName,
			ErrCollectionLoadWaitTimeout,
		)
		slog.WarnContext(ctx, "semantic.collection_load_wait_timeout", "err", err)
		return err
	case <-flight.done:
		if err := ctx.Err(); err != nil {
			return collectionLoadWaitContextError(ctx, collectionName, err)
		}
		return flight.err
	}
}

func collectionLoadWaitContextError(
	ctx context.Context,
	collectionName string,
	err error,
) error {
	wrappedErr := fmt.Errorf("wait for collection %s load: %w", collectionName, err)
	slog.WarnContext(ctx, "collection load wait canceled", "err", wrappedErr)
	return wrappedErr
}

func (controller *collectionResidencyController) newLease(
	collectionName string,
) *collectionLease {
	return &collectionLease{
		once:       sync.Once{},
		controller: controller,
		name:       collectionName,
	}
}

func (lease *collectionLease) Release() {
	if lease == nil {
		return
	}
	lease.once.Do(func() {
		lease.controller.releaseLease(context.Background(), lease.name)
	})
}

func (observation *collectionObservation) Release() {
	observation.ReleaseContext(context.Background())
}

func (observation *collectionObservation) ReleaseContext(ctx context.Context) {
	if observation == nil {
		return
	}
	observation.once.Do(func() {
		observation.controller.releaseObservation(ctx, observation.name)
	})
}

func (pin *collectionPin) Release() {
	if pin == nil {
		return
	}
	pin.once.Do(func() {
		pin.controller.releasePin(context.Background(), pin.name)
	})
}

func (maintenance *collectionMaintenance) Release() {
	maintenance.ReleaseContext(context.Background())
}

func (maintenance *collectionMaintenance) ReleaseContext(ctx context.Context) {
	if maintenance == nil {
		return
	}
	maintenance.once.Do(func() {
		maintenance.controller.releaseMaintenance(ctx, maintenance.names)
	})
}

func (controller *collectionResidencyController) releaseLease(
	ctx context.Context,
	collectionName string,
) {
	controller.mutex.Lock()
	defer controller.mutex.Unlock()
	entry := controller.entries[collectionName]
	if entry == nil || entry.leases == 0 {
		return
	}
	entry.leases--
	controller.notifyLocked(entry)
	if entry.leases == 0 && entry.state == collectionResidencyReady {
		controller.armIdleTimerLocked(ctx, collectionName, entry)
	}
}

func (controller *collectionResidencyController) releaseObservation(
	ctx context.Context,
	collectionName string,
) {
	controller.mutex.Lock()
	defer controller.mutex.Unlock()
	entry := controller.entries[collectionName]
	if entry == nil || entry.observations == 0 {
		return
	}
	entry.observations--
	controller.notifyLocked(entry)
	if entry.observations == 0 && entry.state == collectionResidencyReady {
		controller.resumeIdleTimerLocked(ctx, collectionName, entry)
	}
}

func (controller *collectionResidencyController) releasePin(
	ctx context.Context,
	collectionName string,
) {
	controller.mutex.Lock()
	defer controller.mutex.Unlock()
	entry := controller.entries[collectionName]
	if entry == nil || entry.pins == 0 {
		return
	}
	entry.pins--
	controller.notifyLocked(entry)
	if entry.pins == 0 && entry.state == collectionResidencyReady {
		controller.armIdleTimerLocked(ctx, collectionName, entry)
	}
}

func (controller *collectionResidencyController) releaseMaintenance(
	ctx context.Context,
	collectionNames []string,
) {
	controller.mutex.Lock()
	defer controller.mutex.Unlock()
	controller.releaseMaintenanceLocked(ctx, collectionNames)
}

func (controller *collectionResidencyController) releaseMaintenanceLocked(
	ctx context.Context,
	collectionNames []string,
) {
	for _, collectionName := range collectionNames {
		entry := controller.entries[collectionName]
		if entry == nil || !entry.maintenance {
			continue
		}
		entry.maintenance = false
		controller.notifyLocked(entry)
		if entry.state == collectionResidencyReady {
			controller.armIdleTimerLocked(ctx, collectionName, entry)
		}
	}
}

func (controller *collectionResidencyController) notifyLocked(
	entry *collectionResidencyEntry,
) {
	close(entry.changed)
	entry.changed = make(chan struct{})
}

func (controller *collectionResidencyController) cancelIdleTimerLocked(
	entry *collectionResidencyEntry,
) {
	controller.stopIdleTimerLocked(entry)
	entry.idleDeadline = time.Time{}
}

func (controller *collectionResidencyController) pauseIdleTimerLocked(
	entry *collectionResidencyEntry,
) {
	controller.stopIdleTimerLocked(entry)
}

func (controller *collectionResidencyController) stopIdleTimerLocked(
	entry *collectionResidencyEntry,
) {
	entry.idleGeneration++
	if entry.idleTimer != nil {
		entry.idleTimer.Stop()
		entry.idleTimer = nil
	}
}

func (controller *collectionResidencyController) armIdleTimerLocked(
	ctx context.Context,
	collectionName string,
	entry *collectionResidencyEntry,
) {
	controller.cancelIdleTimerLocked(entry)
	if controller.config.idleTimeout <= 0 || controller.config.unload == nil ||
		entry.leases != 0 || entry.pins != 0 ||
		entry.activeTransition != nil || entry.maintenance {
		return
	}
	entry.idleDeadline = controller.config.clock.Now().Add(controller.config.idleTimeout)
	if entry.observations != 0 {
		return
	}
	generation := entry.idleGeneration
	transitionContext := context.WithoutCancel(ctx)
	entry.idleTimer = controller.config.clock.AfterFunc(controller.config.idleTimeout, func() {
		controller.startUnload(transitionContext, collectionName, generation)
	})
}

func (controller *collectionResidencyController) resumeIdleTimerLocked(
	ctx context.Context,
	collectionName string,
	entry *collectionResidencyEntry,
) {
	controller.stopIdleTimerLocked(entry)
	if controller.config.idleTimeout <= 0 || controller.config.unload == nil ||
		entry.idleDeadline.IsZero() {
		return
	}
	delay := max(entry.idleDeadline.Sub(controller.config.clock.Now()), 0)
	generation := entry.idleGeneration
	transitionContext := context.WithoutCancel(ctx)
	entry.idleTimer = controller.config.clock.AfterFunc(delay, func() {
		controller.startUnload(transitionContext, collectionName, generation)
	})
}

func (controller *collectionResidencyController) startUnload(
	ctx context.Context,
	collectionName string,
	generation uint64,
) {
	controller.mutex.Lock()
	entry := controller.entries[collectionName]
	if controller.closed || entry == nil || entry.idleGeneration != generation ||
		entry.state != collectionResidencyReady || entry.leases != 0 ||
		entry.observations != 0 || entry.pins != 0 || entry.load != nil ||
		entry.activeTransition != nil || entry.maintenance {
		controller.mutex.Unlock()
		return
	}
	entry.idleTimer = nil
	unloadCtx, cancelUnload := context.WithCancel(context.WithoutCancel(ctx))
	entry.activeTransition = cancelUnload
	controller.transitions.Add(1)
	controller.mutex.Unlock()

	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				err := fmt.Errorf("collection unload panic: %v", recovered)
				slog.ErrorContext(unloadCtx, "semantic.collection_unload_panic", "err", err)
				controller.finishUnload(entry, err)
			}
		}()
		controller.runUnload(unloadCtx, collectionName, entry)
	}()
}

func (controller *collectionResidencyController) runUnload(
	ctx context.Context,
	collectionName string,
	entry *collectionResidencyEntry,
) {
	defer controller.transitions.Done()
	err := controller.config.unload(ctx, collectionName)

	controller.finishUnload(entry, err)
}

func (controller *collectionResidencyController) finishUnload(
	entry *collectionResidencyEntry,
	err error,
) {
	controller.mutex.Lock()
	defer controller.mutex.Unlock()
	if entry.activeTransition == nil {
		return
	}
	entry.activeTransition = nil
	controller.notifyLocked(entry)
	if err == nil {
		entry.state = collectionResidencyCold
		entry.idleDeadline = time.Time{}
		return
	}
	entry.state = collectionResidencyUnknown
	entry.idleDeadline = time.Time{}
}

func (controller *collectionResidencyController) Close(ctx context.Context) error {
	if controller == nil {
		return nil
	}

	controller.mutex.Lock()
	if !controller.closed {
		controller.closed = true
		close(controller.closedCh)
		for _, entry := range controller.entries {
			controller.cancelIdleTimerLocked(entry)
			if entry.load != nil {
				entry.load.cancel()
			}
			if entry.activeTransition != nil {
				entry.activeTransition()
			}
			controller.notifyLocked(entry)
		}
	}
	controller.mutex.Unlock()

	done := make(chan struct{})
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				err := fmt.Errorf("collection residency shutdown wait panic: %v", recovered)
				slog.ErrorContext(ctx, "collection residency shutdown wait panic", "err", err)
			}
		}()
		defer close(done)
		controller.transitions.Wait()
	}()
	select {
	case <-ctx.Done():
		err := fmt.Errorf("close collection residency controller: %w", ctx.Err())
		slog.WarnContext(ctx, "collection residency shutdown canceled", "err", err)
		return err
	case <-done:
		return nil
	}
}

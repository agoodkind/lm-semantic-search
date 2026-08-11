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
	"goodkind.io/lm-semantic-search/internal/metrics"
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

// CollectionLease protects one collection for the full duration of a search.
type CollectionLease interface {
	Release()
}

// CollectionPin keeps a staging collection resident across a complete build.
type CollectionPin interface {
	Release()
	ReleaseContext(context.Context)
}

// NoopCollectionPin lets a backend without residency management share the API.
type NoopCollectionPin struct{}

// Release completes the no-op pin contract.
func (NoopCollectionPin) Release() {}

// ReleaseContext completes the no-op pin contract with caller context.
func (NoopCollectionPin) ReleaseContext(context.Context) {}

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
	done      chan struct{}
	cancel    context.CancelFunc
	startedAt time.Time
	err       error
}

type collectionResidencyEntry struct {
	name             string
	state            collectionResidencyState
	leases           int
	idleResetPending bool
	observations     int
	pins             int
	load             *collectionResidencyLoad
	idleTimer        residencyTimer
	idleGeneration   uint64
	idleDeadline     time.Time
	activeTransition context.CancelFunc
	maintenance      bool
	reconciliation   uint64
	changed          chan struct{}
}

type collectionResidencyController struct {
	mutex                    sync.Mutex
	config                   residencyControllerConfig
	entries                  map[string]*collectionResidencyEntry
	reconciliationGeneration uint64
	reconciliationActivity   map[string]uint64
	closed                   bool
	closedCh                 chan struct{}
	transitions              sync.WaitGroup
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
	entry      *collectionResidencyEntry
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
		waitTimeout: time.Duration(service.cfg.MilvusCollectionLoadWaitTimeoutMS) * time.Millisecond,
		idleTimeout: time.Duration(service.cfg.MilvusCollectionIdleTimeoutMS) * time.Millisecond,
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

// AcquireCollection makes a collection searchable and protects it until release.
func (service *Service) AcquireCollection(
	ctx context.Context,
	collectionName string,
) (CollectionLease, error) {
	lease, err := service.residency.Acquire(ctx, collectionName)
	if errors.Is(err, ErrCollectionLoadWaitTimeout) {
		wrappedErr := fmt.Errorf(
			"acquire collection %s: %w",
			collectionName,
			ErrCollectionNotReady,
		)
		slog.WarnContext(ctx, "collection load wait expired", "err", wrappedErr)
		return nil, wrappedErr
	}
	return lease, err
}

func (service *Service) acquireResidentCollection(
	ctx context.Context,
	collectionName string,
) (CollectionLease, bool, error) {
	lease, ready, err := service.residency.AcquireResident(ctx, collectionName)
	if lease == nil {
		return nil, ready, err
	}
	return lease, ready, err
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
		mutex:                    sync.Mutex{},
		config:                   config,
		entries:                  make(map[string]*collectionResidencyEntry),
		reconciliationGeneration: 0,
		reconciliationActivity:   make(map[string]uint64),
		closed:                   false,
		closedCh:                 make(chan struct{}),
		transitions:              sync.WaitGroup{},
	}
}

func (controller *collectionResidencyController) beginReconciliation() uint64 {
	controller.mutex.Lock()
	defer controller.mutex.Unlock()
	controller.reconciliationGeneration++
	generation := controller.reconciliationGeneration
	for _, entry := range controller.entries {
		controller.cancelIdleTimerLocked(entry)
		if isRecoveryCollection(entry.name) {
			entry.reconciliation = 0
			continue
		}
		entry.state = collectionResidencyUnknown
		entry.reconciliation = generation
		if controller.entryHasActivityLocked(entry) {
			entry.reconciliation = 0
		}
		controller.notifyLocked(entry)
	}
	controller.updateStateMetricsLocked()
	return generation
}

func (controller *collectionResidencyController) invalidateResidency() {
	controller.mutex.Lock()
	defer controller.mutex.Unlock()
	controller.reconciliationGeneration++
	for _, entry := range controller.entries {
		controller.cancelIdleTimerLocked(entry)
		if isRecoveryCollection(entry.name) {
			entry.reconciliation = 0
			continue
		}
		entry.state = collectionResidencyUnknown
		entry.reconciliation = 0
		controller.notifyLocked(entry)
	}
	controller.updateStateMetricsLocked()
}

func (controller *collectionResidencyController) applyReconciliation(
	ctx context.Context,
	generation uint64,
	collectionName string,
	state collectionResidencyState,
) {
	controller.mutex.Lock()
	defer controller.mutex.Unlock()
	if isRecoveryCollection(collectionName) {
		if entry := controller.entries[collectionName]; entry != nil {
			controller.cancelIdleTimerLocked(entry)
			entry.reconciliation = 0
		}
		return
	}
	if controller.closed || generation != controller.reconciliationGeneration ||
		controller.reconciliationActivity[collectionName] >= generation {
		return
	}
	entry := controller.entries[collectionName]
	if entry == nil {
		entry = controller.entryLocked(collectionName)
		entry.reconciliation = generation
	}
	if entry.reconciliation != generation {
		return
	}
	controller.cancelIdleTimerLocked(entry)
	entry.state = state
	entry.reconciliation = 0
	controller.notifyLocked(entry)
	controller.updateStateMetricsLocked()
	if state == collectionResidencyReady && !isRecoveryCollection(collectionName) {
		controller.armIdleTimerLocked(ctx, collectionName, entry)
	}
}

func (controller *collectionResidencyController) entryHasActivityLocked(
	entry *collectionResidencyEntry,
) bool {
	return entry.leases != 0 || entry.observations != 0 || entry.pins != 0 ||
		entry.load != nil || entry.activeTransition != nil || entry.maintenance
}

func (controller *collectionResidencyController) markReconciliationActivityLocked(
	collectionName string,
	entry *collectionResidencyEntry,
) {
	entry.reconciliation = 0
	controller.reconciliationActivity[collectionName] = controller.reconciliationGeneration
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
		controller.markReconciliationActivityLocked(collectionName, entry)
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
		entry.idleResetPending = true
		metrics.MilvusCollectionLeaseAcquired()
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
		controller.markReconciliationActivityLocked(collectionName, entry)
		if entry.maintenance ||
			(entry.activeTransition != nil && entry.load == nil) {
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
			entry:      entry,
		}, nil
	}
}

func (controller *collectionResidencyController) Rename(oldName string, newName string) {
	controller.mutex.Lock()
	defer controller.mutex.Unlock()
	entry := controller.entries[oldName]
	controller.reconciliationActivity[oldName] = controller.reconciliationGeneration
	controller.reconciliationActivity[newName] = controller.reconciliationGeneration
	if entry == nil || oldName == newName {
		return
	}
	entry.reconciliation = 0
	if replaced := controller.entries[newName]; replaced != nil && replaced != entry {
		replaced.reconciliation = 0
		controller.cancelIdleTimerLocked(replaced)
		controller.notifyLocked(replaced)
	}
	controller.cancelIdleTimerLocked(entry)
	controller.notifyLocked(entry)
	delete(controller.entries, oldName)
	entry.name = newName
	controller.entries[newName] = entry
	controller.updateStateMetricsLocked()
}

func (controller *collectionResidencyController) Forget(collectionName string) {
	controller.mutex.Lock()
	defer controller.mutex.Unlock()
	controller.reconciliationActivity[collectionName] = controller.reconciliationGeneration
	entry := controller.entries[collectionName]
	if entry == nil {
		return
	}
	controller.cancelIdleTimerLocked(entry)
	if entry.leases != 0 || entry.observations != 0 || entry.pins != 0 {
		entry.state = collectionResidencyUnknown
		entry.reconciliation = 0
		controller.notifyLocked(entry)
		controller.updateStateMetricsLocked()
		return
	}
	controller.notifyLocked(entry)
	delete(controller.entries, collectionName)
	controller.updateStateMetricsLocked()
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
			controller.markReconciliationActivityLocked(collectionName, entry)
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
			name:             collectionName,
			state:            collectionResidencyUnknown,
			leases:           0,
			idleResetPending: false,
			observations:     0,
			pins:             0,
			load:             nil,
			idleTimer:        nil,
			idleGeneration:   0,
			idleDeadline:     time.Time{},
			activeTransition: nil,
			maintenance:      false,
			reconciliation:   controller.reconciliationGeneration,
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
		done:      make(chan struct{}),
		cancel:    cancelLoad,
		startedAt: controller.config.clock.Now(),
		err:       nil,
	}
	entry.state = collectionResidencyLoading
	entry.load = flight
	entry.activeTransition = cancelLoad
	metrics.MilvusCollectionLoadStarted()
	controller.updateStateMetricsLocked()
	logCollectionResidencyEvent(
		loadCtx,
		slog.LevelInfo,
		"semantic.collection_load_started",
		collectionName,
		collectionResidencyLoading,
		0,
		0,
		entry.leases,
		0,
		nil,
	)
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
	elapsed := controller.config.clock.Now().Sub(flight.startedAt)
	metrics.MilvusCollectionLoadDone(elapsed, err != nil)
	entry.load = nil
	entry.activeTransition = nil
	if err == nil {
		entry.state = collectionResidencyReady
		logCollectionResidencyEvent(
			ctx,
			slog.LevelInfo,
			"semantic.collection_load_ready",
			collectionName,
			collectionResidencyReady,
			100,
			elapsed,
			entry.leases,
			0,
			nil,
		)
		if entry.leases == 0 {
			controller.armIdleTimerLocked(ctx, collectionName, entry)
		}
	} else {
		entry.state = collectionResidencyUnknown
	}
	controller.updateStateMetricsLocked()
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
		metrics.MilvusCollectionLoadWaitTimedOut()
		controller.mutex.Lock()
		entry := controller.entries[collectionName]
		leaseCount := 0
		elapsed := controller.config.waitTimeout
		if entry != nil {
			leaseCount = entry.leases
			if entry.load == flight {
				elapsed = controller.config.clock.Now().Sub(flight.startedAt)
			}
		}
		controller.mutex.Unlock()
		logCollectionResidencyEvent(
			ctx,
			slog.LevelWarn,
			"semantic.collection_load_wait_timeout",
			collectionName,
			collectionResidencyLoading,
			0,
			elapsed,
			leaseCount,
			0,
			ErrCollectionNotReady,
		)
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
	lease.ReleaseContext(context.Background())
}

func (lease *collectionLease) ReleaseContext(ctx context.Context) {
	if lease == nil {
		return
	}
	lease.once.Do(func() {
		lease.controller.releaseLease(ctx, lease.name)
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
	pin.ReleaseContext(context.Background())
}

func (pin *collectionPin) ReleaseContext(ctx context.Context) {
	if pin == nil {
		return
	}
	pin.once.Do(func() {
		pin.controller.releasePin(ctx, pin.entry)
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
	metrics.MilvusCollectionLeaseReleased()
	controller.notifyLocked(entry)
	if entry.leases != 0 {
		return
	}
	resetsIdle := entry.idleResetPending
	entry.idleResetPending = false
	if entry.state != collectionResidencyReady {
		return
	}
	if resetsIdle {
		controller.armIdleTimerLocked(ctx, collectionName, entry)
		return
	}
	controller.resumeIdleTimerLocked(ctx, collectionName, entry)
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
	entry *collectionResidencyEntry,
) {
	controller.mutex.Lock()
	defer controller.mutex.Unlock()
	if entry == nil || entry.pins == 0 {
		return
	}
	entry.pins--
	controller.notifyLocked(entry)
	if entry.pins == 0 && entry.state == collectionResidencyReady {
		controller.armIdleTimerLocked(ctx, entry.name, entry)
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

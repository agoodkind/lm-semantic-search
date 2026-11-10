package semantic

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"goodkind.io/lm-semantic-search/internal/metrics"
)

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
	if isRecoveryCollection(collectionName) || controller.config.idleTimeout <= 0 ||
		controller.config.unload == nil ||
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
	if isRecoveryCollection(collectionName) {
		controller.cancelIdleTimerLocked(entry)
		return
	}
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
	if controller.closed || isRecoveryCollection(collectionName) || entry == nil {
		controller.mutex.Unlock()
		return
	}
	if entry.leases != 0 || entry.observations != 0 || entry.pins != 0 || entry.maintenance {
		metrics.MilvusCollectionUnloadSkippedInUse()
		controller.mutex.Unlock()
		return
	}
	if entry.idleGeneration != generation ||
		entry.state != collectionResidencyReady || entry.load != nil ||
		entry.activeTransition != nil {
		controller.mutex.Unlock()
		return
	}
	idle := controller.config.clock.Now().Sub(
		entry.idleDeadline.Add(-controller.config.idleTimeout),
	)
	entry.idleTimer = nil
	unloadCtx, cancelUnload := context.WithCancel(context.WithoutCancel(ctx))
	entry.activeTransition = cancelUnload
	metrics.MilvusCollectionUnloadStarted()
	logCollectionResidencyEvent(
		unloadCtx,
		slog.LevelInfo,
		"semantic.collection_unload_started",
		collectionName,
		collectionResidencyReady,
		0,
		0,
		entry.leases,
		idle,
		nil,
	)
	controller.transitions.Add(1)
	controller.mutex.Unlock()

	startedAt := controller.config.clock.Now()
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				err := fmt.Errorf("collection unload panic: %v", recovered)
				slog.ErrorContext(unloadCtx, "semantic.collection_unload_panic", "err", err)
				controller.finishUnload(
					unloadCtx,
					collectionName,
					entry,
					startedAt,
					idle,
					err,
				)
			}
		}()
		controller.runUnload(unloadCtx, collectionName, entry, startedAt, idle)
	}()
}

func (controller *collectionResidencyController) runUnload(
	ctx context.Context,
	collectionName string,
	entry *collectionResidencyEntry,
	startedAt time.Time,
	idle time.Duration,
) {
	defer controller.transitions.Done()
	err := controller.config.unload(ctx, collectionName)
	controller.finishUnload(ctx, collectionName, entry, startedAt, idle, err)
}

func (controller *collectionResidencyController) finishUnload(
	ctx context.Context,
	collectionName string,
	entry *collectionResidencyEntry,
	startedAt time.Time,
	idle time.Duration,
	err error,
) {
	controller.mutex.Lock()
	defer controller.mutex.Unlock()
	if entry.activeTransition == nil {
		return
	}
	entry.activeTransition = nil
	elapsed := controller.config.clock.Now().Sub(startedAt)
	metrics.MilvusCollectionUnloadDone(elapsed, err != nil)
	controller.notifyLocked(entry)
	if err == nil {
		entry.state = collectionResidencyCold
		entry.idleDeadline = time.Time{}
		controller.updateStateMetricsLocked()
		logCollectionResidencyEvent(
			ctx,
			slog.LevelInfo,
			"semantic.collection_unloaded",
			collectionName,
			collectionResidencyCold,
			100,
			elapsed,
			entry.leases,
			idle,
			nil,
		)
		return
	}
	entry.state = collectionResidencyUnknown
	entry.idleDeadline = time.Time{}
	controller.updateStateMetricsLocked()
	logCollectionResidencyEvent(
		ctx,
		slog.LevelWarn,
		"semantic.collection_unload_failed",
		collectionName,
		collectionResidencyUnknown,
		0,
		elapsed,
		entry.leases,
		idle,
		err,
	)
}

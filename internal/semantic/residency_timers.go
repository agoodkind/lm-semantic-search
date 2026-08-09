package semantic

import (
	"context"
	"fmt"
	"log/slog"
	"time"
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
	if controller.closed || isRecoveryCollection(collectionName) || entry == nil ||
		entry.idleGeneration != generation ||
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

package semantic

import (
	"context"
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/metrics"
)

func (controller *collectionResidencyController) AcquireResident(
	ctx context.Context,
	collectionName string,
) (*collectionLease, bool, error) {
	for {
		if err := ctx.Err(); err != nil {
			wrappedErr := fmt.Errorf("acquire resident collection: %w", err)
			slog.WarnContext(ctx, "resident collection acquire canceled", "err", wrappedErr)
			return nil, false, wrappedErr
		}
		controller.mutex.Lock()
		if controller.closed {
			controller.mutex.Unlock()
			return nil, false, ErrResidencyControllerClosed
		}
		entry := controller.entryLocked(collectionName)
		if entry.maintenance ||
			(entry.activeTransition != nil && entry.load == nil) {
			changed := entry.changed
			controller.mutex.Unlock()
			if err := controller.waitForChange(ctx, changed); err != nil {
				return nil, false, err
			}
			continue
		}
		if entry.state != collectionResidencyReady {
			controller.mutex.Unlock()
			return nil, false, nil
		}
		controller.markReconciliationActivityLocked(collectionName, entry)
		entry.leases++
		metrics.MilvusCollectionLeaseAcquired()
		controller.pauseIdleTimerLocked(entry)
		controller.mutex.Unlock()
		return controller.newLease(collectionName), true, nil
	}
}

package semantic

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"sync"
	"time"
)

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

func (controller *collectionResidencyController) newLease(
	collectionName string,
) *collectionLease {
	return &collectionLease{
		once:       sync.Once{},
		controller: controller,
		name:       collectionName,
	}
}

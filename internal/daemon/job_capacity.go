package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"
)

const (
	// defaultJobCapacityReacquireTimeout gives a read-side waiter five seconds
	// to resume after its already-bounded collection load. That is long enough
	// for scheduler jitter and ten advisory-lock retries, while short enough
	// that a replacement indexing job cannot turn the bounded load outcome into
	// another job-length wait.
	defaultJobCapacityReacquireTimeout = 5 * time.Second
)

type jobCapacityContextKey struct{}

// jobCapacityReacquireError is the terminal outcome when a job cannot regain
// the scarce holds it released around a read. Its cause distinguishes the
// resume deadline from cancellation of the job itself.
type jobCapacityReacquireError struct {
	Timeout time.Duration
	Cause   error
}

func (err *jobCapacityReacquireError) Error() string {
	return fmt.Sprintf(
		"reacquire indexing capacity within %s after read: %v",
		err.Timeout,
		err.Cause,
	)
}

func (err *jobCapacityReacquireError) Unwrap() error {
	return err.Cause
}

type jobCapacity struct {
	manager      *Manager
	slotHeld     bool
	syncLockHeld bool
}

func withJobCapacity(ctx context.Context, capacity *jobCapacity) context.Context {
	return context.WithValue(ctx, jobCapacityContextKey{}, capacity)
}

func jobCapacityFromContext(ctx context.Context) *jobCapacity {
	capacity, _ := ctx.Value(jobCapacityContextKey{}).(*jobCapacity)
	return capacity
}

func (capacity *jobCapacity) acquireSyncLock(ctx context.Context) bool {
	if capacity.syncLockHeld {
		return true
	}
	if !capacity.manager.syncLock.acquireBlocking(ctx) {
		return false
	}
	capacity.syncLockHeld = true
	return true
}

func (capacity *jobCapacity) acquire(ctx context.Context, holdSyncLock bool) bool {
	if !capacity.slotHeld {
		select {
		case capacity.manager.indexSlots <- struct{}{}:
			capacity.slotHeld = true
		case <-ctx.Done():
			return false
		}
	}
	if holdSyncLock && !capacity.acquireSyncLock(ctx) {
		capacity.release(ctx)
		return false
	}
	return true
}

func (capacity *jobCapacity) release(ctx context.Context) {
	if capacity.syncLockHeld {
		capacity.manager.syncLock.release(ctx)
		capacity.syncLockHeld = false
	}
	if capacity.slotHeld {
		<-capacity.manager.indexSlots
		capacity.slotHeld = false
	}
}

func (manager *Manager) runWithoutJobCapacity(
	ctx context.Context,
	operation func() error,
) error {
	capacity := jobCapacityFromContext(ctx)
	if capacity == nil {
		return operation()
	}

	reacquireSyncLock := capacity.syncLockHeld
	capacity.release(ctx)
	operationErr := operation()
	reacquireCtx, cancel := context.WithTimeout(ctx, manager.jobCapacityReacquireTimeout)
	reacquired := capacity.acquire(reacquireCtx, reacquireSyncLock)
	reacquireCause := reacquireCtx.Err()
	cancel()
	if reacquired {
		return operationErr
	}

	if operationErr != nil {
		slog.ErrorContext(ctx, "read failed before indexing capacity reacquire",
			"component", "daemon",
			"subcomponent", "capacity",
			"err", operationErr,
		)
	}
	if reacquireCause == nil {
		reacquireCause = errors.New("indexing capacity unavailable")
	}
	reacquireErr := &jobCapacityReacquireError{
		Timeout: manager.jobCapacityReacquireTimeout,
		Cause:   reacquireCause,
	}
	slog.ErrorContext(ctx, "reacquire indexing capacity after read failed",
		"component", "daemon",
		"subcomponent", "capacity",
		"err", reacquireErr,
	)
	return reacquireErr
}

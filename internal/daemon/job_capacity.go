package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"
)

const (
	// defaultJobCapacityReacquireTimeout gives a read-side waiter five seconds
	// to resume after its already-bounded collection load. That is long enough
	// for scheduler jitter and ten advisory-lock retries, while short enough
	// that a replacement indexing job cannot turn the bounded load outcome into
	// another job-length wait.
	defaultJobCapacityReacquireTimeout = 5 * time.Second

	// defaultJobCapacityReleaseGrace is how long a read may run before the job
	// gives up its indexing slot and the shared sync lock for the rest of it.
	// Whether a read will stall cannot be known before it runs, so this decides
	// by watching: a healthy reuse read answers in milliseconds and keeps both
	// holds, while a read parked behind a collection load that Milvus never
	// finishes crosses the grace and frees them for the minutes it may still
	// take. It equals the resume bound on purpose, so a job never gives up holds
	// for a read shorter than the time it is willing to wait to get them back.
	defaultJobCapacityReleaseGrace = defaultJobCapacityReacquireTimeout
)

type jobCapacityContextKey struct{}

// jobCapacityTimings bounds the two waits around a read that gives up this
// job's scarce holds: how long the read may run before they are released, and
// how long the job may then wait to take them back.
type jobCapacityTimings struct {
	ReleaseGrace time.Duration
	Reacquire    time.Duration
}

func defaultJobCapacityTimings() jobCapacityTimings {
	return jobCapacityTimings{
		ReleaseGrace: defaultJobCapacityReleaseGrace,
		Reacquire:    defaultJobCapacityReacquireTimeout,
	}
}

// jobCapacityReacquireError is the terminal outcome when a job cannot regain
// the scarce holds it released around a stalled read. Its cause distinguishes
// the resume deadline from cancellation of the job itself.
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

// jobCapacity is one job's hold on the two scarce resources an indexing job
// needs: a slot in the daemon-wide concurrency cap and a reference on the
// shared advisory sync lock. The mutex guards the two held flags because the
// stall watchdog releases them from its own goroutine while the job's goroutine
// is inside a read.
type jobCapacity struct {
	manager      *Manager
	mu           sync.Mutex
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
	capacity.mu.Lock()
	defer capacity.mu.Unlock()
	return capacity.acquireSyncLockLocked(ctx)
}

func (capacity *jobCapacity) acquireSyncLockLocked(ctx context.Context) bool {
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
	capacity.mu.Lock()
	defer capacity.mu.Unlock()
	if !capacity.slotHeld {
		select {
		case capacity.manager.indexSlots <- struct{}{}:
			capacity.slotHeld = true
		case <-ctx.Done():
			return false
		}
	}
	if holdSyncLock && !capacity.acquireSyncLockLocked(ctx) {
		capacity.releaseLocked(ctx)
		return false
	}
	return true
}

func (capacity *jobCapacity) release(ctx context.Context) {
	capacity.mu.Lock()
	defer capacity.mu.Unlock()
	capacity.releaseLocked(ctx)
}

// releaseHeld gives up whatever this job currently holds and reports whether
// the sync lock was among them, so a later resume takes back exactly what was
// given up rather than guessing.
func (capacity *jobCapacity) releaseHeld(ctx context.Context) bool {
	capacity.mu.Lock()
	defer capacity.mu.Unlock()
	heldSyncLock := capacity.syncLockHeld
	capacity.releaseLocked(ctx)
	return heldSyncLock
}

func (capacity *jobCapacity) releaseLocked(ctx context.Context) {
	if capacity.syncLockHeld {
		capacity.manager.syncLock.release(ctx)
		capacity.syncLockHeld = false
	}
	if capacity.slotHeld {
		<-capacity.manager.indexSlots
		capacity.slotHeld = false
	}
}

// stallRelease is the watchdog that frees this job's holds once a read has run
// longer than the release grace. releasedSyncLock is written by the watchdog
// goroutine and read only after join returns, so the channel close orders the
// two accesses.
type stallRelease struct {
	stopped          chan struct{}
	finished         chan struct{}
	released         bool
	releasedSyncLock bool
}

// startStallRelease arms the watchdog. It touches the capacity only after the
// grace elapses, which is a window in which the job's own goroutine is inside
// the read and touches nothing.
func startStallRelease(
	ctx context.Context,
	capacity *jobCapacity,
	grace time.Duration,
) *stallRelease {
	watchdog := &stallRelease{
		stopped:          make(chan struct{}),
		finished:         make(chan struct{}),
		released:         false,
		releasedSyncLock: false,
	}
	go func() {
		defer close(watchdog.finished)
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.ErrorContext(ctx, "indexing capacity watchdog panic",
					"component", "daemon",
					"subcomponent", "capacity",
					"err", fmt.Errorf("panic: %v", recovered),
				)
			}
		}()
		graceTimer := time.NewTimer(grace)
		defer graceTimer.Stop()
		select {
		case <-watchdog.stopped:
			return
		case <-graceTimer.C:
		}
		slog.WarnContext(ctx, "releasing indexing capacity for a stalled read",
			"component", "daemon",
			"subcomponent", "capacity",
			"grace_ms", grace.Milliseconds(),
		)
		watchdog.releasedSyncLock = capacity.releaseHeld(ctx)
		watchdog.released = true
	}()
	return watchdog
}

// join stops the watchdog and reports what it released, if anything. It returns
// only once the watchdog goroutine has finished, so the caller owns the
// capacity again from here on.
func (watchdog *stallRelease) join() (released bool, releasedSyncLock bool) {
	close(watchdog.stopped)
	<-watchdog.finished
	return watchdog.released, watchdog.releasedSyncLock
}

// runReleasingCapacityIfStalled runs one store read, giving up this job's
// indexing slot and its reference on the shared sync lock only for as long as
// that read actually stalls.
//
// Whether a read will stall is not knowable before it runs, so the decision is
// made by watching it. The read starts while the job still holds everything,
// and a watchdog releases the holds only once the read has failed to finish
// within the release grace. A healthy reuse read therefore keeps its place and
// pays nothing, while a read parked behind a collection load Milvus never
// finishes stops starving the jobs queued behind it.
//
// Once the holds are gone the job must take them back to keep writing, and that
// resume is bounded: a replacement job or the external adapter may already have
// taken what this job gave up, and waiting for it without a bound would restore
// the job-length stall the release exists to prevent. Failing to resume is a
// terminal, typed outcome rather than a silent continuation.
func (manager *Manager) runReleasingCapacityIfStalled(
	ctx context.Context,
	operation func() error,
) error {
	capacity := jobCapacityFromContext(ctx)
	if capacity == nil {
		return operation()
	}

	watchdog := startStallRelease(ctx, capacity, manager.jobCapacityTimings.ReleaseGrace)
	operationErr := operation()
	released, reacquireSyncLock := watchdog.join()
	if !released {
		return operationErr
	}

	reacquireCtx, cancel := context.WithTimeout(ctx, manager.jobCapacityTimings.Reacquire)
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
		Timeout: manager.jobCapacityTimings.Reacquire,
		Cause:   reacquireCause,
	}
	slog.ErrorContext(ctx, "reacquire indexing capacity after read failed",
		"component", "daemon",
		"subcomponent", "capacity",
		"err", reacquireErr,
	)
	return reacquireErr
}

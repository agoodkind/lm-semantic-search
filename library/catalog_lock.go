package library

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"syscall"
	"time"
)

// writerLockRetryInterval is the wait between attempts to take a kernel lock
// that another process owns.
const writerLockRetryInterval = 50 * time.Millisecond

// writerLock serializes every catalog writer that shares one lock path. The
// mutex serializes goroutines of this process. The kernel flock serializes
// processes, and the kernel releases it when the owning process exits. No
// lease expires while an owner is alive.
type writerLock struct {
	mutex sync.Mutex
	file  *os.File
	path  string
}

// openWriterLock opens or creates the lock file without locking it.
func openWriterLock(ctx context.Context, path string) (*writerLock, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		slog.ErrorContext(ctx, "open catalog writer lock failed", "path", path, "err", err)
		return nil, fmt.Errorf("open catalog writer lock %s: %w", path, err)
	}
	return &writerLock{mutex: sync.Mutex{}, file: file, path: path}, nil
}

// acquire takes the process mutex and then the kernel lock. It waits until the
// lock is free or ctx ends. A context deadline returns an error that wraps
// [ErrDeadline]. The caller runs the returned release exactly once.
func (lock *writerLock) acquire(ctx context.Context) (func(), error) {
	lock.mutex.Lock()
	for {
		err := syscall.Flock(int(lock.file.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
		if err == nil {
			return lock.release, nil
		}
		if !errors.Is(err, syscall.EWOULDBLOCK) {
			lock.mutex.Unlock()
			slog.ErrorContext(ctx, "take catalog writer lock failed", "path", lock.path, "err", err)
			return nil, fmt.Errorf("take catalog writer lock %s: %w", lock.path, err)
		}
		timer := time.NewTimer(writerLockRetryInterval)
		select {
		case <-ctx.Done():
			timer.Stop()
			lock.mutex.Unlock()
			return nil, contextFailure(ctx, "wait for catalog writer lock")
		case <-timer.C:
		}
	}
}

func (lock *writerLock) release() {
	if err := syscall.Flock(int(lock.file.Fd()), syscall.LOCK_UN); err != nil {
		slog.Error("release catalog writer lock failed", "path", lock.path, "err", err)
	}
	lock.mutex.Unlock()
}

// close closes the lock file descriptor. The kernel releases a lock on a
// closed descriptor.
func (lock *writerLock) close() error {
	if err := lock.file.Close(); err != nil {
		slog.Error("close catalog writer lock failed", "path", lock.path, "err", err)
		return fmt.Errorf("close catalog writer lock %s: %w", lock.path, err)
	}
	return nil
}

// contextFailure returns the error for an operation that ctx ended. A deadline
// wraps [ErrDeadline]. A cancellation wraps [context.Canceled].
func contextFailure(ctx context.Context, operation string) error {
	cause := ctx.Err()
	var err error
	if errors.Is(cause, context.DeadlineExceeded) {
		err = fmt.Errorf("%w: %s: %w", ErrDeadline, operation, cause)
	} else {
		err = fmt.Errorf("%s: %w", operation, cause)
	}
	slog.WarnContext(ctx, "library operation ended with its context", "operation", operation, "err", err)
	return err
}

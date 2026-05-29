package daemon

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"sync"
	"time"

	"goodkind.io/claude-context-go/internal/clock"
	"goodkind.io/claude-context-go/internal/store"
)

const (
	defaultSyncLockStale  = 10 * time.Minute
	syncLockRetryInterval = 500 * time.Millisecond
)

// syncLock is a process-wide refcounted hold of the advisory lock that
// coordinates embedding with the upstream TS adapter. The lock is the directory
// at lockPath (~/.context/mcp-sync.lock by default). The first daemon embed to
// acquire creates the directory; the last to release removes it. While the
// daemon holds it, many daemon embeds proceed under the single hold and the
// index-slot semaphore bounds real concurrency; the external TS tool, which
// takes the same lock for a whole sync, backs off while it is present.
//
// A daemon embed that finds the lock already held by the daemon's own refcount
// proceeds. One that finds the directory present with a zero refcount treats it
// as held by the external tool and defers, unless the directory is older than
// the stale window, in which case it reclaims a lock a crashed holder left
// behind.
type syncLock struct {
	lockPath string
	rootDir  string
	staleMS  int

	mu       sync.Mutex
	refcount int
}

// newSyncLock builds a refcounted hold of the lock directory at lockPath.
// rootDir is created before the lock so the first acquire never fails on a
// missing parent. staleMS bounds how long a zero-refcount directory is honored
// before it is treated as abandoned; a non-positive value uses the default.
func newSyncLock(lockPath string, rootDir string, staleMS int) *syncLock {
	return &syncLock{
		lockPath: lockPath,
		rootDir:  rootDir,
		staleMS:  staleMS,
		mu:       sync.Mutex{},
		refcount: 0,
	}
}

// acquire takes one reference to the lock and reports whether the caller may
// embed. A true result means the caller must pair it with exactly one release.
// A false result means the external tool holds the lock, so the caller defers.
func (lock *syncLock) acquire(ctx context.Context) bool {
	lock.mu.Lock()
	defer lock.mu.Unlock()

	if lock.refcount > 0 {
		lock.refcount++
		return true
	}

	if err := store.EnsureDir(lock.rootDir); err != nil {
		slog.ErrorContext(ctx, "ensure sync lock root failed", "path", lock.rootDir, "err", err)
		return false
	}

	if err := os.Mkdir(lock.lockPath, 0o755); err == nil {
		lock.refcount = 1
		return true
	} else if !errors.Is(err, os.ErrExist) {
		slog.ErrorContext(ctx, "acquire sync lock failed", "path", lock.lockPath, "err", err)
		return false
	}

	// The directory exists with no local refcount: either the external TS tool
	// holds it, or a crashed holder left it behind. Reclaim only once it is
	// older than the stale window.
	if !lock.reclaimStaleLocked(ctx) {
		return false
	}
	lock.refcount = 1
	return true
}

// acquireBlocking waits until it can take a reference or ctx is cancelled. User
// index jobs use it because they must complete their embed rather than drop the
// work the way a background converge does; the wait ends when the external tool
// releases the lock or its stale window elapses. It returns false only when ctx
// is cancelled before the lock could be taken, and a true result must be paired
// with exactly one release.
func (lock *syncLock) acquireBlocking(ctx context.Context) bool {
	for {
		if lock.acquire(ctx) {
			return true
		}
		select {
		case <-ctx.Done():
			return false
		case <-time.After(syncLockRetryInterval):
		}
	}
}

// release drops one reference and removes the lock directory when the last
// reference is gone.
func (lock *syncLock) release(ctx context.Context) {
	lock.mu.Lock()
	defer lock.mu.Unlock()

	if lock.refcount == 0 {
		return
	}
	lock.refcount--
	if lock.refcount > 0 {
		return
	}
	if err := os.RemoveAll(lock.lockPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		slog.ErrorContext(ctx, "release sync lock failed", "path", lock.lockPath, "err", err)
	}
}

// reclaimStaleLocked removes and recreates the lock directory when it is older
// than the stale window, so a crashed holder cannot block embedding forever. It
// returns false when the directory is fresh enough to still belong to a live
// external holder. The caller holds lock.mu.
func (lock *syncLock) reclaimStaleLocked(ctx context.Context) bool {
	info, err := os.Stat(lock.lockPath)
	if err != nil {
		slog.ErrorContext(ctx, "inspect sync lock failed", "path", lock.lockPath, "err", err)
		return false
	}
	staleAge := defaultSyncLockStale
	if lock.staleMS > 0 {
		staleAge = time.Duration(lock.staleMS) * time.Millisecond
	}
	if clock.Now().Sub(info.ModTime()) <= staleAge {
		return false
	}
	if err := os.RemoveAll(lock.lockPath); err != nil {
		slog.ErrorContext(ctx, "remove stale sync lock failed", "path", lock.lockPath, "err", err)
		return false
	}
	if err := os.Mkdir(lock.lockPath, 0o755); err != nil {
		slog.ErrorContext(ctx, "reacquire sync lock failed", "path", lock.lockPath, "err", err)
		return false
	}
	return true
}

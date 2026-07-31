package daemon

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

const (
	syncLockHolderModeEnv = "LMS_TEST_SYNC_LOCK_HOLDER"
	syncLockPathEnv       = "LMS_TEST_SYNC_LOCK_PATH"
)

type syncLockAcquireResult struct {
	lease   syncLockLease
	outcome syncLockOutcome
	err     error
}

func newTestSyncLock(t *testing.T) (*syncLock, string, string) {
	t.Helper()
	root := t.TempDir()
	lockPath := filepath.Join(root, "mcp-sync.flock")
	return newSyncLock(lockPath, root), lockPath, root
}

// probeLockHeld opens an independent descriptor and reports whether another
// descriptor holds the kernel lock. It fails the test on a probe error, so it
// must be called from the test goroutine.
func probeLockHeld(t *testing.T, lockPath string) bool {
	t.Helper()
	held, err := lockHeldByAnother(lockPath)
	if err != nil {
		t.Fatalf("probe sync lock returned error: %v", err)
	}
	return held
}

// lockHeldByAnother opens an independent descriptor and reports whether some
// other descriptor holds the kernel lock at lockPath. It returns its error
// rather than failing the test, so a goroutine other than the test's own can
// probe the lock and report through a channel.
func lockHeldByAnother(lockPath string) (bool, error) {
	probe, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return false, fmt.Errorf("open sync lock probe %s: %w", lockPath, err)
	}
	defer func() {
		_ = probe.Close()
	}()

	err = syscall.Flock(int(probe.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
	if errors.Is(err, syscall.EWOULDBLOCK) {
		return true, nil
	}
	if err != nil {
		return false, fmt.Errorf("lock sync lock probe %s: %w", lockPath, err)
	}
	if unlockErr := syscall.Flock(int(probe.Fd()), syscall.LOCK_UN); unlockErr != nil {
		return false, fmt.Errorf("unlock sync lock probe %s: %w", lockPath, unlockErr)
	}
	return false, nil
}

// mustAcquire takes one reference and fails the test when the lock refuses it.
func mustAcquire(t *testing.T, lock *syncLock) syncLockLease {
	t.Helper()
	lease, outcome, err := lock.acquire(context.Background())
	if outcome != syncLockAcquired || err != nil {
		t.Fatalf("acquire returned (%s, %v), want acquired with no error", outcome, err)
	}
	return lease
}

func TestSyncLockHolderProcess(t *testing.T) {
	if os.Getenv(syncLockHolderModeEnv) != "1" {
		return
	}

	lockPath := os.Getenv(syncLockPathEnv)
	holder, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		t.Fatalf("OpenFile holder returned error: %v", err)
	}
	defer func() {
		_ = holder.Close()
	}()
	if err := syscall.Flock(
		int(holder.Fd()),
		syscall.LOCK_EX|syscall.LOCK_NB,
	); err != nil {
		t.Fatalf("Flock holder returned error: %v", err)
	}
	if _, err := fmt.Fprintln(os.Stdout, "locked"); err != nil {
		t.Fatalf("write readiness returned error: %v", err)
	}
	if _, err := io.Copy(io.Discard, os.Stdin); err != nil {
		t.Fatalf("wait for parent returned error: %v", err)
	}
}

// TestSyncLockReleasedWhenHolderProcessDies proves the kernel releases the lock
// when its owner exits without an orderly unlock.
func TestSyncLockReleasedWhenHolderProcessDies(t *testing.T) {
	lock, lockPath, _ := newTestSyncLock(t)
	command := exec.Command(os.Args[0], "-test.run=^TestSyncLockHolderProcess$")
	command.Env = append(
		os.Environ(),
		syncLockHolderModeEnv+"=1",
		syncLockPathEnv+"="+lockPath,
	)
	stdin, err := command.StdinPipe()
	if err != nil {
		t.Fatalf("StdinPipe returned error: %v", err)
	}
	stdout, err := command.StdoutPipe()
	if err != nil {
		t.Fatalf("StdoutPipe returned error: %v", err)
	}
	command.Stderr = os.Stderr
	if err := command.Start(); err != nil {
		t.Fatalf("Start holder returned error: %v", err)
	}
	processFinished := false
	t.Cleanup(func() {
		_ = stdin.Close()
		if !processFinished {
			_ = command.Process.Kill()
			_ = command.Wait()
		}
	})

	ready, err := bufio.NewReader(stdout).ReadString('\n')
	if err != nil {
		t.Fatalf("read holder readiness returned error: %v", err)
	}
	if ready != "locked\n" {
		t.Fatalf("holder readiness = %q, want %q", ready, "locked\n")
	}

	if _, outcome, err := lock.acquire(context.Background()); outcome != syncLockBusy ||
		err != nil {
		t.Fatalf("acquire returned (%s, %v), want busy while child holds the lock", outcome, err)
	}

	if err := command.Process.Kill(); err != nil {
		t.Fatalf("Kill holder returned error: %v", err)
	}
	if err := command.Wait(); err == nil {
		t.Fatal("Wait returned no error after the holder was killed")
	}
	processFinished = true

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	lease, outcome, err := lock.acquireBlocking(ctx)
	if outcome != syncLockAcquired || err != nil {
		t.Fatalf("acquireBlocking returned (%s, %v), want acquired after holder death", outcome, err)
	}
	lease.release(ctx)
}

func TestSyncLockExcludesTwoLiveHolders(t *testing.T) {
	first, lockPath, root := newTestSyncLock(t)
	second := newSyncLock(lockPath, root)
	ctx := context.Background()

	firstLease := mustAcquire(t, first)
	if _, outcome, err := second.acquire(ctx); outcome != syncLockBusy || err != nil {
		t.Fatalf("second acquire returned (%s, %v), want busy", outcome, err)
	}

	firstLease.release(ctx)
	secondLease := mustAcquire(t, second)
	secondLease.release(ctx)
}

func TestSyncLockAcquireIsRefcounted(t *testing.T) {
	lock, lockPath, _ := newTestSyncLock(t)
	ctx := context.Background()

	first := mustAcquire(t, lock)
	second := mustAcquire(t, lock)

	first.release(ctx)
	if !probeLockHeld(t, lockPath) {
		t.Fatal("kernel lock should stay held after the first release")
	}

	second.release(ctx)
	if probeLockHeld(t, lockPath) {
		t.Fatal("kernel lock should be released after the final release")
	}
}

// TestSyncLockStaleReleaseKeepsLaterHolder proves a duplicate release from an
// old lease cannot unlock a later acquisition.
func TestSyncLockStaleReleaseKeepsLaterHolder(t *testing.T) {
	lock, lockPath, _ := newTestSyncLock(t)
	ctx := context.Background()

	first := mustAcquire(t, lock)
	first.release(ctx)
	if probeLockHeld(t, lockPath) {
		t.Fatal("kernel lock should be free after the first holder released")
	}

	second := mustAcquire(t, lock)
	first.release(ctx)

	if !probeLockHeld(t, lockPath) {
		t.Fatal("a stale duplicate release unlocked the later holder")
	}
	if lock.refcount != 1 {
		t.Fatalf("lock refcount = %d, want 1", lock.refcount)
	}

	second.release(ctx)
}

// TestSyncLockAcquireBlockingCancels proves cancellation interrupts a wait
// behind a live holder instead of waiting for the retry interval.
func TestSyncLockAcquireBlockingCancels(t *testing.T) {
	holder, lockPath, root := newTestSyncLock(t)
	waiter := newSyncLock(lockPath, root)
	holderLease := mustAcquire(t, holder)
	t.Cleanup(func() {
		holderLease.release(context.Background())
	})

	if _, outcome, err := waiter.acquire(context.Background()); outcome != syncLockBusy ||
		err != nil {
		t.Fatalf("waiter acquire returned (%s, %v), want busy", outcome, err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	started := make(chan struct{})
	result := make(chan syncLockAcquireResult, 1)
	go func() {
		close(started)
		lease, outcome, err := waiter.acquireBlocking(ctx)
		result <- syncLockAcquireResult{lease: lease, outcome: outcome, err: err}
	}()
	<-started
	cancel()

	select {
	case got := <-result:
		if got.outcome != syncLockCancelled ||
			!errors.Is(got.err, errSyncLockWaitCancelled) {
			t.Fatalf(
				"acquireBlocking returned (%s, %v), want cancelled",
				got.outcome,
				got.err,
			)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("acquireBlocking did not return after cancellation")
	}
}

// TestSyncLockAcquireBlockingFailsFastOnPermanentError proves a lock path that
// cannot be opened as a file returns an error before the first contention retry.
func TestSyncLockAcquireBlockingFailsFastOnPermanentError(t *testing.T) {
	root := t.TempDir()
	lockPath := filepath.Join(root, "mcp-sync.flock")
	if err := os.Mkdir(lockPath, 0o755); err != nil {
		t.Fatalf("Mkdir lock obstruction returned error: %v", err)
	}
	lock := newSyncLock(lockPath, root)

	ctx, cancel := context.WithTimeout(
		context.Background(),
		10*syncLockRetryInterval,
	)
	defer cancel()
	started := time.Now()
	lease, outcome, err := lock.acquireBlocking(ctx)
	elapsed := time.Since(started)

	if outcome != syncLockFailed || err == nil {
		t.Fatalf("acquireBlocking returned (%s, %v), want failed with an error", outcome, err)
	}
	if lease.lock != nil {
		t.Fatal("failed acquire must not return a lease")
	}
	if elapsed >= syncLockRetryInterval {
		t.Fatalf(
			"acquireBlocking took %v, want less than %v",
			elapsed,
			syncLockRetryInterval,
		)
	}
}

func TestSyncLockAcquireBlockingCancelsWhenFree(t *testing.T) {
	lock, lockPath, _ := newTestSyncLock(t)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, outcome, err := lock.acquireBlocking(ctx); outcome != syncLockCancelled ||
		!errors.Is(err, errSyncLockWaitCancelled) {
		t.Fatalf("acquireBlocking returned (%s, %v), want cancelled", outcome, err)
	}
	if lock.refcount != 0 {
		t.Fatalf("lock refcount = %d, want 0", lock.refcount)
	}
	if lock.file != nil {
		t.Fatal("lock file should remain nil")
	}
	if _, err := os.Stat(lockPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("Stat returned error %v, want os.ErrNotExist", err)
	}
}

// TestBeginConvergeSerializesPerCodebase proves the per-codebase guard admits
// one converge per codebase at a time while letting distinct codebases proceed
// concurrently.
func TestBeginConvergeSerializesPerCodebase(t *testing.T) {
	syncer := &BackgroundSync{converging: make(map[string]time.Time)}

	if !syncer.beginConverge("cb1") {
		t.Fatal("first beginConverge for cb1 should succeed")
	}
	if syncer.beginConverge("cb1") {
		t.Fatal("second beginConverge for cb1 must fail while the first runs")
	}
	if !syncer.beginConverge("cb2") {
		t.Fatal("beginConverge for a distinct codebase should succeed concurrently")
	}

	syncer.endConverge("cb1")
	if !syncer.beginConverge("cb1") {
		t.Fatal("beginConverge for cb1 should succeed again after endConverge")
	}
}

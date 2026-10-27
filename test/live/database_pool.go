//go:build live

package live

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"syscall"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v3/milvusclient"
)

// liveDatabasePoolPrefix defines names for pool databases that tests never drop.
// Milvus can drop a database before collection cleanup finishes. After a
// restart, Milvus skips collections in dropped databases and leaves their
// files in object storage.
const (
	liveDatabasePoolPrefix  = "lms_live_pool_"
	liveDatabasePoolSize    = 8
	liveDatabasePoolWait    = 10 * time.Minute
	liveDatabasePoolPoll    = 2 * time.Second
	liveDatabasePoolDirName = "live-database-pool"
	liveDatabasePoolAppName = "lm-semantic-search"
	liveDatabaseLockMode    = 0o600
	liveDatabaseDirMode     = 0o700
)

// liveDatabaseLease pairs a pool database name with its exclusive file lock.
type liveDatabaseLease struct {
	name     string
	lockFile *os.File
}

// acquireLiveDatabase waits up to liveDatabasePoolWait for a free database.
// It polls every liveDatabasePoolPoll and registers release with t.Cleanup
// before the caller registers teardown. Cleanup releases the lease after
// teardown.
func acquireLiveDatabase(
	t *testing.T,
	ctx context.Context,
	operatorMilvus *milvusclient.Client,
) liveDatabaseLease {
	t.Helper()
	lockDirectory := liveDatabasePoolDirectory(t)
	deadline := time.Now().Add(liveDatabasePoolWait)
	for {
		lease, acquired, err := tryAcquireLiveDatabaseSlot(lockDirectory)
		if err != nil {
			t.Fatalf("lock live Milvus database slot in %s: %v", lockDirectory, err)
		}
		if acquired {
			t.Cleanup(lease.release)
			ensureLiveDatabase(t, ctx, operatorMilvus, lease.name)
			t.Logf("leased live Milvus database %s", lease.name)
			return lease
		}
		if time.Now().After(deadline) {
			t.Fatalf("every live Milvus database slot stayed locked for %s", liveDatabasePoolWait)
		}
		time.Sleep(liveDatabasePoolPoll)
	}
}

// liveDatabasePoolDirectory creates the lock directory under the user cache
// directory and returns its path.
func liveDatabasePoolDirectory(t *testing.T) string {
	t.Helper()
	cacheDirectory, err := os.UserCacheDir()
	if err != nil {
		t.Fatalf("resolve user cache directory for live database locks: %v", err)
	}
	lockDirectory := filepath.Join(cacheDirectory, liveDatabasePoolAppName, liveDatabasePoolDirName)
	if err := os.MkdirAll(lockDirectory, liveDatabaseDirMode); err != nil {
		t.Fatalf("create live database lock directory %s: %v", lockDirectory, err)
	}
	return lockDirectory
}

// tryAcquireLiveDatabaseSlot tries each slot once with LOCK_NB to avoid
// waiting for a lock. It returns a lease for the first slot it locks exclusively.
func tryAcquireLiveDatabaseSlot(lockDirectory string) (liveDatabaseLease, bool, error) {
	for slot := range liveDatabasePoolSize {
		databaseName := fmt.Sprintf("%s%02d", liveDatabasePoolPrefix, slot)
		lockPath := filepath.Join(lockDirectory, databaseName+".lock")
		lockFile, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, liveDatabaseLockMode)
		if err != nil {
			return liveDatabaseLease{}, false, fmt.Errorf("open %s: %w", lockPath, err)
		}
		lockErr := syscall.Flock(int(lockFile.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
		if lockErr == nil {
			return liveDatabaseLease{name: databaseName, lockFile: lockFile}, true, nil
		}
		closeErr := lockFile.Close()
		if !errors.Is(lockErr, syscall.EWOULDBLOCK) {
			return liveDatabaseLease{}, false, errors.Join(fmt.Errorf("flock %s: %w", lockPath, lockErr), closeErr)
		}
		if closeErr != nil {
			return liveDatabaseLease{}, false, fmt.Errorf("close %s: %w", lockPath, closeErr)
		}
	}
	return liveDatabaseLease{}, false, nil
}

// ensureLiveDatabase creates the leased database if the database list omits it.
func ensureLiveDatabase(
	t *testing.T,
	ctx context.Context,
	operatorMilvus *milvusclient.Client,
	databaseName string,
) {
	t.Helper()
	databaseNames, err := listMilvusDatabases(operatorMilvus)
	if err != nil {
		t.Fatalf("list Milvus databases before leasing %s: %v", databaseName, err)
	}
	if slices.Contains(databaseNames, databaseName) {
		return
	}
	createContext, cancelCreate := context.WithTimeout(ctx, 15*time.Second)
	defer cancelCreate()
	if err := operatorMilvus.CreateDatabase(
		createContext,
		milvusclient.NewCreateDatabaseOption(databaseName),
	); err != nil {
		t.Fatalf("CreateDatabase(%s) returned error: %v", databaseName, err)
	}
}

// release unlocks and closes the lease's lock file.
func (lease liveDatabaseLease) release() {
	_ = syscall.Flock(int(lease.lockFile.Fd()), syscall.LOCK_UN)
	_ = lease.lockFile.Close()
}

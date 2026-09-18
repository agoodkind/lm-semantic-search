//go:build offlinelive

package offlinelive

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// filesAfterEmptyBuildTimeout bounds the wait for the build the file watcher
// starts. It is far below the periodic sweep interval, and background sync is
// off in this harness, so only the watcher can start the build in time.
const filesAfterEmptyBuildTimeout = 30 * time.Second

// TestFilesAfterEmptyBuildStartFullBuild proves files written into a directory
// whose last build indexed nothing are indexed promptly. That build created no
// collection, so the watcher has nothing to converge individual paths into and
// has to start a full build instead.
func TestFilesAfterEmptyBuildStartFullBuild(t *testing.T) {
	harness := newHarnessWith(t, harnessOptions{fileWatcher: true})

	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatalf("resolve temp directory: %v", err)
	}
	directory := filepath.Join(base, "checkout")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatalf("create checkout directory: %v", err)
	}

	emptyRun := harness.waitForJob(harness.startIndexAt(directory))
	requireCompleted(t, emptyRun)
	if processed := emptyRun.GetProgress().GetFilesProcessed(); processed != 0 {
		t.Fatalf("empty directory run processed %d files, want 0", processed)
	}

	writeGeneratedSources(t, directory, smallRepositoryFileCount)

	waitContext, cancel := context.WithTimeout(context.Background(), filesAfterEmptyBuildTimeout)
	defer cancel()
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()
	for {
		status := harness.indexStatusAt(directory)
		if status.GetCodebase().GetLastSuccessfulRun().GetIndexedFiles() > 0 {
			return
		}
		select {
		case <-waitContext.Done():
			t.Fatalf(
				"no build indexed the files written after an empty build within %s:\n%s",
				filesAfterEmptyBuildTimeout,
				status.GetDisplayText(),
			)
		case <-ticker.C:
		}
	}
}

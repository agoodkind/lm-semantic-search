//go:build offlinelive

package offlinelive

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/model"
)

const (
	// filesAfterEmptyBuildTimeout bounds the wait for the build the file watcher
	// starts. It is far below the periodic sweep interval, and background sync is
	// off in this harness, so only the watcher can index the files in time.
	filesAfterEmptyBuildTimeout = 30 * time.Second

	// firstCheckoutFileCount makes the watcher-started build long enough that
	// more files can land while it is still running.
	firstCheckoutFileCount = 150

	// lateFileCount is how many files land while that build is running.
	lateFileCount = 3
)

// TestFilesAfterEmptyBuildStartFullBuild proves files written into a directory
// whose last build indexed nothing are indexed promptly. That build created no
// collection, so the watcher has nothing to converge individual paths into and
// has to start a full build instead.
func TestFilesAfterEmptyBuildStartFullBuild(t *testing.T) {
	harness := newHarnessWith(t, harnessOptions{fileWatcher: true, backgroundSync: false, maxConcurrentIndexJobs: 0})
	directory := harness.newEmptyIndexedDirectory()

	writeGeneratedSources(t, directory, smallRepositoryFileCount)

	harness.waitForStatus(directory, "no build indexed the files written after an empty build", func(status *pb.GetIndexResponse) bool {
		return status.GetCodebase().GetLastSuccessfulRun().GetIndexedFiles() > 0
	})
}

// TestFilesWrittenDuringBuildAfterEmptyRunAreIndexed proves files that land
// while the watcher-started build is running are indexed after it. That build
// already walked the tree, so they are indexed only if the watcher keeps their
// paths for a converge once the build ends.
func TestFilesWrittenDuringBuildAfterEmptyRunAreIndexed(t *testing.T) {
	harness := newHarnessWith(t, harnessOptions{fileWatcher: true, backgroundSync: false, maxConcurrentIndexJobs: 0})
	directory := harness.newEmptyIndexedDirectory()

	writeGeneratedSources(t, directory, firstCheckoutFileCount)
	harness.waitForStatus(directory, "the watcher did not start a build over the first files", func(status *pb.GetIndexResponse) bool {
		job := status.GetActiveJob()
		return job.GetState() == string(model.JobStateRunning) &&
			job.GetProgress().GetFilesTotal() == firstCheckoutFileCount
	})

	writeGeneratedSourceRange(t, directory, firstCheckoutFileCount, lateFileCount)

	harness.waitForStatusWithin(directory, jobPollTimeout, "the watcher-started build did not complete", func(status *pb.GetIndexResponse) bool {
		return status.GetActiveJob() == nil &&
			status.GetCodebase().GetLastSuccessfulRun().GetIndexedFiles() >= firstCheckoutFileCount
	})

	wantLine := fmt.Sprintf("current_index.indexed_files: %d", firstCheckoutFileCount+lateFileCount)
	harness.waitForStatus(directory, "files written during the build were not indexed after it", func(status *pb.GetIndexResponse) bool {
		return status.GetActiveJob() == nil && strings.Contains(status.GetDisplayText(), wantLine)
	})
}

// TestFilesInEmptyWorktreeWaitForSiblingFirstBuild proves the watcher does not
// start a build for files arriving in a worktree whose last build indexed
// nothing while a sibling's first build is running. It keeps the paths and
// builds them from the sibling's vectors once that first build completes.
func TestFilesInEmptyWorktreeWaitForSiblingFirstBuild(t *testing.T) {
	harness := newHarnessWith(t, harnessOptions{
		fileWatcher:            true,
		backgroundSync:         false,
		maxConcurrentIndexJobs: restartedConcurrentIndexJobs,
		resumeOnBoot:           false,
	})

	repository := newEmptyRepository(t)
	gitRun(t, repository, "-c", "commit.gpgsign=false", "commit", "--quiet", "--allow-empty", "--message", "start empty")
	worktree := addNestedWorktree(t, repository)
	emptyRun := harness.waitForJob(harness.startIndexAt(worktree))
	requireCompleted(t, emptyRun)
	if processed := emptyRun.GetProgress().GetFilesProcessed(); processed != 0 {
		t.Fatalf("empty worktree run processed %d files, want 0", processed)
	}

	writeGeneratedSources(t, repository, heldParentFileCount)
	harness.requireWorktreeBuildWaitsForSibling(repository, worktree, func() {
		writeGeneratedSources(t, worktree, heldParentFileCount)
	})
}

// newEmptyIndexedDirectory creates an empty directory, indexes it, and returns
// its resolved path once that build has completed with no file.
func (harness *harness) newEmptyIndexedDirectory() string {
	harness.t.Helper()

	base, err := filepath.EvalSymlinks(harness.t.TempDir())
	if err != nil {
		harness.t.Fatalf("resolve temp directory: %v", err)
	}
	directory := filepath.Join(base, "checkout")
	if err := os.Mkdir(directory, 0o755); err != nil {
		harness.t.Fatalf("create checkout directory: %v", err)
	}

	emptyRun := harness.waitForJob(harness.startIndexAt(directory))
	requireCompleted(harness.t, emptyRun)
	if processed := emptyRun.GetProgress().GetFilesProcessed(); processed != 0 {
		harness.t.Fatalf("empty directory run processed %d files, want 0", processed)
	}
	return directory
}

// waitForStatus polls the status of path until ready accepts it, failing with
// failure and the last status text after filesAfterEmptyBuildTimeout.
func (harness *harness) waitForStatus(path string, failure string, ready func(*pb.GetIndexResponse) bool) {
	harness.t.Helper()
	harness.waitForStatusWithin(path, filesAfterEmptyBuildTimeout, failure, ready)
}

func (harness *harness) waitForStatusWithin(
	path string,
	timeout time.Duration,
	failure string,
	ready func(*pb.GetIndexResponse) bool,
) {
	harness.t.Helper()

	waitContext, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()
	for {
		status := harness.indexStatusAt(path)
		if ready(status) {
			return
		}
		select {
		case <-waitContext.Done():
			harness.t.Fatalf("%s within %s:\n%s", failure, timeout, status.GetDisplayText())
		case <-ticker.C:
		}
	}
}

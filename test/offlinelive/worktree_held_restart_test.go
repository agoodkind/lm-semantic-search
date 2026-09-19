//go:build offlinelive

package offlinelive

import (
	"os"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/model"
)

const (
	// restartedConcurrentIndexJobs lets a worktree build run beside its sibling's
	// first build, which is the condition under which starting it early embeds
	// content the sibling is about to hold.
	restartedConcurrentIndexJobs = 2

	// firstSweepSettle outlasts the few seconds before the restarted daemon's
	// first background sweep, so the sweep has run its repair and retry passes.
	firstSweepSettle = 10 * time.Second

	// crashAfterFiles is how many files each build checkpoints before the
	// daemon is killed.
	crashAfterFiles = 10

	unreadableDirectoryMode = 0o000
	readableDirectoryMode   = 0o755
)

// TestRestartedInterruptedWorktreeWaitsForSiblingFirstBuild proves the repair
// pass of a restarted daemon does not resume a worktree's interrupted build
// while its sibling's first build is running, and that the worktree builds from
// the sibling's vectors once that first build completes.
func TestRestartedInterruptedWorktreeWaitsForSiblingFirstBuild(t *testing.T) {
	harness := newHarness(t)
	repository, worktree := newCommittedRepositoryWithWorktree(t, heldParentFileCount)

	// An operator indexes both checkouts; the daemon exits before either build
	// finishes, which leaves both registered without a completed run.
	harness.startIndexAt(repository)
	harness.startIndexAt(worktree)
	harness.restart(restartedHeldOptions())

	harness.requireWorktreeBuildWaitsForSibling(repository, worktree)
}

// TestRestartedFailedWorktreeRetryWaitsForSiblingFirstBuild proves the failed
// build retry of a restarted daemon does not rebuild a worktree while its
// sibling's first build is running, and that the worktree builds from the
// sibling's vectors once that first build completes.
func TestRestartedFailedWorktreeRetryWaitsForSiblingFirstBuild(t *testing.T) {
	harness := newHarness(t)
	repository, worktree := newCommittedRepositoryWithWorktree(t, heldParentFileCount)

	// An unreadable checkout makes the worktree's build fail; the permission is
	// restored before the retry, so only the hold can keep the retry waiting.
	setDirectoryMode(t, worktree, unreadableDirectoryMode)
	t.Cleanup(func() {
		setDirectoryMode(t, worktree, readableDirectoryMode)
	})
	failedRun := harness.waitForJob(harness.startIndexAt(worktree))
	if failedRun.GetState() != string(model.JobStateFailed) {
		t.Fatalf("unreadable worktree build ended %q, want %q", failedRun.GetState(), model.JobStateFailed)
	}
	setDirectoryMode(t, worktree, readableDirectoryMode)
	harness.restart(restartedHeldOptions())

	harness.requireWorktreeBuildWaitsForSibling(repository, worktree)
}

// TestCrashResumedWorktreeWaitsForSiblingFirstBuild proves boot resume does not
// resume a worktree's interrupted build while its sibling's first build, which
// boot resume also restarted, is running. The worktree builds from the sibling's
// vectors once that first build completes.
func TestCrashResumedWorktreeWaitsForSiblingFirstBuild(t *testing.T) {
	harness := newUnstartedHarness(t)
	repository, worktree := newCommittedRepositoryWithWorktree(t, heldParentFileCount)
	options := harnessOptions{
		fileWatcher:            false,
		backgroundSync:         false,
		maxConcurrentIndexJobs: restartedConcurrentIndexJobs,
		resumeOnBoot:           true,
	}

	// Both first builds checkpoint some files, then the daemon dies without
	// shutting down, which leaves both resumable from their checkpoints.
	crash := harness.startCrashableDaemon(options)
	harness.startIndexAt(repository)
	harness.startIndexAt(worktree)
	harness.waitForBuildUnderway(repository)
	harness.waitForBuildUnderway(worktree)
	crash()

	harness.start(options, false)
	t.Cleanup(harness.teardown)
	harness.requireWorktreeBuildWaitsForSibling(repository, worktree)
}

// waitForBuildUnderway waits until the build of path has checkpointed files.
func (harness *harness) waitForBuildUnderway(path string) {
	harness.t.Helper()

	harness.waitForStatusWithin(path, jobPollTimeout, "build did not get under way", func(status *pb.GetIndexResponse) bool {
		job := status.GetActiveJob()
		return job.GetState() == string(model.JobStateRunning) &&
			job.GetProgress().GetFilesProcessed() >= crashAfterFiles
	})
}

func restartedHeldOptions() harnessOptions {
	return harnessOptions{
		fileWatcher:            false,
		backgroundSync:         true,
		maxConcurrentIndexJobs: restartedConcurrentIndexJobs,
		resumeOnBoot:           false,
	}
}

// requireWorktreeBuildWaitsForSibling starts the sibling's first build on the
// restarted daemon and asserts the worktree starts no build through the first
// background sweep while it runs, then builds from the sibling's vectors once it
// completes.
func (harness *harness) requireWorktreeBuildWaitsForSibling(repository string, worktree string) {
	harness.t.Helper()

	registered, found := harness.codebaseAt(worktree)
	if !found {
		harness.t.Fatalf("worktree %s is not registered after the restart", worktree)
	}
	earlierJobs := harness.jobIDsFor(registered.GetId())

	parentJobID := harness.startIndexAt(repository)
	deadline := time.Now().Add(firstSweepSettle)
	for time.Now().Before(deadline) {
		if job := harness.indexStatusAt(worktree).GetActiveJob(); job != nil {
			harness.t.Fatalf(
				"worktree build %s started by %q while its sibling's first build runs",
				job.GetId(),
				job.GetClient().GetName(),
			)
		}
		time.Sleep(pollInterval)
	}

	parentRun := harness.waitForJob(parentJobID)
	requireCompleted(harness.t, parentRun)

	waitDeadline := time.Now().Add(worktreeJobAppearTimeout)
	for time.Now().Before(waitDeadline) {
		if job, started := harness.newJobFor(registered.GetId(), earlierJobs); started {
			worktreeRun := harness.waitForJob(job.GetId())
			requireCompleted(harness.t, worktreeRun)
			requireReusedWithoutEmbedding(harness.t, worktreeRun)
			return
		}
		time.Sleep(pollInterval)
	}
	harness.t.Fatalf("worktree build did not start within %s of its sibling's first build completing", worktreeJobAppearTimeout)
}

func (harness *harness) jobIDsFor(codebaseID string) map[string]struct{} {
	harness.t.Helper()

	ids := make(map[string]struct{})
	for _, job := range harness.jobsFor(codebaseID) {
		ids[job.GetId()] = struct{}{}
	}
	return ids
}

// newJobFor returns a job for the codebase that is not among earlierJobs.
func (harness *harness) newJobFor(codebaseID string, earlierJobs map[string]struct{}) (*pb.Job, bool) {
	harness.t.Helper()

	for _, job := range harness.jobsFor(codebaseID) {
		if _, earlier := earlierJobs[job.GetId()]; !earlier {
			return job, true
		}
	}
	return nil, false
}

func setDirectoryMode(t *testing.T, directory string, mode os.FileMode) {
	t.Helper()

	if err := os.Chmod(directory, mode); err != nil {
		t.Fatalf("chmod %s to %o: %v", directory, mode, err)
	}
}

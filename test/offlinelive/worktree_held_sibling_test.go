//go:build offlinelive

package offlinelive

import (
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/model"
)

const (
	// heldParentFileCount makes the sibling's first build long enough that the
	// worktree read and a cancel request both land while it is still running.
	heldParentFileCount = 150

	// heldPastDiscoveryTimer outlasts the daemon's few-second deferred-build
	// delay, so the timer set when the worktree was discovered has fired.
	heldPastDiscoveryTimer = 6 * time.Second

	discoveredStatus = "discovered"

	// waitingForSiblingText is the phrase both the status and the search note use
	// for a worktree held behind its sibling's first build.
	waitingForSiblingText = "sibling worktree's first index"

	// buildStartingText is the discovered search note's claim that the build is
	// under way, which a held worktree must not make.
	buildStartingText = "its build is starting now"
)

// TestWorktreeWaitsForSiblingFirstBuild proves a worktree read while its
// sibling's first build is still embedding is registered and held without a
// job, reports that it is waiting, and builds from the sibling's vectors once
// that first build completes.
func TestWorktreeWaitsForSiblingFirstBuild(t *testing.T) {
	harness := newHarness(t)
	repository, worktree := newCommittedRepositoryWithWorktree(t, heldParentFileCount)

	parentJobID := harness.startIndexAt(repository)
	held := harness.requireHeldWorktree(worktree)

	parentRun := harness.waitForJob(parentJobID)
	requireCompleted(t, parentRun)

	worktreeRun := harness.waitForJob(harness.waitForFirstJobID(held.GetId()))
	requireCompleted(t, worktreeRun)
	startedAt := worktreeRun.GetStartedAt().AsTime()
	parentCompletedAt := parentRun.GetCompletedAt().AsTime()
	if startedAt.Before(parentCompletedAt) {
		t.Fatalf(
			"worktree build started at %s, before its sibling's first build completed at %s",
			startedAt,
			parentCompletedAt,
		)
	}
	requireReusedWithoutEmbedding(t, worktreeRun)
}

// TestWorktreeBuildsAfterSiblingFirstBuildIsCancelled proves a held worktree is
// not stranded when the sibling it waits on ends without content. It builds on
// its own, embedding everything, since there is nothing to reuse.
func TestWorktreeBuildsAfterSiblingFirstBuildIsCancelled(t *testing.T) {
	harness := newHarness(t)
	repository, worktree := newCommittedRepositoryWithWorktree(t, heldParentFileCount)

	parentJobID := harness.startIndexAt(repository)
	held := harness.requireHeldWorktree(worktree)
	// Cancel only after the discovery timer has fired and left the worktree
	// held, so the build that follows can only come from the cancellation.
	harness.requireNoJobsFor(held.GetId(), heldPastDiscoveryTimer)

	if _, err := harness.client.CancelJob(
		correlatedContext(),
		&pb.CancelJobRequest{JobId: parentJobID, Client: harnessClientInfo()},
	); err != nil {
		t.Fatalf("cancel sibling first build %s: %v", parentJobID, err)
	}
	parentRun := harness.waitForJob(parentJobID)
	if parentRun.GetState() != string(model.JobStateCancelled) {
		t.Fatalf(
			"sibling first build ended %q, want %q; the fixture must outlast the cancel request",
			parentRun.GetState(),
			model.JobStateCancelled,
		)
	}

	worktreeRun := harness.waitForJob(harness.waitForFirstJobID(held.GetId()))
	requireCompleted(t, worktreeRun)
	if worktreeRun.GetProgress().GetChunksEmbedded() <= 0 {
		t.Fatalf(
			"worktree build after a cancelled sibling embedded %d chunks, want more than 0",
			worktreeRun.GetProgress().GetChunksEmbedded(),
		)
	}
}

// requireHeldWorktree reads the worktree while its sibling's first build runs
// and asserts the daemon registered it without starting a job and told the
// reader it is waiting, on both the status and the search surface.
func (harness *harness) requireHeldWorktree(worktree string) *pb.Codebase {
	harness.t.Helper()

	status := harness.indexStatusAt(worktree)
	codebase := status.GetCodebase()
	if codebase.GetCanonicalPath() != worktree {
		harness.t.Fatalf(
			"worktree resolved to %q while its sibling's first build runs, want its own codebase at %q:\n%s",
			codebase.GetCanonicalPath(),
			worktree,
			status.GetDisplayText(),
		)
	}
	if codebase.GetStatus() != discoveredStatus {
		harness.t.Fatalf("held worktree status = %q, want %q", codebase.GetStatus(), discoveredStatus)
	}
	if codebase.GetActiveJobId() != "" || status.GetActiveJob() != nil {
		harness.t.Fatalf("held worktree has active job %q, want none", codebase.GetActiveJobId())
	}
	if !strings.Contains(status.GetDisplayText(), waitingForSiblingText) {
		harness.t.Fatalf("held worktree status does not say it is waiting for its sibling:\n%s", status.GetDisplayText())
	}

	search := harness.searchAt(worktree, fixtureQuery)
	if len(search.GetResults()) != 0 {
		harness.t.Fatalf("held worktree search returned %d results, want 0", len(search.GetResults()))
	}
	if !strings.Contains(search.GetDisplayText(), waitingForSiblingText) ||
		strings.Contains(search.GetDisplayText(), buildStartingText) {
		harness.t.Fatalf("held worktree search note does not say it is waiting for its sibling:\n%s", search.GetDisplayText())
	}

	if jobs := harness.jobsFor(codebase.GetId()); len(jobs) != 0 {
		harness.t.Fatalf("held worktree has %d jobs, want 0", len(jobs))
	}
	return codebase
}

// requireNoJobsFor asserts a codebase starts no job for the whole duration.
func (harness *harness) requireNoJobsFor(codebaseID string, duration time.Duration) {
	harness.t.Helper()

	deadline := time.Now().Add(duration)
	for time.Now().Before(deadline) {
		if jobs := harness.jobsFor(codebaseID); len(jobs) != 0 {
			harness.t.Fatalf("held worktree started job %s while its sibling's first build runs", jobs[0].GetId())
		}
		time.Sleep(pollInterval)
	}
}

// newCommittedRepositoryWithWorktree creates a repository holding fileCount
// committed sources and a nested worktree at the same commit.
func newCommittedRepositoryWithWorktree(t *testing.T, fileCount int) (string, string) {
	t.Helper()

	repository := newEmptyRepository(t)
	writeGeneratedSources(t, repository, fileCount)
	gitRun(t, repository, "add", "--all")
	gitRun(t, repository, "commit", "--quiet", "--message", "add sources")
	return repository, addNestedWorktree(t, repository)
}

func (harness *harness) searchAt(path string, query string) *pb.SearchCodeResponse {
	harness.t.Helper()

	response, err := harness.client.SearchCode(
		correlatedContext(),
		&pb.SearchCodeRequest{
			Path:   path,
			Query:  query,
			Limit:  searchResultLimit,
			Client: harnessClientInfo(),
		},
	)
	if err != nil {
		harness.t.Fatalf("search %s: %v", path, err)
	}
	return response
}

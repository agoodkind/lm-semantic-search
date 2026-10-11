//go:build offlinelive

package offlinelive

import (
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

	heldForSiblingStatusField = "held_for_sibling_build=true"
)

// TestSiblingBuildEndDoesNotStartUnheldWorktree proves a sibling's build ending
// starts only worktree builds that a hold stopped. A worktree whose own first
// build an operator cancelled was never held, so neither the sibling's first
// build completing nor a later routine sync of the sibling starts it.
func TestSiblingBuildEndDoesNotStartUnheldWorktree(t *testing.T) {
	harness := newHarness(t)
	repository, worktree := newCommittedRepositoryWithWorktree(t, heldParentFileCount)

	worktreeJobID := harness.startIndexAt(worktree)
	if _, err := harness.client.CancelJob(
		correlatedContext(),
		&pb.CancelJobRequest{JobId: worktreeJobID, Client: harnessClientInfo()},
	); err != nil {
		t.Fatalf("cancel worktree first build %s: %v", worktreeJobID, err)
	}
	if cancelled := harness.waitForJob(worktreeJobID); cancelled.GetState() != string(model.JobStateCancelled) {
		t.Fatalf("worktree first build ended %q, want %q", cancelled.GetState(), model.JobStateCancelled)
	}
	registered, found := harness.codebaseAt(worktree)
	if !found {
		t.Fatalf("worktree %s is not registered", worktree)
	}
	earlierJobs := harness.jobIDsFor(registered.GetId())

	requireCompleted(t, harness.waitForJob(harness.startIndexAt(repository)))
	harness.requireNoNewJobFor(registered.GetId(), earlierJobs, heldPastDiscoveryTimer)

	writeGeneratedSourceRange(t, repository, heldParentFileCount, 1)
	syncResponse, err := harness.client.SyncIndex(
		correlatedContext(),
		&pb.SyncIndexRequest{Path: repository, Client: harnessClientInfo()},
	)
	if err != nil {
		t.Fatalf("sync sibling: %v", err)
	}
	requireCompleted(t, harness.waitForJob(syncResponse.GetJobId()))
	harness.requireNoNewJobFor(registered.GetId(), earlierJobs, heldPastDiscoveryTimer)
}

// requireNoNewJobFor asserts a codebase starts no job beyond earlierJobs for
// the whole duration.
func (harness *harness) requireNoNewJobFor(codebaseID string, earlierJobs map[string]struct{}, duration time.Duration) {
	harness.t.Helper()

	deadline := time.Now().Add(duration)
	for time.Now().Before(deadline) {
		if job, started := harness.newJobFor(codebaseID, earlierJobs); started {
			harness.t.Fatalf("worktree job %s started by %q, but no hold stopped its build", job.GetId(), job.GetClient().GetName())
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
	gitCommitAll(t, repository, "add sources")
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

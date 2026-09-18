//go:build offlinelive

package offlinelive

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

const (
	// smallRepositoryFileCount keeps a build short when the test only needs the
	// parent to hold embedded content.
	smallRepositoryFileCount = 12

	// worktreeJobAppearTimeout bounds the wait for a worktree build the daemon
	// starts on its own. The deferred-build delay is a few seconds.
	worktreeJobAppearTimeout = 30 * time.Second

	worktreeBranch = "feature"
)

// TestWorktreeOfEmptyIndexWaitsForEmbeddedSibling proves a sibling whose only
// completed run indexed no file does not make a worktree eligible for its own
// build, and that once the sibling holds embedded content the worktree builds by
// reusing it rather than embedding again.
func TestWorktreeOfEmptyIndexWaitsForEmbeddedSibling(t *testing.T) {
	harness := newHarness(t)

	repository := newEmptyRepository(t)
	emptyRun := harness.waitForJob(harness.startIndexAt(repository))
	requireCompleted(t, emptyRun)
	if indexed := emptyRun.GetProgress().GetFilesProcessed(); indexed != 0 {
		t.Fatalf("empty repository run processed %d files, want 0", indexed)
	}

	writeGeneratedSources(t, repository, smallRepositoryFileCount)
	gitRun(t, repository, "add", "--all")
	gitRun(t, repository, "commit", "--quiet", "--message", "add sources")
	worktree := addNestedWorktree(t, repository)

	status := harness.indexStatusAt(worktree)
	if status.GetCodebase().GetCanonicalPath() == worktree {
		t.Fatalf(
			"worktree of a repository whose only run indexed no file was registered as %q (status %q); it must wait for a sibling with embedded content:\n%s",
			status.GetCodebase().GetId(),
			status.GetCodebase().GetStatus(),
			status.GetDisplayText(),
		)
	}
	if registered, found := harness.codebaseAt(worktree); found {
		t.Fatalf("worktree codebase %q is registered before any sibling holds embedded content", registered.GetId())
	}

	parentRun := harness.waitForJob(harness.startIndexAt(repository))
	requireCompleted(t, parentRun)

	discovered := harness.indexStatusAt(worktree)
	if discovered.GetCodebase().GetCanonicalPath() != worktree {
		t.Fatalf(
			"worktree resolved to %q after its sibling indexed content, want its own codebase at %q:\n%s",
			discovered.GetCodebase().GetCanonicalPath(),
			worktree,
			discovered.GetDisplayText(),
		)
	}
	worktreeRun := harness.waitForJob(harness.waitForFirstJobID(discovered.GetCodebase().GetId()))
	requireCompleted(t, worktreeRun)
	requireReusedWithoutEmbedding(t, worktreeRun)
}

// newEmptyRepository creates a git repository with no files beyond .git and
// returns its resolved path, the form the daemon reports.
func newEmptyRepository(t *testing.T) string {
	t.Helper()

	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatalf("resolve temp directory: %v", err)
	}
	repository := filepath.Join(base, "repository")
	if err := os.Mkdir(repository, 0o755); err != nil {
		t.Fatalf("create repository directory: %v", err)
	}
	gitRun(t, repository, "init", "--quiet", "--initial-branch", "main")
	return repository
}

// addNestedWorktree adds a linked worktree inside the repository on a new
// branch at the current commit and returns its resolved path.
func addNestedWorktree(t *testing.T, repository string) string {
	t.Helper()

	worktree := filepath.Join(repository, ".worktrees", worktreeBranch)
	gitRun(t, repository, "worktree", "add", "--quiet", "-b", worktreeBranch, worktree)
	return worktree
}

// writeGeneratedSources writes count distinct Go source files, each with enough
// content to produce its own chunk.
func writeGeneratedSources(t *testing.T, directory string, count int) {
	t.Helper()

	for index := range count {
		path := filepath.Join(directory, fmt.Sprintf("unit%04d.go", index))
		content := fmt.Sprintf(
			"package pkg\n\n// Unit%04d returns the running total of %d readings.\nfunc Unit%04d(readings []int) int {\n\ttotal := %d\n\tfor _, reading := range readings {\n\t\ttotal += reading * %d\n\t}\n\treturn total\n}\n",
			index, index, index, index, index+1,
		)
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatalf("write %s: %v", path, err)
		}
	}
}

func (harness *harness) startIndexAt(path string) string {
	harness.t.Helper()

	response, err := harness.client.StartIndex(
		correlatedContext(),
		&pb.StartIndexRequest{
			Path:     path,
			Splitter: &pb.SplitterConfig{Type: "ast"},
			Client:   harnessClientInfo(),
		},
	)
	if err != nil {
		harness.t.Fatalf("start index of %s: %v", path, err)
	}
	if response.GetJobId() == "" {
		harness.t.Fatalf("start index of %s returned an empty job id", path)
	}
	return response.GetJobId()
}

func (harness *harness) indexStatusAt(path string) *pb.GetIndexResponse {
	harness.t.Helper()

	response, err := harness.client.GetIndex(
		correlatedContext(),
		&pb.GetIndexRequest{Path: path, Client: harnessClientInfo()},
	)
	if err != nil {
		harness.t.Fatalf("get index status of %s: %v", path, err)
	}
	return response
}

// codebaseAt returns the registered codebase whose root is exactly path.
func (harness *harness) codebaseAt(path string) (*pb.Codebase, bool) {
	harness.t.Helper()

	response, err := harness.client.ListIndexes(correlatedContext(), &pb.ListIndexesRequest{})
	if err != nil {
		harness.t.Fatalf("list indexes: %v", err)
	}
	for _, codebase := range response.GetIndexes() {
		if codebase.GetCanonicalPath() == path {
			return codebase, true
		}
	}
	return nil, false
}

func (harness *harness) jobsFor(codebaseID string) []*pb.Job {
	harness.t.Helper()

	response, err := harness.client.ListJobs(
		correlatedContext(),
		&pb.ListJobsRequest{CodebaseId: codebaseID},
	)
	if err != nil {
		harness.t.Fatalf("list jobs for %s: %v", codebaseID, err)
	}
	return response.GetJobs()
}

// waitForFirstJobID waits for the daemon to start a job for a codebase on its
// own and returns that job's id.
func (harness *harness) waitForFirstJobID(codebaseID string) string {
	harness.t.Helper()

	waitContext, cancel := context.WithTimeout(context.Background(), worktreeJobAppearTimeout)
	defer cancel()
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()
	for {
		if jobs := harness.jobsFor(codebaseID); len(jobs) > 0 {
			return jobs[0].GetId()
		}
		select {
		case <-waitContext.Done():
			harness.t.Fatalf("no job started for codebase %s within %s", codebaseID, worktreeJobAppearTimeout)
		case <-ticker.C:
		}
	}
}

// requireReusedWithoutEmbedding asserts a build served every chunk from a
// sibling's vectors, which is the outcome when both checkouts are at one commit.
func requireReusedWithoutEmbedding(t *testing.T, job *pb.Job) {
	t.Helper()

	progress := job.GetProgress()
	if progress.GetChunksEmbedded() != 0 {
		t.Fatalf(
			"worktree build embedded %d chunks (reused %d, reuse vectors loaded %d), want 0 embedded",
			progress.GetChunksEmbedded(),
			progress.GetChunksReused(),
			progress.GetReuseVectorsLoaded(),
		)
	}
	if progress.GetChunksReused() <= 0 {
		t.Fatalf(
			"worktree build reused %d chunks (reuse vectors loaded %d), want more than 0",
			progress.GetChunksReused(),
			progress.GetReuseVectorsLoaded(),
		)
	}
}

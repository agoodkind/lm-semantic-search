//go:build offlinelive

package offlinelive

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
	"goodkind.io/lm-semantic-search/internal/view"
)

const (
	siblingWorktreeBranch = "sibling"

	sharedExtraSourceIndex = 500
	targetOnlySourceIndex  = 600

	subdirectoryName        = "component"
	subdirectoryFirstSource = 100
	subdirectoryFileCount   = 6

	failingSourceChunkBudget = 140
)

func (harness *harness) startIndexWithSourceAt(path string, sourcePath string) string {
	harness.t.Helper()

	response, err := harness.client.StartIndex(
		correlatedContext(),
		&pb.StartIndexRequest{
			Path:       path,
			Splitter:   &pb.SplitterConfig{Type: "ast"},
			Client:     harnessClientInfo(),
			SourcePath: sourcePath,
		},
	)
	if err != nil {
		harness.t.Fatalf("StartIndex for path %s with declared source %s returned error %v.", path, sourcePath, err)
	}
	if response.GetJobId() == "" {
		harness.t.Fatalf("StartIndex for path %s with declared source %s returned no job identifier.", path, sourcePath)
	}
	return response.GetJobId()
}

func (harness *harness) startIndexWithChunkBudgetAt(path string, maxJobChunks int32) string {
	harness.t.Helper()

	response, err := harness.client.StartIndex(
		correlatedContext(),
		&pb.StartIndexRequest{
			Path:         path,
			Splitter:     &pb.SplitterConfig{Type: "ast"},
			Client:       harnessClientInfo(),
			MaxJobChunks: maxJobChunks,
		},
	)
	if err != nil {
		harness.t.Fatalf("StartIndex for path %s with chunk budget %d returned error %v.", path, maxJobChunks, err)
	}
	if response.GetJobId() == "" {
		harness.t.Fatalf("StartIndex for path %s with chunk budget %d returned no job identifier.", path, maxJobChunks)
	}
	return response.GetJobId()
}

func (harness *harness) requireReuseReason(path string, wantReason view.WorktreeReuseReason) {
	harness.t.Helper()

	status := harness.indexStatusAt(path)
	gotReason := view.WorktreeReuseReason(status.GetCodebase().GetWorktreeReuseUnavailableReason())
	if gotReason != wantReason {
		harness.t.Fatalf("The status for path %s reports reuse reason %q instead of %q in status text %s.", path, gotReason, wantReason, status.GetDisplayText())
	}
}

func requireEmbeddedWithoutReuse(t *testing.T, job *pb.Job) {
	t.Helper()

	progress := job.GetProgress()
	if progress.GetChunksEmbedded() <= 0 || progress.GetChunksReused() != 0 {
		t.Fatalf("The build embedded %d chunks and reused %d chunks instead of embedding more than 0 and reusing 0.", progress.GetChunksEmbedded(), progress.GetChunksReused())
	}
}

func addNamedNestedWorktree(t *testing.T, repository string, branch string) string {
	t.Helper()

	worktree := filepath.Join(repository, ".worktrees", branch)
	gitRun(t, repository, "worktree", "add", "--quiet", "-b", branch, worktree)
	return worktree
}

func newSmallCommittedRepository(t *testing.T) string {
	t.Helper()

	repository := newEmptyRepository(t)
	writeGeneratedSources(t, repository, smallRepositoryFileCount)
	gitCommitAll(t, repository, "add sources")
	return repository
}

func TestWorktreeReusesDeclaredSourceAndEmbedsOnlyMissingChunks(t *testing.T) {
	harness := newHarness(t)
	repository := newSmallCommittedRepository(t)
	sibling := addNamedNestedWorktree(t, repository, siblingWorktreeBranch)
	target := addNestedWorktree(t, repository)
	writeGeneratedSourceRange(t, sibling, sharedExtraSourceIndex, 1)
	writeGeneratedSourceRange(t, target, sharedExtraSourceIndex, 1)
	writeGeneratedSourceRange(t, target, targetOnlySourceIndex, 1)

	sourceRun := harness.waitForJob(harness.startIndexAt(repository))
	requireCompleted(t, sourceRun)
	sourceChunks := sourceRun.GetProgress().GetChunksEmbedded()
	if sourceChunks <= 0 {
		t.Fatalf("The source build embedded %d chunks instead of more than 0.", sourceChunks)
	}

	siblingRun := harness.waitForJob(harness.startIndexAt(sibling))
	requireCompleted(t, siblingRun)
	requireEmbeddedWithoutReuse(t, siblingRun)
	extraFileChunks := siblingRun.GetProgress().GetChunksEmbedded() - sourceChunks
	if extraFileChunks <= 0 {
		t.Fatalf("The sibling build embedded %d chunks but must embed more than the source build's %d chunks.", siblingRun.GetProgress().GetChunksEmbedded(), sourceChunks)
	}

	targetRun := harness.waitForJob(harness.startIndexWithSourceAt(target, repository))
	requireCompleted(t, targetRun)
	if reused := targetRun.GetProgress().GetChunksReused(); reused != sourceChunks {
		t.Fatalf("The target build reused %d chunks instead of %d.", reused, sourceChunks)
	}
	wantEmbedded := 2 * extraFileChunks
	if embedded := targetRun.GetProgress().GetChunksEmbedded(); embedded != wantEmbedded {
		t.Fatalf("The target build embedded %d chunks instead of %d.", embedded, wantEmbedded)
	}
	harness.requireReuseReason(target, view.WorktreeReuseAvailable)
	if sourcePath := harness.indexStatusAt(target).GetCodebase().GetWorktreeSourcePath(); sourcePath != repository {
		t.Fatalf("The codebase records source path %q instead of %q.", sourcePath, repository)
	}
}

func TestWorktreeWithUnregisteredDeclaredSourceBuildsIndependently(t *testing.T) {
	harness := newHarness(t)
	repository := newSmallCommittedRepository(t)
	target := addNestedWorktree(t, repository)

	targetRun := harness.waitForJob(harness.startIndexWithSourceAt(target, repository))
	requireCompleted(t, targetRun)
	requireEmbeddedWithoutReuse(t, targetRun)
	harness.requireReuseReason(target, view.WorktreeReuseSourceNotRegistered)
}

func TestWorktreeWithUnindexedDeclaredSourceBuildsIndependently(t *testing.T) {
	harness := newHarness(t)
	repository := newEmptyRepository(t)
	emptyRun := harness.waitForJob(harness.startIndexAt(repository))
	requireCompleted(t, emptyRun)
	if processed := emptyRun.GetProgress().GetFilesProcessed(); processed != 0 {
		t.Fatalf("empty repository run processed %d files, want 0", processed)
	}
	writeGeneratedSources(t, repository, smallRepositoryFileCount)
	gitCommitAll(t, repository, "add sources")
	target := addNestedWorktree(t, repository)

	targetRun := harness.waitForJob(harness.startIndexWithSourceAt(target, repository))
	requireCompleted(t, targetRun)
	requireEmbeddedWithoutReuse(t, targetRun)
	harness.requireReuseReason(target, view.WorktreeReuseSourceNotIndexed)
}

func TestWorktreeDoesNotReuseDeclaredSourceWithAnotherEmbeddingModel(t *testing.T) {
	harness := newUnstartedHarness(t)
	repository := newSmallCommittedRepository(t)
	target := addNestedWorktree(t, repository)

	sourceOptions := harnessOptions{offlineEmbeddingModel: offlinemodel.BGESmall}
	harness.start(sourceOptions, true)
	t.Cleanup(harness.teardown)
	sourceRun := harness.waitForJob(harness.startIndexAt(repository))
	requireCompleted(t, sourceRun)
	if sourceModel := sourceRun.GetConfig().GetEmbeddingModel(); sourceModel != offlinemodel.BGESmall {
		t.Fatalf("The source job used embedding model %q instead of %q.", sourceModel, offlinemodel.BGESmall)
	}

	harness.restart(harnessOptions{offlineEmbeddingModel: offlinemodel.EmbeddingGemma})
	targetRun := harness.waitForJob(harness.startIndexWithSourceAt(target, repository))
	requireCompleted(t, targetRun)
	requireEmbeddedWithoutReuse(t, targetRun)
	harness.requireReuseReason(target, view.WorktreeReuseSourceEmbeddingConfigurationDiffers)
}

func TestWorktreeBuildsWithoutReuseWhenDeclaredSourceCollectionIsUnreadable(t *testing.T) {
	harness := newHarness(t)
	repository := newSmallCommittedRepository(t)
	target := addNestedWorktree(t, repository)

	requireCompleted(t, harness.waitForJob(harness.startIndexAt(repository)))
	sourceCollection := harness.indexStatusAt(repository).GetCodebase().GetCollectionName()
	metadataPath := filepath.Join(harness.config.StateRoot, "localvec", sourceCollection, "metadata.jsonl")
	if _, err := os.Stat(metadataPath); err != nil {
		t.Fatalf("The source collection's metadata file at %s is missing according to error %v.", metadataPath, err)
	}
	if err := os.WriteFile(metadataPath, []byte("unreadable row\nunreadable row\n"), 0o600); err != nil {
		t.Fatalf("Overwriting the metadata file at %s failed with error %v.", metadataPath, err)
	}
	harness.restart(harnessOptions{})

	targetRun := harness.waitForJob(harness.startIndexWithSourceAt(target, repository))
	requireCompleted(t, targetRun)
	requireEmbeddedWithoutReuse(t, targetRun)
	harness.requireReuseReason(target, view.WorktreeReuseAvailable)
}

func TestWorktreeWithoutDeclaredSourceEmbedsEveryChunk(t *testing.T) {
	harness := newHarness(t)
	repository := newSmallCommittedRepository(t)
	target := addNestedWorktree(t, repository)

	sourceRun := harness.waitForJob(harness.startIndexAt(repository))
	requireCompleted(t, sourceRun)

	targetRun := harness.waitForJob(harness.startIndexAt(target))
	requireCompleted(t, targetRun)
	requireEmbeddedWithoutReuse(t, targetRun)
	if embedded := targetRun.GetProgress().GetChunksEmbedded(); embedded != sourceRun.GetProgress().GetChunksEmbedded() {
		t.Fatalf("The worktree without a declared source embedded %d chunks instead of matching the identical sibling's %d chunks.", embedded, sourceRun.GetProgress().GetChunksEmbedded())
	}
	if strings.Contains(harness.indexStatusAt(target).GetDisplayText(), "worktree_reuse_unavailable_reason=") {
		t.Fatalf("The status must omit the reuse reason field from status text %s.", harness.indexStatusAt(target).GetDisplayText())
	}
}

func TestParentBuildReusesAndAbsorbsIndexedSubdirectory(t *testing.T) {
	harness := newHarness(t)
	repository := newSmallCommittedRepository(t)
	subdirectory := filepath.Join(repository, subdirectoryName)
	if err := os.Mkdir(subdirectory, 0o755); err != nil {
		t.Fatalf("Directory creation at %s failed with error %v.", subdirectory, err)
	}
	writeGeneratedSourceRange(t, subdirectory, subdirectoryFirstSource, subdirectoryFileCount)
	gitCommitAll(t, repository, "add component")

	subdirectoryRun := harness.waitForJob(harness.startIndexAt(subdirectory))
	requireCompleted(t, subdirectoryRun)
	subdirectoryChunks := subdirectoryRun.GetProgress().GetChunksEmbedded()
	if subdirectoryChunks <= 0 {
		t.Fatalf("The subdirectory build embedded %d chunks instead of more than 0.", subdirectoryChunks)
	}

	parentRun := harness.waitForJob(harness.startIndexAt(repository))
	requireCompleted(t, parentRun)
	if reused := parentRun.GetProgress().GetChunksReused(); reused != subdirectoryChunks {
		t.Fatalf("The parent build reused %d chunks instead of %d.", reused, subdirectoryChunks)
	}
	if embedded := parentRun.GetProgress().GetChunksEmbedded(); embedded <= 0 {
		t.Fatalf("The parent build embedded %d chunks instead of more than 0.", embedded)
	}
	if absorbed, found := harness.codebaseAt(subdirectory); found {
		t.Fatalf("The subdirectory codebase at %s remains registered with identifier %s.", subdirectory, absorbed.GetId())
	}
}

func (harness *harness) newEmptyWorktreeWithDeclaredSource() (string, string) {
	harness.t.Helper()

	repository := newEmptyRepository(harness.t)
	gitRun(harness.t, repository, "-c", "commit.gpgsign=false", "commit", "--quiet", "--allow-empty", "--message", "start empty")
	worktree := addNestedWorktree(harness.t, repository)
	emptyRun := harness.waitForJob(harness.startIndexWithSourceAt(worktree, repository))
	requireCompleted(harness.t, emptyRun)
	if processed := emptyRun.GetProgress().GetFilesProcessed(); processed != 0 {
		harness.t.Fatalf("empty worktree run processed %d files, want 0", processed)
	}
	return repository, worktree
}

func (harness *harness) waitForHeldForDeclaredSource(worktree string) *pb.Codebase {
	harness.t.Helper()

	harness.waitForStatus(worktree, "The worktree did not report a wait for the declared source", func(status *pb.GetIndexResponse) bool {
		return status.GetActiveJob() == nil && strings.Contains(status.GetDisplayText(), heldForSiblingStatusField)
	})
	registered, found := harness.codebaseAt(worktree)
	if !found {
		harness.t.Fatalf("worktree %s is not registered", worktree)
	}
	return registered
}

func (harness *harness) requireIndependentBuildAfterSourceEnds(
	worktree string,
	registered *pb.Codebase,
	earlierJobs map[string]struct{},
	sourceRun *pb.Job,
) {
	harness.t.Helper()

	harness.waitForStatusWithin(worktree, worktreeJobAppearTimeout, "No worktree build started after the source build ended", func(*pb.GetIndexResponse) bool {
		_, started := harness.newJobFor(registered.GetId(), earlierJobs)
		return started
	})
	worktreeJob, _ := harness.newJobFor(registered.GetId(), earlierJobs)
	worktreeRun := harness.waitForJob(worktreeJob.GetId())
	requireCompleted(harness.t, worktreeRun)
	requireEmbeddedWithoutReuse(harness.t, worktreeRun)
	if worktreeRun.GetStartedAt().AsTime().Before(sourceRun.GetCompletedAt().AsTime()) {
		harness.t.Fatalf("The worktree build started at %s before the source build ended at %s.", worktreeRun.GetStartedAt().AsTime(), sourceRun.GetCompletedAt().AsTime())
	}
}

func declaredSourceWaitOptions() harnessOptions {
	return harnessOptions{
		fileWatcher:            true,
		backgroundSync:         false,
		maxConcurrentIndexJobs: restartedConcurrentIndexJobs,
		resumeOnBoot:           false,
		syncIntervalMS:         0,
	}
}

func TestWorktreeBuildsAfterDeclaredSourceFirstBuildIsCancelled(t *testing.T) {
	harness := newHarnessWith(t, declaredSourceWaitOptions())
	repository, worktree := harness.newEmptyWorktreeWithDeclaredSource()
	writeGeneratedSources(t, repository, heldParentFileCount)

	sourceJobID := harness.startIndexAt(repository)
	writeGeneratedSources(t, worktree, heldParentFileCount)
	registered := harness.waitForHeldForDeclaredSource(worktree)
	earlierJobs := harness.jobIDsFor(registered.GetId())

	if _, err := harness.client.CancelJob(
		correlatedContext(),
		&pb.CancelJobRequest{JobId: sourceJobID, Client: harnessClientInfo()},
	); err != nil {
		t.Fatalf("Cancellation of source build %s failed with error %v.", sourceJobID, err)
	}
	sourceRun := harness.waitForJob(sourceJobID)
	if sourceRun.GetState() != string(model.JobStateCancelled) {
		t.Fatalf("The source build ended in state %q instead of %q.", sourceRun.GetState(), model.JobStateCancelled)
	}
	harness.requireIndependentBuildAfterSourceEnds(worktree, registered, earlierJobs, sourceRun)
}

func TestWorktreeBuildsAfterDeclaredSourceFirstBuildFails(t *testing.T) {
	harness := newHarnessWith(t, declaredSourceWaitOptions())
	repository, worktree := harness.newEmptyWorktreeWithDeclaredSource()
	writeGeneratedSources(t, repository, heldParentFileCount)

	sourceJobID := harness.startIndexWithChunkBudgetAt(repository, failingSourceChunkBudget)
	writeGeneratedSources(t, worktree, heldParentFileCount)
	registered := harness.waitForHeldForDeclaredSource(worktree)
	earlierJobs := harness.jobIDsFor(registered.GetId())

	sourceRun := harness.waitForJob(sourceJobID)
	if sourceRun.GetState() != string(model.JobStateFailed) {
		t.Fatalf("The source build ended in state %q instead of %q.", sourceRun.GetState(), model.JobStateFailed)
	}
	harness.requireIndependentBuildAfterSourceEnds(worktree, registered, earlierJobs, sourceRun)
}

func TestUnrelatedSiblingFirstBuildDoesNotDelayWorktree(t *testing.T) {
	harness := newHarnessWith(t, declaredSourceWaitOptions())
	repository, worktree := harness.newEmptyWorktreeWithDeclaredSource()
	sibling := addNamedNestedWorktree(t, repository, siblingWorktreeBranch)
	writeGeneratedSources(t, sibling, heldParentFileCount)

	siblingJobID := harness.startIndexAt(sibling)
	harness.waitForBuildUnderway(sibling)
	writeGeneratedSources(t, worktree, smallRepositoryFileCount)

	harness.waitForStatusWithin(worktree, jobPollTimeout, "The worktree did not index files during the sibling build", func(status *pb.GetIndexResponse) bool {
		return status.GetCodebase().GetLastSuccessfulRun().GetIndexedFiles() > 0
	})
	siblingJob, err := harness.client.GetJob(correlatedContext(), &pb.GetJobRequest{JobId: siblingJobID})
	if err != nil {
		t.Fatalf("GetJob for job %s returned error %v.", siblingJobID, err)
	}
	if state := siblingJob.GetJob().GetState(); state != string(model.JobStateRunning) {
		t.Fatalf("The sibling build reports state %q instead of the required running state %q.", state, model.JobStateRunning)
	}
	if strings.Contains(harness.indexStatusAt(worktree).GetDisplayText(), heldForSiblingStatusField) {
		t.Fatalf("The worktree must not report a wait in status text %s.", harness.indexStatusAt(worktree).GetDisplayText())
	}
	requireCompleted(t, harness.waitForJob(siblingJobID))
}

func TestDiscoveredWorktreeBuildsDuringSiblingFirstBuildWithoutReuse(t *testing.T) {
	harness := newHarnessWith(t, harnessOptions{maxConcurrentIndexJobs: restartedConcurrentIndexJobs})
	repository, worktree := newCommittedRepositoryWithWorktree(t, heldParentFileCount)

	siblingJobID := harness.startIndexAt(repository)
	status := harness.indexStatusAt(worktree)
	discovered := status.GetCodebase()
	if discovered.GetCanonicalPath() != worktree || discovered.GetStatus() != discoveredStatus {
		t.Fatalf("The worktree at its own path %q reports status %q instead of \"discovered\" in status text %s.", discovered.GetCanonicalPath(), discovered.GetStatus(), status.GetDisplayText())
	}
	if strings.Contains(status.GetDisplayText(), heldForSiblingStatusField) {
		t.Fatalf("The discovered worktree must not report a wait in status text %s.", status.GetDisplayText())
	}
	harness.requireReuseReason(worktree, view.WorktreeReuseNoSourceRecorded)

	worktreeRun := harness.waitForJob(harness.waitForFirstJobID(discovered.GetId()))
	requireCompleted(t, worktreeRun)
	requireEmbeddedWithoutReuse(t, worktreeRun)
	requireCompleted(t, harness.waitForJob(siblingJobID))
}

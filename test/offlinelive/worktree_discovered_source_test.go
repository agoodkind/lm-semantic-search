//go:build offlinelive

package offlinelive

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
	"goodkind.io/lm-semantic-search/internal/view"
)

const (
	deferredBuildClientName = "worktree-deferred-build"

	sameCommitTargetBranch  = "same-commit-target"
	otherCommitTargetBranch = "other-commit-target"
	earlierIndexedBranch    = "alpha"
	laterIndexedBranch      = "beta"

	revisedSourceCount = 10
)

func addNestedWorktreeFrom(t *testing.T, repository string, branch string, startPoint string) string {
	t.Helper()

	worktree := filepath.Join(repository, ".worktrees", branch)
	gitRun(t, repository, "worktree", "add", "--quiet", "-b", branch, worktree, startPoint)
	return worktree
}

func addNestedWorktreeWithOwnCommit(t *testing.T, repository string, branch string) string {
	t.Helper()

	worktree := addNamedNestedWorktree(t, repository, branch)
	writeGeneratedSourceRange(t, worktree, targetOnlySourceIndex, 1)
	gitCommitAll(t, worktree, "add target source")
	return worktree
}

func (harness *harness) requireDiscoveredSource(worktree string, wantSourcePath string) *pb.Codebase {
	harness.t.Helper()

	wantSourceID := ""
	if wantSourcePath != "" {
		source, found := harness.codebaseAt(wantSourcePath)
		if !found {
			harness.t.Fatalf("The expected source path %s is not a registered codebase.", wantSourcePath)
		}
		wantSourceID = source.GetId()
	}
	status := harness.indexStatusAt(worktree)
	discovered := status.GetCodebase()
	if discovered.GetCanonicalPath() != worktree || discovered.GetStatus() != discoveredStatus {
		harness.t.Fatalf("The status read for worktree %s returned canonical path %q and status %q instead of the worktree path with status discovered. The status text was %s.", worktree, discovered.GetCanonicalPath(), discovered.GetStatus(), status.GetDisplayText())
	}
	if discovered.GetWorktreeSourcePath() != wantSourcePath ||
		discovered.GetWorktreeSourceId() != wantSourceID ||
		discovered.GetWorktreeStartingCommit() != "" {
		harness.t.Fatalf(
			"The discovered worktree %s records source path %q, source identifier %q, and starting commit %q instead of source path %q, source identifier %q, and an empty starting commit.",
			worktree,
			discovered.GetWorktreeSourcePath(),
			discovered.GetWorktreeSourceId(),
			discovered.GetWorktreeStartingCommit(),
			wantSourcePath,
			wantSourceID,
		)
	}
	if jobs := harness.jobsFor(discovered.GetId()); len(jobs) != 0 {
		harness.t.Fatalf("The discovered worktree %s has %d jobs before its first build instead of 0.", worktree, len(jobs))
	}
	return discovered
}

func (harness *harness) waitForAutomaticBuild(discovered *pb.Codebase) *pb.Job {
	harness.t.Helper()

	run := harness.waitForJob(harness.waitForFirstJobID(discovered.GetId()))
	requireCompleted(harness.t, run)
	if clientName := run.GetClient().GetName(); clientName != deferredBuildClientName {
		harness.t.Fatalf("The first build of worktree %s came from client %q instead of client %q.", discovered.GetCanonicalPath(), clientName, deferredBuildClientName)
	}
	return run
}

func (harness *harness) indexInOrder(earlier string, later string) {
	harness.t.Helper()

	earlierRun := harness.waitForJob(harness.startIndexAt(earlier))
	requireCompleted(harness.t, earlierRun)
	laterRun := harness.waitForJob(harness.startIndexAt(later))
	requireCompleted(harness.t, laterRun)
	if !laterRun.GetCompletedAt().AsTime().After(earlierRun.GetCompletedAt().AsTime()) {
		harness.t.Fatalf("The build of %s completed at %s, but the build of %s completed at %s or later.", later, laterRun.GetCompletedAt().AsTime(), earlier, earlierRun.GetCompletedAt().AsTime())
	}
}

func TestDiscoveredWorktreeRecordsIndexedMainCheckoutAndEmbedsOnlyAbsentFiles(t *testing.T) {
	harness := newHarness(t)
	repository := newSmallCommittedRepository(t)
	target := addNestedWorktree(t, repository)
	writeGeneratedSourceRange(t, target, targetOnlySourceIndex, 1)

	sourceRun := harness.waitForJob(harness.startIndexAt(repository))
	requireCompleted(t, sourceRun)
	sourceChunks := sourceRun.GetProgress().GetChunksEmbedded()
	if sourceChunks <= 0 {
		t.Fatalf("The source build embedded %d chunks instead of more than 0.", sourceChunks)
	}

	discovered := harness.requireDiscoveredSource(target, repository)
	harness.requireReuseReason(target, view.WorktreeReuseAvailable)

	targetRun := harness.waitForAutomaticBuild(discovered)
	if reused := targetRun.GetProgress().GetChunksReused(); reused != sourceChunks {
		t.Fatalf("The automatic build reused %d chunks instead of the source build's %d chunks.", reused, sourceChunks)
	}
	targetChunks := harness.indexStatusAt(target).GetCodebase().GetLastSuccessfulRun().GetTotalChunks()
	wantEmbedded := targetChunks - sourceChunks
	if embedded := targetRun.GetProgress().GetChunksEmbedded(); wantEmbedded <= 0 || embedded != wantEmbedded {
		t.Fatalf("The automatic build embedded %d chunks instead of the required positive count %d. The chunk totals are %d for the target and %d for the source.", embedded, wantEmbedded, targetChunks, sourceChunks)
	}
	if sourcePath := harness.indexStatusAt(target).GetCodebase().GetWorktreeSourcePath(); sourcePath != repository {
		t.Fatalf("The codebase records source path %q after the build instead of the main checkout %q.", sourcePath, repository)
	}
}

func TestDiscoveredWorktreeSelectsSourceByHeadCommitThenMainCheckout(t *testing.T) {
	harness := newHarness(t)
	repository := newSmallCommittedRepository(t)
	sibling := addNamedNestedWorktree(t, repository, siblingWorktreeBranch)
	writeGeneratedSourceRange(t, sibling, sharedExtraSourceIndex, 1)
	gitCommitAll(t, sibling, "add sibling source")
	sameCommitTarget := addNestedWorktreeFrom(t, repository, sameCommitTargetBranch, siblingWorktreeBranch)
	otherCommitTarget := addNestedWorktreeWithOwnCommit(t, repository, otherCommitTargetBranch)
	harness.indexInOrder(repository, sibling)

	discoveredAtOtherCommit := harness.requireDiscoveredSource(otherCommitTarget, repository)
	discoveredAtSameCommit := harness.requireDiscoveredSource(sameCommitTarget, sibling)

	harness.waitForAutomaticBuild(discoveredAtOtherCommit)
	requireReusedWithoutEmbedding(t, harness.waitForAutomaticBuild(discoveredAtSameCommit))
}

func TestDiscoveredWorktreeSelectsLatestIndexedSiblingWithoutIndexedMainCheckout(t *testing.T) {
	harness := newHarness(t)
	repository := newSmallCommittedRepository(t)
	earlierIndexed := addNamedNestedWorktree(t, repository, earlierIndexedBranch)
	laterIndexed := addNamedNestedWorktree(t, repository, laterIndexedBranch)
	target := addNestedWorktreeWithOwnCommit(t, repository, otherCommitTargetBranch)
	harness.indexInOrder(earlierIndexed, laterIndexed)

	discovered := harness.requireDiscoveredSource(target, laterIndexed)

	harness.waitForAutomaticBuild(discovered)
}

func rewriteGeneratedSourceRange(t *testing.T, directory string, first int, count int) {
	t.Helper()

	for index := first; index < first+count; index++ {
		path := filepath.Join(directory, generatedSourceName(index))
		content := fmt.Sprintf(
			"package pkg\n\n// Revised%04d returns the largest of the readings above %d.\nfunc Revised%04d(readings []int) int {\n\tlargest := %d\n\tfor _, reading := range readings {\n\t\tif reading > largest {\n\t\t\tlargest = reading\n\t\t}\n\t}\n\treturn largest\n}\n",
			index, index, index, index,
		)
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatalf("Writing the revised source file %s failed with error %v.", path, err)
		}
	}
}

func newRepositoryWithRevisedSiblingAndTarget(t *testing.T) (string, string, string) {
	t.Helper()

	repository := newSmallCommittedRepository(t)
	sibling := addNamedNestedWorktree(t, repository, siblingWorktreeBranch)
	rewriteGeneratedSourceRange(t, sibling, 0, revisedSourceCount)
	gitCommitAll(t, sibling, "revise sources")
	target := addNestedWorktreeFrom(t, repository, otherCommitTargetBranch, siblingWorktreeBranch)
	writeGeneratedSourceRange(t, target, targetOnlySourceIndex, 1)
	gitCommitAll(t, target, "add target source")
	return repository, sibling, target
}

func (harness *harness) removeMerkleSnapshot(path string) {
	harness.t.Helper()

	codebase, found := harness.codebaseAt(path)
	if !found {
		harness.t.Fatalf("The registry does not contain the expected codebase at %s.", path)
	}
	snapshotPath := filepath.Join(harness.config.MerkleDir, codebase.GetId()+".json")
	if err := os.Remove(snapshotPath); err != nil {
		harness.t.Fatalf("Removing the Merkle snapshot %s for the codebase at %s failed with error %v.", snapshotPath, path, err)
	}
}

func TestDiscoveredWorktreeSelectsSiblingWithLargestContentOverlapOverMainCheckout(t *testing.T) {
	harness := newHarness(t)
	repository, sibling, target := newRepositoryWithRevisedSiblingAndTarget(t)
	harness.indexInOrder(sibling, repository)
	siblingChunks := harness.indexStatusAt(sibling).GetCodebase().GetLastSuccessfulRun().GetTotalChunks()
	if siblingChunks <= 0 {
		t.Fatalf("The sibling worktree at %s recorded %d chunks in its last successful build instead of more than 0.", sibling, siblingChunks)
	}

	discovered := harness.requireDiscoveredSource(target, sibling)

	targetRun := harness.waitForAutomaticBuild(discovered)
	if reused := targetRun.GetProgress().GetChunksReused(); reused != siblingChunks {
		t.Fatalf("The target worktree's automatic build reused %d chunks instead of the sibling build's %d chunks.", reused, siblingChunks)
	}
	targetChunks := harness.indexStatusAt(target).GetCodebase().GetLastSuccessfulRun().GetTotalChunks()
	wantEmbedded := targetChunks - siblingChunks
	if embedded := targetRun.GetProgress().GetChunksEmbedded(); wantEmbedded <= 0 || embedded != wantEmbedded {
		t.Fatalf("The target worktree's automatic build embedded %d chunks instead of the required positive count %d. The chunk totals are %d for the target and %d for the sibling.", embedded, wantEmbedded, targetChunks, siblingChunks)
	}
}

func TestDiscoveredWorktreeSelectsMainCheckoutWhenCandidateSnapshotsAreAbsent(t *testing.T) {
	harness := newHarness(t)
	repository, sibling, target := newRepositoryWithRevisedSiblingAndTarget(t)
	harness.indexInOrder(sibling, repository)
	harness.removeMerkleSnapshot(repository)
	harness.removeMerkleSnapshot(sibling)

	discovered := harness.requireDiscoveredSource(target, repository)

	harness.waitForAutomaticBuild(discovered)
}

func TestIndexRequestWithSourcePathReplacesDiscoveredSource(t *testing.T) {
	harness := newHarness(t)
	repository := newSmallCommittedRepository(t)
	sibling := addNamedNestedWorktree(t, repository, siblingWorktreeBranch)
	target := addNestedWorktree(t, repository)
	harness.indexInOrder(repository, sibling)
	siblingCodebase, found := harness.codebaseAt(sibling)
	if !found {
		t.Fatalf("The expected source path %s is not a registered codebase.", sibling)
	}

	harness.requireDiscoveredSource(target, repository)

	targetRun := harness.waitForJob(harness.startIndexWithSourceAt(target, sibling))
	requireCompleted(t, targetRun)
	requireReusedWithoutEmbedding(t, targetRun)
	replaced := harness.indexStatusAt(target).GetCodebase()
	if replaced.GetWorktreeSourcePath() != sibling || replaced.GetWorktreeSourceId() != siblingCodebase.GetId() {
		t.Fatalf("Target %s records source path %q and source identifier %q after the index request with a source path instead of source path %q and source identifier %q.", target, replaced.GetWorktreeSourcePath(), replaced.GetWorktreeSourceId(), sibling, siblingCodebase.GetId())
	}
}

func TestDiscoveredWorktreeRecordsNoSourceWhenIndexedSiblingUsesAnotherEmbeddingModel(t *testing.T) {
	harness := newUnstartedHarness(t)
	repository := newSmallCommittedRepository(t)
	target := addNestedWorktree(t, repository)

	harness.start(harnessOptions{offlineEmbeddingModel: offlinemodel.BGESmall}, true)
	t.Cleanup(harness.teardown)
	sourceRun := harness.waitForJob(harness.startIndexAt(repository))
	requireCompleted(t, sourceRun)
	if sourceModel := sourceRun.GetConfig().GetEmbeddingModel(); sourceModel != offlinemodel.BGESmall {
		t.Fatalf("The sibling job used embedding model %q instead of %q.", sourceModel, offlinemodel.BGESmall)
	}

	harness.restart(harnessOptions{offlineEmbeddingModel: offlinemodel.EmbeddingGemma})
	discovered := harness.requireDiscoveredSource(target, "")
	harness.requireReuseReason(target, view.WorktreeReuseNoSourceRecorded)

	targetRun := harness.waitForAutomaticBuild(discovered)
	requireEmbeddedWithoutReuse(t, targetRun)
	if embedded := targetRun.GetProgress().GetChunksEmbedded(); embedded != sourceRun.GetProgress().GetChunksEmbedded() {
		t.Fatalf("The discovered worktree embedded %d chunks instead of the sibling's %d chunks.", embedded, sourceRun.GetProgress().GetChunksEmbedded())
	}
}

//go:build offlinelive

package offlinelive

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/metrics"
)

// TestWatcherRenameAndMoveBackKeepsCurrentPath indexes generated sources with
// the file watcher on and periodic sync off. It renames unit0000.go to
// renamed_unit0000.go and then renames the file back. After each rename, it waits
// for the watcher to converge. The live Merkle snapshot must record the current
// path and must not record the previous path. Search must return the current
// path and must not return the previous path. The two converges must copy chunks
// for 2 paths, and the embedder must receive 0 inputs.
func TestWatcherRenameAndMoveBackKeepsCurrentPath(t *testing.T) {
	harness := newHarnessWith(t, harnessOptions{
		fileWatcher:            true,
		backgroundSync:         false,
		maxConcurrentIndexJobs: 0,
	})

	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatalf("resolve temp directory: %v", err)
	}
	directory := filepath.Join(base, "checkout")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatalf("create checkout directory: %v", err)
	}
	writeGeneratedSources(t, directory, smallRepositoryFileCount)
	requireCompleted(t, harness.waitForJob(harness.startIndexAt(directory)))

	before := metrics.Read()
	sourceName := generatedSourceName(renamedSourceIndex)
	sourcePath := filepath.Join(directory, sourceName)
	destinationPath := filepath.Join(directory, renamedSourceName)
	if err := os.Rename(sourcePath, destinationPath); err != nil {
		t.Fatalf("rename %s to %s: %v", sourcePath, destinationPath, err)
	}
	harness.waitForRenameConverge(directory, renamedSourceName)
	harness.requireCurrentPathOnly(directory, renamedSourceName, sourceName, "rename")

	if err := os.Rename(destinationPath, sourcePath); err != nil {
		t.Fatalf("rename %s to %s: %v", destinationPath, sourcePath, err)
	}
	harness.waitForRenameConverge(directory, sourceName)
	harness.requireCurrentPathOnly(directory, sourceName, renamedSourceName, "move back")
	after := metrics.Read()

	copied := after.ConvergeCopyChunksTotal - before.ConvergeCopyChunksTotal
	upserted := after.ConvergeUpsertTotal - before.ConvergeUpsertTotal
	if copied != 2 {
		t.Fatalf("The two rename converges copied chunks for %d paths and upserted %d paths. The test expected 2 copied paths.", copied, upserted)
	}
	if embedded := after.EmbedVectorsTotal - before.EmbedVectorsTotal; embedded != 0 {
		t.Fatalf("The embedder received %d inputs during the two rename converges. The test expected 0 inputs.", embedded)
	}
}

// requireCurrentPathOnly reads the live Merkle snapshot of the codebase at
// directory and fails if the snapshot does not record current or records
// previous. It searches for the generated source text and fails if any result
// has the previous path or if the search does not return the current path.
// The step argument labels failure messages.
func (harness *harness) requireCurrentPathOnly(directory string, current string, previous string, step string) {
	harness.t.Helper()

	snapshotPath := harness.indexStatusAt(directory).GetCodebase().GetMerkleSnapshotPath()
	snapshot, err := merkle.ReadSnapshot(snapshotPath)
	if err != nil {
		harness.t.Fatalf("read the live Merkle snapshot %s after the %s: %v", snapshotPath, step, err)
	}
	if !snapshot.HasFile(current) || snapshot.HasFile(previous) {
		harness.t.Fatalf("After the %s, the live Merkle snapshot records %t for %s and %t for %s. The test expected only %s.", step, snapshot.HasFile(current), current, snapshot.HasFile(previous), previous, current)
	}

	query := fmt.Sprintf("Unit%04d returns the running total of %d readings", renamedSourceIndex, renamedSourceIndex)
	search := harness.searchAt(directory, query)
	foundCurrent := false
	for _, result := range search.GetResults() {
		switch result.GetRelativePath() {
		case previous:
			harness.t.Fatalf("After the %s, the search returned the old path %s:\n%s", step, previous, search.GetDisplayText())
		case current:
			foundCurrent = true
		}
	}
	if !foundCurrent {
		harness.t.Fatalf("After the %s, the search did not return path %s:\n%s", step, current, search.GetDisplayText())
	}
}

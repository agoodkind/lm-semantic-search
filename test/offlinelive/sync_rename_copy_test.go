//go:build offlinelive

package offlinelive

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/metrics"
	"goodkind.io/lm-semantic-search/internal/store"
)

// TestSweepSyncCopiesRenamedSource indexes a directory with the file watcher
// off, renames one tracked file, and waits for the sync job that the periodic
// sweep starts. It checks that the sync does not call the embedder and that the
// live Merkle snapshot records the new path without the source path. It reads
// the chunk cache file and checks that it contains chunks for the new path and
// does not contain chunks for the old path. It also checks that a search returns
// the new path.
func TestSweepSyncCopiesRenamedSource(t *testing.T) {
	harness := newHarnessWith(t, harnessOptions{
		fileWatcher:            false,
		backgroundSync:         true,
		maxConcurrentIndexJobs: 0,
		syncIntervalMS:         repeatedSweepIntervalMS,
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

	codebaseID := harness.indexStatusAt(directory).GetCodebase().GetId()
	earlierJobs := harness.jobIDsFor(codebaseID)
	before := metrics.Read()
	sourceName := generatedSourceName(renamedSourceIndex)
	if err := os.Rename(filepath.Join(directory, sourceName), filepath.Join(directory, renamedSourceName)); err != nil {
		t.Fatalf("rename %s to %s: %v", sourceName, renamedSourceName, err)
	}
	harness.waitForScheduledSync(codebaseID, earlierJobs)
	after := metrics.Read()

	if embedded := after.EmbedVectorsTotal - before.EmbedVectorsTotal; embedded != 0 {
		t.Fatalf("The embedder received %d inputs during the sweep sync of the rename. The test expected 0 inputs.", embedded)
	}
	snapshotPath := harness.indexStatusAt(directory).GetCodebase().GetMerkleSnapshotPath()
	snapshot, err := merkle.ReadSnapshot(snapshotPath)
	if err != nil {
		t.Fatalf("read the live Merkle snapshot %s after the sweep sync: %v", snapshotPath, err)
	}
	if !snapshot.HasFile(renamedSourceName) || snapshot.HasFile(sourceName) {
		t.Fatalf("The live Merkle snapshot records %t for %s and %t for %s. The test expected only the new path.", snapshot.HasFile(renamedSourceName), renamedSourceName, snapshot.HasFile(sourceName), sourceName)
	}
	chunkPath := filepath.Join(harness.config.ChunksDir, codebaseID+".json")
	chunks, err := store.ReadChunks(chunkPath)
	if err != nil {
		t.Fatalf("read the chunk cache %s after the sweep sync: %v", chunkPath, err)
	}
	cachedPaths := make(map[string]int, len(chunks))
	for _, chunk := range chunks {
		cachedPaths[chunk.RelativePath]++
	}
	if cachedPaths[renamedSourceName] == 0 || cachedPaths[sourceName] != 0 {
		t.Fatalf("The chunk cache %s records %d chunks for %s and %d chunks for %s. The test expected chunks only for the new path.", chunkPath, cachedPaths[renamedSourceName], renamedSourceName, cachedPaths[sourceName], sourceName)
	}

	query := fmt.Sprintf("Unit%04d returns the running total of %d readings", renamedSourceIndex, renamedSourceIndex)
	search := harness.searchAt(directory, query)
	for _, result := range search.GetResults() {
		if result.GetRelativePath() == renamedSourceName {
			return
		}
	}
	t.Fatalf("The search for the renamed source did not return path %s:\n%s", renamedSourceName, search.GetDisplayText())
}

//go:build offlinelive

package offlinelive

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/metrics"
)

const (
	// renamedSourceIndex is the generated source that the test renames.
	renamedSourceIndex = 0

	// renamedSourceName is the path of that source after the rename. The path is
	// in the codebase root, and the rename creates no directory.
	renamedSourceName = "renamed_unit0000.go"

	// renameConvergeTimeout bounds the wait for the converge that the file
	// watcher starts after the rename.
	renameConvergeTimeout = 30 * time.Second

	// editedSourceIndex is the generated source that the test edits before the
	// periodic sweep. It differs from renamedSourceIndex.
	editedSourceIndex = 1

	// editedSourceContent replaces the content of the edited source.
	editedSourceContent = "package pkg\n\n// Unit0001 returns the edited total of its readings.\nfunc Unit0001(readings []int) int {\n\treturn len(readings)\n}\n"

	// scheduledSyncClient is the client name of a sync job that the periodic
	// sweep starts.
	scheduledSyncClient = "daemon-sync"

	// scheduledSyncTimeout bounds the wait for the sweep to start a sync job
	// after the edit. The first sweep runs a few seconds after the daemon starts,
	// and later sweeps run every repeatedSweepIntervalMS.
	scheduledSyncTimeout = 30 * time.Second
)

// TestRenameAfterFullBuildAndSyncCopiesChunks indexes a directory with a full
// build, edits one tracked file, and waits for the sync job that the periodic
// sweep starts. It restarts the daemon with the file watcher on, renames
// another tracked file, and waits for the watcher converge. The live Merkle
// snapshot must record an inode entry for each file after the full build, after
// the sweep sync job, and after the restart. The converge must copy the stored
// chunks to the new path without an embedder call, and a search must return
// the new path.
func TestRenameAfterFullBuildAndSyncCopiesChunks(t *testing.T) {
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
	harness.requireInodeEntryPerFile(directory, "full build")

	codebaseID := harness.indexStatusAt(directory).GetCodebase().GetId()
	earlierJobs := harness.jobIDsFor(codebaseID)
	editedPath := filepath.Join(directory, generatedSourceName(editedSourceIndex))
	if err := os.WriteFile(editedPath, []byte(editedSourceContent), 0o644); err != nil {
		t.Fatalf("edit %s: %v", editedPath, err)
	}
	harness.waitForScheduledSync(codebaseID, earlierJobs)
	harness.requireInodeEntryPerFile(directory, "periodic sweep sync job")

	harness.restart(harnessOptions{fileWatcher: true, backgroundSync: false, maxConcurrentIndexJobs: 0})
	harness.requireInodeEntryPerFile(directory, "daemon restart")

	before := metrics.Read()
	sourcePath := filepath.Join(directory, generatedSourceName(renamedSourceIndex))
	destinationPath := filepath.Join(directory, renamedSourceName)
	if err := os.Rename(sourcePath, destinationPath); err != nil {
		t.Fatalf("rename %s to %s: %v", sourcePath, destinationPath, err)
	}
	harness.waitForRenameConverge(directory, before)
	after := metrics.Read()

	copied := after.ConvergeCopyChunksTotal - before.ConvergeCopyChunksTotal
	upserted := after.ConvergeUpsertTotal - before.ConvergeUpsertTotal
	if copied != 1 {
		t.Fatalf("The rename converge copied chunks for %d paths and upserted %d paths. The test expected 1 copied path.", copied, upserted)
	}
	if embedded := after.EmbedVectorsTotal - before.EmbedVectorsTotal; embedded != 0 {
		t.Fatalf("The embedder received %d inputs during the rename converge. The test expected 0 inputs.", embedded)
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

// requireInodeEntryPerFile reads the live Merkle snapshot of the codebase at
// directory and fails unless the snapshot records one inode entry for each
// recorded file. A full build writes its staging checkpoint and then renames
// that file to the live path. The check after a full build therefore reads the
// staging checkpoint content. The job argument is the label of the job that
// wrote the snapshot in the log line and the failure messages.
func (harness *harness) requireInodeEntryPerFile(directory string, job string) {
	harness.t.Helper()

	snapshotPath := harness.indexStatusAt(directory).GetCodebase().GetMerkleSnapshotPath()
	snapshot, err := merkle.ReadSnapshot(snapshotPath)
	if err != nil {
		harness.t.Fatalf("read the live Merkle snapshot %s after the %s: %v", snapshotPath, job, err)
	}
	harness.t.Logf("The live Merkle snapshot %s after the %s records %d files and %d inode entries.", snapshotPath, job, len(snapshot.Files), len(snapshot.Inodes))
	if len(snapshot.Files) == 0 {
		harness.t.Fatalf("The live Merkle snapshot after the %s records 0 files.", job)
	}
	for relativePath := range snapshot.Files {
		if _, found := snapshot.Inodes[relativePath]; !found {
			harness.t.Fatalf("The live Merkle snapshot after the %s records file %s without an inode entry. It records %d files and %d inode entries.", job, relativePath, len(snapshot.Files), len(snapshot.Inodes))
		}
	}
}

// waitForScheduledSync waits for the first job of the codebase that is not in
// earlierJobs and waits for that job to finish. The job must complete, and
// every new job of the codebase must have the scheduledSyncClient client.
func (harness *harness) waitForScheduledSync(codebaseID string, earlierJobs map[string]struct{}) {
	harness.t.Helper()

	waitContext, cancel := context.WithTimeout(context.Background(), scheduledSyncTimeout)
	defer cancel()
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()
	for {
		if job, started := harness.newJobFor(codebaseID, earlierJobs); started {
			requireCompleted(harness.t, harness.waitForJob(job.GetId()))
			break
		}
		select {
		case <-waitContext.Done():
			harness.t.Fatalf("The periodic sweep did not start a sync job for codebase %s within %s.", codebaseID, scheduledSyncTimeout)
		case <-ticker.C:
		}
	}
	for _, job := range harness.jobsFor(codebaseID) {
		if _, earlier := earlierJobs[job.GetId()]; earlier {
			continue
		}
		if client := job.GetClient().GetName(); client != scheduledSyncClient {
			harness.t.Fatalf("Job %s of codebase %s has client %q. The test expected only %q jobs after the edit.", job.GetId(), codebaseID, client, scheduledSyncClient)
		}
	}
}

// waitForRenameConverge waits until a converge after before records a chunk
// copy or upsert for a path and the codebase at directory has no active job.
func (harness *harness) waitForRenameConverge(directory string, before metrics.Snapshot) {
	harness.t.Helper()

	waitContext, cancel := context.WithTimeout(context.Background(), renameConvergeTimeout)
	defer cancel()
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()
	for {
		current := metrics.Read()
		converged := current.ConvergeCopyChunksTotal > before.ConvergeCopyChunksTotal ||
			current.ConvergeUpsertTotal > before.ConvergeUpsertTotal
		status := harness.indexStatusAt(directory)
		if converged && status.GetActiveJob() == nil {
			return
		}
		select {
		case <-waitContext.Done():
			harness.t.Fatalf("The watcher did not converge the renamed source within %s:\n%s", renameConvergeTimeout, status.GetDisplayText())
		case <-ticker.C:
		}
	}
}

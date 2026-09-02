package daemon

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

// The live chunk count is read from the collection a promoted run created, and
// a run creates one only when it committed files. Keying the gate on the run
// record alone let two shapes through to a collection that was never created,
// and each row count against it writes a "count collection rows failed" line
// for the whole life of the job.
//
// The cases below pair each skip with the query that must still happen, so
// neither half can pass by doing nothing: liveCollection proves the gate lets a
// real collection through, and would fail if the gate were simply always closed.
func TestFillLiveChunkTotalQueriesOnlyACollectionThatWasCreated(t *testing.T) {
	manager, _, repoPath := newTestManager(t)

	committedRun := &model.IndexRunSummary{IndexedFiles: 4, TotalChunks: 120, Status: "completed"}
	zeroFileRun := &model.IndexRunSummary{IndexedFiles: 0, TotalChunks: 0, Status: "completed"}

	cases := []struct {
		name          string
		lastRun       *model.IndexRunSummary
		status        model.CodebaseStatus
		operation     string
		wantCalls     int32
		wantChunkText string
	}{
		{
			// The first build writes to staging and promotes at the end, so there is
			// no live collection to count while it runs.
			name:      "first build has no promoted collection",
			lastRun:   nil,
			status:    model.CodebaseStatusIndexing,
			operation: "index",
			wantCalls: 0,
		},
		{
			// A run that indexed no file promoted nothing, so no collection exists
			// however many later jobs the codebase runs.
			name:      "zero file run created no collection",
			lastRun:   zeroFileRun,
			status:    model.CodebaseStatusIndexing,
			operation: "index",
			wantCalls: 0,
		},
		{
			// The operation clause used to be the only thing holding this shut, so a
			// watcher-fired sync on the same codebase walked straight past it.
			name:      "zero file run followed by a sync",
			lastRun:   zeroFileRun,
			status:    model.CodebaseStatusIndexing,
			operation: "sync",
			wantCalls: 0,
		},
		{
			// A forced reindex over a real index counts the live collection that is
			// still serving while the staging build runs.
			name:      "reindex over a committed run counts the live collection",
			lastRun:   committedRun,
			status:    model.CodebaseStatusIndexing,
			operation: "streaming_reindex",
			wantCalls: 1,
		},
		{
			// Adoption takes a path whose collection already exists and records it as
			// indexed without a run summary, so this shape owns rows to count while
			// carrying the same empty run record a first build carries. Reading the
			// run record alone left an adopted codebase uncounted until its first
			// sync finished.
			name:      "adopted codebase counts the collection it was adopted for",
			lastRun:   nil,
			status:    model.CodebaseStatusIndexed,
			operation: "sync",
			wantCalls: 1,
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			var countCalls atomic.Int32
			manager.semantic = &fakeSemantic{
				count: func(context.Context, string) (int32, error) {
					countCalls.Add(1)
					return 512, nil
				},
			}
			codebase := model.Codebase{
				ID:                "cb_" + testCase.name,
				CanonicalPath:     repoPath,
				Status:            testCase.status,
				LastSuccessfulRun: testCase.lastRun,
			}
			job := model.Job{
				ID:            "job_" + testCase.name,
				CanonicalPath: repoPath,
				Operation:     testCase.operation,
				State:         model.JobStateRunning,
				Progress:      model.Progress{RunMode: model.RunModeFirstBuild},
			}

			logs := captureLogs(t)
			manager.fillLiveChunkTotal(context.Background(), codebase, &job)

			if got := countCalls.Load(); got != testCase.wantCalls {
				t.Fatalf("collection row count ran %d time(s), want %d", got, testCase.wantCalls)
			}
			if testCase.wantCalls == 0 {
				if job.Progress.ChunksTotal != 0 {
					t.Fatalf("ChunksTotal = %d, want 0 for a codebase with no live collection", job.Progress.ChunksTotal)
				}
				if len(logs.linesContaining("level=DEBUG", "no live collection to count yet", codebase.ID)) == 0 {
					t.Fatalf("the skipped count was not explained in the daemon log:\n%s", logs.text())
				}
				return
			}
			if job.Progress.ChunksTotal != 512 {
				t.Fatalf("ChunksTotal = %d, want the live collection's 512", job.Progress.ChunksTotal)
			}
			if lines := logs.linesContaining("no live collection to count yet", codebase.ID); len(lines) > 0 {
				t.Fatalf("a codebase with a live collection was reported as having none:\n%s", strings.Join(lines, "\n"))
			}
		})
	}
}

// The run-completion totals read the collection too, on a path the status read
// does not share, so the same absent collection produced the same false fault
// there. A run that committed no file has nothing to count, and the pair below
// keeps the skip from being an always-closed gate.
func TestCodebaseTotalsQueriesOnlyACollectionThatWasCreated(t *testing.T) {
	manager, _, repoPath := newTestManager(t)

	cases := []struct {
		name       string
		working    map[string]string
		wantCalls  int32
		wantChunks int32
	}{
		{
			// A run whose working set is empty committed no file, so it promoted no
			// collection and the fallback total is already the whole answer.
			name:       "run that committed no file is not counted",
			working:    map[string]string{},
			wantCalls:  0,
			wantChunks: 7,
		},
		{
			// A run that committed files promoted a collection, and its live row
			// count replaces the running fallback.
			name:       "run that committed files counts the live collection",
			working:    map[string]string{"main.go": "hash"},
			wantCalls:  1,
			wantChunks: 512,
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			var countCalls atomic.Int32
			manager.semantic = &fakeSemantic{
				count: func(context.Context, string) (int32, error) {
					countCalls.Add(1)
					return 512, nil
				},
			}

			_, chunkCount := manager.codebaseTotals(context.Background(), repoPath, testCase.working, 7)

			if got := countCalls.Load(); got != testCase.wantCalls {
				t.Fatalf("collection row count ran %d time(s), want %d", got, testCase.wantCalls)
			}
			if chunkCount != testCase.wantChunks {
				t.Fatalf("chunk total = %d, want %d", chunkCount, testCase.wantChunks)
			}
		})
	}
}

// indexEmptyCodebase indexes a directory with no indexable file and returns the
// completed record plus the checkpoint path it never wrote. It fails the test
// unless the run really did land at indexed with zero files and no checkpoint,
// so a later quiet-log assertion cannot pass because the setup was wrong.
func indexEmptyCodebase(t *testing.T, manager *Manager) (model.Codebase, string) {
	t.Helper()

	emptyRepo := filepath.Join(t.TempDir(), "empty-repo")
	if err := os.MkdirAll(emptyRepo, 0o755); err != nil {
		t.Fatalf("MkdirAll returned error: %v", err)
	}

	if _, _, _, _, err := manager.StartIndex(context.Background(), emptyRepo, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget); err != nil {
		t.Fatalf("StartIndex returned error: %v", err)
	}
	waitForCodebaseStatus(t, manager, emptyRepo, model.CodebaseStatusIndexed)

	codebase, _, found, _, err := manager.GetIndex(context.Background(), emptyRepo)
	if err != nil || !found {
		t.Fatalf("GetIndex returned err=%v found=%v", err, found)
	}
	if codebase.LastSuccessfulRun == nil {
		t.Fatal("LastSuccessfulRun is nil; the empty codebase did not record a completed run")
	}
	if codebase.LastSuccessfulRun.IndexedFiles != 0 {
		t.Fatalf("LastSuccessfulRun.IndexedFiles = %d, want 0 for an empty codebase", codebase.LastSuccessfulRun.IndexedFiles)
	}
	snapshotPath := manager.snapshotPathForCodebase(codebase)
	if _, statErr := os.Stat(snapshotPath); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("empty codebase already owns a checkpoint at %s (%v); the guard would pass vacuously", snapshotPath, statErr)
	}
	return codebase, snapshotPath
}

// A successful run that indexed zero files writes no checkpoint, so the file it
// never wrote must not read as a lost index. The watcher drives ConvergePaths on
// file activity rather than once at boot, so a required read here would repeat
// "this index lost state" for as long as the empty codebase stays tracked.
func TestConvergePathsKeepsEmptyCodebaseQuietAboutItsAbsentCheckpoint(t *testing.T) {
	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{}

	codebase, snapshotPath := indexEmptyCodebase(t, manager)

	logs := captureLogs(t)
	if _, err := manager.ConvergePaths(context.Background(), codebase.ID, []string{"new.go"}, nil); err != nil {
		t.Fatalf("ConvergePaths returned error: %v", err)
	}

	if lines := logs.linesContaining("read Merkle snapshot failed", snapshotPath); len(lines) > 0 {
		t.Fatalf("a healthy empty codebase was reported as having lost its index:\n%s", strings.Join(lines, "\n"))
	}
}

// The sweep runs every five minutes over every indexed codebase, so a wrong
// verdict here repeats forever rather than once. An empty codebase must come
// back unchanged: its absent checkpoint is the empty one it would have written,
// which matches the empty capture, so the sweep neither reports damage nor
// enqueues a sync for a codebase where nothing happened.
func TestRunSyncAllKeepsEmptyCodebaseQuietAndEnqueuesNothing(t *testing.T) {
	manager, cfg, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{
		listCollections:      func(context.Context) ([]string, error) { return []string{}, nil },
		hasCollectionForPath: func(context.Context, string) (bool, error) { return false, nil },
	}

	codebase, snapshotPath := indexEmptyCodebase(t, manager)
	jobsBefore := len(manager.ListJobs(""))

	logs := captureLogs(t)
	syncer := NewBackgroundSync(cfg, manager)
	syncer.runSyncAll(context.Background(), "test")
	syncer.runSyncAll(context.Background(), "test")

	if lines := logs.linesContaining("read Merkle snapshot failed", snapshotPath); len(lines) > 0 {
		t.Fatalf("the sweep reported a healthy empty codebase as having lost its index:\n%s", strings.Join(lines, "\n"))
	}
	if lines := logs.linesContaining("level=ERROR", codebase.ID); len(lines) > 0 {
		t.Fatalf("the sweep reported an error against a healthy empty codebase:\n%s", strings.Join(lines, "\n"))
	}
	if jobsAfter := len(manager.ListJobs("")); jobsAfter != jobsBefore {
		t.Fatalf("the sweep started %d job(s) for a codebase where nothing changed, want 0", jobsAfter-jobsBefore)
	}
}

// doctor is the operator's own read of daemon health, so the same absent
// checkpoint must not turn a healthy empty repository into a standing
// diagnostic there either.
func TestDoctorKeepsEmptyCodebaseOutOfTheDiagnosticList(t *testing.T) {
	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{}

	codebase, _ := indexEmptyCodebase(t, manager)

	for _, diagnostic := range manager.Doctor(context.Background()) {
		if strings.Contains(diagnostic, codebase.CanonicalPath) {
			t.Fatalf("doctor reported a healthy empty codebase as a problem: %q", diagnostic)
		}
	}
}

// The quiet-absence rule must not have gone so far that a real loss is silent.
// A codebase whose last run indexed files and whose checkpoint then vanished is
// the one shape where the sweep has to say so.
func TestRunSyncAllReportsACheckpointLostAfterACommittedRun(t *testing.T) {
	manager, cfg, repoPath := newTestManager(t)
	// The collection is present, as it is for any codebase that really indexed
	// files, so the repair pass stays inert and the sweep reaches the checkpoint
	// read this test is about.
	manager.semantic = &fakeSemantic{}

	if _, _, _, _, err := manager.StartIndex(context.Background(), repoPath, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget); err != nil {
		t.Fatalf("StartIndex returned error: %v", err)
	}
	waitForCodebaseStatus(t, manager, repoPath, model.CodebaseStatusIndexed)

	codebase, _, found, _, err := manager.GetIndex(context.Background(), repoPath)
	if err != nil || !found {
		t.Fatalf("GetIndex returned err=%v found=%v", err, found)
	}
	if codebase.LastSuccessfulRun == nil || codebase.LastSuccessfulRun.IndexedFiles == 0 {
		t.Fatalf("LastSuccessfulRun = %+v, want a run that indexed files", codebase.LastSuccessfulRun)
	}
	snapshotPath := manager.snapshotPathForCodebase(codebase)
	if _, statErr := os.Stat(snapshotPath); statErr != nil {
		t.Fatalf("the committed run wrote no checkpoint at %s (%v); the loss below would not be a loss", snapshotPath, statErr)
	}
	if err := os.Remove(snapshotPath); err != nil {
		t.Fatalf("Remove returned error: %v", err)
	}

	logs := captureLogs(t)
	syncer := NewBackgroundSync(cfg, manager)
	syncer.runSyncAll(context.Background(), "test")

	if len(logs.linesContaining("level=ERROR", "read Merkle snapshot failed", snapshotPath)) == 0 {
		t.Fatalf("a checkpoint lost after a committed run was not reported:\n%s", logs.text())
	}
	// The sweep also starts the resync the loss calls for. Let it finish before
	// the test returns, so its writes do not race the temp-directory cleanup.
	waitForCodebaseStatus(t, manager, repoPath, model.CodebaseStatusIndexed)
	waitForCondition(t, func() bool {
		for _, job := range manager.ListJobs("") {
			if !isTerminalJobState(job.State) {
				return false
			}
		}
		return true
	})
}

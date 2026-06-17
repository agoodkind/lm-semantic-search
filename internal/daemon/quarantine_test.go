package daemon

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"

	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
)

func writeSyntheticSnapshot(t *testing.T, manager *Manager, codebase model.Codebase, fileCount int) {
	t.Helper()
	files := make(map[string]string, fileCount)
	for i := 0; i < fileCount; i++ {
		files[fmt.Sprintf("f%03d.go", i)] = fmt.Sprintf("hash-%03d", i)
	}
	snapshot := merkle.Snapshot{
		ConfigDigest: codebase.EffectiveConfig.IgnoreDigest,
		Files:        files,
		Inodes:       nil,
	}
	if err := merkle.WriteSnapshot(manager.snapshotPathForCodebase(codebase), snapshot); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}
}

func TestConvergePathsRootMissingMarksMissingAndSkipsDelete(t *testing.T) {
	t.Parallel()

	manager, _, repoPath := newTestManager(t)
	indexConfig := defaultIndexConfig()
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)
	codebase := newCodebaseRecord(repoPath)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = indexConfig
	codebase.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 1}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	writeSyntheticSnapshot(t, manager, codebase, 1)

	var reindexCalls atomic.Int32
	manager.semantic = &fakeSemantic{
		reindex: func(_ context.Context, _ string, _ []model.StoredChunk, _ []string) error {
			reindexCalls.Add(1)
			return nil
		},
	}

	if err := os.RemoveAll(repoPath); err != nil {
		t.Fatalf("RemoveAll returned error: %v", err)
	}
	if err := manager.ConvergePaths(context.Background(), codebase.ID, []string{"f000.go"}); err != nil {
		t.Fatalf("ConvergePaths returned error: %v", err)
	}

	manager.mu.Lock()
	readCodebase, found := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if !found {
		t.Fatalf("tracked codebase %s disappeared", codebase.ID)
	}
	if readCodebase.Status != model.CodebaseStatusMissing {
		t.Fatalf("status = %q, want missing", readCodebase.Status)
	}
	if got := reindexCalls.Load(); got != 0 {
		t.Fatalf("reindex calls = %d, want 0 when the root disappears", got)
	}

	snapshot, err := merkle.ReadSnapshot(manager.snapshotPathForCodebase(codebase))
	if err != nil {
		t.Fatalf("ReadSnapshot returned error: %v", err)
	}
	if len(snapshot.Files) != 1 {
		t.Fatalf("snapshot files = %d, want 1 unchanged entry after missing-root hold", len(snapshot.Files))
	}
}

func TestConvergePathsLargeWatcherDeleteQuarantines(t *testing.T) {
	t.Parallel()

	manager, _, repoPath := newTestManager(t)
	indexConfig := defaultIndexConfig()
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)
	codebase := newCodebaseRecord(repoPath)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = indexConfig
	codebase.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 120, TotalChunks: 120}
	codebase.LiveFileTotal = 120
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	writeSyntheticSnapshot(t, manager, codebase, 120)

	var reindexCalls atomic.Int32
	manager.semantic = &fakeSemantic{
		reindex: func(_ context.Context, _ string, _ []model.StoredChunk, _ []string) error {
			reindexCalls.Add(1)
			return nil
		},
	}

	relativePaths := make([]string, 0, 110)
	for i := 0; i < 110; i++ {
		relativePaths = append(relativePaths, fmt.Sprintf("f%03d.go", i))
	}
	if err := manager.ConvergePaths(context.Background(), codebase.ID, relativePaths); err != nil {
		t.Fatalf("ConvergePaths returned error: %v", err)
	}

	manager.mu.Lock()
	readCodebase, found := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if !found {
		t.Fatalf("tracked codebase %s disappeared", codebase.ID)
	}
	if readCodebase.Status != model.CodebaseStatusQuarantined {
		t.Fatalf("status = %q, want quarantined", readCodebase.Status)
	}
	if readCodebase.Quarantine == nil {
		t.Fatal("Quarantine = nil, want recorded quarantine state")
	}
	if readCodebase.Quarantine.LastMissingCount != 110 || readCodebase.Quarantine.LastTotalCount != 120 {
		t.Fatalf("quarantine counts = %+v, want 110 of 120", readCodebase.Quarantine)
	}
	if got := reindexCalls.Load(); got != 0 {
		t.Fatalf("reindex calls = %d, want 0 while the watcher delete wave is quarantined", got)
	}

	snapshot, err := merkle.ReadSnapshot(manager.snapshotPathForCodebase(codebase))
	if err != nil {
		t.Fatalf("ReadSnapshot returned error: %v", err)
	}
	if len(snapshot.Files) != 120 {
		t.Fatalf("snapshot files = %d, want 120 unchanged entries while quarantined", len(snapshot.Files))
	}
}

func TestSyncIndexLargeRemovalQuarantinesBeforeDelete(t *testing.T) {
	t.Parallel()

	manager, _, repoPath := newTestManager(t)
	indexConfig := defaultIndexConfig()

	for i := 0; i < 119; i++ {
		path := filepath.Join(repoPath, fmt.Sprintf("f%03d.go", i))
		if err := os.WriteFile(path, []byte("package main\n"), 0o644); err != nil {
			t.Fatalf("WriteFile(%s) returned error: %v", path, err)
		}
	}

	var reindexCalls atomic.Int32
	manager.semantic = &fakeSemantic{
		reindex: func(_ context.Context, _ string, _ []model.StoredChunk, _ []string) error {
			reindexCalls.Add(1)
			return nil
		},
	}

	if _, _, _, _, err := manager.StartIndex(context.Background(), repoPath, testClientInfo(), indexConfig, false); err != nil {
		t.Fatalf("StartIndex returned error: %v", err)
	}
	indexed := waitForCodebaseSettled(t, manager, repoPath)
	reindexCalls.Store(0)

	for i := 0; i < 110; i++ {
		if err := os.Remove(filepath.Join(repoPath, fmt.Sprintf("f%03d.go", i))); err != nil {
			t.Fatalf("Remove returned error: %v", err)
		}
	}

	job, _, _, err := manager.SyncIndex(context.Background(), repoPath, testClientInfo())
	if err != nil {
		t.Fatalf("SyncIndex returned error: %v", err)
	}
	waitForCondition(t, func() bool {
		codebase, _, found, _, getErr := manager.GetIndex(context.Background(), repoPath)
		if getErr != nil || !found {
			return false
		}
		return codebase.Status == model.CodebaseStatusQuarantined
	})

	quarantinedCodebase, _, found, _, err := manager.GetIndex(context.Background(), repoPath)
	if err != nil || !found {
		t.Fatalf("GetIndex returned err=%v found=%v", err, found)
	}
	if quarantinedCodebase.Quarantine == nil {
		t.Fatal("Quarantine = nil, want recorded quarantine state")
	}
	if quarantinedCodebase.Quarantine.LastMissingCount != 110 || quarantinedCodebase.Quarantine.LastTotalCount != 120 {
		t.Fatalf("quarantine counts = %+v, want 110 of 120", quarantinedCodebase.Quarantine)
	}
	recordedJob, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("GetJob(%s) did not find the quarantined sync job", job.ID)
	}
	if recordedJob.State != model.JobStateFailed {
		t.Fatalf("job state = %q, want failed terminal quarantine job", recordedJob.State)
	}
	if recordedJob.Error == nil || recordedJob.Error.Message == "" {
		t.Fatal("quarantine job missing error message")
	}
	if got := reindexCalls.Load(); got != 0 {
		t.Fatalf("reindex calls = %d, want 0 before quarantine release", got)
	}

	snapshot, err := merkle.ReadSnapshot(manager.snapshotPathForCodebase(indexed))
	if err != nil {
		t.Fatalf("ReadSnapshot returned error: %v", err)
	}
	if len(snapshot.Files) != 120 {
		t.Fatalf("snapshot files = %d, want 120 unchanged entries after quarantine", len(snapshot.Files))
	}
}

func TestHandleQuarantinedCodebaseClearsOnEmptyDiff(t *testing.T) {
	t.Parallel()

	manager, cfg, repoPath := newTestManager(t)
	indexConfig := defaultIndexConfig()
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)
	codebase := newCodebaseRecord(repoPath)
	codebase.Status = model.CodebaseStatusQuarantined
	codebase.EffectiveConfig = indexConfig
	codebase.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 1}
	codebase.Quarantine = &model.QuarantineState{
		Reason:           quarantineReasonWatcherLargeDelete,
		ObservationCount: 1,
		LastTrigger:      quarantineTriggerWatcher,
		LastMissingCount: 110,
		LastTotalCount:   120,
	}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	content := "package main\n"
	if err := os.WriteFile(filepath.Join(repoPath, "main.go"), []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	snapshot := merkle.Snapshot{
		ConfigDigest: indexConfig.IgnoreDigest,
		Files:        map[string]string{"main.go": hashText(content)},
		Inodes:       nil,
	}
	if err := merkle.WriteSnapshot(manager.snapshotPathForCodebase(codebase), snapshot); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	syncer := NewBackgroundSync(cfg, manager)
	syncer.handleQuarantinedCodebase(context.Background(), codebase)

	manager.mu.Lock()
	readCodebase, found := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if !found {
		t.Fatalf("tracked codebase %s disappeared", codebase.ID)
	}
	if readCodebase.Status != model.CodebaseStatusIndexed {
		t.Fatalf("status = %q, want indexed after empty diff", readCodebase.Status)
	}
	if readCodebase.Quarantine != nil {
		t.Fatalf("Quarantine = %+v, want cleared after empty diff", readCodebase.Quarantine)
	}
}

func TestAssessDeltaDeleteWaveFlagsLargeRemoval(t *testing.T) {
	t.Parallel()

	codebase := model.Codebase{
		Kind:              model.CodebaseKindCode,
		Status:            model.CodebaseStatusIndexed,
		LastSuccessfulRun: &model.IndexRunSummary{IndexedFiles: 120, TotalChunks: 120},
		LiveFileTotal:     120,
	}
	snapshot := merkle.Snapshot{Files: make(map[string]string, 120)}
	for i := 0; i < 120; i++ {
		snapshot.Files[fmt.Sprintf("f%03d.go", i)] = fmt.Sprintf("hash-%03d", i)
	}
	diff := merkle.Diff{Added: nil, Modified: nil, Removed: make([]string, 110)}
	for i := 0; i < 110; i++ {
		diff.Removed[i] = fmt.Sprintf("f%03d.go", i)
	}
	signal, suspicious := assessDeltaDeleteWave(codebase, diff, snapshot)
	if !suspicious {
		t.Fatal("assessDeltaDeleteWave returned suspicious=false, want true for 110 of 120 removed")
	}
	if signal.missingCount != 110 || signal.totalCount != 120 {
		t.Fatalf("signal = %+v, want 110 of 120", signal)
	}
}

func TestAssessDeltaDeleteWaveSkipsAfterFullScanConfirmation(t *testing.T) {
	t.Parallel()

	codebase := model.Codebase{
		Kind:              model.CodebaseKindCode,
		Status:            model.CodebaseStatusQuarantined,
		LastSuccessfulRun: &model.IndexRunSummary{IndexedFiles: 120, TotalChunks: 120},
		LiveFileTotal:     120,
		Quarantine: &model.QuarantineState{
			ObservationCount: quarantineConfirmationObservations,
			LastTrigger:      quarantineTriggerFullScan,
		},
	}
	snapshot := merkle.Snapshot{Files: make(map[string]string, 120)}
	for i := 0; i < 120; i++ {
		snapshot.Files[fmt.Sprintf("f%03d.go", i)] = fmt.Sprintf("hash-%03d", i)
	}
	diff := merkle.Diff{Added: nil, Modified: nil, Removed: make([]string, 110)}
	for i := 0; i < 110; i++ {
		diff.Removed[i] = fmt.Sprintf("f%03d.go", i)
	}
	if _, suspicious := assessDeltaDeleteWave(codebase, diff, snapshot); suspicious {
		t.Fatal("assessDeltaDeleteWave returned suspicious=true after confirmation, want false so destructive sync can proceed")
	}
}

// TestAssessWatcherDeleteWaveClydeDesktopScenario reproduces the exact failure
// that motivated this fix: a healthy repo whose 151 tracked files are all
// present, while a flood of untracked .git churn paths are absent. The raw
// batch carries 1,671 absent untracked paths, which the old code counted
// against the 151 tracked total to produce the impossible "1,671 of 151" and a
// false quarantine.
func TestAssessWatcherDeleteWaveClydeDesktopScenario(t *testing.T) {
	t.Parallel()
	root := t.TempDir()

	const trackedCount = 151
	files := make(map[string]string, trackedCount)
	trackedPaths := make([]string, 0, trackedCount)
	if err := os.MkdirAll(filepath.Join(root, "src"), 0o755); err != nil {
		t.Fatalf("MkdirAll returned error: %v", err)
	}
	for i := 0; i < trackedCount; i++ {
		rel := fmt.Sprintf("src/f%03d.go", i)
		files[rel] = fmt.Sprintf("hash-%03d", i)
		trackedPaths = append(trackedPaths, rel)
		if err := os.WriteFile(filepath.Join(root, rel), []byte("package main\n"), 0o644); err != nil {
			t.Fatalf("WriteFile(%s) returned error: %v", rel, err)
		}
	}
	snapshot := merkle.Snapshot{Files: files}

	batch := append([]string{}, trackedPaths...)
	for i := 0; i < 1671; i++ {
		batch = append(batch, fmt.Sprintf(".git/objects/%02x/%04x.pack", i%256, i))
	}

	codebase := model.Codebase{
		Kind:              model.CodebaseKindCode,
		Status:            model.CodebaseStatusIndexed,
		LastSuccessfulRun: &model.IndexRunSummary{IndexedFiles: trackedCount, TotalChunks: trackedCount},
		LiveFileTotal:     trackedCount,
	}

	signal, suspicious := assessWatcherDeleteWave(codebase, snapshot, root, batch)
	if suspicious {
		t.Fatalf("assessWatcherDeleteWave quarantined a healthy repo: signal = %+v; untracked churn must not count toward the tracked-file total", signal)
	}
	if signal.missingCount > signal.totalCount {
		t.Fatalf("missingCount %d exceeds totalCount %d; the numerator must never exceed the tracked denominator", signal.missingCount, signal.totalCount)
	}
}

// TestAssessWatcherDeleteWaveCountsTrackedOnlyInMixedBatch confirms a genuine
// tracked mass-delete still quarantines while untracked churn in the same batch
// is excluded from the count.
func TestAssessWatcherDeleteWaveCountsTrackedOnlyInMixedBatch(t *testing.T) {
	t.Parallel()
	root := t.TempDir()

	const trackedCount = 120
	files := make(map[string]string, trackedCount)
	for i := 0; i < trackedCount; i++ {
		files[fmt.Sprintf("f%03d.go", i)] = fmt.Sprintf("hash-%03d", i)
	}
	for i := 110; i < trackedCount; i++ {
		rel := fmt.Sprintf("f%03d.go", i)
		if err := os.WriteFile(filepath.Join(root, rel), []byte("package main\n"), 0o644); err != nil {
			t.Fatalf("WriteFile(%s) returned error: %v", rel, err)
		}
	}
	snapshot := merkle.Snapshot{Files: files}

	batch := make([]string, 0, trackedCount+2000)
	for i := 0; i < trackedCount; i++ {
		batch = append(batch, fmt.Sprintf("f%03d.go", i))
	}
	for i := 0; i < 2000; i++ {
		batch = append(batch, fmt.Sprintf("node_modules/pkg/%04d.js", i))
	}

	codebase := model.Codebase{
		Kind:              model.CodebaseKindCode,
		Status:            model.CodebaseStatusIndexed,
		LastSuccessfulRun: &model.IndexRunSummary{IndexedFiles: trackedCount, TotalChunks: trackedCount},
		LiveFileTotal:     trackedCount,
	}

	signal, suspicious := assessWatcherDeleteWave(codebase, snapshot, root, batch)
	if !suspicious {
		t.Fatal("assessWatcherDeleteWave returned suspicious=false, want true for 110 genuine tracked deletes")
	}
	if signal.missingCount != 110 || signal.totalCount != 120 {
		t.Fatalf("signal = %+v, want 110 of 120 with the untracked node_modules churn excluded", signal)
	}
	if signal.reason != quarantineReasonWatcherLargeDelete {
		t.Fatalf("reason = %q, want the watcher large-delete reason", signal.reason)
	}
}

// TestAssessWatcherDeleteWaveEmptySnapshotDefersToFullScan confirms that with
// no tracked baseline in memory the watcher prefilter does not quarantine and
// instead leaves the decision to the authoritative full scan.
func TestAssessWatcherDeleteWaveEmptySnapshotDefersToFullScan(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	snapshot := merkle.Snapshot{Files: map[string]string{}}
	batch := make([]string, 0, 1600)
	for i := 0; i < 1600; i++ {
		batch = append(batch, fmt.Sprintf("f%04d.go", i))
	}
	codebase := model.Codebase{
		Kind:              model.CodebaseKindCode,
		Status:            model.CodebaseStatusIndexed,
		LastSuccessfulRun: &model.IndexRunSummary{IndexedFiles: 151, TotalChunks: 151},
		LiveFileTotal:     151,
	}
	if _, suspicious := assessWatcherDeleteWave(codebase, snapshot, root, batch); suspicious {
		t.Fatal("assessWatcherDeleteWave quarantined on an empty in-memory snapshot; the prefilter must defer to the full scan")
	}
}

// TestAssessWatcherDeleteWaveVCSOperationPausesBelowRatio confirms that a large
// tracked removal that is below the normal ratio still pauses (quarantines)
// when a git operation is mid-flight, so index rows are not deleted for files
// that a rebase or checkout will restore.
func TestAssessWatcherDeleteWaveVCSOperationPausesBelowRatio(t *testing.T) {
	t.Parallel()
	root := t.TempDir()

	const trackedCount = 500
	files := make(map[string]string, trackedCount)
	for i := 0; i < trackedCount; i++ {
		files[fmt.Sprintf("f%04d.go", i)] = fmt.Sprintf("hash-%04d", i)
	}
	snapshot := merkle.Snapshot{Files: files}

	batch := make([]string, 0, 120)
	for i := 0; i < 120; i++ {
		batch = append(batch, fmt.Sprintf("f%04d.go", i))
	}

	codebase := model.Codebase{
		Kind:              model.CodebaseKindCode,
		Status:            model.CodebaseStatusIndexed,
		LastSuccessfulRun: &model.IndexRunSummary{IndexedFiles: trackedCount, TotalChunks: trackedCount},
		LiveFileTotal:     trackedCount,
	}

	if _, suspicious := assessWatcherDeleteWave(codebase, snapshot, root, batch); suspicious {
		t.Fatal("precondition failed: 120 of 500 should not trip the normal ratio rule")
	}

	if err := os.MkdirAll(filepath.Join(root, ".git", "rebase-merge"), 0o755); err != nil {
		t.Fatalf("MkdirAll returned error: %v", err)
	}
	signal, suspicious := assessWatcherDeleteWave(codebase, snapshot, root, batch)
	if !suspicious {
		t.Fatal("assessWatcherDeleteWave returned suspicious=false during a rebase; a large removal mid-git-operation must pause")
	}
	if signal.reason != quarantineReasonVCSTransient {
		t.Fatalf("reason = %q, want the VCS-transient reason", signal.reason)
	}
	if signal.missingCount != 120 || signal.totalCount != 500 {
		t.Fatalf("signal = %+v, want 120 of 500", signal)
	}
}

// TestHandleQuarantinedCodebaseHoldsDuringVCSOperation confirms the
// confirmation gate never clears a quarantine or advances toward destructive
// sync while a git operation is in progress, even when the on-disk tree looks
// clean.
func TestHandleQuarantinedCodebaseHoldsDuringVCSOperation(t *testing.T) {
	t.Parallel()

	manager, cfg, repoPath := newTestManager(t)
	indexConfig := defaultIndexConfig()
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)
	codebase := newCodebaseRecord(repoPath)
	codebase.Status = model.CodebaseStatusQuarantined
	codebase.EffectiveConfig = indexConfig
	codebase.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 1}
	codebase.Quarantine = &model.QuarantineState{
		Reason:           quarantineReasonWatcherLargeDelete,
		ObservationCount: 1,
		LastTrigger:      quarantineTriggerWatcher,
		LastMissingCount: 110,
		LastTotalCount:   120,
	}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	content := "package main\n"
	if err := os.WriteFile(filepath.Join(repoPath, "main.go"), []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	snapshot := merkle.Snapshot{
		ConfigDigest: indexConfig.IgnoreDigest,
		Files:        map[string]string{"main.go": hashText(content)},
		Inodes:       nil,
	}
	if err := merkle.WriteSnapshot(manager.snapshotPathForCodebase(codebase), snapshot); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(repoPath, ".git", "rebase-merge"), 0o755); err != nil {
		t.Fatalf("MkdirAll returned error: %v", err)
	}

	syncer := NewBackgroundSync(cfg, manager)
	syncer.handleQuarantinedCodebase(context.Background(), codebase)

	manager.mu.Lock()
	readCodebase, found := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if !found {
		t.Fatalf("tracked codebase %s disappeared", codebase.ID)
	}
	if readCodebase.Status != model.CodebaseStatusQuarantined {
		t.Fatalf("status = %q, want quarantine held during a git operation", readCodebase.Status)
	}
	if readCodebase.Quarantine == nil {
		t.Fatal("Quarantine = nil, want quarantine retained during a git operation")
	}
}

package daemon

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
)

func TestBootstrapAdmissionHaltDropsStagingAndDoesNotPromote(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 1)
	manager.config.MaxJobChunks = 1
	fake := &fakeSemantic{
		hasStaging: func(context.Context, string) (bool, error) { return true, nil },
	}
	manager.semantic = fake
	manager.runner = runnerWithChunks(map[string][]string{
		"main.go": {"chunk one", "chunk two"},
	})

	repoPath := newMultiFileRepo(t, "main.go")
	cfg := defaultIndexConfig()
	cfg.IgnoreDigest = "sha256:admission-bootstrap"
	codebaseID, job := seedBootstrapCodebase(t, manager, repoPath, cfg)

	manager.runBootstrap(context.Background(), job, newCodeItemSource(manager.runner, manager.indexability, codebaseID, repoPath, cfg))

	completed, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("job %s was not found", job.ID)
	}
	if completed.State != model.JobStateFailed {
		t.Fatalf("job state = %q, want failed", completed.State)
	}
	if completed.Error == nil || completed.Error.Code != adapterr.CodeIndexBudgetExceeded {
		t.Fatalf("job error = %+v, want admission budget code", completed.Error)
	}
	if len(fake.stageCallsSnapshot()) != 0 {
		t.Fatalf("StageReindex calls = %v, want none before over-budget write", fake.stageCallsSnapshot())
	}
	if len(fake.promotedSnapshot()) != 0 {
		t.Fatalf("PromoteStaging calls = %v, want none", fake.promotedSnapshot())
	}
	if len(fake.droppedStagingSnapshot()) == 0 {
		t.Fatal("DropStaging was not called after admission halt")
	}
}

func TestForcedRebuildAdmissionHaltPreservesLiveSnapshot(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 1)
	manager.config.MaxJobChunks = 1
	manager.semantic = &fakeSemantic{
		hasStaging: func(context.Context, string) (bool, error) { return true, nil },
	}
	manager.runner = runnerWithChunks(map[string][]string{
		"main.go": {"new one", "new two"},
	})

	repoPath := newMultiFileRepo(t, "main.go")
	cfg := defaultIndexConfig()
	cfg.IgnoreDigest = "sha256:admission-forced"
	codebaseID, job := seedBootstrapCodebase(t, manager, repoPath, cfg)
	job.Forced = true

	liveSnapshot := merkle.Snapshot{
		ConfigDigest: cfg.IgnoreDigest,
		Files:        map[string]string{"main.go": hashText("old live content")},
		Inodes:       nil,
	}
	if err := merkle.WriteSnapshot(manager.merklePath(codebaseID), liveSnapshot); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}
	manager.mu.Lock()
	codebase := manager.codebases[codebaseID]
	codebase.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 1, Status: "completed"}
	manager.codebases[codebaseID] = codebase
	manager.mu.Unlock()

	manager.runBootstrap(context.Background(), job, newCodeItemSource(manager.runner, manager.indexability, codebaseID, repoPath, cfg))

	after, err := merkle.ReadSnapshot(manager.merklePath(codebaseID))
	if err != nil {
		t.Fatalf("ReadSnapshot returned error: %v", err)
	}
	if !merkle.Equal(after, liveSnapshot) {
		t.Fatalf("live snapshot changed after halted forced rebuild: got %+v want %+v", after, liveSnapshot)
	}
	stagingPath := manager.stagingMerklePath(codebaseID)
	if _, err := os.Stat(stagingPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("staging snapshot still exists after halt: %v", err)
	}
}

func TestDeltaAdmissionHaltDoesNotWriteOffendingFile(t *testing.T) {
	manager, codebase, job, fake := newAdmissionDeltaFixture(t)
	manager.config.MaxJobChunks = 1
	manager.runner = runnerWithChunks(map[string][]string{
		"added.go": {"chunk one", "chunk two"},
	})
	if err := os.WriteFile(filepath.Join(job.CanonicalPath, "added.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	source := newCodeItemSource(manager.runner, manager.indexability, job.CodebaseID, job.CanonicalPath, job.Config).withCollectionName(codebase.CollectionName)

	handled := manager.runDeltaSync(context.Background(), job, source)
	if !handled {
		t.Fatal("runDeltaSync returned false, want handled admission halt")
	}

	completed, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("job %s was not found", job.ID)
	}
	if completed.State != model.JobStateFailed {
		t.Fatalf("job state = %q, want failed", completed.State)
	}
	if completed.Error == nil || completed.Error.Code != adapterr.CodeIndexBudgetExceeded {
		t.Fatalf("job error = %+v, want admission budget code", completed.Error)
	}
	if calls := fake.reindexCallsSnapshot(); len(calls) != 0 {
		t.Fatalf("live reindex calls = %v, want none for offending file", calls)
	}
	snapshot, err := merkle.ReadSnapshot(manager.merklePath(codebase.ID))
	if err != nil {
		t.Fatalf("ReadSnapshot returned error: %v", err)
	}
	if snapshot.HasFile("added.go") {
		t.Fatalf("snapshot recorded added.go after refused write: %+v", snapshot.Files)
	}
}

func TestSiblingExpectedAdmissionHaltTrips(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 1)
	manager.config.MaxJobChunks = 100
	manager.config.ExpectedJobGrowthFactor = 2
	manager.config.ExpectedJobGrowthFloor = 1
	manager.semantic = &fakeSemantic{
		hasStaging: func(context.Context, string) (bool, error) { return true, nil },
	}

	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(base, "feature")
	makeLinkedWorktree(t, mainRoot, "feature", worktreeDir, "feature")
	worktreeRoot := evalSym(t, worktreeDir)
	if err := os.WriteFile(filepath.Join(worktreeRoot, "feature.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}

	cfg := defaultIndexConfig()
	cfg.IgnoreDigest = "sha256:admission-sibling"
	registerSiblingCodebase(t, manager, mainRoot, func(codebase *model.Codebase) {
		codebase.Status = model.CodebaseStatusIndexed
		codebase.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 1, Status: "completed"}
		codebase.EffectiveConfig = cfg
	})
	manager.runner = runnerWithChunks(map[string][]string{
		"feature.go": {"one", "two", "three"},
	})
	codebaseID, job := seedBootstrapCodebase(t, manager, worktreeRoot, cfg)

	manager.runBootstrap(context.Background(), job, newCodeItemSource(manager.runner, manager.indexability, codebaseID, worktreeRoot, cfg))

	completed, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("job %s was not found", job.ID)
	}
	if completed.State != model.JobStateFailed {
		t.Fatalf("job state = %q, want failed", completed.State)
	}
	if completed.Error == nil || completed.Error.Code != adapterr.CodeIndexBudgetExceeded {
		t.Fatalf("job error = %+v, want admission budget code", completed.Error)
	}
}

func TestAdmissionAllowsNormalLargeBuildWithinSlack(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 1)
	manager.config.MaxJobChunks = 100
	manager.config.ExpectedJobGrowthFactor = 2
	manager.config.ExpectedJobGrowthFloor = 50
	manager.semantic = &fakeSemantic{
		hasStaging: func(context.Context, string) (bool, error) { return true, nil },
		count:      func(context.Context, string) (int32, error) { return 40, nil },
	}
	chunks := make([]string, 0, 40)
	for i := 0; i < 40; i++ {
		chunks = append(chunks, "chunk")
	}
	manager.runner = runnerWithChunks(map[string][]string{
		"main.go": chunks,
	})

	repoPath := newMultiFileRepo(t, "main.go")
	cfg := defaultIndexConfig()
	cfg.IgnoreDigest = "sha256:admission-normal"
	codebaseID, job := seedBootstrapCodebase(t, manager, repoPath, cfg)
	manager.mu.Lock()
	codebase := manager.codebases[codebaseID]
	codebase.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 10, Status: "completed"}
	manager.codebases[codebaseID] = codebase
	manager.mu.Unlock()

	manager.runBootstrap(context.Background(), job, newCodeItemSource(manager.runner, manager.indexability, codebaseID, repoPath, cfg))

	completed, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("job %s was not found", job.ID)
	}
	if completed.State != model.JobStateCompleted {
		t.Fatalf("job state = %q, want completed; error = %+v", completed.State, completed.Error)
	}
}

func TestConvergePathsRoutesUpsertThroughAdmission(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 1)
	manager.config.MaxJobChunks = 1
	fake := &fakeSemantic{}
	manager.semantic = fake
	manager.runner = runnerWithChunks(map[string][]string{
		"main.go": {"chunk one", "chunk two"},
	})
	repoPath := newMultiFileRepo(t, "main.go")
	cfg := defaultIndexConfig()
	cfg.IgnoreDigest = "sha256:admission-converge"
	codebase := newCodebaseRecord(repoPath)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = cfg
	codebase.MerkleSnapshotPath = manager.merklePath(codebase.ID)
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	err := manager.ConvergePaths(context.Background(), codebase.ID, []string{"main.go"})
	if err != nil {
		t.Fatalf("ConvergePaths returned error: %v", err)
	}
	if calls := fake.reindexCallsSnapshot(); len(calls) != 0 {
		t.Fatalf("converge Reindex calls = %v, want none after admission refusal", calls)
	}
}

func TestAdmissionHaltedFailedBuildIsNotAutoRetried(t *testing.T) {
	t.Parallel()

	manager, cfg, repoPath := newTestManager(t)
	codebase := registerFailedBuildCodebase(t, manager, repoPath)
	manager.mu.Lock()
	updated := manager.codebases[codebase.ID]
	updated.LastFailedRun = &model.IndexRunFailure{
		Message:  "index budget exceeded",
		Code:     adapterr.CodeIndexBudgetExceeded,
		FailedAt: clock.Now(),
	}
	manager.codebases[codebase.ID] = updated
	manager.mu.Unlock()
	syncer := NewBackgroundSync(cfg, manager)

	syncer.runSyncAll(context.Background(), "test")

	if got := len(manager.ListJobs(codebase.ID)); got != 0 {
		t.Fatalf("retry jobs for admission halt = %d, want 0", got)
	}
}

func TestRequestAdmissionBudgetCannotRaiseServerCaps(t *testing.T) {
	manager, _, _ := newTestManager(t)
	manager.config.MaxJobChunks = 10
	manager.config.MaxJobBytes = 100

	raised := newAdmissionState(manager.config, model.AdmissionBudget{
		MaxJobChunks: 20,
		MaxJobBytes:  200,
	}, 0)
	if raised.maxChunks != 10 {
		t.Fatalf("raised maxChunks = %d, want server cap 10", raised.maxChunks)
	}
	if raised.maxBytes != 100 {
		t.Fatalf("raised maxBytes = %d, want server cap 100", raised.maxBytes)
	}

	tightened := newAdmissionState(manager.config, model.AdmissionBudget{
		MaxJobChunks: 4,
		MaxJobBytes:  40,
	}, 0)
	if tightened.maxChunks != 4 {
		t.Fatalf("tightened maxChunks = %d, want request cap 4", tightened.maxChunks)
	}
	if tightened.maxBytes != 40 {
		t.Fatalf("tightened maxBytes = %d, want request cap 40", tightened.maxBytes)
	}
}

func TestRequestAdmissionBudgetTightensUnlimitedServerCaps(t *testing.T) {
	manager, _, _ := newTestManager(t)
	manager.config.MaxJobChunks = 0
	manager.config.MaxJobBytes = 0

	state := newAdmissionState(manager.config, model.AdmissionBudget{
		MaxJobChunks: 4,
		MaxJobBytes:  40,
	}, 0)
	if state.maxChunks != 4 {
		t.Fatalf("maxChunks = %d, want request cap 4", state.maxChunks)
	}
	if state.maxBytes != 40 {
		t.Fatalf("maxBytes = %d, want request cap 40", state.maxBytes)
	}
}

func TestAdmissionForCodebaseUsesLiveChunkTotalBaseline(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	cfg := defaultIndexConfig()
	cfg.IgnoreDigest = "sha256:admission-live-baseline"
	codebase := newCodebaseRecord(repoPath)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = cfg
	codebase.LastSuccessfulRun = &model.IndexRunSummary{
		IndexedFiles: 1,
		TotalChunks:  3,
		Status:       "completed",
	}
	codebase.LiveChunkTotal = 9
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	state := manager.admissionForCodebase(codebase)

	if state.expectedChunks != 9 {
		t.Fatalf("expectedChunks = %d, want LiveChunkTotal 9", state.expectedChunks)
	}
}

func runnerWithChunks(chunksByPath map[string][]string) fakeRunner {
	return fakeRunner{
		indexOne: func(_ context.Context, _ string, relativePath string, _ model.IndexConfig) (indexer.OneFileResult, error) {
			chunkTexts := chunksByPath[relativePath]
			chunks := make([]model.StoredChunk, 0, len(chunkTexts))
			for index, content := range chunkTexts {
				chunks = append(chunks, model.StoredChunk{
					Content:       content,
					RelativePath:  relativePath,
					StartLine:     int32(index + 1),
					EndLine:       int32(index + 1),
					Language:      "go",
					FileExtension: ".go",
				})
			}
			return indexer.OneFileResult{
				Chunks:   chunks,
				FileHash: hashText(relativePath + ":changed"),
				Skipped:  false,
				Removed:  false,
			}, nil
		},
	}
}

func newAdmissionDeltaFixture(t *testing.T) (*Manager, model.Codebase, model.Job, *fakeSemantic) {
	t.Helper()

	manager, _, _ := newTestManager(t)
	fake := &fakeSemantic{}
	manager.semantic = fake

	repoPath := newMultiFileRepo(t, "main.go")
	cfg := defaultIndexConfig()
	cfg.IgnoreDigest = "sha256:admission-delta"
	codebase := newCodebaseRecord(repoPath)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.CollectionName = "cc_admission_delta"
	codebase.EffectiveConfig = cfg
	codebase.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 1, Status: "completed"}
	codebase.MerkleSnapshotPath = manager.merklePath(codebase.ID)
	initialSnapshot := merkle.Snapshot{
		ConfigDigest: cfg.IgnoreDigest,
		Files:        map[string]string{"main.go": hashText("package main\n// main.go\n")},
		Inodes:       nil,
	}
	if err := merkle.WriteSnapshot(manager.merklePath(codebase.ID), initialSnapshot); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}
	job := newQueuedJob(codebase.ID, repoPath, repoPath, testClientInfo(), string(jobOperationSync), false, cfg, emptyAdmissionBudget, clock.Now())
	job.State = model.JobStateRunning
	codebase.ActiveJobID = job.ID

	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	return manager, codebase, job, fake
}

func (f *fakeSemantic) stageCallsSnapshot() []reindexCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.stageCalls)
}

func (f *fakeSemantic) reindexCallsSnapshot() []reindexCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.reindexCalls)
}

func (f *fakeSemantic) promotedSnapshot() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.promoted)
}

func (f *fakeSemantic) droppedStagingSnapshot() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.droppedStaging)
}

package daemon

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func TestRunDeltaSyncFailsWhenChunkCacheUnreadable(t *testing.T) {
	manager, codebase, job, _ := newAdmissionDeltaFixture(t)
	manager.runner = runnerWithChunks(map[string][]string{
		"main.go": {"replacement chunk"},
	})
	if err := os.WriteFile(filepath.Join(job.CanonicalPath, "main.go"), []byte("package main\nfunc Changed() {}\n"), 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	corruptCache := []byte("{not-json")
	if err := os.WriteFile(manager.chunkPath(codebase.ID), corruptCache, 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	source := newCodeItemSource(manager.runner, manager.indexability, job.CodebaseID, job.CanonicalPath, job.Config).withCollectionName(codebase.CollectionName)

	handled, _ := manager.runDeltaSync(context.Background(), job, source)
	if !handled {
		t.Fatal("runDeltaSync returned false, want handled failure")
	}

	completed, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("job %s was not found", job.ID)
	}
	if completed.State != model.JobStateFailed {
		t.Fatalf("job state = %q, want failed", completed.State)
	}
	afterCache, err := os.ReadFile(manager.chunkPath(codebase.ID))
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	if !slices.Equal(afterCache, corruptCache) {
		t.Fatalf("chunk cache was rewritten after read failure: got %q want %q", string(afterCache), string(corruptCache))
	}
}

// TestRunDeltaSyncKeepsPriorTotalBytesWhenChunkCacheMissing proves that when the
// code chunk cache file is missing (deleted externally), a delta carries the
// prior whole-codebase TotalBytes forward instead of recomputing it from only
// the changed files, which would shrink the reported total dramatically.
func TestRunDeltaSyncKeepsPriorTotalBytesWhenChunkCacheMissing(t *testing.T) {
	manager, codebase, job, _ := newAdmissionDeltaFixture(t)
	const priorBytes = 10000
	manager.mu.Lock()
	stored := manager.codebases[codebase.ID]
	stored.LastSuccessfulRun.TotalBytes = priorBytes
	manager.codebases[codebase.ID] = stored
	manager.mu.Unlock()

	manager.runner = runnerWithChunks(map[string][]string{
		"main.go": {"replacement chunk"},
	})
	if err := os.WriteFile(filepath.Join(job.CanonicalPath, "main.go"), []byte("package main\nfunc Changed() {}\n"), 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	// No chunk cache file is written, so store.ReadChunks reports os.ErrNotExist.
	source := newCodeItemSource(manager.runner, manager.indexability, job.CodebaseID, job.CanonicalPath, job.Config).withCollectionName(codebase.CollectionName)

	handled, graphTask := manager.runDeltaSync(context.Background(), job, source)
	if !handled {
		t.Fatal("runDeltaSync returned false, want handled completion")
	}
	manager.runGraphIndexTask(context.Background(), graphTask)

	completed, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("job %s was not found", job.ID)
	}
	if completed.State != model.JobStateCompleted {
		t.Fatalf("job state = %q, want completed", completed.State)
	}

	manager.mu.Lock()
	after := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if after.LastSuccessfulRun == nil {
		t.Fatal("LastSuccessfulRun is nil after delta sync")
	}
	if after.LastSuccessfulRun.TotalBytes != priorBytes {
		t.Fatalf("LastSuccessfulRun.TotalBytes = %d, want prior %d carried forward when the chunk cache is missing", after.LastSuccessfulRun.TotalBytes, priorBytes)
	}
}

func TestRunDeltaSyncCreatesGraphTaskForLegacyEmptyKindCodebase(t *testing.T) {
	manager, codebase, job, _ := newAdmissionDeltaFixture(t)
	manager.mu.Lock()
	stored := manager.codebases[codebase.ID]
	stored.Kind = ""
	manager.codebases[codebase.ID] = stored
	manager.mu.Unlock()

	manager.runner = runnerWithChunks(map[string][]string{
		"main.go": {"replacement chunk"},
	})
	if err := os.WriteFile(filepath.Join(job.CanonicalPath, "main.go"), []byte("package main\nfunc Changed() {}\n"), 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	source := newCodeItemSource(manager.runner, manager.indexability, job.CodebaseID, job.CanonicalPath, job.Config).withCollectionName(codebase.CollectionName)

	handled, graphTask := manager.runDeltaSync(context.Background(), job, source)
	if !handled {
		t.Fatal("runDeltaSync returned false, want handled completion")
	}
	if graphTask == nil {
		t.Fatal("graphTask is nil, want graph task for legacy empty-kind codebase")
	}
	if graphTask.codebaseID != codebase.ID {
		t.Fatalf("graphTask codebaseID = %q, want %q", graphTask.codebaseID, codebase.ID)
	}
}

func TestRunDeltaSyncReadsOnlyTargetAndDeclaredSource(t *testing.T) {
	t.Run("added file reuses declared source vectors", func(t *testing.T) {
		addedContent := "package feature\n\nfunc Added() string { return \"shared\" }\n"
		manager, codebase, job, fake := newWorktreeDeltaReuseFixture(t, map[string]model.ReuseVector{
			hashText(addedContent): {Vector: []float32{1, 2, 3}},
		}, true)
		writeDeltaFixtureFile(t, job.CanonicalPath, "added.go", addedContent)
		manager.runner = deltaReuseRunner(map[string]string{"added.go": addedContent, "feature.go": addedContent})

		completed := syncWorktreeDeltaReuseFixture(t, manager, codebase, job)
		if completed.Progress.ReuseVectorsLoaded <= 0 {
			t.Fatalf("The sync with a declared source must load more than 0 reuse vectors instead of %d.", completed.Progress.ReuseVectorsLoaded)
		}
		if completed.Progress.ChunksEmbedded != 0 {
			t.Fatalf("ChunksEmbedded = %d, want 0 for identical added content", completed.Progress.ChunksEmbedded)
		}
		if completed.Progress.ChunksReused <= 0 {
			t.Fatalf("ChunksReused = %d, want > 0 for identical added content", completed.Progress.ChunksReused)
		}
		wantCollections := []string{"cc_feature", "cc_repo"}
		if readCollections := fake.reuseContentCollectionNames(); !slices.Equal(readCollections, wantCollections) {
			t.Fatalf("The content lookups read collections %v instead of the expected collections %v.", readCollections, wantCollections)
		}
		if calls := fake.reuseCollectionsSnapshot(); len(calls) != 0 {
			t.Fatalf("The sync must not load a full collection in recorded calls %v.", calls)
		}
	})

	t.Run("added file without declared source reads only the target", func(t *testing.T) {
		addedContent := "package feature\n\nfunc Added() string { return \"shared\" }\n"
		manager, codebase, job, fake := newWorktreeDeltaReuseFixture(t, map[string]model.ReuseVector{
			hashText(addedContent): {Vector: []float32{1, 2, 3}},
		}, false)
		writeDeltaFixtureFile(t, job.CanonicalPath, "added.go", addedContent)
		manager.runner = deltaReuseRunner(map[string]string{"added.go": addedContent, "feature.go": addedContent})

		completed := syncWorktreeDeltaReuseFixture(t, manager, codebase, job)
		if completed.Progress.ChunksEmbedded <= 0 {
			t.Fatalf("The sync without a declared source must embed more than 0 chunks instead of %d.", completed.Progress.ChunksEmbedded)
		}
		if completed.Progress.ChunksReused != 0 {
			t.Fatalf("The sync without a declared source must reuse 0 chunks instead of %d.", completed.Progress.ChunksReused)
		}
		readCollections := fake.reuseContentCollectionNames()
		if len(readCollections) == 0 {
			t.Fatal("The sync recorded no content lookup.")
		}
		wantCollections := []string{"cc_feature"}
		if !slices.Equal(readCollections, wantCollections) {
			t.Fatalf("The content lookups read collections %v instead of only the target collection %v.", readCollections, wantCollections)
		}
		if calls := fake.reuseCollectionsSnapshot(); len(calls) != 0 {
			t.Fatalf("The sync must not load a full collection in recorded calls %v.", calls)
		}
	})

	t.Run("modified only does not seed siblings", func(t *testing.T) {
		manager, codebase, job, fake := newWorktreeDeltaReuseFixture(t, nil, false)
		modifiedContent := "package feature\n\nfunc Modified() string { return \"changed\" }\n"
		writeDeltaFixtureFile(t, job.CanonicalPath, "feature.go", modifiedContent)
		manager.runner = deltaReuseRunner(map[string]string{"feature.go": modifiedContent})

		syncWorktreeDeltaReuseFixture(t, manager, codebase, job)
		if calls := fake.reuseCollectionsSnapshot(); len(calls) != 0 {
			t.Fatalf("modified-only delta loaded sibling reuse collections: %v", calls)
		}
	})
}

func syncWorktreeDeltaReuseFixture(t *testing.T, manager *Manager, codebase model.Codebase, fixtureJob model.Job) model.Job {
	t.Helper()

	manager.mu.Lock()
	registered := manager.codebases[codebase.ID]
	registered.ActiveJobID = ""
	manager.codebases[codebase.ID] = registered
	delete(manager.jobs, fixtureJob.ID)
	manager.mu.Unlock()

	syncJob, _, _, err := manager.SyncIndex(context.Background(), codebase.CanonicalPath, testClientInfo())
	if err != nil {
		t.Fatalf("SyncIndex returned error: %v", err)
	}
	var completed model.Job
	waitForCondition(t, func() bool {
		current, found := manager.GetJob(syncJob.ID)
		if !found {
			return false
		}
		terminal := current.State == model.JobStateCompleted ||
			current.State == model.JobStateFailed ||
			current.State == model.JobStateCancelled
		if !terminal {
			return false
		}
		completed = current
		return true
	})
	if completed.State != model.JobStateCompleted {
		t.Fatalf("job state = %q, want completed", completed.State)
	}
	return completed
}

func newWorktreeDeltaReuseFixture(t *testing.T, sourceVectors map[string]model.ReuseVector, declareSource bool) (*Manager, model.Codebase, model.Job, *fakeSemantic) {
	t.Helper()

	manager, _, _ := newTestManager(t)
	cfg := defaultIndexConfig()
	cfg.IgnoreDigest = "sha256:delta-sibling-reuse"

	fake := &fakeSemantic{
		collectionName:       func(path string) string { return "cc_" + filepath.Base(path) },
		hasCollectionForPath: func(context.Context, string) (bool, error) { return true, nil },
		loadReuse: func(context.Context, []string) (map[string]model.ReuseVector, error) {
			return cloneReuseVectors(sourceVectors), nil
		},
		loadReuseForContents: func(_ context.Context, collectionName string, chunks []model.StoredChunk) (map[string]model.ReuseVector, error) {
			found := map[string]model.ReuseVector{}
			if collectionName != "cc_repo" {
				return found, nil
			}
			for _, chunk := range chunks {
				if vector, present := sourceVectors[hashText(chunk.Content)]; present {
					found[hashText(chunk.Content)] = model.ReuseVector{
						Vector:           append([]float32(nil), vector.Vector...),
						SourceModelKnown: vector.SourceModelKnown,
					}
				}
			}
			return found, nil
		},
		reindexWithReuse: reportWorktreeDeltaReuse,
	}
	fake.stageReindexWithReuse = reportWorktreeDeltaReuse
	manager.semantic = fake

	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(base, "feature")
	makeLinkedWorktree(t, mainRoot, "feature", worktreeDir, "feature")
	worktreeRoot := evalSym(t, worktreeDir)

	registerSiblingCodebase(t, manager, mainRoot, func(codebase *model.Codebase) {
		codebase.Status = model.CodebaseStatusIndexed
		codebase.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 1, Status: "completed"}
		codebase.EffectiveConfig = cfg
		codebase.EffectiveConfig.EmbeddingProvider = fake.EmbeddingProviderName()
		codebase.EffectiveConfig.EmbeddingModel = manager.config.EmbeddingModel
		if manager.config.EmbeddingDimension > 0 {
			codebase.EffectiveConfig.EmbeddingDimension = manager.config.EmbeddingDimension
		}
	})

	worktreeCodebase := newCodebaseRecord(worktreeRoot)
	worktreeCodebase.Status = model.CodebaseStatusIndexed
	worktreeCodebase.CollectionName = "cc_feature"
	worktreeCodebase.EffectiveConfig = cfg
	worktreeCodebase.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 1, Status: "completed"}
	worktreeCodebase.MerkleSnapshotPath = manager.merklePath(worktreeCodebase.ID)
	if declareSource {
		worktreeCodebase.WorktreeSourcePath = evalSym(t, mainRoot)
	}

	initialSnapshot, err := merkle.Capture(context.Background(), manager.indexability, worktreeCodebase.ID, worktreeRoot, cfg)
	if err != nil {
		t.Fatalf("Capture returned error: %v", err)
	}
	initialSnapshot.ConfigDigest = cfg.IgnoreDigest
	if err := merkle.WriteSnapshot(manager.merklePath(worktreeCodebase.ID), initialSnapshot); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	job := newQueuedJob(worktreeCodebase.ID, worktreeRoot, worktreeRoot, testClientInfo(), string(jobOperationSync), false, cfg, emptyAdmissionBudget, clock.Now())
	job.State = model.JobStateRunning
	worktreeCodebase.ActiveJobID = job.ID

	manager.mu.Lock()
	manager.codebases[worktreeCodebase.ID] = worktreeCodebase
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	return manager, worktreeCodebase, job, fake
}

func reportWorktreeDeltaReuse(_ context.Context, _ string, chunks []model.StoredChunk, _ []string, progress func(semantic.Progress), reuse map[string]model.ReuseVector) error {
	if progress == nil {
		return nil
	}
	chunkCount := safeInt32(len(chunks))
	var reused int32
	var embedded int32
	for _, chunk := range chunks {
		if _, present := reuse[hashText(chunk.Content)]; present {
			reused++
		} else {
			embedded++
		}
	}
	progress(semantic.Progress{ChunksProcessed: chunkCount, ChunksReused: reused, ChunksEmbedded: embedded})
	return nil
}

func cloneReuseVectors(reuse map[string]model.ReuseVector) map[string]model.ReuseVector {
	cloned := make(map[string]model.ReuseVector, len(reuse))
	for key, vector := range reuse {
		cloned[key] = model.ReuseVector{
			Vector:           append([]float32(nil), vector.Vector...),
			SourceModelKnown: vector.SourceModelKnown,
		}
	}
	return cloned
}

func deltaReuseRunner(contents map[string]string) fakeRunner {
	return fakeRunner{
		indexOne: func(_ context.Context, _ string, relativePath string, _ model.IndexConfig) (indexer.OneFileResult, error) {
			content := contents[relativePath]
			return indexer.OneFileResult{
				Chunks: []model.StoredChunk{{
					Content:       content,
					RelativePath:  relativePath,
					StartLine:     1,
					EndLine:       3,
					Language:      "go",
					FileExtension: ".go",
				}},
				FileHash: hashText(content),
				Skipped:  false,
				Removed:  false,
			}, nil
		},
	}
}

func writeDeltaFixtureFile(t *testing.T, root string, relativePath string, content string) {
	t.Helper()

	path := filepath.Join(root, relativePath)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile(%s) returned error: %v", path, err)
	}
}

func (f *fakeSemantic) reuseCollectionsSnapshot() [][]string {
	f.mu.Lock()
	defer f.mu.Unlock()

	out := make([][]string, 0, len(f.reuseCollections))
	for _, call := range f.reuseCollections {
		out = append(out, append([]string(nil), call...))
	}
	return out
}

func (f *fakeSemantic) reuseContentCollectionNames() []string {
	f.mu.Lock()
	defer f.mu.Unlock()

	names := make([]string, 0)
	for _, call := range f.reuseContentCollections {
		for _, name := range call {
			if !slices.Contains(names, name) {
				names = append(names, name)
			}
		}
	}
	slices.Sort(names)
	return names
}

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

func TestRunDeltaSyncSeedsSiblingReuseOnlyForAddedFiles(t *testing.T) {
	t.Run("added file reuses sibling vectors", func(t *testing.T) {
		addedContent := "package feature\n\nfunc Added() string { return \"shared\" }\n"
		manager, codebase, job, fake := newWorktreeDeltaReuseFixture(t, map[string][]float32{
			hashText(addedContent): {1, 2, 3},
		})
		writeDeltaFixtureFile(t, job.CanonicalPath, "added.go", addedContent)
		manager.runner = deltaReuseRunner(map[string]string{"added.go": addedContent})
		source := newCodeItemSource(manager.runner, manager.indexability, job.CodebaseID, job.CanonicalPath, job.Config).withCollectionName(codebase.CollectionName)

		handled, graphTask := manager.runDeltaSync(context.Background(), job, source)
		if !handled {
			t.Fatal("runDeltaSync returned false, want it to handle the added-file delta")
		}
		manager.runGraphIndexTask(context.Background(), graphTask)

		completed, found := manager.GetJob(job.ID)
		if !found {
			t.Fatalf("job %s was not found after runDeltaSync", job.ID)
		}
		if completed.State != model.JobStateCompleted {
			t.Fatalf("job state = %q, want completed", completed.State)
		}
		if completed.Progress.ReuseVectorsLoaded <= 0 {
			t.Fatalf("ReuseVectorsLoaded = %d, want > 0 from sibling seed", completed.Progress.ReuseVectorsLoaded)
		}
		if completed.Progress.ChunksEmbedded != 0 {
			t.Fatalf("ChunksEmbedded = %d, want 0 for identical added content", completed.Progress.ChunksEmbedded)
		}
		if completed.Progress.ChunksReused <= 0 {
			t.Fatalf("ChunksReused = %d, want > 0 for identical added content", completed.Progress.ChunksReused)
		}
		if !fake.requestedReuseCollection("cc_repo") {
			t.Fatalf("sibling collection cc_repo was not loaded; calls = %v", fake.reuseCollectionsSnapshot())
		}

		snapshot, err := merkle.ReadSnapshot(manager.merklePath(codebase.ID))
		if err != nil {
			t.Fatalf("ReadSnapshot returned error: %v", err)
		}
		if _, present := snapshot.Files["added.go"]; !present {
			t.Fatalf("snapshot missing added.go after delta sync; have %v", snapshot.Files)
		}
	})

	t.Run("modified only does not seed siblings", func(t *testing.T) {
		manager, codebase, job, fake := newWorktreeDeltaReuseFixture(t, nil)
		modifiedContent := "package feature\n\nfunc Modified() string { return \"changed\" }\n"
		writeDeltaFixtureFile(t, job.CanonicalPath, "feature.go", modifiedContent)
		manager.runner = deltaReuseRunner(map[string]string{"feature.go": modifiedContent})
		source := newCodeItemSource(manager.runner, manager.indexability, job.CodebaseID, job.CanonicalPath, job.Config).withCollectionName(codebase.CollectionName)

		handled, graphTask := manager.runDeltaSync(context.Background(), job, source)
		if !handled {
			t.Fatal("runDeltaSync returned false, want it to handle the modified-only delta")
		}
		manager.runGraphIndexTask(context.Background(), graphTask)

		completed, found := manager.GetJob(job.ID)
		if !found {
			t.Fatalf("job %s was not found after runDeltaSync", job.ID)
		}
		if completed.State != model.JobStateCompleted {
			t.Fatalf("job state = %q, want completed", completed.State)
		}
		if calls := fake.reuseCollectionsSnapshot(); len(calls) != 0 {
			t.Fatalf("modified-only delta loaded sibling reuse collections: %v", calls)
		}
	})

	t.Run("document-kind added delta seeds siblings like code", func(t *testing.T) {
		addedContent := "package feature\n\nfunc AddedDocumentKind() string { return \"shared\" }\n"
		manager, codebase, job, fake := newWorktreeDeltaReuseFixture(t, map[string][]float32{
			hashText(addedContent): {1, 2, 3},
		})
		manager.mu.Lock()
		documentCodebase := manager.codebases[codebase.ID]
		documentCodebase.Kind = model.CodebaseKindDocument
		manager.codebases[codebase.ID] = documentCodebase
		manager.mu.Unlock()

		writeDeltaFixtureFile(t, job.CanonicalPath, "document-kind.go", addedContent)
		manager.runner = deltaReuseRunner(map[string]string{"document-kind.go": addedContent})
		source := newCodeItemSource(manager.runner, manager.indexability, job.CodebaseID, job.CanonicalPath, job.Config).withCollectionName(codebase.CollectionName)

		handled, graphTask := manager.runDeltaSync(context.Background(), job, source)
		if !handled {
			t.Fatal("runDeltaSync returned false, want it to handle the added document-kind delta")
		}
		manager.runGraphIndexTask(context.Background(), graphTask)

		completed, found := manager.GetJob(job.ID)
		if !found {
			t.Fatalf("job %s was not found after runDeltaSync", job.ID)
		}
		if completed.State != model.JobStateCompleted {
			t.Fatalf("job state = %q, want completed", completed.State)
		}
		// A chat:/// document collection has no git siblings in practice, so
		// conversations still rely on per-item reuse. This fixture uses a linked
		// worktree to pin that the build-wide rule itself is Kind neutral.
		if !fake.requestedReuseCollection("cc_repo") {
			t.Fatalf("sibling collection cc_repo was not loaded; calls = %v", fake.reuseCollectionsSnapshot())
		}
		if completed.Progress.ReuseVectorsLoaded <= 0 {
			t.Fatalf("ReuseVectorsLoaded = %d, want > 0 from sibling seed", completed.Progress.ReuseVectorsLoaded)
		}
		if completed.Progress.ChunksEmbedded != 0 {
			t.Fatalf("ChunksEmbedded = %d, want 0 for identical document-kind content", completed.Progress.ChunksEmbedded)
		}
		if completed.Progress.ChunksReused <= 0 {
			t.Fatalf("ChunksReused = %d, want > 0 for identical document-kind content", completed.Progress.ChunksReused)
		}
	})
}

func newWorktreeDeltaReuseFixture(t *testing.T, loadReuse map[string][]float32) (*Manager, model.Codebase, model.Job, *fakeSemantic) {
	t.Helper()

	manager, _, _ := newTestManager(t)
	cfg := defaultIndexConfig()
	cfg.IgnoreDigest = "sha256:delta-sibling-reuse"

	fake := &fakeSemantic{
		collectionName:       func(path string) string { return "cc_" + filepath.Base(path) },
		hasCollectionForPath: func(context.Context, string) (bool, error) { return true, nil },
		loadReuse: func(context.Context, []string) (map[string][]float32, error) {
			return cloneReuseVectors(loadReuse), nil
		},
		reindexWithReuse: func(_ context.Context, _ string, chunks []model.StoredChunk, _ []string, progress func(semantic.Progress), reuse map[string][]float32) error {
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
		},
	}
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
	})

	worktreeCodebase := newCodebaseRecord(worktreeRoot)
	worktreeCodebase.Status = model.CodebaseStatusIndexed
	worktreeCodebase.CollectionName = "cc_feature"
	worktreeCodebase.EffectiveConfig = cfg
	worktreeCodebase.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 1, Status: "completed"}
	worktreeCodebase.MerkleSnapshotPath = manager.merklePath(worktreeCodebase.ID)

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

func cloneReuseVectors(reuse map[string][]float32) map[string][]float32 {
	cloned := make(map[string][]float32, len(reuse))
	for key, vector := range reuse {
		cloned[key] = append([]float32(nil), vector...)
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

func (f *fakeSemantic) requestedReuseCollection(collectionName string) bool {
	for _, call := range f.reuseCollectionsSnapshot() {
		for _, name := range call {
			if name == collectionName {
				return true
			}
		}
	}
	return false
}

package daemon

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func TestReadPathsDoNotMutateMissingCollectionState(t *testing.T) {
	t.Parallel()

	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}

	indexConfig := defaultIndexConfig()
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)

	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = indexConfig
	codebase.CollectionName = "missing_collection"
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	manager.semantic = &fakeSemantic{
		collectionName:       func(string) string { return "missing_collection" },
		listCollections:      func(context.Context) ([]string, error) { return []string{}, nil },
		hasCollectionForPath: func(context.Context, string) (bool, error) { return false, nil },
	}

	readCodebase, _, found, _, err := manager.GetIndex(context.Background(), repoPath)
	if err != nil {
		t.Fatalf("GetIndex returned error: %v", err)
	}
	if !found {
		t.Fatal("GetIndex stopped finding the tracked codebase")
	}
	if readCodebase.Status != model.CodebaseStatusIndexed {
		t.Fatalf("GetIndex mutated status to %q, want %q", readCodebase.Status, model.CodebaseStatusIndexed)
	}

	indexes := manager.ListIndexes(context.Background())
	if len(indexes) != 1 {
		t.Fatalf("ListIndexes returned %d codebases, want 1", len(indexes))
	}
	if indexes[0].Status != model.CodebaseStatusIndexed {
		t.Fatalf("ListIndexes mutated status to %q, want %q", indexes[0].Status, model.CodebaseStatusIndexed)
	}
	if jobs := manager.ListJobs(""); len(jobs) != 0 {
		t.Fatalf("read paths queued %d jobs, want 0", len(jobs))
	}
}

func TestListIndexesDoesNotMutateMissingCollectionState(t *testing.T) {
	t.Parallel()

	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}

	indexConfig := defaultIndexConfig()
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)

	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = indexConfig
	codebase.CollectionName = "missing_collection"
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	manager.semantic = &fakeSemantic{
		collectionName:       func(string) string { return "missing_collection" },
		listCollections:      func(context.Context) ([]string, error) { return []string{}, nil },
		hasCollectionForPath: func(context.Context, string) (bool, error) { return false, nil },
	}

	indexes := manager.ListIndexes(context.Background())
	if len(indexes) != 1 {
		t.Fatalf("ListIndexes returned %d codebases, want 1", len(indexes))
	}
	if indexes[0].Status != model.CodebaseStatusIndexed {
		t.Fatalf("ListIndexes mutated status to %q, want %q", indexes[0].Status, model.CodebaseStatusIndexed)
	}
	if jobs := manager.ListJobs(""); len(jobs) != 0 {
		t.Fatalf("ListIndexes queued %d jobs, want 0", len(jobs))
	}

	readCodebase, _, found, _, err := manager.GetIndex(context.Background(), repoPath)
	if err != nil {
		t.Fatalf("GetIndex returned error: %v", err)
	}
	if !found {
		t.Fatal("tracked codebase disappeared after ListIndexes")
	}
	if readCodebase.Status != model.CodebaseStatusIndexed {
		t.Fatalf("ListIndexes persisted status %q, want %q", readCodebase.Status, model.CodebaseStatusIndexed)
	}
}

func TestRunSyncAllRepairsMissingCollectionWithFullIndex(t *testing.T) {
	t.Parallel()

	manager, cfg, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}

	indexConfig := defaultIndexConfig()
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)

	codebase := newCodebaseRecord(canonical)
	codebase.ID = "cb-repair"
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = indexConfig
	codebase.CollectionName = "missing_collection"
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	started := make(chan struct{}, 1)
	release := make(chan struct{})
	manager.runner = fakeRunner{
		indexOne: func(ctx context.Context, root string, relativePath string, cfg model.IndexConfig) (indexer.OneFileResult, error) {
			select {
			case started <- struct{}{}:
			default:
			}
			<-release
			content := "package main\n"
			return indexer.OneFileResult{
				Chunks: []model.StoredChunk{{
					Content:       content,
					RelativePath:  relativePath,
					StartLine:     1,
					EndLine:       1,
					Language:      "go",
					FileExtension: ".go",
				}},
				FileHash: hashText(content),
				Skipped:  false,
				Removed:  false,
			}, nil
		},
	}
	manager.semantic = &fakeSemantic{
		collectionName:       func(string) string { return "missing_collection" },
		listCollections:      func(context.Context) ([]string, error) { return []string{}, nil },
		hasCollectionForPath: func(context.Context, string) (bool, error) { return false, nil },
	}

	syncer := NewBackgroundSync(cfg, manager)
	syncer.runSyncAll(context.Background(), "test")

	waitForCodebaseStatus(t, manager, repoPath, model.CodebaseStatusIndexing)
	<-started

	jobs := manager.ListJobs("")
	if len(jobs) != 1 {
		t.Fatalf("ListJobs returned %d jobs, want 1", len(jobs))
	}
	if jobOperation(jobs[0].Operation) != jobOperationIndex {
		t.Fatalf("repair job Operation = %q, want %q", jobs[0].Operation, jobOperationIndex)
	}

	syncer.runSyncAll(context.Background(), "test")
	if jobs := manager.ListJobs(""); len(jobs) != 1 {
		t.Fatalf("second repair pass queued %d jobs, want 1", len(jobs))
	}

	close(release)
	waitForCodebaseStatus(t, manager, repoPath, model.CodebaseStatusIndexed)
}

func TestRunSyncAllLeavesRegistryUntouchedWhenCollectionListFails(t *testing.T) {
	t.Parallel()

	manager, cfg, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}

	indexConfig := defaultIndexConfig()
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)

	codebase := newCodebaseRecord(canonical)
	codebase.ID = "cb-list-fail"
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = indexConfig
	codebase.CollectionName = "missing_collection"
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	if err := merkle.WriteSnapshot(
		manager.merklePath(codebase.ID),
		merkle.Snapshot{
			ConfigDigest: indexConfig.IgnoreDigest,
			Files:        map[string]string{"main.go": hashText("package main\n")},
			Inodes:       nil,
		},
	); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	manager.semantic = &fakeSemantic{
		collectionName:       func(string) string { return "missing_collection" },
		listCollections:      func(context.Context) ([]string, error) { return nil, errors.New("list failed") },
		hasCollectionForPath: func(context.Context, string) (bool, error) { return false, nil },
	}

	syncer := NewBackgroundSync(cfg, manager)
	syncer.runSyncAll(context.Background(), "test")

	readCodebase, _, found, _, err := manager.GetIndex(context.Background(), repoPath)
	if err != nil {
		t.Fatalf("GetIndex returned error: %v", err)
	}
	if !found {
		t.Fatal("GetIndex stopped finding the tracked codebase")
	}
	if readCodebase.Status != model.CodebaseStatusIndexed {
		t.Fatalf("repair-on-error mutated status to %q, want %q", readCodebase.Status, model.CodebaseStatusIndexed)
	}
	if jobs := manager.ListJobs(""); len(jobs) != 0 {
		t.Fatalf("repair-on-error queued %d jobs, want 0", len(jobs))
	}
	if _, statErr := os.Stat(manager.config.RegistryPath); statErr != nil {
		t.Fatalf("registry stat returned error: %v", statErr)
	}
}

func TestRunSyncAllLeavesRegistryUntouchedWhenSemanticUnavailable(t *testing.T) {
	t.Parallel()

	manager, cfg, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}

	indexConfig := defaultIndexConfig()
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)

	codebase := newCodebaseRecord(canonical)
	codebase.ID = "cb-unavailable"
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = indexConfig
	codebase.CollectionName = "missing_collection"
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.semantic = nil
	manager.mu.Unlock()
	if err := merkle.WriteSnapshot(
		manager.merklePath(codebase.ID),
		merkle.Snapshot{
			ConfigDigest: indexConfig.IgnoreDigest,
			Files:        map[string]string{"main.go": hashText("package main\n")},
			Inodes:       nil,
		},
	); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	syncer := NewBackgroundSync(cfg, manager)
	syncer.runSyncAll(context.Background(), "test")

	readCodebase, _, found, _, err := manager.GetIndex(context.Background(), repoPath)
	if err != nil {
		t.Fatalf("GetIndex returned error: %v", err)
	}
	if !found {
		t.Fatal("GetIndex stopped finding the tracked codebase")
	}
	if readCodebase.Status != model.CodebaseStatusIndexed {
		t.Fatalf("semantic-unavailable repair mutated status to %q, want %q", readCodebase.Status, model.CodebaseStatusIndexed)
	}
	if jobs := manager.ListJobs(""); len(jobs) != 0 {
		t.Fatalf("semantic-unavailable repair queued %d jobs, want 0", len(jobs))
	}
}

func TestForceReindexUnchangedRepoBootstrapsWhenCollectionDisappears(t *testing.T) {
	t.Parallel()

	manager, _, repoPath := newTestManager(t)
	collectionProbeCount := atomic.Int32{}
	manager.semantic = &fakeSemantic{
		collectionName: func(string) string { return "force_collection" },
		count:          func(context.Context, string) (int32, error) { return 1, nil },
		hasCollectionForPath: func(context.Context, string) (bool, error) {
			switch collectionProbeCount.Add(1) {
			case 1:
				return false, nil
			case 2:
				return true, nil
			case 3:
				return false, nil
			default:
				return false, nil
			}
		},
		reindexEmit: func(progress func(semantic.Progress)) {
			progress(semantic.Progress{ChunksProcessed: 1, ChunksReused: 0, ChunksEmbedded: 1})
		},
	}
	manager.runner = fakeRunner{
		indexOne: func(ctx context.Context, root string, relativePath string, cfg model.IndexConfig) (indexer.OneFileResult, error) {
			content := "package main\n"
			return indexer.OneFileResult{
				Chunks: []model.StoredChunk{{
					Content:       content,
					RelativePath:  relativePath,
					StartLine:     1,
					EndLine:       1,
					Language:      "go",
					FileExtension: ".go",
				}},
				FileHash: hashText(content),
				Skipped:  false,
				Removed:  false,
			}, nil
		},
	}

	if _, _, _, _, err := manager.StartIndex(context.Background(), repoPath, testClientInfo(), defaultIndexConfig(), false); err != nil {
		t.Fatalf("initial StartIndex returned error: %v", err)
	}
	waitForCodebaseStatus(t, manager, repoPath, model.CodebaseStatusIndexed)

	embeddedCount := atomic.Int32{}
	manager.runner = fakeRunner{
		indexOne: func(ctx context.Context, root string, relativePath string, cfg model.IndexConfig) (indexer.OneFileResult, error) {
			embeddedCount.Add(1)
			content := "package main\n"
			return indexer.OneFileResult{
				Chunks: []model.StoredChunk{{
					Content:       content,
					RelativePath:  relativePath,
					StartLine:     1,
					EndLine:       1,
					Language:      "go",
					FileExtension: ".go",
				}},
				FileHash: hashText(content),
				Skipped:  false,
				Removed:  false,
			}, nil
		},
	}

	job, _, _, _, err := manager.StartIndex(context.Background(), repoPath, testClientInfo(), defaultIndexConfig(), true)
	if err != nil {
		t.Fatalf("force StartIndex returned error: %v", err)
	}
	if jobOperation(job.Operation) != jobOperationStreamingReindex {
		t.Fatalf("force job Operation = %q, want %q", job.Operation, jobOperationStreamingReindex)
	}
	waitForCodebaseStatus(t, manager, repoPath, model.CodebaseStatusIndexed)

	if embeddedCount.Load() == 0 {
		t.Fatal("force reindex did not embed any files after collection loss")
	}
	completedJob, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("GetJob(%s) not found", job.ID)
	}
	if completedJob.Progress.FilesEmbedded == 0 {
		t.Fatalf("completed job FilesEmbedded = %d, want > 0", completedJob.Progress.FilesEmbedded)
	}
	if completedJob.Progress.ChunksGenerated == 0 {
		t.Fatalf("completed job ChunksGenerated = %d, want > 0", completedJob.Progress.ChunksGenerated)
	}
}

func TestStaleRetryUnchangedRepoBootstrapsWhenCollectionDisappears(t *testing.T) {
	t.Parallel()

	manager, _, repoPath := newTestManager(t)
	manager.semantic = &fakeSemantic{
		collectionName:       func(string) string { return "stale_collection" },
		count:                func(context.Context, string) (int32, error) { return 1, nil },
		hasCollectionForPath: func(context.Context, string) (bool, error) { return false, nil },
		reindexEmit: func(progress func(semantic.Progress)) {
			progress(semantic.Progress{ChunksProcessed: 1, ChunksReused: 0, ChunksEmbedded: 1})
		},
	}
	manager.runner = fakeRunner{
		indexOne: func(ctx context.Context, root string, relativePath string, cfg model.IndexConfig) (indexer.OneFileResult, error) {
			content := "package main\n"
			return indexer.OneFileResult{
				Chunks: []model.StoredChunk{{
					Content:       content,
					RelativePath:  relativePath,
					StartLine:     1,
					EndLine:       1,
					Language:      "go",
					FileExtension: ".go",
				}},
				FileHash: hashText(content),
				Skipped:  false,
				Removed:  false,
			}, nil
		},
	}

	if _, _, _, _, err := manager.StartIndex(context.Background(), repoPath, testClientInfo(), defaultIndexConfig(), false); err != nil {
		t.Fatalf("initial StartIndex returned error: %v", err)
	}
	waitForCodebaseStatus(t, manager, repoPath, model.CodebaseStatusIndexed)

	codebase, _, found, _, err := manager.GetIndex(context.Background(), repoPath)
	if err != nil || !found {
		t.Fatalf("GetIndex returned err=%v found=%v", err, found)
	}
	manager.mu.Lock()
	codebase.Status = model.CodebaseStatusStale
	manager.codebases[codebase.ID] = codebase
	if saveErr := manager.saveLocked(); saveErr != nil {
		manager.mu.Unlock()
		t.Fatalf("saveLocked returned error: %v", saveErr)
	}
	manager.mu.Unlock()

	collectionProbeCount := atomic.Int32{}
	manager.semantic = &fakeSemantic{
		collectionName: func(string) string { return "stale_collection" },
		count:          func(context.Context, string) (int32, error) { return 1, nil },
		hasCollectionForPath: func(context.Context, string) (bool, error) {
			switch collectionProbeCount.Add(1) {
			case 1:
				return true, nil
			case 2:
				return false, nil
			default:
				return false, nil
			}
		},
		reindexEmit: func(progress func(semantic.Progress)) {
			progress(semantic.Progress{ChunksProcessed: 1, ChunksReused: 0, ChunksEmbedded: 1})
		},
	}

	embeddedCount := atomic.Int32{}
	manager.runner = fakeRunner{
		indexOne: func(ctx context.Context, root string, relativePath string, cfg model.IndexConfig) (indexer.OneFileResult, error) {
			embeddedCount.Add(1)
			content := "package main\n"
			return indexer.OneFileResult{
				Chunks: []model.StoredChunk{{
					Content:       content,
					RelativePath:  relativePath,
					StartLine:     1,
					EndLine:       1,
					Language:      "go",
					FileExtension: ".go",
				}},
				FileHash: hashText(content),
				Skipped:  false,
				Removed:  false,
			}, nil
		},
	}

	job, _, _, _, err := manager.StartIndex(context.Background(), repoPath, testClientInfo(), defaultIndexConfig(), false)
	if err != nil {
		t.Fatalf("stale retry StartIndex returned error: %v", err)
	}
	if jobOperation(job.Operation) != jobOperationStreamingReindex {
		t.Fatalf("stale retry job Operation = %q, want %q", job.Operation, jobOperationStreamingReindex)
	}
	waitForCodebaseStatus(t, manager, repoPath, model.CodebaseStatusIndexed)

	if embeddedCount.Load() == 0 {
		t.Fatal("stale retry did not embed any files after collection loss")
	}
	completedJob, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("GetJob(%s) not found", job.ID)
	}
	if completedJob.Progress.FilesEmbedded == 0 {
		t.Fatalf("completed stale retry FilesEmbedded = %d, want > 0", completedJob.Progress.FilesEmbedded)
	}
	if completedJob.Progress.ChunksGenerated == 0 {
		t.Fatalf("completed stale retry ChunksGenerated = %d, want > 0", completedJob.Progress.ChunksGenerated)
	}
}

func TestSearchCodeMissingCollectionReportsAutomaticRepairWithoutQueuingWork(t *testing.T) {
	t.Parallel()

	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}

	indexConfig := defaultIndexConfig()
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)
	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusStale
	codebase.EffectiveConfig = indexConfig
	codebase.CollectionName = "missing_collection"
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	manager.semantic = &fakeSemantic{
		search: func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error) {
			return nil, semantic.ErrCollectionMissing
		},
	}

	server := NewGRPCServer(manager, nil)
	response, err := server.SearchCode(context.Background(), &pb.SearchCodeRequest{
		Path:  repoPath,
		Query: "SearchableThing",
		Limit: 5,
	})
	if err != nil {
		t.Fatalf("SearchCode returned error: %v", err)
	}
	if strings.Contains(response.GetDisplayText(), "force=true") {
		t.Fatalf("search display text still mentions force=true: %q", response.GetDisplayText())
	}
	if !strings.Contains(strings.ToLower(response.GetDisplayText()), "automatic rebuild") {
		t.Fatalf("search display text missing automatic rebuild note: %q", response.GetDisplayText())
	}
	if jobs := manager.ListJobs(""); len(jobs) != 0 {
		t.Fatalf("search queued %d jobs, want 0", len(jobs))
	}
}

func TestSearchCodeMissingCollectionErrorDoesNotMentionForceTrue(t *testing.T) {
	t.Parallel()

	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}

	indexConfig := defaultIndexConfig()
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)
	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = indexConfig
	codebase.CollectionName = "missing_collection"
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	manager.semantic = &fakeSemantic{
		search: func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error) {
			return nil, semantic.ErrCollectionMissing
		},
	}

	_, err = manager.SearchCode(context.Background(), repoPath, "SearchableThing", 5, nil)
	if err == nil {
		t.Fatal("SearchCode returned nil error, want missing collection error")
	}
	if strings.Contains(err.Error(), "force=true") {
		t.Fatalf("search error still mentions force=true: %q", err.Error())
	}
}

func TestCollectionLossDoesNotPruneUntilClearIndex(t *testing.T) {
	t.Parallel()

	manager, cfg, repoPath := newTestManager(t)
	initialSemantic := &fakeSemantic{
		collectionName: func(string) string { return "collection_loss_test" },
		count:          func(context.Context, string) (int32, error) { return 1, nil },
	}
	manager.semantic = initialSemantic
	manager.runner = fakeRunner{
		indexOne: func(ctx context.Context, root string, relativePath string, cfg model.IndexConfig) (indexer.OneFileResult, error) {
			content := "package main\n"
			return indexer.OneFileResult{
				Chunks: []model.StoredChunk{{
					Content:       content,
					RelativePath:  relativePath,
					StartLine:     1,
					EndLine:       1,
					Language:      "go",
					FileExtension: ".go",
				}},
				FileHash: hashText(content),
				Skipped:  false,
				Removed:  false,
			}, nil
		},
	}

	_, codebase, _, _, err := manager.StartIndex(context.Background(), repoPath, testClientInfo(), defaultIndexConfig(), false)
	if err != nil {
		t.Fatalf("StartIndex returned error: %v", err)
	}
	waitForCodebaseStatus(t, manager, repoPath, model.CodebaseStatusIndexed)

	chunkPath := filepath.Join(cfg.ChunksDir, codebase.ID+".json")
	merklePath := filepath.Join(cfg.MerkleDir, codebase.ID+".json")
	release := make(chan struct{})
	missingSemantic := &fakeSemantic{
		collectionName:       func(string) string { return "collection_loss_test" },
		listCollections:      func(context.Context) ([]string, error) { return []string{}, nil },
		hasCollectionForPath: func(context.Context, string) (bool, error) { return false, nil },
		search: func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error) {
			return nil, semantic.ErrCollectionMissing
		},
	}
	manager.semantic = missingSemantic
	manager.runner = fakeRunner{
		indexOne: func(ctx context.Context, root string, relativePath string, cfg model.IndexConfig) (indexer.OneFileResult, error) {
			<-release
			content := "package main\n"
			return indexer.OneFileResult{
				Chunks: []model.StoredChunk{{
					Content:       content,
					RelativePath:  relativePath,
					StartLine:     1,
					EndLine:       1,
					Language:      "go",
					FileExtension: ".go",
				}},
				FileHash: hashText(content),
				Skipped:  false,
				Removed:  false,
			}, nil
		},
	}

	if _, _, found, _, err := manager.GetIndex(context.Background(), repoPath); err != nil || !found {
		t.Fatalf("GetIndex returned err=%v found=%v", err, found)
	}
	if indexes := manager.ListIndexes(context.Background()); len(indexes) != 1 {
		t.Fatalf("ListIndexes returned %d entries, want 1", len(indexes))
	}

	syncer := NewBackgroundSync(cfg, manager)
	syncer.runSyncAll(context.Background(), "test")
	waitForCodebaseStatus(t, manager, repoPath, model.CodebaseStatusIndexing)

	server := NewGRPCServer(manager, nil)
	searchResponse, err := server.SearchCode(context.Background(), &pb.SearchCodeRequest{
		Path:  repoPath,
		Query: "SmokeNeedle",
		Limit: 5,
	})
	if err != nil {
		t.Fatalf("SearchCode returned error: %v", err)
	}
	if !strings.Contains(strings.ToLower(searchResponse.GetDisplayText()), "automatic rebuild") {
		t.Fatalf("search output missing automatic rebuild note: %q", searchResponse.GetDisplayText())
	}

	if _, _, found, _, err := manager.GetIndex(context.Background(), repoPath); err != nil || !found {
		t.Fatalf("GetIndex after repair returned err=%v found=%v", err, found)
	}
	if _, err := os.Stat(chunkPath); err != nil {
		t.Fatalf("chunk file missing before clear: %v", err)
	}
	if _, err := os.Stat(merklePath); err != nil {
		t.Fatalf("merkle file missing before clear: %v", err)
	}

	if _, err := manager.ClearIndex(context.Background(), repoPath, testClientInfo()); err != nil {
		t.Fatalf("ClearIndex returned error: %v", err)
	}
	close(release)

	if _, err := os.Stat(chunkPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("chunk file still present after clear: %v", err)
	}
	if _, err := os.Stat(merklePath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("merkle file still present after clear: %v", err)
	}
	if len(missingSemantic.dropped) != 1 || missingSemantic.dropped[0] != codebase.CanonicalPath {
		t.Fatalf("semantic drop calls = %v, want [%s]", missingSemantic.dropped, codebase.CanonicalPath)
	}
	if _, _, found, _, err := manager.GetIndex(context.Background(), repoPath); err != nil || found {
		t.Fatalf("GetIndex after clear returned err=%v found=%v, want found=false", err, found)
	}
}

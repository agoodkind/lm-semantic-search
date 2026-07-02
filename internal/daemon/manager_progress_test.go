package daemon

import (
	"context"
	"testing"

	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// TestDeltaProgressAccumulatesReuseSplit proves that the per-file reindex folds
// each file's reused-vs-embedded chunk split into the run totals, and that
// reportDeltaProgress carries those totals onto the job's model.Progress with
// ChunksReused holding the reused count and ChunksEmbedded holding the embedded
// count.
func TestDeltaProgressAccumulatesReuseSplit(t *testing.T) {
	manager, _, _ := newTestManager(t)

	emissions := []semantic.Progress{
		{Phase: "", OverallPercent: 0, EmbeddingBatchesTotal: 0, EmbeddingBatchesCompleted: 0, CollectionRowsWritten: 4200, ChunksProcessed: 4200, ChunksReused: 4000, ChunksEmbedded: 200},
		{Phase: "", OverallPercent: 0, EmbeddingBatchesTotal: 0, EmbeddingBatchesCompleted: 0, CollectionRowsWritten: 3261, ChunksProcessed: 3261, ChunksReused: 2989, ChunksEmbedded: 272},
	}
	call := 0
	manager.semantic = &fakeSemantic{
		reindexEmit: func(progress func(semantic.Progress)) {
			progress(emissions[call])
			call++
		},
	}

	cfg := defaultIndexConfig()
	codebaseID := "cb-progress"
	canonical := "/tmp/cb-progress"
	manager.mu.Lock()
	manager.codebases[codebaseID] = model.Codebase{
		ID:              codebaseID,
		CanonicalPath:   canonical,
		Status:          model.CodebaseStatusIndexing,
		EffectiveConfig: cfg,
	}
	manager.mu.Unlock()

	job := newQueuedJob(codebaseID, canonical, canonical, testClientInfo(), string(jobOperationIndex), false, cfg, emptyAdmissionBudget, clock.Now())
	manager.mu.Lock()
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	state := deltaState{
		semantic:    true,
		staging:     false,
		reuse:       nil,
		chunkCounts: &chunkCounters{reused: 0, embedded: 0},
	}

	for _, relativePath := range []string{"a.go", "b.go"} {
		chunks := []model.StoredChunk{{Content: "x", RelativePath: relativePath}}
		if outcome := manager.applyReindexForState(context.Background(), job, state, chunks, semantic.RemovePaths([]string{relativePath}), "test reindex"); outcome.fallback || outcome.handled {
			t.Fatalf("applyReindexForState returned terminal outcome %+v", outcome)
		}
	}

	processed, reused, embedded, loaded := state.chunkSplit()
	manager.reportDeltaProgress(job.ID, 2, 2, 2, indexer.Result{IndexedFiles: 2, TotalChunks: 7461}, processed, reused, embedded, loaded, "file")

	got, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("GetJob(%s) not found", job.ID)
	}
	if got.Progress.ChunksReused != 6989 {
		t.Fatalf("Progress.ChunksReused = %d, want 6989", got.Progress.ChunksReused)
	}
	if got.Progress.ChunksProcessed != 7461 {
		t.Fatalf("Progress.ChunksProcessed = %d, want 7461", got.Progress.ChunksProcessed)
	}
	if got.Progress.ChunksEmbedded != 472 {
		t.Fatalf("Progress.ChunksEmbedded = %d, want 472", got.Progress.ChunksEmbedded)
	}
	if got.Progress.ChunksGenerated != 472 {
		t.Fatalf("Progress.ChunksGenerated = %d, want 472", got.Progress.ChunksGenerated)
	}
}

func TestCodeItemReuseLoadsExactPath(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	fake := &fakeSemantic{
		loadReuseForPath: func(_ context.Context, collectionName string, relativePath string) (map[string][]float32, error) {
			if collectionName != "code_chunks_live" {
				t.Fatalf("collectionName = %q, want code_chunks_live", collectionName)
			}
			if relativePath != "src/file.go" {
				t.Fatalf("relativePath = %q, want src/file.go", relativePath)
			}
			return map[string][]float32{"target-hash": {1, 2, 3}}, nil
		},
		loadReuseForPrefix: func(_ context.Context, _ string, prefix string) (map[string][]float32, error) {
			t.Fatalf("code reuse called prefix loader with %q; want exact path loader", prefix)
			return nil, nil
		},
	}
	manager.semantic = fake

	source := newCodeItemSource(manager.runner, manager.indexability, "cb", "/tmp/code", defaultIndexConfig()).withCollectionName("code_chunks_live")
	state := deltaState{
		source:           source,
		semantic:         true,
		itemReuseEnabled: true,
		reuse:            map[string][]float32{"base-hash": {9}},
		chunkCounts:      &chunkCounters{},
	}

	reuse, loaded := manager.itemReuse(context.Background(), state, "src/file.go")
	if loaded != 1 {
		t.Fatalf("loaded = %d, want 1 exact-path reuse candidate", loaded)
	}
	if _, ok := reuse["target-hash"]; !ok {
		t.Fatalf("reuse map missing exact-path candidate: %v", reuse)
	}
	if _, ok := reuse["base-hash"]; !ok {
		t.Fatalf("reuse map lost base candidate: %v", reuse)
	}
	pathCalls := fake.reusePathCallsSnapshot()
	if len(pathCalls) != 1 || pathCalls[0].Path != "src/file.go" {
		t.Fatalf("path reuse calls = %+v, want one call for src/file.go", pathCalls)
	}
	prefixCalls := fake.reusePrefixCallsSnapshot()
	if len(prefixCalls) != 0 {
		t.Fatalf("prefix reuse calls = %+v, want none for code reuse", prefixCalls)
	}
}

func TestItemReuseSkipsPerFileLoadsDuringBootstrap(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	fake := &fakeSemantic{
		loadReuseForPath: func(_ context.Context, _ string, _ string) (map[string][]float32, error) {
			t.Fatal("bootstrap item reuse should not probe the live collection by exact path")
			return nil, nil
		},
		loadReuseForPrefix: func(_ context.Context, _ string, _ string) (map[string][]float32, error) {
			t.Fatal("bootstrap item reuse should not probe the live collection by prefix")
			return nil, nil
		},
	}
	manager.semantic = fake

	source := newCodeItemSource(manager.runner, manager.indexability, "cb", "/tmp/code", defaultIndexConfig()).withCollectionName("code_chunks_live")
	state := deltaState{
		source:      source,
		semantic:    true,
		staging:     true,
		reuse:       map[string][]float32{"base-hash": {9}},
		chunkCounts: &chunkCounters{},
	}

	reuse, loaded := manager.itemReuse(context.Background(), state, "src/file.go")
	if loaded != 0 {
		t.Fatalf("loaded = %d, want 0 during bootstrap", loaded)
	}
	if len(reuse) != 1 || reuse["base-hash"][0] != 9 {
		t.Fatalf("reuse = %v, want the unchanged build-wide reuse map", reuse)
	}
	if pathCalls := fake.reusePathCallsSnapshot(); len(pathCalls) != 0 {
		t.Fatalf("path reuse calls = %+v, want none during bootstrap", pathCalls)
	}
	if prefixCalls := fake.reusePrefixCallsSnapshot(); len(prefixCalls) != 0 {
		t.Fatalf("prefix reuse calls = %+v, want none during bootstrap", prefixCalls)
	}
}

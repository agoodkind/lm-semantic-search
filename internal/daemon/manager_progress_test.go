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
// ChunksReused holding the reused count and ChunksGenerated the embedded count.
func TestDeltaProgressAccumulatesReuseSplit(t *testing.T) {
	manager, _, _ := newTestManager(t)

	emissions := []semantic.Progress{
		{Phase: "", OverallPercent: 0, EmbeddingBatchesTotal: 0, EmbeddingBatchesCompleted: 0, CollectionRowsWritten: 0, ChunksReused: 4000, ChunksEmbedded: 200},
		{Phase: "", OverallPercent: 0, EmbeddingBatchesTotal: 0, EmbeddingBatchesCompleted: 0, CollectionRowsWritten: 0, ChunksReused: 2989, ChunksEmbedded: 272},
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

	job := newQueuedJob(codebaseID, canonical, canonical, testClientInfo(), string(jobOperationIndex), false, cfg, clock.Now())
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

	reused, embedded := state.chunkSplit()
	manager.reportDeltaProgress(job.ID, 2, 2, 2, indexer.Result{IndexedFiles: 2, TotalChunks: 7461}, reused, embedded, "file")

	got, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("GetJob(%s) not found", job.ID)
	}
	if got.Progress.ChunksReused != 6989 {
		t.Fatalf("Progress.ChunksReused = %d, want 6989", got.Progress.ChunksReused)
	}
	if got.Progress.ChunksGenerated != 472 {
		t.Fatalf("Progress.ChunksGenerated = %d, want 472", got.Progress.ChunksGenerated)
	}
}

package localvec

import (
	"context"
	"fmt"
	"os"
	"slices"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func TestLoadReuseVectorsForContentsUsesCandidateIndexNotRowScan(t *testing.T) {
	t.Parallel()

	const codebasePath = "/tmp/localvec-candidate-index"
	provider := &fakeEmbeddingProvider{vectors: map[string][]float32{
		"first":  {1, 0},
		"second": {0, 1},
	}}
	store, err := newStoreWithProvider(
		config.Config{StateRoot: t.TempDir()},
		provider,
	)
	if err != nil {
		t.Fatalf("newStoreWithProvider returned error: %v", err)
	}
	stageAndPromote(
		t,
		store,
		codebasePath,
		[]model.StoredChunk{
			{Content: "first", RelativePath: "first.go"},
			{Content: "second", RelativePath: "second.go"},
		},
		semantic.StoreColumnSetCode,
	)

	stored, err := store.collectionForName(store.CollectionName(codebasePath), false)
	if err != nil {
		t.Fatalf("collectionForName returned error: %v", err)
	}
	stored.mutex.Lock()
	if err := stored.loadLocked(); err != nil {
		stored.mutex.Unlock()
		t.Fatalf("loadLocked returned error: %v", err)
	}
	if len(stored.rows) != 2 {
		stored.mutex.Unlock()
		t.Fatalf("stored rows = %d, want 2", len(stored.rows))
	}
	candidateContent := stored.rows[0].Content
	candidateKey := stored.rows[0].ContentVectorKey
	wantVector := slices.Clone(stored.rows[0].Vector)
	// Poison the later row after collection state has been built. A row scan
	// observes this duplicate key and returns the wrong vector. The candidate
	// index keeps the original key-to-row position.
	stored.rows[1].ContentVectorKey = candidateKey
	stored.mutex.Unlock()

	reuse, err := store.LoadReuseVectorsForContents(
		context.Background(),
		store.CollectionName(codebasePath),
		[]model.StoredChunk{{Content: candidateContent}},
	)
	if err != nil {
		t.Fatalf("LoadReuseVectorsForContents returned error: %v", err)
	}
	gotVector := reuse[semantic.ContentVectorKey(candidateContent)]
	if !slices.Equal(gotVector, wantVector) {
		t.Fatalf("reused vector = %v, want indexed vector %v", gotVector, wantVector)
	}
}

func TestLoadReuseVectorsForContentsDoesNotCopyWholeCollection(t *testing.T) {
	const (
		collectionName = "local_code_chunks_allocation"
		rowCount       = 256
		vectorSize     = 1024
	)
	store, err := newStoreWithProvider(
		config.Config{StateRoot: t.TempDir()},
		nil,
	)
	if err != nil {
		t.Fatalf("newStoreWithProvider returned error: %v", err)
	}
	rows := make([]row, rowCount)
	for rowIndex := range rows {
		content := fmt.Sprintf("candidate-%d", rowIndex)
		rows[rowIndex] = row{
			Content:          content,
			ContentVectorKey: semantic.ContentVectorKey(content),
			Vector:           make([]float32, vectorSize),
		}
	}
	stored := newCollection(collectionName, "")
	stored.rows = rows
	stored.reuseRows = buildReuseRowIndex(rows)
	stored.loaded = true
	stored.exists = true
	store.collections[collectionName] = stored

	candidate := model.StoredChunk{Content: rows[0].Content}
	var latest map[string][]float32
	allocations := testing.AllocsPerRun(10, func() {
		latest, err = store.LoadReuseVectorsForContents(
			context.Background(),
			collectionName,
			[]model.StoredChunk{candidate},
		)
	})
	if err != nil {
		t.Fatalf("LoadReuseVectorsForContents returned error: %v", err)
	}
	if len(latest) != 1 {
		t.Fatalf("reused vectors = %d, want 1", len(latest))
	}
	if allocations > 10 {
		t.Fatalf("allocations per candidate lookup = %.0f, want at most 10", allocations)
	}
}

func TestLoadReuseVectorsForContentsRebuildsIndexAfterMutationAndReload(t *testing.T) {
	t.Parallel()

	const codebasePath = "/tmp/localvec-candidate-index-rebuild"
	provider := &fakeEmbeddingProvider{vectors: map[string][]float32{
		"first":  {1, 0},
		"second": {0, 1},
	}}
	configuration := config.Config{StateRoot: t.TempDir()}
	store, err := newStoreWithProvider(configuration, provider)
	if err != nil {
		t.Fatalf("newStoreWithProvider returned error: %v", err)
	}
	stageAndPromote(
		t,
		store,
		codebasePath,
		[]model.StoredChunk{{Content: "first", RelativePath: "first.go"}},
		semantic.StoreColumnSetCode,
	)
	if err := store.Reindex(
		context.Background(),
		codebasePath,
		[]model.StoredChunk{{Content: "second", RelativePath: "second.go"}},
		semantic.Removal{},
		nil,
		nil,
		semantic.StoreColumnSetCode,
	); err != nil {
		t.Fatalf("Reindex returned error: %v", err)
	}
	assertReusedVector(t, store, codebasePath, "second", []float32{0, 1})

	reloaded, err := newStoreWithProvider(configuration, provider)
	if err != nil {
		t.Fatalf("newStoreWithProvider for reload returned error: %v", err)
	}
	assertReusedVector(t, reloaded, codebasePath, "second", []float32{0, 1})
}

func TestLocalReuseLookupP95BelowConfiguredEmbedding(t *testing.T) {
	if os.Getenv("LMS_LOCALVEC_REUSE_PERF") != "1" {
		t.Skip("set LMS_LOCALVEC_REUSE_PERF=1 to measure the configured embedder")
	}
	configuration, err := config.Default()
	if err != nil {
		t.Fatalf("load configured embedder: %v", err)
	}
	configuration.StateRoot = t.TempDir()
	store, err := New(context.Background(), configuration)
	if err != nil {
		t.Fatalf("create local store with configured embedder: %v", err)
	}

	const (
		codebasePath = "/tmp/localvec-reuse-performance"
		sampleCount  = 20
	)
	contents := make([]string, 0, sampleCount)
	chunks := make([]model.StoredChunk, 0, sampleCount)
	for sampleIndex := range sampleCount {
		content := fmt.Sprintf("harmless local reuse performance control %02d", sampleIndex)
		contents = append(contents, content)
		chunks = append(chunks, model.StoredChunk{
			Content:      content,
			RelativePath: fmt.Sprintf("control-%02d.txt", sampleIndex),
		})
	}
	stageAndPromote(t, store, codebasePath, chunks, semantic.StoreColumnSetCode)
	if _, err := store.embedder.EmbedBatch(
		context.Background(),
		[]string{"harmless local reuse warmup"},
	); err != nil {
		t.Fatalf("warm configured embedder: %v", err)
	}
	if _, err := store.LoadReuseVectorsForContents(
		context.Background(),
		store.CollectionName(codebasePath),
		[]model.StoredChunk{{Content: contents[0]}},
	); err != nil {
		t.Fatalf("warm candidate lookup: %v", err)
	}

	lookupDurations := make([]time.Duration, 0, sampleCount)
	embedDurations := make([]time.Duration, 0, sampleCount)
	for _, content := range contents {
		lookupStarted := time.Now()
		reuse, lookupErr := store.LoadReuseVectorsForContents(
			context.Background(),
			store.CollectionName(codebasePath),
			[]model.StoredChunk{{Content: content}},
		)
		lookupDurations = append(lookupDurations, time.Since(lookupStarted))
		if lookupErr != nil {
			t.Fatalf("candidate lookup for %q: %v", content, lookupErr)
		}
		if len(reuse) != 1 {
			t.Fatalf("candidate lookup for %q returned %d vectors, want 1", content, len(reuse))
		}

		embedStarted := time.Now()
		result, embedErr := store.embedder.EmbedBatch(context.Background(), []string{content})
		embedDurations = append(embedDurations, time.Since(embedStarted))
		if embedErr != nil {
			t.Fatalf("embed performance control %q: %v", content, embedErr)
		}
		if len(result.Vectors) != 1 || len(result.Vectors[0]) == 0 {
			t.Fatalf("embed performance control %q returned no vector", content)
		}
	}

	lookupP95 := localDurationPercentile(lookupDurations, 95)
	embedP95 := localDurationPercentile(embedDurations, 95)
	ratio := float64(lookupP95) / float64(embedP95)
	t.Logf(
		"lookup_ms min=%.3f p50=%.3f p95=%.3f max=%.3f; embed_ms min=%.3f p50=%.3f p95=%.3f max=%.3f; ratio=%.4f",
		localDurationMilliseconds(localDurationPercentile(lookupDurations, 0)),
		localDurationMilliseconds(localDurationPercentile(lookupDurations, 50)),
		localDurationMilliseconds(lookupP95),
		localDurationMilliseconds(localDurationPercentile(lookupDurations, 100)),
		localDurationMilliseconds(localDurationPercentile(embedDurations, 0)),
		localDurationMilliseconds(localDurationPercentile(embedDurations, 50)),
		localDurationMilliseconds(embedP95),
		localDurationMilliseconds(localDurationPercentile(embedDurations, 100)),
		ratio,
	)
	if lookupP95*10 >= embedP95 {
		t.Fatalf(
			"candidate lookup p95 %s is %.2f%% of embedding p95 %s, want below 10%%",
			lookupP95,
			ratio*100,
			embedP95,
		)
	}
}

func localDurationPercentile(values []time.Duration, percentile int) time.Duration {
	sorted := slices.Clone(values)
	slices.Sort(sorted)
	if percentile <= 0 {
		return sorted[0]
	}
	index := (len(sorted)*percentile+99)/100 - 1
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return sorted[index]
}

func localDurationMilliseconds(duration time.Duration) float64 {
	return float64(duration) / float64(time.Millisecond)
}

func assertReusedVector(
	t *testing.T,
	store *Store,
	codebasePath string,
	content string,
	want []float32,
) {
	t.Helper()
	reuse, err := store.LoadReuseVectorsForContents(
		context.Background(),
		store.CollectionName(codebasePath),
		[]model.StoredChunk{{Content: content}},
	)
	if err != nil {
		t.Fatalf("LoadReuseVectorsForContents returned error: %v", err)
	}
	got := reuse[semantic.ContentVectorKey(content)]
	if !slices.Equal(got, want) {
		t.Fatalf("reused vector = %v, want %v", got, want)
	}
}

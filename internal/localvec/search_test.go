package localvec

import (
	"context"
	"fmt"
	"math"
	"slices"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func TestSearchAppliesExtensionAndRelativePathPrefixFilters(t *testing.T) {
	t.Parallel()

	const codebasePath = "/tmp/localvec-search-filters"
	provider := &fakeEmbeddingProvider{
		vectors: map[string][]float32{
			"best outside": {1, 0},
			"best wrong":   {0.9, 0.1},
			"kept":         {0.8, 0.2},
			"query":        {1, 0},
		},
	}
	store, err := newStoreWithProvider(
		config.Config{StateRoot: t.TempDir()},
		provider,
	)
	if err != nil {
		t.Fatalf("newStoreWithProvider returned error: %v", err)
	}
	chunks := []model.StoredChunk{
		{
			Content:       "best outside",
			RelativePath:  "other/outside.go",
			FileExtension: ".go",
		},
		{
			Content:       "best wrong",
			RelativePath:  "scope/wrong.py",
			FileExtension: ".py",
		},
		{
			Content:       "kept",
			RelativePath:  "scope/kept.go",
			FileExtension: ".go",
		},
	}
	stageAndPromote(t, store, codebasePath, chunks, semantic.CodeColumns())

	results, err := store.Search(
		context.Background(),
		codebasePath,
		"query",
		10,
		[]string{"go"},
		"scope",
	)
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	if len(results) != 1 || results[0].Content != "kept" {
		t.Fatalf("Search results = %+v, want only kept", results)
	}
}

func hitContents(hits []semantic.CollectionHit) []string {
	contents := make([]string, 0, len(hits))
	for _, hit := range hits {
		contents = append(contents, hit.Chunk.Content)
	}
	return contents
}

func TestSearchAboveExactThresholdReturnsNearestNeighbor(t *testing.T) {
	t.Parallel()

	const (
		codebasePath = "/tmp/localvec-hnsw-nearest"
		resultLimit  = 8
	)
	store := newSearchTestStore(t)
	chunks, reuse := largeSearchFixture(exactSearchThreshold + 1)
	stageAndPromoteWithReuse(t, store, codebasePath, chunks, reuse)

	results, err := store.Search(
		context.Background(),
		codebasePath,
		"query",
		resultLimit,
		nil,
		"",
	)
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	if len(results) != resultLimit {
		t.Fatalf("Search returned %d results, want %d", len(results), resultLimit)
	}
	if !slices.ContainsFunc(results, func(chunk model.StoredChunk) bool {
		return chunk.Content == "nearest"
	}) {
		t.Fatalf("Search results = %+v, want nearest in top %d", results, resultLimit)
	}
}

func TestSearchAboveExactThresholdAdaptivelyOverfetchesAfterFiltering(t *testing.T) {
	t.Parallel()

	const (
		codebasePath  = "/tmp/localvec-hnsw-filter"
		resultLimit   = 8
		filteredCount = resultLimit*initialSearchOverfetchFactor + 1
	)
	store := newSearchTestStore(t)
	chunks, reuse := largeFilteredSearchFixture(
		exactSearchThreshold+1,
		filteredCount,
	)
	stageAndPromoteWithReuse(t, store, codebasePath, chunks, reuse)

	results, err := store.Search(
		context.Background(),
		codebasePath,
		"query",
		resultLimit,
		[]string{".go"},
		"",
	)
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	if len(results) != 1 || results[0].Content != "kept" {
		t.Fatalf("Search results = %+v, want kept", results)
	}
}

const (
	tiedDenseConversation = "claude:dense"
	tiedFarConversation   = "claude:far"
	tiedDenseRows         = 300
	tiedFarRows           = 3
)

func newSearchTestStore(t *testing.T) *Store {
	t.Helper()
	store, err := newStoreWithProvider(
		config.Config{StateRoot: t.TempDir()},
		&fakeEmbeddingProvider{
			vectors: map[string][]float32{"query": {1, 0}},
		},
	)
	if err != nil {
		t.Fatalf("newStoreWithProvider returned error: %v", err)
	}
	return store
}

func largeSearchFixture(count int) ([]model.StoredChunk, map[string][]float32) {
	const firstFarAngle = 0.01

	chunks := make([]model.StoredChunk, 0, count)
	reuse := make(map[string][]float32, count)
	farCount := count - 1
	for index := 0; index < count; index++ {
		var content string
		var vector []float32
		if index == count-1 {
			content = "nearest"
			vector = []float32{1, 0}
		} else {
			content = fmt.Sprintf("far-%d", index)
			fraction := float64(index) / float64(max(farCount-1, 1))
			angle := firstFarAngle + (math.Pi-firstFarAngle)*fraction
			vector = vectorAtAngle(angle)
		}
		chunks = append(chunks, model.StoredChunk{
			Content:       content,
			RelativePath:  content + ".go",
			FileExtension: ".go",
		})
		reuse[semantic.ContentVectorKey(content)] = vector
	}
	return chunks, reuse
}

func largeFilteredSearchFixture(
	count int,
	filteredCount int,
) ([]model.StoredChunk, map[string][]float32) {
	const (
		keptAngle     = 0.01
		firstFarAngle = 0.02
	)

	chunks := make([]model.StoredChunk, 0, count)
	reuse := make(map[string][]float32, count)
	farCount := count - filteredCount - 1
	for index := 0; index < count; index++ {
		var content string
		var extension string
		var vector []float32
		if index < filteredCount {
			content = fmt.Sprintf("filtered-%d", index)
			extension = ".py"
			vector = []float32{1, 0}
		} else if index == filteredCount {
			content = "kept"
			extension = ".go"
			vector = vectorAtAngle(keptAngle)
		} else {
			content = fmt.Sprintf("far-%d", index)
			extension = ".txt"
			farIndex := index - filteredCount - 1
			fraction := float64(farIndex) / float64(max(farCount-1, 1))
			angle := firstFarAngle + (math.Pi-firstFarAngle)*fraction
			vector = vectorAtAngle(angle)
		}
		chunks = append(chunks, model.StoredChunk{
			Content:       content,
			RelativePath:  content + extension,
			FileExtension: extension,
		})
		reuse[semantic.ContentVectorKey(content)] = vector
	}
	return chunks, reuse
}

func vectorAtAngle(angle float64) []float32 {
	return []float32{float32(math.Cos(angle)), float32(math.Sin(angle))}
}

func stageAndPromoteWithReuse(
	t *testing.T,
	store *Store,
	codebasePath string,
	chunks []model.StoredChunk,
	reuse map[string][]float32,
) {
	t.Helper()
	if err := store.StageReindex(
		context.Background(),
		codebasePath,
		chunks,
		semantic.Removal{},
		nil,
		reuse,
		semantic.CodeColumns(),
	); err != nil {
		t.Fatalf("StageReindex returned error: %v", err)
	}
	if err := store.PromoteStaging(context.Background(), codebasePath); err != nil {
		t.Fatalf("PromoteStaging returned error: %v", err)
	}
}

func stageAndPromote(
	t *testing.T,
	store *Store,
	codebasePath string,
	chunks []model.StoredChunk,
	columnSet semantic.StoreColumnSet,
) {
	t.Helper()
	if err := store.StageReindex(
		context.Background(),
		codebasePath,
		chunks,
		semantic.Removal{},
		nil,
		nil,
		columnSet,
	); err != nil {
		t.Fatalf("StageReindex returned error: %v", err)
	}
	if err := store.PromoteStaging(context.Background(), codebasePath); err != nil {
		t.Fatalf("PromoteStaging returned error: %v", err)
	}
}

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
	stageAndPromote(t, store, codebasePath, chunks, semantic.StoreColumnSetCode)

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

func TestConversationSearchAppliesFiltersScoreAndPerConversationLimit(
	t *testing.T,
) {
	t.Parallel()

	const codebasePath = "chat:///local-search"
	provider := &fakeEmbeddingProvider{
		vectors: map[string][]float32{
			"a best":    {1, 0},
			"a second":  {0.9, 0.1},
			"b kept":    {0.8, 0.2},
			"b too low": {0.1, 0.9},
			"filtered":  {0.95, 0.05},
			"query":     {1, 0},
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
		conversationChunk("a best", "claude:a", "assistant", 1, 100),
		conversationChunk("a second", "claude:a", "assistant", 2, 101),
		conversationChunk("b kept", "claude:b", "assistant", 1, 102),
		conversationChunk("b too low", "claude:b", "assistant", 2, 103),
		conversationChunk("filtered", "cursor:c", "user", 1, 104),
	}
	stageAndPromote(
		t,
		store,
		codebasePath,
		chunks,
		semantic.StoreColumnSetConversation,
	)

	filter := semantic.ConversationFilter{
		Providers:        []string{"claude"},
		Roles:            []string{"ASSISTANT"},
		FromUnix:         100,
		UntilUnix:        104,
		MessageIndexFrom: 1,
	}
	results, err := store.SearchConversationCollectionCapped(
		context.Background(),
		store.CollectionName(codebasePath),
		"query",
		10,
		1,
		0.5,
		filter,
	)
	if err != nil {
		t.Fatalf("SearchConversationCollectionCapped returned error: %v", err)
	}
	gotContents := make([]string, 0, len(results))
	for _, result := range results {
		gotContents = append(gotContents, result.Content)
	}
	wantContents := []string{"a best", "b kept"}
	if !slices.Equal(gotContents, wantContents) {
		t.Fatalf("conversation contents = %v, want %v", gotContents, wantContents)
	}
}

func TestConversationPartIndexRejectsNegativeMessageIndex(t *testing.T) {
	t.Parallel()

	if _, err := conversationPartIndex("conv/test/-1", "test"); err == nil {
		t.Fatal("conversationPartIndex returned nil error for a negative message index")
	}
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
		semantic.StoreColumnSetCode,
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

func conversationChunk(
	content string,
	conversationID string,
	role string,
	messageIndex int32,
	timestampUnix int64,
) model.StoredChunk {
	return model.StoredChunk{
		Content:        content,
		RelativePath:   "conv/" + conversationID + "/message",
		ConversationID: conversationID,
		MessageIndex:   messageIndex,
		Role:           role,
		TimestampUnix:  timestampUnix,
		WorkspaceRoot:  "/workspace",
	}
}

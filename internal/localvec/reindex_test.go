package localvec

import (
	"context"
	"errors"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func TestReindexMissingCollectionSkipsEmbedding(t *testing.T) {
	t.Parallel()

	store, err := newStoreWithProvider(
		config.Config{StateRoot: t.TempDir()},
		&fakeEmbeddingProvider{vectors: map[string][]float32{}},
	)
	if err != nil {
		t.Fatalf("newStoreWithProvider returned error: %v", err)
	}
	err = store.Reindex(
		context.Background(),
		"/tmp/localvec-missing-reindex",
		[]model.StoredChunk{{Content: "must not embed", RelativePath: "missing.go"}},
		semantic.Removal{},
		nil,
		nil,
		semantic.StoreColumnSetCode,
	)
	if !errors.Is(err, semantic.ErrCollectionMissing) {
		t.Fatalf("Reindex error = %v, want ErrCollectionMissing", err)
	}
}

func TestReindexDeletesPathsAndPrefixesBeforeAppending(t *testing.T) {
	t.Parallel()

	const codebasePath = "/tmp/localvec-reindex-delete"
	provider := &fakeEmbeddingProvider{
		vectors: map[string][]float32{
			"exact old":  {1, 0},
			"prefix one": {0, 1},
			"prefix two": {0.5, 0.5},
			"untouched":  {0.25, 0.75},
			"replacement": {
				0.75,
				0.25,
			},
		},
	}
	store, err := newStoreWithProvider(
		config.Config{StateRoot: t.TempDir()},
		provider,
	)
	if err != nil {
		t.Fatalf("newStoreWithProvider returned error: %v", err)
	}
	initial := []model.StoredChunk{
		{Content: "exact old", RelativePath: "exact.go"},
		{Content: "prefix one", RelativePath: "conv/a/one"},
		{Content: "prefix two", RelativePath: "conv/a/two"},
		{Content: "untouched", RelativePath: "keep.go"},
	}
	stageAndPromote(t, store, codebasePath, initial, semantic.StoreColumnSetCode)

	removal := semantic.Removal{
		Paths:    []string{"exact.go"},
		Prefixes: []string{"conv/a/"},
	}
	replacement := []model.StoredChunk{
		{Content: "replacement", RelativePath: "exact.go"},
	}
	if err := store.Reindex(
		context.Background(),
		codebasePath,
		replacement,
		removal,
		nil,
		nil,
		semantic.StoreColumnSetCode,
	); err != nil {
		t.Fatalf("Reindex returned error: %v", err)
	}

	count, err := store.Count(context.Background(), codebasePath)
	if err != nil {
		t.Fatalf("Count returned error: %v", err)
	}
	if count != 2 {
		t.Fatalf("Count = %d, want 2", count)
	}
	exactReuse, err := store.LoadReuseVectorsForPath(
		context.Background(),
		store.CollectionName(codebasePath),
		"exact.go",
	)
	if err != nil {
		t.Fatalf("LoadReuseVectorsForPath returned error: %v", err)
	}
	if _, found := exactReuse[semantic.ContentVectorKey("exact old")]; found {
		t.Fatal("deleted exact-path row remained in the collection")
	}
	if _, found := exactReuse[semantic.ContentVectorKey("replacement")]; !found {
		t.Fatal("replacement row is missing from the collection")
	}
	prefixReuse, err := store.LoadReuseVectorsForPrefix(
		context.Background(),
		store.CollectionName(codebasePath),
		"conv/a/",
	)
	if err != nil {
		t.Fatalf("LoadReuseVectorsForPrefix returned error: %v", err)
	}
	if len(prefixReuse) != 0 {
		t.Fatalf("prefix reuse = %v, want empty", prefixReuse)
	}
}

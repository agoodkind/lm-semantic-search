package localvec

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

type fakeEmbeddingProvider struct {
	vectors map[string][]float32
}

func (provider *fakeEmbeddingProvider) Embed(
	_ context.Context,
	text string,
) ([]float32, error) {
	vector, found := provider.vectors[text]
	if !found {
		return nil, fmt.Errorf("no fake vector for %q", text)
	}
	return slices.Clone(vector), nil
}

func (provider *fakeEmbeddingProvider) EmbedBatch(
	ctx context.Context,
	texts []string,
) ([][]float32, error) {
	vectors := make([][]float32, 0, len(texts))
	for _, text := range texts {
		vector, err := provider.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		vectors = append(vectors, vector)
	}
	return vectors, nil
}

func (provider *fakeEmbeddingProvider) ProviderName() string {
	return "fake"
}

func (provider *fakeEmbeddingProvider) Health(context.Context) error {
	return nil
}

func TestStoreRoundTripSearchDeletePruneAndReuse(t *testing.T) {
	t.Parallel()

	const codebasePath = "/tmp/localvec-round-trip"
	provider := &fakeEmbeddingProvider{
		vectors: map[string][]float32{
			"first":  {1, 0},
			"second": {0.8, 0.6},
			"third":  {0, 1},
			"query":  {1, 0},
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
			Content:       "first",
			RelativePath:  "first.go",
			StartLine:     1,
			EndLine:       2,
			Language:      "go",
			FileExtension: ".go",
		},
		{
			Content:       "second",
			RelativePath:  "second.go",
			StartLine:     3,
			EndLine:       4,
			Language:      "go",
			FileExtension: ".go",
		},
		{
			Content:       "third",
			RelativePath:  "third.go",
			StartLine:     5,
			EndLine:       6,
			Language:      "go",
			FileExtension: ".go",
		},
	}
	if err := store.StageReindex(
		context.Background(),
		codebasePath,
		chunks,
		semantic.Removal{},
		nil,
		nil,
		semantic.StoreColumnSetCode,
	); err != nil {
		t.Fatalf("StageReindex returned error: %v", err)
	}
	if err := store.PromoteStaging(context.Background(), codebasePath); err != nil {
		t.Fatalf("PromoteStaging returned error: %v", err)
	}

	results, err := store.Search(
		context.Background(),
		codebasePath,
		"query",
		2,
		nil,
		"",
	)
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	gotContents := make([]string, 0, len(results))
	for _, result := range results {
		gotContents = append(gotContents, result.Content)
	}
	wantContents := []string{"first", "second"}
	if !slices.Equal(gotContents, wantContents) {
		t.Fatalf("Search contents = %v, want %v", gotContents, wantContents)
	}

	count, err := store.Count(context.Background(), codebasePath)
	if err != nil {
		t.Fatalf("Count returned error: %v", err)
	}
	if count != 3 {
		t.Fatalf("Count = %d, want 3", count)
	}

	reuse, err := store.LoadReuseVectorsForPath(
		context.Background(),
		store.CollectionName(codebasePath),
		"second.go",
	)
	if err != nil {
		t.Fatalf("LoadReuseVectorsForPath returned error: %v", err)
	}
	reusedVector, found := reuse[semantic.ContentVectorKey("second")]
	if !found {
		t.Fatal("reuse map is missing the second chunk content key")
	}
	if !slices.Equal(reusedVector, []float32{0.8, 0.6}) {
		t.Fatalf("reused vector = %v, want [0.8 0.6]", reusedVector)
	}

	if err := store.Reindex(
		context.Background(),
		codebasePath,
		nil,
		semantic.RemovePaths([]string{"second.go"}),
		nil,
		nil,
		semantic.StoreColumnSetCode,
	); err != nil {
		t.Fatalf("Reindex delete returned error: %v", err)
	}
	count, err = store.Count(context.Background(), codebasePath)
	if err != nil {
		t.Fatalf("Count after delete returned error: %v", err)
	}
	if count != 2 {
		t.Fatalf("Count after delete = %d, want 2", count)
	}

	if err := store.PruneToCurrent(
		context.Background(),
		codebasePath,
		[]string{"first.go"},
	); err != nil {
		t.Fatalf("PruneToCurrent returned error: %v", err)
	}
	count, err = store.Count(context.Background(), codebasePath)
	if err != nil {
		t.Fatalf("Count after prune returned error: %v", err)
	}
	if count != 1 {
		t.Fatalf("Count after prune = %d, want 1", count)
	}
}

func TestStoreRecoversInterruptedPromotion(t *testing.T) {
	t.Parallel()

	const codebasePath = "/tmp/localvec-promotion-recovery"
	testCases := []struct {
		name            string
		installStaging  bool
		expectedContent string
	}{
		{
			name:            "restores backup before staging rename",
			installStaging:  false,
			expectedContent: "old",
		},
		{
			name:            "removes backup after staging rename",
			installStaging:  true,
			expectedContent: "new",
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			stateRoot := t.TempDir()
			provider := &fakeEmbeddingProvider{
				vectors: map[string][]float32{
					"old":   {1, 0},
					"new":   {0, 1},
					"query": {1, 0},
				},
			}
			store, err := newStoreWithProvider(
				config.Config{StateRoot: stateRoot},
				provider,
			)
			if err != nil {
				t.Fatalf("newStoreWithProvider returned error: %v", err)
			}
			stageAndPromote(
				t,
				store,
				codebasePath,
				[]model.StoredChunk{{
					Content:       "old",
					RelativePath:  "old.go",
					FileExtension: ".go",
				}},
				semantic.StoreColumnSetCode,
			)
			if err := store.StageReindex(
				context.Background(),
				codebasePath,
				[]model.StoredChunk{{
					Content:       "new",
					RelativePath:  "new.go",
					FileExtension: ".go",
				}},
				semantic.Removal{},
				nil,
				nil,
				semantic.StoreColumnSetCode,
			); err != nil {
				t.Fatalf("StageReindex returned error: %v", err)
			}

			collectionName := store.CollectionName(codebasePath)
			livePath := filepath.Join(store.root, collectionName)
			stagingPath := livePath + stagingCollectionSuffix
			backupPath := livePath + backupCollectionSuffix
			if err := os.Rename(livePath, backupPath); err != nil {
				t.Fatalf("rename live collection to backup: %v", err)
			}
			if testCase.installStaging {
				if err := os.Rename(stagingPath, livePath); err != nil {
					t.Fatalf("rename staging collection to live: %v", err)
				}
			}

			reopened, err := newStoreWithProvider(
				config.Config{StateRoot: stateRoot},
				provider,
			)
			if err != nil {
				t.Fatalf("reopen local vector store: %v", err)
			}
			listed, err := reopened.ListCollections(context.Background())
			if err != nil {
				t.Fatalf("ListCollections returned error: %v", err)
			}
			if !slices.Equal(listed, []string{collectionName}) {
				t.Fatalf("ListCollections returned %v, want %q", listed, collectionName)
			}

			results, err := reopened.Search(
				context.Background(),
				codebasePath,
				"query",
				1,
				nil,
				"",
			)
			if err != nil {
				t.Fatalf("Search after recovery returned error: %v", err)
			}
			if len(results) != 1 || results[0].Content != testCase.expectedContent {
				t.Fatalf(
					"Search after recovery returned %+v, want %q",
					results,
					testCase.expectedContent,
				)
			}
			if _, err := os.Stat(livePath); err != nil {
				t.Fatalf("inspect recovered live collection: %v", err)
			}
			if _, err := os.Stat(backupPath); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("backup path still exists after recovery: %v", err)
			}
		})
	}
}

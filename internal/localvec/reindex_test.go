package localvec

import (
	"context"
	"errors"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/embedding"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// recordingEmbeddingProvider embeds any text with a deterministic non-zero
// vector and records every text it embedded, so a test can prove that each split
// child reached the embedder rather than being truncated to a prefix.
type recordingEmbeddingProvider struct {
	embedded []string
}

type rejectingEmbeddingProvider struct {
	rejected map[string]bool
	batches  [][]string
}

func (provider *recordingEmbeddingProvider) Embed(
	_ context.Context,
	text string,
) ([]float32, error) {
	provider.embedded = append(provider.embedded, text)
	return []float32{1, float32(len(text) + 1)}, nil
}

func (provider *recordingEmbeddingProvider) EmbedBatch(
	ctx context.Context,
	texts []string,
) (embedding.BatchResult, error) {
	vectors := make([][]float32, 0, len(texts))
	for _, text := range texts {
		vector, err := provider.Embed(ctx, text)
		if err != nil {
			return embedding.BatchResult{}, err
		}
		vectors = append(vectors, vector)
	}
	return embedding.BatchResult{Vectors: vectors}, nil
}

func (provider *recordingEmbeddingProvider) ProviderName() string {
	return "recording"
}

func (provider *recordingEmbeddingProvider) Health(context.Context) error {
	return nil
}

func (provider *rejectingEmbeddingProvider) Embed(
	_ context.Context,
	text string,
) ([]float32, error) {
	return []float32{1, float32(len(text) + 1)}, nil
}

func (provider *rejectingEmbeddingProvider) EmbedBatch(
	_ context.Context,
	texts []string,
) (embedding.BatchResult, error) {
	provider.batches = append(provider.batches, append([]string(nil), texts...))
	vectors := make([][]float32, len(texts))
	skipped := make([]embedding.SkippedInput, 0, len(texts))
	for index, text := range texts {
		if provider.rejected[text] {
			skipped = append(skipped, embedding.SkippedInput{
				Index:  index,
				Reason: "endpoint_policy_rejection",
			})
			continue
		}
		vectors[index] = []float32{1, float32(len(text) + 1)}
	}
	return embedding.BatchResult{Vectors: vectors, Skipped: skipped}, nil
}

func (provider *rejectingEmbeddingProvider) ProviderName() string {
	return "rejecting"
}

func (provider *rejectingEmbeddingProvider) Health(context.Context) error {
	return nil
}

func assertRowsCoverContentWithDistinctIDs(t *testing.T, store *Store, codebasePath string, content string, embedded []string) {
	t.Helper()
	if strings.Join(embedded, "") != content {
		t.Fatal("embedded pieces did not reproduce the content; a child was truncated or lost")
	}
	rows, exists, _, err := readRows(filepath.Join(store.root, store.CollectionName(codebasePath), metadataFileName))
	if err != nil {
		t.Fatalf("readRows returned error: %v", err)
	}
	if !exists {
		t.Fatal("collection file is missing after promote")
	}
	if len(rows) < 3 {
		t.Fatalf("stored %d rows, want at least 3 split pieces", len(rows))
	}
	ids := make(map[string]bool, len(rows))
	for index := range rows {
		if ids[rows[index].ID] {
			t.Fatalf("duplicate primary key %q among split rows", rows[index].ID)
		}
		ids[rows[index].ID] = true
	}
}

func TestStageReindexSplitsOversizedChunkIntoDistinctRows(t *testing.T) {
	t.Parallel()

	const codebasePath = "/tmp/localvec-oversized-stage"
	provider := &recordingEmbeddingProvider{}
	store, err := newStoreWithProvider(config.Config{StateRoot: t.TempDir()}, provider)
	if err != nil {
		t.Fatalf("newStoreWithProvider returned error: %v", err)
	}

	byteBudget := config.EmbedChunkByteBudget(0)
	// Repeated content makes the full-budget pieces byte-identical, so the store
	// must give each a distinct primary key or the rows would collide.
	content := strings.Repeat("x", byteBudget*2+5)
	stageAndPromote(t, store, codebasePath, []model.StoredChunk{{Content: content, RelativePath: "big.go"}}, semantic.StoreColumnSetCode)

	assertRowsCoverContentWithDistinctIDs(t, store, codebasePath, content, provider.embedded)
}

func TestReindexReportsDroppedChunks(t *testing.T) {
	t.Parallel()

	const codebasePath = "/tmp/localvec-reindex-dropped"
	provider := &rejectingEmbeddingProvider{
		rejected: map[string]bool{"drop": true},
	}
	store, err := newStoreWithProvider(config.Config{StateRoot: t.TempDir()}, provider)
	if err != nil {
		t.Fatalf("newStoreWithProvider returned error: %v", err)
	}
	stageAndPromote(
		t,
		store,
		codebasePath,
		[]model.StoredChunk{{Content: "seed", RelativePath: "seed.go"}},
		semantic.StoreColumnSetCode,
	)
	provider.batches = nil

	var reports []semantic.Progress
	err = store.Reindex(
		context.Background(),
		codebasePath,
		[]model.StoredChunk{
			{Content: "keep", RelativePath: "keep.go"},
			{Content: "drop", RelativePath: "drop.go"},
		},
		semantic.Removal{},
		func(progress semantic.Progress) {
			reports = append(reports, progress)
		},
		nil,
		semantic.StoreColumnSetCode,
		semantic.ReuseFromCorpus,
	)
	if err != nil {
		t.Fatalf("Reindex returned error: %v", err)
	}
	if len(provider.batches) != 1 || len(provider.batches[0]) != 2 {
		t.Fatalf("embed batches = %v, want one packed batch with two inputs", provider.batches)
	}
	if len(reports) != 1 {
		t.Fatalf("progress reports = %d, want 1", len(reports))
	}
	if reports[0].ChunksDropped != 1 {
		t.Fatalf("ChunksDropped = %d, want 1", reports[0].ChunksDropped)
	}
	if reports[0].CollectionRowsWritten != 1 {
		t.Fatalf(
			"CollectionRowsWritten = %d, want 1",
			reports[0].CollectionRowsWritten,
		)
	}
}

func TestStageReindexReportsProgressWhenEveryChunkDropped(t *testing.T) {
	t.Parallel()

	const codebasePath = "/tmp/localvec-stage-all-dropped"
	provider := &rejectingEmbeddingProvider{
		rejected: map[string]bool{
			"drop one": true,
			"drop two": true,
		},
	}
	store, err := newStoreWithProvider(config.Config{StateRoot: t.TempDir()}, provider)
	if err != nil {
		t.Fatalf("newStoreWithProvider returned error: %v", err)
	}

	var reports []semantic.Progress
	err = store.StageReindex(
		context.Background(),
		codebasePath,
		[]model.StoredChunk{
			{Content: "drop one", RelativePath: "one.go"},
			{Content: "drop two", RelativePath: "two.go"},
		},
		semantic.Removal{},
		func(progress semantic.Progress) {
			reports = append(reports, progress)
		},
		nil,
		semantic.StoreColumnSetCode,
		semantic.ReuseFromCorpus,
	)
	if err != nil {
		t.Fatalf("StageReindex returned error: %v", err)
	}
	if len(provider.batches) != 1 || len(provider.batches[0]) != 2 {
		t.Fatalf("embed batches = %v, want one packed batch with two inputs", provider.batches)
	}
	if len(reports) != 1 {
		t.Fatalf("progress reports = %d, want 1", len(reports))
	}
	if reports[0].ChunksDropped != 2 {
		t.Fatalf("ChunksDropped = %d, want 2", reports[0].ChunksDropped)
	}
	if reports[0].CollectionRowsWritten != 0 {
		t.Fatalf(
			"CollectionRowsWritten = %d, want 0",
			reports[0].CollectionRowsWritten,
		)
	}
}

func TestReindexPrefixDeleteRemovesEverySplitPiece(t *testing.T) {
	t.Parallel()

	const codebasePath = "/tmp/localvec-oversized-prefix-delete"
	provider := &recordingEmbeddingProvider{}
	store, err := newStoreWithProvider(config.Config{StateRoot: t.TempDir()}, provider)
	if err != nil {
		t.Fatalf("newStoreWithProvider returned error: %v", err)
	}

	byteBudget := config.EmbedChunkByteBudget(0)
	// One oversized message splits into many pieces that all share its relativePath
	// prefix, so a message delete-by-prefix must remove every piece.
	content := strings.Repeat("z", byteBudget*2+5)
	stageAndPromote(t, store, codebasePath, []model.StoredChunk{{Content: content, RelativePath: "conv/c/msg"}}, semantic.StoreColumnSetCode)

	before, err := store.Count(context.Background(), codebasePath)
	if err != nil {
		t.Fatalf("Count returned error: %v", err)
	}
	if before < 3 {
		t.Fatalf("stored %d pieces before delete, want at least 3", before)
	}

	if err := store.Reindex(
		context.Background(),
		codebasePath,
		nil,
		semantic.Removal{Prefixes: []string{"conv/c/"}},
		nil,
		nil,
		semantic.StoreColumnSetCode,
		semantic.ReuseFromCorpus,
	); err != nil {
		t.Fatalf("Reindex returned error: %v", err)
	}

	after, err := store.Count(context.Background(), codebasePath)
	if err != nil {
		t.Fatalf("Count returned error: %v", err)
	}
	if after != 0 {
		t.Fatalf("Count after prefix delete = %d, want 0 (every split piece removed)", after)
	}
}

func TestReindexSplitsOversizedChunkIntoDistinctRows(t *testing.T) {
	t.Parallel()

	const codebasePath = "/tmp/localvec-oversized-reindex"
	provider := &recordingEmbeddingProvider{}
	store, err := newStoreWithProvider(config.Config{StateRoot: t.TempDir()}, provider)
	if err != nil {
		t.Fatalf("newStoreWithProvider returned error: %v", err)
	}

	// A first small index creates the live collection Reindex requires.
	stageAndPromote(t, store, codebasePath, []model.StoredChunk{{Content: "seed", RelativePath: "seed.go"}}, semantic.StoreColumnSetCode)
	provider.embedded = nil

	byteBudget := config.EmbedChunkByteBudget(0)
	content := strings.Repeat("y", byteBudget*2+5)
	if err := store.Reindex(
		context.Background(),
		codebasePath,
		[]model.StoredChunk{{Content: content, RelativePath: "big.go"}},
		semantic.Removal{},
		nil,
		nil,
		semantic.StoreColumnSetCode,
		semantic.ReuseFromCorpus,
	); err != nil {
		t.Fatalf("Reindex returned error: %v", err)
	}

	if strings.Join(provider.embedded, "") != content {
		t.Fatal("Reindex embedded pieces did not reproduce the content; a child was truncated or lost")
	}
	rows, exists, _, err := readRows(filepath.Join(store.root, store.CollectionName(codebasePath), metadataFileName))
	if err != nil {
		t.Fatalf("readRows returned error: %v", err)
	}
	if !exists {
		t.Fatal("collection file is missing after reindex")
	}
	bigRowIDs := make(map[string]bool)
	bigPieces := 0
	for index := range rows {
		if rows[index].RelativePath != "big.go" {
			continue
		}
		bigPieces++
		if bigRowIDs[rows[index].ID] {
			t.Fatalf("duplicate primary key %q among reindexed split rows", rows[index].ID)
		}
		bigRowIDs[rows[index].ID] = true
	}
	if bigPieces < 3 {
		t.Fatalf("reindexed %d big.go pieces, want at least 3", bigPieces)
	}
}

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
		semantic.ReuseFromCorpus,
	)
	if !errors.Is(err, semantic.ErrCollectionMissing) {
		t.Fatalf("Reindex error = %v, want ErrCollectionMissing", err)
	}
}

func TestSplitRowsRoundTripAndCopyWithDistinctIdentities(t *testing.T) {
	t.Parallel()

	const (
		codebasePath    = "/tmp/localvec-split-copy"
		sourcePath      = "src/big.go"
		destinationPath = "src/renamed.go"
	)
	store, err := newStoreWithProvider(
		config.Config{StateRoot: t.TempDir()},
		&fakeEmbeddingProvider{vectors: map[string][]float32{
			"same": {1, 0},
		}},
	)
	if err != nil {
		t.Fatalf("newStoreWithProvider returned error: %v", err)
	}
	chunks := []model.StoredChunk{
		{
			Content:      "same",
			RelativePath: sourcePath,
			StartLine:    1,
			EndLine:      400,
			SplitPart:    1,
		},
		{
			Content:      "same",
			RelativePath: sourcePath,
			StartLine:    1,
			EndLine:      400,
			SplitPart:    513,
		},
	}
	stageAndPromote(t, store, codebasePath, chunks, semantic.StoreColumnSetCode)

	copied, err := store.CopyChunks(
		context.Background(),
		codebasePath,
		sourcePath,
		destinationPath,
	)
	if err != nil {
		t.Fatalf("CopyChunks returned error: %v", err)
	}
	if copied != 2 {
		t.Fatalf("CopyChunks copied %d rows, want 2", copied)
	}
	stored, err := store.collectionForName(store.CollectionName(codebasePath), false)
	if err != nil {
		t.Fatalf("collectionForName returned error: %v", err)
	}
	rows, exists, err := stored.snapshot()
	if err != nil {
		t.Fatalf("snapshot returned error: %v", err)
	}
	if !exists {
		t.Fatal("snapshot reported collection missing")
	}
	if len(rows) != 2 {
		t.Fatalf("stored rows = %d, want 2", len(rows))
	}
	if rows[0].ID == rows[1].ID {
		t.Fatalf("copied rows share destination ID %q", rows[0].ID)
	}
	parts := []int32{rows[0].chunk(0).SplitPart, rows[1].chunk(0).SplitPart}
	slices.Sort(parts)
	if !slices.Equal(parts, []int32{1, 513}) {
		t.Fatalf("round-tripped split parts = %v, want [1 513]", parts)
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
		semantic.ReuseFromCorpus,
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

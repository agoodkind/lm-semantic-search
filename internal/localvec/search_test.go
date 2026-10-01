package localvec

import (
	"context"
	"fmt"
	"math"
	"slices"
	"strings"
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

// TestCollectionSearchEvaluatesNestedFilterTree proves the local store keeps a
// row only when the whole tree is true. An any node keeps either branch. A not
// node rejects its child's matches. A leaf on a column the row format lacks is
// unknown, which a not node keeps unknown. Every hit decodes the declared
// scalar cells from the row.
func TestCollectionSearchEvaluatesNestedFilterTree(t *testing.T) {
	t.Parallel()

	const codebasePath = "chat:///local-tree"
	provider := &fakeEmbeddingProvider{
		vectors: map[string][]float32{
			"first":  {1, 0},
			"second": {0.9, 0.1},
			"third":  {0.8, 0.2},
			"query":  {1, 0},
		},
	}
	store, err := newStoreWithProvider(config.Config{StateRoot: t.TempDir()}, provider)
	if err != nil {
		t.Fatalf("newStoreWithProvider returned error: %v", err)
	}
	stageAndPromote(t, store, codebasePath, []model.StoredChunk{
		documentChunk("first", "claude:a", "user", 0, 100),
		documentChunk("second", "codex:b", "assistant", 1, 200),
		documentChunk("third", "claude:c", "assistant", 2, 300),
	}, semantic.ColumnsForDeclaration(documentDeclaration()))

	declaration := documentDeclaration()
	declaration.Scalars = append(declaration.Scalars, model.ScalarColumn{Name: "priority", Type: model.ScalarTypeInt64, Nullable: true, MaxLength: 0})
	lower := int64(250)
	search := func(filter semantic.CollectionFilter) []semantic.CollectionHit {
		t.Helper()
		hits, searchErr := store.SearchCollection(context.Background(), semantic.CollectionSearch{
			CollectionName: store.CollectionName(codebasePath),
			Query:          "query",
			Limit:          10,
			MinScore:       0,
			Filter:         &filter,
			GroupBy:        "",
			PerGroupLimit:  0,
			Declaration:    declaration,
		})
		if searchErr != nil {
			t.Fatalf("SearchCollection returned error: %v", searchErr)
		}
		return hits
	}

	either := semantic.AnyOf(
		semantic.ColumnEquals("role", semantic.StringScalar("user")),
		semantic.ColumnRange("timestampUnix", &lower, nil),
	)
	if got, want := hitContents(search(either)), []string{"first", "third"}; !slices.Equal(got, want) {
		t.Fatalf("any node contents = %v, want %v", got, want)
	}
	notClaude := semantic.Negate(semantic.ColumnIn("provider", semantic.StringValues([]string{"claude"})))
	if got, want := hitContents(search(notClaude)), []string{"second"}; !slices.Equal(got, want) {
		t.Fatalf("not node contents = %v, want %v", got, want)
	}
	unknownPriority := semantic.Negate(semantic.ColumnEquals("priority", semantic.Int64Scalar(1)))
	if got := hitContents(search(unknownPriority)); len(got) != 0 {
		t.Fatalf("not over an absent column kept %v, want none", got)
	}
	if got, want := hitContents(search(semantic.ColumnIsNull("priority"))), []string{"first", "second", "third"}; !slices.Equal(got, want) {
		t.Fatalf("is_null on an absent column kept %v, want every row", got)
	}

	hits := search(semantic.ColumnEquals("itemId", semantic.StringScalar("claude:a")))
	if len(hits) != 1 {
		t.Fatalf("equality kept %d hits, want 1", len(hits))
	}
	role, _ := hits[0].Scalar("role")
	if role != semantic.ValueCell("role", semantic.StringScalar("user")) {
		t.Fatalf("role cell = %+v, want the lowercased stored role", role)
	}
	providerCell, _ := hits[0].Scalar("provider")
	if providerCell != semantic.ValueCell("provider", semantic.StringScalar("claude")) {
		t.Fatalf("provider cell = %+v, want claude", providerCell)
	}
	priority, _ := hits[0].Scalar("priority")
	if priority != semantic.AbsentCell("priority") {
		t.Fatalf("priority cell = %+v, want absent", priority)
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
	tiedDenseDocument = "claude:dense"
	tiedFarDocument   = "claude:far"
	tiedDenseRows     = 300
	tiedFarRows       = 3
)

// tiedDocumentFixture builds total document rows. The dense
// document has tiedDenseRows rows with the query vector, so they tie at
// the top score. The far document has tiedFarRows rows with the opposite
// vector, so they score lowest. Every other row is its own document at a
// distinct angle between them.
func tiedDocumentFixture(total int) ([]model.StoredChunk, map[string][]float32) {
	const firstOtherAngle = 0.01
	chunks := make([]model.StoredChunk, 0, total)
	reuse := make(map[string][]float32, total)
	add := func(content string, documentID string, messageIndex int, vector []float32) {
		chunks = append(chunks, model.StoredChunk{
			Content:      content,
			RelativePath: fmt.Sprintf("conv/%s/%d", documentID, messageIndex),
			Scalars:      map[string]model.ScalarValue{"itemId": {Type: model.ScalarTypeString, String: documentID}},
		})
		reuse[semantic.ContentVectorKey(content)] = vector
	}
	for messageIndex := range tiedDenseRows {
		add(fmt.Sprintf("dense-%04d", messageIndex), tiedDenseDocument, messageIndex, []float32{1, 0})
	}
	for messageIndex := range tiedFarRows {
		add(fmt.Sprintf("far-%d", messageIndex), tiedFarDocument, messageIndex, []float32{-1, 0})
	}
	otherCount := total - tiedDenseRows - tiedFarRows
	for index := range otherCount {
		fraction := float64(index) / float64(max(otherCount-1, 1))
		angle := firstOtherAngle + (math.Pi-2*firstOtherAngle)*fraction
		add(fmt.Sprintf("other-%05d", index), fmt.Sprintf("claude:other-%05d", index), 0, vectorAtAngle(angle))
	}
	return chunks, reuse
}

// TestCollectionSearchSmallerLimitIsPrefixAboveExactThreshold proves collection
// search ranks a fixed candidate set above the 4,096-row exact threshold of
// code search, both below and above semantic.CollectionRankingDepth rows. The
// dense document's rows tie at the top score, and a cap of two per
// document keeps two of them. Every smaller limit returns a prefix of limit
// 20, repeated searches return the same rows, and a search scoped to the far
// document finds its rows although they score lowest.
func TestCollectionSearchSmallerLimitIsPrefixAboveExactThreshold(t *testing.T) {
	t.Parallel()

	for _, total := range []int{exactSearchThreshold + 104, semantic.CollectionRankingDepth + 616} {
		t.Run(fmt.Sprintf("%d rows", total), func(t *testing.T) {
			t.Parallel()

			codebasePath := fmt.Sprintf("chat:///local-prefix-%d", total)
			store := newSearchTestStore(t)
			chunks, reuse := tiedDocumentFixture(total)
			stageAndPromoteWithReuse(t, store, codebasePath, chunks, reuse)
			search := func(limit int32, filter *semantic.CollectionFilter) []string {
				t.Helper()
				hits, err := store.SearchCollection(context.Background(), semantic.CollectionSearch{
					CollectionName: store.CollectionName(codebasePath),
					Query:          "query",
					Limit:          limit,
					MinScore:       0,
					Filter:         filter,
					GroupBy:        "itemId",
					PerGroupLimit:  2,
					Declaration:    documentDeclaration(),
				})
				if err != nil {
					t.Fatalf("SearchCollection returned error: %v", err)
				}
				paths := make([]string, 0, len(hits))
				for _, hit := range hits {
					paths = append(paths, hit.Chunk.RelativePath)
				}
				return paths
			}

			larger := search(20, nil)
			if len(larger) != 20 {
				t.Fatalf("limit 20 returned %d rows, want 20", len(larger))
			}
			for _, path := range larger[:2] {
				if !strings.HasPrefix(path, "conv/"+tiedDenseDocument+"/") {
					t.Fatalf("top rows %v, want two dense rows first", larger[:2])
				}
			}
			for _, limit := range []int32{1, 2, 3, 5, 10} {
				if smaller := search(limit, nil); !slices.Equal(smaller, larger[:len(smaller)]) || len(smaller) != int(limit) {
					t.Fatalf("limit %d rows %v are not the first %d rows of %v", limit, smaller, limit, larger)
				}
			}
			for range 3 {
				if again := search(20, nil); !slices.Equal(again, larger) {
					t.Fatalf("repeated search rows %v, want %v", again, larger)
				}
			}

			scope := semantic.ColumnIn("itemId", semantic.StringValues([]string{tiedFarDocument}))
			far := search(10, &scope)
			if len(far) != 2 || !strings.HasPrefix(far[0], "conv/"+tiedFarDocument+"/") {
				t.Fatalf("scoped search rows %v, want the two far rows the cap keeps", far)
			}
		})
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

func documentDeclaration() model.CollectionDeclaration {
	return model.CollectionDeclaration{ItemIDColumn: "itemId", Scalars: []model.ScalarColumn{
		{Name: "itemId", Type: model.ScalarTypeString, MaxLength: 256},
		{Name: "role", Type: model.ScalarTypeString, MaxLength: 64},
		{Name: "provider", Type: model.ScalarTypeString, MaxLength: 64},
		{Name: "timestampUnix", Type: model.ScalarTypeInt64},
	}}
}

func documentChunk(content string, documentID string, role string, sequence int32, created int64) model.StoredChunk {
	category, _, _ := strings.Cut(documentID, ":")
	return model.StoredChunk{
		Content: content, RelativePath: "items/" + documentID + "/row", StartLine: sequence,
		Scalars: map[string]model.ScalarValue{
			"itemId":        {Type: model.ScalarTypeString, String: documentID},
			"role":          {Type: model.ScalarTypeString, String: role},
			"provider":      {Type: model.ScalarTypeString, String: category},
			"timestampUnix": {Type: model.ScalarTypeInt64, Int64: created},
		},
	}
}

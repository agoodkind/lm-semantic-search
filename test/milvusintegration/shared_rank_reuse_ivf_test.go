//go:build milvusintegration

package milvusintegration

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/index"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"
)

const (
	ivfReuseCollection  = "rank_reuse_ivf"
	ivfReuseClusters    = 16
	ivfReuseLeaders     = 75
	ivfReuseMiddle      = 228
	ivfReuseTieSize     = 12
	ivfReuseFillers     = 1500
	ivfReuseTieNudge    = 1e-7
	ivfReuseShared      = 5
	ivfReuseInsideTime  = int64(1_790_000_000)
	ivfReuseOutsideTime = int64(1_700_000_000)
	ivfReuseRangeLower  = int64(1_789_000_000)
	ivfReuseRangeUpper  = int64(1_791_000_000)
	ivfReuseQueries     = 4
	ivfReuseBatch       = 500
)

var ivfReuseWords = []string{"milvus", "segment", "compaction", "index", "vector", "queue", "retention", "backup", "search", "filter"}

// alignedVector returns a unit vector with the given cosine similarity to query.
func alignedVector(generator *rand.Rand, query []float32, cosine float64) []float32 {
	noise := normalizedVector(generator)
	dot := 0.0
	for position := range noise {
		dot += float64(noise[position]) * float64(query[position])
	}
	orthogonal := make([]float64, len(noise))
	norm := 0.0
	for position := range noise {
		orthogonal[position] = float64(noise[position]) - dot*float64(query[position])
		norm += orthogonal[position] * orthogonal[position]
	}
	norm = math.Sqrt(norm)
	side := math.Sqrt(1 - cosine*cosine)
	vector := make([]float32, len(noise))
	for position := range vector {
		vector[position] = float32(cosine*float64(query[position]) + side*orthogonal[position]/norm)
	}
	return vector
}

// tieGroup returns size copies of base, each with one coordinate changed by ivfReuseTieNudge. The copies have distinct bytes and vector keys, but IVF_SQ8 encodes them identically.
func tieGroup(base []float32, size int) [][]float32 {
	group := make([][]float32, 0, size)
	for member := range size {
		variant := slices.Clone(base)
		variant[member%len(variant)] += float32(ivfReuseTieNudge) * float32(member/len(variant)+1)
		group = append(group, variant)
	}
	return group
}

func ivfReuseRow(id string, rowIndex int, generator *rand.Rand, vector []float32) collection.Row {
	timestamp := ivfReuseOutsideTime
	if rowIndex%2 == 0 {
		timestamp = ivfReuseInsideTime
	}
	content := ivfReuseWords[generator.IntN(len(ivfReuseWords))] + " " + ivfReuseWords[generator.IntN(len(ivfReuseWords))]
	return collection.Row{
		ID:            id,
		Content:       content,
		RelativePath:  "conv/ivf/" + id,
		FileExtension: ".md",
		Metadata:      `{"kind":"message"}`,
		Vector:        vector,
		Scalars: map[string]collection.ScalarValue{
			"conversationId": collection.StringScalar(fmt.Sprintf("conv-ivf-%02d", rowIndex%7)),
			"provider":       collection.StringScalar("claude"),
			"workspaceRoot":  collection.StringScalar("/ws/0"),
			"archived":       collection.BoolScalar(false),
			"timestampUnix":  collection.Int64Scalar(timestamp),
			"label":          {Type: collection.ScalarTypeString, Null: true},
		},
	}
}

// ivfReuseRows builds vectors around query with tie groups at dense positions 76 to 87 and 316 to 327. The vector at position 78 stores five occurrences.
func ivfReuseRows(generator *rand.Rand, query []float32) []collection.Row {
	vectors := make([][]float32, 0, ivfReuseLeaders+ivfReuseMiddle+2*ivfReuseTieSize+ivfReuseFillers)
	for leader := range ivfReuseLeaders {
		vectors = append(vectors, alignedVector(generator, query, 0.95-0.5*float64(leader)/float64(ivfReuseLeaders)))
	}
	vectors = append(vectors, tieGroup(alignedVector(generator, query, 0.42), ivfReuseTieSize)...)
	for middle := range ivfReuseMiddle {
		vectors = append(vectors, alignedVector(generator, query, 0.40-0.25*float64(middle)/float64(ivfReuseMiddle)))
	}
	vectors = append(vectors, tieGroup(alignedVector(generator, query, 0.12), ivfReuseTieSize)...)
	for range ivfReuseFillers {
		vectors = append(vectors, alignedVector(generator, query, -0.9*generator.Float64()))
	}
	rows := make([]collection.Row, 0, len(vectors)+ivfReuseShared)
	for vectorIndex, vector := range vectors {
		rows = append(rows, ivfReuseRow(fmt.Sprintf("ivf-%05d", vectorIndex), vectorIndex, generator, vector))
		if vectorIndex == 77 {
			for copyIndex := 1; copyIndex < ivfReuseShared; copyIndex++ {
				rows = append(rows, ivfReuseRow(fmt.Sprintf("ivf-%05d-%d", vectorIndex, copyIndex), vectorIndex+copyIndex, generator, slices.Clone(vector)))
			}
		}
	}
	return rows
}

// vectorScores returns dense scores from a direct vector collection search in descending order.
func vectorScores(t *testing.T, client *milvusclient.Client, query []float32, limit int) []float32 {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), milvusCallTimeout)
	defer cancel()
	annParam := index.NewCustomAnnParam()
	annParam.WithExtraParam("nprobe", ivfReuseClusters)
	option := milvusclient.NewSearchOption(milvusstore.VectorCollectionName(ivfReuseCollection), limit, []entity.Vector{entity.FloatVector(query)}).
		WithANNSField(milvusstore.DenseVectorField).
		WithAnnParam(annParam)
	resultSets, err := client.Search(ctx, option)
	if err != nil {
		t.Fatalf("direct vector search: %v", err)
	}
	scores := slices.Clone(resultSets[0].Scores)
	slices.SortFunc(scores, func(left float32, right float32) int {
		switch {
		case left > right:
			return -1
		case left < right:
			return 1
		default:
			return 0
		}
	})
	return scores
}

// TestSharedStoreRankReuseMatchesUnscopedOnIVFSQ8Ties checks score ties and reuse, then compares scoped and unscoped Rank results across limits, filters, and queries using a Milvus fixture with an IVF_SQ8 index.
func TestSharedStoreRankReuseMatchesUnscopedOnIVFSQ8Ties(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, "4g")
	client := dialMilvus(t, stack)
	ctx := context.Background()
	options := milvusstore.Options{
		Hybrid:            true,
		EmbeddingModel:    "generated-model",
		DenseIndex:        index.NewIvfSQ8Index(entity.COSINE, ivfReuseClusters),
		DenseSearchParams: map[string]string{"nprobe": fmt.Sprint(ivfReuseClusters)},
	}
	newStore := func() *milvusstore.SharedStore {
		store, err := milvusstore.NewShared(client, options)
		if err != nil {
			t.Fatalf("new shared store: %v", err)
		}
		return store
	}
	writer := newStore()
	if err := writer.EnsureCollection(ctx, collection.EnsureRequest{Collection: ivfReuseCollection, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
		t.Fatalf("ensure shared collection: %v", err)
	}
	vectorsName := milvusstore.VectorCollectionName(ivfReuseCollection)
	loadSharedCollections(t, client, ivfReuseCollection, vectorsName)
	generator := rand.New(rand.NewPCG(sharedSeed+11, sharedSeed+11))
	query := normalizedVector(generator)
	rows := ivfReuseRows(generator, query)
	for start := 0; start < len(rows); start += ivfReuseBatch {
		if err := writer.Upsert(ctx, ivfReuseCollection, sharedDeclaration, rows[start:min(start+ivfReuseBatch, len(rows))]); err != nil {
			t.Fatalf("upsert at %d: %v", start, err)
		}
	}
	sealAndIndex(t, client, ivfReuseCollection, false)
	sealAndIndex(t, client, vectorsName, true)

	scores := vectorScores(t, client, query, 4*80+1)
	if len(scores) < 4*80+1 || scores[79] != scores[80] || scores[319] != scores[320] {
		t.Fatalf("fixture lacks ties across the dense cutoffs: scores at 80/81 %v, at 320/321 %v", scores[79:81], scores[319:321])
	}

	lower := ivfReuseRangeLower
	upper := ivfReuseRangeUpper
	dateFilter := collection.ColumnRange("timestampUnix", &lower, &upper)
	conversationFilter := collection.ColumnIn("conversationId", collection.StringValues([]string{"conv-ivf-01", "conv-ivf-04"}))
	filters := []struct {
		name   string
		filter *collection.Filter
	}{{"none", nil}, {"date", &dateFilter}, {"conversation", &conversationFilter}}
	queries := [][]float32{query}
	for range ivfReuseQueries - 1 {
		queries = append(queries, normalizedVector(generator))
	}
	limits := []int32{20, 80, 320}
	compared := 0
	for _, filterCase := range filters {
		for queryIndex, vector := range queries {
			request := collection.SearchRequest{Collection: ivfReuseCollection, Query: "milvus segment", Vector: vector, Filter: filterCase.filter, Declaration: sharedDeclaration}
			unscoped := newStore()
			again := newStore()
			scopedStore := newStore()
			scope := rankReuseScope(t, ctx)
			for _, limit := range limits {
				request.Limit = limit
				want, err := unscoped.Rank(ctx, request)
				if err != nil {
					t.Fatalf("%s query %d limit %d: unscoped rank: %v", filterCase.name, queryIndex, limit, err)
				}
				repeat, err := again.Rank(ctx, request)
				if err != nil {
					t.Fatalf("%s query %d limit %d: repeated unscoped rank: %v", filterCase.name, queryIndex, limit, err)
				}
				if !slices.Equal(want, repeat) {
					t.Fatalf("%s query %d limit %d: two unscoped Rank calls on fixed data differ", filterCase.name, queryIndex, limit)
				}
				got, err := scopedStore.Rank(scope, request)
				if err != nil {
					t.Fatalf("%s query %d limit %d: scoped rank: %v", filterCase.name, queryIndex, limit, err)
				}
				if !slices.Equal(want, got) {
					t.Fatalf("%s query %d limit %d: scoped Rank returned %d candidates that differ from the %d unscoped candidates", filterCase.name, queryIndex, limit, len(got), len(want))
				}
				compared++
			}
		}
	}
	requireScopedRankReusesSearch(t, ctx, newStore, sharedDeclaration, queries[1])
	t.Logf("scoped Rank equaled unscoped Rank in all %d comparisons on IVF_SQ8 with ties across the 80 and 320 key cutoffs", compared)
}

// requireScopedRankReusesSearch runs scoped Rank, inserts the query vector through another store, and checks that the same scope omits the new row while an unscoped Rank and a new scope return it.
func requireScopedRankReusesSearch(t *testing.T, ctx context.Context, newStore func() *milvusstore.SharedStore, declaration collection.Declaration, vector []float32) {
	t.Helper()
	reader := newStore()
	request := collection.SearchRequest{Collection: ivfReuseCollection, Query: "milvus segment", Vector: vector, Declaration: declaration, Limit: 20}
	scope := rankReuseScope(t, ctx)
	if _, err := reader.Rank(scope, request); err != nil {
		t.Fatalf("first scoped rank: %v", err)
	}
	newRow := ivfReuseRow("ivf-new", 1, rand.New(rand.NewPCG(1, 1)), slices.Clone(vector))
	newRow.Content = "unrelated words"
	if err := newStore().Upsert(ctx, ivfReuseCollection, declaration, []collection.Row{newRow}); err != nil {
		t.Fatalf("upsert through a second store: %v", err)
	}
	time.Sleep(sharedVisibilityWait)
	request.Limit = 80
	reused, err := reader.Rank(scope, request)
	if err != nil {
		t.Fatalf("second scoped rank: %v", err)
	}
	if containsCandidate(reused, newRow.ID) {
		t.Fatalf("the second Rank in one scope searched Milvus again and returned row %s written after the first Rank", newRow.ID)
	}
	unscoped, err := reader.Rank(ctx, request)
	if err != nil {
		t.Fatalf("unscoped rank: %v", err)
	}
	fresh, err := reader.Rank(rankReuseScope(t, ctx), request)
	if err != nil {
		t.Fatalf("rank in a new scope: %v", err)
	}
	if !containsCandidate(unscoped, newRow.ID) || !containsCandidate(fresh, newRow.ID) {
		t.Fatalf("Rank without a scope or in a new scope did not return row %s that a second store wrote", newRow.ID)
	}
}

// TestSharedStoreRankReuseMatchesUnscopedOnHNSW checks that scoped Rank succeeds and matches unscoped Rank using a fixture with an HNSW index.
func TestSharedStoreRankReuseMatchesUnscopedOnHNSW(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, "4g")
	client := dialMilvus(t, stack)
	ctx := context.Background()
	options := milvusstore.Options{
		Hybrid:            true,
		EmbeddingModel:    "generated-model",
		DenseIndex:        index.NewHNSWIndex(entity.COSINE, 16, 64),
		DenseSearchParams: map[string]string{"ef": "64"},
	}
	store, err := milvusstore.NewShared(client, options)
	if err != nil {
		t.Fatalf("new shared store: %v", err)
	}
	if err := store.EnsureCollection(ctx, collection.EnsureRequest{Collection: ivfReuseCollection, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
		t.Fatalf("ensure shared collection: %v", err)
	}
	vectorsName := milvusstore.VectorCollectionName(ivfReuseCollection)
	loadSharedCollections(t, client, ivfReuseCollection, vectorsName)
	generator := rand.New(rand.NewPCG(sharedSeed+13, sharedSeed+13))
	query := normalizedVector(generator)
	rows := ivfReuseRows(generator, query)
	for start := 0; start < len(rows); start += ivfReuseBatch {
		if err := store.Upsert(ctx, ivfReuseCollection, sharedDeclaration, rows[start:min(start+ivfReuseBatch, len(rows))]); err != nil {
			t.Fatalf("upsert at %d: %v", start, err)
		}
	}
	sealAndIndex(t, client, ivfReuseCollection, false)
	sealAndIndex(t, client, vectorsName, true)
	request := collection.SearchRequest{Collection: ivfReuseCollection, Query: "milvus segment", Vector: query, Declaration: sharedDeclaration, Limit: 10}
	want, err := store.Rank(ctx, request)
	if err != nil {
		t.Fatalf("unscoped rank: %v", err)
	}
	got, err := store.Rank(rankReuseScope(t, ctx), request)
	if err != nil {
		t.Fatalf("scoped rank with ef 64: %v", err)
	}
	if !slices.Equal(want, got) {
		t.Fatalf("scoped Rank on HNSW returned %d candidates that differ from the %d unscoped candidates", len(got), len(want))
	}
}

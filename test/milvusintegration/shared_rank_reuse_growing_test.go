//go:build milvusintegration

package milvusintegration

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"net"
	"net/http"
	"os/exec"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/index"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"
)

const (
	growingCollection   = "growing_reuse"
	growingRows         = 20000
	growingDimension    = 4096
	growingInsideEvery  = 40
	growingQueries      = 20
	growingBatch        = 200
	growingClusters     = 128
	growingWait         = 10 * time.Second
	growingInterimKey   = "querynode.segcore.interimindex.enableindex"
	growingInterimIndex = "IVF_FLAT_CC"
)

// interimIndexEnabled reads the throwaway Milvus configuration endpoint and reports whether Milvus enables the interim index for growing segments.
func interimIndexEnabled(t *testing.T, stack throwawayStack) bool {
	t.Helper()
	response, err := http.Get("http://" + net.JoinHostPort("127.0.0.1", strconv.Itoa(stack.healthPort)) + "/api/v1/_cluster/configs")
	if err != nil {
		t.Fatalf("read Milvus config: %v", err)
	}
	defer func() { _ = response.Body.Close() }()
	values := map[string]string{}
	if err := json.NewDecoder(response.Body).Decode(&values); err != nil {
		t.Fatalf("decode Milvus config: %v", err)
	}
	return strings.HasPrefix(values[growingInterimKey], "true")
}

// wideVector returns a random unit vector with growingDimension values.
func wideVector(generator *rand.Rand) []float32 {
	vector := make([]float32, growingDimension)
	norm := 0.0
	for position := range vector {
		value := generator.NormFloat64()
		vector[position] = float32(value)
		norm += value * value
	}
	scale := float32(1 / math.Sqrt(norm))
	for position := range vector {
		vector[position] *= scale
	}
	return vector
}

// rankTwice runs Rank with limits 20 and 80 in sequence using one context and returns both candidate lists.
func rankTwice(t *testing.T, ctx context.Context, store *milvusstore.SharedStore, request collection.SearchRequest) [][]milvusstore.Candidate {
	t.Helper()
	results := make([][]milvusstore.Candidate, 0, 2)
	for _, limit := range []int32{reuseFirstLimit, reuseSecondLimit} {
		request.Limit = limit
		candidates, err := store.Rank(ctx, request)
		if err != nil {
			t.Fatalf("rank at limit %d: %v", limit, err)
		}
		results = append(results, candidates)
	}
	return results
}

// TestSharedStoreRankReuseMatchesUnscopedOnGrowingSegments writes unflushed rows to an IVF_SQ8 pair with the interim index enabled. It requires a direct top 80 search to differ from a top 320 prefix for a query and checks scoped Rank against unscoped Rank with a date filter.
func TestSharedStoreRankReuseMatchesUnscopedOnGrowingSegments(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, "6g")
	client := dialMilvus(t, stack)
	ctx := context.Background()
	if !interimIndexEnabled(t, stack) {
		t.Fatalf("the throwaway Milvus reports %s other than true; growing segments need the interim index for this test", growingInterimKey)
	}
	options := milvusstore.Options{
		Hybrid:            true,
		EmbeddingModel:    "generated-model",
		DenseIndex:        index.NewIvfSQ8Index(entity.COSINE, growingClusters),
		DenseSearchParams: map[string]string{"nprobe": strconv.Itoa(growingClusters)},
	}
	store, err := milvusstore.NewShared(client, options)
	if err != nil {
		t.Fatalf("new shared store: %v", err)
	}
	if err := store.EnsureCollection(ctx, collection.EnsureRequest{Collection: growingCollection, Declaration: sharedDeclaration, Dimension: growingDimension}); err != nil {
		t.Fatalf("ensure shared collection: %v", err)
	}
	vectorsName := milvusstore.VectorCollectionName(growingCollection)
	loadSharedCollections(t, client, growingCollection, vectorsName)
	generator := rand.New(rand.NewPCG(sharedSeed+31, sharedSeed+31))
	rows := make([]collection.Row, 0, growingRows)
	for rowIndex := range growingRows {
		row := reuseRow(fmt.Sprintf("g-%05d", rowIndex), reuseSparseQuery, wideVector(generator))
		if rowIndex%growingInsideEvery != 0 {
			row.Scalars["timestampUnix"] = collection.Int64Scalar(overflowOutsideTime)
		}
		rows = append(rows, row)
	}
	for start := 0; start < len(rows); start += growingBatch {
		if err := store.Upsert(ctx, growingCollection, sharedDeclaration, rows[start:min(start+growingBatch, len(rows))]); err != nil {
			t.Fatalf("upsert at %d: %v", start, err)
		}
	}
	time.Sleep(growingWait)
	logs, err := exec.Command("docker", "logs", stack.project+"-milvus").CombinedOutput()
	if err != nil {
		t.Fatalf("read Milvus logs: %v", err)
	}
	if !strings.Contains(string(logs), growingInterimIndex) {
		t.Fatalf("the Milvus log does not mention %s; the growing segment did not build an interim index", growingInterimIndex)
	}

	lower := overflowRangeLower
	upper := overflowRangeUpper
	filter := collection.ColumnRange("timestampUnix", &lower, &upper)
	superset, err := milvusstore.SupersetExpression(&filter, sharedDeclaration)
	if err != nil {
		t.Fatalf("superset expression: %v", err)
	}
	directKeys := func(vector []float32, limit int) []string {
		annParam := index.NewCustomAnnParam()
		annParam.WithExtraParam("nprobe", growingClusters)
		resultSets, err := client.Search(ctx, milvusclient.NewSearchOption(vectorsName, limit, []entity.Vector{entity.FloatVector(vector)}).
			WithANNSField(milvusstore.DenseVectorField).WithAnnParam(annParam).WithFilter(superset))
		if err != nil {
			t.Fatalf("direct vector search: %v", err)
		}
		keys := make([]string, 0, resultSets[0].ResultCount)
		for position := range resultSets[0].ResultCount {
			key, err := resultSets[0].IDs.GetAsString(position)
			if err != nil {
				t.Fatalf("read vector key: %v", err)
			}
			keys = append(keys, key)
		}
		return keys
	}
	limitDependent := 0
	for range growingQueries {
		vector := wideVector(generator)
		small := directKeys(vector, 80)
		large := directKeys(vector, 320)
		if !slices.Equal(small, large[:min(len(small), len(large))]) {
			limitDependent++
		}
		request := collection.SearchRequest{Collection: growingCollection, Query: reuseSparseQuery, Vector: vector, Filter: &filter, Declaration: sharedDeclaration}
		want := rankTwice(t, ctx, store, request)
		repeat := rankTwice(t, ctx, store, request)
		got := rankTwice(t, rankReuseScope(t, ctx), store, request)
		for call := range want {
			if !slices.Equal(want[call], repeat[call]) {
				t.Fatalf("two unscoped Rank calls %d differ on unchanged growing data", call+1)
			}
			if !slices.Equal(want[call], got[call]) {
				t.Fatalf("scoped Rank call %d returned candidates that differ from unscoped Rank on growing segments", call+1)
			}
		}
	}
	if limitDependent == 0 {
		t.Fatalf("top-80 equaled the first 80 keys of top-320 for all %d queries; the fixture needs limit dependent interim search", growingQueries)
	}
	t.Logf("growing segments: top-80 differed from the first 80 keys of top-320 for %d of %d queries; scoped Rank equaled unscoped Rank for every query", limitDependent, growingQueries)
}

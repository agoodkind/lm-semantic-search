//go:build milvusintegration

package milvusintegration

import (
	"bufio"
	"context"
	"fmt"
	"math/rand/v2"
	"net"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/index"
	"goodkind.io/lm-semantic-search/collection"
	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"
)

const (
	workCollection = "rank_reuse_work"
	workSettle     = 3 * time.Second
	workQueries    = 5
)

// vectorSearchCount returns the total Search requests for collectionName from the throwaway Milvus proxy metrics endpoint.
func vectorSearchCount(t *testing.T, stack throwawayStack, collectionName string) int {
	t.Helper()
	response, err := http.Get("http://" + net.JoinHostPort("127.0.0.1", strconv.Itoa(stack.healthPort)) + "/metrics")
	if err != nil {
		t.Fatalf("read Milvus metrics: %v", err)
	}
	defer func() { _ = response.Body.Close() }()
	label := `collection_name="` + collectionName + `"`
	total := 0.0
	scanner := bufio.NewScanner(response.Body)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "milvus_proxy_req_count{") || !strings.Contains(line, label) ||
			!strings.Contains(line, `function_name="Search"`) || !strings.Contains(line, `status="total"`) {
			continue
		}
		value, parseErr := strconv.ParseFloat(line[strings.LastIndex(line, " ")+1:], 64)
		if parseErr != nil {
			t.Fatalf("parse Milvus metric %q: %v", line, parseErr)
		}
		total += value
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan Milvus metrics: %v", err)
	}
	return int(total)
}

// TestSharedStoreRankReuseEndsBackgroundWork compares Milvus search counts for requests with one or two Rank calls, with and without a scope. It checks the allowed search counts and verifies that Milvus serves zero searches in the three seconds after each request ends.
func TestSharedStoreRankReuseEndsBackgroundWork(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, "4g")
	client := dialMilvus(t, stack)
	ctx := context.Background()
	options := milvusstore.Options{Hybrid: true, EmbeddingModel: "generated-model", DenseIndex: index.NewFlatIndex(entity.COSINE)}
	store, err := milvusstore.NewShared(client, options)
	if err != nil {
		t.Fatalf("new shared store: %v", err)
	}
	if err := store.EnsureCollection(ctx, collection.EnsureRequest{Collection: workCollection, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
		t.Fatalf("ensure shared collection: %v", err)
	}
	vectorsName := milvusstore.VectorCollectionName(workCollection)
	loadSharedCollections(t, client, workCollection, vectorsName)
	generator := rand.New(rand.NewPCG(sharedSeed+41, sharedSeed+41))
	rows := make([]collection.Row, 0, reuseRows)
	for rowIndex := range reuseRows {
		rows = append(rows, reuseRow(fmt.Sprintf("work-%04d", rowIndex), reuseSparseQuery, normalizedVector(generator)))
	}
	if err := store.Upsert(ctx, workCollection, sharedDeclaration, rows); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	time.Sleep(sharedVisibilityWait)

	// measure runs one request pattern and returns the Milvus search counts through the end of the request and during workSettle after the request ends.
	measure := func(scoped bool, limits []int32, vector []float32) (int, int) {
		before := vectorSearchCount(t, stack, vectorsName)
		requestCtx := ctx
		end := func() {}
		if scoped {
			requestCtx, end = milvusstore.WithRankReuse(ctx)
		}
		request := collection.SearchRequest{Collection: workCollection, Query: reuseSparseQuery, Vector: vector, Declaration: sharedDeclaration}
		for _, limit := range limits {
			request.Limit = limit
			if _, err := store.Rank(requestCtx, request); err != nil {
				t.Fatalf("rank at limit %d: %v", limit, err)
			}
		}
		end()
		ended := vectorSearchCount(t, stack, vectorsName)
		time.Sleep(workSettle)
		settled := vectorSearchCount(t, stack, vectorsName)
		return ended - before, settled - ended
	}
	patterns := []struct {
		name   string
		limits []int32
	}{{"one call", []int32{reuseFirstLimit}}, {"two calls", []int32{reuseFirstLimit, reuseSecondLimit}}}
	for _, pattern := range patterns {
		for range workQueries {
			vector := normalizedVector(generator)
			unscoped, unscopedLate := measure(false, pattern.limits, vector)
			scoped, scopedLate := measure(true, pattern.limits, vector)
			if unscopedLate != 0 || scopedLate != 0 {
				t.Fatalf("%s: Milvus served %d unscoped and %d scoped vector searches after the request ended", pattern.name, unscopedLate, scopedLate)
			}
			if scoped > unscoped+1 {
				t.Fatalf("%s: a scoped request made %d vector searches, an unscoped request made %d", pattern.name, scoped, unscoped)
			}
			if len(pattern.limits) > 1 && scoped > unscoped {
				t.Fatalf("%s: a scoped request made %d vector searches, more than the %d of an unscoped request", pattern.name, scoped, unscoped)
			}
			t.Logf("%s: unscoped %d vector searches, scoped %d; 0 after the request ended", pattern.name, unscoped, scoped)
		}
	}
}

//go:build milvusintegration

package milvusintegration

import (
	"context"
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/index"
	"goodkind.io/lm-semantic-search/collection"
	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"
)

const (
	reuseCollection   = "rank_reuse"
	reuseRows         = 400
	reuseFirstLimit   = 20
	reuseSecondLimit  = 80
	reuseSparseQuery  = "reuse occurrence text"
	reuseOtherContent = "unrelated words"
	reuseTimestamp    = int64(1_790_000_000)
	reuseConversation = "conv-reuse"
	reuseNewRowID     = "reuse-new"
)

func reuseRow(id string, content string, vector []float32) collection.Row {
	return collection.Row{
		ID:            id,
		Content:       content,
		RelativePath:  "conv/reuse/" + id,
		FileExtension: ".md",
		Metadata:      `{"kind":"message"}`,
		Vector:        vector,
		Scalars: map[string]collection.ScalarValue{
			"conversationId": collection.StringScalar(reuseConversation),
			"provider":       collection.StringScalar("claude"),
			"workspaceRoot":  collection.StringScalar("/ws/0"),
			"archived":       collection.BoolScalar(false),
			"timestampUnix":  collection.Int64Scalar(reuseTimestamp),
			"label":          {Type: collection.ScalarTypeString, Null: true},
		},
	}
}

func containsCandidate(candidates []milvusstore.Candidate, id string) bool {
	return slices.ContainsFunc(candidates, func(candidate milvusstore.Candidate) bool { return candidate.PrimaryKey == id })
}

// TestSharedStoreRankReuseMatchesFreshStoreAndSeesOtherWriters checks that scoped Rank results match a fresh SharedStore and that unscoped and newly scoped Rank calls return a matching row written by another SharedStore.
func TestSharedStoreRankReuseMatchesFreshStoreAndSeesOtherWriters(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, "4g")
	client := dialMilvus(t, stack)
	ctx := context.Background()
	options := milvusstore.Options{Hybrid: true, EmbeddingModel: "generated-model", DenseIndex: index.NewFlatIndex(entity.COSINE)}
	newStore := func() *milvusstore.SharedStore {
		store, err := milvusstore.NewShared(client, options)
		if err != nil {
			t.Fatalf("new shared store: %v", err)
		}
		return store
	}
	reader := newStore()
	if err := reader.EnsureCollection(ctx, collection.EnsureRequest{Collection: reuseCollection, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
		t.Fatalf("ensure shared collection: %v", err)
	}
	loadSharedCollections(t, client, reuseCollection, milvusstore.VectorCollectionName(reuseCollection))
	generator := rand.New(rand.NewPCG(sharedSeed+7, sharedSeed+7))
	rows := make([]collection.Row, 0, reuseRows)
	for rowIndex := range reuseRows {
		rows = append(rows, reuseRow(fmt.Sprintf("reuse-%04d", rowIndex), reuseSparseQuery, normalizedVector(generator)))
	}
	if err := reader.Upsert(ctx, reuseCollection, sharedDeclaration, rows); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	time.Sleep(sharedVisibilityWait)
	query := normalizedVector(generator)
	request := collection.SearchRequest{Collection: reuseCollection, Query: reuseSparseQuery, Vector: query, Declaration: sharedDeclaration}

	scoped := milvusstore.WithRankReuse(ctx)
	request.Limit = reuseFirstLimit
	if _, err := reader.Rank(scoped, request); err != nil {
		t.Fatalf("first scoped rank: %v", err)
	}
	request.Limit = reuseSecondLimit
	reused, err := reader.Rank(scoped, request)
	if err != nil {
		t.Fatalf("second scoped rank: %v", err)
	}
	fresh, err := newStore().Rank(ctx, request)
	if err != nil {
		t.Fatalf("fresh rank: %v", err)
	}
	if !slices.Equal(reused, fresh) {
		t.Fatalf("second scoped Rank returned %d candidates that differ from the %d candidates of a new store", len(reused), len(fresh))
	}

	writer := newStore()
	if err := writer.Upsert(ctx, reuseCollection, sharedDeclaration, []collection.Row{reuseRow(reuseNewRowID, reuseOtherContent, slices.Clone(query))}); err != nil {
		t.Fatalf("upsert through the second store: %v", err)
	}
	time.Sleep(sharedVisibilityWait)
	unscoped, err := reader.Rank(ctx, request)
	if err != nil {
		t.Fatalf("unscoped rank after the second store wrote: %v", err)
	}
	if !containsCandidate(unscoped, reuseNewRowID) {
		t.Fatalf("Rank without a reuse scope did not return row %s that a second SharedStore wrote", reuseNewRowID)
	}
	nextRequest, err := reader.Rank(milvusstore.WithRankReuse(ctx), request)
	if err != nil {
		t.Fatalf("rank in a new scope: %v", err)
	}
	if !containsCandidate(nextRequest, reuseNewRowID) {
		t.Fatalf("Rank in a new reuse scope did not return row %s that a second SharedStore wrote", reuseNewRowID)
	}
	t.Logf("second scoped Rank matched a new store at %d candidates; unscoped and new-scope Rank returned the row a second store wrote", len(reused))
}

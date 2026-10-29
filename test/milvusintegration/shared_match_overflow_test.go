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
	overflowCollection    = "match_overflow"
	overflowCrowdRows     = 34_000
	overflowNearestRows   = 20
	overflowUpsertBatch   = 1_000
	overflowInsideTime    = int64(1_790_000_000)
	overflowOutsideTime   = int64(1_700_000_000)
	overflowRangeLower    = int64(1_789_000_000)
	overflowRangeUpper    = int64(1_791_000_000)
	overflowSearchLimit   = 20
	overflowSparseQuery   = "unmatchedterm"
	overflowContent       = "repeated occurrence text"
	overflowCrowdPrefix   = "a"
	overflowNearestPrefix = "z"
	overflowConversation  = "conv-overflow"
)

// overflowRows returns count occurrence rows storing vector, with IDs and paths starting with prefix, and the sorted IDs of matching rows.
// Rows with an even index match the date filter.
func overflowRows(prefix string, count int, vector []float32) ([]collection.Row, []string) {
	rows := make([]collection.Row, 0, count)
	matching := make([]string, 0, count/2)
	for rowIndex := range count {
		id := fmt.Sprintf("%s-%05d", prefix, rowIndex)
		timestamp := overflowOutsideTime
		if rowIndex%2 == 0 {
			timestamp = overflowInsideTime
			matching = append(matching, id)
		}
		rows = append(rows, collection.Row{
			ID:            id,
			Content:       overflowContent,
			RelativePath:  fmt.Sprintf("conv/overflow/%s", id),
			StartLine:     int32(rowIndex),
			EndLine:       int32(rowIndex),
			FileExtension: ".md",
			Metadata:      `{"kind":"message"}`,
			Vector:        slices.Clone(vector),
			Scalars: map[string]collection.ScalarValue{
				"conversationId": collection.StringScalar(overflowConversation),
				"provider":       collection.StringScalar("claude"),
				"workspaceRoot":  collection.StringScalar("/ws/0"),
				"archived":       collection.BoolScalar(false),
				"timestampUnix":  collection.Int64Scalar(timestamp),
				"label":          {Type: collection.ScalarTypeString, Null: true},
			},
		})
	}
	slices.Sort(matching)
	return rows, matching
}

// TestSharedStoreDenseLegReadsEveryMatchingOccurrence checks Rank with a date filter and limit 20.
// The test writes 20 z rows storing the query vector and 34,000 a rows storing a second vector.
// The filter matches 10 z rows and 17,000 a rows. The sparse query does not match any row.
// One query returns at most 16,384 rows. Matching z IDs sort after all matching a IDs.
// Rank must return the 10 matching z IDs, followed by the 70 smallest matching a IDs.
// This result requires the dense leg to read every matching occurrence of both vectors.
func TestSharedStoreDenseLegReadsEveryMatchingOccurrence(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, "4g")
	client := dialMilvus(t, stack)
	ctx := context.Background()
	options := milvusstore.Options{Hybrid: true, EmbeddingModel: "generated-model", DenseIndex: index.NewFlatIndex(entity.COSINE)}
	shared, err := milvusstore.NewShared(client, options)
	if err != nil {
		t.Fatalf("new shared store: %v", err)
	}
	if err := shared.EnsureCollection(ctx, collection.EnsureRequest{Collection: overflowCollection, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
		t.Fatalf("ensure shared collection: %v", err)
	}
	loadSharedCollections(t, client, overflowCollection, milvusstore.VectorCollectionName(overflowCollection))

	generator := rand.New(rand.NewPCG(sharedSeed, sharedSeed))
	nearestVector := normalizedVector(generator)
	crowdVector := normalizedVector(generator)
	nearestRows, nearestMatching := overflowRows(overflowNearestPrefix, overflowNearestRows, nearestVector)
	crowdRows, crowdMatching := overflowRows(overflowCrowdPrefix, overflowCrowdRows, crowdVector)
	rows := append(nearestRows, crowdRows...)
	for start := 0; start < len(rows); start += overflowUpsertBatch {
		if err := shared.Upsert(ctx, overflowCollection, sharedDeclaration, rows[start:min(start+overflowUpsertBatch, len(rows))]); err != nil {
			t.Fatalf("shared upsert at %d: %v", start, err)
		}
	}
	time.Sleep(sharedVisibilityWait)
	if keys := distinctVectorKeys(rows); len(keys) != 2 {
		t.Fatalf("rows reference %d vectors, want 2", len(keys))
	}

	lower := overflowRangeLower
	upper := overflowRangeUpper
	filter := collection.ColumnRange("timestampUnix", &lower, &upper)
	candidates, err := shared.Rank(ctx, collection.SearchRequest{
		Collection:  overflowCollection,
		Query:       overflowSparseQuery,
		Vector:      nearestVector,
		Limit:       overflowSearchLimit,
		Filter:      &filter,
		Declaration: sharedDeclaration,
	})
	if err != nil {
		t.Fatalf("rank: %v", err)
	}
	depth := min(max(overflowSearchLimit*4, 64), collection.RankingDepth)
	want := append(slices.Clone(nearestMatching), crowdMatching[:depth-len(nearestMatching)]...)
	got := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		got = append(got, candidate.PrimaryKey)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("dense leg returned %d candidates %v, want the %d matching rows of the nearest vector and then the smallest matching IDs of the other vector %v",
			len(got), got, len(nearestMatching), want)
	}
	t.Logf("dense leg read the %d matching rows of the nearest vector and %d matching rows of the other vector, and kept %d rows in vector score and ID order",
		len(nearestMatching), len(crowdMatching), depth)
}

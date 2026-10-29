//go:build milvusintegration

package milvusintegration

import (
	"context"
	"fmt"
	"math/rand/v2"
	"sort"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/index"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"
)

const (
	nprobeRows       = 4096
	nprobeClusters   = 64
	nprobeQueries    = 20
	nprobeLimit      = 10
	nprobeSeed       = 29
	nprobeMinRecall  = 0.99
	nprobeIndexWait  = 3 * time.Minute
	nprobeBaseline   = "nprobe_baseline"
	nprobeShared     = "nprobe_shared"
	nprobeAbsentText = "zzqxnprobeabsent"
)

func nprobeRowsFor(generator *rand.Rand) []collection.Row {
	rows := make([]collection.Row, 0, nprobeRows)
	for rowIndex := range nprobeRows {
		rows = append(rows, collection.Row{
			ID:                fmt.Sprintf("row-%05d", rowIndex),
			Content:           fmt.Sprintf("dense row %d", rowIndex),
			RelativePath:      fmt.Sprintf("conv/conv-000/%05d", rowIndex),
			StartLine:         int32(rowIndex),
			EndLine:           int32(rowIndex),
			FileExtension:     ".md",
			Metadata:          `{"kind":"message"}`,
			SplitPart:         0,
			SplitPartRecorded: false,
			Vector:            normalizedVector(generator),
			Scalars: map[string]collection.ScalarValue{
				"conversationId": collection.StringScalar("conv-000"),
				"provider":       collection.StringScalar("codex"),
				"workspaceRoot":  collection.StringScalar("/ws/0"),
				"archived":       collection.BoolScalar(false),
				"timestampUnix":  collection.Int64Scalar(1_780_000_000),
				"label":          {Type: collection.ScalarTypeString, Null: true},
			},
		})
	}
	return rows
}

// exactTopIDs returns the IDs of the limit rows with the largest cosine
// similarity to query. Every vector is unit length.
func exactTopIDs(rows []collection.Row, query []float32, limit int) []string {
	type scored struct {
		id    string
		score float64
	}
	ranked := make([]scored, 0, len(rows))
	for _, row := range rows {
		dot := 0.0
		for position, value := range row.Vector {
			dot += float64(value) * float64(query[position])
		}
		ranked = append(ranked, scored{id: row.ID, score: dot})
	}
	sort.Slice(ranked, func(left int, right int) bool { return ranked[left].score > ranked[right].score })
	ids := make([]string, 0, limit)
	for _, row := range ranked[:limit] {
		ids = append(ids, row.id)
	}
	return ids
}

// sealAndIndex flushes a collection, waits until its dense index covers every
// row when dense is true, and loads it again. Milvus searches a growing
// segment exhaustively, and only an indexed sealed segment applies nprobe.
func sealAndIndex(t *testing.T, client *milvusclient.Client, name string, dense bool) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), nprobeIndexWait)
	defer cancel()
	task, err := client.Flush(ctx, milvusclient.NewFlushOption(name))
	if err != nil {
		t.Fatalf("flush %s: %v", name, err)
	}
	if err := task.Await(ctx); err != nil {
		t.Fatalf("await flush %s: %v", name, err)
	}
	for dense {
		description, err := client.DescribeIndex(ctx, milvusclient.NewDescribeIndexOption(name, milvusstore.DenseVectorField))
		if err != nil {
			t.Fatalf("describe index %s: %v", name, err)
		}
		if description.TotalRows > 0 && description.IndexedRows == description.TotalRows && description.PendingIndexRows == 0 {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatalf("index of %s covers %d of %d rows after %s", name, description.IndexedRows, description.TotalRows, nprobeIndexWait)
		case <-time.After(time.Second):
		}
	}
	if err := client.ReleaseCollection(ctx, milvusclient.NewReleaseCollectionOption(name)); err != nil {
		t.Fatalf("release %s: %v", name, err)
	}
	loadTask, err := client.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(name))
	if err != nil {
		t.Fatalf("load %s: %v", name, err)
	}
	if err := loadTask.Await(ctx); err != nil {
		t.Fatalf("await load %s: %v", name, err)
	}
}

// TestDenseSearchParamsReachTheIndex indexes 4,096 random unit vectors with
// IVF_FLAT over 64 clusters in a single-collection store and in a shared pair.
// Both stores search with nprobe 64, which scans every cluster of an exact
// IVF_FLAT index. Each search must return the exact top rows. Milvus scans 8
// clusters when a search does not send nprobe in its index parameters.
func TestDenseSearchParamsReachTheIndex(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, "2g")
	client := dialMilvus(t, stack)
	ctx := context.Background()
	options := milvusstore.Options{
		Hybrid:            true,
		EmbeddingModel:    "generated-model",
		DenseIndex:        index.NewIvfFlatIndex(entity.COSINE, nprobeClusters),
		DenseSearchParams: map[string]string{"nprobe": fmt.Sprint(nprobeClusters)},
	}
	shared, err := milvusstore.NewShared(client, options)
	if err != nil {
		t.Fatalf("new shared store: %v", err)
	}
	generator := rand.New(rand.NewPCG(nprobeSeed, nprobeSeed))
	rows := nprobeRowsFor(generator)
	targets := []struct {
		label       string
		store       collection.Store
		collection  string
		collections []string
		indexed     string
	}{
		{label: "single", store: milvusstore.New(client, options), collection: nprobeBaseline, collections: []string{nprobeBaseline}, indexed: nprobeBaseline},
		{label: "shared", store: shared, collection: nprobeShared, collections: []string{nprobeShared, milvusstore.VectorCollectionName(nprobeShared)}, indexed: milvusstore.VectorCollectionName(nprobeShared)},
	}
	for _, target := range targets {
		if err := target.store.EnsureCollection(ctx, collection.EnsureRequest{Collection: target.collection, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
			t.Fatalf("%s: ensure: %v", target.label, err)
		}
		loadSharedCollections(t, client, target.collections...)
		if err := target.store.Upsert(ctx, target.collection, sharedDeclaration, rows); err != nil {
			t.Fatalf("%s: upsert: %v", target.label, err)
		}
		for _, name := range target.collections {
			sealAndIndex(t, client, name, name == target.indexed)
		}
		matched := 0
		for range nprobeQueries {
			query := normalizedVector(generator)
			hits, err := target.store.Search(ctx, collection.SearchRequest{
				Collection:  target.collection,
				Query:       nprobeAbsentText,
				Vector:      query,
				Limit:       nprobeLimit,
				Declaration: sharedDeclaration,
			})
			if err != nil {
				t.Fatalf("%s: search: %v", target.label, err)
			}
			want := map[string]bool{}
			for _, id := range exactTopIDs(rows, query, nprobeLimit) {
				want[id] = true
			}
			for _, hit := range hits {
				if want[hit.ID] {
					matched++
				}
			}
		}
		recall := float64(matched) / float64(nprobeQueries*nprobeLimit)
		t.Logf("%s: recall@%d over %d queries = %.3f", target.label, nprobeLimit, nprobeQueries, recall)
		if recall < nprobeMinRecall {
			t.Fatalf("%s: recall %.3f below %.2f with nprobe %d of %d clusters", target.label, recall, nprobeMinRecall, nprobeClusters, nprobeClusters)
		}
	}
}

//go:build milvusintegration

package milvusintegration

import (
	"context"
	"math"
	"math/rand/v2"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v3/column"
	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/index"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"
)

const (
	scalarUpdateCollection = "shared_scalar_update"
	scalarUpdateZeroTime   = int64(-62135596800)
	scalarUpdateOldTime    = int64(1_780_000_000)
	scalarUpdateNewTimeA   = int64(1_790_000_000)
	scalarUpdateNewTimeC   = int64(1_790_000_100)
	scalarUpdateOtherTime  = int64(1_700_000_000)
	scalarUpdateRangeLow   = int64(1_789_999_000)
	scalarUpdateRangeHigh  = int64(1_790_001_000)
	scalarUpdateQuery      = "zzzunmatchedterm"
)

// storedScalarRow contains the occurrence row fields that the scalar update
// test compares before and after an update.
type storedScalarRow struct {
	content        string
	relativePath   string
	metadata       string
	conversationID string
	vectorKey      string
	timestampUnix  int64
}

// TestSharedStoreUpdateRowScalarsWidensVectorRanges stores four occurrences
// on a temporary Milvus stack. Two occurrences share a vector, and two have
// Go zero timestamps. The test checks that updates with a missing ID or empty
// metadata fail without changing stored rows. It updates the timestamps and
// metadata of the two zero time rows, then checks that other fields, vector
// bytes, and the vector count remain unchanged. It also checks that a
// date-filtered search with a query that matches no text returns both updated
// rows through dense search.
func TestSharedStoreUpdateRowScalarsWidensVectorRanges(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, "4g")
	client := dialMilvus(t, stack)
	ctx := context.Background()
	shared, err := milvusstore.NewShared(client, milvusstore.Options{Hybrid: true, EmbeddingModel: "generated-model", DenseIndex: index.NewFlatIndex(entity.COSINE)})
	if err != nil {
		t.Fatalf("new shared store: %v", err)
	}
	if err := shared.EnsureCollection(ctx, collection.EnsureRequest{Collection: scalarUpdateCollection, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
		t.Fatalf("ensure %s: %v", scalarUpdateCollection, err)
	}
	vectorCollection := milvusstore.VectorCollectionName(scalarUpdateCollection)
	loadSharedCollections(t, client, scalarUpdateCollection, vectorCollection)

	generator := rand.New(rand.NewPCG(sharedSeed, sharedSeed+7))
	shared1, shared2, shared3 := normalizedVector(generator), normalizedVector(generator), normalizedVector(generator)
	rows := []collection.Row{
		scalarUpdateRow("upd-a", "conv-a", "first occurrence of the shared vector", shared1, scalarUpdateZeroTime),
		scalarUpdateRow("upd-b", "conv-b", "second occurrence of the shared vector", shared1, scalarUpdateOldTime),
		scalarUpdateRow("upd-c", "conv-c", "only occurrence of the second vector", shared2, scalarUpdateZeroTime),
		scalarUpdateRow("upd-d", "conv-d", "untouched occurrence of the third vector", shared3, scalarUpdateOtherTime),
	}
	if err := shared.Upsert(ctx, scalarUpdateCollection, sharedDeclaration, rows); err != nil {
		t.Fatalf("shared upsert: %v", err)
	}
	time.Sleep(sharedVisibilityWait)
	before := readScalarRows(t, client, rows)
	vectorsBefore := readVectorBits(t, client, vectorCollection, [][]float32{shared1, shared2, shared3})
	for _, vector := range [][]float32{shared1, shared2} {
		if hits := scalarUpdateSearch(t, shared, vector); len(hits) != 0 {
			t.Fatalf("date-filtered search returned rows before the update: %v", hitIDs(hits))
		}
	}

	timestampColumn := []collection.ScalarColumn{{Name: "timestampUnix", Type: collection.ScalarTypeInt64, Nullable: false, MaxLength: 0}}
	missingRow := scalarUpdateChange("upd-missing", scalarUpdateNewTimeA)
	if _, err := shared.UpdateRowScalars(ctx, scalarUpdateCollection, timestampColumn, []collection.Row{scalarUpdateChange("upd-a", scalarUpdateNewTimeA), missingRow}); err == nil {
		t.Fatal("update with a missing id did not fail")
	}
	emptyMetadata := scalarUpdateChange("upd-a", scalarUpdateNewTimeA)
	emptyMetadata.Metadata = ""
	if _, err := shared.UpdateRowScalars(ctx, scalarUpdateCollection, timestampColumn, []collection.Row{emptyMetadata}); err == nil {
		t.Fatal("update with empty metadata did not fail")
	}
	time.Sleep(sharedVisibilityWait)
	if after := readScalarRows(t, client, rows); !scalarRowsEqual(before, after) {
		t.Fatalf("rejected update changed stored rows; before and after: %+v, %+v", before, after)
	}

	updated, err := shared.UpdateRowScalars(ctx, scalarUpdateCollection, timestampColumn, []collection.Row{
		scalarUpdateChange("upd-a", scalarUpdateNewTimeA),
		scalarUpdateChange("upd-c", scalarUpdateNewTimeC),
	})
	if err != nil || updated != 2 {
		t.Fatalf("row scalar update failed; count and error: %d, %v", updated, err)
	}
	time.Sleep(sharedVisibilityWait)

	after := readScalarRows(t, client, rows)
	want := map[string]storedScalarRow{}
	for id, row := range before {
		want[id] = row
	}
	for id, newTime := range map[string]int64{"upd-a": scalarUpdateNewTimeA, "upd-c": scalarUpdateNewTimeC} {
		row := want[id]
		row.timestampUnix = newTime
		row.metadata = scalarUpdateChange(id, newTime).Metadata
		want[id] = row
	}
	if !scalarRowsEqual(after, want) {
		t.Fatalf("stored rows after the update differ; got and want: %+v, %+v", after, want)
	}
	if vectorsAfter := readVectorBits(t, client, vectorCollection, [][]float32{shared1, shared2, shared3}); !vectorBitsEqual(vectorsBefore, vectorsAfter) {
		t.Fatal("vector bytes changed after the update")
	}
	if got := countRows(t, client, vectorCollection); got != 3 {
		t.Fatalf("vector collection row count is wrong: %d", got)
	}
	for _, vector := range [][]float32{shared1, shared2} {
		ids := hitIDs(scalarUpdateSearch(t, shared, vector))
		slices.Sort(ids)
		if !slices.Equal(ids, []string{"upd-a", "upd-c"}) {
			t.Fatalf("date-filtered search did not return exactly the updated rows: %v", ids)
		}
	}
}

func scalarUpdateRow(id string, conversation string, content string, vector []float32, timestampUnix int64) collection.Row {
	return collection.Row{
		ID:                id,
		Content:           content,
		RelativePath:      "conv/" + conversation + "/0",
		StartLine:         0,
		EndLine:           0,
		FileExtension:     ".md",
		Metadata:          `{"kind":"message"}`,
		SplitPart:         0,
		SplitPartRecorded: false,
		Vector:            slices.Clone(vector),
		Scalars: map[string]collection.ScalarValue{
			"conversationId": collection.StringScalar(conversation),
			"provider":       collection.StringScalar("cursor"),
			"workspaceRoot":  collection.StringScalar("/ws/0"),
			"archived":       collection.BoolScalar(false),
			"timestampUnix":  collection.Int64Scalar(timestampUnix),
			"label":          {Type: collection.ScalarTypeString, Null: true},
		},
	}
}

func scalarUpdateChange(id string, timestampUnix int64) collection.Row {
	return collection.Row{
		ID:                id,
		Content:           "",
		RelativePath:      "",
		StartLine:         0,
		EndLine:           0,
		FileExtension:     "",
		Metadata:          `{"kind":"message","timestamp_unix":` + strconv.FormatInt(timestampUnix, 10) + `}`,
		SplitPart:         0,
		SplitPartRecorded: false,
		Vector:            nil,
		Scalars:           map[string]collection.ScalarValue{"timestampUnix": collection.Int64Scalar(timestampUnix)},
	}
}

func scalarUpdateSearch(t *testing.T, shared *milvusstore.SharedStore, vector []float32) []collection.Hit {
	t.Helper()
	lower, upper := scalarUpdateRangeLow, scalarUpdateRangeHigh
	filter := collection.ColumnRange("timestampUnix", &lower, &upper)
	hits, err := shared.Search(context.Background(), collection.SearchRequest{
		Collection:  scalarUpdateCollection,
		Query:       scalarUpdateQuery,
		Vector:      vector,
		Limit:       sharedSearchLimit,
		Filter:      &filter,
		Declaration: sharedDeclaration,
	})
	if err != nil {
		t.Fatalf("shared search failed: %v", err)
	}
	return hits
}

func hitIDs(hits []collection.Hit) []string {
	ids := make([]string, 0, len(hits))
	for _, hit := range hits {
		ids = append(ids, hit.ID)
	}
	return ids
}

func readScalarRows(t *testing.T, client *milvusclient.Client, rows []collection.Row) map[string]storedScalarRow {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), milvusCallTimeout)
	defer cancel()
	ids := make([]string, 0, len(rows))
	for _, row := range rows {
		ids = append(ids, row.ID)
	}
	resultSet, err := client.Query(ctx, milvusclient.NewQueryOption(scalarUpdateCollection).
		WithIDs(column.NewColumnVarChar(milvusstore.IDField, ids)).
		WithOutputFields(milvusstore.IDField, milvusstore.ContentField, milvusstore.RelativePathField, milvusstore.MetadataField, "conversationId", milvusstore.VectorKeyField, "timestampUnix").
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		t.Fatalf("read occurrence rows: %v", err)
	}
	stored := map[string]storedScalarRow{}
	for rowIndex := range resultSet.ResultCount {
		id, _ := resultSet.GetColumn(milvusstore.IDField).GetAsString(rowIndex)
		content, _ := resultSet.GetColumn(milvusstore.ContentField).GetAsString(rowIndex)
		path, _ := resultSet.GetColumn(milvusstore.RelativePathField).GetAsString(rowIndex)
		metadata, _ := resultSet.GetColumn(milvusstore.MetadataField).GetAsString(rowIndex)
		conversation, _ := resultSet.GetColumn("conversationId").GetAsString(rowIndex)
		key, _ := resultSet.GetColumn(milvusstore.VectorKeyField).GetAsString(rowIndex)
		timestampUnix, _ := resultSet.GetColumn("timestampUnix").GetAsInt64(rowIndex)
		stored[id] = storedScalarRow{content: content, relativePath: path, metadata: metadata, conversationID: conversation, vectorKey: key, timestampUnix: timestampUnix}
	}
	if len(stored) != len(rows) {
		t.Fatalf("occurrence query returned the wrong number of rows; got and want: %d, %d", len(stored), len(rows))
	}
	return stored
}

func readVectorBits(t *testing.T, client *milvusclient.Client, vectorCollection string, vectors [][]float32) map[string][]uint32 {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), milvusCallTimeout)
	defer cancel()
	keys := make([]string, 0, len(vectors))
	for _, vector := range vectors {
		keys = append(keys, milvusstore.VectorKey(vector))
	}
	resultSet, err := client.Query(ctx, milvusclient.NewQueryOption(vectorCollection).
		WithIDs(column.NewColumnVarChar(milvusstore.VectorKeyField, keys)).
		WithOutputFields(milvusstore.VectorKeyField, milvusstore.DenseVectorField).
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		t.Fatalf("read vector rows: %v", err)
	}
	bits := map[string][]uint32{}
	for rowIndex := range resultSet.ResultCount {
		key, _ := resultSet.GetColumn(milvusstore.VectorKeyField).GetAsString(rowIndex)
		vector, vectorErr := milvusstore.VectorAt(resultSet.GetColumn(milvusstore.DenseVectorField), rowIndex)
		if vectorErr != nil {
			t.Fatalf("read vector rows: %v", vectorErr)
		}
		rowBits := make([]uint32, 0, len(vector))
		for _, value := range vector {
			rowBits = append(rowBits, math.Float32bits(value))
		}
		bits[key] = rowBits
	}
	if len(bits) != len(vectors) {
		t.Fatalf("vector collection row count is wrong: %d", len(bits))
	}
	return bits
}

func scalarRowsEqual(left map[string]storedScalarRow, right map[string]storedScalarRow) bool {
	if len(left) != len(right) {
		return false
	}
	for id, row := range left {
		if right[id] != row {
			return false
		}
	}
	return true
}

func vectorBitsEqual(left map[string][]uint32, right map[string][]uint32) bool {
	if len(left) != len(right) {
		return false
	}
	for key, bits := range left {
		if !slices.Equal(bits, right[key]) {
			return false
		}
	}
	return true
}

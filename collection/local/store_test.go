package local_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"goodkind.io/lm-semantic-search/collection"
	"goodkind.io/lm-semantic-search/collection/local"
)

const (
	testCollection  = "conv_chunks_memory"
	itemColumn      = "itemId"
	archivedColumn  = "archived"
	workspaceColumn = "workspaceRoot"
	testDimension   = 3
)

func testDeclaration() collection.Declaration {
	return collection.Declaration{
		ItemIDColumn: itemColumn,
		Scalars: []collection.ScalarColumn{
			{Name: itemColumn, Type: collection.ScalarTypeString, Nullable: true, MaxLength: 256},
			{Name: archivedColumn, Type: collection.ScalarTypeBool, Nullable: true, MaxLength: 0},
			{Name: workspaceColumn, Type: collection.ScalarTypeString, Nullable: true, MaxLength: 1024},
		},
	}
}

func testRow(id string, relativePath string, vector []float32, scalars map[string]collection.ScalarValue) collection.Row {
	return collection.Row{
		ID:                id,
		Content:           "content of " + id,
		RelativePath:      relativePath,
		StartLine:         0,
		EndLine:           0,
		FileExtension:     "",
		Metadata:          "{}",
		SplitPart:         0,
		SplitPartRecorded: false,
		Vector:            vector,
		Scalars:           scalars,
	}
}

func openStore(t *testing.T, root string) *local.Store {
	t.Helper()
	store, err := local.Open(local.Options{Root: root, EmbeddingModel: "test-model"})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(store.Close)
	return store
}

func newTestStore(t *testing.T, rows []collection.Row) *local.Store {
	t.Helper()
	store := openStore(t, t.TempDir())
	request := collection.EnsureRequest{Collection: testCollection, Declaration: testDeclaration(), Dimension: testDimension}
	if err := store.EnsureCollection(context.Background(), request); err != nil {
		t.Fatalf("EnsureCollection: %v", err)
	}
	if err := store.Upsert(context.Background(), testCollection, testDeclaration(), rows); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	return store
}

func searchIDs(t *testing.T, store *local.Store, request collection.SearchRequest) []string {
	t.Helper()
	request.Collection = testCollection
	request.Vector = []float32{1, 0, 0}
	request.Declaration = testDeclaration()
	hits, err := store.Search(context.Background(), request)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	ids := make([]string, 0, len(hits))
	for _, hit := range hits {
		ids = append(ids, hit.ID)
	}
	return ids
}

func queryAll(t *testing.T, store *local.Store) []collection.Hit {
	t.Helper()
	request := collection.QueryRequest{Collection: testCollection, Declaration: testDeclaration(), Filter: nil, Limit: 100}
	hits, err := store.Query(context.Background(), request)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	return hits
}

func rankingRows() []collection.Row {
	itemA := map[string]collection.ScalarValue{
		itemColumn:     collection.StringScalar("a"),
		archivedColumn: collection.BoolScalar(false),
	}
	return []collection.Row{
		testRow("a0", "doc/a/0", []float32{1, 0, 0}, itemA),
		testRow("a1", "doc/a/1", []float32{0.9, 0.1, 0}, itemA),
		testRow("a2", "doc/a/2", []float32{0.8, 0.2, 0}, itemA),
		testRow("b0", "doc/b/0", []float32{0.5, 0.5, 0}, map[string]collection.ScalarValue{
			itemColumn: collection.StringScalar("b"),
		}),
		testRow("c0", "doc/c/0", []float32{5, 0, 0}, map[string]collection.ScalarValue{
			itemColumn:     collection.StringScalar("c"),
			archivedColumn: collection.BoolScalar(true),
		}),
		testRow("n0", "doc/n/0", []float32{0, 1, 0}, nil),
	}
}

func TestSearchRanksByCosineSimilarity(t *testing.T) {
	t.Parallel()

	store := newTestStore(t, rankingRows())
	got := searchIDs(t, store, collection.SearchRequest{Limit: 10})
	want := []string{"a0", "c0", "a1", "a2", "b0", "n0"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ranked IDs = %v, want %v", got, want)
	}
	prefix := searchIDs(t, store, collection.SearchRequest{Limit: 2})
	if !reflect.DeepEqual(prefix, want[:2]) {
		t.Fatalf("limit 2 returned %v, want the prefix %v", prefix, want[:2])
	}
}

func TestSearchAppliesFilterGroupCapAndScoreFloor(t *testing.T) {
	t.Parallel()

	store := newTestStore(t, rankingRows())

	notArchived := collection.Negate(collection.ColumnEquals(archivedColumn, collection.BoolScalar(true)))
	filtered := searchIDs(t, store, collection.SearchRequest{Limit: 10, Filter: &notArchived})
	if want := []string{"a0", "a1", "a2"}; !reflect.DeepEqual(filtered, want) {
		t.Fatalf("filtered IDs = %v, want %v (b0 and n0 do not store an archived value)", filtered, want)
	}

	capped := searchIDs(t, store, collection.SearchRequest{Limit: 10, GroupBy: itemColumn, PerGroupLimit: 1})
	if want := []string{"a0", "c0", "b0", "n0"}; !reflect.DeepEqual(capped, want) {
		t.Fatalf("capped IDs = %v, want %v", capped, want)
	}

	floored := searchIDs(t, store, collection.SearchRequest{Limit: 10, MinScore: 0.75})
	if want := []string{"a0", "c0", "a1", "a2"}; !reflect.DeepEqual(floored, want) {
		t.Fatalf("floored IDs = %v, want %v (b0 scores 0.707)", floored, want)
	}
}

func TestUpsertReplacesByIDAndRejectsWholeBatch(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := newTestStore(t, rankingRows())

	replacement := testRow("a0", "doc/a/0", []float32{0, 0, 1}, nil)
	replacement.Content = "replaced"
	if err := store.Upsert(ctx, testCollection, testDeclaration(), []collection.Row{replacement}); err != nil {
		t.Fatalf("Upsert replacement: %v", err)
	}
	all := queryAll(t, store)
	if len(all) != 6 || all[0].ID != "a0" || all[0].Content != "replaced" {
		t.Fatalf("Query returned %d rows with first %+v, want 6 rows and a0 replaced", len(all), all[0])
	}
	if cell := all[0].Scalars[itemColumn]; cell.State != collection.ScalarCellNull {
		t.Fatalf("replaced a0 item cell = %+v, want null", cell)
	}

	batch := []collection.Row{
		testRow("new", "doc/new/0", []float32{1, 0, 0}, nil),
		testRow("narrow", "doc/narrow/0", []float32{1, 0}, nil),
	}
	if err := store.Upsert(ctx, testCollection, testDeclaration(), batch); err == nil {
		t.Fatal("Upsert accepted a vector of width 2 in a collection of width 3")
	}
	if after := queryAll(t, store); len(after) != 6 {
		t.Fatalf("store has %d rows after a rejected batch, want 6", len(after))
	}

	empty := openStore(t, t.TempDir())
	_, err := empty.Search(ctx, collection.SearchRequest{Collection: testCollection, Vector: []float32{1, 0, 0}, Limit: 1, Declaration: testDeclaration()})
	if !errors.Is(err, collection.ErrCollectionMissing) {
		t.Fatalf("Search on a new store returned %v, want ErrCollectionMissing", err)
	}
}

func itemRows() []collection.Row {
	return []collection.Row{
		testRow("a0", "conv/a/0", []float32{1, 0, 0}, map[string]collection.ScalarValue{
			itemColumn:      collection.StringScalar("a"),
			workspaceColumn: collection.StringScalar(""),
		}),
		testRow("a1", "conv/a/1", []float32{0, 1, 0}, map[string]collection.ScalarValue{
			itemColumn: collection.StringScalar("a"),
		}),
		testRow("b0", "conv/b/0", []float32{0, 0, 1}, map[string]collection.ScalarValue{
			itemColumn:      collection.StringScalar("b"),
			workspaceColumn: collection.StringScalar("/work/b"),
		}),
		testRow("legacy", "conv/a/5", []float32{1, 1, 0}, nil),
		testRow("z0", "conv/z/0", []float32{1, 0, 1}, map[string]collection.ScalarValue{
			itemColumn: collection.StringScalar("z"),
		}),
	}
}

func TestBackfillScalarsFillsMissingValuesOfStreamedItems(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := newTestStore(t, itemRows())
	backfill := collection.ScalarBackfill{
		ItemColumn: itemColumn,
		Columns: []collection.ScalarColumn{
			{Name: workspaceColumn, Type: collection.ScalarTypeString, Nullable: true, MaxLength: 1024},
		},
		Values: map[string]map[string]collection.ScalarValue{
			"a": {workspaceColumn: collection.StringScalar("/work/a")},
		},
		LegacyPathFamilies: []string{"conv/"},
		DryRun:             true,
	}
	workspaces := func() map[string]string {
		t.Helper()
		values := make(map[string]string)
		for _, row := range queryAll(t, store) {
			cell := row.Scalars[workspaceColumn]
			values[row.ID] = string(cell.State) + ":" + cell.Value.String
		}
		return values
	}
	before := workspaces()

	changed, orphan, err := store.BackfillScalars(ctx, testCollection, backfill)
	if err != nil {
		t.Fatalf("dry run: %v", err)
	}
	if changed != 3 || orphan != 1 {
		t.Fatalf("dry run counted changed=%d orphan=%d, want 3 and 1", changed, orphan)
	}
	if after := workspaces(); !reflect.DeepEqual(after, before) {
		t.Fatalf("dry run changed stored values: %v, want %v", after, before)
	}

	backfill.DryRun = false
	changed, orphan, err = store.BackfillScalars(ctx, testCollection, backfill)
	if err != nil {
		t.Fatalf("backfill: %v", err)
	}
	if changed != 3 || orphan != 1 {
		t.Fatalf("backfill counted changed=%d orphan=%d, want 3 and 1", changed, orphan)
	}
	want := map[string]string{
		"a0":     "value:/work/a",
		"a1":     "value:/work/a",
		"b0":     "value:/work/b",
		"legacy": "value:/work/a",
		"z0":     "null:",
	}
	if after := workspaces(); !reflect.DeepEqual(after, want) {
		t.Fatalf("stored workspaces = %v, want %v", after, want)
	}
}

func TestItemRowsReadAndDelete(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := newTestStore(t, itemRows())

	rows, err := store.QueryRows(ctx, collection.RowsRequest{
		Collection:    testCollection,
		Declaration:   testDeclaration(),
		ItemIDs:       []string{"a"},
		PathPrefixes:  []string{"conv/a/"},
		IncludeVector: true,
	})
	if err != nil {
		t.Fatalf("QueryRows: %v", err)
	}
	ids := make([]string, 0, len(rows))
	for _, row := range rows {
		ids = append(ids, row.ID)
		if len(row.Vector) != testDimension || row.ContentHash == "" || row.EmbeddingModel != "test-model" {
			t.Fatalf("row %s = %+v, want a vector, a content hash, and the store's model", row.ID, row)
		}
	}
	if want := []string{"a0", "a1", "legacy"}; !reflect.DeepEqual(ids, want) {
		t.Fatalf("QueryRows IDs = %v, want %v", ids, want)
	}

	removed, err := store.DeleteItems(ctx, collection.DeleteItemsRequest{
		Collection:   testCollection,
		Declaration:  testDeclaration(),
		ItemIDs:      []string{"a"},
		PathPrefixes: []string{"conv/a/"},
	})
	if err != nil {
		t.Fatalf("DeleteItems: %v", err)
	}
	if removed != 3 {
		t.Fatalf("DeleteItems removed %d rows, want 3", removed)
	}
	remaining := queryAll(t, store)
	if len(remaining) != 2 || remaining[0].ID != "b0" || remaining[1].ID != "z0" {
		t.Fatalf("remaining rows = %+v, want b0 and z0", remaining)
	}

	itemZ := collection.ColumnEquals(itemColumn, collection.StringScalar("z"))
	deleted, err := store.Delete(ctx, testCollection, itemZ)
	if err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if deleted != 1 {
		t.Fatalf("Delete removed %d rows, want 1", deleted)
	}
}

func TestReopenedStoreReadsSavedRows(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	first, err := local.Open(local.Options{Root: root, EmbeddingModel: "test-model"})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	ensure := collection.EnsureRequest{Collection: testCollection, Declaration: testDeclaration(), Dimension: testDimension}
	if err := first.EnsureCollection(context.Background(), ensure); err != nil {
		t.Fatalf("EnsureCollection: %v", err)
	}
	if err := first.Upsert(context.Background(), testCollection, testDeclaration(), rankingRows()); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	want := searchIDs(t, first, collection.SearchRequest{Limit: 10})
	first.Close()

	second := openStore(t, root)
	if got := searchIDs(t, second, collection.SearchRequest{Limit: 10}); !reflect.DeepEqual(got, want) {
		t.Fatalf("reopened store ranked %v, want %v", got, want)
	}
	if err := second.EnsureCollection(context.Background(), ensure); err != nil {
		t.Fatalf("EnsureCollection on a reopened store: %v", err)
	}
	if got := len(queryAll(t, second)); got != len(want) {
		t.Fatalf("reopened store has %d rows after EnsureCollection, want %d", got, len(want))
	}
}

// An interrupted append leaves an index file that lists fewer rows than the row
// file. The test restores the index file from before the second write.
func TestReopenedStoreRebuildsStaleIndex(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	indexPath := filepath.Join(root, testCollection, "index.usearch")
	rows := rankingRows()
	first, err := local.Open(local.Options{Root: root, EmbeddingModel: "test-model"})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	ensure := collection.EnsureRequest{Collection: testCollection, Declaration: testDeclaration(), Dimension: testDimension}
	if err := first.EnsureCollection(context.Background(), ensure); err != nil {
		t.Fatalf("EnsureCollection: %v", err)
	}
	if err := first.Upsert(context.Background(), testCollection, testDeclaration(), rows[:2]); err != nil {
		t.Fatalf("first Upsert: %v", err)
	}
	staleIndex, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatalf("read index file: %v", err)
	}
	if err := first.Upsert(context.Background(), testCollection, testDeclaration(), rows[2:]); err != nil {
		t.Fatalf("second Upsert: %v", err)
	}
	want := searchIDs(t, first, collection.SearchRequest{Limit: 10})
	first.Close()
	if err := os.WriteFile(indexPath, staleIndex, 0o600); err != nil {
		t.Fatalf("restore stale index file: %v", err)
	}

	second := openStore(t, root)
	if got := searchIDs(t, second, collection.SearchRequest{Limit: 10}); !reflect.DeepEqual(got, want) {
		t.Fatalf("store with a stale index ranked %v, want %v", got, want)
	}
}

package library_test

import (
	"cmp"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/embedded"
)

// searchNamespace declares the scalar columns that the search tests filter
// and group by.
func searchNamespace(id string) library.NamespaceSpec {
	return library.NamespaceSpec{
		ID:     id,
		Policy: library.ReplaceAllowed,
		Scalars: []library.ScalarColumn{
			{Name: "workspace", Type: library.String, Nullable: true, Mutable: true, MaxLength: 64},
			{Name: "archived", Type: library.Bool, Nullable: false, Mutable: true, MaxLength: 0},
			{Name: "message_index", Type: library.Int64, Nullable: false, Mutable: false, MaxLength: 0},
			{Name: "conversation", Type: library.String, Nullable: false, Mutable: false, MaxLength: 64},
		},
	}
}

// searchTopics are the distinct texts of the search corpus. Several
// occurrences repeat each topic, so repeated content shares one vector.
var searchTopics = []string{
	"restart the daemon without a bind gap",
	"reload the proxy configuration file",
	"the embedding endpoint rejected an oversized input",
	"milvus flat cosine exact scoring over selected ids",
	"sqlite write ahead log checkpoint and busy timeout",
	"graphite stack restack after the parent branch moved",
	"conversation search returned a short page",
	"cursor pagination over a persisted result snapshot",
	"group quota limits hits per conversation",
	"reciprocal rank fusion with k equal to sixty",
	"bm25 document frequency counts every occurrence",
	"a literal prefix is never a sql wildcard",
	"null differs from an absent scalar value",
	"the kernel file lock serializes writers",
	"strong consistency reads verify the checksum",
	"temporary query database on disk with a page limit",
	"deadline exceeded returns no partial page",
	"reprojection changes mutable metadata only",
	"owner replacement removes previous occurrences",
	"append only namespaces reject deletion",
}

// workspaceValues are the workspace scalars the corpus cycles through. The
// empty string marks an absent column and "null" marks an explicit null.
var workspaceValues = []string{"/w/alpha", "/w/alpha/sub", "/w/50%_off", "/w/50xyoff", "null", "", "/w/beta"}

// searchCorpusRow is one occurrence of the search corpus with the owner that
// publishes it.
type searchCorpusRow struct {
	owner      string
	occurrence library.Occurrence
}

func searchCorpus(ownerSuffix string) []searchCorpusRow {
	owners := []string{"conv-a" + ownerSuffix, "conv-b" + ownerSuffix, "conv-c" + ownerSuffix}
	var rows []searchCorpusRow
	for index := range 60 {
		owner := owners[index%len(owners)]
		topic := searchTopics[(index*7)%len(searchTopics)]
		scalars := map[string]library.ScalarValue{
			"message_index": {Type: library.Int64, Null: false, String: "", Bool: false, Int64: int64(index)},
			"conversation":  {Type: library.String, Null: false, String: owner, Bool: false, Int64: 0},
		}
		switch workspace := workspaceValues[index%len(workspaceValues)]; workspace {
		case "":
		case "null":
			scalars["workspace"] = library.ScalarValue{Type: library.String, Null: true, String: "", Bool: false, Int64: 0}
		default:
			scalars["workspace"] = library.ScalarValue{Type: library.String, Null: false, String: workspace, Bool: false, Int64: 0}
		}
		if index%3 != 2 {
			scalars["archived"] = library.ScalarValue{Type: library.Bool, Null: false, String: "", Bool: index%4 == 0, Int64: 0}
		}
		rows = append(rows, searchCorpusRow{
			owner: owner,
			occurrence: library.Occurrence{
				RowKey: fmt.Sprintf("m%03d", index),
				// Repeated topics share one vector and one score. (index*3)%7
				// gives each repeat of a topic its own SortKey, in an order
				// that differs from the owner order.
				SortKey:        fmt.Sprintf("s%02d", (index*3)%7),
				SourceText:     fmt.Sprintf("%s (message %d)", topic, index%9),
				SearchText:     topic,
				EmbeddingInput: topic,
				Scalars:        scalars,
			},
		})
	}
	return rows
}

// searchFixture is an open library over the embedded pool with a published
// corpus, and the test's own record of every occurrence.
type searchFixture struct {
	ctx        context.Context
	library    *library.Library
	pool       *embedded.Store
	poolRoot   string
	descriptor library.StoreDescriptor
	embedder   library.Embedder
	records    map[library.OccurrenceID]*oracleRecord
}

// oracleRecord is the test's own copy of one published occurrence with its
// published and projected scalars.
type oracleRecord struct {
	id        library.OccurrenceID
	sortKey   string
	source    string
	published map[string]library.ScalarValue
	projected map[string]library.ScalarValue
}

func (record *oracleRecord) effective() map[string]library.ScalarValue {
	values := make(map[string]library.ScalarValue, len(record.published)+len(record.projected))
	for name, value := range record.published {
		values[name] = value
	}
	for name, value := range record.projected {
		values[name] = value
	}
	return values
}

func denseConfig(descriptor library.StoreDescriptor, pool *embedded.Store, embedder library.Embedder) library.Config {
	return library.Config{
		Store:            descriptor,
		Vectors:          pool,
		Embedder:         embedder,
		SearchMode:       library.Dense,
		QueryBlockSize:   7,
		QueryWorkers:     3,
		MaxBatchRows:     256,
		MaxBatchBytes:    0,
		MaxPageSize:      100,
		MaxQueryBytes:    1024,
		MaxFilterDepth:   8,
		MaxFilterValues:  32,
		AnalyzerIdentity: "",
	}
}

func newSearchFixture(t *testing.T, configure func(*library.Config)) *searchFixture {
	t.Helper()
	ctx := context.Background()
	root := t.TempDir()
	poolRoot := filepath.Join(root, "pool")
	fixture := &searchFixture{
		ctx:        ctx,
		library:    nil,
		pool:       embeddedPool(t, poolRoot),
		poolRoot:   poolRoot,
		descriptor: embeddedDescriptor(t, filepath.Join(root, "catalog")),
		embedder:   offlineEmbedder(t),
		records:    map[library.OccurrenceID]*oracleRecord{},
	}
	fixture.open(t, configure)
	for _, namespace := range []string{"chat", "other"} {
		if err := fixture.library.RegisterNamespace(ctx, searchNamespace(namespace)); err != nil {
			t.Fatalf("RegisterNamespace(%s): %v", namespace, err)
		}
	}
	fixture.publish(t, "chat", searchCorpus(""), 1)
	fixture.publish(t, "other", searchCorpus("-other"), 1)
	fixture.project(t, "conv-b", 1, map[string]map[string]library.ScalarValue{
		"m001": {"workspace": {Type: library.String, Null: false, String: "/w/alpha", Bool: false, Int64: 0}},
		"m004": {"workspace": {Type: library.String, Null: true, String: "", Bool: false, Int64: 0}},
		"m007": {"archived": {Type: library.Bool, Null: false, String: "", Bool: true, Int64: 0}},
	})
	return fixture
}

func (fixture *searchFixture) open(t *testing.T, configure func(*library.Config)) {
	t.Helper()
	config := denseConfig(fixture.descriptor, fixture.pool, fixture.embedder)
	if configure != nil {
		configure(&config)
	}
	opened, err := library.Open(fixture.ctx, config)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	fixture.library = opened
	t.Cleanup(func() {
		if err := opened.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
}

// publish replaces each owner's rows with its rows from corpus.
func (fixture *searchFixture) publish(t *testing.T, namespace string, corpus []searchCorpusRow, order uint64) {
	t.Helper()
	byOwner := map[string][]library.Occurrence{}
	for _, row := range corpus {
		byOwner[row.owner] = append(byOwner[row.owner], row.occurrence)
	}
	for _, owner := range slices.Sorted(maps.Keys(byOwner)) {
		fixture.replaceOwner(t, namespace, owner, byOwner[owner], order)
	}
}

func (fixture *searchFixture) replaceOwner(t *testing.T, namespace string, owner string, rows []library.Occurrence, order uint64) {
	t.Helper()
	_, err := fixture.library.Apply(fixture.ctx, library.Batch{
		Namespace:        namespace,
		OwnerID:          owner,
		GenerationOrder:  order,
		IdempotencyToken: fmt.Sprintf("%s-%d", owner, order),
		Mode:             library.Replace,
		Rows:             rows,
	})
	if err != nil {
		t.Fatalf("Apply %s/%s order %d: %v", namespace, owner, order, err)
	}
	for id := range fixture.records {
		if id.Namespace == namespace && id.OwnerID == owner {
			delete(fixture.records, id)
		}
	}
	for _, row := range rows {
		id := library.OccurrenceID{Namespace: namespace, OwnerID: owner, RowKey: row.RowKey}
		fixture.records[id] = &oracleRecord{
			id:        id,
			sortKey:   row.SortKey,
			source:    row.SourceText,
			published: row.Scalars,
			projected: map[string]library.ScalarValue{},
		}
	}
}

func (fixture *searchFixture) project(t *testing.T, owner string, order uint64, rows map[string]map[string]library.ScalarValue) {
	t.Helper()
	_, err := fixture.library.ReprojectScalars(fixture.ctx, library.ScalarProjection{
		Namespace:        "chat",
		OwnerID:          owner,
		ProjectionOrder:  order,
		IdempotencyToken: fmt.Sprintf("project-%s-%d", owner, order),
		Rows:             rows,
	})
	if err != nil {
		t.Fatalf("ReprojectScalars %s order %d: %v", owner, order, err)
	}
	for rowKey, values := range rows {
		record := fixture.records[library.OccurrenceID{Namespace: "chat", OwnerID: owner, RowKey: rowKey}]
		for name, value := range values {
			record.projected[name] = value
		}
	}
}

// catalogVectorIDs reads the vector ID of every published occurrence of the
// namespace from the catalog with a separate read-only SQLite connection.
func (fixture *searchFixture) catalogVectorIDs(t *testing.T, namespace string) map[library.OccurrenceID]string {
	t.Helper()
	database := openCatalogReadOnly(t, fixture.descriptor)
	rows, err := database.QueryContext(fixture.ctx, `SELECT owner_id, row_key, vector_id FROM occurrences WHERE namespace = ?`, namespace)
	if err != nil {
		t.Fatalf("read occurrence vectors: %v", err)
	}
	defer func() { _ = rows.Close() }()
	vectorIDs := map[library.OccurrenceID]string{}
	for rows.Next() {
		var owner, rowKey, vectorID string
		if err := rows.Scan(&owner, &rowKey, &vectorID); err != nil {
			t.Fatalf("scan occurrence vector: %v", err)
		}
		vectorIDs[library.OccurrenceID{Namespace: namespace, OwnerID: owner, RowKey: rowKey}] = vectorID
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("read occurrence vectors: %v", err)
	}
	return vectorIDs
}

func openCatalogReadOnly(t *testing.T, descriptor library.StoreDescriptor) *sql.DB {
	t.Helper()
	database, err := sql.Open("sqlite3", "file:"+descriptor.CatalogPath+"?mode=ro")
	if err != nil {
		t.Fatalf("open catalog read-only: %v", err)
	}
	t.Cleanup(func() { _ = database.Close() })
	return database
}

// oracle ranks the fixture's own records for request. It takes each vector's
// exact score from the vector store and applies the filter, the MinScore
// floor, the group quota, and the final order without the library's search
// code.
func (fixture *searchFixture) oracle(t *testing.T, request library.SearchRequest) []library.SearchHit {
	t.Helper()
	vectorIDs := fixture.catalogVectorIDs(t, request.Namespace)
	queryVectors, err := fixture.embedder.EmbedBatch(fixture.ctx, []string{request.Query})
	if err != nil {
		t.Fatalf("oracle query embedding: %v", err)
	}
	distinct := map[string]bool{}
	for _, vectorID := range vectorIDs {
		distinct[vectorID] = true
	}
	ids := slices.Sorted(maps.Keys(distinct))
	scores, err := fixture.pool.ScoreExact(fixture.ctx, queryVectors[0], ids)
	if err != nil {
		t.Fatalf("oracle exact scores: %v", err)
	}
	scoreByVector := map[string]float64{}
	for _, score := range scores {
		scoreByVector[score.ID] = score.Score
	}
	var hits []oracleHit
	for id, record := range fixture.records {
		if id.Namespace != request.Namespace {
			continue
		}
		effective := record.effective()
		if request.Filter != nil && evaluateFilter(*request.Filter, effective) != truthTrue {
			continue
		}
		score := scoreByVector[vectorIDs[id]]
		if request.MinScore > 0 && score < request.MinScore {
			continue
		}
		hits = append(hits, oracleHit{record: record, score: score, group: groupKey(effective, request.GroupBy)})
	}
	slices.SortFunc(hits, compareOracleHits)
	return applyGroupQuota(hits, request)
}

type oracleHit struct {
	record *oracleRecord
	score  float64
	group  string
}

func compareOracleHits(left oracleHit, right oracleHit) int {
	if left.score != right.score {
		return cmp.Compare(right.score, left.score)
	}
	return cmp.Or(
		strings.Compare(left.record.sortKey, right.record.sortKey),
		strings.Compare(left.record.id.OwnerID, right.record.id.OwnerID),
		strings.Compare(left.record.id.RowKey, right.record.id.RowKey),
	)
}

func applyGroupQuota(hits []oracleHit, request library.SearchRequest) []library.SearchHit {
	perGroup := map[string]int{}
	var result []library.SearchHit
	for _, hit := range hits {
		if request.GroupBy != "" {
			if perGroup[hit.group] == request.PerGroupLimit {
				continue
			}
			perGroup[hit.group]++
		}
		result = append(result, library.SearchHit{
			ID:         hit.record.id,
			SourceText: hit.record.source,
			Scalars:    hit.record.effective(),
			Score:      hit.score,
		})
	}
	return result
}

func groupKey(values map[string]library.ScalarValue, column string) string {
	if column == "" {
		return ""
	}
	value, present := values[column]
	switch {
	case !present:
		return "absent"
	case value.Null:
		return "null"
	default:
		return fmt.Sprintf("value:%v", scalarComparable(value))
	}
}

// truth is a three-valued filter result.
type truth int8

const (
	truthFalse truth = iota
	truthUnknown
	truthTrue
)

// evaluateFilter evaluates a filter with SQL three-valued logic over one
// occurrence's effective scalars. A comparison with a null or absent value is
// unknown. IsNull and IsPresent are never unknown.
func evaluateFilter(filter library.Filter, values map[string]library.ScalarValue) truth {
	switch filter.Op {
	case library.All:
		result := truthTrue
		for _, child := range filter.Children {
			result = min(result, evaluateFilter(child, values))
		}
		return result
	case library.Any:
		result := truthFalse
		for _, child := range filter.Children {
			result = max(result, evaluateFilter(child, values))
		}
		return result
	case library.Not:
		return truthTrue - evaluateFilter(filter.Children[0], values)
	case library.IsNull:
		value, present := values[filter.Column]
		return truthOf(present && value.Null)
	case library.IsPresent:
		_, present := values[filter.Column]
		return truthOf(present)
	}
	value, present := values[filter.Column]
	if !present || value.Null {
		return truthUnknown
	}
	switch filter.Op {
	case library.Equal, library.In:
		for _, candidate := range filter.Values {
			if scalarComparable(candidate) == scalarComparable(value) {
				return truthTrue
			}
		}
		return truthFalse
	case library.Range:
		inside := (filter.Lower == nil || value.Int64 >= filter.Lower.Int64) && (filter.Upper == nil || value.Int64 < filter.Upper.Int64)
		return truthOf(inside)
	default:
		return truthOf(strings.HasPrefix(value.String, filter.Prefix))
	}
}

func truthOf(condition bool) truth {
	if condition {
		return truthTrue
	}
	return truthFalse
}

func scalarComparable(value library.ScalarValue) any {
	switch value.Type {
	case library.Bool:
		return value.Bool
	case library.Int64:
		return value.Int64
	default:
		return value.String
	}
}

// pageAll pages request to exhaustion at pageSize. It fails on a short page
// before the end, an empty page, or a cursor on the last page.
func pageAll(t *testing.T, searcher *library.Library, request library.SearchRequest, pageSize int) []library.SearchHit {
	t.Helper()
	request.PageSize = pageSize
	request.Cursor = ""
	var hits []library.SearchHit
	for pages := 0; ; pages++ {
		page, err := searcher.Search(context.Background(), request)
		if err != nil {
			t.Fatalf("Search page %d at page size %d: %v", pages, pageSize, err)
		}
		if page.HasMore && len(page.Hits) != pageSize {
			t.Fatalf("page %d at page size %d has %d hits and HasMore", pages, pageSize, len(page.Hits))
		}
		if pages > 0 && len(page.Hits) == 0 {
			t.Fatalf("page %d at page size %d is empty after a cursor", pages, pageSize)
		}
		if !page.HasMore && page.NextCursor != "" {
			t.Fatalf("page %d at page size %d has a cursor without HasMore", pages, pageSize)
		}
		hits = append(hits, page.Hits...)
		if !page.HasMore {
			return hits
		}
		request.Cursor = page.NextCursor
	}
}

func assertHitsEqual(t *testing.T, label string, got []library.SearchHit, want []library.SearchHit) {
	t.Helper()
	seen := map[library.OccurrenceID]int{}
	for index, hit := range got {
		if first, repeated := seen[hit.ID]; repeated {
			t.Fatalf("%s: hit %d repeats %+v from position %d", label, index, hit.ID, first)
		}
		seen[hit.ID] = index
	}
	if len(got) != len(want) {
		t.Fatalf("%s: %d hits, oracle has %d", label, len(got), len(want))
	}
	for index := range want {
		if !reflect.DeepEqual(normalizeHit(got[index]), normalizeHit(want[index])) {
			t.Fatalf("%s: hit %d = %+v, oracle %+v", label, index, got[index], want[index])
		}
	}
}

func normalizeHit(hit library.SearchHit) library.SearchHit {
	if hit.Scalars == nil {
		hit.Scalars = map[string]library.ScalarValue{}
	}
	return hit
}

func stringValue(text string) library.ScalarValue {
	return library.ScalarValue{Type: library.String, Null: false, String: text, Bool: false, Int64: 0}
}

func int64Bound(value int64) *library.ScalarValue {
	return &library.ScalarValue{Type: library.Int64, Null: false, String: "", Bool: false, Int64: value}
}

// searchRequests covers every filter operator, null and absent values, a
// literal prefix with SQL wildcard characters, a projected value, grouping,
// and a namespace boundary.
func searchRequests() map[string]library.SearchRequest {
	base := library.SearchRequest{Namespace: "chat", Query: "restart the daemon without a bind gap"}
	requests := map[string]library.SearchRequest{"no filter": base}
	add := func(name string, filter library.Filter) {
		request := base
		request.Filter = &filter
		requests[name] = request
	}
	add("nested workspace and archived", library.Filter{Op: library.All, Children: []library.Filter{
		{Op: library.Any, Children: []library.Filter{
			{Op: library.Equal, Column: "workspace", Values: []library.ScalarValue{stringValue("/w/alpha")}},
			{Op: library.Prefix, Column: "workspace", Prefix: "/w/alpha/"},
		}},
		{Op: library.Not, Children: []library.Filter{
			{Op: library.Equal, Column: "archived", Values: []library.ScalarValue{{Type: library.Bool, Null: false, String: "", Bool: true, Int64: 0}}},
		}},
	}})
	add("literal prefix", library.Filter{Op: library.Prefix, Column: "workspace", Prefix: "/w/50%_"})
	add("not equal skips null and absent", library.Filter{Op: library.Not, Children: []library.Filter{
		{Op: library.Equal, Column: "workspace", Values: []library.ScalarValue{stringValue("/w/alpha")}},
	}})
	add("is null", library.Filter{Op: library.IsNull, Column: "workspace"})
	add("not is null keeps absent", library.Filter{Op: library.Not, Children: []library.Filter{{Op: library.IsNull, Column: "workspace"}}})
	add("is present", library.Filter{Op: library.IsPresent, Column: "archived"})
	add("range", library.Filter{Op: library.Range, Column: "message_index", Lower: int64Bound(10), Upper: int64Bound(40)})
	add("upper range or conversation", library.Filter{Op: library.Any, Children: []library.Filter{
		{Op: library.Range, Column: "message_index", Upper: int64Bound(5)},
		{Op: library.In, Column: "conversation", Values: []library.ScalarValue{stringValue("conv-c"), stringValue("conv-z")}},
	}})
	add("not any range or prefix", library.Filter{Op: library.Not, Children: []library.Filter{
		{Op: library.Any, Children: []library.Filter{
			{Op: library.Range, Column: "message_index", Lower: int64Bound(20), Upper: int64Bound(45)},
			{Op: library.Prefix, Column: "workspace", Prefix: "/w/alpha"},
		}},
	}})
	add("not all in and equal", library.Filter{Op: library.Not, Children: []library.Filter{
		{Op: library.All, Children: []library.Filter{
			{Op: library.In, Column: "conversation", Values: []library.ScalarValue{stringValue("conv-a"), stringValue("conv-b")}},
			{Op: library.Equal, Column: "archived", Values: []library.ScalarValue{{Type: library.Bool, Null: false, String: "", Bool: false, Int64: 0}}},
		}},
	}})
	grouped := base
	grouped.GroupBy = "workspace"
	grouped.PerGroupLimit = 2
	requests["group by workspace"] = grouped
	other := base
	other.Namespace = "other"
	requests["other namespace"] = other
	return requests
}

func TestSearchPagesMatchTheExhaustiveOracle(t *testing.T) {
	fixture := newSearchFixture(t, nil)
	requests := searchRequests()
	for _, name := range slices.Sorted(maps.Keys(requests)) {
		request := requests[name]
		want := fixture.oracle(t, request)
		if len(want) == 0 && name != "is null" {
			t.Fatalf("%s: the oracle selects no occurrence; the corpus does not exercise the request", name)
		}
		for _, pageSize := range []int{1, 10, 100} {
			got := pageAll(t, fixture.library, request, pageSize)
			assertHitsEqual(t, fmt.Sprintf("%s at page size %d", name, pageSize), got, want)
		}
	}
}

func TestSearchAppliesMinScoreBeforePaging(t *testing.T) {
	fixture := newSearchFixture(t, nil)
	request := library.SearchRequest{Namespace: "chat", Query: "cursor pagination over a persisted result snapshot"}
	all := fixture.oracle(t, request)
	floor := 0.0
	for index := 20; index+1 < len(all); index++ {
		if all[index].Score > all[index+1].Score {
			floor = (all[index].Score + all[index+1].Score) / 2
			break
		}
	}
	if floor == 0 {
		t.Fatal("the corpus has no score gap after position 20")
	}
	request.MinScore = floor
	want := fixture.oracle(t, request)
	if len(want) == 0 || len(want) == len(all) {
		t.Fatalf("MinScore %v keeps %d of %d hits; the floor does not split the result", floor, len(want), len(all))
	}
	for _, pageSize := range []int{1, 10, 100} {
		assertHitsEqual(t, fmt.Sprintf("MinScore at page size %d", pageSize), pageAll(t, fixture.library, request, pageSize), want)
	}
}

func TestSearchPageSequenceIgnoresWritesAfterPageOne(t *testing.T) {
	fixture := newSearchFixture(t, nil)
	request := library.SearchRequest{Namespace: "chat", Query: "owner replacement removes previous occurrences", PageSize: 10}
	want := fixture.oracle(t, request)
	first, err := fixture.library.Search(fixture.ctx, request)
	if err != nil || !first.HasMore {
		t.Fatalf("page one = %d hits, HasMore %v, %v; want a cursor", len(first.Hits), first.HasMore, err)
	}
	pinned := fixture.snapshotPins(t)
	if len(pinned) != len(want) {
		t.Fatalf("the snapshot pins %d results, the oracle ranks %d", len(pinned), len(want))
	}
	vectorIDs := fixture.catalogVectorIDs(t, "chat")
	for _, pin := range pinned {
		if pin.vectorID != vectorIDs[pin.id] || pin.sourceBlobID == "" {
			t.Fatalf("snapshot row %+v does not pin vector %s", pin, vectorIDs[pin.id])
		}
	}

	replacement := []library.Occurrence{{
		RowKey: "new", SortKey: "s00", SourceText: "replacement text", SearchText: "replacement text",
		EmbeddingInput: "owner replacement removes previous occurrences", Scalars: nil,
	}}
	fixture.replaceOwner(t, "chat", "conv-a", replacement, 2)
	fixture.project(t, "conv-b", 2, map[string]map[string]library.ScalarValue{
		"m010": {"workspace": stringValue("/w/changed")},
	})
	_, err = fixture.library.Apply(fixture.ctx, library.Batch{
		Namespace: "chat", OwnerID: "conv-c", GenerationOrder: 2, IdempotencyToken: "conv-c-append", Mode: library.Append,
		Rows: []library.Occurrence{{
			RowKey: "appended", SortKey: "s00", SourceText: "appended", SearchText: "appended",
			EmbeddingInput: "owner replacement removes previous occurrences", Scalars: nil,
		}},
	})
	if err != nil {
		t.Fatalf("Append between pages: %v", err)
	}
	fixture.records[library.OccurrenceID{Namespace: "chat", OwnerID: "conv-c", RowKey: "appended"}] = &oracleRecord{
		id: library.OccurrenceID{Namespace: "chat", OwnerID: "conv-c", RowKey: "appended"}, sortKey: "s00", source: "appended",
		published: nil, projected: map[string]library.ScalarValue{},
	}

	got := slices.Clone(first.Hits)
	request.Cursor = first.NextCursor
	for request.Cursor != "" {
		page, err := fixture.library.Search(fixture.ctx, request)
		if err != nil {
			t.Fatalf("cursor page after writes: %v", err)
		}
		got = append(got, page.Hits...)
		request.Cursor = page.NextCursor
	}
	assertHitsEqual(t, "pages after writes", got, want)

	fresh := library.SearchRequest{Namespace: "chat", Query: request.Query}
	assertHitsEqual(t, "fresh search after writes", pageAll(t, fixture.library, fresh, 100), fixture.oracle(t, fresh))
}

// snapshotPin is one search_results row with the references it pins.
type snapshotPin struct {
	id           library.OccurrenceID
	vectorID     string
	sourceBlobID string
}

func (fixture *searchFixture) snapshotPins(t *testing.T) []snapshotPin {
	t.Helper()
	database := openCatalogReadOnly(t, fixture.descriptor)
	rows, err := database.QueryContext(fixture.ctx,
		`SELECT r.namespace, r.owner_id, r.row_key, r.vector_id, r.source_blob_id FROM search_results r
		JOIN search_snapshots s ON s.snapshot_id = r.snapshot_id ORDER BY r.snapshot_id, r.ordinal`,
	)
	if err != nil {
		t.Fatalf("read snapshot pins: %v", err)
	}
	defer func() { _ = rows.Close() }()
	var pins []snapshotPin
	for rows.Next() {
		var pin snapshotPin
		if err := rows.Scan(&pin.id.Namespace, &pin.id.OwnerID, &pin.id.RowKey, &pin.vectorID, &pin.sourceBlobID); err != nil {
			t.Fatalf("scan snapshot pin: %v", err)
		}
		pins = append(pins, pin)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("read snapshot pins: %v", err)
	}
	return pins
}

func TestSearchWritesNoSnapshotForOneCompletePage(t *testing.T) {
	fixture := newSearchFixture(t, nil)
	request := library.SearchRequest{Namespace: "chat", Query: "group quota limits hits per conversation", PageSize: 100}
	page, err := fixture.library.Search(fixture.ctx, request)
	if err != nil || page.HasMore || page.NextCursor != "" {
		t.Fatalf("complete page = HasMore %v cursor %q, %v", page.HasMore, page.NextCursor, err)
	}
	assertHitsEqual(t, "one complete page", page.Hits, fixture.oracle(t, request))
	if pins := fixture.snapshotPins(t); len(pins) != 0 {
		t.Fatalf("a complete first page persisted %d snapshot rows", len(pins))
	}
}

func TestSearchRejectsForeignAndExpiredCursors(t *testing.T) {
	fixture := newSearchFixture(t, func(config *library.Config) { config.SnapshotTTL = 300 * time.Millisecond })
	request := library.SearchRequest{Namespace: "chat", Query: "deadline exceeded returns no partial page", PageSize: 5}
	first, err := fixture.library.Search(fixture.ctx, request)
	if err != nil || first.NextCursor == "" {
		t.Fatalf("page one = %v, cursor %q", err, first.NextCursor)
	}

	changed := request
	changed.Cursor = first.NextCursor
	changed.Query = "another query"
	if _, err := fixture.library.Search(fixture.ctx, changed); !errors.Is(err, library.ErrCursorMismatch) {
		t.Fatalf("cursor with another query = %v, want ErrCursorMismatch", err)
	}
	changed = request
	changed.Cursor = first.NextCursor
	changed.Filter = &library.Filter{Op: library.IsPresent, Column: "archived"}
	if _, err := fixture.library.Search(fixture.ctx, changed); !errors.Is(err, library.ErrCursorMismatch) {
		t.Fatalf("cursor with another filter = %v, want ErrCursorMismatch", err)
	}
	changed = request
	changed.Cursor = "not-a-cursor"
	if _, err := fixture.library.Search(fixture.ctx, changed); !errors.Is(err, library.ErrInvalidRequest) {
		t.Fatalf("undecodable cursor = %v, want ErrInvalidRequest", err)
	}

	time.Sleep(400 * time.Millisecond)
	expired := request
	expired.Cursor = first.NextCursor
	page, err := fixture.library.Search(fixture.ctx, expired)
	if !errors.Is(err, library.ErrCursorExpired) || len(page.Hits) != 0 {
		t.Fatalf("expired cursor = %d hits, %v; want no page and ErrCursorExpired", len(page.Hits), err)
	}
	second, err := fixture.library.Search(fixture.ctx, request)
	if err != nil || second.NextCursor == "" {
		t.Fatalf("second page one = %v, cursor %q", err, second.NextCursor)
	}
	oracleCount := len(fixture.oracle(t, request))
	if pins := fixture.snapshotPins(t); len(pins) != oracleCount {
		t.Fatalf("after cleanup the catalog has %d snapshot rows, want only the %d of the new snapshot", len(pins), oracleCount)
	}
	if rows := fixture.countRows(t, "search_results"); rows != oracleCount {
		t.Fatalf("after cleanup search_results has %d rows, want only the %d of the new snapshot", rows, oracleCount)
	}
	if rows := fixture.countRows(t, "search_snapshots"); rows != 1 {
		t.Fatalf("after cleanup search_snapshots has %d rows, want only the new snapshot", rows)
	}
}

// countRows counts every row of a catalog table, including rows that no
// other table references.
func (fixture *searchFixture) countRows(t *testing.T, table string) int {
	t.Helper()
	queries := map[string]string{
		"search_results":   `SELECT COUNT(*) FROM search_results`,
		"search_snapshots": `SELECT COUNT(*) FROM search_snapshots`,
	}
	var count int
	if err := openCatalogReadOnly(t, fixture.descriptor).QueryRowContext(fixture.ctx, queries[table]).Scan(&count); err != nil {
		t.Fatalf("count %s rows: %v", table, err)
	}
	return count
}

func TestSearchCursorRejectsAnotherRankConfiguration(t *testing.T) {
	fixture := newSearchFixture(t, nil)
	request := library.SearchRequest{Namespace: "chat", Query: "reload the proxy configuration file", PageSize: 5}
	first, err := fixture.library.Search(fixture.ctx, request)
	if err != nil || first.NextCursor == "" {
		t.Fatalf("page one = %v, cursor %q", err, first.NextCursor)
	}
	if err := fixture.library.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	fixture.open(t, func(config *library.Config) { config.QueryInstructionPrefix = "query: " })
	request.Cursor = first.NextCursor
	if _, err := fixture.library.Search(fixture.ctx, request); !errors.Is(err, library.ErrCursorMismatch) {
		t.Fatalf("cursor after a rank configuration change = %v, want ErrCursorMismatch", err)
	}
}

// TestSearchReturnsResourceLimitWhenTheQueryDatabaseIsFull raises
// MaxTemporaryBytes one query database page at a time. Every search below the
// size that the query needs fails with ErrResourceLimit and no page, and at
// least one of them fails after the query database schema exists.
func TestSearchReturnsResourceLimitWhenTheQueryDatabaseIsFull(t *testing.T) {
	fixture := newSearchFixture(t, nil)
	request := library.SearchRequest{Namespace: "chat", Query: "strong consistency reads verify the checksum", PageSize: 1}
	const pageBytes = 4096
	const maxPages = 1024
	dataFailures := 0
	for pages := 1; pages <= maxPages; pages++ {
		config := denseConfig(fixture.descriptor, fixture.pool, fixture.embedder)
		config.MaxTemporaryBytes = int64(pages * pageBytes)
		limited, err := library.Open(fixture.ctx, config)
		if err != nil {
			t.Fatalf("Open with %d query database pages: %v", pages, err)
		}
		page, err := limited.Search(fixture.ctx, request)
		if closeErr := limited.Close(); closeErr != nil {
			t.Fatalf("Close: %v", closeErr)
		}
		if err == nil {
			if dataFailures == 0 {
				t.Fatalf("Search succeeded at %d pages with no earlier failure after the schema", pages)
			}
			t.Logf("Search failed after the schema at %d budgets and succeeded at %d pages", dataFailures, pages)
			return
		}
		if !errors.Is(err, library.ErrResourceLimit) || len(page.Hits) != 0 || page.NextCursor != "" {
			t.Fatalf("Search at %d pages = %d hits, %v; want no page and ErrResourceLimit", pages, len(page.Hits), err)
		}
		if !strings.Contains(err.Error(), "create query database schema") {
			dataFailures++
		}
	}
	t.Fatalf("Search did not succeed within %d query database pages", maxPages)
}

func TestSearchFailsWithoutAPageOnBudgetDeadlineAndVectorLoss(t *testing.T) {
	request := library.SearchRequest{Namespace: "chat", Query: "strong consistency reads verify the checksum", PageSize: 1}
	cases := []struct {
		name      string
		configure func(*library.Config)
		damage    func(*testing.T, *searchFixture)
		want      error
	}{
		{name: "snapshot disk", configure: func(config *library.Config) { config.MaxSnapshotBytes = 512 }, damage: nil, want: library.ErrResourceLimit},
		{name: "deadline", configure: func(config *library.Config) { config.QueryTimeout = time.Nanosecond }, damage: nil, want: library.ErrDeadline},
		{name: "missing vector", configure: nil, damage: removeOneVector, want: library.ErrVectorMissing},
		{name: "corrupt vector", configure: nil, damage: corruptOneVector, want: library.ErrVectorCorrupt},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			fixture := newSearchFixture(t, testCase.configure)
			if testCase.damage != nil {
				testCase.damage(t, fixture)
			}
			page, err := fixture.library.Search(fixture.ctx, request)
			if !errors.Is(err, testCase.want) {
				t.Fatalf("Search = %v, want %v", err, testCase.want)
			}
			if len(page.Hits) != 0 || page.HasMore || page.NextCursor != "" {
				t.Fatalf("failed Search returned a page: %+v", page)
			}
			if pins := fixture.snapshotPins(t); len(pins) != 0 {
				t.Fatalf("failed Search persisted %d snapshot rows", len(pins))
			}
			entries, err := os.ReadDir(filepath.Dir(fixture.descriptor.CatalogPath))
			if err != nil {
				t.Fatalf("read catalog directory: %v", err)
			}
			for _, entry := range entries {
				if strings.HasPrefix(entry.Name(), ".lms-query-") {
					t.Fatalf("failed Search left query database %s", entry.Name())
				}
			}
		})
	}
}

// TestSearchVerifiesVectorIdentityBeforeScoring changes one eligible vector's
// checksum in the catalog and leaves the pool unchanged. VerifyStrong compares
// the pool with the checksum that the snapshot copies from the catalog;
// ScoreExact reads only the pool.
func TestSearchVerifiesVectorIdentityBeforeScoring(t *testing.T) {
	fixture := newSearchFixture(t, nil)
	vectorID := fixture.catalogVectorIDs(t, "chat")[library.OccurrenceID{Namespace: "chat", OwnerID: "conv-a", RowKey: "m000"}]
	database, err := sql.Open("sqlite3", "file:"+fixture.descriptor.CatalogPath)
	if err != nil {
		t.Fatalf("open catalog: %v", err)
	}
	t.Cleanup(func() { _ = database.Close() })
	if _, err := database.ExecContext(fixture.ctx,
		`UPDATE vectors SET vector_checksum = ? WHERE vector_id = ?`, strings.Repeat("0", 64), vectorID); err != nil {
		t.Fatalf("change catalog checksum: %v", err)
	}
	page, err := fixture.library.Search(fixture.ctx, library.SearchRequest{Namespace: "chat", Query: "alpha", PageSize: 5})
	if !errors.Is(err, library.ErrVectorCorrupt) || len(page.Hits) != 0 {
		t.Fatalf("Search with a changed catalog checksum = %d hits, %v; want no page and ErrVectorCorrupt", len(page.Hits), err)
	}
}

// TestSearchRemovesStaleQueryDatabases leaves one query database file older
// than twice the QueryTimeout and one recent file in the catalog directory.
// Page one deletes the old file and keeps the recent one.
func TestSearchRemovesStaleQueryDatabases(t *testing.T) {
	fixture := newSearchFixture(t, func(config *library.Config) { config.QueryTimeout = time.Minute })
	directory := filepath.Dir(fixture.descriptor.CatalogPath)
	stale := filepath.Join(directory, ".lms-query-stale.sqlite")
	recent := filepath.Join(directory, ".lms-query-recent.sqlite")
	for _, path := range []string{stale, recent} {
		if err := os.WriteFile(path, []byte("left by a stopped search"), 0o600); err != nil {
			t.Fatalf("write %s: %v", path, err)
		}
	}
	old := time.Now().Add(-3 * time.Minute)
	if err := os.Chtimes(stale, old, old); err != nil {
		t.Fatalf("age %s: %v", stale, err)
	}
	if _, err := fixture.library.Search(fixture.ctx, library.SearchRequest{Namespace: "chat", Query: "alpha", PageSize: 5}); err != nil {
		t.Fatalf("Search: %v", err)
	}
	if _, err := os.Stat(stale); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("stale query database after Search: %v, want it removed", err)
	}
	if _, err := os.Stat(recent); err != nil {
		t.Fatalf("recent query database after Search: %v, want it kept", err)
	}
}

// vectorFile returns the embedded pool file of one eligible vector.
func (fixture *searchFixture) vectorFile(t *testing.T) string {
	t.Helper()
	vectorIDs := fixture.catalogVectorIDs(t, "chat")
	id := vectorIDs[library.OccurrenceID{Namespace: "chat", OwnerID: "conv-a", RowKey: "m000"}]
	sum := sha256.Sum256([]byte(id))
	return filepath.Join(fixture.poolRoot, "vectors", hex.EncodeToString(sum[:1]), hex.EncodeToString(sum[1:2]), id+".vec")
}

func removeOneVector(t *testing.T, fixture *searchFixture) {
	t.Helper()
	if err := os.Remove(fixture.vectorFile(t)); err != nil {
		t.Fatalf("remove vector file: %v", err)
	}
}

func corruptOneVector(t *testing.T, fixture *searchFixture) {
	t.Helper()
	path := fixture.vectorFile(t)
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read vector file: %v", err)
	}
	content[len(content)-1] ^= 0xff
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatalf("write vector file: %v", err)
	}
}

// hybridRRFK is the default reciprocal rank fusion constant.
const hybridRRFK = 60

// hybridOracle ranks the fixture's own records for request with reciprocal
// rank fusion. Dense ranks come from exact store scores. lexicalOrder lists
// the occurrences with a lexical match in lexical rank order.
func (fixture *searchFixture) hybridOracle(
	t *testing.T,
	request library.SearchRequest,
	lexicalOrder []library.OccurrenceID,
) []library.SearchHit {
	t.Helper()
	dense := fixture.oracle(t, library.SearchRequest{Namespace: request.Namespace, Query: request.Query, Filter: request.Filter})
	slices.SortFunc(dense, func(left library.SearchHit, right library.SearchHit) int {
		if left.Score != right.Score {
			return cmp.Compare(right.Score, left.Score)
		}
		return cmp.Or(strings.Compare(left.ID.OwnerID, right.ID.OwnerID), strings.Compare(left.ID.RowKey, right.ID.RowKey))
	})
	eligible := map[library.OccurrenceID]bool{}
	for _, hit := range dense {
		eligible[hit.ID] = true
	}
	lexicalRank := map[library.OccurrenceID]int{}
	for _, id := range lexicalOrder {
		if eligible[id] {
			lexicalRank[id] = len(lexicalRank) + 1
		}
	}
	fused := make([]oracleHit, 0, len(dense))
	for index, hit := range dense {
		score := 1.0 / float64(hybridRRFK+index+1)
		if rank, matched := lexicalRank[hit.ID]; matched {
			score += 1.0 / float64(hybridRRFK+rank)
		}
		if request.MinScore > 0 && score < request.MinScore {
			continue
		}
		record := fixture.records[hit.ID]
		fused = append(fused, oracleHit{record: record, score: score, group: groupKey(record.effective(), request.GroupBy)})
	}
	slices.SortFunc(fused, compareOracleHits)
	return applyGroupQuota(fused, request)
}

// lexicalWorkspaces are the workspace scalars of the five lexical rows: a
// value, absent, another value, the first value again, and null.
var lexicalWorkspaces = []string{"/w/alpha", "", "/w/beta", "/w/alpha", "null"}

func TestSearchHybridFusesDenseAndLexicalRanks(t *testing.T) {
	fixture := newSearchFixture(t, func(config *library.Config) { config.SearchMode = library.Hybrid })
	var rows []library.Occurrence
	var short, long []library.OccurrenceID
	for index := range 5 {
		text := "zzqx"
		if index >= 3 {
			text = "zzqx alpha beta gamma delta epsilon"
		}
		rowKey := fmt.Sprintf("lex%d", index)
		scalars := map[string]library.ScalarValue{}
		switch workspace := lexicalWorkspaces[index]; workspace {
		case "":
		case "null":
			scalars["workspace"] = library.ScalarValue{Type: library.String, Null: true, String: "", Bool: false, Int64: 0}
		default:
			scalars["workspace"] = stringValue(workspace)
		}
		rows = append(rows, library.Occurrence{
			RowKey: rowKey, SortKey: fmt.Sprintf("s%02d", index), SourceText: text, SearchText: text,
			EmbeddingInput: searchTopics[index], Scalars: scalars,
		})
		id := library.OccurrenceID{Namespace: "chat", OwnerID: "conv-lex", RowKey: rowKey}
		if index < 3 {
			short = append(short, id)
		} else {
			long = append(long, id)
		}
	}
	fixture.replaceOwner(t, "chat", "conv-lex", rows, 1)

	// Equal short texts tie on BM25 and rank by occurrence ID. The long texts
	// have the same term frequency and a longer document, so they score lower.
	matched := library.SearchRequest{Namespace: "chat", Query: "zzqx"}
	want := fixture.hybridOracle(t, matched, append(slices.Clone(short), long...))
	for _, pageSize := range []int{1, 10, 100} {
		assertHitsEqual(t, fmt.Sprintf("hybrid zzqx at page size %d", pageSize), pageAll(t, fixture.library, matched, pageSize), want)
	}

	// Both ranks count only the occurrences that the filter selects. lex1 has
	// no workspace and lex2 has another workspace, so neither is eligible.
	filtered := matched
	filtered.Filter = &library.Filter{Op: library.Any, Children: []library.Filter{
		{Op: library.Prefix, Column: "workspace", Prefix: "/w/alpha"},
		{Op: library.IsNull, Column: "workspace"},
	}}
	want = fixture.hybridOracle(t, filtered, append(slices.Clone(short), long...))
	assertHitsEqual(t, "hybrid filtered", pageAll(t, fixture.library, filtered, 10), want)

	grouped := matched
	grouped.GroupBy = "workspace"
	grouped.PerGroupLimit = 1
	want = fixture.hybridOracle(t, grouped, append(slices.Clone(short), long...))
	assertHitsEqual(t, "hybrid grouped", pageAll(t, fixture.library, grouped, 10), want)

	all := fixture.hybridOracle(t, matched, append(slices.Clone(short), long...))
	floored := matched
	for index := 3; index+1 < len(all); index++ {
		if all[index].Score > all[index+1].Score {
			floored.MinScore = (all[index].Score + all[index+1].Score) / 2
			break
		}
	}
	want = fixture.hybridOracle(t, floored, append(slices.Clone(short), long...))
	if len(want) == 0 || len(want) == len(all) {
		t.Fatalf("hybrid MinScore %v keeps %d of %d hits; the floor does not split the result", floored.MinScore, len(want), len(all))
	}
	assertHitsEqual(t, "hybrid MinScore", pageAll(t, fixture.library, floored, 10), want)

	// A query with no analyzed terms ranks by the dense term alone.
	unanalyzed := library.SearchRequest{Namespace: "chat", Query: "!!! ???"}
	want = fixture.hybridOracle(t, unanalyzed, nil)
	assertHitsEqual(t, "hybrid without analyzed terms", pageAll(t, fixture.library, unanalyzed, 10), want)
}

func TestSearchRejectsInvalidRequestsBeforeScoring(t *testing.T) {
	fixture := newSearchFixture(t, nil)
	for name, request := range map[string]library.SearchRequest{
		"unregistered namespace": {Namespace: "missing", Query: "alpha", PageSize: 1},
		"NUL in query":           {Namespace: "chat", Query: "alpha\x00beta", PageSize: 1},
		"undeclared filter":      {Namespace: "chat", Query: "alpha", PageSize: 1, Filter: &library.Filter{Op: library.IsNull, Column: "provider"}},
		"page over limit":        {Namespace: "chat", Query: "alpha", PageSize: 101},
	} {
		if _, err := fixture.library.Search(fixture.ctx, request); !errors.Is(err, library.ErrInvalidRequest) {
			t.Fatalf("%s: Search = %v, want ErrInvalidRequest", name, err)
		}
	}
}

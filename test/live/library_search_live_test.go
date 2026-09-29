//go:build live

package live

import (
	"cmp"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/library"
)

const (
	// searchLiveDistinctInputs is the number of distinct embedding inputs in
	// the completeness corpus. It exceeds the 16,384 fixed ranking depth of the
	// current collection search.
	searchLiveDistinctInputs = 20480
	// searchLiveOwners is the number of owners of the completeness corpus.
	searchLiveOwners = 160
	// searchLivePriorDepth is the fixed candidate depth of the current
	// collection search.
	searchLivePriorDepth = 16384
	// searchLiveOracleBlock is the ID count of one oracle scoring request.
	searchLiveOracleBlock = 1024
	// searchLiveTimeout bounds the completeness test, which embeds every input
	// through the production endpoint. The endpoint measured 3.0 to 5.9 inputs
	// per second, so 20,480 inputs take 58 to 114 minutes.
	searchLiveTimeout = 140 * time.Minute
	// searchLiveProgressOwners is the number of owners between two publish
	// progress log lines.
	searchLiveProgressOwners = 16
	// searchLiveMetricsEnv is the environment variable with the path of a file
	// that receives the measured search metrics as JSON.
	searchLiveMetricsEnv = "LMS_LIBRARY_SEARCH_METRICS"
)

// searchLiveVocabulary is combined by index into distinct embedding inputs.
var searchLiveVocabulary = []string{
	"daemon", "reload", "bind", "socket", "embedding", "vector", "cursor", "snapshot", "page", "filter",
	"workspace", "archive", "graphite", "stack", "restack", "milvus", "sqlite", "checkpoint", "lock", "writer",
	"lexical", "analyzer", "token", "query", "ranking", "fusion", "quota", "group", "deadline", "budget",
}

// searchLiveNamespace declares the scalars that the live requests filter and
// group by.
func searchLiveNamespace() library.NamespaceSpec {
	return library.NamespaceSpec{
		ID:     "search",
		Policy: library.ReplaceAllowed,
		Scalars: []library.ScalarColumn{
			{Name: "owner", Type: library.String, Nullable: false, Mutable: false, MaxLength: 64},
			{Name: "group", Type: library.Int64, Nullable: false, Mutable: false, MaxLength: 0},
			{Name: "workspace", Type: library.String, Nullable: true, Mutable: true, MaxLength: 128},
			{Name: "archived", Type: library.Bool, Nullable: false, Mutable: true, MaxLength: 0},
		},
	}
}

func searchLiveInput(index int) string {
	words := len(searchLiveVocabulary)
	return fmt.Sprintf("entry %05d on %s %s %s", index,
		searchLiveVocabulary[index%words], searchLiveVocabulary[(index/words)%words], searchLiveVocabulary[(index/(words*words))%words])
}

func searchLiveOwner(index int) string {
	return fmt.Sprintf("owner-%03d", index%searchLiveOwners)
}

// searchLiveRow builds one occurrence of distinct input index. A duplicate
// occurrence reuses the input and vector with another row key, source text,
// and SortKey.
func searchLiveRow(index int, duplicate bool) library.Occurrence {
	scalars := map[string]library.ScalarValue{
		"owner": {Type: library.String, Null: false, String: searchLiveOwner(index), Bool: false, Int64: 0},
		"group": {Type: library.Int64, Null: false, String: "", Bool: false, Int64: int64(index % 23)},
	}
	switch index % 5 {
	case 0:
		scalars["workspace"] = library.ScalarValue{Type: library.String, Null: false, String: "/repo/a", Bool: false, Int64: 0}
	case 1:
		scalars["workspace"] = library.ScalarValue{Type: library.String, Null: false, String: "/repo/a/b", Bool: false, Int64: 0}
	case 2:
		scalars["workspace"] = library.ScalarValue{Type: library.String, Null: false, String: "/repo/c", Bool: false, Int64: 0}
	case 3:
		scalars["workspace"] = library.ScalarValue{Type: library.String, Null: true, String: "", Bool: false, Int64: 0}
	}
	if index%7 != 0 {
		scalars["archived"] = library.ScalarValue{Type: library.Bool, Null: false, String: "", Bool: index%3 == 0, Int64: 0}
	}
	rowKey := fmt.Sprintf("r%05d", index)
	source := "source " + searchLiveInput(index)
	sortKey := fmt.Sprintf("%02d", (index*7)%11)
	if duplicate {
		rowKey += "-copy"
		source += " (copy)"
		sortKey = fmt.Sprintf("%02d", (index*5)%11)
	}
	return library.Occurrence{
		RowKey:         rowKey,
		SortKey:        sortKey,
		SourceText:     source,
		SearchText:     searchLiveInput(index),
		EmbeddingInput: searchLiveInput(index),
		Scalars:        scalars,
	}
}

// searchLiveCorpus returns every owner's rows: one occurrence per distinct
// input and a duplicate occurrence of every eighth input in the next owner.
func searchLiveCorpus(distinct int) map[string][]library.Occurrence {
	byOwner := map[string][]library.Occurrence{}
	for index := range distinct {
		owner := searchLiveOwner(index)
		byOwner[owner] = append(byOwner[owner], searchLiveRow(index, false))
		if index%8 == 0 {
			next := searchLiveOwner(index + 1)
			byOwner[next] = append(byOwner[next], searchLiveRow(index, true))
		}
	}
	return byOwner
}

// searchLiveStore is one open library over an isolated Milvus collection and
// the test's own record of every occurrence.
type searchLiveStore struct {
	testbed    *libraryHarness
	ctx        context.Context
	collection string
	descriptor library.StoreDescriptor
	vectors    library.VectorStore
	library    *library.Library
	records    map[library.OccurrenceID]library.Occurrence
}

func searchLiveConfig(store *searchLiveStore) library.Config {
	return library.Config{
		Store:          store.descriptor,
		Vectors:        store.vectors,
		Embedder:       store.testbed.embedder,
		SearchMode:     library.Dense,
		MaxPageSize:    100,
		MaxQueryBytes:  4096,
		MaxFilterDepth: 16,
		QueryTimeout:   10 * time.Minute,
	}
}

func newSearchLiveStore(t *testing.T, storeContext context.Context, liveHarness *libraryHarness, name string) *searchLiveStore {
	t.Helper()
	store := &searchLiveStore{
		testbed:    liveHarness,
		ctx:        storeContext,
		collection: name + "_vectors",
		descriptor: liveHarness.descriptor(name),
		vectors:    nil,
		library:    nil,
		records:    map[library.OccurrenceID]library.Occurrence{},
	}
	store.vectors = liveHarness.vectorStore(store.collection)
	store.library = store.open(t, nil)
	if err := store.library.RegisterNamespace(storeContext, searchLiveNamespace()); err != nil {
		t.Fatalf("register namespace: %v", err)
	}
	return store
}

func (store *searchLiveStore) open(t *testing.T, configure func(*library.Config)) *library.Library {
	t.Helper()
	config := searchLiveConfig(store)
	if configure != nil {
		configure(&config)
	}
	opened, err := library.Open(store.ctx, config)
	if err != nil {
		t.Fatalf("open library: %v", err)
	}
	t.Cleanup(func() {
		if err := opened.Close(); err != nil {
			t.Errorf("close library: %v", err)
		}
	})
	return opened
}

func (store *searchLiveStore) replaceOwner(t *testing.T, owner string, order uint64, rows []library.Occurrence) {
	t.Helper()
	_, err := store.library.Apply(store.ctx, library.Batch{
		Namespace: "search", OwnerID: owner, GenerationOrder: order,
		IdempotencyToken: fmt.Sprintf("%s-%d", owner, order), Mode: library.Replace, Rows: rows,
	})
	if err != nil {
		t.Fatalf("apply owner %s order %d: %v", owner, order, err)
	}
	for id := range store.records {
		if id.OwnerID == owner {
			delete(store.records, id)
		}
	}
	for _, row := range rows {
		store.records[library.OccurrenceID{Namespace: "search", OwnerID: owner, RowKey: row.RowKey}] = row
	}
}

// publish replaces every owner and logs the published occurrence count and
// rate every searchLiveProgressOwners owners.
func (store *searchLiveStore) publish(t *testing.T, byOwner map[string][]library.Occurrence) {
	t.Helper()
	started := time.Now()
	published := 0
	for index, owner := range slices.Sorted(searchLiveKeys(byOwner)) {
		store.replaceOwner(t, owner, 1, byOwner[owner])
		published += len(byOwner[owner])
		if (index+1)%searchLiveProgressOwners == 0 {
			elapsed := time.Since(started)
			t.Logf("published %d owners, %d occurrences in %.0f s (%.2f occurrences per second) at %s",
				index+1, published, elapsed.Seconds(), float64(published)/elapsed.Seconds(), time.Now().UTC().Format(time.RFC3339))
		}
	}
}

func searchLiveKeys[K comparable, V any](values map[K]V) func(func(K) bool) {
	return func(yield func(K) bool) {
		for key := range values {
			if !yield(key) {
				return
			}
		}
	}
}

// catalogVectorIDs reads each occurrence's vector ID from the catalog with a
// separate read-only connection.
func (store *searchLiveStore) catalogVectorIDs(t *testing.T) map[library.OccurrenceID]string {
	t.Helper()
	database := searchLiveCatalog(t, store.descriptor)
	rows, err := database.QueryContext(store.ctx, `SELECT owner_id, row_key, vector_id FROM occurrences WHERE namespace = 'search'`)
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
		vectorIDs[library.OccurrenceID{Namespace: "search", OwnerID: owner, RowKey: rowKey}] = vectorID
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("read occurrence vectors: %v", err)
	}
	return vectorIDs
}

func searchLiveCatalog(t *testing.T, descriptor library.StoreDescriptor) *sql.DB {
	t.Helper()
	database, err := sql.Open("sqlite3", "file:"+descriptor.CatalogPath+"?mode=ro")
	if err != nil {
		t.Fatalf("open catalog read-only: %v", err)
	}
	t.Cleanup(func() { _ = database.Close() })
	return database
}

// deterministicQueryVector embeds the query three times through the
// production adapter and requires equal bytes. The oracle scores with the
// same vector that Search embeds.
func (store *searchLiveStore) deterministicQueryVector(t *testing.T, query string) []float32 {
	t.Helper()
	var first []float32
	for attempt := range 3 {
		vectors, err := store.testbed.embedder.EmbedBatch(store.ctx, []string{query})
		if err != nil {
			t.Fatalf("embed oracle query: %v", err)
		}
		if attempt == 0 {
			first = vectors[0]
			continue
		}
		if !slices.Equal(first, vectors[0]) {
			t.Fatalf("the embedding endpoint returned different vectors for one query on attempt %d; the oracle needs a deterministic query vector", attempt)
		}
	}
	return first
}

// searchLivePageSizes are the page sizes that every live request pages to
// exhaustion.
var searchLivePageSizes = []int{1, 10, 100}

// searchLiveCosineTolerance is the largest accepted difference between a
// Milvus COSINE score, computed in float32, and the test's float64 cosine.
const searchLiveCosineTolerance = 1e-5

// searchLiveVectorReadBlock is the ID count of one strong vector read.
const searchLiveVectorReadBlock = 256

// searchLiveWorkerCounts are the QueryWorkers values that the cache
// measurement compares.
var searchLiveWorkerCounts = []int{2, 4, 8}

// searchLiveWarmSamples is the number of page one searches after the first
// search of a library at one catalog revision.
const searchLiveWarmSamples = 3

// measureVerificationCache opens a library for each QueryWorkers value and
// request, which starts with an empty verification cache, and times page one
// of the unfiltered and filtered requests: the first search verifies every
// eligible vector, and the next searches at the same catalog revision verify
// none.
func (store *searchLiveStore) measureVerificationCache(
	t *testing.T,
	metrics *searchLiveMetrics,
	requests map[string]library.SearchRequest,
) {
	t.Helper()
	recorder := &searchLivePhaseRecorder{mutex: sync.Mutex{}, records: nil}
	previousLogger := slog.Default()
	slog.SetDefault(slog.New(recorder))
	defer slog.SetDefault(previousLogger)
	metrics.PageOneCache = map[string]string{}
	for _, workers := range searchLiveWorkerCounts {
		for _, name := range []string{"unfiltered", "filtered"} {
			searcher := store.open(t, func(config *library.Config) { config.QueryWorkers = workers })
			request := requests[name]
			request.PageSize = 10
			for sample := range 1 + searchLiveWarmSamples {
				started := time.Now()
				if _, err := searcher.Search(store.ctx, request); err != nil {
					t.Fatalf("%s page one with %d workers, sample %d: %v", name, workers, sample, err)
				}
				elapsed := time.Since(started)
				records := recorder.take()
				if len(records) != 1 {
					t.Fatalf("%s page one with %d workers wrote %d phase records", name, workers, len(records))
				}
				phase := records[0]
				state := "warm"
				if sample == 0 {
					state = "cold"
				}
				if state == "warm" && phase["verified_vectors"] != 0 {
					t.Fatalf("%s warm page one with %d workers verified %v vectors, want 0", name, workers, phase["verified_vectors"])
				}
				summary := fmt.Sprintf("total_ms %.1f dense_ms %.1f verify_ms %.1f score_ms %.1f verified_vectors %.0f",
					float64(elapsed.Microseconds())/1000, phase["dense_ms"], phase["verify_ms"], phase["score_ms"], phase["verified_vectors"])
				key := fmt.Sprintf("%s/workers %d/%s %d", name, workers, state, sample)
				metrics.PageOneCache[key] = summary
				t.Logf("page one %s: %s", key, summary)
			}
		}
	}
}

// checkOracleCosine reads every published vector with a strong Milvus query,
// computes its cosine with the query vector in float64, and compares it with
// the oracle score that a Milvus search returned for the same ID.
func (store *searchLiveStore) checkOracleCosine(t *testing.T, query string) {
	t.Helper()
	vectorIDs := store.catalogVectorIDs(t)
	distinct := map[string]bool{}
	for _, id := range vectorIDs {
		distinct[id] = true
	}
	ids := slices.Sorted(searchLiveKeys(distinct))
	queryVector := store.deterministicQueryVector(t, query)
	scores := store.oracleScores(t, queryVector, ids)
	largest := 0.0
	for start := 0; start < len(ids); start += searchLiveVectorReadBlock {
		block := ids[start:min(start+searchLiveVectorReadBlock, len(ids))]
		result, err := store.testbed.milvus.Query(store.ctx, milvusclient.NewQueryOption(store.collection).
			WithIDs(column.NewColumnVarChar("vector_id", block)).
			WithOutputFields("vector_id", "vector").
			WithConsistencyLevel(entity.ClStrong))
		if err != nil {
			t.Fatalf("strong read of vectors from %d: %v", start, err)
		}
		vectors, ok := result.GetColumn("vector").(*column.ColumnFloatVector)
		if !ok || result.ResultCount != len(block) {
			t.Fatalf("strong read of vectors from %d returned %d rows for %d IDs", start, result.ResultCount, len(block))
		}
		for row := range result.ResultCount {
			id, err := result.GetColumn("vector_id").GetAsString(row)
			if err != nil {
				t.Fatalf("decode vector ID: %v", err)
			}
			difference := math.Abs(searchLiveCosine(queryVector, vectors.Data()[row]) - scores[id])
			largest = max(largest, difference)
			if difference > searchLiveCosineTolerance {
				t.Fatalf("vector %s: Milvus score %v, the test's cosine differs by %v", id, scores[id], difference)
			}
		}
	}
	t.Logf("oracle scores of %d vectors equal the test's float64 cosine within %v (largest difference %v)", len(ids), searchLiveCosineTolerance, largest)
}

// searchLiveCosine returns the cosine similarity of two vectors in float64.
func searchLiveCosine(left []float32, right []float32) float64 {
	var dot, leftNorm, rightNorm float64
	for index := range left {
		dot += float64(left[index]) * float64(right[index])
		leftNorm += float64(left[index]) * float64(left[index])
		rightNorm += float64(right[index]) * float64(right[index])
	}
	return dot / (math.Sqrt(leftNorm) * math.Sqrt(rightNorm))
}

// oracleScores asks Milvus directly for the exact COSINE score of every
// vector ID with its own client and batching.
func (store *searchLiveStore) oracleScores(t *testing.T, queryVector []float32, ids []string) map[string]float64 {
	t.Helper()
	scores := make(map[string]float64, len(ids))
	for start := 0; start < len(ids); start += searchLiveOracleBlock {
		block := ids[start:min(start+searchLiveOracleBlock, len(ids))]
		results, err := store.testbed.milvus.Search(store.ctx,
			milvusclient.NewSearchOption(store.collection, len(block), []entity.Vector{entity.FloatVector(queryVector)}).
				WithANNSField("vector").
				WithFilter("vector_id in {oracle_ids}").
				WithTemplateParam("oracle_ids", block).
				WithConsistencyLevel(entity.ClStrong),
		)
		if err != nil {
			t.Fatalf("oracle Milvus search: %v", err)
		}
		for _, result := range results {
			for row := range result.ResultCount {
				id, err := result.IDs.GetAsString(row)
				if err != nil {
					t.Fatalf("decode oracle ID: %v", err)
				}
				scores[id] = float64(result.Scores[row])
			}
		}
	}
	if len(scores) != len(ids) {
		t.Fatalf("oracle Milvus search returned %d scores for %d IDs", len(scores), len(ids))
	}
	return scores
}

// searchLiveOracleHit is one ranked occurrence of the oracle.
type searchLiveOracleHit struct {
	id    library.OccurrenceID
	row   library.Occurrence
	score float64
}

// oracle ranks the test's own records for request with Milvus scores and Go
// filter evaluation, and returns the ranked hits and each hit's rank in the
// unfiltered order.
func (store *searchLiveStore) oracle(t *testing.T, request library.SearchRequest) ([]library.SearchHit, map[library.OccurrenceID]int) {
	t.Helper()
	vectorIDs := store.catalogVectorIDs(t)
	distinct := map[string]bool{}
	for _, id := range vectorIDs {
		distinct[id] = true
	}
	scores := store.oracleScores(t, store.deterministicQueryVector(t, request.Query), slices.Sorted(searchLiveKeys(distinct)))
	var all []searchLiveOracleHit
	for id, row := range store.records {
		all = append(all, searchLiveOracleHit{id: id, row: row, score: scores[vectorIDs[id]]})
	}
	slices.SortFunc(all, func(left searchLiveOracleHit, right searchLiveOracleHit) int {
		if left.score != right.score {
			return cmp.Compare(right.score, left.score)
		}
		return cmp.Or(strings.Compare(left.row.SortKey, right.row.SortKey), strings.Compare(left.id.OwnerID, right.id.OwnerID),
			strings.Compare(left.id.RowKey, right.id.RowKey))
	})
	unfilteredRank := map[library.OccurrenceID]int{}
	perGroup := map[string]int{}
	var hits []library.SearchHit
	for rank, hit := range all {
		unfilteredRank[hit.id] = rank + 1
		if request.Filter != nil && searchLiveEvaluate(*request.Filter, hit.row.Scalars) != searchLiveTrue {
			continue
		}
		if request.MinScore > 0 && hit.score < request.MinScore {
			continue
		}
		if request.GroupBy != "" {
			key := searchLiveGroupKey(hit.row.Scalars, request.GroupBy)
			if perGroup[key] == request.PerGroupLimit {
				continue
			}
			perGroup[key]++
		}
		scalars := hit.row.Scalars
		if scalars == nil {
			scalars = map[string]library.ScalarValue{}
		}
		hits = append(hits, library.SearchHit{ID: hit.id, SourceText: hit.row.SourceText, Scalars: scalars, Score: hit.score})
	}
	return hits, unfilteredRank
}

func searchLiveGroupKey(values map[string]library.ScalarValue, column string) string {
	value, present := values[column]
	switch {
	case !present:
		return "absent"
	case value.Null:
		return "null"
	default:
		return fmt.Sprintf("value:%v", searchLiveComparable(value))
	}
}

const (
	searchLiveFalse int8 = iota
	searchLiveUnknown
	searchLiveTrue
)

// searchLiveEvaluate evaluates a filter with SQL three-valued logic. A
// comparison with a null or absent value is unknown.
func searchLiveEvaluate(filter library.Filter, values map[string]library.ScalarValue) int8 {
	switch filter.Op {
	case library.All:
		result := searchLiveTrue
		for _, child := range filter.Children {
			result = min(result, searchLiveEvaluate(child, values))
		}
		return result
	case library.Any:
		result := searchLiveFalse
		for _, child := range filter.Children {
			result = max(result, searchLiveEvaluate(child, values))
		}
		return result
	case library.Not:
		return searchLiveTrue - searchLiveEvaluate(filter.Children[0], values)
	case library.IsNull:
		value, present := values[filter.Column]
		return searchLiveTruth(present && value.Null)
	case library.IsPresent:
		_, present := values[filter.Column]
		return searchLiveTruth(present)
	}
	value, present := values[filter.Column]
	if !present || value.Null {
		return searchLiveUnknown
	}
	switch filter.Op {
	case library.Equal, library.In:
		for _, candidate := range filter.Values {
			if searchLiveComparable(candidate) == searchLiveComparable(value) {
				return searchLiveTrue
			}
		}
		return searchLiveFalse
	case library.Range:
		return searchLiveTruth((filter.Lower == nil || value.Int64 >= filter.Lower.Int64) && (filter.Upper == nil || value.Int64 < filter.Upper.Int64))
	default:
		return searchLiveTruth(strings.HasPrefix(value.String, filter.Prefix))
	}
}

func searchLiveTruth(condition bool) int8 {
	if condition {
		return searchLiveTrue
	}
	return searchLiveFalse
}

func searchLiveComparable(value library.ScalarValue) any {
	switch value.Type {
	case library.Bool:
		return value.Bool
	case library.Int64:
		return value.Int64
	default:
		return value.String
	}
}

// searchLiveMetrics are the measured costs of the completeness corpus.
type searchLiveMetrics struct {
	DistinctVectors      int                `json:"distinct_vectors"`
	Occurrences          int                `json:"occurrences"`
	PublishSeconds       float64            `json:"publish_seconds"`
	FirstPageSeconds     map[string]float64 `json:"first_page_seconds"`
	CursorPageP50Millis  float64            `json:"cursor_page_p50_ms"`
	CursorPageP95Millis  float64            `json:"cursor_page_p95_ms"`
	CursorPageMaxMillis  float64            `json:"cursor_page_max_ms"`
	QueryDatabaseBytes   int64              `json:"query_database_peak_bytes"`
	SnapshotResultBytes  int64              `json:"snapshot_result_bytes"`
	PeakRSSBytes         int64              `json:"test_process_peak_rss_bytes"`
	MilvusLiveVectors    int64              `json:"milvus_live_vectors"`
	CompactedSegmentRows int64              `json:"compacted_segment_rows"`
	MeasuredVectors      int                `json:"measured_vectors"`
	PageOneCache         map[string]string  `json:"page_one_cache"`
	VerifyStrongSeconds  float64            `json:"verify_strong_seconds_all_vectors"`
	ScoreExactSeconds    float64            `json:"score_exact_seconds_all_vectors"`
	cursorDurations      []time.Duration
}

// pageAll pages request to exhaustion at pageSize and records the page one
// and cursor page latencies. It fails on a short page before the end, an
// empty page after a cursor, or a cursor on the last page.
func (store *searchLiveStore) pageAll(
	t *testing.T,
	searcher *library.Library,
	request library.SearchRequest,
	pageSize int,
	metrics *searchLiveMetrics,
	label string,
) []library.SearchHit {
	t.Helper()
	request.PageSize = pageSize
	request.Cursor = ""
	var hits []library.SearchHit
	for pages := 0; ; pages++ {
		started := time.Now()
		page, err := searcher.Search(store.ctx, request)
		elapsed := time.Since(started)
		if err != nil {
			t.Fatalf("%s: page %d at page size %d: %v", label, pages, pageSize, err)
		}
		if pages == 0 {
			metrics.FirstPageSeconds[fmt.Sprintf("%s/%d", label, pageSize)] = elapsed.Seconds()
		} else {
			metrics.cursorDurations = append(metrics.cursorDurations, elapsed)
		}
		if page.HasMore && len(page.Hits) != pageSize {
			t.Fatalf("%s: page %d at page size %d has %d hits and HasMore", label, pages, pageSize, len(page.Hits))
		}
		if pages > 0 && len(page.Hits) == 0 {
			t.Fatalf("%s: page %d at page size %d is empty after a cursor", label, pages, pageSize)
		}
		if !page.HasMore && page.NextCursor != "" {
			t.Fatalf("%s: last page %d at page size %d has a cursor", label, pages, pageSize)
		}
		hits = append(hits, page.Hits...)
		if !page.HasMore {
			return hits
		}
		request.Cursor = page.NextCursor
	}
}

func searchLiveAssertHits(t *testing.T, label string, got []library.SearchHit, want []library.SearchHit) {
	t.Helper()
	seen := map[library.OccurrenceID]int{}
	for index, hit := range got {
		if first, repeated := seen[hit.ID]; repeated {
			t.Fatalf("%s: hit %d repeats %+v from position %d", label, index, hit.ID, first)
		}
		seen[hit.ID] = index
	}
	missing := 0
	for _, hit := range want {
		if _, found := seen[hit.ID]; !found {
			missing++
		}
	}
	if len(got) != len(want) || missing != 0 {
		t.Fatalf("%s: %d hits, oracle %d, %d oracle hits missing", label, len(got), len(want), missing)
	}
	for index := range want {
		gotHit := got[index]
		if gotHit.Scalars == nil {
			gotHit.Scalars = map[string]library.ScalarValue{}
		}
		if !reflect.DeepEqual(gotHit, want[index]) {
			t.Fatalf("%s: hit %d = %+v, oracle %+v", label, index, gotHit, want[index])
		}
	}
}

func stringScalar(text string) library.ScalarValue {
	return library.ScalarValue{Type: library.String, Null: false, String: text, Bool: false, Int64: 0}
}

func int64Scalar(value int64) *library.ScalarValue {
	return &library.ScalarValue{Type: library.Int64, Null: false, String: "", Bool: false, Int64: value}
}

// TestLibrarySearchCompletePagesMatchTheExhaustiveOracle publishes more than
// 20,000 distinct vectors through the production embedding adapter and real
// Milvus, pages unfiltered, filtered, and grouped requests to exhaustion, and
// compares every page with an exhaustive per-occurrence oracle. It then
// checks cursor replay, writes between pages, deadline and disk budgets,
// snapshot expiry, and vector loss.
func TestLibrarySearchCompletePagesMatchTheExhaustiveOracle(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), searchLiveTimeout)
	defer cancel()
	store := newSearchLiveStore(t, ctx, newLibraryHarness(t), "search_complete")
	corpus := searchLiveCorpus(searchLiveDistinctInputs)
	metrics := &searchLiveMetrics{FirstPageSeconds: map[string]float64{}}
	started := time.Now()
	t.Logf("embedding window start %s", started.UTC().Format(time.RFC3339))
	store.publish(t, corpus)
	metrics.PublishSeconds = time.Since(started).Seconds()
	t.Logf("publish finished %s after %.1f s", time.Now().UTC().Format(time.RFC3339), metrics.PublishSeconds)
	vectorIDs := store.catalogVectorIDs(t)
	distinct := map[string]bool{}
	for _, id := range vectorIDs {
		distinct[id] = true
	}
	metrics.DistinctVectors = len(distinct)
	metrics.Occurrences = len(vectorIDs)
	if metrics.DistinctVectors <= 20000 || metrics.Occurrences != len(store.records) {
		t.Fatalf("corpus has %d distinct vectors and %d catalog occurrences for %d records; want more than 20000 vectors",
			metrics.DistinctVectors, metrics.Occurrences, len(store.records))
	}

	stopWatch := watchQueryDatabases(filepath.Dir(store.descriptor.CatalogPath), &metrics.QueryDatabaseBytes)
	requests := searchLiveRequests()
	for _, name := range slices.Sorted(searchLiveKeys(requests)) {
		request := requests[name]
		want, unfilteredRank := store.oracle(t, request)
		if name == "filtered" {
			deepest := 0
			for _, hit := range want {
				deepest = max(deepest, unfilteredRank[hit.ID])
			}
			if deepest <= searchLivePriorDepth {
				t.Fatalf("filtered request: the deepest eligible hit has unfiltered rank %d, within the prior depth %d", deepest, searchLivePriorDepth)
			}
		}
		for _, pageSize := range searchLivePageSizes {
			got := store.pageAll(t, store.library, request, pageSize, metrics, name)
			searchLiveAssertHits(t, fmt.Sprintf("%s at page size %d", name, pageSize), got, want)
			t.Logf("%s at page size %d: %d hits equal the oracle hit for hit", name, pageSize, len(got))
		}
	}
	stopWatch()
	store.checkOracleCosine(t, requests["unfiltered"].Query)
	store.measureVerificationCache(t, metrics, requests)
	store.checkHybrid(t, metrics)
	store.measureVectorReads(t, metrics)

	store.checkCursorReplayAndWrites(t)
	store.checkBudgetsAndExpiry(t)
	store.recordStorage(t, metrics)
	store.checkVectorLoss(t)
	finishSearchLiveMetrics(t, metrics)
	t.Logf("embedding window end %s", time.Now().UTC().Format(time.RFC3339))
}

const (
	// searchLiveBM25Collection is the oracle collection with one row per
	// occurrence. A BM25 function fills its sparse field from its text field.
	searchLiveBM25Collection = "search_bm25_oracle"
	searchLiveBM25IDField    = "id"
	searchLiveBM25TextField  = "text"
	searchLiveBM25Sparse     = "sparse"
	searchLiveBM25TextLength = 1024
	// searchLiveHybridQuery avoids the words that every input contains. Its
	// lexical matches stay below the 16,384 result limit of one Milvus search.
	searchLiveHybridQuery = "restack fusion quota"
	// searchLiveRRFK is the default reciprocal rank fusion constant.
	searchLiveRRFK = 60
)

// bm25OracleScores inserts the SearchText of every record into a BM25
// collection, one row per occurrence, and returns the Milvus 2.6.18 BM25
// score of every occurrence that matches query. Milvus counts one document
// per row, and each occurrence is one row in the corpus statistics.
func (store *searchLiveStore) bm25OracleScores(t *testing.T, query string) map[library.OccurrenceID]float32 {
	t.Helper()
	client := store.testbed.milvus
	schema := entity.NewSchema().
		WithField(entity.NewField().WithName(searchLiveBM25IDField).WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName(searchLiveBM25TextField).WithDataType(entity.FieldTypeVarChar).
			WithMaxLength(searchLiveBM25TextLength).WithEnableAnalyzer(true)).
		WithField(entity.NewField().WithName(searchLiveBM25Sparse).WithDataType(entity.FieldTypeSparseVector)).
		WithFunction(entity.NewFunction().WithName("text_bm25").WithType(entity.FunctionTypeBM25).
			WithInputFields(searchLiveBM25TextField).WithOutputFields(searchLiveBM25Sparse))
	indexParams := map[string]string{
		index.IndexTypeKey:    string(index.SparseInverted),
		index.MetricTypeKey:   string(entity.BM25),
		"bm25_k1":             "1.2",
		"bm25_b":              "0.75",
		"inverted_index_algo": "TAAT_NAIVE",
	}
	if err := client.CreateCollection(store.ctx, milvusclient.NewCreateCollectionOption(searchLiveBM25Collection, schema).
		WithIndexOptions(milvusclient.NewCreateIndexOption(searchLiveBM25Collection, searchLiveBM25Sparse,
			index.NewGenericIndex(searchLiveBM25Sparse, indexParams)).WithIndexName(searchLiveBM25Sparse))); err != nil {
		t.Fatalf("create BM25 oracle collection: %v", err)
	}
	ids := slices.SortedFunc(searchLiveKeys(store.records), func(left library.OccurrenceID, right library.OccurrenceID) int {
		return cmp.Or(strings.Compare(left.OwnerID, right.OwnerID), strings.Compare(left.RowKey, right.RowKey))
	})
	rowIDs := make([]int64, len(ids))
	texts := make([]string, len(ids))
	for index, id := range ids {
		rowIDs[index] = int64(index)
		texts[index] = store.records[id].SearchText
	}
	if _, err := client.Insert(store.ctx, milvusclient.NewColumnBasedInsertOption(searchLiveBM25Collection).
		WithInt64Column(searchLiveBM25IDField, rowIDs).WithVarcharColumn(searchLiveBM25TextField, texts)); err != nil {
		t.Fatalf("insert BM25 oracle rows: %v", err)
	}
	store.awaitBM25Index(t, len(ids))
	results, err := client.Search(store.ctx, milvusclient.NewSearchOption(searchLiveBM25Collection, searchLivePriorDepth,
		[]entity.Vector{entity.Text(query)}).WithANNSField(searchLiveBM25Sparse).WithConsistencyLevel(entity.ClStrong))
	if err != nil || len(results) != 1 {
		t.Fatalf("BM25 oracle search: %d result sets, %v", len(results), err)
	}
	if results[0].ResultCount >= searchLivePriorDepth {
		t.Fatalf("BM25 oracle search returned %d rows, at the search limit; the oracle cannot list every match", results[0].ResultCount)
	}
	scores := make(map[library.OccurrenceID]float32, results[0].ResultCount)
	for position := range results[0].ResultCount {
		rowID, err := results[0].IDs.GetAsInt64(position)
		if err != nil {
			t.Fatalf("decode BM25 oracle row: %v", err)
		}
		scores[ids[rowID]] = results[0].Scores[position]
	}
	return scores
}

// awaitBM25Index flushes the oracle collection, waits until its BM25 index
// covers every row, and loads it.
func (store *searchLiveStore) awaitBM25Index(t *testing.T, rows int) {
	t.Helper()
	client := store.testbed.milvus
	flush, err := client.Flush(store.ctx, milvusclient.NewFlushOption(searchLiveBM25Collection))
	if err != nil {
		t.Fatalf("flush BM25 oracle collection: %v", err)
	}
	if err := flush.Await(store.ctx); err != nil {
		t.Fatalf("await BM25 oracle flush: %v", err)
	}
	for {
		description, err := client.DescribeIndex(store.ctx, milvusclient.NewDescribeIndexOption(searchLiveBM25Collection, searchLiveBM25Sparse))
		if err != nil {
			t.Fatalf("describe BM25 oracle index: %v", err)
		}
		if description.IndexedRows >= int64(rows) && description.PendingIndexRows == 0 {
			break
		}
		time.Sleep(200 * time.Millisecond)
	}
	load, err := client.LoadCollection(store.ctx, milvusclient.NewLoadCollectionOption(searchLiveBM25Collection))
	if err != nil {
		t.Fatalf("load BM25 oracle collection: %v", err)
	}
	if err := load.Await(store.ctx); err != nil {
		t.Fatalf("await BM25 oracle load: %v", err)
	}
}

// hybridOracle ranks the eligible records of request with reciprocal rank
// fusion of the dense rank and the BM25 rank. Both ranks count only eligible
// occurrences and break score ties by occurrence ID.
func (store *searchLiveStore) hybridOracle(t *testing.T, request library.SearchRequest, bm25 map[library.OccurrenceID]float32) []library.SearchHit {
	t.Helper()
	vectorIDs := store.catalogVectorIDs(t)
	distinct := map[string]bool{}
	for _, id := range vectorIDs {
		distinct[id] = true
	}
	dense := store.oracleScores(t, store.deterministicQueryVector(t, request.Query), slices.Sorted(searchLiveKeys(distinct)))
	var eligible []library.OccurrenceID
	for id, row := range store.records {
		if request.Filter == nil || searchLiveEvaluate(*request.Filter, row.Scalars) == searchLiveTrue {
			eligible = append(eligible, id)
		}
	}
	byID := func(left library.OccurrenceID, right library.OccurrenceID) int {
		return cmp.Or(strings.Compare(left.OwnerID, right.OwnerID), strings.Compare(left.RowKey, right.RowKey))
	}
	denseOrder := slices.Clone(eligible)
	slices.SortFunc(denseOrder, func(left library.OccurrenceID, right library.OccurrenceID) int {
		leftScore, rightScore := dense[vectorIDs[left]], dense[vectorIDs[right]]
		if leftScore != rightScore {
			return cmp.Compare(rightScore, leftScore)
		}
		return byID(left, right)
	})
	var lexicalOrder []library.OccurrenceID
	for _, id := range eligible {
		if _, matched := bm25[id]; matched {
			lexicalOrder = append(lexicalOrder, id)
		}
	}
	slices.SortFunc(lexicalOrder, func(left library.OccurrenceID, right library.OccurrenceID) int {
		if bm25[left] != bm25[right] {
			return cmp.Compare(bm25[right], bm25[left])
		}
		return byID(left, right)
	})
	fused := make(map[library.OccurrenceID]float64, len(eligible))
	for rank, id := range denseOrder {
		fused[id] = 1.0 / float64(searchLiveRRFK+rank+1)
	}
	for rank, id := range lexicalOrder {
		fused[id] += 1.0 / float64(searchLiveRRFK+rank+1)
	}
	slices.SortFunc(eligible, func(left library.OccurrenceID, right library.OccurrenceID) int {
		if fused[left] != fused[right] {
			return cmp.Compare(fused[right], fused[left])
		}
		return cmp.Or(strings.Compare(store.records[left].SortKey, store.records[right].SortKey), byID(left, right))
	})
	hits := make([]library.SearchHit, 0, len(eligible))
	perGroup := map[string]int{}
	for _, id := range eligible {
		if request.MinScore > 0 && fused[id] < request.MinScore {
			break
		}
		row := store.records[id]
		if request.GroupBy != "" {
			key := searchLiveGroupKey(row.Scalars, request.GroupBy)
			if perGroup[key] == request.PerGroupLimit {
				continue
			}
			perGroup[key]++
		}
		scalars := row.Scalars
		if scalars == nil {
			scalars = map[string]library.ScalarValue{}
		}
		hits = append(hits, library.SearchHit{ID: id, SourceText: row.SourceText, Scalars: scalars, Score: fused[id]})
	}
	return hits
}

// checkHybrid opens a Hybrid library over the same catalog and pool and pages
// unfiltered, filtered, grouped, and MinScore requests against the RRF oracle
// with Milvus BM25 scores.
func (store *searchLiveStore) checkHybrid(t *testing.T, metrics *searchLiveMetrics) {
	t.Helper()
	bm25 := store.bm25OracleScores(t, searchLiveHybridQuery)
	t.Logf("BM25 oracle: %d of %d occurrences match %q", len(bm25), len(store.records), searchLiveHybridQuery)
	hybrid := store.open(t, func(config *library.Config) { config.SearchMode = library.Hybrid })
	requests := map[string]library.SearchRequest{
		"hybrid unfiltered": {Namespace: "search", Query: searchLiveHybridQuery},
		"hybrid filtered": {Namespace: "search", Query: searchLiveHybridQuery, Filter: &library.Filter{
			Op: library.Range, Column: "group", Lower: int64Scalar(3), Upper: int64Scalar(9),
		}},
		"hybrid grouped": {Namespace: "search", Query: searchLiveHybridQuery, GroupBy: "owner", PerGroupLimit: 3},
	}
	unfloored := store.hybridOracle(t, requests["hybrid unfiltered"], bm25)
	floorIndex := len(unfloored) / 2
	for floorIndex+1 < len(unfloored) && unfloored[floorIndex].Score == unfloored[floorIndex+1].Score {
		floorIndex++
	}
	if floorIndex+1 >= len(unfloored) {
		t.Fatalf("hybrid oracle has no score gap after position %d of %d", len(unfloored)/2, len(unfloored))
	}
	requests["hybrid MinScore"] = library.SearchRequest{
		Namespace: "search", Query: searchLiveHybridQuery,
		MinScore: (unfloored[floorIndex].Score + unfloored[floorIndex+1].Score) / 2,
	}
	for _, name := range slices.Sorted(searchLiveKeys(requests)) {
		request := requests[name]
		want := store.hybridOracle(t, request, bm25)
		for _, pageSize := range searchLivePageSizes {
			got := store.pageAll(t, hybrid, request, pageSize, metrics, name)
			searchLiveAssertHits(t, fmt.Sprintf("%s at page size %d", name, pageSize), got, want)
			t.Logf("%s at page size %d: %d hits equal the oracle hit for hit", name, pageSize, len(got))
		}
	}
}

// searchLiveRequests are an unfiltered request, a filtered request that
// selects hits beyond the prior fixed depth, and a grouped request.
func searchLiveRequests() map[string]library.SearchRequest {
	base := library.SearchRequest{Namespace: "search", Query: "reload the daemon socket after a writer lock checkpoint"}
	filtered := base
	filtered.Filter = &library.Filter{Op: library.All, Children: []library.Filter{
		{Op: library.Range, Column: "group", Lower: int64Scalar(3), Upper: int64Scalar(5)},
		{Op: library.Not, Children: []library.Filter{{Op: library.Equal, Column: "workspace", Values: []library.ScalarValue{stringScalar("/repo/c")}}}},
	}}
	grouped := base
	grouped.Filter = &library.Filter{Op: library.Prefix, Column: "workspace", Prefix: "/repo/a"}
	grouped.GroupBy = "owner"
	grouped.PerGroupLimit = 5
	return map[string]library.SearchRequest{"unfiltered": base, "filtered": filtered, "grouped": grouped}
}

// checkCursorReplayAndWrites replays one cursor twice and changes the corpus
// between pages. The page sequence after page one equals the oracle before
// the writes.
func (store *searchLiveStore) checkCursorReplayAndWrites(t *testing.T) {
	t.Helper()
	request := library.SearchRequest{Namespace: "search", Query: "graphite restack of a stale stack", PageSize: 100}
	want, _ := store.oracle(t, request)
	first, err := store.library.Search(store.ctx, request)
	if err != nil || first.NextCursor == "" {
		t.Fatalf("page one before writes = %v, cursor %q", err, first.NextCursor)
	}
	request.Cursor = first.NextCursor
	replayOne, errOne := store.library.Search(store.ctx, request)
	replayTwo, errTwo := store.library.Search(store.ctx, request)
	if errOne != nil || errTwo != nil || !reflect.DeepEqual(replayOne, replayTwo) {
		t.Fatalf("replaying one cursor returned different pages: %v, %v", errOne, errTwo)
	}

	store.replaceOwner(t, "owner-000", 2, []library.Occurrence{searchLiveRow(0, false)})
	_, err = store.library.ReprojectScalars(store.ctx, library.ScalarProjection{
		Namespace: "search", OwnerID: "owner-001", ProjectionOrder: 1, IdempotencyToken: "project-owner-001",
		Rows: map[string]map[string]library.ScalarValue{"r00001": {"workspace": stringScalar("/repo/changed")}},
	})
	if err != nil {
		t.Fatalf("reproject between pages: %v", err)
	}
	projectedID := library.OccurrenceID{Namespace: "search", OwnerID: "owner-001", RowKey: "r00001"}
	projected := store.records[projectedID]
	projectedScalars := make(map[string]library.ScalarValue, len(projected.Scalars))
	for name, value := range projected.Scalars {
		projectedScalars[name] = value
	}
	projectedScalars["workspace"] = stringScalar("/repo/changed")
	projected.Scalars = projectedScalars
	store.records[projectedID] = projected

	got := slices.Clone(first.Hits)
	for request.Cursor != "" {
		page, err := store.library.Search(store.ctx, request)
		if err != nil {
			t.Fatalf("cursor page after writes: %v", err)
		}
		got = append(got, page.Hits...)
		request.Cursor = page.NextCursor
	}
	searchLiveAssertHits(t, "pages after writes", got, want)
	fresh := library.SearchRequest{Namespace: "search", Query: request.Query}
	freshWant, _ := store.oracle(t, fresh)
	freshMetrics := &searchLiveMetrics{FirstPageSeconds: map[string]float64{}}
	searchLiveAssertHits(t, "fresh search after writes", store.pageAll(t, store.library, fresh, 100, freshMetrics, "fresh"), freshWant)
}

// checkBudgetsAndExpiry opens more libraries over the same catalog and pool
// with small budgets and a short snapshot TTL.
func (store *searchLiveStore) checkBudgetsAndExpiry(t *testing.T) {
	t.Helper()
	request := library.SearchRequest{Namespace: "search", Query: "deadline and budget of a query", PageSize: 10}
	cases := []struct {
		name      string
		configure func(*library.Config)
		want      error
	}{
		{name: "deadline", configure: func(config *library.Config) { config.QueryTimeout = 200 * time.Millisecond }, want: library.ErrDeadline},
		{name: "temporary disk", configure: func(config *library.Config) { config.MaxTemporaryBytes = 256 * 1024 }, want: library.ErrResourceLimit},
		{name: "snapshot disk", configure: func(config *library.Config) { config.MaxSnapshotBytes = 64 * 1024 }, want: library.ErrResourceLimit},
	}
	for _, testCase := range cases {
		limited := store.open(t, testCase.configure)
		page, err := limited.Search(store.ctx, request)
		if !errors.Is(err, testCase.want) || len(page.Hits) != 0 || page.NextCursor != "" {
			t.Fatalf("%s: Search = %d hits, cursor %q, %v; want no page and %v", testCase.name, len(page.Hits), page.NextCursor, err, testCase.want)
		}
	}
	expiring := store.open(t, func(config *library.Config) { config.SnapshotTTL = 2 * time.Second })
	first, err := expiring.Search(store.ctx, request)
	if err != nil || first.NextCursor == "" {
		t.Fatalf("page one with a short TTL = %v, cursor %q", err, first.NextCursor)
	}
	time.Sleep(3 * time.Second)
	request.Cursor = first.NextCursor
	if _, err := expiring.Search(store.ctx, request); !errors.Is(err, library.ErrCursorExpired) {
		t.Fatalf("cursor after the TTL = %v, want ErrCursorExpired", err)
	}
}

// recordStorage records the snapshot bytes, the Milvus live vector count, and
// the physical segment rows after a flush and a completed compaction.
func (store *searchLiveStore) recordStorage(t *testing.T, metrics *searchLiveMetrics) {
	t.Helper()
	database := searchLiveCatalog(t, store.descriptor)
	var snapshotBytes sql.NullInt64
	if err := database.QueryRowContext(store.ctx,
		`SELECT SUM(json_extract(rank_config, '$.result_bytes')) FROM search_snapshots`).Scan(&snapshotBytes); err != nil {
		t.Fatalf("read snapshot bytes: %v", err)
	}
	metrics.SnapshotResultBytes = snapshotBytes.Int64
	metrics.MilvusLiveVectors = store.milvusVectorCount(t)
	client := store.testbed.milvus
	flush, err := client.Flush(store.ctx, milvusclient.NewFlushOption(store.collection))
	if err != nil {
		t.Fatalf("flush vector collection: %v", err)
	}
	if err := flush.Await(store.ctx); err != nil {
		t.Fatalf("await flush: %v", err)
	}
	compactionID, err := client.Compact(store.ctx, milvusclient.NewCompactOption(store.collection))
	if err != nil {
		t.Fatalf("compact vector collection: %v", err)
	}
	for {
		state, err := client.GetCompactionState(store.ctx, milvusclient.NewGetCompactionStateOption(compactionID))
		if err != nil {
			t.Fatalf("read compaction state: %v", err)
		}
		if state == entity.CompactionStateCompleted {
			break
		}
		time.Sleep(time.Second)
	}
	segments, err := client.GetPersistentSegmentInfo(store.ctx, milvusclient.NewGetPersistentSegmentInfoOption(store.collection))
	if err != nil {
		t.Fatalf("read segment info: %v", err)
	}
	for _, segment := range segments {
		metrics.CompactedSegmentRows += segment.NumRows
	}
}

// milvusVectorCount counts the live vectors of the collection with a strong
// read on the test context. The harness context ends after 10 minutes.
func (store *searchLiveStore) milvusVectorCount(t *testing.T) int64 {
	t.Helper()
	result, err := store.testbed.milvus.Query(store.ctx,
		milvusclient.NewQueryOption(store.collection).
			WithFilter(`vector_id != ""`).
			WithOutputFields("count(*)").
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		t.Fatalf("count vectors in %s: %v", store.collection, err)
	}
	count, err := result.GetColumn("count(*)").GetAsInt64(0)
	if err != nil {
		t.Fatalf("decode vector count of %s: %v", store.collection, err)
	}
	return count
}

// measureVectorReads times VerifyStrong and ScoreExact over every published
// vector in blocks of the default QueryBlockSize, one block at a time.
func (store *searchLiveStore) measureVectorReads(t *testing.T, metrics *searchLiveMetrics) {
	t.Helper()
	const blockSize = 512
	database := searchLiveCatalog(t, store.descriptor)
	rows, err := database.QueryContext(store.ctx, `SELECT DISTINCT v.vector_id, v.identity_digest, v.vector_checksum
		FROM occurrences o JOIN vectors v ON v.vector_id = o.vector_id WHERE o.namespace = 'search' ORDER BY v.vector_id`)
	if err != nil {
		t.Fatalf("read vector identities: %v", err)
	}
	var identities []library.VectorIdentity
	for rows.Next() {
		var identity library.VectorIdentity
		if err := rows.Scan(&identity.ID, &identity.IdentityDigest, &identity.Checksum); err != nil {
			t.Fatalf("scan vector identity: %v", err)
		}
		identities = append(identities, identity)
	}
	if err := errors.Join(rows.Err(), rows.Close()); err != nil {
		t.Fatalf("read vector identities: %v", err)
	}
	queryVector := store.deterministicQueryVector(t, searchLiveRequests()["unfiltered"].Query)
	var verifyTime, scoreTime time.Duration
	for start := 0; start < len(identities); start += blockSize {
		block := identities[start:min(start+blockSize, len(identities))]
		ids := make([]string, 0, len(block))
		for _, identity := range block {
			ids = append(ids, identity.ID)
		}
		started := time.Now()
		if err := store.vectors.VerifyStrong(store.ctx, block); err != nil {
			t.Fatalf("VerifyStrong block at %d: %v", start, err)
		}
		verifyTime += time.Since(started)
		started = time.Now()
		if _, err := store.vectors.ScoreExact(store.ctx, queryVector, ids); err != nil {
			t.Fatalf("ScoreExact block at %d: %v", start, err)
		}
		scoreTime += time.Since(started)
	}
	metrics.MeasuredVectors = len(identities)
	metrics.VerifyStrongSeconds = verifyTime.Seconds()
	metrics.ScoreExactSeconds = scoreTime.Seconds()
}

// checkVectorLoss changes one eligible vector's values in Milvus without
// changing its identity fields, which only VerifyStrong detects, publishes a
// new owner to advance the catalog visibility revision, and then deletes
// eligible vectors.
func (store *searchLiveStore) checkVectorLoss(t *testing.T) {
	t.Helper()
	request := library.SearchRequest{Namespace: "search", Query: "vector loss", PageSize: 10}
	vectorIDs := store.catalogVectorIDs(t)
	corruptID := vectorIDs[library.OccurrenceID{Namespace: "search", OwnerID: "owner-002", RowKey: "r00002"}]
	missingID := vectorIDs[library.OccurrenceID{Namespace: "search", OwnerID: "owner-003", RowKey: "r00003"}]
	database := searchLiveCatalog(t, store.descriptor)
	var digest, checksum string
	if err := database.QueryRowContext(store.ctx,
		`SELECT identity_digest, vector_checksum FROM vectors WHERE vector_id = ?`, corruptID).Scan(&digest, &checksum); err != nil {
		t.Fatalf("read vector identity: %v", err)
	}
	replacement := make([]float32, libraryLiveDimension)
	for index := range replacement {
		replacement[index] = float32(math.Sin(float64(index + 1)))
	}
	if _, err := store.testbed.milvus.Upsert(store.ctx, milvusclient.NewColumnBasedInsertOption(store.collection,
		column.NewColumnVarChar("vector_id", []string{corruptID}),
		column.NewColumnFloatVector("vector", libraryLiveDimension, [][]float32{replacement}),
		column.NewColumnVarChar("identity_digest", []string{digest}),
		column.NewColumnVarChar("vector_checksum", []string{checksum}),
	)); err != nil {
		t.Fatalf("replace vector values: %v", err)
	}
	// Earlier searches verified every vector at this catalog visibility
	// revision. A search at the same revision reuses that verification and
	// scores the changed vector. A publication advances the revision, and the
	// next search verifies again.
	page, err := store.library.Search(store.ctx, request)
	if err != nil || len(page.Hits) == 0 {
		t.Fatalf("Search over a changed vector at the verified revision = %d hits, %v; want a page", len(page.Hits), err)
	}
	store.replaceOwner(t, "owner-verification", 1, []library.Occurrence{searchLiveRow(900002, false)})
	page, err = store.library.Search(store.ctx, request)
	if !errors.Is(err, library.ErrVectorCorrupt) || len(page.Hits) != 0 {
		t.Fatalf("Search over a changed vector after a publication = %d hits, %v; want no page and ErrVectorCorrupt", len(page.Hits), err)
	}
	if _, err := store.testbed.milvus.Delete(store.ctx,
		milvusclient.NewDeleteOption(store.collection).WithStringIDs("vector_id", []string{corruptID, missingID})); err != nil {
		t.Fatalf("delete vectors: %v", err)
	}
	page, err = store.library.Search(store.ctx, request)
	if !errors.Is(err, library.ErrVectorMissing) || len(page.Hits) != 0 {
		t.Fatalf("Search with deleted vectors = %d hits, %v; want no page and ErrVectorMissing", len(page.Hits), err)
	}
}

// watchQueryDatabases records the largest query database file in directory
// until the returned function stops the watch.
func watchQueryDatabases(directory string, peak *int64) func() {
	done := make(chan struct{})
	var group sync.WaitGroup
	group.Go(func() {
		ticker := time.NewTicker(20 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				entries, err := os.ReadDir(directory)
				if err != nil {
					continue
				}
				for _, entry := range entries {
					if !strings.HasPrefix(entry.Name(), ".lms-query-") {
						continue
					}
					if info, err := entry.Info(); err == nil {
						*peak = max(*peak, info.Size())
					}
				}
			}
		}
	})
	return func() {
		close(done)
		group.Wait()
	}
}

func finishSearchLiveMetrics(t *testing.T, metrics *searchLiveMetrics) {
	t.Helper()
	durations := slices.Clone(metrics.cursorDurations)
	slices.Sort(durations)
	if len(durations) > 0 {
		metrics.CursorPageP50Millis = float64(durations[len(durations)/2].Microseconds()) / 1000
		metrics.CursorPageP95Millis = float64(durations[len(durations)*95/100].Microseconds()) / 1000
		metrics.CursorPageMaxMillis = float64(durations[len(durations)-1].Microseconds()) / 1000
	}
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err == nil {
		metrics.PeakRSSBytes = usage.Maxrss
	}
	encoded, err := json.Marshal(metrics)
	if err != nil {
		t.Fatalf("encode metrics: %v", err)
	}
	t.Logf("search metrics %s", encoded)
	if path := os.Getenv(searchLiveMetricsEnv); path != "" {
		if err := os.WriteFile(path, encoded, 0o600); err != nil {
			t.Fatalf("write metrics to %s: %v", path, err)
		}
	}
}

// searchLivePhaseSamples is the number of page one searches in each writer
// case of the lock test.
const searchLivePhaseSamples = 20

// searchLivePhaseRecorder keeps the numeric attributes of every "library
// search phases" log record that Search writes after page one.
type searchLivePhaseRecorder struct {
	mutex   sync.Mutex
	records []map[string]float64
}

// Enabled accepts every level, including the debug level of the phase record.
func (recorder *searchLivePhaseRecorder) Enabled(context.Context, slog.Level) bool { return true }

// Handle stores the attributes of a phase record and ignores other records.
func (recorder *searchLivePhaseRecorder) Handle(_ context.Context, record slog.Record) error {
	if record.Message != "library search phases" {
		return nil
	}
	values := map[string]float64{}
	record.Attrs(func(attribute slog.Attr) bool {
		switch attribute.Value.Kind() {
		case slog.KindFloat64:
			values[attribute.Key] = attribute.Value.Float64()
		case slog.KindInt64:
			values[attribute.Key] = float64(attribute.Value.Int64())
		default:
		}
		return true
	})
	recorder.mutex.Lock()
	recorder.records = append(recorder.records, values)
	recorder.mutex.Unlock()
	return nil
}

// WithAttrs returns the same recorder.
func (recorder *searchLivePhaseRecorder) WithAttrs([]slog.Attr) slog.Handler { return recorder }

// WithGroup returns the same recorder.
func (recorder *searchLivePhaseRecorder) WithGroup(string) slog.Handler { return recorder }

// take returns the stored records and clears them.
func (recorder *searchLivePhaseRecorder) take() []map[string]float64 {
	recorder.mutex.Lock()
	defer recorder.mutex.Unlock()
	taken := recorder.records
	recorder.records = nil
	return taken
}

// searchLivePhaseKeys are the phase attributes of the page one log record.
var searchLivePhaseKeys = []string{
	"plan_ms", "read_ms", "embed_ms", "dense_ms", "verify_ms", "score_ms", "rank_ms", "write_wait_ms", "write_ms", "hits_ms",
}

func searchLivePercentiles(values []float64) string {
	sorted := slices.Clone(values)
	slices.Sort(sorted)
	return fmt.Sprintf("p50 %.1f p95 %.1f max %.1f", sorted[len(sorted)/2], sorted[len(sorted)*95/100], sorted[len(sorted)-1])
}

func searchLivePhaseSummary(records []map[string]float64) string {
	parts := make([]string, 0, len(searchLivePhaseKeys))
	for _, key := range searchLivePhaseKeys {
		values := make([]float64, 0, len(records))
		for _, record := range records {
			values = append(values, record[key])
		}
		parts = append(parts, key+" "+searchLivePercentiles(values))
	}
	return strings.Join(parts, "; ")
}

const (
	// searchLiveLargeGenerationRows is the row count of one publication in the
	// lock test. Every row reuses an input that the test already embedded.
	searchLiveLargeGenerationRows = 600
	// searchLiveStageRows is the row count of one Stage batch, below the
	// default MaxBatchRows.
	searchLiveStageRows = 200
)

// publishLargeGeneration replaces owner-publish with searchLiveLargeGenerationRows
// rows in Stage batches and one CommitGeneration. CommitGeneration writes every
// row, scalar, and lexical change in one SQLite write transaction.
func publishLargeGeneration(ctx context.Context, searcher *library.Library, order uint64) error {
	rows := make([]library.Occurrence, 0, searchLiveLargeGenerationRows)
	for index := range searchLiveLargeGenerationRows {
		rows = append(rows, searchLiveRow(index, false))
	}
	key := library.GenerationKey{
		Namespace: "search", OwnerID: "owner-publish", GenerationOrder: order, IdempotencyToken: fmt.Sprintf("publish-%d", order),
	}
	for start := 0; start < len(rows); start += searchLiveStageRows {
		batch := library.StageBatch{Key: key, Mode: library.Replace, Rows: rows[start:min(start+searchLiveStageRows, len(rows))]}
		if err := searcher.Stage(ctx, batch); err != nil {
			return fmt.Errorf("stage rows from %d: %w", start, err)
		}
	}
	seal, err := library.SealRows(rows)
	if err != nil {
		return fmt.Errorf("seal rows: %w", err)
	}
	if _, err := searcher.CommitGeneration(ctx, key, seal); err != nil {
		return fmt.Errorf("commit generation: %w", err)
	}
	return nil
}

// TestLibrarySearchDoesNotWaitForTheWriterLock measures page one latency with
// no writer, while the test process has acquired the kernel writer lock at
// LockPath, and while publications run. A publication that waits for the
// lock proves that the lock was acquired.
func TestLibrarySearchDoesNotWaitForTheWriterLock(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()
	store := newSearchLiveStore(t, ctx, newLibraryHarness(t), "search_lock")
	store.publish(t, searchLiveCorpus(600))
	recorder := &searchLivePhaseRecorder{mutex: sync.Mutex{}, records: nil}
	previousLogger := slog.Default()
	slog.SetDefault(slog.New(recorder))
	defer slog.SetDefault(previousLogger)
	request := library.SearchRequest{Namespace: "search", Query: "writer lock and page one latency", PageSize: 10}
	measure := func(label string) string {
		durations := make([]float64, 0, searchLivePhaseSamples)
		for range searchLivePhaseSamples {
			started := time.Now()
			page, err := store.library.Search(ctx, request)
			if err != nil || !page.HasMore {
				t.Fatalf("%s: page one = %v, HasMore %v", label, err, page.HasMore)
			}
			durations = append(durations, float64(time.Since(started).Microseconds())/1000)
		}
		records := recorder.take()
		if len(records) != searchLivePhaseSamples {
			t.Fatalf("%s: captured %d phase records for %d searches", label, len(records), searchLivePhaseSamples)
		}
		return "total_ms " + searchLivePercentiles(durations) + "; " + searchLivePhaseSummary(records)
	}
	baseline := measure("no writer")

	lockFile, err := os.OpenFile(store.descriptor.LockPath, os.O_RDWR, 0)
	if err != nil {
		t.Fatalf("open lock file: %v", err)
	}
	defer func() { _ = lockFile.Close() }()
	if err := syscall.Flock(int(lockFile.Fd()), syscall.LOCK_EX); err != nil {
		t.Fatalf("acquire the writer lock: %v", err)
	}
	applied := make(chan error, 1)
	go func() {
		_, err := store.library.Apply(ctx, library.Batch{
			Namespace: "search", OwnerID: "owner-lock", GenerationOrder: 1, IdempotencyToken: "lock-1",
			Mode: library.Replace, Rows: []library.Occurrence{searchLiveRow(900001, false)},
		})
		applied <- err
	}()
	locked := measure("writer lock acquired")
	select {
	case err := <-applied:
		t.Fatalf("Apply finished while the test had acquired the writer lock: %v", err)
	default:
	}
	if err := syscall.Flock(int(lockFile.Fd()), syscall.LOCK_UN); err != nil {
		t.Fatalf("release the writer lock: %v", err)
	}
	if err := <-applied; err != nil {
		t.Fatalf("Apply after the lock release: %v", err)
	}

	stop := make(chan struct{})
	publications := make(chan int, 1)
	go func() {
		count := 0
		for order := uint64(1); ; order++ {
			select {
			case <-stop:
				publications <- count
				return
			default:
			}
			if err := publishLargeGeneration(ctx, store.library, order); err != nil {
				t.Logf("large publication %d failed: %v", order, err)
				publications <- -1
				return
			}
			count++
		}
	}()
	publishing := measure("publications running")
	close(stop)
	if count := <-publications; count < 1 {
		t.Fatalf("the concurrent publication loop committed %d publications", count)
	} else {
		t.Logf("the concurrent publication loop committed %d generations of %d rows", count, searchLiveLargeGenerationRows)
	}
	t.Logf("page one phases with no writer: %s", baseline)
	t.Logf("page one phases with the writer lock acquired: %s", locked)
	t.Logf("page one phases with publications running: %s", publishing)
}

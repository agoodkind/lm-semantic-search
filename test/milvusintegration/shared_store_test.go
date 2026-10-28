//go:build milvusintegration

package milvusintegration

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"sort"
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
	sharedBaselineCollection = "shared_baseline"
	sharedLayoutCollection   = "shared_layout"
	sharedDimension          = 32
	sharedDistinctContents   = 300
	sharedRows               = 1200
	sharedConversations      = 40
	sharedWorkspaces         = 6
	sharedQueries            = 25
	sharedSearchLimit        = 10
	sharedScoreTolerance     = 1e-6
	sharedVisibilityWait     = 3 * time.Second
	sharedSeed               = 17
)

var sharedProviders = []string{"claude", "codex", "cursor", "zed"}

var sharedDeclaration = collection.Declaration{
	ItemIDColumn: "conversationId",
	Scalars: []collection.ScalarColumn{
		{Name: "conversationId", Type: collection.ScalarTypeString, Nullable: false, MaxLength: 128},
		{Name: "provider", Type: collection.ScalarTypeString, Nullable: false, MaxLength: 32},
		{Name: "workspaceRoot", Type: collection.ScalarTypeString, Nullable: false, MaxLength: 128},
		{Name: "archived", Type: collection.ScalarTypeBool, Nullable: false, MaxLength: 0},
		{Name: "timestampUnix", Type: collection.ScalarTypeInt64, Nullable: false, MaxLength: 0},
		{Name: "label", Type: collection.ScalarTypeString, Nullable: true, MaxLength: 64},
	},
}

// sharedCorpus is generated conversation rows. Some rows repeat content and
// vector bytes, and some repeat content with different vector bytes.
type sharedCorpus struct {
	rows      []collection.Row
	contents  []string
	vectors   [][]float32
	variantOf map[int][]float32
}

func normalizedVector(generator *rand.Rand) []float32 {
	vector := make([]float32, sharedDimension)
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

func buildSharedCorpus() sharedCorpus {
	generator := rand.New(rand.NewPCG(sharedSeed, sharedSeed))
	words := []string{"index", "segment", "query", "vector", "restart", "collection", "worker", "flush", "compaction", "drop", "schema", "filter", "ranking", "storage", "backup"}
	corpus := sharedCorpus{variantOf: map[int][]float32{}}
	for contentIndex := range sharedDistinctContents {
		picked := make([]string, 0, 8)
		for range 8 {
			picked = append(picked, words[generator.IntN(len(words))])
		}
		corpus.contents = append(corpus.contents, strings.Join(picked, " ")+fmt.Sprintf(" note %d", contentIndex))
		corpus.vectors = append(corpus.vectors, normalizedVector(generator))
		if contentIndex%25 == 0 {
			corpus.variantOf[contentIndex] = normalizedVector(generator)
		}
	}
	for rowIndex := range sharedRows {
		contentIndex := rowIndex
		if rowIndex >= sharedDistinctContents {
			contentIndex = int(float64(sharedDistinctContents) * math.Pow(generator.Float64(), 3))
		}
		vector := corpus.vectors[contentIndex]
		if variant, found := corpus.variantOf[contentIndex]; found && rowIndex%2 == 1 {
			vector = variant
		}
		conversation := fmt.Sprintf("conv-%03d", generator.IntN(sharedConversations))
		label := collection.ScalarValue{Type: collection.ScalarTypeString, Null: true}
		corpus.rows = append(corpus.rows, collection.Row{
			ID:                fmt.Sprintf("row-%05d", rowIndex),
			Content:           corpus.contents[contentIndex],
			RelativePath:      fmt.Sprintf("conv/%s/%05d", conversation, rowIndex),
			StartLine:         int32(rowIndex),
			EndLine:           int32(rowIndex),
			FileExtension:     ".md",
			Metadata:          `{"kind":"message"}`,
			SplitPart:         0,
			SplitPartRecorded: false,
			Vector:            slices.Clone(vector),
			Scalars: map[string]collection.ScalarValue{
				"conversationId": collection.StringScalar(conversation),
				"provider":       collection.StringScalar(sharedProviders[generator.IntN(len(sharedProviders))]),
				"workspaceRoot":  collection.StringScalar(fmt.Sprintf("/ws/%d", generator.IntN(sharedWorkspaces))),
				"archived":       collection.BoolScalar(generator.IntN(10) == 0),
				"timestampUnix":  collection.Int64Scalar(1_780_000_000 + int64(generator.IntN(1_000_000))),
				"label":          label,
			},
		})
	}
	return corpus
}

func distinctVectorKeys(rows []collection.Row) map[string]struct{} {
	keys := map[string]struct{}{}
	for _, row := range rows {
		keys[milvusstore.VectorKey(row.Vector)] = struct{}{}
	}
	return keys
}

func sharedFilters() []struct {
	name   string
	filter *collection.Filter
} {
	lower := int64(1_780_200_000)
	upper := int64(1_780_500_000)
	providerFilter := collection.ColumnEquals("provider", collection.StringScalar("codex"))
	workspaceFilter := collection.AllOf(
		collection.ColumnEquals("workspaceRoot", collection.StringScalar("/ws/2")),
		collection.ColumnEquals("archived", collection.BoolScalar(false)))
	conversationFilter := collection.ColumnIn("conversationId", collection.StringValues([]string{"conv-003", "conv-017"}))
	timeFilter := collection.ColumnRange("timestampUnix", &lower, &upper)
	anyFilter := collection.AnyOf(
		collection.ColumnEquals("provider", collection.StringScalar("zed")),
		collection.ColumnEquals("conversationId", collection.StringScalar("conv-005")))
	notFilter := collection.Negate(collection.ColumnEquals("provider", collection.StringScalar("claude")))
	return []struct {
		name   string
		filter *collection.Filter
	}{
		{name: "none", filter: nil},
		{name: "provider", filter: &providerFilter},
		{name: "workspace_active", filter: &workspaceFilter},
		{name: "conversation_in", filter: &conversationFilter},
		{name: "time_range", filter: &timeFilter},
		{name: "any", filter: &anyFilter},
		{name: "not", filter: &notFilter},
	}
}

// rankedLegRow is one row of a full exact ranking leg of the baseline
// collection.
type rankedLegRow struct {
	id           string
	relativePath string
	vectorKey    string
	score        float32
}

// baselineLeg runs one ranking leg on the baseline collection over every row
// the filter matches, orders rows with equal scores by vector key (dense only)
// and then primary key, and cuts the leg at depth.
func baselineLeg(t *testing.T, client *milvusclient.Client, label string, baselineName string, field string, query entity.Vector, expression string, vectorKeyByID map[string]string, dense bool, depth int) []rankedLegRow {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), milvusCallTimeout)
	defer cancel()
	option := milvusclient.NewSearchOption(baselineName, collection.RankingDepth, []entity.Vector{query}).
		WithANNSField(field).
		WithOutputFields(milvusstore.RelativePathField).
		WithConsistencyLevel(entity.ClStrong)
	if expression != "" {
		option = option.WithFilter(expression)
	}
	resultSets, err := client.Search(ctx, option)
	if err != nil {
		t.Fatalf("%s: baseline %s leg: %v", label, field, err)
	}
	rows := make([]rankedLegRow, 0)
	if len(resultSets) > 0 {
		pathColumn := resultSets[0].GetColumn(milvusstore.RelativePathField)
		for index := range resultSets[0].ResultCount {
			id, idErr := resultSets[0].IDs.GetAsString(index)
			if idErr != nil {
				t.Fatalf("%s: read baseline %s leg id: %v", label, field, idErr)
			}
			relativePath, pathErr := pathColumn.GetAsString(index)
			if pathErr != nil {
				t.Fatalf("%s: read baseline %s leg path: %v", label, field, pathErr)
			}
			rows = append(rows, rankedLegRow{id: id, relativePath: relativePath, vectorKey: vectorKeyByID[id], score: resultSets[0].Scores[index]})
		}
	}
	sort.SliceStable(rows, func(left int, right int) bool {
		if rows[left].score != rows[right].score {
			return rows[left].score > rows[right].score
		}
		if dense && rows[left].vectorKey != rows[right].vectorKey {
			return rows[left].vectorKey < rows[right].vectorKey
		}
		return rows[left].id < rows[right].id
	})
	if len(rows) > depth {
		rows = rows[:depth]
	}
	return rows
}

// expectedHits fuses the cut baseline legs with RRF, 1 / (60 + rank) summed
// over legs and rounded to float32, and orders the fused rows by score,
// relativePath, and primary key, the order [milvusstore.SelectCandidates] uses.
func expectedHits(t *testing.T, client *milvusclient.Client, label string, baselineName string, request collection.SearchRequest, vectorKeyByID map[string]string) []rankedLegRow {
	t.Helper()
	expression, err := collection.CompileInline(request.Filter)
	if err != nil {
		t.Fatalf("%s: compile filter: %v", label, err)
	}
	depth := min(max(int(request.Limit)*4, 64), collection.RankingDepth)
	dense := baselineLeg(t, client, label, baselineName, milvusstore.DenseVectorField, entity.FloatVector(request.Vector), expression, vectorKeyByID, true, depth)
	sparse := baselineLeg(t, client, label, baselineName, milvusstore.SparseVectorField, entity.Text(request.Query), expression, vectorKeyByID, false, depth)
	scores := map[string]float64{}
	rows := map[string]rankedLegRow{}
	for _, leg := range [][]rankedLegRow{dense, sparse} {
		for position, row := range leg {
			scores[row.id] += 1.0 / float64(60+position+1)
			rows[row.id] = row
		}
	}
	fused := make([]rankedLegRow, 0, len(scores))
	for id, score := range scores {
		row := rows[id]
		row.score = float32(score)
		fused = append(fused, row)
	}
	sort.Slice(fused, func(left int, right int) bool {
		if fused[left].score != fused[right].score {
			return fused[left].score > fused[right].score
		}
		if fused[left].relativePath != fused[right].relativePath {
			return fused[left].relativePath < fused[right].relativePath
		}
		return fused[left].id < fused[right].id
	})
	if len(fused) > int(request.Limit) {
		fused = fused[:request.Limit]
	}
	return fused
}

// requireExpectedHits checks that the shared hits equal the expected hits row
// for row and score for score. It returns the number of positions where the
// baseline store returned another row than the expected one.
func requireExpectedHits(t *testing.T, client *milvusclient.Client, label string, baselineName string, request collection.SearchRequest, vectorKeyByID map[string]string, baseline []collection.Hit, shared []collection.Hit) int {
	t.Helper()
	expected := expectedHits(t, client, label, baselineName, request, vectorKeyByID)
	if len(shared) != len(expected) {
		t.Fatalf("%s: shared returned %d hits, expected %d", label, len(shared), len(expected))
	}
	for position, want := range expected {
		if shared[position].ID != want.id || math.Abs(shared[position].Score-float64(want.score)) > sharedScoreTolerance {
			t.Fatalf("%s: position %d is %s with score %.9f in shared, expected %s with score %.9f", label, position, shared[position].ID, shared[position].Score, want.id, want.score)
		}
	}
	differences := 0
	for position, want := range expected {
		if position >= len(baseline) || baseline[position].ID != want.id {
			differences++
		}
	}
	return differences
}

// requireVectorReferencesConsistent checks that every vector key an occurrence
// row stores exists in the vector collection and that every vector row has at
// least one occurrence.
func requireVectorReferencesConsistent(t *testing.T, client *milvusclient.Client, stage string, sharedName string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), milvusCallTimeout)
	defer cancel()
	referenced := map[string][]string{}
	occurrences, err := client.Query(ctx, milvusclient.NewQueryOption(sharedName).
		WithFilter(milvusstore.IDField+` != ""`).
		WithOutputFields(milvusstore.IDField, milvusstore.VectorKeyField).
		WithLimit(16384).
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		t.Fatalf("%s: read occurrence vector keys: %v", stage, err)
	}
	for index := range occurrences.ResultCount {
		id, _ := occurrences.GetColumn(milvusstore.IDField).GetAsString(index)
		key, _ := occurrences.GetColumn(milvusstore.VectorKeyField).GetAsString(index)
		referenced[key] = append(referenced[key], id)
	}
	vectors, err := client.Query(ctx, milvusclient.NewQueryOption(milvusstore.VectorCollectionName(sharedName)).
		WithFilter(milvusstore.VectorKeyField+` != ""`).
		WithOutputFields(milvusstore.VectorKeyField).
		WithLimit(16384).
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		t.Fatalf("%s: read vector keys: %v", stage, err)
	}
	stored := map[string]struct{}{}
	for index := range vectors.ResultCount {
		key, _ := vectors.GetColumn(milvusstore.VectorKeyField).GetAsString(index)
		stored[key] = struct{}{}
	}
	for key, ids := range referenced {
		if _, found := stored[key]; !found {
			t.Fatalf("%s: occurrences %v reference vector %s, which the vector collection lacks", stage, ids, key)
		}
	}
	for key := range stored {
		if _, found := referenced[key]; !found {
			t.Fatalf("%s: vector %s has no occurrence", stage, key)
		}
	}
}

func vectorKeysByID(rows []collection.Row) map[string]string {
	keys := make(map[string]string, len(rows))
	for _, row := range rows {
		keys[row.ID] = milvusstore.VectorKey(row.Vector)
	}
	return keys
}

func countRows(t *testing.T, client *milvusclient.Client, collectionName string) int64 {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), milvusCallTimeout)
	defer cancel()
	resultSet, err := client.Query(ctx, milvusclient.NewQueryOption(collectionName).
		WithFilter("").
		WithOutputFields(milvusstore.CountOutputField).
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		t.Fatalf("count rows of %s: %v", collectionName, err)
	}
	count, err := resultSet.GetColumn(milvusstore.CountOutputField).GetAsInt64(0)
	if err != nil {
		t.Fatalf("read row count of %s: %v", collectionName, err)
	}
	return count
}

func loadSharedCollections(t *testing.T, client *milvusclient.Client, names ...string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), milvusCallTimeout)
	defer cancel()
	for _, name := range names {
		task, err := client.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(name))
		if err != nil {
			t.Fatalf("load %s: %v", name, err)
		}
		if err := task.Await(ctx); err != nil {
			t.Fatalf("await load of %s: %v", name, err)
		}
	}
}

// searchTargets is the single-collection baseline and the shared occurrence
// collection one search comparison reads.
type searchTargets struct {
	baseline string
	shared   string
}

var defaultTargets = searchTargets{baseline: sharedBaselineCollection, shared: sharedLayoutCollection}

func compareSearches(t *testing.T, client *milvusclient.Client, stage string, targets searchTargets, baseline *milvusstore.Store, shared *milvusstore.SharedStore, corpus sharedCorpus, vectorKeyByID map[string]string) {
	t.Helper()
	ctx := context.Background()
	generator := rand.New(rand.NewPCG(sharedSeed+1, sharedSeed+1))
	baselineDifferences := 0
	checkedPositions := 0
	for _, filterCase := range sharedFilters() {
		for queryIndex := range sharedQueries {
			contentIndex := generator.IntN(len(corpus.contents))
			query := corpus.contents[contentIndex]
			vector := slices.Clone(corpus.vectors[contentIndex])
			for position := range vector {
				vector[position] += float32(generator.NormFloat64() * 0.05)
			}
			request := collection.SearchRequest{
				Collection:  targets.baseline,
				Query:       query,
				Vector:      vector,
				Limit:       sharedSearchLimit,
				Filter:      filterCase.filter,
				Declaration: sharedDeclaration,
			}
			baselineHits, err := baseline.Search(ctx, request)
			if err != nil {
				t.Fatalf("%s %s query %d: baseline search: %v", stage, filterCase.name, queryIndex, err)
			}
			request.Collection = targets.shared
			sharedHits, err := shared.Search(ctx, request)
			if err != nil {
				t.Fatalf("%s %s query %d: shared search: %v", stage, filterCase.name, queryIndex, err)
			}
			label := fmt.Sprintf("%s %s query %d", stage, filterCase.name, queryIndex)
			baselineDifferences += requireExpectedHits(t, client, label, targets.baseline, request, vectorKeyByID, baselineHits, sharedHits)
			checkedPositions += len(sharedHits)
		}
	}
	t.Logf("%s: shared equals the expected ranking at all %d positions; the single-collection store returned another row at %d of them", stage, checkedPositions, baselineDifferences)
}

// TestSharedStoreMatchesSingleCollectionStore writes the same generated rows to
// a single-collection store and a shared vector store on a throwaway Milvus
// with exact FLAT indexes. It checks that every filtered and unfiltered search
// returns the same scores and rows, that stored vectors keep their exact bytes,
// that the vector collection stores each distinct vector once, and that
// deletes, replacements, and a backfill keep both stores equal.
func TestSharedStoreMatchesSingleCollectionStore(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, "4g")
	client := dialMilvus(t, stack)
	ctx := context.Background()
	options := milvusstore.Options{Hybrid: true, EmbeddingModel: "generated-model", DenseIndex: index.NewFlatIndex(entity.COSINE)}
	baseline := milvusstore.New(client, options)
	shared, err := milvusstore.NewShared(client, options)
	if err != nil {
		t.Fatalf("new shared store: %v", err)
	}
	corpus := buildSharedCorpus()
	for _, ensure := range []struct {
		store collection.Store
		name  string
	}{{store: baseline, name: sharedBaselineCollection}, {store: shared, name: sharedLayoutCollection}} {
		if err := ensure.store.EnsureCollection(ctx, collection.EnsureRequest{Collection: ensure.name, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
			t.Fatalf("ensure %s: %v", ensure.name, err)
		}
	}
	vectorCollection := milvusstore.VectorCollectionName(sharedLayoutCollection)
	loadSharedCollections(t, client, sharedBaselineCollection, sharedLayoutCollection, vectorCollection)
	for start := 0; start < len(corpus.rows); start += 200 {
		batch := corpus.rows[start:min(start+200, len(corpus.rows))]
		if err := baseline.Upsert(ctx, sharedBaselineCollection, sharedDeclaration, batch); err != nil {
			t.Fatalf("baseline upsert: %v", err)
		}
		if err := shared.Upsert(ctx, sharedLayoutCollection, sharedDeclaration, batch); err != nil {
			t.Fatalf("shared upsert: %v", err)
		}
	}
	time.Sleep(sharedVisibilityWait)

	wantVectors := int64(len(distinctVectorKeys(corpus.rows)))
	if got := countRows(t, client, vectorCollection); got != wantVectors {
		t.Fatalf("vector collection has %d rows, want %d distinct vectors", got, wantVectors)
	}
	if got := countRows(t, client, sharedLayoutCollection); got != int64(len(corpus.rows)) {
		t.Fatalf("occurrence collection has %d rows, want %d", got, len(corpus.rows))
	}
	t.Logf("stored %d occurrences with %d distinct vectors", len(corpus.rows), wantVectors)

	itemIDs := make([]string, 0, sharedConversations)
	for conversation := range sharedConversations {
		itemIDs = append(itemIDs, fmt.Sprintf("conv-%03d", conversation))
	}
	storedRows, err := shared.QueryRows(ctx, collection.RowsRequest{Collection: sharedLayoutCollection, Declaration: sharedDeclaration, ItemIDs: itemIDs, IncludeVector: true})
	if err != nil {
		t.Fatalf("shared query rows: %v", err)
	}
	if len(storedRows) != len(corpus.rows) {
		t.Fatalf("shared query rows returned %d rows, want %d", len(storedRows), len(corpus.rows))
	}
	wantVectorByID := map[string]string{}
	for _, row := range corpus.rows {
		wantVectorByID[row.ID] = milvusstore.VectorKey(row.Vector)
	}
	for _, row := range storedRows {
		if milvusstore.VectorKey(row.Vector) != wantVectorByID[row.ID] {
			t.Fatalf("row %s vector bytes changed", row.ID)
		}
	}

	requireVectorReferencesConsistent(t, client, "initial", sharedLayoutCollection)
	compareSearches(t, client, "initial", defaultTargets, baseline, shared, corpus, wantVectorByID)

	deleteRequest := collection.DeleteItemsRequest{Declaration: sharedDeclaration, ItemIDs: []string{"conv-001", "conv-002", "conv-003"}}
	deleteRequest.Collection = sharedBaselineCollection
	baselineRemoved, err := baseline.DeleteItems(ctx, deleteRequest)
	if err != nil {
		t.Fatalf("baseline delete items: %v", err)
	}
	deleteRequest.Collection = sharedLayoutCollection
	sharedRemoved, err := shared.DeleteItems(ctx, deleteRequest)
	if err != nil {
		t.Fatalf("shared delete items: %v", err)
	}
	if baselineRemoved != sharedRemoved {
		t.Fatalf("delete removed %d baseline rows and %d shared rows", baselineRemoved, sharedRemoved)
	}
	remaining := make([]collection.Row, 0, len(corpus.rows))
	for _, row := range corpus.rows {
		if !slices.Contains(deleteRequest.ItemIDs, row.Scalars["conversationId"].String) {
			remaining = append(remaining, row)
		}
	}
	requireVectorReferencesConsistent(t, client, "after delete", sharedLayoutCollection)

	replacements := make([]collection.Row, 0, 20)
	generator := rand.New(rand.NewPCG(sharedSeed+2, sharedSeed+2))
	for _, row := range remaining[:20] {
		replaced := row
		replaced.Vector = normalizedVector(generator)
		replaced.Scalars = map[string]collection.ScalarValue{}
		for name, value := range row.Scalars {
			replaced.Scalars[name] = value
		}
		replacements = append(replacements, replaced)
	}
	if err := baseline.Upsert(ctx, sharedBaselineCollection, sharedDeclaration, replacements); err != nil {
		t.Fatalf("baseline replacement upsert: %v", err)
	}
	if err := shared.Upsert(ctx, sharedLayoutCollection, sharedDeclaration, replacements); err != nil {
		t.Fatalf("shared replacement upsert: %v", err)
	}
	copy(remaining[:20], replacements)
	time.Sleep(sharedVisibilityWait)
	if got, want := countRows(t, client, vectorCollection), int64(len(distinctVectorKeys(remaining))); got != want {
		t.Fatalf("after delete and replacement the vector collection has %d rows, want %d", got, want)
	}
	requireVectorReferencesConsistent(t, client, "after replacement", sharedLayoutCollection)

	remainingVectorByID := vectorKeysByID(remaining)
	backfill := collection.ScalarBackfill{
		ItemColumn: "conversationId",
		Columns:    []collection.ScalarColumn{sharedDeclaration.Scalars[5]},
		Values: map[string]map[string]collection.ScalarValue{
			"conv-010": {"label": collection.StringScalar("reviewed")},
			"conv-011": {"label": collection.StringScalar("reviewed")},
		},
	}
	if _, _, err := baseline.BackfillScalars(ctx, sharedBaselineCollection, backfill); err != nil {
		t.Fatalf("backfill %s: %v", sharedBaselineCollection, err)
	}
	if _, _, err := shared.BackfillScalars(ctx, sharedLayoutCollection, backfill); err != nil {
		t.Fatalf("backfill %s: %v", sharedLayoutCollection, err)
	}
	requireVectorReferencesConsistent(t, client, "after backfill", sharedLayoutCollection)
	time.Sleep(sharedVisibilityWait)
	labelFilter := collection.ColumnEquals("label", collection.StringScalar("reviewed"))
	for queryIndex := range sharedQueries {
		contentIndex := generator.IntN(len(corpus.contents))
		request := collection.SearchRequest{Collection: sharedBaselineCollection, Query: corpus.contents[contentIndex], Vector: corpus.vectors[contentIndex], Limit: sharedSearchLimit, Filter: &labelFilter, Declaration: sharedDeclaration}
		baselineHits, err := baseline.Search(ctx, request)
		if err != nil {
			t.Fatalf("baseline label search: %v", err)
		}
		request.Collection = sharedLayoutCollection
		sharedHits, err := shared.Search(ctx, request)
		if err != nil {
			t.Fatalf("shared label search: %v", err)
		}
		requireExpectedHits(t, client, fmt.Sprintf("backfill label query %d", queryIndex), sharedBaselineCollection, request, remainingVectorByID, baselineHits, sharedHits)
	}
	compareSearches(t, client, "after delete, replacement, and backfill", defaultTargets, baseline, shared, corpus, remainingVectorByID)
}

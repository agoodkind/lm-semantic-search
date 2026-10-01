//go:build live

package live

import (
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/model"
	"google.golang.org/protobuf/types/known/structpb"
)

const (
	paritySearchQuery        = "needle"
	parityBulkItems          = 300
	parityDenseRows          = 30
	parityLegacyItem         = "source-a:legacy"
	parityDenseItem          = "source-a:dense"
	parityFullLimit          = 1000
	parityPageLimit          = 10
	parityGroupLimit         = 2
	parityBatchSize          = 256
	parityVisibilityTimeout  = 60 * time.Second
	parityVisibilityInterval = 500 * time.Millisecond
	parityBaseCreated        = 1_700_000_000
	parityDenseCreated       = 1_600_000_000
	parityLegacyCreated      = 1_500_000_000
	parityLegacyRows         = 3
)

type searchParityRow struct {
	rowKey   string
	itemID   string
	source   string
	category string
	created  int64
	sequence int64
	parent   *string
	location *string
	hidden   *bool
	tag      *string
}

type parityCorpus struct {
	items   map[string][]*pb.CollectionRow
	rows    []searchParityRow
	bulkIDs []string
}

func stringPointer(value string) *string {
	return &value
}

func boolPointer(value bool) *bool {
	return &value
}

func bulkItemID(index int) string {
	source := "source-a"
	if index%2 == 1 {
		source = "source-b"
	}
	return fmt.Sprintf("%s:bulk-%03d", source, index)
}

func buildParityCorpus() parityCorpus {
	corpus := parityCorpus{items: map[string][]*pb.CollectionRow{}, rows: nil, bulkIDs: nil}
	for index := range parityBulkItems {
		itemID := bulkItemID(index)
		corpus.bulkIDs = append(corpus.bulkIDs, itemID)
		source, _, _ := strings.Cut(itemID, ":")
		location := fmt.Sprintf("/work/w%d", index%3)
		hidden := index%4 == 0
		parent := ""
		if index%5 == 0 && index != 0 {
			parent = bulkItemID(0)
		}
		tag := ""
		if index%2 == 0 {
			tag = "tag-v2"
		}
		categories := []string{"Primary", "secondary"}
		if index%3 == 0 {
			categories[0] = "primary"
		}
		for sequence, category := range categories {
			timestamp := int64(parityBaseCreated + index*10 + sequence)
			corpus.rows = append(corpus.rows, searchParityRow{
				rowKey:   fmt.Sprintf("items/%s/%d", itemID, sequence),
				itemID:   itemID,
				source:   source,
				category: strings.ToLower(category),
				created:  timestamp,
				sequence: int64(sequence),
				parent:   stringPointer(parent),
				location: stringPointer(location),
				hidden:   boolPointer(hidden),
				tag:      stringPointer(tag),
			})
		}
	}
	for sequence := range parityDenseRows {
		category := "secondary"
		if sequence%2 == 0 {
			category = "primary"
		}
		timestamp := int64(parityDenseCreated + sequence)
		corpus.rows = append(corpus.rows, searchParityRow{
			rowKey:   fmt.Sprintf("items/%s/%d", parityDenseItem, sequence),
			itemID:   parityDenseItem,
			source:   "source-a",
			category: category,
			created:  timestamp,
			sequence: int64(sequence),
			parent:   stringPointer(""),
			location: stringPointer("/work/dense"),
			hidden:   boolPointer(false),
			tag:      stringPointer("tag-dense"),
		})
	}
	for sequence := range parityLegacyRows {
		corpus.rows = append(corpus.rows, searchParityRow{
			rowKey:   fmt.Sprintf("items/%s/%d", parityLegacyItem, sequence),
			itemID:   parityLegacyItem,
			source:   "source-a",
			category: "primary",
			created:  int64(parityLegacyCreated + sequence),
			sequence: int64(sequence),
			parent:   nil,
			location: nil,
			hidden:   nil,
			tag:      nil,
		})
	}

	for _, row := range corpus.rows {
		text := fmt.Sprintf("bulk item %s entry %d covers design note ingestion step", row.itemID, row.sequence)
		if row.itemID == parityDenseItem {
			text = paritySearchQuery
		}
		scalars := []*pb.CollectionScalarValue{
			{Column: "source", Value: &pb.CollectionScalarValue_StringValue{StringValue: row.source}},
			{Column: "category", Value: &pb.CollectionScalarValue_StringValue{StringValue: row.category}},
			{Column: "created", Value: &pb.CollectionScalarValue_Int64Value{Int64Value: row.created}},
			{Column: "sequence", Value: &pb.CollectionScalarValue_Int64Value{Int64Value: row.sequence}},
		}
		for name, value := range map[string]*string{"parentId": row.parent, "location": row.location, "tag": row.tag} {
			if value != nil {
				scalars = append(scalars, &pb.CollectionScalarValue{Column: name, Value: &pb.CollectionScalarValue_StringValue{StringValue: *value}})
			}
		}
		if row.hidden != nil {
			scalars = append(scalars, &pb.CollectionScalarValue{Column: "hidden", Value: &pb.CollectionScalarValue_BoolValue{BoolValue: *row.hidden}})
		}
		corpus.items[row.itemID] = append(corpus.items[row.itemID], &pb.CollectionRow{RowKey: row.rowKey, ItemId: row.itemID, Text: text, Scalars: scalars})
	}
	return corpus
}

func (h *harness) genericSearch(request *pb.SearchCollectionRequest) []*pb.CollectionSearchHit {
	h.t.Helper()
	request.CollectionId = h.collectionID
	request.Query = paritySearchQuery
	response, err := h.client.SearchCollection(correlatedContext(), request)
	if err != nil {
		h.t.Fatalf("SearchCollection returned error: %v", err)
	}
	return response.GetHits()
}

func (h *harness) waitForCorpusVisibility(corpus parityCorpus) {
	h.t.Helper()
	deadline := time.Now().Add(parityVisibilityTimeout)
	visible := 0
	for time.Now().Before(deadline) {
		visible = len(h.genericSearch(&pb.SearchCollectionRequest{Limit: parityFullLimit}))
		if visible == len(corpus.rows) {
			return
		}
		time.Sleep(parityVisibilityInterval)
	}
	h.t.Fatalf("searchable rows = %d after %s, want %d", visible, parityVisibilityTimeout, len(corpus.rows))
}

func parityStringValue(value string) *pb.CollectionFilterValue {
	return &pb.CollectionFilterValue{Value: &pb.CollectionFilterValue_StringValue{StringValue: value}}
}

func parityBoolValue(value bool) *pb.CollectionFilterValue {
	return &pb.CollectionFilterValue{Value: &pb.CollectionFilterValue_BoolValue{BoolValue: value}}
}

func parityEquals(column string, value *pb.CollectionFilterValue) *pb.CollectionFilter {
	return &pb.CollectionFilter{Node: &pb.CollectionFilter_Equals{Equals: &pb.CollectionFilterEquals{Column: column, Value: value}}}
}

func parityIn(column string, values ...string) *pb.CollectionFilter {
	wireValues := make([]*pb.CollectionFilterValue, 0, len(values))
	for _, value := range values {
		wireValues = append(wireValues, parityStringValue(value))
	}
	return &pb.CollectionFilter{Node: &pb.CollectionFilter_InSet{InSet: &pb.CollectionFilterIn{Column: column, Values: wireValues}}}
}

func parityRange(column string, lower *int64, upper *int64) *pb.CollectionFilter {
	return &pb.CollectionFilter{Node: &pb.CollectionFilter_Range{Range: &pb.CollectionFilterRange{Column: column, Lower: lower, Upper: upper}}}
}

func parityAll(filters ...*pb.CollectionFilter) *pb.CollectionFilter {
	return &pb.CollectionFilter{Node: &pb.CollectionFilter_AllOf{AllOf: &pb.CollectionFilterGroup{Filters: filters}}}
}

func parityAny(filters ...*pb.CollectionFilter) *pb.CollectionFilter {
	return &pb.CollectionFilter{Node: &pb.CollectionFilter_AnyOf{AnyOf: &pb.CollectionFilterGroup{Filters: filters}}}
}

func parityNegate(filter *pb.CollectionFilter) *pb.CollectionFilter {
	return &pb.CollectionFilter{Node: &pb.CollectionFilter_Negate{Negate: filter}}
}

func parityIsNull(column string) *pb.CollectionFilter {
	return &pb.CollectionFilter{Node: &pb.CollectionFilter_IsNull{IsNull: &pb.CollectionFilterColumn{Column: column}}}
}

func parityIsPresent(column string) *pb.CollectionFilter {
	return &pb.CollectionFilter{Node: &pb.CollectionFilter_IsPresent{IsPresent: &pb.CollectionFilterColumn{Column: column}}}
}

func parityBound(value int64) *int64 {
	return &value
}

func expectedRowKeys(corpus parityCorpus, keep func(searchParityRow) bool) []string {
	keys := make([]string, 0, len(corpus.rows))
	for _, row := range corpus.rows {
		if keep(row) {
			keys = append(keys, row.rowKey)
		}
	}
	slices.Sort(keys)
	return keys
}

func sortedKeysOfHits(hits []*pb.CollectionSearchHit) []string {
	keys := make([]string, 0, len(hits))
	for _, hit := range hits {
		keys = append(keys, hit.GetRowKey())
	}
	slices.Sort(keys)
	return keys
}

func groupOfRowKey(rowKey string) string {
	trimmed := strings.TrimPrefix(rowKey, "items/")
	separator := strings.LastIndex(trimmed, "/")
	if separator < 0 {
		return trimmed
	}
	return trimmed[:separator]
}

func requireSameRanking(t *testing.T, label string, got []*pb.CollectionSearchHit, want []*pb.CollectionSearchHit) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d hits, want %d", label, len(got), len(want))
	}
	for index := range want {
		if got[index].GetRowKey() != want[index].GetRowKey() || got[index].GetScore() != want[index].GetScore() {
			t.Fatalf("%s: rank %d is %s score %v, want %s score %v", label, index, got[index].GetRowKey(), got[index].GetScore(), want[index].GetRowKey(), want[index].GetScore())
		}
	}
}

func hitKeys(hits []*pb.CollectionSearchHit) []string {
	keys := make([]string, 0, len(hits))
	for _, hit := range hits {
		keys = append(keys, hit.GetRowKey())
	}
	return keys
}

func capHits(hits []*pb.CollectionSearchHit, perGroupLimit int, minScore float64, limit int) []*pb.CollectionSearchHit {
	kept := make([]*pb.CollectionSearchHit, 0, limit)
	perGroup := map[string]int{}
	for _, hit := range hits {
		if minScore > 0 && hit.GetScore() < minScore {
			continue
		}
		if perGroupLimit > 0 {
			group := groupOfRowKey(hit.GetRowKey())
			if perGroup[group] >= perGroupLimit {
				continue
			}
			perGroup[group]++
		}
		kept = append(kept, hit)
		if len(kept) >= limit {
			break
		}
	}
	return kept
}

func (h *harness) searchCallCount() int {
	return h.callRecorder.count("Search", h.collectionName) + h.callRecorder.count("HybridSearch", h.collectionName)
}

func (h *harness) storedScalarRows(rowKeys []string) map[string]map[string]*pb.CollectionHitScalar {
	h.t.Helper()
	declared := searchFixtureDeclaration().Scalars
	outputFields := []string{relativePathField}
	for _, declaredColumn := range declared {
		outputFields = append(outputFields, declaredColumn.Name)
	}
	stored := make(map[string]map[string]*pb.CollectionHitScalar, len(rowKeys))
	for start := 0; start < len(rowKeys); start += parityBatchSize {
		batch := rowKeys[start:min(start+parityBatchSize, len(rowKeys))]
		quoted := make([]string, 0, len(batch))
		for _, key := range batch {
			quoted = append(quoted, strconv.Quote(key))
		}
		resultSet, err := h.milvus.Query(correlatedContext(), milvusclient.NewQueryOption(h.collectionName).
			WithFilter(relativePathField+" in ["+strings.Join(quoted, ", ")+"]").
			WithOutputFields(outputFields...).
			WithConsistencyLevel(entity.ClStrong))
		if err != nil {
			h.t.Fatalf("query stored scalars: %v", err)
		}
		pathColumn := resultSet.GetColumn(relativePathField)
		for rowIndex := range resultSet.ResultCount {
			path, pathErr := pathColumn.GetAsString(rowIndex)
			if pathErr != nil {
				h.t.Fatalf("read stored relativePath %d: %v", rowIndex, pathErr)
			}
			cells := make(map[string]*pb.CollectionHitScalar, len(declared))
			for _, declaredColumn := range declared {
				cells[declaredColumn.Name] = h.storedCell(resultSet.GetColumn(declaredColumn.Name), declaredColumn.Name, rowIndex)
			}
			stored[path] = cells
		}
	}
	return stored
}

func (h *harness) storedCell(valueColumn column.Column, name string, rowIndex int) *pb.CollectionHitScalar {
	h.t.Helper()
	cell := &pb.CollectionHitScalar{Column: name}
	if valueColumn == nil {
		return cell
	}
	isNull, err := valueColumn.IsNull(rowIndex)
	if err != nil {
		h.t.Fatalf("read stored null state of %s: %v", name, err)
	}
	if isNull {
		cell.Value = &pb.CollectionHitScalar_NullValue{NullValue: structpb.NullValue_NULL_VALUE}
		return cell
	}
	switch valueColumn.Type() {
	case entity.FieldTypeBool:
		value, valueErr := valueColumn.GetAsBool(rowIndex)
		if valueErr != nil {
			h.t.Fatalf("read stored %s: %v", name, valueErr)
		}
		cell.Value = &pb.CollectionHitScalar_BoolValue{BoolValue: value}
	case entity.FieldTypeInt64:
		value, valueErr := valueColumn.GetAsInt64(rowIndex)
		if valueErr != nil {
			h.t.Fatalf("read stored %s: %v", name, valueErr)
		}
		cell.Value = &pb.CollectionHitScalar_Int64Value{Int64Value: value}
	default:
		value, valueErr := valueColumn.GetAsString(rowIndex)
		if valueErr != nil {
			h.t.Fatalf("read stored %s: %v", name, valueErr)
		}
		cell.Value = &pb.CollectionHitScalar_StringValue{StringValue: value}
	}
	return cell
}

func TestGenericCollectionSearchLive(t *testing.T) {
	h := newSearchFixtureHarness(t)
	corpus := buildParityCorpus()
	for _, itemID := range slices.Sorted(maps.Keys(corpus.items)) {
		requireCompleted(t, h.upsert(map[string][]*pb.CollectionRow{itemID: corpus.items[itemID]}, pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_RETAIN, false, false), "corpus item ingest")
	}
	h.waitForCorpusVisibility(corpus)

	t.Run("filters match the corpus", func(t *testing.T) {
		cases := []struct {
			name    string
			generic *pb.CollectionFilter
			keep    func(searchParityRow) bool
		}{
			{
				name:    "source",
				generic: parityAll(parityIn("source", "source-b")),
				keep:    func(row searchParityRow) bool { return row.source == "source-b" },
			},
			{
				name:    "category",
				generic: parityAll(parityIn("category", "primary")),
				keep:    func(row searchParityRow) bool { return row.category == "primary" },
			},
			{
				name:    "time bounds",
				generic: parityAll(parityRange("created", parityBound(parityBaseCreated+1000), nil), parityRange("created", nil, parityBound(parityBaseCreated+1500))),
				keep: func(row searchParityRow) bool {
					return row.created >= parityBaseCreated+1000 && row.created < parityBaseCreated+1500
				},
			},
			{
				name:    "entry index bounds",
				generic: parityAll(parityRange("sequence", parityBound(1), nil), parityRange("sequence", nil, parityBound(5))),
				keep:    func(row searchParityRow) bool { return row.sequence >= 1 && row.sequence < 5 },
			},
			{
				name:    "parent",
				generic: parityAll(parityEquals("parentId", parityStringValue(bulkItemID(0)))),
				keep:    func(row searchParityRow) bool { return row.parent != nil && *row.parent == bulkItemID(0) },
			},
			{
				name:    "location",
				generic: parityAll(parityIn("location", "/work/w1", "/work/dense")),
				keep: func(row searchParityRow) bool {
					return row.location != nil && (*row.location == "/work/w1" || *row.location == "/work/dense")
				},
			},
			{
				name:    "hidden true excludes null",
				generic: parityAll(parityEquals("hidden", parityBoolValue(true))),
				keep:    func(row searchParityRow) bool { return row.hidden != nil && *row.hidden },
			},
			{
				name:    "hidden false excludes null",
				generic: parityAll(parityEquals("hidden", parityBoolValue(false))),
				keep:    func(row searchParityRow) bool { return row.hidden != nil && !*row.hidden },
			},
			{
				name:    "more than 256 item ids",
				generic: parityAll(parityIn("itemId", corpus.bulkIDs...)),
				keep:    func(row searchParityRow) bool { return strings.Contains(row.itemID, ":bulk-") },
			},
		}
		for _, testCase := range cases {
			generic := h.genericSearch(&pb.SearchCollectionRequest{Limit: parityFullLimit, Filter: testCase.generic})
			if got, want := sortedKeysOfHits(generic), expectedRowKeys(corpus, testCase.keep); !slices.Equal(got, want) {
				t.Fatalf("%s: generic returned %d rows, want %d from the corpus definition", testCase.name, len(got), len(want))
			}
		}
	})

	t.Run("null tests and negation follow the declared null semantics", func(t *testing.T) {
		cases := []struct {
			name   string
			filter *pb.CollectionFilter
			keep   func(searchParityRow) bool
		}{
			{name: "is_null hidden", filter: parityIsNull("hidden"), keep: func(row searchParityRow) bool { return row.hidden == nil }},
			{
				name: "int64 and bool membership",
				filter: parityAll(
					&pb.CollectionFilter{Node: &pb.CollectionFilter_InSet{InSet: &pb.CollectionFilterIn{Column: "sequence", Values: []*pb.CollectionFilterValue{
						{Value: &pb.CollectionFilterValue_Int64Value{Int64Value: 1}},
						{Value: &pb.CollectionFilterValue_Int64Value{Int64Value: 2}},
					}}}},
					&pb.CollectionFilter{Node: &pb.CollectionFilter_InSet{InSet: &pb.CollectionFilterIn{Column: "hidden", Values: []*pb.CollectionFilterValue{parityBoolValue(true)}}}},
				),
				keep: func(row searchParityRow) bool {
					return (row.sequence == 1 || row.sequence == 2) && row.hidden != nil && *row.hidden
				},
			},
			{name: "is_present location", filter: parityIsPresent("location"), keep: func(row searchParityRow) bool { return row.location != nil }},
			{
				name:   "negated comparison excludes null",
				filter: parityNegate(parityEquals("hidden", parityBoolValue(true))),
				keep:   func(row searchParityRow) bool { return row.hidden != nil && !*row.hidden },
			},
			{
				name:   "negated group over null columns",
				filter: parityNegate(parityAny(parityEquals("hidden", parityBoolValue(true)), parityIn("tag", "tag-v2"))),
				keep: func(row searchParityRow) bool {
					return row.hidden != nil && !*row.hidden && row.tag != nil && *row.tag != "tag-v2"
				},
			},
		}
		for _, testCase := range cases {
			generic := h.genericSearch(&pb.SearchCollectionRequest{Limit: parityFullLimit, Filter: testCase.filter})
			if got, want := sortedKeysOfHits(generic), expectedRowKeys(corpus, testCase.keep); !slices.Equal(got, want) {
				t.Fatalf("%s: generic returned %d rows, want %d from the corpus definition", testCase.name, len(got), len(want))
			}
		}
	})

	t.Run("group cap fills past an overfilled top", func(t *testing.T) {
		full := h.genericSearch(&pb.SearchCollectionRequest{Limit: parityFullLimit})
		denseInTop := 0
		for _, hit := range full[:parityPageLimit] {
			if groupOfRowKey(hit.GetRowKey()) == parityDenseItem {
				denseInTop++
			}
		}
		if denseInTop <= parityGroupLimit {
			t.Fatalf("top %d ranks have %d dense rows, want more than the group limit %d", parityPageLimit, denseInTop, parityGroupLimit)
		}
		h.callRecorder.reset()
		capped := h.genericSearch(&pb.SearchCollectionRequest{Limit: parityPageLimit, GroupBy: "itemId", PerGroupLimit: parityGroupLimit})
		if calls := h.searchCallCount(); calls != 1 {
			t.Fatalf("capped search ran %d ranking searches, want 1", calls)
		}
		requireSameRanking(t, "group cap fill", capped, capHits(full, parityGroupLimit, 0, parityPageLimit))
	})

	t.Run("more than 256 ids rank in one search", func(t *testing.T) {
		scope := append(slices.Clone(corpus.bulkIDs), parityDenseItem)
		inScope := map[string]bool{}
		for _, itemID := range scope {
			inScope[itemID] = true
		}
		full := h.genericSearch(&pb.SearchCollectionRequest{Limit: parityFullLimit})
		scopedFull := make([]*pb.CollectionSearchHit, 0, len(full))
		for _, hit := range full {
			if inScope[groupOfRowKey(hit.GetRowKey())] {
				scopedFull = append(scopedFull, hit)
			}
		}
		h.callRecorder.reset()
		generic := h.genericSearch(&pb.SearchCollectionRequest{Limit: parityPageLimit, GroupBy: "itemId", PerGroupLimit: parityGroupLimit, Filter: parityAll(parityIn("itemId", scope...))})
		if calls := h.searchCallCount(); calls != 1 {
			t.Fatalf("scoped search ran %d ranking searches, want 1", calls)
		}
		requireSameRanking(t, "large scope", generic, capHits(scopedFull, parityGroupLimit, 0, parityPageLimit))
	})

	t.Run("rankings are stable and smaller limits are prefixes", func(t *testing.T) {
		full := hitKeys(h.genericSearch(&pb.SearchCollectionRequest{Limit: parityFullLimit}))
		for range 3 {
			if again := hitKeys(h.genericSearch(&pb.SearchCollectionRequest{Limit: parityFullLimit})); !slices.Equal(again, full) {
				t.Fatalf("repeated full ranking differs: %d rows versus %d", len(again), len(full))
			}
		}
		larger := h.genericSearch(&pb.SearchCollectionRequest{Limit: 2 * parityPageLimit, GroupBy: "itemId", PerGroupLimit: parityGroupLimit})
		for _, limit := range []int32{1, 3, 5, parityPageLimit, 15} {
			smaller := h.genericSearch(&pb.SearchCollectionRequest{Limit: limit, GroupBy: "itemId", PerGroupLimit: parityGroupLimit})
			if len(smaller) > len(larger) || !slices.Equal(hitKeys(smaller), hitKeys(larger)[:len(smaller)]) {
				t.Fatalf("limit %d rows %v are not a prefix of %v", limit, hitKeys(smaller), hitKeys(larger))
			}
		}
	})

	t.Run("score floor", func(t *testing.T) {
		full := h.genericSearch(&pb.SearchCollectionRequest{Limit: parityFullLimit})
		floor := full[len(full)/4].GetScore()
		floored := h.genericSearch(&pb.SearchCollectionRequest{Limit: parityFullLimit, MinScore: floor})
		requireSameRanking(t, "score floor", floored, capHits(full, 0, floor, parityFullLimit))
	})

	t.Run("stored rows match row keys and scalar echoes", func(t *testing.T) {
		full := h.genericSearch(&pb.SearchCollectionRequest{Limit: parityFullLimit})
		keys := make([]string, 0, len(full))
		for _, hit := range full {
			keys = append(keys, hit.GetRowKey())
		}
		stored := h.storedScalarRows(keys)
		nullEchoes := 0
		for _, hit := range full {
			storedCells, found := stored[hit.GetRowKey()]
			if !found {
				t.Fatalf("hit row key %s has no stored row with that relativePath", hit.GetRowKey())
			}
			for _, echoed := range hit.GetScalars() {
				want := storedCells[echoed.GetColumn()]
				if want == nil || echoed.String() != want.String() {
					t.Fatalf("row %s scalar %s echoes %v, stored %v", hit.GetRowKey(), echoed.GetColumn(), echoed, want)
				}
				if _, isNull := echoed.GetValue().(*pb.CollectionHitScalar_NullValue); isNull {
					nullEchoes++
				}
			}
		}
		if nullEchoes == 0 {
			t.Fatal("no hit echoed a null scalar, want the optional null values")
		}
	})

	t.Run("item fingerprints and scoped rows", func(t *testing.T) {
		for _, itemID := range []string{parityDenseItem, bulkItemID(7), parityLegacyItem} {
			state, err := h.client.GetCollectionItemState(correlatedContext(), &pb.GetCollectionItemStateRequest{CollectionId: h.collectionID, ItemId: itemID})
			if err != nil {
				t.Fatalf("GetCollectionItemState: %v", err)
			}
			if got := state.GetIndexedFingerprint(); got == "" || got != h.parityCheckpoint(h.codebaseID)[itemID] {
				t.Fatalf("item %s fingerprint = %q, want the stored manifest", itemID, got)
			}
			scoped := h.genericSearch(&pb.SearchCollectionRequest{Limit: parityFullLimit, Filter: parityAll(parityIn("itemId", itemID))})
			want := expectedRowKeys(corpus, func(row searchParityRow) bool { return row.itemID == itemID })
			if got := sortedKeysOfHits(scoped); !slices.Equal(got, want) {
				t.Fatalf("item %s rows = %v, want %v", itemID, got, want)
			}
		}
	})
}

func searchFixtureDeclaration() model.CollectionDeclaration {
	return model.CollectionDeclaration{ItemIDColumn: "itemId", Scalars: []model.ScalarColumn{
		{Name: "itemId", Type: model.ScalarTypeString, MaxLength: 512},
		{Name: "parentId", Type: model.ScalarTypeString, Nullable: true, MaxLength: 512},
		{Name: "category", Type: model.ScalarTypeString, MaxLength: 64},
		{Name: "source", Type: model.ScalarTypeString, MaxLength: 64},
		{Name: "location", Type: model.ScalarTypeString, Nullable: true, MaxLength: 512},
		{Name: "hidden", Type: model.ScalarTypeBool, Nullable: true},
		{Name: "created", Type: model.ScalarTypeInt64},
		{Name: "sequence", Type: model.ScalarTypeInt64},
		{Name: "tag", Type: model.ScalarTypeString, Nullable: true, MaxLength: 512},
	}}
}

func newSearchFixtureHarness(t *testing.T) *harness {
	h := newHarnessWithOptions(t, nil, 0, true, true)
	collectionID := "live-generic-search-" + randomID()
	declaration := searchFixtureDeclaration()
	scalars := make([]*pb.ScalarColumnDeclaration, 0, len(declaration.Scalars))
	for _, scalar := range declaration.Scalars {
		scalars = append(scalars, &pb.ScalarColumnDeclaration{Column: scalar.Name, Type: liveScalarType(scalar.Type), Nullable: scalar.Nullable, MaxLength: scalar.MaxLength})
	}
	response, err := h.client.RegisterCollection(correlatedContext(), &pb.RegisterCollectionRequest{CollectionId: collectionID, ItemIdColumn: declaration.ItemIDColumn, Scalars: scalars, Client: &pb.ClientInfo{Name: "live-harness"}})
	if err != nil {
		t.Fatalf("RegisterCollection: %v", err)
	}
	h.trackCollectionFamily(response.GetCollectionName())
	h.collectionID = collectionID
	h.collectionName = response.GetCollectionName()
	h.codebaseID = response.GetCodebaseId()
	return h
}

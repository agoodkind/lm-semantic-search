package daemon

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"
)

const (
	searchCorpusCollection = "search-corpus"
	searchCorpusQuery      = "needle"
	alphaItem              = "claude:alpha"
	betaItem               = "codex:beta"
	alphaFingerprint       = "fingerprint-alpha"
	betaFingerprint        = "fingerprint-beta"
	alphaTag               = "rules-v1"
	searchCorpusLimit      = 50
	exactMatchScoreFloor   = 0.9999
)

func (daemon *offlineCollectionDaemon) searchCollection(request *pb.SearchCollectionRequest) (*pb.SearchCollectionResponse, error) {
	return daemon.client.SearchCollection(grpcutil.WithCorrelation(context.Background()), request)
}

func (daemon *offlineCollectionDaemon) mustSearchCollection(request *pb.SearchCollectionRequest) []*pb.CollectionSearchHit {
	daemon.t.Helper()
	response, err := daemon.searchCollection(request)
	if err != nil {
		daemon.t.Fatalf("SearchCollection returned error: %v", err)
	}
	return response.GetHits()
}

func hitRowKeys(hits []*pb.CollectionSearchHit) []string {
	keys := make([]string, 0, len(hits))
	for _, hit := range hits {
		keys = append(keys, hit.GetRowKey())
	}
	return keys
}

func sortedHitRowKeys(hits []*pb.CollectionSearchHit) []string {
	keys := hitRowKeys(hits)
	slices.Sort(keys)
	return keys
}

func sortedRowKeys(keys ...string) []string {
	sorted := slices.Clone(keys)
	slices.Sort(sorted)
	return sorted
}

func stringValue(value string) *pb.CollectionFilterValue {
	return &pb.CollectionFilterValue{Value: &pb.CollectionFilterValue_StringValue{StringValue: value}}
}

func boolValue(value bool) *pb.CollectionFilterValue {
	return &pb.CollectionFilterValue{Value: &pb.CollectionFilterValue_BoolValue{BoolValue: value}}
}

func int64Value(value int64) *pb.CollectionFilterValue {
	return &pb.CollectionFilterValue{Value: &pb.CollectionFilterValue_Int64Value{Int64Value: value}}
}

func equalsFilter(column string, value *pb.CollectionFilterValue) *pb.CollectionFilter {
	return &pb.CollectionFilter{Node: &pb.CollectionFilter_Equals{Equals: &pb.CollectionFilterEquals{Column: column, Value: value}}}
}

func inFilter(column string, values ...*pb.CollectionFilterValue) *pb.CollectionFilter {
	return &pb.CollectionFilter{Node: &pb.CollectionFilter_InSet{InSet: &pb.CollectionFilterIn{Column: column, Values: values}}}
}

func rangeFilter(column string, lower *int64, upper *int64) *pb.CollectionFilter {
	return &pb.CollectionFilter{Node: &pb.CollectionFilter_Range{Range: &pb.CollectionFilterRange{Column: column, Lower: lower, Upper: upper}}}
}

func allOfFilter(filters ...*pb.CollectionFilter) *pb.CollectionFilter {
	return &pb.CollectionFilter{Node: &pb.CollectionFilter_AllOf{AllOf: &pb.CollectionFilterGroup{Filters: filters}}}
}

func anyOfFilter(filters ...*pb.CollectionFilter) *pb.CollectionFilter {
	return &pb.CollectionFilter{Node: &pb.CollectionFilter_AnyOf{AnyOf: &pb.CollectionFilterGroup{Filters: filters}}}
}

func negateFilter(filter *pb.CollectionFilter) *pb.CollectionFilter {
	return &pb.CollectionFilter{Node: &pb.CollectionFilter_Negate{Negate: filter}}
}

func isNullFilter(column string) *pb.CollectionFilter {
	return &pb.CollectionFilter{Node: &pb.CollectionFilter_IsNull{IsNull: &pb.CollectionFilterColumn{Column: column}}}
}

func isPresentFilter(column string) *pb.CollectionFilter {
	return &pb.CollectionFilter{Node: &pb.CollectionFilter_IsPresent{IsPresent: &pb.CollectionFilterColumn{Column: column}}}
}

func bound(value int64) *int64 {
	return &value
}

// TestSearchCollectionAppliesEachPredicate searches the corpus through the
// public RPC with one filter per case. Each case covers one leaf predicate or
// a nested boolean expression and requires the exact set of matching rows.
func TestSearchCollectionAppliesEachPredicate(t *testing.T) {
	t.Parallel()
	daemon := newSearchCorpusDaemon(t)

	alphaRows := []string{rowKey(alphaItem, 0), rowKey(alphaItem, 1), rowKey(alphaItem, 2)}
	betaRows := []string{rowKey(betaItem, 0), rowKey(betaItem, 1)}
	allRows := append(slices.Clone(alphaRows), betaRows...)
	cases := []struct {
		name   string
		filter *pb.CollectionFilter
		want   []string
	}{
		{name: "no filter", filter: nil, want: allRows},
		{name: "string equality", filter: equalsFilter("source", stringValue("codex")), want: betaRows},
		{name: "bool equality", filter: equalsFilter("hidden", boolValue(true)), want: betaRows},
		{name: "int64 equality", filter: equalsFilter("sequence", int64Value(1)), want: []string{rowKey(alphaItem, 1), rowKey(betaItem, 1)}},
		{name: "parent equality", filter: equalsFilter("parentId", stringValue(alphaItem)), want: betaRows},
		{name: "workspace membership", filter: inFilter("location", stringValue("/work/alpha"), stringValue("/work/missing")), want: alphaRows},
		{name: "lowercased role membership", filter: inFilter("category", stringValue("user")), want: []string{rowKey(alphaItem, 0), rowKey(betaItem, 0)}},
		{name: "item id membership", filter: inFilter("itemId", stringValue(betaItem)), want: betaRows},
		{name: "inclusive lower and exclusive upper time bound", filter: rangeFilter("created", bound(1010), bound(2000)), want: []string{rowKey(alphaItem, 1), rowKey(alphaItem, 2)}},
		{name: "exclusive upper message index bound", filter: rangeFilter("sequence", nil, bound(1)), want: []string{rowKey(alphaItem, 0), rowKey(betaItem, 0)}},
		{name: "inclusive lower bound only", filter: rangeFilter("created", bound(2010), nil), want: []string{rowKey(betaItem, 1)}},
		{name: "null test on concrete rows", filter: isNullFilter("location"), want: nil},
		{name: "presence of an empty string", filter: isPresentFilter("tag"), want: allRows},
		{name: "empty string equality", filter: equalsFilter("tag", stringValue("")), want: betaRows},
		{
			name: "nested all, any, and negate",
			filter: anyOfFilter(
				allOfFilter(equalsFilter("source", stringValue("claude")), rangeFilter("sequence", bound(2), nil)),
				negateFilter(inFilter("category", stringValue("assistant"))),
			),
			want: []string{rowKey(alphaItem, 2), rowKey(alphaItem, 0), rowKey(betaItem, 0)},
		},
		{
			name:   "negated group",
			filter: negateFilter(anyOfFilter(equalsFilter("hidden", boolValue(true)), equalsFilter("category", stringValue("user")))),
			want:   []string{rowKey(alphaItem, 1), rowKey(alphaItem, 2)},
		},
	}
	for _, testCase := range cases {
		hits := daemon.mustSearchCollection(&pb.SearchCollectionRequest{
			CollectionId: searchCorpusCollection,
			Query:        searchCorpusQuery,
			Limit:        searchCorpusLimit,
			Filter:       testCase.filter,
		})
		if got, want := sortedHitRowKeys(hits), sortedRowKeys(testCase.want...); !slices.Equal(got, want) {
			t.Fatalf("%s: row keys = %v, want %v", testCase.name, got, want)
		}
	}
}

// TestSearchCollectionRejectsInvalidRequests sends invalid filters and group
// columns through the public RPC. Each fails with InvalidArgument before the
// search runs. A column violation reports the rejected column in ErrorInfo.
func TestSearchCollectionRejectsInvalidRequests(t *testing.T) {
	t.Parallel()
	daemon := newSearchCorpusDaemon(t)

	deep := equalsFilter("source", stringValue("claude"))
	for range maxCollectionFilterDepth {
		deep = negateFilter(deep)
	}
	oversized := make([]*pb.CollectionFilterValue, 0, maxCollectionFilterValues+1)
	for range maxCollectionFilterValues + 1 {
		oversized = append(oversized, stringValue("x"))
	}
	columnCases := []struct {
		name       string
		request    *pb.SearchCollectionRequest
		wantColumn string
	}{
		{name: "unknown filter column", request: &pb.SearchCollectionRequest{Filter: equalsFilter("title", stringValue("x"))}, wantColumn: "title"},
		{name: "unknown nested column", request: &pb.SearchCollectionRequest{Filter: allOfFilter(equalsFilter("source", stringValue("claude")), negateFilter(isNullFilter("author")))}, wantColumn: "author"},
		{name: "string value for an int64 column", request: &pb.SearchCollectionRequest{Filter: equalsFilter("sequence", stringValue("1"))}, wantColumn: "sequence"},
		{name: "bool value for a string column", request: &pb.SearchCollectionRequest{Filter: inFilter("category", stringValue("user"), boolValue(true))}, wantColumn: "category"},
		{name: "range on a string column", request: &pb.SearchCollectionRequest{Filter: rangeFilter("category", bound(1), nil)}, wantColumn: "category"},
		{name: "range without a bound", request: &pb.SearchCollectionRequest{Filter: rangeFilter("created", nil, nil)}, wantColumn: "created"},
		{name: "empty membership set", request: &pb.SearchCollectionRequest{Filter: inFilter("source")}, wantColumn: "source"},
		{name: "oversized membership set", request: &pb.SearchCollectionRequest{Filter: inFilter("itemId", oversized...)}, wantColumn: "itemId"},
		{name: "literal without a value", request: &pb.SearchCollectionRequest{Filter: equalsFilter("source", &pb.CollectionFilterValue{})}, wantColumn: "source"},
		{name: "unknown group column", request: &pb.SearchCollectionRequest{GroupBy: "title", PerGroupLimit: 1}, wantColumn: "title"},
	}
	for _, testCase := range columnCases {
		testCase.request.CollectionId = searchCorpusCollection
		testCase.request.Query = searchCorpusQuery
		_, err := daemon.searchCollection(testCase.request)
		requireColumnError(t, err, codes.InvalidArgument, "invalid_argument", testCase.wantColumn)
	}

	argumentCases := []struct {
		name    string
		request *pb.SearchCollectionRequest
	}{
		{name: "excessive depth", request: &pb.SearchCollectionRequest{Filter: deep}},
		{name: "empty all group", request: &pb.SearchCollectionRequest{Filter: allOfFilter()}},
		{name: "empty any group", request: &pb.SearchCollectionRequest{Filter: anyOfFilter()}},
		{name: "node without a member", request: &pb.SearchCollectionRequest{Filter: &pb.CollectionFilter{}}},
		{name: "group limit without group column", request: &pb.SearchCollectionRequest{PerGroupLimit: 1}},
		{name: "negative group limit", request: &pb.SearchCollectionRequest{GroupBy: "itemId", PerGroupLimit: -1}},
	}
	for _, testCase := range argumentCases {
		testCase.request.CollectionId = searchCorpusCollection
		testCase.request.Query = searchCorpusQuery
		_, err := daemon.searchCollection(testCase.request)
		if status.Code(err) != codes.InvalidArgument {
			t.Fatalf("%s: error = %v, want InvalidArgument", testCase.name, err)
		}
	}

	_, err := daemon.searchCollection(&pb.SearchCollectionRequest{CollectionId: "never-registered", Query: searchCorpusQuery})
	if status.Code(err) != codes.NotFound {
		t.Fatalf("unregistered collection error = %v, want NotFound", err)
	}
	if records := daemon.documentCollectionRecords("never-registered"); len(records) != 0 {
		t.Fatalf("search registered %d records for an unknown collection, want 0", len(records))
	}
}

// TestSearchCollectionAppliesGroupCapAndScoreFloor proves the per-group cap
// and the score floor. The three alpha rows equal the query and take the first
// ranks. Without a cap the limit fills with alpha rows. A cap of one per
// item keeps one alpha row and fills the rest from beta. The score
// floor keeps only the exact matches.
func TestSearchCollectionAppliesGroupCapAndScoreFloor(t *testing.T) {
	t.Parallel()
	daemon := newSearchCorpusDaemon(t)

	uncapped := daemon.mustSearchCollection(&pb.SearchCollectionRequest{CollectionId: searchCorpusCollection, Query: searchCorpusQuery, Limit: 3})
	for _, hit := range uncapped {
		if !strings.HasPrefix(hit.GetRowKey(), "items/"+alphaItem+"/") {
			t.Fatalf("uncapped top 3 = %v, want only alpha rows", hitRowKeys(uncapped))
		}
	}

	capped := daemon.mustSearchCollection(&pb.SearchCollectionRequest{
		CollectionId:  searchCorpusCollection,
		Query:         searchCorpusQuery,
		Limit:         3,
		GroupBy:       "itemId",
		PerGroupLimit: 1,
	})
	if len(capped) != 2 {
		t.Fatalf("capped hits = %v, want one row per item", hitRowKeys(capped))
	}
	if !strings.HasPrefix(capped[0].GetRowKey(), "items/"+alphaItem+"/") || !strings.HasPrefix(capped[1].GetRowKey(), "items/"+betaItem+"/") {
		t.Fatalf("capped hits = %v, want the best alpha row then the best beta row", hitRowKeys(capped))
	}
	if capped[0].GetScore() < capped[1].GetScore() {
		t.Fatalf("capped scores %v then %v are not descending", capped[0].GetScore(), capped[1].GetScore())
	}

	larger := hitRowKeys(daemon.mustSearchCollection(&pb.SearchCollectionRequest{CollectionId: searchCorpusCollection, Query: searchCorpusQuery, Limit: searchCorpusLimit}))
	for limit := int32(1); limit <= 5; limit++ {
		smaller := hitRowKeys(daemon.mustSearchCollection(&pb.SearchCollectionRequest{CollectionId: searchCorpusCollection, Query: searchCorpusQuery, Limit: limit}))
		if len(smaller) > len(larger) || !slices.Equal(smaller, larger[:len(smaller)]) {
			t.Fatalf("limit %d rows %v are not a prefix of %v", limit, smaller, larger)
		}
	}

	floored := daemon.mustSearchCollection(&pb.SearchCollectionRequest{
		CollectionId: searchCorpusCollection,
		Query:        searchCorpusQuery,
		Limit:        searchCorpusLimit,
		MinScore:     exactMatchScoreFloor,
	})
	if got, want := sortedHitRowKeys(floored), sortedRowKeys(rowKey(alphaItem, 0), rowKey(alphaItem, 1), rowKey(alphaItem, 2)); !slices.Equal(got, want) {
		t.Fatalf("score floor kept %v, want only the exact matches %v", got, want)
	}
	for _, hit := range floored {
		if hit.GetScore() < exactMatchScoreFloor {
			t.Fatalf("hit %s scored %v, below the floor %v", hit.GetRowKey(), hit.GetScore(), exactMatchScoreFloor)
		}
	}
}

// TestSearchCollectionEchoesDeclaredScalars proves every hit lists every
// declared scalar column in declaration order with its stored value,
// including the load rules tag and the lowercased role.
func TestSearchCollectionEchoesDeclaredScalars(t *testing.T) {
	t.Parallel()
	daemon := newSearchCorpusDaemon(t)

	hits := daemon.mustSearchCollection(&pb.SearchCollectionRequest{
		CollectionId: searchCorpusCollection,
		Query:        searchCorpusQuery,
		Limit:        1,
		Filter:       allOfFilter(inFilter("itemId", stringValue(betaItem)), equalsFilter("sequence", int64Value(0))),
	})
	if len(hits) != 1 {
		t.Fatalf("hits = %v, want one beta row", hitRowKeys(hits))
	}
	hit := hits[0]
	if hit.GetRowKey() != rowKey(betaItem, 0) || hit.GetContent() != "how does the haystack index work" {
		t.Fatalf("hit = %s %q, want the first beta row", hit.GetRowKey(), hit.GetContent())
	}
	want := []*pb.CollectionHitScalar{
		{Column: "itemId", Value: &pb.CollectionHitScalar_StringValue{StringValue: betaItem}},
		{Column: "parentId", Value: &pb.CollectionHitScalar_StringValue{StringValue: alphaItem}},
		{Column: "category", Value: &pb.CollectionHitScalar_StringValue{StringValue: "user"}},
		{Column: "source", Value: &pb.CollectionHitScalar_StringValue{StringValue: "codex"}},
		{Column: "location", Value: &pb.CollectionHitScalar_StringValue{StringValue: "/work/beta"}},
		{Column: "hidden", Value: &pb.CollectionHitScalar_BoolValue{BoolValue: true}},
		{Column: "created", Value: &pb.CollectionHitScalar_Int64Value{Int64Value: 2000}},
		{Column: "sequence", Value: &pb.CollectionHitScalar_Int64Value{Int64Value: 0}},
		{Column: "tag", Value: &pb.CollectionHitScalar_StringValue{StringValue: ""}},
	}
	requireHitScalars(t, hit.GetScalars(), want)

	alphaHits := daemon.mustSearchCollection(&pb.SearchCollectionRequest{
		CollectionId: searchCorpusCollection,
		Query:        searchCorpusQuery,
		Limit:        1,
		Filter:       inFilter("itemId", stringValue(alphaItem)),
	})
	if len(alphaHits) != 1 {
		t.Fatalf("alpha hits = %v, want one", hitRowKeys(alphaHits))
	}
	tag := alphaHits[0].GetScalars()[len(want)-1]
	if tag.GetColumn() != "tag" || tag.GetStringValue() != alphaTag {
		t.Fatalf("alpha tag scalar = %v, want %q", tag, alphaTag)
	}
}

func requireHitScalars(t *testing.T, got []*pb.CollectionHitScalar, want []*pb.CollectionHitScalar) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("hit scalars = %v, want %v", got, want)
	}
	for index := range want {
		if got[index].GetColumn() != want[index].GetColumn() || got[index].GetValue() == nil {
			t.Fatalf("hit scalar %d = %v, want %v", index, got[index], want[index])
		}
		if got[index].String() != want[index].String() {
			t.Fatalf("hit scalar %d = %v, want %v", index, got[index], want[index])
		}
	}
}

// TestCollectionHitScalarsDistinguishNullFromAbsent proves the wire shape of
// the three cell states. The offline store has no null values, and the live
// battery proves null and absent values from real Milvus rows.
func TestCollectionHitScalarsDistinguishNullFromAbsent(t *testing.T) {
	t.Parallel()

	scalars := collectionHitScalarsToPB([]semantic.ScalarCell{
		semantic.NullCell("location"),
		semantic.AbsentCell("priority"),
		semantic.ValueCell("hidden", semantic.BoolScalar(false)),
	})
	if _, isNull := scalars[0].GetValue().(*pb.CollectionHitScalar_NullValue); !isNull || scalars[0].GetNullValue() != structpb.NullValue_NULL_VALUE {
		t.Fatalf("null cell = %v, want null_value", scalars[0])
	}
	if scalars[1].GetValue() != nil {
		t.Fatalf("absent cell = %v, want no value", scalars[1])
	}
	if _, isBool := scalars[2].GetValue().(*pb.CollectionHitScalar_BoolValue); !isBool || scalars[2].GetBoolValue() {
		t.Fatalf("bool cell = %v, want bool_value false", scalars[2])
	}
}

// TestGetCollectionItemStateReadsCheckpoint proves the item state RPC returns
// the checkpoint fingerprint for an indexed item and the empty fingerprint for
// an unknown item and an unregistered collection, without registering it.
func TestGetCollectionItemStateReadsCheckpoint(t *testing.T) {
	t.Parallel()
	daemon := newSearchCorpusDaemon(t)

	itemState := func(collectionID string, itemID string) string {
		t.Helper()
		response, err := daemon.client.GetCollectionItemState(grpcutil.WithCorrelation(context.Background()), &pb.GetCollectionItemStateRequest{CollectionId: collectionID, ItemId: itemID})
		if err != nil {
			t.Fatalf("GetCollectionItemState(%s, %s) returned error: %v", collectionID, itemID, err)
		}
		return response.GetIndexedFingerprint()
	}
	if got := itemState(searchCorpusCollection, alphaItem); got != alphaFingerprint {
		t.Fatalf("alpha fingerprint = %q, want %q", got, alphaFingerprint)
	}
	if got := itemState(searchCorpusCollection, "claude:missing"); got != "" {
		t.Fatalf("unknown item fingerprint = %q, want empty", got)
	}
	if got := itemState("never-registered", alphaItem); got != "" {
		t.Fatalf("unregistered collection fingerprint = %q, want empty", got)
	}
	if records := daemon.documentCollectionRecords("never-registered"); len(records) != 0 {
		t.Fatalf("item state registered %d records, want 0", len(records))
	}
	_, err := daemon.client.GetCollectionItemState(grpcutil.WithCorrelation(context.Background()), &pb.GetCollectionItemStateRequest{CollectionId: searchCorpusCollection})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("missing item id error = %v, want InvalidArgument", err)
	}
}

func requireMaintenanceStatus(t *testing.T, method string, err error) {
	t.Helper()
	grpcStatus, ok := status.FromError(err)
	if !ok || grpcStatus.Code() != codes.FailedPrecondition {
		t.Fatalf("%s during maintenance returned %v, want FailedPrecondition", method, err)
	}
	if !strings.Contains(grpcStatus.Message(), "daemon is in maintenance mode (backup)") {
		t.Fatalf("%s maintenance message = %q, want the maintenance refusal", method, grpcStatus.Message())
	}
	if len(grpcStatus.Details()) != 0 {
		t.Fatalf("%s maintenance status has details %v, want the unchanged status without details", method, grpcStatus.Details())
	}
}

func rowKey(itemID string, sequence int32) string {
	return fmt.Sprintf("items/%s/%d", itemID, sequence)
}

func newSearchCorpusDaemon(t *testing.T) *offlineCollectionDaemon {
	t.Helper()
	daemon := newOfflineCollectionDaemon(t)
	columns := []*pb.ScalarColumnDeclaration{
		{Column: "itemId", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_STRING, MaxLength: 256},
		{Column: "parentId", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_STRING, MaxLength: 256},
		{Column: "category", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_STRING, MaxLength: 64},
		{Column: "source", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_STRING, MaxLength: 32},
		{Column: "location", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_STRING, MaxLength: 1024},
		{Column: "hidden", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_BOOL},
		{Column: "created", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_INT64},
		{Column: "sequence", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_INT64},
		{Column: "tag", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_STRING, MaxLength: 256},
	}
	if _, err := daemon.registerCollection(searchCorpusCollection, "itemId", columns); err != nil {
		t.Fatalf("RegisterCollection: %v", err)
	}
	rows := make([]*pb.CollectionRow, 0, 5)
	appendRow := func(itemID string, sequence int32, category string, created int64, text string, parent string, location string, hidden bool, tag string, source string) {
		rows = append(rows, &pb.CollectionRow{RowKey: rowKey(itemID, sequence), ItemId: itemID, Text: text, Scalars: []*pb.CollectionScalarValue{
			stringScalar("itemId", itemID), stringScalar("parentId", parent), stringScalar("category", category), stringScalar("source", source), stringScalar("location", location), boolScalar("hidden", hidden), int64Scalar("created", created), int64Scalar("sequence", int64(sequence)), stringScalar("tag", tag),
		}})
	}
	appendRow(alphaItem, 0, "user", 1000, searchCorpusQuery, "", "/work/alpha", false, alphaTag, "claude")
	appendRow(alphaItem, 1, "assistant", 1010, searchCorpusQuery, "", "/work/alpha", false, alphaTag, "claude")
	appendRow(alphaItem, 2, "assistant", 1020, searchCorpusQuery, "", "/work/alpha", false, alphaTag, "claude")
	appendRow(betaItem, 0, "user", 2000, "how does the haystack index work", alphaItem, "/work/beta", true, "", "codex")
	appendRow(betaItem, 1, "assistant", 2010, "the index embeds every message", alphaItem, "/work/beta", true, "", "codex")
	daemon.upsertItems(collectionHeader(searchCorpusCollection, pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_RETAIN, false, false), rows, map[string]string{alphaItem: alphaFingerprint, betaItem: betaFingerprint})
	return daemon
}

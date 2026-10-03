package daemon

import (
	"context"
	"slices"
	"strings"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"
)

const (
	searchCorpusCollection = "search-corpus"
	searchCorpusQuery      = "needle"
	alphaConversation      = "claude:alpha"
	betaConversation       = "codex:beta"
	alphaFingerprint       = "fingerprint-alpha"
	betaFingerprint        = "fingerprint-beta"
	alphaLoadRules         = "rules-v1"
	searchCorpusLimit      = 50
	exactMatchScoreFloor   = 0.9999
)

// searchCorpusDocuments is the offline search corpus. Every alpha message text
// equals the query. The deterministic test embedder therefore scores all three
// alpha rows 1.0, and they fill the first ranks. The beta rows store the other
// scalar values: another provider and workspace, a parent, archived true, and
// an empty load rules tag.
func searchCorpusDocuments() []*pb.ConversationDocument {
	return []*pb.ConversationDocument{
		{ConversationId: alphaConversation, MessageIndex: 0, Role: "user", TimestampUnix: 1000, Text: searchCorpusQuery, WorkspaceRoot: "/work/alpha", Archived: false, LoadRules: alphaLoadRules},
		{ConversationId: alphaConversation, MessageIndex: 1, Role: "assistant", TimestampUnix: 1010, Text: searchCorpusQuery, WorkspaceRoot: "/work/alpha", Archived: false, LoadRules: alphaLoadRules},
		{ConversationId: alphaConversation, MessageIndex: 2, Role: "assistant", TimestampUnix: 1020, Text: searchCorpusQuery, WorkspaceRoot: "/work/alpha", Archived: false, LoadRules: alphaLoadRules},
		{ConversationId: betaConversation, ParentConversationId: alphaConversation, MessageIndex: 0, Role: "User", TimestampUnix: 2000, Text: "how does the haystack index work", WorkspaceRoot: "/work/beta", Archived: true},
		{ConversationId: betaConversation, ParentConversationId: alphaConversation, MessageIndex: 1, Role: "assistant", TimestampUnix: 2010, Text: "the index embeds every message", WorkspaceRoot: "/work/beta", Archived: true},
	}
}

// ingestDocuments streams documents and their manifest through the
// conversation upsert RPC and waits for the ingest job to complete.
func (daemon *offlineCollectionDaemon) ingestSearchDocuments(collectionID string, documents []*pb.ConversationDocument, manifest []*pb.ConversationFingerprint) {
	daemon.t.Helper()
	stream, err := daemon.client.UpsertConversationDocumentsStream(grpcutil.WithCorrelation(context.Background()))
	if err != nil {
		daemon.t.Fatalf("open UpsertConversationDocumentsStream returned error: %v", err)
	}
	chunks := []*pb.UpsertConversationDocumentsChunk{
		{Chunk: &pb.UpsertConversationDocumentsChunk_Header{Header: &pb.UpsertConversationDocumentsHeader{
			CollectionId: collectionID,
			Client:       &pb.ClientInfo{Name: "collection-search-test"},
		}}},
		{Chunk: &pb.UpsertConversationDocumentsChunk_Documents{Documents: &pb.UpsertConversationDocumentsDocuments{Documents: documents}}},
		{Chunk: &pb.UpsertConversationDocumentsChunk_Manifest{Manifest: &pb.UpsertConversationDocumentsManifest{Manifest: manifest}}},
	}
	for _, chunk := range chunks {
		if err := stream.Send(chunk); err != nil {
			daemon.t.Fatalf("send upsert chunk returned error: %v", err)
		}
	}
	response, err := stream.CloseAndRecv()
	if err != nil {
		daemon.t.Fatalf("CloseAndRecv returned error: %v", err)
	}
	job := waitForRPCJobTerminal(daemon.t, daemon.client, response.GetJobId())
	if job.GetState() != string(model.JobStateCompleted) {
		daemon.t.Fatalf("ingest job state = %q, want completed: %+v", job.GetState(), job.GetError())
	}
}

func newSearchCorpusDaemon(t *testing.T) *offlineCollectionDaemon {
	t.Helper()
	daemon := newOfflineCollectionDaemon(t)
	if _, err := daemon.registerCollection(searchCorpusCollection, "conversationId", conversationScalarsPB()); err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}
	daemon.ingestSearchDocuments(searchCorpusCollection, searchCorpusDocuments(), []*pb.ConversationFingerprint{
		{ConversationId: alphaConversation, Fingerprint: alphaFingerprint},
		{ConversationId: betaConversation, Fingerprint: betaFingerprint},
	})
	return daemon
}

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

func rowKey(conversationID string, messageIndex int32) string {
	return conversationRelativePath(conversationID, messageIndex, 0, false)
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

	alphaRows := []string{rowKey(alphaConversation, 0), rowKey(alphaConversation, 1), rowKey(alphaConversation, 2)}
	betaRows := []string{rowKey(betaConversation, 0), rowKey(betaConversation, 1)}
	allRows := append(slices.Clone(alphaRows), betaRows...)
	cases := []struct {
		name   string
		filter *pb.CollectionFilter
		want   []string
	}{
		{name: "no filter", filter: nil, want: allRows},
		{name: "string equality", filter: equalsFilter("provider", stringValue("codex")), want: betaRows},
		{name: "bool equality", filter: equalsFilter("archived", boolValue(true)), want: betaRows},
		{name: "int64 equality", filter: equalsFilter("messageIndex", int64Value(1)), want: []string{rowKey(alphaConversation, 1), rowKey(betaConversation, 1)}},
		{name: "parent equality", filter: equalsFilter("parentConversationId", stringValue(alphaConversation)), want: betaRows},
		{name: "workspace membership", filter: inFilter("workspaceRoot", stringValue("/work/alpha"), stringValue("/work/missing")), want: alphaRows},
		{name: "lowercased role membership", filter: inFilter("role", stringValue("user")), want: []string{rowKey(alphaConversation, 0), rowKey(betaConversation, 0)}},
		{name: "item id membership", filter: inFilter("conversationId", stringValue(betaConversation)), want: betaRows},
		{name: "inclusive lower and exclusive upper time bound", filter: rangeFilter("timestampUnix", bound(1010), bound(2000)), want: []string{rowKey(alphaConversation, 1), rowKey(alphaConversation, 2)}},
		{name: "exclusive upper message index bound", filter: rangeFilter("messageIndex", nil, bound(1)), want: []string{rowKey(alphaConversation, 0), rowKey(betaConversation, 0)}},
		{name: "inclusive lower bound only", filter: rangeFilter("timestampUnix", bound(2010), nil), want: []string{rowKey(betaConversation, 1)}},
		{name: "null test on concrete rows", filter: isNullFilter("workspaceRoot"), want: nil},
		{name: "presence of an empty string", filter: isPresentFilter("loadRules"), want: allRows},
		{name: "empty string equality", filter: equalsFilter("loadRules", stringValue("")), want: betaRows},
		{
			name: "nested all, any, and negate",
			filter: anyOfFilter(
				allOfFilter(equalsFilter("provider", stringValue("claude")), rangeFilter("messageIndex", bound(2), nil)),
				negateFilter(inFilter("role", stringValue("assistant"))),
			),
			want: []string{rowKey(alphaConversation, 2), rowKey(alphaConversation, 0), rowKey(betaConversation, 0)},
		},
		{
			name:   "negated group",
			filter: negateFilter(anyOfFilter(equalsFilter("archived", boolValue(true)), equalsFilter("role", stringValue("user")))),
			want:   []string{rowKey(alphaConversation, 1), rowKey(alphaConversation, 2)},
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

	deep := equalsFilter("provider", stringValue("claude"))
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
		{name: "unknown nested column", request: &pb.SearchCollectionRequest{Filter: allOfFilter(equalsFilter("provider", stringValue("claude")), negateFilter(isNullFilter("author")))}, wantColumn: "author"},
		{name: "string value for an int64 column", request: &pb.SearchCollectionRequest{Filter: equalsFilter("messageIndex", stringValue("1"))}, wantColumn: "messageIndex"},
		{name: "bool value for a string column", request: &pb.SearchCollectionRequest{Filter: inFilter("role", stringValue("user"), boolValue(true))}, wantColumn: "role"},
		{name: "range on a string column", request: &pb.SearchCollectionRequest{Filter: rangeFilter("role", bound(1), nil)}, wantColumn: "role"},
		{name: "range without a bound", request: &pb.SearchCollectionRequest{Filter: rangeFilter("timestampUnix", nil, nil)}, wantColumn: "timestampUnix"},
		{name: "empty membership set", request: &pb.SearchCollectionRequest{Filter: inFilter("provider")}, wantColumn: "provider"},
		{name: "oversized membership set", request: &pb.SearchCollectionRequest{Filter: inFilter("conversationId", oversized...)}, wantColumn: "conversationId"},
		{name: "literal without a value", request: &pb.SearchCollectionRequest{Filter: equalsFilter("provider", &pb.CollectionFilterValue{})}, wantColumn: "provider"},
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
		{name: "negative group limit", request: &pb.SearchCollectionRequest{GroupBy: "conversationId", PerGroupLimit: -1}},
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
// conversation keeps one alpha row and fills the rest from beta. The score
// floor keeps only the exact matches.
func TestSearchCollectionAppliesGroupCapAndScoreFloor(t *testing.T) {
	t.Parallel()
	daemon := newSearchCorpusDaemon(t)

	uncapped := daemon.mustSearchCollection(&pb.SearchCollectionRequest{CollectionId: searchCorpusCollection, Query: searchCorpusQuery, Limit: 3})
	for _, hit := range uncapped {
		if !strings.HasPrefix(hit.GetRowKey(), "conv/"+alphaConversation+"/") {
			t.Fatalf("uncapped top 3 = %v, want only alpha rows", hitRowKeys(uncapped))
		}
	}

	capped := daemon.mustSearchCollection(&pb.SearchCollectionRequest{
		CollectionId:  searchCorpusCollection,
		Query:         searchCorpusQuery,
		Limit:         3,
		GroupBy:       "conversationId",
		PerGroupLimit: 1,
	})
	if len(capped) != 2 {
		t.Fatalf("capped hits = %v, want one row per conversation", hitRowKeys(capped))
	}
	if !strings.HasPrefix(capped[0].GetRowKey(), "conv/"+alphaConversation+"/") || !strings.HasPrefix(capped[1].GetRowKey(), "conv/"+betaConversation+"/") {
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
	if got, want := sortedHitRowKeys(floored), sortedRowKeys(rowKey(alphaConversation, 0), rowKey(alphaConversation, 1), rowKey(alphaConversation, 2)); !slices.Equal(got, want) {
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
		Filter:       allOfFilter(inFilter("conversationId", stringValue(betaConversation)), equalsFilter("messageIndex", int64Value(0))),
	})
	if len(hits) != 1 {
		t.Fatalf("hits = %v, want one beta row", hitRowKeys(hits))
	}
	hit := hits[0]
	if hit.GetRowKey() != rowKey(betaConversation, 0) || hit.GetContent() != "how does the haystack index work" {
		t.Fatalf("hit = %s %q, want the first beta row", hit.GetRowKey(), hit.GetContent())
	}
	want := []*pb.CollectionHitScalar{
		{Column: "conversationId", Value: &pb.CollectionHitScalar_StringValue{StringValue: betaConversation}},
		{Column: "parentConversationId", Value: &pb.CollectionHitScalar_StringValue{StringValue: alphaConversation}},
		{Column: "role", Value: &pb.CollectionHitScalar_StringValue{StringValue: "user"}},
		{Column: "provider", Value: &pb.CollectionHitScalar_StringValue{StringValue: "codex"}},
		{Column: "workspaceRoot", Value: &pb.CollectionHitScalar_StringValue{StringValue: "/work/beta"}},
		{Column: "archived", Value: &pb.CollectionHitScalar_BoolValue{BoolValue: true}},
		{Column: "timestampUnix", Value: &pb.CollectionHitScalar_Int64Value{Int64Value: 2000}},
		{Column: "messageIndex", Value: &pb.CollectionHitScalar_Int64Value{Int64Value: 0}},
		{Column: "loadRules", Value: &pb.CollectionHitScalar_StringValue{StringValue: ""}},
	}
	requireHitScalars(t, hit.GetScalars(), want)

	alphaHits := daemon.mustSearchCollection(&pb.SearchCollectionRequest{
		CollectionId: searchCorpusCollection,
		Query:        searchCorpusQuery,
		Limit:        1,
		Filter:       inFilter("conversationId", stringValue(alphaConversation)),
	})
	if len(alphaHits) != 1 {
		t.Fatalf("alpha hits = %v, want one", hitRowKeys(alphaHits))
	}
	loadRules := alphaHits[0].GetScalars()[len(want)-1]
	if loadRules.GetColumn() != "loadRules" || loadRules.GetStringValue() != alphaLoadRules {
		t.Fatalf("alpha loadRules scalar = %v, want %q", loadRules, alphaLoadRules)
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
		semantic.NullCell("workspaceRoot"),
		semantic.AbsentCell("priority"),
		semantic.ValueCell("archived", semantic.BoolScalar(false)),
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
	if got := itemState(searchCorpusCollection, alphaConversation); got != alphaFingerprint {
		t.Fatalf("alpha fingerprint = %q, want %q", got, alphaFingerprint)
	}
	if got := itemState(searchCorpusCollection, "claude:missing"); got != "" {
		t.Fatalf("unknown item fingerprint = %q, want empty", got)
	}
	if got := itemState("never-registered", alphaConversation); got != "" {
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

// TestConversationSearchRPCsMatchGenericSearch proves both old conversation
// search RPCs return the same ordered rows and scores as the generic search
// with the equivalent typed filter, and that SearchWithinConversation returns
// the item state fingerprint.
func TestConversationSearchRPCsMatchGenericSearch(t *testing.T) {
	t.Parallel()
	daemon := newSearchCorpusDaemon(t)
	archived := true
	cases := []struct {
		name                 string
		oldFilter            *pb.ConversationSearchFilter
		perConversationLimit int32
		generic              *pb.SearchCollectionRequest
	}{
		{name: "empty filter", oldFilter: nil, generic: &pb.SearchCollectionRequest{}},
		{
			name:      "providers and uppercase roles",
			oldFilter: &pb.ConversationSearchFilter{Providers: []string{"claude"}, Roles: []string{"ASSISTANT"}},
			generic:   &pb.SearchCollectionRequest{Filter: allOfFilter(inFilter("provider", stringValue("claude")), inFilter("role", stringValue("assistant")))},
		},
		{
			name:      "time and message index bounds",
			oldFilter: &pb.ConversationSearchFilter{FromUnix: 1010, UntilUnix: 2010, MessageIndexFrom: 1, MessageIndexUntil: 3},
			generic: &pb.SearchCollectionRequest{Filter: allOfFilter(
				rangeFilter("timestampUnix", bound(1010), nil), rangeFilter("timestampUnix", nil, bound(2010)),
				rangeFilter("messageIndex", bound(1), nil), rangeFilter("messageIndex", nil, bound(3)),
			)},
		},
		{
			name:      "parent, workspace, and archived",
			oldFilter: &pb.ConversationSearchFilter{ParentConversationId: alphaConversation, WorkspaceRoots: []string{"/work/beta"}, Archived: &archived},
			generic: &pb.SearchCollectionRequest{Filter: allOfFilter(
				inFilter("workspaceRoot", stringValue("/work/beta")), equalsFilter("parentConversationId", stringValue(alphaConversation)), equalsFilter("archived", boolValue(true)),
			)},
		},
		{
			name:                 "per conversation limit and score floor",
			oldFilter:            &pb.ConversationSearchFilter{MinScore: 0.5},
			perConversationLimit: 1,
			generic:              &pb.SearchCollectionRequest{MinScore: 0.5, GroupBy: "conversationId", PerGroupLimit: 1},
		},
	}
	for _, testCase := range cases {
		old, err := daemon.client.SearchConversations(grpcutil.WithCorrelation(context.Background()), &pb.SearchConversationsRequest{
			CollectionId:         searchCorpusCollection,
			Query:                searchCorpusQuery,
			Filter:               testCase.oldFilter,
			PerConversationLimit: testCase.perConversationLimit,
		})
		if err != nil {
			t.Fatalf("%s: SearchConversations returned error: %v", testCase.name, err)
		}
		testCase.generic.CollectionId = searchCorpusCollection
		testCase.generic.Query = searchCorpusQuery
		generic := daemon.mustSearchCollection(testCase.generic)
		requireConversationParity(t, testCase.name, old.GetResults(), generic)
	}

	within, err := daemon.client.SearchWithinConversation(grpcutil.WithCorrelation(context.Background()), &pb.SearchWithinConversationRequest{
		CollectionId:   searchCorpusCollection,
		ConversationId: betaConversation,
		Query:          searchCorpusQuery,
	})
	if err != nil {
		t.Fatalf("SearchWithinConversation returned error: %v", err)
	}
	if within.GetIndexedFingerprint() != betaFingerprint {
		t.Fatalf("within fingerprint = %q, want %q", within.GetIndexedFingerprint(), betaFingerprint)
	}
	scoped := daemon.mustSearchCollection(&pb.SearchCollectionRequest{
		CollectionId: searchCorpusCollection,
		Query:        searchCorpusQuery,
		Filter:       allOfFilter(inFilter("conversationId", stringValue(betaConversation))),
	})
	requireConversationParity(t, "within conversation", within.GetResults(), scoped)

	unregistered, err := daemon.client.SearchConversations(grpcutil.WithCorrelation(context.Background()), &pb.SearchConversationsRequest{CollectionId: "never-registered", Query: searchCorpusQuery})
	if err != nil || len(unregistered.GetResults()) != 0 {
		t.Fatalf("SearchConversations on an unregistered collection = %v, %v, want no results", unregistered.GetResults(), err)
	}
	if records := daemon.documentCollectionRecords("never-registered"); len(records) != 0 {
		t.Fatalf("SearchConversations registered %d records, want 0", len(records))
	}
}

// requireConversationParity compares the fields the old and generic responses
// share: row order, score, content, conversation id, and message index.
func requireConversationParity(t *testing.T, label string, old []*pb.ConversationSearchResult, generic []*pb.CollectionSearchHit) {
	t.Helper()
	if len(old) == 0 {
		t.Fatalf("%s: old RPC returned no results, want a non-empty comparison", label)
	}
	if len(old) != len(generic) {
		t.Fatalf("%s: old RPC returned %d results, generic returned %d", label, len(old), len(generic))
	}
	for index := range old {
		hit := generic[index]
		if hit.GetRowKey() != rowKey(old[index].GetConversationId(), old[index].GetMessageIndex()) {
			t.Fatalf("%s: result %d row key %s, want %s", label, index, hit.GetRowKey(), rowKey(old[index].GetConversationId(), old[index].GetMessageIndex()))
		}
		if hit.GetScore() != old[index].GetScore() || hit.GetContent() != old[index].GetContent() {
			t.Fatalf("%s: result %d score/content = %v %q, want %v %q", label, index, hit.GetScore(), hit.GetContent(), old[index].GetScore(), old[index].GetContent())
		}
		if hit.GetScalars()[0].GetStringValue() != old[index].GetConversationId() {
			t.Fatalf("%s: result %d conversationId scalar %v, want %s", label, index, hit.GetScalars()[0], old[index].GetConversationId())
		}
	}
}

// TestSearchCollectionRefusesDuringMaintenance proves maintenance mode fails
// the generic search with the maintenance reason, and the old conversation
// search RPCs keep their maintenance status.
func TestSearchCollectionRefusesDuringMaintenance(t *testing.T) {
	t.Parallel()
	daemon := newSearchCorpusDaemon(t)
	if _, err := daemon.client.SetMaintenanceMode(grpcutil.WithCorrelation(context.Background()), &pb.SetMaintenanceModeRequest{Enabled: true, Reason: "backup"}); err != nil {
		t.Fatalf("SetMaintenanceMode returned error: %v", err)
	}

	_, err := daemon.searchCollection(&pb.SearchCollectionRequest{CollectionId: searchCorpusCollection, Query: searchCorpusQuery})
	requireColumnError(t, err, codes.FailedPrecondition, "maintenance", "")

	_, err = daemon.client.SearchConversations(grpcutil.WithCorrelation(context.Background()), &pb.SearchConversationsRequest{CollectionId: searchCorpusCollection, Query: searchCorpusQuery})
	requireMaintenanceStatus(t, "SearchConversations", err)
	_, err = daemon.client.SearchWithinConversation(grpcutil.WithCorrelation(context.Background()), &pb.SearchWithinConversationRequest{CollectionId: searchCorpusCollection, ConversationId: alphaConversation, Query: searchCorpusQuery})
	requireMaintenanceStatus(t, "SearchWithinConversation", err)
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

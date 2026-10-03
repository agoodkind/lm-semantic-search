//go:build live

package live

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/protobuf/types/known/structpb"
)

const (
	paritySearchQuery        = "needle"
	parityBulkConversations  = 300
	parityDenseMessages      = 30
	parityLegacyConversation = "claude:legacy"
	parityDenseConversation  = "claude:dense"
	parityFullLimit          = 1000
	parityPageLimit          = 10
	parityGroupLimit         = 2
	parityBatchSize          = 256
	parityVisibilityTimeout  = 60 * time.Second
	parityVisibilityInterval = 500 * time.Millisecond
	parityBaseTimestamp      = 1_700_000_000
	parityDenseTimestamp     = 1_600_000_000
	parityLegacyTimestamp    = 1_500_000_000
	parityLegacyMessages     = 3
)

// parityRow is the scalar metadata of one corpus row. A nil pointer is a null
// stored value, which only the legacy rows written directly to Milvus have.
type searchParityRow struct {
	rowKey         string
	conversationID string
	provider       string
	role           string
	timestampUnix  int64
	messageIndex   int64
	parent         *string
	workspaceRoot  *string
	archived       *bool
	loadRules      *string
}

type parityCorpus struct {
	conversations map[string][]*pb.ConversationDocument
	rows          []searchParityRow
	bulkIDs       []string
}

func stringPointer(value string) *string {
	return &value
}

func boolPointer(value bool) *bool {
	return &value
}

func bulkConversationID(index int) string {
	provider := "claude"
	if index%2 == 1 {
		provider = "codex"
	}
	return fmt.Sprintf("%s:bulk-%03d", provider, index)
}

// buildParityCorpus returns the synthetic parity corpus. Three hundred bulk
// conversations exceed one membership batch and vary every scalar column. The
// dense conversation has thirty messages. Each dense message text equals the
// query, and the dense rows fill the first ranked pages.
func buildParityCorpus() parityCorpus {
	corpus := parityCorpus{conversations: map[string][]*pb.ConversationDocument{}, rows: nil, bulkIDs: nil}
	for index := range parityBulkConversations {
		conversationID := bulkConversationID(index)
		corpus.bulkIDs = append(corpus.bulkIDs, conversationID)
		provider, _, _ := strings.Cut(conversationID, ":")
		workspaceRoot := fmt.Sprintf("/work/w%d", index%3)
		archived := index%4 == 0
		parent := ""
		if index%5 == 0 && index != 0 {
			parent = bulkConversationID(0)
		}
		loadRules := ""
		if index%2 == 0 {
			loadRules = "rules-v2"
		}
		roles := []string{"User", "assistant"}
		if index%3 == 0 {
			roles[0] = "user"
		}
		for messageIndex, role := range roles {
			timestamp := int64(parityBaseTimestamp + index*10 + messageIndex)
			corpus.conversations[conversationID] = append(corpus.conversations[conversationID], &pb.ConversationDocument{
				ConversationId:       conversationID,
				ParentConversationId: parent,
				MessageIndex:         int32(messageIndex),
				Role:                 role,
				TimestampUnix:        timestamp,
				Text:                 fmt.Sprintf("bulk conversation %03d message %d covers topic %d", index, messageIndex, index%7),
				WorkspaceRoot:        workspaceRoot,
				Archived:             archived,
				LoadRules:            loadRules,
			})
			corpus.rows = append(corpus.rows, searchParityRow{
				rowKey:         fmt.Sprintf("conv/%s/%d", conversationID, messageIndex),
				conversationID: conversationID,
				provider:       provider,
				role:           strings.ToLower(role),
				timestampUnix:  timestamp,
				messageIndex:   int64(messageIndex),
				parent:         stringPointer(parent),
				workspaceRoot:  stringPointer(workspaceRoot),
				archived:       boolPointer(archived),
				loadRules:      stringPointer(loadRules),
			})
		}
	}
	for messageIndex := range parityDenseMessages {
		role := "assistant"
		if messageIndex%2 == 0 {
			role = "user"
		}
		timestamp := int64(parityDenseTimestamp + messageIndex)
		corpus.conversations[parityDenseConversation] = append(corpus.conversations[parityDenseConversation], &pb.ConversationDocument{
			ConversationId: parityDenseConversation,
			MessageIndex:   int32(messageIndex),
			Role:           role,
			TimestampUnix:  timestamp,
			Text:           paritySearchQuery,
			WorkspaceRoot:  "/work/dense",
			LoadRules:      "rules-dense",
		})
		corpus.rows = append(corpus.rows, searchParityRow{
			rowKey:         fmt.Sprintf("conv/%s/%d", parityDenseConversation, messageIndex),
			conversationID: parityDenseConversation,
			provider:       "claude",
			role:           role,
			timestampUnix:  timestamp,
			messageIndex:   int64(messageIndex),
			parent:         stringPointer(""),
			workspaceRoot:  stringPointer("/work/dense"),
			archived:       boolPointer(false),
			loadRules:      stringPointer("rules-dense"),
		})
	}
	for messageIndex := range parityLegacyMessages {
		corpus.rows = append(corpus.rows, searchParityRow{
			rowKey:         fmt.Sprintf("conv/%s/%d", parityLegacyConversation, messageIndex),
			conversationID: parityLegacyConversation,
			provider:       "claude",
			role:           "user",
			timestampUnix:  int64(parityLegacyTimestamp + messageIndex),
			messageIndex:   int64(messageIndex),
			parent:         nil,
			workspaceRoot:  nil,
			archived:       nil,
			loadRules:      nil,
		})
	}
	return corpus
}

// insertLegacyRows writes the legacy rows directly to the harness collection
// with null parent, workspace, archived, and load rules values, the shape of a
// row the enrichment backfill has not updated.
func (h *harness) insertLegacyRows(corpus parityCorpus) {
	h.t.Helper()
	legacy := make([]searchParityRow, 0, parityLegacyMessages)
	for _, row := range corpus.rows {
		if row.conversationID == parityLegacyConversation {
			legacy = append(legacy, row)
		}
	}
	count := len(legacy)
	ids := make([]string, 0, count)
	contents := make([]string, 0, count)
	paths := make([]string, 0, count)
	metadata := make([]string, 0, count)
	vectors := make([][]float32, 0, count)
	conversationIDs := make([]string, 0, count)
	providers := make([]string, 0, count)
	roles := make([]string, 0, count)
	timestamps := make([]int64, 0, count)
	messageIndexes := make([]int64, 0, count)
	for _, row := range legacy {
		content := fmt.Sprintf("legacy message %d before scalar enrichment", row.messageIndex)
		ids = append(ids, "legacy_"+strconv.FormatInt(row.messageIndex, 10))
		contents = append(contents, content)
		paths = append(paths, row.rowKey)
		metadata = append(metadata, fmt.Sprintf(`{"conversation_id":%q,"message_index":%d,"role":%q,"timestamp_unix":%d}`, row.conversationID, row.messageIndex, row.role, row.timestampUnix))
		vector := make([]float32, 0, fakeEmbeddingDimension)
		for _, value := range deterministicVector(content, fakeEmbeddingDimension) {
			vector = append(vector, float32(value))
		}
		vectors = append(vectors, vector)
		conversationIDs = append(conversationIDs, row.conversationID)
		providers = append(providers, row.provider)
		roles = append(roles, row.role)
		timestamps = append(timestamps, row.timestampUnix)
		messageIndexes = append(messageIndexes, row.messageIndex)
	}
	allNull := make([]bool, count)
	nullColumns := make([]column.Column, 0, 7)
	for _, name := range []string{"parentConversationId", "workspaceRoot", "loadRules", "contentHash", "embeddingModel"} {
		nullColumn, err := column.NewNullableColumnVarChar(name, make([]string, count), allNull, column.WithSparseNullableMode[string](true))
		if err != nil {
			h.t.Fatalf("build null column %s: %v", name, err)
		}
		nullColumns = append(nullColumns, nullColumn)
	}
	archivedColumn, err := column.NewNullableColumnBool("archived", make([]bool, count), allNull, column.WithSparseNullableMode[bool](true))
	if err != nil {
		h.t.Fatalf("build null archived column: %v", err)
	}
	splitPartColumn, err := column.NewNullableColumnInt64("splitPart", make([]int64, count), allNull, column.WithSparseNullableMode[int64](true))
	if err != nil {
		h.t.Fatalf("build null splitPart column: %v", err)
	}
	nullColumns = append(nullColumns, archivedColumn, splitPartColumn)
	zeros := make([]int64, count)
	extensions := make([]string, count)
	insertOption := milvusclient.NewColumnBasedInsertOption(h.collectionName).
		WithVarcharColumn("id", ids).
		WithVarcharColumn("content", contents).
		WithVarcharColumn(relativePathField, paths).
		WithInt64Column("startLine", zeros).
		WithInt64Column("endLine", zeros).
		WithVarcharColumn("fileExtension", extensions).
		WithVarcharColumn("metadata", metadata).
		WithFloatVectorColumn("vector", fakeEmbeddingDimension, vectors).
		WithVarcharColumn("conversationId", conversationIDs).
		WithVarcharColumn("provider", providers).
		WithVarcharColumn("role", roles).
		WithInt64Column("timestampUnix", timestamps).
		WithInt64Column("messageIndex", messageIndexes).
		WithColumns(nullColumns...)
	if _, err := h.milvus.Insert(correlatedContext(), insertOption); err != nil {
		h.t.Fatalf("insert legacy rows into %s: %v", h.collectionName, err)
	}
}

// genericSearch runs the generic RPC on the harness collection.
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

// oldSearch runs the old conversation search RPC on the harness collection.
func (h *harness) oldSearch(filter *pb.ConversationSearchFilter, limit int32, perConversationLimit int32) []*pb.ConversationSearchResult {
	h.t.Helper()
	response, err := h.client.SearchConversations(correlatedContext(), &pb.SearchConversationsRequest{
		CollectionId:         h.collectionID,
		Query:                paritySearchQuery,
		Limit:                limit,
		Filter:               filter,
		PerConversationLimit: perConversationLimit,
	})
	if err != nil {
		h.t.Fatalf("SearchConversations returned error: %v", err)
	}
	return response.GetResults()
}

// waitForCorpusVisibility polls the generic search until every corpus row is
// searchable. Milvus search reads at bounded consistency. A direct insert
// becomes visible to search after a short delay.
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
	trimmed := strings.TrimPrefix(rowKey, "conv/")
	separator := strings.LastIndex(trimmed, "/")
	if separator < 0 {
		return trimmed
	}
	return trimmed[:separator]
}

// requireOldGenericParity requires the old and generic responses to list the
// same rows in the same order with equal scores and content. A mismatch
// reports row keys and scores only.
func requireOldGenericParity(t *testing.T, label string, old []*pb.ConversationSearchResult, generic []*pb.CollectionSearchHit) {
	t.Helper()
	if len(old) != len(generic) {
		t.Fatalf("%s: old RPC returned %d rows, generic returned %d", label, len(old), len(generic))
	}
	for index := range old {
		oldKey := fmt.Sprintf("conv/%s/%d", old[index].GetConversationId(), old[index].GetMessageIndex())
		if oldKey != generic[index].GetRowKey() || old[index].GetScore() != generic[index].GetScore() {
			t.Fatalf("%s: rank %d old %s score %v, generic %s score %v", label, index, oldKey, old[index].GetScore(), generic[index].GetRowKey(), generic[index].GetScore())
		}
		if old[index].GetContent() != generic[index].GetContent() {
			t.Fatalf("%s: rank %d row %s content differs between the old and generic RPCs", label, index, oldKey)
		}
	}
}

// requireSameRanking requires two rankings to list the same rows in the same
// order with equal scores.
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

// capHits applies a per-group cap, a score floor, and a limit to a ranked list,
// the reduction the daemon applies after its store search.
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

// storedScalarRows reads the stored relativePath and declared scalar values of
// rowKeys directly from Milvus at strong consistency.
func (h *harness) storedScalarRows(rowKeys []string) map[string]map[string]*pb.CollectionHitScalar {
	h.t.Helper()
	declared := semantic.ConversationDeclaration().Scalars
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

// TestGenericCollectionSearchParity is the read-only parity battery on an
// isolated harness collection. It compares the old conversation RPCs with the
// generic RPC on the same corpus: provider, role, time, message index, parent,
// workspace, and archived filters, a scope of more than 256 conversation ids,
// group caps, the score floor, and within-conversation fingerprints. Both RPCs
// must return identical ordered rows. It checks every filter result against the
// corpus definition, proves the group cap fills past an overfilled top in one
// ranking search, proves a scope of more than 256 ids ranks in one search,
// proves repeated rankings are identical and smaller limits are prefixes, and
// reads the stored rows directly to confirm row keys and every scalar echo,
// including null values. Failure messages report row keys, scores, and scalar
// metadata, never message text.
func TestGenericCollectionSearchParity(t *testing.T) {
	h := newHarness(t)
	corpus := buildParityCorpus()
	ingested := h.upsert(corpus.conversations, pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN, false, false)
	requireCompleted(t, ingested, "parity corpus ingest")
	h.insertLegacyRows(corpus)
	h.waitForCorpusVisibility(corpus)

	t.Run("filters match the corpus and the old RPC", func(t *testing.T) {
		archivedTrue := true
		archivedFalse := false
		cases := []struct {
			name      string
			oldFilter *pb.ConversationSearchFilter
			generic   *pb.CollectionFilter
			keep      func(searchParityRow) bool
		}{
			{
				name:      "provider",
				oldFilter: &pb.ConversationSearchFilter{Providers: []string{"codex"}},
				generic:   parityAll(parityIn("provider", "codex")),
				keep:      func(row searchParityRow) bool { return row.provider == "codex" },
			},
			{
				name:      "uppercase role",
				oldFilter: &pb.ConversationSearchFilter{Roles: []string{"USER"}},
				generic:   parityAll(parityIn("role", "user")),
				keep:      func(row searchParityRow) bool { return row.role == "user" },
			},
			{
				name:      "time bounds",
				oldFilter: &pb.ConversationSearchFilter{FromUnix: parityBaseTimestamp + 1000, UntilUnix: parityBaseTimestamp + 1500},
				generic:   parityAll(parityRange("timestampUnix", parityBound(parityBaseTimestamp+1000), nil), parityRange("timestampUnix", nil, parityBound(parityBaseTimestamp+1500))),
				keep: func(row searchParityRow) bool {
					return row.timestampUnix >= parityBaseTimestamp+1000 && row.timestampUnix < parityBaseTimestamp+1500
				},
			},
			{
				name:      "message index bounds",
				oldFilter: &pb.ConversationSearchFilter{MessageIndexFrom: 1, MessageIndexUntil: 5},
				generic:   parityAll(parityRange("messageIndex", parityBound(1), nil), parityRange("messageIndex", nil, parityBound(5))),
				keep:      func(row searchParityRow) bool { return row.messageIndex >= 1 && row.messageIndex < 5 },
			},
			{
				name:      "parent",
				oldFilter: &pb.ConversationSearchFilter{ParentConversationId: bulkConversationID(0)},
				generic:   parityAll(parityEquals("parentConversationId", parityStringValue(bulkConversationID(0)))),
				keep:      func(row searchParityRow) bool { return row.parent != nil && *row.parent == bulkConversationID(0) },
			},
			{
				name:      "workspace",
				oldFilter: &pb.ConversationSearchFilter{WorkspaceRoots: []string{"/work/w1", "/work/dense"}},
				generic:   parityAll(parityIn("workspaceRoot", "/work/w1", "/work/dense")),
				keep: func(row searchParityRow) bool {
					return row.workspaceRoot != nil && (*row.workspaceRoot == "/work/w1" || *row.workspaceRoot == "/work/dense")
				},
			},
			{
				name:      "archived true excludes null",
				oldFilter: &pb.ConversationSearchFilter{Archived: &archivedTrue},
				generic:   parityAll(parityEquals("archived", parityBoolValue(true))),
				keep:      func(row searchParityRow) bool { return row.archived != nil && *row.archived },
			},
			{
				name:      "archived false excludes null",
				oldFilter: &pb.ConversationSearchFilter{Archived: &archivedFalse},
				generic:   parityAll(parityEquals("archived", parityBoolValue(false))),
				keep:      func(row searchParityRow) bool { return row.archived != nil && !*row.archived },
			},
			{
				name:      "more than 256 conversation ids",
				oldFilter: &pb.ConversationSearchFilter{ConversationIds: corpus.bulkIDs},
				generic:   parityAll(parityIn("conversationId", corpus.bulkIDs...)),
				keep:      func(row searchParityRow) bool { return strings.Contains(row.conversationID, ":bulk-") },
			},
		}
		for _, testCase := range cases {
			old := h.oldSearch(testCase.oldFilter, parityFullLimit, 0)
			generic := h.genericSearch(&pb.SearchCollectionRequest{Limit: parityFullLimit, Filter: testCase.generic})
			requireOldGenericParity(t, testCase.name, old, generic)
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
			{name: "is_null archived", filter: parityIsNull("archived"), keep: func(row searchParityRow) bool { return row.archived == nil }},
			{
				name: "int64 and bool membership",
				filter: parityAll(
					&pb.CollectionFilter{Node: &pb.CollectionFilter_InSet{InSet: &pb.CollectionFilterIn{Column: "messageIndex", Values: []*pb.CollectionFilterValue{
						{Value: &pb.CollectionFilterValue_Int64Value{Int64Value: 1}},
						{Value: &pb.CollectionFilterValue_Int64Value{Int64Value: 2}},
					}}}},
					&pb.CollectionFilter{Node: &pb.CollectionFilter_InSet{InSet: &pb.CollectionFilterIn{Column: "archived", Values: []*pb.CollectionFilterValue{parityBoolValue(true)}}}},
				),
				keep: func(row searchParityRow) bool {
					return (row.messageIndex == 1 || row.messageIndex == 2) && row.archived != nil && *row.archived
				},
			},
			{name: "is_present workspaceRoot", filter: parityIsPresent("workspaceRoot"), keep: func(row searchParityRow) bool { return row.workspaceRoot != nil }},
			{
				name:   "negated comparison excludes null",
				filter: parityNegate(parityEquals("archived", parityBoolValue(true))),
				keep:   func(row searchParityRow) bool { return row.archived != nil && !*row.archived },
			},
			{
				name:   "negated group over null columns",
				filter: parityNegate(parityAny(parityEquals("archived", parityBoolValue(true)), parityIn("loadRules", "rules-v2"))),
				keep: func(row searchParityRow) bool {
					return row.archived != nil && !*row.archived && row.loadRules != nil && *row.loadRules != "rules-v2"
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
			if groupOfRowKey(hit.GetRowKey()) == parityDenseConversation {
				denseInTop++
			}
		}
		if denseInTop <= parityGroupLimit {
			t.Fatalf("top %d ranks have %d dense rows, want more than the group limit %d", parityPageLimit, denseInTop, parityGroupLimit)
		}
		h.callRecorder.reset()
		capped := h.genericSearch(&pb.SearchCollectionRequest{Limit: parityPageLimit, GroupBy: "conversationId", PerGroupLimit: parityGroupLimit})
		if calls := h.searchCallCount(); calls != 1 {
			t.Fatalf("capped search ran %d ranking searches, want 1", calls)
		}
		requireSameRanking(t, "group cap fill", capped, capHits(full, parityGroupLimit, 0, parityPageLimit))
		old := h.oldSearch(nil, parityPageLimit, parityGroupLimit)
		requireOldGenericParity(t, "group cap fill old RPC", old, capped)
	})

	t.Run("more than 256 ids rank in one search", func(t *testing.T) {
		scope := append(slices.Clone(corpus.bulkIDs), parityDenseConversation)
		inScope := map[string]bool{}
		for _, conversationID := range scope {
			inScope[conversationID] = true
		}
		full := h.genericSearch(&pb.SearchCollectionRequest{Limit: parityFullLimit})
		scopedFull := make([]*pb.CollectionSearchHit, 0, len(full))
		for _, hit := range full {
			if inScope[groupOfRowKey(hit.GetRowKey())] {
				scopedFull = append(scopedFull, hit)
			}
		}
		h.callRecorder.reset()
		generic := h.genericSearch(&pb.SearchCollectionRequest{Limit: parityPageLimit, GroupBy: "conversationId", PerGroupLimit: parityGroupLimit, Filter: parityAll(parityIn("conversationId", scope...))})
		if calls := h.searchCallCount(); calls != 1 {
			t.Fatalf("scoped search ran %d ranking searches, want 1", calls)
		}
		requireSameRanking(t, "large scope", generic, capHits(scopedFull, parityGroupLimit, 0, parityPageLimit))
		old := h.oldSearch(&pb.ConversationSearchFilter{ConversationIds: scope}, parityPageLimit, parityGroupLimit)
		requireOldGenericParity(t, "large scope old RPC", old, generic)
	})

	t.Run("rankings are stable and smaller limits are prefixes", func(t *testing.T) {
		full := hitKeys(h.genericSearch(&pb.SearchCollectionRequest{Limit: parityFullLimit}))
		for range 3 {
			if again := hitKeys(h.genericSearch(&pb.SearchCollectionRequest{Limit: parityFullLimit})); !slices.Equal(again, full) {
				t.Fatalf("repeated full ranking differs: %d rows versus %d", len(again), len(full))
			}
		}
		larger := h.genericSearch(&pb.SearchCollectionRequest{Limit: 2 * parityPageLimit, GroupBy: "conversationId", PerGroupLimit: parityGroupLimit})
		for _, limit := range []int32{1, 3, 5, parityPageLimit, 15} {
			smaller := h.genericSearch(&pb.SearchCollectionRequest{Limit: limit, GroupBy: "conversationId", PerGroupLimit: parityGroupLimit})
			if len(smaller) > len(larger) || !slices.Equal(hitKeys(smaller), hitKeys(larger)[:len(smaller)]) {
				t.Fatalf("limit %d rows %v are not a prefix of %v", limit, hitKeys(smaller), hitKeys(larger))
			}
			old := h.oldSearch(nil, limit, parityGroupLimit)
			requireOldGenericParity(t, fmt.Sprintf("prefix limit %d old RPC", limit), old, smaller)
		}
	})

	t.Run("score floor", func(t *testing.T) {
		full := h.genericSearch(&pb.SearchCollectionRequest{Limit: parityFullLimit})
		floor := full[len(full)/4].GetScore()
		floored := h.genericSearch(&pb.SearchCollectionRequest{Limit: parityFullLimit, MinScore: floor})
		requireSameRanking(t, "score floor", floored, capHits(full, 0, floor, parityFullLimit))
		old := h.oldSearch(&pb.ConversationSearchFilter{MinScore: floor}, parityFullLimit, 0)
		requireOldGenericParity(t, "score floor old RPC", old, floored)
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
			t.Fatal("no hit echoed a null scalar, want the legacy rows' null values")
		}
	})

	t.Run("within-conversation fingerprints", func(t *testing.T) {
		for _, conversationID := range []string{parityDenseConversation, bulkConversationID(7), parityLegacyConversation} {
			within, err := h.client.SearchWithinConversation(correlatedContext(), &pb.SearchWithinConversationRequest{
				CollectionId:   h.collectionID,
				ConversationId: conversationID,
				Query:          paritySearchQuery,
				Limit:          parityFullLimit,
			})
			if err != nil {
				t.Fatalf("SearchWithinConversation(%s) returned error: %v", conversationID, err)
			}
			state, err := h.client.GetCollectionItemState(correlatedContext(), &pb.GetCollectionItemStateRequest{CollectionId: h.collectionID, ItemId: conversationID})
			if err != nil {
				t.Fatalf("GetCollectionItemState(%s) returned error: %v", conversationID, err)
			}
			wantFingerprint := ""
			if documents, ingestedConversation := corpus.conversations[conversationID]; ingestedConversation {
				wantFingerprint = fingerprint(documents)
			}
			if within.GetIndexedFingerprint() != wantFingerprint || state.GetIndexedFingerprint() != wantFingerprint {
				t.Fatalf("%s fingerprints: within %q, item state %q, want %q", conversationID, within.GetIndexedFingerprint(), state.GetIndexedFingerprint(), wantFingerprint)
			}
			scoped := h.genericSearch(&pb.SearchCollectionRequest{Limit: parityFullLimit, Filter: parityAll(parityIn("conversationId", conversationID))})
			requireOldGenericParity(t, "within "+conversationID, within.GetResults(), scoped)
			wantKeys := expectedRowKeys(corpus, func(row searchParityRow) bool { return row.conversationID == conversationID })
			if got := sortedKeysOfHits(scoped); !slices.Equal(got, wantKeys) {
				t.Fatalf("within %s returned %d rows, want %d", conversationID, len(got), len(wantKeys))
			}
		}
	})
}

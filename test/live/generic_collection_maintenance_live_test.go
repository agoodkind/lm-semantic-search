//go:build live

package live

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// maintenanceStoredFields are the stored fields a maintenance comparison reads
// besides the vector. The vector compares bit for bit, and content compares by
// hash, so a failure message prints no transcript text.
var maintenanceStoredFields = []string{
	relativePathField,
	"contentHash",
	"embeddingModel",
	"metadata",
	"startLine",
	"endLine",
	"fileExtension",
	"splitPart",
	semantic.ConversationIDColumn,
	semantic.ConversationParentColumn,
	semantic.ConversationRoleColumn,
	semantic.ConversationProviderColumn,
	semantic.ConversationWorkspaceRootColumn,
	semantic.ConversationArchivedColumn,
	semantic.ConversationTimestampColumn,
	semantic.ConversationMessageIndexColumn,
	semantic.ConversationLoadRulesColumn,
}

// maintenanceRow is one stored Milvus row reduced to comparable strings: the
// hash of its content, the bits of its vector, and the rendered value of every
// field in maintenanceStoredFields.
type maintenanceRow map[string]string

// TestGenericCollectionMaintenanceParity ingests the same synthetic transcripts
// into a conversation collection and a generic collection with the
// conversation declaration in a real temporary Milvus database. Conversations
// A and C store no workspace root, and B stores one. The old and generic dry
// runs report equal counts and write nothing. The old backfill fills the
// conversation collection and the generic backfill fills the generic
// collection. Both change only A's empty workspace roots and keep every other
// field and every vector bit for bit. The old and generic deletes of A then
// remove only A's rows. Both collections store equal rows after each step, and
// no step changes a checkpoint.
func TestGenericCollectionMaintenanceParity(t *testing.T) {
	h := newHarness(t)
	genericCollectionID := "live-maintenance-" + randomID()
	registration, err := h.client.RegisterCollection(correlatedContext(), &pb.RegisterCollectionRequest{
		CollectionId: genericCollectionID,
		ItemIdColumn: semantic.ConversationDeclaration().ItemIDColumn,
		Scalars:      parityDeclarationPB(),
		Client:       &pb.ClientInfo{Name: "live-harness"},
	})
	if err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}
	h.trackCollectionFamily(registration.GetCollectionName())

	conversationA := parityConversationID("claude", "maintenance-a")
	conversationB := parityConversationID("codex", "maintenance-b")
	conversationC := parityConversationID("claude", "maintenance-c")
	convs := map[string][]*pb.ConversationDocument{
		conversationA: withoutWorkspace(parityTranscript(conversationA, "")),
		conversationB: parityTranscript(conversationB, conversationA),
		conversationC: withoutWorkspace(parityTranscript(conversationC, "")),
	}
	requireCompleted(t, h.upsert(convs, pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN, false, false), "conversation ingest")
	requireCompleted(t, h.upsertGeneric(genericCollectionID, convs, parityManifest(convs), pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_RETAIN, false, false), "generic ingest")
	h.requireParity(registration, "ingest")

	conversationBefore := h.maintenanceRows(h.collectionName)
	genericBefore := h.maintenanceRows(registration.GetCollectionName())
	conversationCheckpoint := h.parityCheckpoint(h.codebaseID)
	genericCheckpoint := h.parityCheckpoint(registration.GetCodebaseId())
	rowsA := maintenanceRowIDs(conversationBefore, conversationA)
	rowsB := maintenanceRowIDs(conversationBefore, conversationB)
	rowsC := maintenanceRowIDs(conversationBefore, conversationC)
	if len(rowsA) < 4 || len(rowsB) < 4 || len(rowsC) < 4 {
		t.Fatalf("stored rows A %d, B %d, C %d, want split text, tool, and thinking rows for each", len(rowsA), len(rowsB), len(rowsC))
	}
	// The partial update omits the BM25 input and output fields. The set of rows
	// a full-text search matches before and after the backfill shows that every
	// row, including A's filled rows, keeps its sparse vector. BM25 scores are
	// not compared: each upsert changes the collection's BM25 statistics.
	fullTextBefore := make(map[string][]string, 2)
	if h.config.HybridMode {
		for _, collectionName := range []string{h.collectionName, registration.GetCollectionName()} {
			fullTextBefore[collectionName] = h.fullTextHits(collectionName)
			if !containsAny(fullTextBefore[collectionName], rowsA) {
				t.Fatalf("full-text search of %s matched no row of A", collectionName)
			}
		}
	}

	entries := []*pb.BackfillConversationScalarEntry{
		{ConversationId: conversationA, WorkspaceRoot: "/work/maintenance-a", Archived: true},
		{ConversationId: conversationB, WorkspaceRoot: "/work/renamed", Archived: true},
	}
	columns := []string{semantic.ConversationWorkspaceRootColumn, semantic.ConversationArchivedColumn}
	items := []*pb.BackfillCollectionItem{
		maintenanceItem(conversationA, "/work/maintenance-a", true),
		maintenanceItem(conversationB, "/work/renamed", true),
	}
	h.requireBackfillCounts("old dry run", h.backfillConversation(h.collectionID, true, entries), len(rowsA), len(rowsC))
	h.requireBackfillCounts("generic dry run on the conversation collection", h.backfillCollection(h.collectionID, columns, true, items), len(rowsA), len(rowsC))
	h.requireBackfillCounts("generic dry run on the generic collection", h.backfillCollection(genericCollectionID, columns, true, items), len(rowsA), len(rowsC))
	h.requireMaintenanceRows("dry runs", h.collectionName, conversationBefore, nil)
	h.requireMaintenanceRows("dry runs", registration.GetCollectionName(), genericBefore, nil)

	h.requireBackfillCounts("old backfill", h.backfillConversation(h.collectionID, false, entries), len(rowsA), len(rowsC))
	h.requireBackfillCounts("generic backfill", h.backfillCollection(genericCollectionID, columns, false, items), len(rowsA), len(rowsC))
	filled := make(map[string]maintenanceRow, len(rowsA))
	for _, rowID := range rowsA {
		filled[rowID] = maintenanceRow{semantic.ConversationWorkspaceRootColumn: "/work/maintenance-a"}
	}
	h.requireMaintenanceRows("old backfill", h.collectionName, conversationBefore, filled)
	h.requireMaintenanceRows("generic backfill", registration.GetCollectionName(), genericBefore, filled)
	h.requireParity(registration, "backfill")
	if h.config.HybridMode {
		for _, collectionName := range []string{h.collectionName, registration.GetCollectionName()} {
			if got := h.fullTextHits(collectionName); !reflect.DeepEqual(got, fullTextBefore[collectionName]) {
				t.Fatalf("full-text hits of %s changed after the backfill: %v, want %v", collectionName, got, fullTextBefore[collectionName])
			}
		}
	}
	h.requireBackfillCounts("generic dry run after the backfill", h.backfillCollection(genericCollectionID, columns, true, items), 0, len(rowsC))

	afterBackfill := h.maintenanceRows(h.collectionName)
	requireCompleted(t, h.waitJob(h.deleteConversation(h.collectionID, conversationA)), "old delete")
	requireCompleted(t, h.waitJob(h.deleteCollectionItem(genericCollectionID, conversationA)), "generic delete")
	kept := make(map[string]maintenanceRow, len(afterBackfill))
	for rowID, row := range afterBackfill {
		if row[semantic.ConversationIDColumn] != conversationA {
			kept[rowID] = row
		}
	}
	h.requireMaintenanceRows("old delete", h.collectionName, kept, nil)
	h.requireParity(registration, "delete")
	if got := h.parityCheckpoint(h.codebaseID); !reflect.DeepEqual(got, conversationCheckpoint) {
		t.Fatalf("conversation checkpoint changed: %v, want %v", got, conversationCheckpoint)
	}
	if got := h.parityCheckpoint(registration.GetCodebaseId()); !reflect.DeepEqual(got, genericCheckpoint) {
		t.Fatalf("generic checkpoint changed: %v, want %v", got, genericCheckpoint)
	}
}

// withoutWorkspace returns copies of documents that store no workspace root.
func withoutWorkspace(documents []*pb.ConversationDocument) []*pb.ConversationDocument {
	copies := make([]*pb.ConversationDocument, 0, len(documents))
	for _, document := range documents {
		copied := &pb.ConversationDocument{
			ConversationId:       document.GetConversationId(),
			ParentConversationId: document.GetParentConversationId(),
			MessageIndex:         document.GetMessageIndex(),
			Role:                 document.GetRole(),
			TimestampUnix:        document.GetTimestampUnix(),
			Text:                 document.GetText(),
			Thinking:             document.GetThinking(),
			Tools:                document.GetTools(),
			LoadRules:            document.GetLoadRules(),
		}
		copies = append(copies, copied)
	}
	return copies
}

func maintenanceItem(conversationID string, workspaceRoot string, archived bool) *pb.BackfillCollectionItem {
	return &pb.BackfillCollectionItem{
		ItemId: conversationID,
		Scalars: []*pb.CollectionScalarValue{
			{Column: semantic.ConversationWorkspaceRootColumn, Value: &pb.CollectionScalarValue_StringValue{StringValue: workspaceRoot}},
			{Column: semantic.ConversationArchivedColumn, Value: &pb.CollectionScalarValue_BoolValue{BoolValue: archived}},
		},
	}
}

// maintenanceRowIDs returns the ids of the stored rows of one conversation.
func maintenanceRowIDs(rows map[string]maintenanceRow, conversationID string) []string {
	ids := make([]string, 0)
	for rowID, row := range rows {
		if row[semantic.ConversationIDColumn] == conversationID {
			ids = append(ids, rowID)
		}
	}
	return ids
}

// backfillCounts is the changed and orphan count one backfill RPC returned.
type backfillCounts struct {
	changed int64
	orphan  int64
}

func (h *harness) requireBackfillCounts(step string, counts backfillCounts, wantChanged int, wantOrphan int) {
	h.t.Helper()
	if counts.changed != int64(wantChanged) || counts.orphan != int64(wantOrphan) {
		h.t.Fatalf("%s: changed %d orphan %d, want changed %d orphan %d", step, counts.changed, counts.orphan, wantChanged, wantOrphan)
	}
}

// backfillConversation sends one old conversation scalar backfill stream.
func (h *harness) backfillConversation(collectionID string, dryRun bool, entries []*pb.BackfillConversationScalarEntry) backfillCounts {
	h.t.Helper()
	stream, err := h.client.BackfillConversationScalars(correlatedContext())
	if err != nil {
		h.t.Fatalf("open BackfillConversationScalars returned error: %v", err)
	}
	for _, chunk := range []*pb.BackfillConversationScalarsChunk{
		{Chunk: &pb.BackfillConversationScalarsChunk_Header{Header: &pb.BackfillConversationScalarsHeader{CollectionId: collectionID, DryRun: dryRun, Client: &pb.ClientInfo{Name: "live-harness"}}}},
		{Chunk: &pb.BackfillConversationScalarsChunk_Entries{Entries: &pb.BackfillConversationScalarsEntries{Entries: entries}}},
	} {
		if err := stream.Send(chunk); err != nil && !errors.Is(err, io.EOF) {
			h.t.Fatalf("send conversation backfill chunk returned error: %v", err)
		}
	}
	response, err := stream.CloseAndRecv()
	if err != nil {
		h.t.Fatalf("BackfillConversationScalars returned error: %v", err)
	}
	return backfillCounts{changed: response.GetChanged(), orphan: response.GetOrphan()}
}

// backfillCollection sends one generic scalar backfill stream.
func (h *harness) backfillCollection(collectionID string, columns []string, dryRun bool, items []*pb.BackfillCollectionItem) backfillCounts {
	h.t.Helper()
	stream, err := h.client.BackfillCollectionScalars(correlatedContext())
	if err != nil {
		h.t.Fatalf("open BackfillCollectionScalars returned error: %v", err)
	}
	for _, frame := range []*pb.BackfillCollectionScalarsStreamRequest{
		{Chunk: &pb.BackfillCollectionScalarsStreamRequest_Header{Header: &pb.BackfillCollectionScalarsHeader{CollectionId: collectionID, Columns: columns, DryRun: dryRun, Client: &pb.ClientInfo{Name: "live-harness"}}}},
		{Chunk: &pb.BackfillCollectionScalarsStreamRequest_Items{Items: &pb.BackfillCollectionScalarsItems{Items: items}}},
	} {
		if err := stream.Send(frame); err != nil && !errors.Is(err, io.EOF) {
			h.t.Fatalf("send collection backfill frame returned error: %v", err)
		}
	}
	response, err := stream.CloseAndRecv()
	if err != nil {
		h.t.Fatalf("BackfillCollectionScalars returned error: %v", err)
	}
	return backfillCounts{changed: response.GetChanged(), orphan: response.GetOrphan()}
}

// deleteConversation queues one old conversation delete and returns its job id.
func (h *harness) deleteConversation(collectionID string, conversationID string) string {
	h.t.Helper()
	response, err := h.client.DeleteConversation(correlatedContext(), &pb.DeleteConversationRequest{CollectionId: collectionID, ConversationId: conversationID, Client: &pb.ClientInfo{Name: "live-harness"}})
	if err != nil {
		h.t.Fatalf("DeleteConversation returned error: %v", err)
	}
	return response.GetJobId()
}

// deleteCollectionItem queues one generic item delete and returns its job id.
func (h *harness) deleteCollectionItem(collectionID string, itemID string) string {
	h.t.Helper()
	response, err := h.client.DeleteCollectionItem(correlatedContext(), &pb.DeleteCollectionItemRequest{CollectionId: collectionID, ItemId: itemID, Client: &pb.ClientInfo{Name: "live-harness"}})
	if err != nil {
		h.t.Fatalf("DeleteCollectionItem returned error: %v", err)
	}
	return response.GetJobId()
}

// requireMaintenanceRows requires collectionName to store the rows of want,
// except the fields changes lists per row id, which must equal the listed
// values.
func (h *harness) requireMaintenanceRows(step string, collectionName string, want map[string]maintenanceRow, changes map[string]maintenanceRow) {
	h.t.Helper()
	got := h.maintenanceRows(collectionName)
	if len(got) != len(want) {
		h.t.Fatalf("%s: %s stores %d rows, want %d", step, collectionName, len(got), len(want))
	}
	for rowID, wantRow := range want {
		gotRow, found := got[rowID]
		if !found {
			h.t.Fatalf("%s: %s lost row %s", step, collectionName, rowID)
		}
		for field, wantValue := range wantRow {
			if changed, listed := changes[rowID][field]; listed {
				wantValue = changed
			}
			if gotRow[field] != wantValue {
				h.t.Fatalf("%s: %s row %s field %s = %q, want %q", step, collectionName, rowID, field, gotRow[field], wantValue)
			}
		}
	}
}

// fullTextHitLimit exceeds the row count of the maintenance fixture, so one
// full-text search returns every row with a query term.
const fullTextHitLimit = 1000

// fullTextHits runs one BM25 full-text search against the sparse vector field
// at strong consistency and returns the sorted ids of every row it matches.
func (h *harness) fullTextHits(collectionName string) []string {
	h.t.Helper()
	results, err := h.milvus.Search(context.Background(), milvusclient.NewSearchOption(collectionName, fullTextHitLimit, []entity.Vector{entity.Text("design note ingestion step")}).
		WithANNSField("sparse_vector").
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		h.t.Fatalf("full-text search of %s: %v", collectionName, err)
	}
	hits := make([]string, 0)
	for _, result := range results {
		for rowIndex := range result.ResultCount {
			rowID, err := result.IDs.GetAsString(rowIndex)
			if err != nil {
				h.t.Fatalf("read full-text hit id at %d: %v", rowIndex, err)
			}
			hits = append(hits, rowID)
		}
	}
	slices.Sort(hits)
	return hits
}

// containsAny reports whether values contains any of candidates.
func containsAny(values []string, candidates []string) bool {
	for _, candidate := range candidates {
		if slices.Contains(values, candidate) {
			return true
		}
	}
	return false
}

// maintenanceRows reads every row of a collection at strong consistency,
// keyed by row id.
func (h *harness) maintenanceRows(collectionName string) map[string]maintenanceRow {
	h.t.Helper()
	outputFields := append([]string{"id", "content", "vector"}, maintenanceStoredFields...)
	result, err := h.milvus.Query(context.Background(), milvusclient.NewQueryOption(collectionName).
		WithFilter(`id != ""`).
		WithOutputFields(outputFields...).
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		h.t.Fatalf("query maintenance rows from %s: %v", collectionName, err)
	}
	rows := make(map[string]maintenanceRow, result.ResultCount)
	for rowIndex := range result.ResultCount {
		row := maintenanceRow{}
		for _, field := range maintenanceStoredFields {
			row[field] = parityColumnValue(h.t, result.GetColumn(field), field, rowIndex)
		}
		contentHash := sha256.Sum256([]byte(parityColumnValue(h.t, result.GetColumn("content"), "content", rowIndex)))
		row["content"] = hex.EncodeToString(contentHash[:])
		vectorValue, err := result.GetColumn("vector").Get(rowIndex)
		if err != nil {
			h.t.Fatalf("read vector at %d: %v", rowIndex, err)
		}
		vector, isVector := vectorValue.(entity.FloatVector)
		if !isVector {
			h.t.Fatalf("vector at %d has type %T", rowIndex, vectorValue)
		}
		bits := make([]string, 0, len(vector))
		for _, component := range vector {
			bits = append(bits, strconv.FormatUint(uint64(math.Float32bits(component)), 16))
		}
		row["vector"] = strings.Join(bits, ",")
		rows[parityColumnValue(h.t, result.GetColumn("id"), "id", rowIndex)] = row
	}
	return rows
}

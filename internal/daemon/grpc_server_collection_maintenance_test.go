package daemon

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/model"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// localRowFields decodes every stored row of a local collection into its raw
// JSON fields, keyed by row id. A row compares byte for byte field by field,
// including its vector.
func (daemon *offlineCollectionDaemon) localRowFields(collectionName string) map[string]map[string]json.RawMessage {
	daemon.t.Helper()
	rows := make(map[string]map[string]json.RawMessage)
	for _, line := range strings.Split(strings.TrimSpace(string(daemon.localRowsFile(collectionName))), "\n") {
		if line == "" {
			continue
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal([]byte(line), &fields); err != nil {
			daemon.t.Fatalf("decode local row: %v", err)
		}
		var rowID string
		if err := json.Unmarshal(fields["id"], &rowID); err != nil {
			daemon.t.Fatalf("decode local row id: %v", err)
		}
		rows[rowID] = fields
	}
	return rows
}

// localRowsFile returns the exact bytes of a local collection's row file.
func (daemon *offlineCollectionDaemon) localRowsFile(collectionName string) []byte {
	daemon.t.Helper()
	return readFileBytes(daemon.t, filepath.Join(daemon.config.StateRoot, "localvec", collectionName, "metadata.jsonl"))
}

// ingestDocuments streams conversation documents and their manifest through the
// conversation upsert RPC and waits for the ingest job to complete.
func (daemon *offlineCollectionDaemon) ingestDocuments(collectionID string, documents []*pb.ConversationDocument) {
	daemon.t.Helper()
	manifest := make(map[string]string)
	for _, document := range documents {
		manifest[document.GetConversationId()] = "fingerprint-" + document.GetConversationId()
	}
	fingerprints := make([]*pb.ConversationFingerprint, 0, len(manifest))
	for conversationID, fingerprint := range manifest {
		fingerprints = append(fingerprints, &pb.ConversationFingerprint{ConversationId: conversationID, Fingerprint: fingerprint})
	}
	stream, err := daemon.client.UpsertConversationDocumentsStream(grpcutil.WithCorrelation(context.Background()))
	if err != nil {
		daemon.t.Fatalf("open UpsertConversationDocumentsStream returned error: %v", err)
	}
	for _, chunk := range []*pb.UpsertConversationDocumentsChunk{
		{Chunk: &pb.UpsertConversationDocumentsChunk_Header{Header: &pb.UpsertConversationDocumentsHeader{CollectionId: collectionID, Client: &pb.ClientInfo{Name: "maintenance-test"}}}},
		{Chunk: &pb.UpsertConversationDocumentsChunk_Documents{Documents: &pb.UpsertConversationDocumentsDocuments{Documents: documents}}},
		{Chunk: &pb.UpsertConversationDocumentsChunk_Manifest{Manifest: &pb.UpsertConversationDocumentsManifest{Manifest: fingerprints}}},
	} {
		if err := stream.Send(chunk); err != nil {
			daemon.t.Fatalf("send conversation chunk returned error: %v", err)
		}
	}
	response, err := stream.CloseAndRecv()
	if err != nil {
		daemon.t.Fatalf("conversation CloseAndRecv returned error: %v", err)
	}
	if job := waitForRPCJobTerminal(daemon.t, daemon.client, response.GetJobId()); job.GetState() != string(model.JobStateCompleted) {
		daemon.t.Fatalf("conversation ingest state = %q: %+v", job.GetState(), job.GetError())
	}
}

// backfillConversation sends one old conversation scalar backfill stream.
func (daemon *offlineCollectionDaemon) backfillConversation(collectionID string, dryRun bool, entries []*pb.BackfillConversationScalarEntry) (*pb.BackfillConversationScalarsResponse, error) {
	daemon.t.Helper()
	stream, err := daemon.client.BackfillConversationScalars(grpcutil.WithCorrelation(context.Background()))
	if err != nil {
		daemon.t.Fatalf("open BackfillConversationScalars returned error: %v", err)
	}
	for _, chunk := range []*pb.BackfillConversationScalarsChunk{
		{Chunk: &pb.BackfillConversationScalarsChunk_Header{Header: &pb.BackfillConversationScalarsHeader{CollectionId: collectionID, DryRun: dryRun, Client: &pb.ClientInfo{Name: "maintenance-test"}}}},
		{Chunk: &pb.BackfillConversationScalarsChunk_Entries{Entries: &pb.BackfillConversationScalarsEntries{Entries: entries}}},
	} {
		if err := stream.Send(chunk); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			daemon.t.Fatalf("send conversation backfill chunk returned error: %v", err)
		}
	}
	return stream.CloseAndRecv()
}

// sendBackfillFrames sends frames on one BackfillCollectionScalars stream and
// returns the response or the stream's final status error.
func (daemon *offlineCollectionDaemon) sendBackfillFrames(frames []*pb.BackfillCollectionScalarsStreamRequest) (*pb.BackfillCollectionScalarsResponse, error) {
	daemon.t.Helper()
	stream, err := daemon.client.BackfillCollectionScalars(grpcutil.WithCorrelation(context.Background()))
	if err != nil {
		daemon.t.Fatalf("open BackfillCollectionScalars returned error: %v", err)
	}
	for _, frame := range frames {
		if err := stream.Send(frame); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			daemon.t.Fatalf("send collection backfill frame returned error: %v", err)
		}
	}
	return stream.CloseAndRecv()
}

// backfillItems sends one generic scalar backfill with a header and one items
// frame.
func (daemon *offlineCollectionDaemon) backfillItems(collectionID string, columns []string, dryRun bool, items []*pb.BackfillCollectionItem) (*pb.BackfillCollectionScalarsResponse, error) {
	daemon.t.Helper()
	return daemon.sendBackfillFrames(backfillFrames(backfillHeader(collectionID, columns, dryRun), items))
}

// deleteItem deletes one item through the generic RPC and waits for the job.
func (daemon *offlineCollectionDaemon) deleteItem(collectionID string, itemID string) {
	daemon.t.Helper()
	response, err := daemon.client.DeleteCollectionItem(grpcutil.WithCorrelation(context.Background()), &pb.DeleteCollectionItemRequest{CollectionId: collectionID, ItemId: itemID, Client: &pb.ClientInfo{Name: "maintenance-test"}})
	if err != nil {
		daemon.t.Fatalf("DeleteCollectionItem returned error: %v", err)
	}
	if !strings.Contains(response.GetDisplayText(), "Started document delete job "+response.GetJobId()) {
		daemon.t.Fatalf("DeleteCollectionItem DisplayText = %q, want the delete job start text", response.GetDisplayText())
	}
	if job := waitForRPCJobTerminal(daemon.t, daemon.client, response.GetJobId()); job.GetState() != string(model.JobStateCompleted) {
		daemon.t.Fatalf("item delete state = %q: %+v", job.GetState(), job.GetError())
	}
}

// deleteConversation deletes one conversation through the old RPC and waits
// for the job.
func (daemon *offlineCollectionDaemon) deleteConversation(collectionID string, conversationID string) {
	daemon.t.Helper()
	response, err := daemon.client.DeleteConversation(grpcutil.WithCorrelation(context.Background()), &pb.DeleteConversationRequest{CollectionId: collectionID, ConversationId: conversationID, Client: &pb.ClientInfo{Name: "maintenance-test"}})
	if err != nil {
		daemon.t.Fatalf("DeleteConversation returned error: %v", err)
	}
	if job := waitForRPCJobTerminal(daemon.t, daemon.client, response.GetJobId()); job.GetState() != string(model.JobStateCompleted) {
		daemon.t.Fatalf("conversation delete state = %q: %+v", job.GetState(), job.GetError())
	}
}

func backfillHeader(collectionID string, columns []string, dryRun bool) *pb.BackfillCollectionScalarsHeader {
	return &pb.BackfillCollectionScalarsHeader{CollectionId: collectionID, Columns: columns, DryRun: dryRun, Client: &pb.ClientInfo{Name: "maintenance-test"}}
}

func backfillFrames(header *pb.BackfillCollectionScalarsHeader, items []*pb.BackfillCollectionItem) []*pb.BackfillCollectionScalarsStreamRequest {
	return []*pb.BackfillCollectionScalarsStreamRequest{
		{Chunk: &pb.BackfillCollectionScalarsStreamRequest_Header{Header: header}},
		{Chunk: &pb.BackfillCollectionScalarsStreamRequest_Items{Items: &pb.BackfillCollectionScalarsItems{Items: items}}},
	}
}

func backfillItem(itemID string, scalars ...*pb.CollectionScalarValue) *pb.BackfillCollectionItem {
	return &pb.BackfillCollectionItem{ItemId: itemID, Scalars: scalars}
}

// maintenanceDocuments is a two-message conversation. The assistant message
// has a tool call and thinking text, which store convtool/ and convthink/ rows
// beside the conv/ rows.
func maintenanceDocuments(conversationID string, workspaceRoot string) []*pb.ConversationDocument {
	return []*pb.ConversationDocument{
		{ConversationId: conversationID, MessageIndex: 0, Role: "user", TimestampUnix: 1712345678, Text: "which scalars does the backfill fill in " + conversationID, WorkspaceRoot: workspaceRoot},
		{
			ConversationId: conversationID, MessageIndex: 1, Role: "assistant", TimestampUnix: 1712345679, Text: "the backfill fills only null or empty values in " + conversationID,
			WorkspaceRoot: workspaceRoot, Thinking: "checking the declaration of " + conversationID,
			Tools: []*pb.ConversationToolCall{{Name: "Read", Display: "declaration.go", LangHint: "go"}},
		},
	}
}

// rowIDsOf returns the ids of the rows of one conversation, sorted.
func rowIDsOf(t *testing.T, rows map[string]map[string]json.RawMessage, conversationID string) []string {
	t.Helper()
	ids := make([]string, 0)
	for rowID, fields := range rows {
		var stored string
		if raw, found := fields["conversationId"]; found {
			if err := json.Unmarshal(raw, &stored); err != nil {
				t.Fatalf("decode conversationId of %s: %v", rowID, err)
			}
		}
		if stored == conversationID {
			ids = append(ids, rowID)
		}
	}
	sort.Strings(ids)
	return ids
}

// requireRowFieldsEqual requires after to store the rows of before with equal
// raw fields, except the fields changes lists per row id, which must equal the
// listed raw JSON. A listed field with a nil value must be absent.
func requireRowFieldsEqual(t *testing.T, label string, before map[string]map[string]json.RawMessage, after map[string]map[string]json.RawMessage, changes map[string]map[string]json.RawMessage) {
	t.Helper()
	if len(before) != len(after) {
		t.Fatalf("%s: %d rows after, want %d", label, len(after), len(before))
	}
	for rowID, beforeFields := range before {
		afterFields, found := after[rowID]
		if !found {
			t.Fatalf("%s: row %s is gone", label, rowID)
		}
		keys := make(map[string]struct{})
		for key := range beforeFields {
			keys[key] = struct{}{}
		}
		for key := range afterFields {
			keys[key] = struct{}{}
		}
		for key := range keys {
			want, changed := changes[rowID][key]
			if !changed {
				want = beforeFields[key]
			}
			if !bytes.Equal(afterFields[key], want) {
				t.Fatalf("%s: row %s field %s = %s, want %s", label, rowID, key, afterFields[key], want)
			}
		}
	}
}

func requireBackfillCounts(t *testing.T, label string, changed int64, orphan int64, wantChanged int, wantOrphan int) {
	t.Helper()
	if changed != int64(wantChanged) || orphan != int64(wantOrphan) {
		t.Fatalf("%s: changed %d orphan %d, want changed %d orphan %d", label, changed, orphan, wantChanged, wantOrphan)
	}
}

// TestBackfillCollectionScalarsMatchesConversationBackfill ingests the same
// three conversations into two conversation collections. Conversations A and C
// store no workspace root, and B stores one. The old and generic dry runs
// report equal counts on the same stored rows and write nothing. The old RPC
// then fills one collection, and the generic RPC fills the other. Both fill
// only A's empty workspace roots, keep archived, keep every vector and every
// other field byte for byte, store equal rows, and leave the checkpoints
// unchanged.
func TestBackfillCollectionScalarsMatchesConversationBackfill(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)
	conversationA := "claude:backfill-a"
	conversationB := "codex:backfill-b"
	conversationC := "claude:backfill-c"
	documents := append(append(maintenanceDocuments(conversationA, ""), maintenanceDocuments(conversationB, "/work/b")...), maintenanceDocuments(conversationC, "")...)
	oldRegistration, err := daemon.registerConversationCollection("conv-backfill-old")
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	genericRegistration, err := daemon.registerConversationCollection("conv-backfill-generic")
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	daemon.ingestDocuments("conv-backfill-old", documents)
	daemon.ingestDocuments("conv-backfill-generic", documents)
	oldBefore := daemon.localRowFields(oldRegistration.GetCollectionName())
	genericBefore := daemon.localRowFields(genericRegistration.GetCollectionName())
	if !reflect.DeepEqual(oldBefore, genericBefore) {
		t.Fatal("the two collections stored different rows for the same documents")
	}
	rowsA := rowIDsOf(t, oldBefore, conversationA)
	rowsC := rowIDsOf(t, oldBefore, conversationC)
	if len(rowsA) < 3 || len(rowsC) < 3 || len(rowIDsOf(t, oldBefore, conversationB)) < 3 {
		t.Fatalf("stored rows A %d, B %d, C %d, want text, tool, and thinking rows for each", len(rowsA), len(rowIDsOf(t, oldBefore, conversationB)), len(rowsC))
	}
	oldCheckpoint := readFileBytes(t, daemon.checkpointPath(oldRegistration.GetCodebaseId()))
	genericCheckpoint := readFileBytes(t, daemon.checkpointPath(genericRegistration.GetCodebaseId()))
	oldFile := daemon.localRowsFile(oldRegistration.GetCollectionName())

	entries := []*pb.BackfillConversationScalarEntry{
		{ConversationId: conversationA, WorkspaceRoot: "/work/a", Archived: true},
		{ConversationId: conversationB, WorkspaceRoot: "/work/b-renamed", Archived: true},
	}
	columns := []string{"workspaceRoot", "archived"}
	items := []*pb.BackfillCollectionItem{
		backfillItem(conversationA, stringScalar("workspaceRoot", "/work/a"), boolScalar("archived", true)),
		backfillItem(conversationB, stringScalar("workspaceRoot", "/work/b-renamed"), boolScalar("archived", true)),
	}
	oldDryRun, err := daemon.backfillConversation("conv-backfill-old", true, entries)
	if err != nil {
		t.Fatalf("old dry run returned error: %v", err)
	}
	requireBackfillCounts(t, "old dry run", oldDryRun.GetChanged(), oldDryRun.GetOrphan(), len(rowsA), len(rowsC))
	if !strings.Contains(oldDryRun.GetDisplayText(), "Dry run counted conversation scalars for collection 'conv-backfill-old'") {
		t.Fatalf("old dry run DisplayText = %q", oldDryRun.GetDisplayText())
	}
	genericDryRun, err := daemon.backfillItems("conv-backfill-old", columns, true, items)
	if err != nil {
		t.Fatalf("generic dry run returned error: %v", err)
	}
	requireBackfillCounts(t, "generic dry run", genericDryRun.GetChanged(), genericDryRun.GetOrphan(), len(rowsA), len(rowsC))
	if !strings.Contains(genericDryRun.GetDisplayText(), "Dry run counted document scalars for collection 'conv-backfill-old'") {
		t.Fatalf("generic dry run DisplayText = %q", genericDryRun.GetDisplayText())
	}
	if !bytes.Equal(daemon.localRowsFile(oldRegistration.GetCollectionName()), oldFile) {
		t.Fatal("a dry run rewrote the stored rows")
	}

	oldRun, err := daemon.backfillConversation("conv-backfill-old", false, entries)
	if err != nil {
		t.Fatalf("old backfill returned error: %v", err)
	}
	requireBackfillCounts(t, "old backfill", oldRun.GetChanged(), oldRun.GetOrphan(), len(rowsA), len(rowsC))
	genericRun, err := daemon.backfillItems("conv-backfill-generic", columns, false, items)
	if err != nil {
		t.Fatalf("generic backfill returned error: %v", err)
	}
	requireBackfillCounts(t, "generic backfill", genericRun.GetChanged(), genericRun.GetOrphan(), len(rowsA), len(rowsC))

	filled := make(map[string]map[string]json.RawMessage, len(rowsA))
	for _, rowID := range rowsA {
		filled[rowID] = map[string]json.RawMessage{"workspaceRoot": json.RawMessage(`"/work/a"`)}
	}
	oldAfter := daemon.localRowFields(oldRegistration.GetCollectionName())
	requireRowFieldsEqual(t, "old backfill", oldBefore, oldAfter, filled)
	requireRowFieldsEqual(t, "generic backfill", genericBefore, daemon.localRowFields(genericRegistration.GetCollectionName()), filled)
	if !bytes.Equal(daemon.localRowsFile(oldRegistration.GetCollectionName()), daemon.localRowsFile(genericRegistration.GetCollectionName())) {
		t.Fatal("the old and generic backfills stored different rows")
	}
	if !bytes.Equal(readFileBytes(t, daemon.checkpointPath(oldRegistration.GetCodebaseId())), oldCheckpoint) ||
		!bytes.Equal(readFileBytes(t, daemon.checkpointPath(genericRegistration.GetCodebaseId())), genericCheckpoint) {
		t.Fatal("a scalar backfill changed a Merkle checkpoint")
	}

	again, err := daemon.backfillItems("conv-backfill-generic", columns, true, items)
	if err != nil {
		t.Fatalf("second generic dry run returned error: %v", err)
	}
	requireBackfillCounts(t, "second generic dry run", again.GetChanged(), again.GetOrphan(), 0, len(rowsC))
}

// TestBackfillCollectionScalarsFillsOnlyMissingValues backfills a generic
// collection with typed scalars. A null title, a null pinned value, and an
// empty title take the item's values. A set title and a set pinned value keep
// theirs, an omitted item stays unchanged as an orphan, and every vector, row
// key, text, and checkpoint stays as stored.
func TestBackfillCollectionScalarsFillsOnlyMissingValues(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)
	registered, err := daemon.registerCollection("docs-backfill", "docId", documentScalars())
	if err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}
	rows := []*pb.CollectionRow{
		{RowKey: "a/0", ItemId: "doc-a", Text: "alpha zero", Scalars: []*pb.CollectionScalarValue{nullScalar("title"), int64Scalar("rank", 1)}},
		{RowKey: "a/1", ItemId: "doc-a", Text: "alpha one", Scalars: []*pb.CollectionScalarValue{stringScalar("title", "given title"), boolScalar("pinned", true), int64Scalar("rank", 2)}},
		{RowKey: "b/0", ItemId: "doc-b", Text: "bravo zero", Scalars: []*pb.CollectionScalarValue{stringScalar("title", ""), boolScalar("pinned", false), int64Scalar("rank", 3)}},
		{RowKey: "c/0", ItemId: "doc-c", Text: "charlie zero", Scalars: []*pb.CollectionScalarValue{int64Scalar("rank", 4)}},
	}
	daemon.upsertItems(collectionHeader("docs-backfill", pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_UNSPECIFIED, false, false), rows, map[string]string{"doc-a": "fp-a", "doc-b": "fp-b", "doc-c": "fp-c"})
	before := daemon.localRows(registered.GetCollectionName())
	beforeFields := daemon.localRowFields(registered.GetCollectionName())
	checkpoint := readFileBytes(t, daemon.checkpointPath(registered.GetCodebaseId()))
	items := []*pb.BackfillCollectionItem{
		backfillItem("doc-a", stringScalar("title", "title from backfill a"), boolScalar("pinned", false)),
		backfillItem("doc-b", stringScalar("title", "title from backfill b"), boolScalar("pinned", true)),
	}

	dryRun, err := daemon.backfillItems("docs-backfill", []string{"title", "pinned"}, true, items)
	if err != nil {
		t.Fatalf("dry run returned error: %v", err)
	}
	requireBackfillCounts(t, "dry run", dryRun.GetChanged(), dryRun.GetOrphan(), 2, 1)
	run, err := daemon.backfillItems("docs-backfill", []string{"title", "pinned"}, false, items)
	if err != nil {
		t.Fatalf("backfill returned error: %v", err)
	}
	requireBackfillCounts(t, "backfill", run.GetChanged(), run.GetOrphan(), 2, 1)

	after := daemon.localRows(registered.GetCollectionName())
	wantScalars := map[string]map[string]model.ScalarValue{
		"a/0": {
			"docId":  {Type: model.ScalarTypeString, String: "doc-a"},
			"title":  {Type: model.ScalarTypeString, String: "title from backfill a"},
			"pinned": {Type: model.ScalarTypeBool, Bool: false},
			"rank":   {Type: model.ScalarTypeInt64, Int64: 1},
		},
		"a/1": rowByPath(t, before, "a/1").Scalars,
		"b/0": {
			"docId":  {Type: model.ScalarTypeString, String: "doc-b"},
			"title":  {Type: model.ScalarTypeString, String: "title from backfill b"},
			"pinned": {Type: model.ScalarTypeBool, Bool: false},
			"rank":   {Type: model.ScalarTypeInt64, Int64: 3},
		},
		"c/0": rowByPath(t, before, "c/0").Scalars,
	}
	for rowKey, want := range wantScalars {
		if got := rowByPath(t, after, rowKey).Scalars; !reflect.DeepEqual(got, want) {
			t.Fatalf("row %s scalars = %+v, want %+v", rowKey, got, want)
		}
	}
	changes := make(map[string]map[string]json.RawMessage)
	for _, row := range after {
		encoded, err := json.Marshal(row.Scalars)
		if err != nil {
			t.Fatalf("encode scalars of %s: %v", row.RelativePath, err)
		}
		changes[row.ID] = map[string]json.RawMessage{"scalars": encoded}
	}
	requireRowFieldsEqual(t, "typed backfill", beforeFields, daemon.localRowFields(registered.GetCollectionName()), changes)
	if !bytes.Equal(readFileBytes(t, daemon.checkpointPath(registered.GetCodebaseId())), checkpoint) {
		t.Fatal("a scalar backfill changed the Merkle checkpoint")
	}

	again, err := daemon.backfillItems("docs-backfill", []string{"title", "pinned"}, true, items)
	if err != nil {
		t.Fatalf("second dry run returned error: %v", err)
	}
	requireBackfillCounts(t, "second dry run", again.GetChanged(), again.GetOrphan(), 0, 1)
}

// TestBackfillCollectionScalarsRejectsInvalidRequests sends backfills that
// break the saved declaration or the stream order. Each fails with
// InvalidArgument and reports the rejected column in ErrorInfo when there is
// one. An unregistered collection fails with NotFound. No request writes a row.
func TestBackfillCollectionScalarsRejectsInvalidRequests(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)
	registered, err := daemon.registerCollection("docs-backfill-invalid", "docId", documentScalars())
	if err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}
	daemon.upsertItems(collectionHeader("docs-backfill-invalid", pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_UNSPECIFIED, false, false), []*pb.CollectionRow{documentRow("a/0", "doc-a", "alpha", 1)}, map[string]string{"doc-a": "fp-a"})
	conversation, err := daemon.registerConversationCollection("conv-backfill-invalid")
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	daemon.ingestDocuments("conv-backfill-invalid", maintenanceDocuments("claude:invalid-a", ""))
	genericFile := daemon.localRowsFile(registered.GetCollectionName())
	conversationFile := daemon.localRowsFile(conversation.GetCollectionName())

	titleOnly := []string{"title"}
	columnCases := []struct {
		name         string
		collectionID string
		columns      []string
		items        []*pb.BackfillCollectionItem
		wantColumn   string
	}{
		{name: "undeclared header column", collectionID: "docs-backfill-invalid", columns: []string{"author"}, wantColumn: "author"},
		{name: "item id column", collectionID: "docs-backfill-invalid", columns: []string{"docId"}, wantColumn: "docId"},
		{name: "header column twice", collectionID: "docs-backfill-invalid", columns: []string{"title", "title"}, wantColumn: "title"},
		{name: "column that is never missing", collectionID: "docs-backfill-invalid", columns: []string{"rank"}, wantColumn: "rank"},
		{name: "wrong type", collectionID: "docs-backfill-invalid", columns: titleOnly, items: []*pb.BackfillCollectionItem{backfillItem("doc-a", int64Scalar("title", 1))}, wantColumn: "title"},
		{name: "value outside the header", collectionID: "docs-backfill-invalid", columns: titleOnly, items: []*pb.BackfillCollectionItem{backfillItem("doc-a", stringScalar("title", "t"), boolScalar("pinned", true))}, wantColumn: "pinned"},
		{name: "null value", collectionID: "docs-backfill-invalid", columns: titleOnly, items: []*pb.BackfillCollectionItem{backfillItem("doc-a", nullScalar("title"))}, wantColumn: "title"},
		{name: "missing header column value", collectionID: "docs-backfill-invalid", columns: []string{"title", "pinned"}, items: []*pb.BackfillCollectionItem{backfillItem("doc-a", stringScalar("title", "t"))}, wantColumn: "pinned"},
		{name: "value set twice", collectionID: "docs-backfill-invalid", columns: titleOnly, items: []*pb.BackfillCollectionItem{backfillItem("doc-a", stringScalar("title", "t"), stringScalar("title", "u"))}, wantColumn: "title"},
		{name: "string over max length", collectionID: "docs-backfill-invalid", columns: titleOnly, items: []*pb.BackfillCollectionItem{backfillItem("doc-a", stringScalar("title", strings.Repeat("t", 513)))}, wantColumn: "title"},
		{name: "conversation provider", collectionID: "conv-backfill-invalid", columns: []string{"provider"}, wantColumn: "provider"},
		{name: "conversation message index", collectionID: "conv-backfill-invalid", columns: []string{"messageIndex"}, wantColumn: "messageIndex"},
	}
	for _, testCase := range columnCases {
		for _, dryRun := range []bool{true, false} {
			_, err := daemon.backfillItems(testCase.collectionID, testCase.columns, dryRun, testCase.items)
			requireColumnError(t, err, codes.InvalidArgument, "invalid_argument", testCase.wantColumn)
		}
	}

	header := &pb.BackfillCollectionScalarsStreamRequest{Chunk: &pb.BackfillCollectionScalarsStreamRequest_Header{Header: backfillHeader("docs-backfill-invalid", titleOnly, false)}}
	itemsFrame := &pb.BackfillCollectionScalarsStreamRequest{Chunk: &pb.BackfillCollectionScalarsStreamRequest_Items{Items: &pb.BackfillCollectionScalarsItems{Items: []*pb.BackfillCollectionItem{backfillItem("doc-a", stringScalar("title", "t"))}}}}
	frameCases := map[string][]*pb.BackfillCollectionScalarsStreamRequest{
		"items before header": {itemsFrame, header},
		"duplicate header":    {header, header},
		"no header":           nil,
		"no columns":          backfillFrames(backfillHeader("docs-backfill-invalid", nil, false), nil),
		"empty item id":       backfillFrames(backfillHeader("docs-backfill-invalid", titleOnly, false), []*pb.BackfillCollectionItem{backfillItem(" ", stringScalar("title", "t"))}),
		"item id twice": backfillFrames(backfillHeader("docs-backfill-invalid", titleOnly, false), []*pb.BackfillCollectionItem{
			backfillItem("doc-a", stringScalar("title", "t")), backfillItem("doc-a", stringScalar("title", "u")),
		}),
	}
	for name, frames := range frameCases {
		if _, err := daemon.sendBackfillFrames(frames); status.Code(err) != codes.InvalidArgument {
			t.Fatalf("%s returned %v, want InvalidArgument", name, err)
		}
	}
	if _, err := daemon.backfillItems("docs-backfill-unregistered", titleOnly, true, nil); status.Code(err) != codes.NotFound {
		t.Fatalf("backfill of an unregistered collection returned %v, want NotFound", err)
	}
	if !bytes.Equal(daemon.localRowsFile(registered.GetCollectionName()), genericFile) ||
		!bytes.Equal(daemon.localRowsFile(conversation.GetCollectionName()), conversationFile) {
		t.Fatal("a rejected backfill rewrote stored rows")
	}
}

// TestDeleteCollectionItemRemovesOnlyThatItem deletes one item of a generic
// collection and one conversation through each delete RPC. Each delete removes
// only the rows of its item, keeps every other row byte for byte, and leaves
// the checkpoint unchanged. The old and generic conversation deletes store
// equal rows. The generic RPC fails with NotFound for an unregistered
// collection and with InvalidArgument without an item id.
func TestDeleteCollectionItemRemovesOnlyThatItem(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)
	registered, err := daemon.registerCollection("docs-delete", "docId", documentScalars())
	if err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}
	daemon.upsertItems(
		collectionHeader("docs-delete", pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_UNSPECIFIED, false, false),
		[]*pb.CollectionRow{documentRow("a/0", "doc-a", "alpha zero", 1), documentRow("a/1", "doc-a", "alpha one", 2), documentRow("b/0", "doc-b", "bravo zero", 3), documentRow("b/1", "doc-b", "bravo one", 4)},
		map[string]string{"doc-a": "fp-a", "doc-b": "fp-b"},
	)
	before := daemon.localRowFields(registered.GetCollectionName())
	checkpoint := readFileBytes(t, daemon.checkpointPath(registered.GetCodebaseId()))
	daemon.deleteItem("docs-delete", "doc-a")
	after := daemon.localRowFields(registered.GetCollectionName())
	kept := make(map[string]map[string]json.RawMessage)
	for rowID, fields := range before {
		if !bytes.Contains(fields["relativePath"], []byte(`"a/`)) {
			kept[rowID] = fields
		}
	}
	if len(kept) != 2 {
		t.Fatalf("fixture stored %d rows of doc-b, want 2", len(kept))
	}
	requireRowFieldsEqual(t, "generic item delete", kept, after, nil)
	if !bytes.Equal(readFileBytes(t, daemon.checkpointPath(registered.GetCodebaseId())), checkpoint) {
		t.Fatal("an item delete changed the Merkle checkpoint")
	}

	documents := append(maintenanceDocuments("claude:delete-a", "/work/a"), maintenanceDocuments("claude:delete-b", "/work/b")...)
	oldRegistration, err := daemon.registerConversationCollection("conv-delete-old")
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	genericRegistration, err := daemon.registerConversationCollection("conv-delete-generic")
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	daemon.ingestDocuments("conv-delete-old", documents)
	daemon.ingestDocuments("conv-delete-generic", documents)
	conversationBefore := daemon.localRowFields(oldRegistration.GetCollectionName())
	oldCheckpoint := readFileBytes(t, daemon.checkpointPath(oldRegistration.GetCodebaseId()))
	daemon.deleteConversation("conv-delete-old", "claude:delete-a")
	daemon.deleteItem("conv-delete-generic", "claude:delete-a")
	keptConversation := make(map[string]map[string]json.RawMessage)
	for _, rowID := range rowIDsOf(t, conversationBefore, "claude:delete-b") {
		keptConversation[rowID] = conversationBefore[rowID]
	}
	requireRowFieldsEqual(t, "old conversation delete", keptConversation, daemon.localRowFields(oldRegistration.GetCollectionName()), nil)
	requireRowFieldsEqual(t, "generic conversation delete", keptConversation, daemon.localRowFields(genericRegistration.GetCollectionName()), nil)
	if !bytes.Equal(readFileBytes(t, daemon.checkpointPath(oldRegistration.GetCodebaseId())), oldCheckpoint) {
		t.Fatal("a conversation delete changed the Merkle checkpoint")
	}

	if _, err := daemon.client.DeleteCollectionItem(grpcutil.WithCorrelation(context.Background()), &pb.DeleteCollectionItemRequest{CollectionId: "docs-delete-unregistered", ItemId: "doc-a"}); status.Code(err) != codes.NotFound {
		t.Fatalf("delete from an unregistered collection returned %v, want NotFound", err)
	}
	if _, err := daemon.client.DeleteCollectionItem(grpcutil.WithCorrelation(context.Background()), &pb.DeleteCollectionItemRequest{CollectionId: "docs-delete", ItemId: " "}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("delete without an item id returned %v, want InvalidArgument", err)
	}
}

// TestCollectionMaintenanceSelectsLegacyConversationRows strips conversationId
// from the stored rows of one conversation, the shape of rows written before
// the column existed. The generic backfill attributes those rows to the
// conversation by their conv/, convtool/, and convthink/ path prefixes and
// fills them. The generic delete then removes them by the same prefixes and
// keeps the other conversation's rows.
func TestCollectionMaintenanceSelectsLegacyConversationRows(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)
	legacyID := "claude:legacy-a"
	currentID := "claude:current-b"
	registration, err := daemon.registerConversationCollection("conv-legacy-rows")
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	daemon.ingestDocuments("conv-legacy-rows", append(maintenanceDocuments(legacyID, ""), maintenanceDocuments(currentID, "/work/b")...))
	rowsPath := filepath.Join(daemon.config.StateRoot, "localvec", registration.GetCollectionName(), "metadata.jsonl")
	legacyRows := 0
	daemon.restart(func() {
		lines := strings.Split(strings.TrimSpace(string(readFileBytes(t, rowsPath))), "\n")
		rewritten := make([]string, 0, len(lines))
		for _, line := range lines {
			var fields map[string]json.RawMessage
			if err := json.Unmarshal([]byte(line), &fields); err != nil {
				t.Fatalf("decode local row: %v", err)
			}
			if string(fields["conversationId"]) == strconv.Quote(legacyID) {
				delete(fields, "conversationId")
				legacyRows++
			}
			encoded, err := json.Marshal(fields)
			if err != nil {
				t.Fatalf("encode local row: %v", err)
			}
			rewritten = append(rewritten, string(encoded))
		}
		if err := os.WriteFile(rowsPath, []byte(strings.Join(rewritten, "\n")+"\n"), 0o600); err != nil {
			t.Fatalf("write local rows: %v", err)
		}
	})
	if legacyRows < 3 {
		t.Fatalf("stripped conversationId from %d rows, want the text, tool, and thinking rows", legacyRows)
	}

	columns := []string{"workspaceRoot", "archived"}
	items := []*pb.BackfillCollectionItem{backfillItem(legacyID, stringScalar("workspaceRoot", "/work/legacy"), boolScalar("archived", false))}
	run, err := daemon.backfillItems("conv-legacy-rows", columns, false, items)
	if err != nil {
		t.Fatalf("backfill returned error: %v", err)
	}
	requireBackfillCounts(t, "legacy backfill", run.GetChanged(), run.GetOrphan(), legacyRows, 0)
	for _, row := range daemon.localRows(registration.GetCollectionName()) {
		legacy := strings.Contains(row.RelativePath, "/"+legacyID+"/")
		if legacy && !strings.Contains(row.Line, `"workspaceRoot":"/work/legacy"`) {
			t.Fatalf("legacy row %s kept an empty workspace root", row.RelativePath)
		}
		if !legacy && !strings.Contains(row.Line, `"workspaceRoot":"/work/b"`) {
			t.Fatalf("row %s lost its workspace root", row.RelativePath)
		}
	}

	daemon.deleteItem("conv-legacy-rows", legacyID)
	remaining := daemon.localRows(registration.GetCollectionName())
	if len(remaining) < 3 {
		t.Fatalf("delete kept %d rows, want the rows of %s", len(remaining), currentID)
	}
	for _, row := range remaining {
		if row.ConversationID != currentID {
			t.Fatalf("delete kept row %s of another conversation", row.RelativePath)
		}
	}
}

// TestCollectionMaintenanceRefusesDuringMaintenance puts the daemon in
// maintenance mode. The generic and old backfill and delete RPCs all fail with
// the maintenance refusal, the generic RPCs refuse before they resolve an
// unregistered collection, and no request writes a row or queues a job.
func TestCollectionMaintenanceRefusesDuringMaintenance(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)
	registration, err := daemon.registerConversationCollection("conv-maintenance")
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	daemon.ingestDocuments("conv-maintenance", maintenanceDocuments("claude:maintenance-a", ""))
	rowsFile := daemon.localRowsFile(registration.GetCollectionName())
	jobsBefore, err := daemon.client.ListJobs(grpcutil.WithCorrelation(context.Background()), &pb.ListJobsRequest{CodebaseId: registration.GetCodebaseId()})
	if err != nil {
		t.Fatalf("ListJobs returned error: %v", err)
	}
	if _, err := daemon.client.SetMaintenanceMode(grpcutil.WithCorrelation(context.Background()), &pb.SetMaintenanceModeRequest{Enabled: true, Reason: "store backup", Client: &pb.ClientInfo{Name: "maintenance-test"}}); err != nil {
		t.Fatalf("SetMaintenanceMode returned error: %v", err)
	}

	entries := []*pb.BackfillConversationScalarEntry{{ConversationId: "claude:maintenance-a", WorkspaceRoot: "/work/a"}}
	items := []*pb.BackfillCollectionItem{backfillItem("claude:maintenance-a", stringScalar("workspaceRoot", "/work/a"), boolScalar("archived", false))}
	columns := []string{"workspaceRoot", "archived"}
	refusals := make(map[string]error)
	_, refusals["old backfill"] = daemon.backfillConversation("conv-maintenance", false, entries)
	_, refusals["generic backfill"] = daemon.backfillItems("conv-maintenance", columns, false, items)
	_, refusals["generic backfill of an unregistered collection"] = daemon.backfillItems("conv-maintenance-unregistered", columns, false, items)
	_, refusals["old delete"] = daemon.client.DeleteConversation(grpcutil.WithCorrelation(context.Background()), &pb.DeleteConversationRequest{CollectionId: "conv-maintenance", ConversationId: "claude:maintenance-a"})
	_, refusals["generic delete"] = daemon.client.DeleteCollectionItem(grpcutil.WithCorrelation(context.Background()), &pb.DeleteCollectionItemRequest{CollectionId: "conv-maintenance", ItemId: "claude:maintenance-a"})
	_, refusals["generic delete from an unregistered collection"] = daemon.client.DeleteCollectionItem(grpcutil.WithCorrelation(context.Background()), &pb.DeleteCollectionItemRequest{CollectionId: "conv-maintenance-unregistered", ItemId: "claude:maintenance-a"})
	for name, err := range refusals {
		grpcStatus, _ := status.FromError(err)
		if grpcStatus.Code() != codes.FailedPrecondition || !strings.Contains(grpcStatus.Message(), "daemon is in maintenance mode (store backup)") {
			t.Fatalf("%s during maintenance returned %v, want the maintenance refusal", name, err)
		}
	}
	if !bytes.Equal(daemon.localRowsFile(registration.GetCollectionName()), rowsFile) {
		t.Fatal("a refused request rewrote stored rows")
	}
	jobsAfter, err := daemon.client.ListJobs(grpcutil.WithCorrelation(context.Background()), &pb.ListJobsRequest{CodebaseId: registration.GetCodebaseId()})
	if err != nil {
		t.Fatalf("ListJobs returned error: %v", err)
	}
	if len(jobsAfter.GetJobs()) != len(jobsBefore.GetJobs()) {
		t.Fatalf("refused requests queued %d jobs", len(jobsAfter.GetJobs())-len(jobsBefore.GetJobs()))
	}
}

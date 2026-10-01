//go:build live

package live

import (
	"context"
	"fmt"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/merkle"
)

func TestGenericCollectionTypedScalarsLive(t *testing.T) {
	h := newHarness(t)
	collectionID := "live-typed-" + randomID()
	register := func() *pb.RegisterCollectionResponse {
		response, err := h.client.RegisterCollection(correlatedContext(), &pb.RegisterCollectionRequest{
			CollectionId: collectionID,
			ItemIdColumn: "docId",
			Scalars:      typedDeclarationPB(),
			Client:       &pb.ClientInfo{Name: "live-harness"},
		})
		if err != nil {
			t.Fatalf("RegisterCollection returned error: %v", err)
		}
		return response
	}
	registration := register()
	h.trackCollectionFamily(registration.GetCollectionName())
	retain := pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_RETAIN

	rows := []*pb.CollectionRow{typedRow("a/0", "doc-a", "alpha zero", 1, true), typedRow("a/1", "doc-a", "alpha one", 2, false), typedRow("b/0", "doc-b", "bravo zero", 3, true)}
	rows[2].Scalars = append(rows[2].Scalars[:1], rows[2].Scalars[2:]...)
	h.requireTypedJob(collectionID, rows, map[string]string{"doc-a": "fp-a1", "doc-b": "fp-b1"}, retain, false, false, "first ingest")
	stored := h.typedRows(registration.GetCollectionName())
	want := map[string]string{
		"a/0": "docId=doc-a title=title a/0 pinned=true rank=1",
		"a/1": "docId=doc-a title=title a/1 pinned=false rank=2",
		"b/0": "docId=doc-b title=title b/0 pinned=null rank=3",
	}
	if !reflect.DeepEqual(stored, want) {
		t.Fatalf("stored typed rows = %v, want %v", stored, want)
	}

	h.restart(nil)
	register()
	withExtra := append(slices.Clone(rows[:2]), typedRow("a/2", "doc-a", "alpha two", 4, true))
	h.requireTypedJob(collectionID, withExtra, map[string]string{"doc-a": "fp-a1"}, retain, true, false, "backfill after restart")
	if stored := h.typedRows(registration.GetCollectionName()); len(stored) != 4 || stored["a/2"] == "" {
		t.Fatalf("rows after backfill = %v, want a/2 added", stored)
	}

	forced := []*pb.CollectionRow{typedRow("a/0", "doc-a", "alpha zero edited", 9, false)}
	h.requireTypedJob(collectionID, forced, map[string]string{"doc-a": "fp-a2"}, retain, false, true, "force")
	stored = h.typedRows(registration.GetCollectionName())
	wantForced := map[string]string{
		"a/0": "docId=doc-a title=title a/0 pinned=false rank=9",
		"b/0": "docId=doc-b title=title b/0 pinned=null rank=3",
	}
	if !reflect.DeepEqual(stored, wantForced) {
		t.Fatalf("rows after force = %v, want %v", stored, wantForced)
	}

	h.requireTypedJob(collectionID, nil, map[string]string{"doc-a": "fp-a2"}, pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_AUTHORITATIVE, false, false, "authoritative")
	if stored := h.typedRows(registration.GetCollectionName()); !reflect.DeepEqual(stored, map[string]string{"a/0": wantForced["a/0"]}) {
		t.Fatalf("rows after authoritative = %v, want only a/0", stored)
	}
	if got := h.parityCheckpoint(registration.GetCodebaseId()); !reflect.DeepEqual(got, map[string]string{"doc-a": "fp-a2"}) {
		t.Fatalf("checkpoint after authoritative = %v", got)
	}
}

func typedDeclarationPB() []*pb.ScalarColumnDeclaration {
	return []*pb.ScalarColumnDeclaration{
		{Column: "docId", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_STRING, Nullable: false, MaxLength: 128},
		{Column: "title", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_STRING, Nullable: true, MaxLength: 256},
		{Column: "pinned", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_BOOL, Nullable: true, MaxLength: 0},
		{Column: "rank", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_INT64, Nullable: false, MaxLength: 0},
	}
}

func typedRow(rowKey string, itemID string, text string, rank int64, pinned bool) *pb.CollectionRow {
	return &pb.CollectionRow{
		RowKey: rowKey,
		ItemId: itemID,
		Text:   text,
		Scalars: []*pb.CollectionScalarValue{
			{Column: "title", Value: &pb.CollectionScalarValue_StringValue{StringValue: "title " + rowKey}},
			{Column: "pinned", Value: &pb.CollectionScalarValue_BoolValue{BoolValue: pinned}},
			{Column: "rank", Value: &pb.CollectionScalarValue_Int64Value{Int64Value: rank}},
		},
	}
}

func (h *harness) requireTypedJob(collectionID string, rows []*pb.CollectionRow, manifest map[string]string, reconcile pb.CollectionReconcileMode, backfill bool, force bool, step string) {
	h.t.Helper()
	response, err := h.sendGeneric(collectionID, rows, manifest, reconcile, backfill, force)
	if err != nil {
		h.t.Fatalf("%s: UpsertCollectionItemsStream returned error: %v", step, err)
	}
	requireCompleted(h.t, h.waitJob(response.GetJobId()), step)
}

func (h *harness) typedRows(collectionName string) map[string]string {
	h.t.Helper()
	result, err := h.milvus.Query(context.Background(), milvusclient.NewQueryOption(collectionName).
		WithFilter(`id != ""`).
		WithOutputFields(relativePathField, "docId", "title", "pinned", "rank").
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		h.t.Fatalf("query typed rows from %s: %v", collectionName, err)
	}
	rows := make(map[string]string, result.ResultCount)
	for rowIndex := range result.ResultCount {
		values := make([]string, 0, 4)
		for _, columnName := range []string{"docId", "title", "pinned", "rank"} {
			values = append(values, columnName+"="+parityColumnValue(h.t, result.GetColumn(columnName), columnName, rowIndex))
		}
		rows[parityColumnValue(h.t, result.GetColumn(relativePathField), relativePathField, rowIndex)] = strings.Join(values, " ")
	}
	return rows
}

func (h *harness) sendGeneric(collectionID string, rows []*pb.CollectionRow, manifest map[string]string, reconcile pb.CollectionReconcileMode, backfill bool, force bool) (*pb.UpsertCollectionItemsStreamResponse, error) {
	h.t.Helper()
	stream, err := h.client.UpsertCollectionItemsStream(correlatedContext())
	if err != nil {
		h.t.Fatalf("open UpsertCollectionItemsStream returned error: %v", err)
	}
	fingerprints := make([]*pb.CollectionItemFingerprint, 0, len(manifest))
	for itemID, value := range manifest {
		fingerprints = append(fingerprints, &pb.CollectionItemFingerprint{ItemId: itemID, Fingerprint: value})
	}
	frames := []*pb.UpsertCollectionItemsStreamRequest{
		{Chunk: &pb.UpsertCollectionItemsStreamRequest_Header{Header: &pb.UpsertCollectionItemsHeader{
			CollectionId: collectionID, Client: &pb.ClientInfo{Name: "live-harness"}, ReconcileMode: reconcile, BackfillDelivered: backfill, ForceReexamine: force,
		}}},
		{Chunk: &pb.UpsertCollectionItemsStreamRequest_Rows{Rows: &pb.UpsertCollectionItemsRows{Rows: rows}}},
		{Chunk: &pb.UpsertCollectionItemsStreamRequest_Manifest{Manifest: &pb.UpsertCollectionItemsManifest{Manifest: fingerprints}}},
	}
	for _, frame := range frames {
		if err := stream.Send(frame); err != nil {
			break
		}
	}
	return stream.CloseAndRecv()
}

func (h *harness) parityCheckpoint(codebaseID string) map[string]string {
	h.t.Helper()
	snapshot, err := merkle.ReadSnapshot(filepath.Join(h.config.MerkleDir, codebaseID+".json"))
	if err != nil {
		h.t.Fatalf("read checkpoint for %s: %v", codebaseID, err)
	}
	return snapshot.Files
}

func parityColumnValue(t *testing.T, valueColumn column.Column, columnName string, rowIndex int) string {
	t.Helper()
	if valueColumn == nil {
		t.Fatalf("query omitted column %s", columnName)
	}
	isNull, err := valueColumn.IsNull(rowIndex)
	if err != nil {
		t.Fatalf("read null state of %s at %d: %v", columnName, rowIndex, err)
	}
	if isNull {
		return "null"
	}
	value, err := valueColumn.Get(rowIndex)
	if err != nil {
		t.Fatalf("read %s at %d: %v", columnName, rowIndex, err)
	}
	return fmt.Sprint(value)
}

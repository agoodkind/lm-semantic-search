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
)

var maintenanceStoredFields = []string{
	relativePathField,
	"contentHash",
	"embeddingModel",
	"metadata",
	"startLine",
	"endLine",
	"fileExtension",
	"splitPart",
	"itemId",
	"parentId",
	"category",
	"source",
	"location",
	"hidden",
	"created",
	"sequence",
	"tag",
}

type maintenanceRow map[string]string

func TestGenericCollectionMaintenanceLive(t *testing.T) {
	h := newSearchFixtureHarness(t)
	corpus := buildParityCorpus()
	itemA := bulkItemID(0)
	itemB := bulkItemID(1)
	itemC := bulkItemID(2)
	items := map[string][]*pb.CollectionRow{itemA: corpus.items[itemA], itemB: corpus.items[itemB], itemC: corpus.items[itemC]}
	for _, itemID := range []string{itemA, itemC} {
		for _, row := range items[itemID] {
			for _, scalar := range row.Scalars {
				if scalar.Column == "location" {
					scalar.Value = &pb.CollectionScalarValue_StringValue{StringValue: ""}
				}
			}
		}
	}
	requireCompleted(t, h.upsert(items, pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_RETAIN, false, false), "generic ingest")
	before := h.maintenanceRows(h.collectionName)
	checkpoint := h.parityCheckpoint(h.codebaseID)
	rowsA := maintenanceRowIDs(before, itemA)
	rowsB := maintenanceRowIDs(before, itemB)
	rowsC := maintenanceRowIDs(before, itemC)
	if len(rowsA) != 2 || len(rowsB) != 2 || len(rowsC) != 2 {
		t.Fatalf("stored row counts = %d/%d/%d, want 2/2/2", len(rowsA), len(rowsB), len(rowsC))
	}
	var fullTextBefore []string
	if h.config.HybridMode {
		fullTextBefore = h.fullTextHits(h.collectionName)
		if !containsAny(fullTextBefore, rowsA) {
			t.Fatal("full-text search matched no row of item A")
		}
	}
	columns := []string{"location", "hidden"}
	updates := []*pb.BackfillCollectionItem{maintenanceItem(itemA, "/work/filled-a", true), maintenanceItem(itemB, "/work/renamed", true)}
	h.requireBackfillCounts("dry run", h.backfillCollection(h.collectionID, columns, true, updates), len(rowsA), len(rowsC))
	h.requireMaintenanceRows("dry run", h.collectionName, before, nil)
	h.requireBackfillCounts("backfill", h.backfillCollection(h.collectionID, columns, false, updates), len(rowsA), len(rowsC))
	changes := make(map[string]maintenanceRow, len(rowsA))
	for _, rowID := range rowsA {
		changes[rowID] = maintenanceRow{"location": "/work/filled-a"}
	}
	h.requireMaintenanceRows("backfill", h.collectionName, before, changes)
	if h.config.HybridMode {
		if got := h.fullTextHits(h.collectionName); !reflect.DeepEqual(got, fullTextBefore) {
			t.Fatalf("full-text hits changed = %v, want %v", got, fullTextBefore)
		}
	}
	h.requireBackfillCounts("backfill complete", h.backfillCollection(h.collectionID, columns, true, updates), 0, len(rowsC))
	after := h.maintenanceRows(h.collectionName)
	requireCompleted(t, h.waitJob(h.deleteCollectionItem(h.collectionID, itemA)), "generic delete")
	kept := make(map[string]maintenanceRow, len(after))
	for rowID, row := range after {
		if row["itemId"] != itemA {
			kept[rowID] = row
		}
	}
	h.requireMaintenanceRows("delete", h.collectionName, kept, nil)
	if got := h.parityCheckpoint(h.codebaseID); !reflect.DeepEqual(got, checkpoint) {
		t.Fatalf("checkpoint changed = %v, want %v", got, checkpoint)
	}
}

func maintenanceItem(itemID string, location string, hidden bool) *pb.BackfillCollectionItem {
	return &pb.BackfillCollectionItem{
		ItemId: itemID,
		Scalars: []*pb.CollectionScalarValue{
			{Column: "location", Value: &pb.CollectionScalarValue_StringValue{StringValue: location}},
			{Column: "hidden", Value: &pb.CollectionScalarValue_BoolValue{BoolValue: hidden}},
		},
	}
}

func maintenanceRowIDs(rows map[string]maintenanceRow, itemID string) []string {
	ids := make([]string, 0)
	for rowID, row := range rows {
		if row["itemId"] == itemID {
			ids = append(ids, rowID)
		}
	}
	return ids
}

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

func (h *harness) deleteCollectionItem(collectionID string, itemID string) string {
	h.t.Helper()
	response, err := h.client.DeleteCollectionItem(correlatedContext(), &pb.DeleteCollectionItemRequest{CollectionId: collectionID, ItemId: itemID, Client: &pb.ClientInfo{Name: "live-harness"}})
	if err != nil {
		h.t.Fatalf("DeleteCollectionItem returned error: %v", err)
	}
	return response.GetJobId()
}

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

const fullTextHitLimit = 1000

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

func containsAny(values []string, candidates []string) bool {
	for _, candidate := range candidates {
		if slices.Contains(values, candidate) {
			return true
		}
	}
	return false
}

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

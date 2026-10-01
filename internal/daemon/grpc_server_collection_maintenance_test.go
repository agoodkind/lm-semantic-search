package daemon

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/model"
	"google.golang.org/genproto/googleapis/rpc/errdetails"
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

// embeddingGate blocks every embeddings request of a local test embedder while
// the gate is armed, until open runs.
type embeddingGate struct {
	armed    atomic.Bool
	released chan struct{}
	once     sync.Once
}

func (gate *embeddingGate) open() {
	gate.once.Do(func() { close(gate.released) })
}

// newGatedEmbeddingServer starts a local embedding server that answers like
// newTestEmbeddingServer. While gate is armed, it blocks each embeddings request
// until the gate opens.
func newGatedEmbeddingServer(t *testing.T, gate *embeddingGate) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if gate.armed.Load() && strings.HasSuffix(request.URL.Path, "/embeddings") {
			<-gate.released
		}
		testEmbeddingHandler(writer, request)
	}))
	t.Cleanup(server.Close)
	return server
}

// requireActiveJobConflict requires err to be the refusal of an active job:
// FailedPrecondition with ErrorInfo reason active_job_conflict and the active
// job id in ErrorInfo metadata key active_job_id.
func requireActiveJobConflict(t *testing.T, label string, err error, activeJobID string) {
	t.Helper()
	grpcStatus, ok := status.FromError(err)
	if !ok || grpcStatus.Code() != codes.FailedPrecondition {
		t.Fatalf("%s returned %v, want FailedPrecondition", label, err)
	}
	var errorInfo *errdetails.ErrorInfo
	for _, detail := range grpcStatus.Details() {
		if info, isErrorInfo := detail.(*errdetails.ErrorInfo); isErrorInfo {
			errorInfo = info
		}
	}
	if errorInfo == nil {
		t.Fatalf("%s status %v has no ErrorInfo detail", label, err)
	}
	if errorInfo.GetReason() != "active_job_conflict" {
		t.Fatalf("%s ErrorInfo reason = %q, want active_job_conflict", label, errorInfo.GetReason())
	}
	if got := errorInfo.GetMetadata()["active_job_id"]; got != activeJobID {
		t.Fatalf("%s ErrorInfo active_job_id = %q, want %q (metadata %v)", label, got, activeJobID, errorInfo.GetMetadata())
	}
	if !strings.Contains(grpcStatus.Message(), "conflicting active job "+activeJobID) {
		t.Fatalf("%s message = %q, want the conflicting active job text", label, grpcStatus.Message())
	}
}

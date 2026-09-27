package daemon

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// storedLocalRow is one persisted local vector row, decoded from the local
// store's metadata file. Line is the row's exact persisted JSON line.
type storedLocalRow struct {
	ID             string                       `json:"id"`
	RelativePath   string                       `json:"relativePath"`
	Content        string                       `json:"content"`
	ConversationID string                       `json:"conversationId"`
	Role           string                       `json:"role"`
	SplitPart      int32                        `json:"splitPart"`
	Scalars        map[string]model.ScalarValue `json:"scalars"`
	Line           string                       `json:"-"`
}

// localRows reads every stored row of a local collection, sorted by relative
// path and id.
func (daemon *offlineCollectionDaemon) localRows(collectionName string) []storedLocalRow {
	daemon.t.Helper()
	path := filepath.Join(daemon.config.StateRoot, "localvec", collectionName, "metadata.jsonl")
	content, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		daemon.t.Fatalf("read local rows: %v", err)
	}
	rows := make([]storedLocalRow, 0)
	for _, line := range strings.Split(strings.TrimSpace(string(content)), "\n") {
		if line == "" {
			continue
		}
		var row storedLocalRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			daemon.t.Fatalf("decode local row: %v", err)
		}
		row.Line = line
		rows = append(rows, row)
	}
	sort.Slice(rows, func(first int, second int) bool {
		return rows[first].RelativePath+rows[first].ID < rows[second].RelativePath+rows[second].ID
	})
	return rows
}

func (daemon *offlineCollectionDaemon) checkpointFiles(codebaseID string) map[string]string {
	daemon.t.Helper()
	snapshot, err := merkle.ReadSnapshot(daemon.checkpointPath(codebaseID))
	if err != nil {
		daemon.t.Fatalf("ReadSnapshot returned error: %v", err)
	}
	return snapshot.Files
}

// sendCollectionStream sends frames on one UpsertCollectionItemsStream and
// returns the response or the stream's final status error.
func (daemon *offlineCollectionDaemon) sendCollectionStream(frames []*pb.UpsertCollectionItemsStreamRequest) (*pb.UpsertCollectionItemsStreamResponse, error) {
	daemon.t.Helper()
	stream, err := daemon.client.UpsertCollectionItemsStream(grpcutil.WithCorrelation(context.Background()))
	if err != nil {
		daemon.t.Fatalf("open UpsertCollectionItemsStream returned error: %v", err)
	}
	for _, frame := range frames {
		if sendErr := stream.Send(frame); sendErr != nil {
			if errors.Is(sendErr, io.EOF) {
				break
			}
			daemon.t.Fatalf("send collection frame returned error: %v", sendErr)
		}
	}
	return stream.CloseAndRecv()
}

// upsertItems streams rows and a manifest through the generic RPC and waits
// for the ingest job to complete.
func (daemon *offlineCollectionDaemon) upsertItems(header *pb.UpsertCollectionItemsHeader, rows []*pb.CollectionRow, manifest map[string]string) {
	daemon.t.Helper()
	response, err := daemon.sendCollectionStream(collectionFrames(header, rows, manifest))
	if err != nil {
		daemon.t.Fatalf("UpsertCollectionItemsStream returned error: %v", err)
	}
	job := waitForRPCJobTerminal(daemon.t, daemon.client, response.GetJobId())
	if job.GetState() != string(model.JobStateCompleted) {
		daemon.t.Fatalf("collection ingest job state = %q, want completed: %+v", job.GetState(), job.GetError())
	}
}

// upsertConversation sends documents and a manifest with the conversation
// upsert RPC and waits for the ingest job to complete.
func (daemon *offlineCollectionDaemon) upsertConversation(collectionID string, documents []*pb.ConversationDocument, manifest map[string]string, backfill bool) {
	daemon.t.Helper()
	stream, err := daemon.client.UpsertConversationDocumentsStream(grpcutil.WithCorrelation(context.Background()))
	if err != nil {
		daemon.t.Fatalf("open UpsertConversationDocumentsStream returned error: %v", err)
	}
	fingerprints := make([]*pb.ConversationFingerprint, 0, len(manifest))
	for conversationID, fingerprint := range manifest {
		fingerprints = append(fingerprints, &pb.ConversationFingerprint{ConversationId: conversationID, Fingerprint: fingerprint})
	}
	for _, chunk := range []*pb.UpsertConversationDocumentsChunk{
		{Chunk: &pb.UpsertConversationDocumentsChunk_Header{Header: &pb.UpsertConversationDocumentsHeader{CollectionId: collectionID, BackfillDelivered: backfill}}},
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
		daemon.t.Fatalf("conversation ingest job state = %q, want completed: %+v", job.GetState(), job.GetError())
	}
}

// requireEqualConversationRows requires byte-identical stored rows and equal
// checkpoints. One collection is registered by the conversation RPC. The other
// collection is registered by the generic RPC with the conversation declaration.
// It returns the stored rows.
func (daemon *offlineCollectionDaemon) requireEqualConversationRows(step string, conversation *pb.RegisterConversationCollectionResponse, generic *pb.RegisterCollectionResponse) []storedLocalRow {
	daemon.t.Helper()
	conversationRows := daemon.localRows(conversation.GetCollectionName())
	genericRows := daemon.localRows(generic.GetCollectionName())
	if len(conversationRows) != len(genericRows) {
		daemon.t.Fatalf("%s: row paths differ: conversation %v, generic %v", step, rowPaths(conversationRows), rowPaths(genericRows))
	}
	for index := range conversationRows {
		if conversationRows[index].Line != genericRows[index].Line {
			daemon.t.Fatalf("%s: stored row %s differs between the conversation and generic streams", step, conversationRows[index].RelativePath)
		}
	}
	if !reflect.DeepEqual(daemon.checkpointFiles(conversation.GetCodebaseId()), daemon.checkpointFiles(generic.GetCodebaseId())) {
		daemon.t.Fatalf("%s: checkpoints differ between the conversation and generic streams", step)
	}
	return conversationRows
}

// storeBlankTextRow writes one message text row with a single space as its
// content straight into a collection's store. An older pipeline wrote such a
// row for message text it did not keep, and current ingest never writes one.
func (daemon *offlineCollectionDaemon) storeBlankTextRow(collectionID string, conversationID string, messageIndex int32) {
	daemon.t.Helper()
	blank := model.StoredChunk{
		Content:              " ",
		RelativePath:         "conv/" + conversationID + "/" + strconv.Itoa(int(messageIndex)),
		StartLine:            0,
		EndLine:              0,
		Language:             "",
		FileExtension:        "",
		ConversationID:       conversationID,
		ParentConversationID: "",
		MessageIndex:         messageIndex,
		Role:                 "user",
		TimestampUnix:        0,
		WorkspaceRoot:        "",
		Archived:             false,
		SplitPart:            0,
		SplitPartRecorded:    true,
		LoadRules:            "",
		Scalars:              nil,
		Score:                0,
	}
	// The embedder skips blank content. The reuse map supplies the row's vector.
	reuse := map[string][]float32{semantic.ContentVectorKey(blank.Content): {1, 0, 0}}
	err := daemon.manager.semantic.Reindex(context.Background(), conversationCanonicalPath(collectionID), []model.StoredChunk{blank}, semantic.RemovePaths(nil), nil, reuse, semantic.ConversationColumns())
	if err != nil {
		daemon.t.Fatalf("store blank text row in %s: %v", collectionID, err)
	}
}

// conversationRowScalars returns the scalar values of one generic row of the
// parity transcripts, which are Claude conversations in /work with the
// rules-v1 load rules.
func conversationRowScalars(messageIndex int64, role string, timestamp int64) []*pb.CollectionScalarValue {
	return []*pb.CollectionScalarValue{
		stringScalar("role", role), int64Scalar("messageIndex", messageIndex), int64Scalar("timestampUnix", timestamp),
		stringScalar("workspaceRoot", "/work"), stringScalar("loadRules", "rules-v1"), stringScalar("provider", "claude"),
		stringScalar("parentConversationId", ""), boolScalar("archived", false),
	}
}

func (daemon *offlineCollectionDaemon) syncItems(collectionID string, manifest map[string]string) []string {
	daemon.t.Helper()
	response, err := daemon.client.SyncCollectionManifest(grpcutil.WithCorrelation(context.Background()), &pb.SyncCollectionManifestRequest{
		CollectionId: collectionID,
		Manifest:     collectionFingerprints(manifest),
		Client:       &pb.ClientInfo{Name: "collection-test"},
	})
	if err != nil {
		daemon.t.Fatalf("SyncCollectionManifest returned error: %v", err)
	}
	return response.GetNeededItemIds()
}

func collectionHeader(collectionID string, mode pb.CollectionReconcileMode, backfill bool, force bool) *pb.UpsertCollectionItemsHeader {
	return &pb.UpsertCollectionItemsHeader{
		CollectionId:      collectionID,
		Client:            &pb.ClientInfo{Name: "collection-test"},
		ReconcileMode:     mode,
		BackfillDelivered: backfill,
		ForceReexamine:    force,
	}
}

func collectionFrames(header *pb.UpsertCollectionItemsHeader, rows []*pb.CollectionRow, manifest map[string]string) []*pb.UpsertCollectionItemsStreamRequest {
	frames := []*pb.UpsertCollectionItemsStreamRequest{
		{Chunk: &pb.UpsertCollectionItemsStreamRequest_Header{Header: header}},
		{Chunk: &pb.UpsertCollectionItemsStreamRequest_Rows{Rows: &pb.UpsertCollectionItemsRows{Rows: rows}}},
	}
	if manifest != nil {
		frames = append(frames, &pb.UpsertCollectionItemsStreamRequest{Chunk: &pb.UpsertCollectionItemsStreamRequest_Manifest{Manifest: &pb.UpsertCollectionItemsManifest{Manifest: collectionFingerprints(manifest)}}})
	}
	return frames
}

func collectionFingerprints(manifest map[string]string) []*pb.CollectionItemFingerprint {
	fingerprints := make([]*pb.CollectionItemFingerprint, 0, len(manifest))
	for itemID, fingerprint := range manifest {
		fingerprints = append(fingerprints, &pb.CollectionItemFingerprint{ItemId: itemID, Fingerprint: fingerprint})
	}
	return fingerprints
}

func stringScalar(column string, value string) *pb.CollectionScalarValue {
	return &pb.CollectionScalarValue{Column: column, Value: &pb.CollectionScalarValue_StringValue{StringValue: value}}
}

func boolScalar(column string, value bool) *pb.CollectionScalarValue {
	return &pb.CollectionScalarValue{Column: column, Value: &pb.CollectionScalarValue_BoolValue{BoolValue: value}}
}

func int64Scalar(column string, value int64) *pb.CollectionScalarValue {
	return &pb.CollectionScalarValue{Column: column, Value: &pb.CollectionScalarValue_Int64Value{Int64Value: value}}
}

func nullScalar(column string) *pb.CollectionScalarValue {
	return &pb.CollectionScalarValue{Column: column, Value: nil}
}

// documentRow builds one row of the documentScalars declaration.
func documentRow(rowKey string, itemID string, text string, rank int64) *pb.CollectionRow {
	return &pb.CollectionRow{
		RowKey: rowKey,
		ItemId: itemID,
		Text:   text,
		Scalars: []*pb.CollectionScalarValue{
			stringScalar("title", "title of "+rowKey),
			boolScalar("pinned", rank%2 == 0),
			int64Scalar("rank", rank),
		},
	}
}

func rowPaths(rows []storedLocalRow) []string {
	paths := make([]string, 0, len(rows))
	for _, row := range rows {
		paths = append(paths, row.RelativePath)
	}
	return paths
}

func distinctRowPaths(rows []storedLocalRow) []string {
	paths := rowPaths(rows)
	sort.Strings(paths)
	return slices.Compact(paths)
}

// assembleParts concatenates the content of every row under prefix in part
// order, then split position order. The local store splits a part again at the
// embedding budget and keeps the part path on each piece.
func assembleParts(rows []storedLocalRow, prefix string) string {
	parts := make([]storedLocalRow, 0)
	for _, row := range rows {
		if strings.HasPrefix(row.RelativePath, prefix) {
			parts = append(parts, row)
		}
	}
	partIndex := func(row storedLocalRow) int {
		index, _ := strconv.Atoi(strings.TrimPrefix(row.RelativePath, prefix))
		return index
	}
	sort.Slice(parts, func(first int, second int) bool {
		if partIndex(parts[first]) != partIndex(parts[second]) {
			return partIndex(parts[first]) < partIndex(parts[second])
		}
		return parts[first].SplitPart < parts[second].SplitPart
	})
	var assembled strings.Builder
	for _, part := range parts {
		assembled.WriteString(part.Content)
	}
	return assembled.String()
}

// storedPartContent concatenates the content of every row stored at exactly
// relativePath in split position order. The local store splits one part again
// at the embedding budget and keeps the part path on each piece.
func storedPartContent(rows []storedLocalRow, relativePath string) string {
	pieces := make([]storedLocalRow, 0)
	for _, row := range rows {
		if row.RelativePath == relativePath {
			pieces = append(pieces, row)
		}
	}
	sort.Slice(pieces, func(first int, second int) bool {
		return pieces[first].SplitPart < pieces[second].SplitPart
	})
	var content strings.Builder
	for _, piece := range pieces {
		content.WriteString(piece.Content)
	}
	return content.String()
}

func rowByPath(t *testing.T, rows []storedLocalRow, relativePath string) storedLocalRow {
	t.Helper()
	for _, row := range rows {
		if row.RelativePath == relativePath {
			return row
		}
	}
	t.Fatalf("no stored row at %s in %v", relativePath, rowPaths(rows))
	return storedLocalRow{}
}

// TestUpsertCollectionItemsStreamStoresTypedRows registers a generic
// collection, syncs a manifest, and upserts rows with typed and null scalar
// values and one text longer than the split budget. The stored rows use the
// row keys and part suffixes, store the typed values with the item id column
// set from item_id, and the checkpoint records the manifest. An unchanged
// manifest then needs no items, the conversation upsert RPC rejects the
// generic collection, and the collection registers again after a restart.
func TestUpsertCollectionItemsStreamStoresTypedRows(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)
	registered, err := daemon.registerCollection("docs-typed", "docId", documentScalars())
	if err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}
	manifest := map[string]string{"doc-a": "fp-a1", "doc-b": "fp-b1"}
	if needed := daemon.syncItems("docs-typed", manifest); !slices.Equal(needed, []string{"doc-a", "doc-b"}) {
		t.Fatalf("needed before ingest = %v, want [doc-a doc-b]", needed)
	}

	longText := strings.Repeat("long body sentence. ", 4000)
	nullTitle := &pb.CollectionRow{
		RowKey:  "doc-b/summary",
		ItemId:  "doc-b",
		Text:    "summary of the second document",
		Scalars: []*pb.CollectionScalarValue{nullScalar("title"), int64Scalar("rank", 7), stringScalar("docId", "doc-b")},
	}
	daemon.upsertItems(
		collectionHeader("docs-typed", pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_UNSPECIFIED, false, false),
		[]*pb.CollectionRow{documentRow("doc-a/title", "doc-a", "first document heading", 2), documentRow("doc-a/body", "doc-a", longText, 3), nullTitle},
		manifest,
	)

	rows := daemon.localRows(registered.GetCollectionName())
	budget := daemon.manager.conversationChunkByteBudget
	partCount := (len(longText) + budget - 1) / budget
	wantPaths := make([]string, 0, partCount+2)
	for part := range partCount {
		wantPaths = append(wantPaths, "doc-a/body/"+strconv.Itoa(part))
	}
	wantPaths = append(wantPaths, "doc-a/title", "doc-b/summary")
	sort.Strings(wantPaths)
	if got := distinctRowPaths(rows); partCount < 2 || !slices.Equal(got, wantPaths) {
		t.Fatalf("stored row paths = %v, want %v", got, wantPaths)
	}
	if assembled := assembleParts(rows, "doc-a/body/"); assembled != longText {
		t.Fatal("stored body parts do not reassemble the delivered text")
	}
	title := rowByPath(t, rows, "doc-a/title")
	wantTitleScalars := map[string]model.ScalarValue{
		"docId":  {Type: model.ScalarTypeString, String: "doc-a"},
		"title":  {Type: model.ScalarTypeString, String: "title of doc-a/title"},
		"pinned": {Type: model.ScalarTypeBool, Bool: true},
		"rank":   {Type: model.ScalarTypeInt64, Int64: 2},
	}
	if !reflect.DeepEqual(title.Scalars, wantTitleScalars) {
		t.Fatalf("doc-a/title scalars = %+v, want %+v", title.Scalars, wantTitleScalars)
	}
	if title.ConversationID != "" || title.Role != "" {
		t.Fatalf("generic row stored conversation fields %q/%q, want none", title.ConversationID, title.Role)
	}
	summary := rowByPath(t, rows, "doc-b/summary")
	if value := summary.Scalars["title"]; !value.Null {
		t.Fatalf("doc-b/summary title = %+v, want null", value)
	}
	if _, present := summary.Scalars["pinned"]; present {
		t.Fatalf("doc-b/summary stored an unset pinned value: %+v", summary.Scalars)
	}
	if got := daemon.checkpointFiles(registered.GetCodebaseId()); !reflect.DeepEqual(got, manifest) {
		t.Fatalf("checkpoint = %v, want %v", got, manifest)
	}
	if needed := daemon.syncItems("docs-typed", manifest); len(needed) != 0 {
		t.Fatalf("needed after ingest = %v, want none", needed)
	}

	conversationStream, err := daemon.client.UpsertConversationDocumentsStream(grpcutil.WithCorrelation(context.Background()))
	if err != nil {
		t.Fatalf("open UpsertConversationDocumentsStream returned error: %v", err)
	}
	for _, chunk := range []*pb.UpsertConversationDocumentsChunk{
		{Chunk: &pb.UpsertConversationDocumentsChunk_Header{Header: &pb.UpsertConversationDocumentsHeader{CollectionId: "docs-typed"}}},
		{Chunk: &pb.UpsertConversationDocumentsChunk_Documents{Documents: &pb.UpsertConversationDocumentsDocuments{Documents: []*pb.ConversationDocument{{ConversationId: "doc-a", Role: "user", Text: "x"}}}}},
	} {
		if err := conversationStream.Send(chunk); err != nil {
			t.Fatalf("send conversation chunk returned error: %v", err)
		}
	}
	if _, err := conversationStream.CloseAndRecv(); status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("conversation upsert into a generic collection returned %v, want FailedPrecondition", err)
	}

	daemon.restart(nil)
	if _, err := daemon.registerCollection("docs-typed", "docId", documentScalars()); err != nil {
		t.Fatalf("RegisterCollection after ingest and restart returned error: %v", err)
	}
	if needed := daemon.syncItems("docs-typed", manifest); len(needed) != 0 {
		t.Fatalf("needed after restart = %v, want none", needed)
	}
}

// TestUpsertCollectionItemsRetainAndAuthoritative ingests two items, then
// sends a manifest that omits one. Retain keeps the omitted item's rows and
// checkpoint entry. Authoritative removes them by item id. An authoritative
// stream without a manifest fails before it queues a job.
func TestUpsertCollectionItemsRetainAndAuthoritative(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)
	registered, err := daemon.registerCollection("docs-absence", "docId", documentScalars())
	if err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}
	unspecified := pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_UNSPECIFIED
	daemon.upsertItems(
		collectionHeader("docs-absence", unspecified, false, false),
		[]*pb.CollectionRow{documentRow("a/0", "doc-a", "alpha", 1), documentRow("b/0", "doc-b", "bravo", 2), documentRow("b/1", "doc-b", "bravo two", 3)},
		map[string]string{"doc-a": "fp-a1", "doc-b": "fp-b1"},
	)

	daemon.upsertItems(
		collectionHeader("docs-absence", pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_RETAIN, false, false),
		[]*pb.CollectionRow{documentRow("a/1", "doc-a", "alpha appended", 4)},
		map[string]string{"doc-a": "fp-a2"},
	)
	rows := daemon.localRows(registered.GetCollectionName())
	if got := rowPaths(rows); !slices.Equal(got, []string{"a/0", "a/1", "b/0", "b/1"}) {
		t.Fatalf("rows after retain = %v, want [a/0 a/1 b/0 b/1]", got)
	}
	if got := daemon.checkpointFiles(registered.GetCodebaseId()); !reflect.DeepEqual(got, map[string]string{"doc-a": "fp-a2", "doc-b": "fp-b1"}) {
		t.Fatalf("checkpoint after retain = %v", got)
	}

	authoritative := pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_AUTHORITATIVE
	_, err = daemon.sendCollectionStream(collectionFrames(collectionHeader("docs-absence", authoritative, false, false), nil, nil))
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("authoritative upsert without a manifest returned %v, want InvalidArgument", err)
	}

	daemon.upsertItems(collectionHeader("docs-absence", authoritative, false, false), nil, map[string]string{"doc-a": "fp-a2"})
	rows = daemon.localRows(registered.GetCollectionName())
	if got := rowPaths(rows); !slices.Equal(got, []string{"a/0", "a/1"}) {
		t.Fatalf("rows after authoritative = %v, want [a/0 a/1]", got)
	}
	if got := daemon.checkpointFiles(registered.GetCodebaseId()); !reflect.DeepEqual(got, map[string]string{"doc-a": "fp-a2"}) {
		t.Fatalf("checkpoint after authoritative = %v", got)
	}
}

// TestUpsertCollectionItemsBackfillAndForce covers the delivery flags. An
// unchanged fingerprint skips a delivered item. Backfill adds a delivered row
// family the store lacks and keeps present rows. A changed fingerprint adds
// absent families and keeps a present family with new text. Force replaces
// every row of the item, which removes a row the delivery no longer lists.
func TestUpsertCollectionItemsBackfillAndForce(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)
	registered, err := daemon.registerCollection("docs-flags", "docId", documentScalars())
	if err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}
	unspecified := pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_UNSPECIFIED
	daemon.upsertItems(collectionHeader("docs-flags", unspecified, false, false), []*pb.CollectionRow{documentRow("a/0", "doc-a", "original zero", 1), documentRow("a/1", "doc-a", "original one", 2)}, map[string]string{"doc-a": "fp-1"})

	withExtra := []*pb.CollectionRow{documentRow("a/0", "doc-a", "original zero", 1), documentRow("a/1", "doc-a", "original one", 2), documentRow("a/2", "doc-a", "added two", 3)}
	daemon.upsertItems(collectionHeader("docs-flags", unspecified, false, false), withExtra, map[string]string{"doc-a": "fp-1"})
	rows := daemon.localRows(registered.GetCollectionName())
	if got := rowPaths(rows); !slices.Equal(got, []string{"a/0", "a/1"}) {
		t.Fatalf("rows after an unchanged fingerprint = %v, want [a/0 a/1]", got)
	}

	daemon.upsertItems(collectionHeader("docs-flags", unspecified, true, false), withExtra, map[string]string{"doc-a": "fp-1"})
	rows = daemon.localRows(registered.GetCollectionName())
	if got := rowPaths(rows); !slices.Equal(got, []string{"a/0", "a/1", "a/2"}) {
		t.Fatalf("rows after backfill = %v, want [a/0 a/1 a/2]", got)
	}

	changed := []*pb.CollectionRow{documentRow("a/0", "doc-a", "edited zero", 1), documentRow("a/3", "doc-a", "added three", 4)}
	daemon.upsertItems(collectionHeader("docs-flags", unspecified, false, false), changed, map[string]string{"doc-a": "fp-2"})
	rows = daemon.localRows(registered.GetCollectionName())
	if got := rowPaths(rows); !slices.Equal(got, []string{"a/0", "a/1", "a/2", "a/3"}) {
		t.Fatalf("rows after a changed fingerprint = %v, want [a/0 a/1 a/2 a/3]", got)
	}
	if content := rowByPath(t, rows, "a/0").Content; content != "original zero" {
		t.Fatalf("present family a/0 content = %q, want the stored text", content)
	}

	daemon.upsertItems(collectionHeader("docs-flags", unspecified, true, true), changed, map[string]string{"doc-a": "fp-2"})
	rows = daemon.localRows(registered.GetCollectionName())
	if got := rowPaths(rows); !slices.Equal(got, []string{"a/0", "a/3"}) {
		t.Fatalf("rows after force = %v, want [a/0 a/3]", got)
	}
	if content := rowByPath(t, rows, "a/0").Content; content != "edited zero" {
		t.Fatalf("forced a/0 content = %q, want the delivered text", content)
	}
	if got := daemon.checkpointFiles(registered.GetCodebaseId()); !reflect.DeepEqual(got, map[string]string{"doc-a": "fp-2"}) {
		t.Fatalf("checkpoint after force = %v", got)
	}
}

// TestUpsertCollectionItemsContinuationPrefix upserts two rows longer than the
// split budget into a collection with a generic declaration. The row with a
// continuation prefix splits at the budget less the prefix length plus one,
// and every part after the first starts with the prefix and a newline. The row
// without a prefix splits at the full budget and stores its text unchanged.
func TestUpsertCollectionItemsContinuationPrefix(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)
	registered, err := daemon.registerCollection("docs-prefix", "docId", documentScalars())
	if err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}
	budget := daemon.manager.conversationChunkByteBudget
	const prefix = "Section A"
	text := strings.TrimSpace(strings.Repeat("prefixed body sentence. ", 2*budget/24+1))
	prefixed := documentRow("doc-a/prefixed", "doc-a", text, 1)
	prefixed.ContinuationPrefix = prefix
	plain := documentRow("doc-a/plain", "doc-a", text, 2)
	daemon.upsertItems(
		collectionHeader("docs-prefix", pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_UNSPECIFIED, false, false),
		[]*pb.CollectionRow{prefixed, plain},
		map[string]string{"doc-a": "fp-a1"},
	)

	rows := daemon.localRows(registered.GetCollectionName())
	if got := len(storedPartContent(rows, "doc-a/plain/0")); got != budget {
		t.Fatalf("row without a prefix stores a first part of %d bytes, want the full budget, %d", got, budget)
	}
	if assembleParts(rows, "doc-a/plain/") != text {
		t.Fatal("row without a prefix does not reassemble its text")
	}
	firstPart := storedPartContent(rows, "doc-a/prefixed/0")
	if len(firstPart) != budget-len(prefix)-1 {
		t.Fatalf("row with a prefix stores a first part of %d bytes, want %d", len(firstPart), budget-len(prefix)-1)
	}
	var rebuilt strings.Builder
	rebuilt.WriteString(firstPart)
	for part := 1; ; part++ {
		content := storedPartContent(rows, "doc-a/prefixed/"+strconv.Itoa(part))
		if content == "" {
			break
		}
		remainder, found := strings.CutPrefix(content, prefix+"\n")
		if !found {
			t.Fatalf("part %d of the row with a prefix starts with %.20q, want the prefix line", part, content)
		}
		rebuilt.WriteString(remainder)
	}
	if rebuilt.String() != text {
		t.Fatal("parts of the row with a prefix, without their prefix lines, do not reassemble its text")
	}
}

// TestUpsertCollectionItemsDerivedFingerprintCoversContinuationPrefix upserts
// one row three times without a manifest frame. The engine derives each
// fingerprint from the row. Changing only the continuation prefix gives
// the item a new fingerprint in the checkpoint. The same row without the prefix
// restores the first fingerprint.
func TestUpsertCollectionItemsDerivedFingerprintCoversContinuationPrefix(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)
	registered, err := daemon.registerCollection("docs-derived", "docId", documentScalars())
	if err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}
	header := collectionHeader("docs-derived", pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_UNSPECIFIED, false, false)
	plain := documentRow("a/0", "doc-a", "derived fingerprint body", 1)
	prefixed := documentRow("a/0", "doc-a", "derived fingerprint body", 1)
	prefixed.ContinuationPrefix = "Section A"

	daemon.upsertItems(header, []*pb.CollectionRow{plain}, nil)
	withoutPrefix := daemon.checkpointFiles(registered.GetCodebaseId())["doc-a"]
	daemon.upsertItems(header, []*pb.CollectionRow{prefixed}, nil)
	withPrefix := daemon.checkpointFiles(registered.GetCodebaseId())["doc-a"]
	if withoutPrefix == "" || withPrefix == withoutPrefix {
		t.Fatalf("derived fingerprint with a continuation prefix = %q, and without one = %q; the prefix should change the fingerprint", withPrefix, withoutPrefix)
	}
	daemon.upsertItems(header, []*pb.CollectionRow{plain}, nil)
	if restored := daemon.checkpointFiles(registered.GetCodebaseId())["doc-a"]; restored != withoutPrefix {
		t.Fatalf("derived fingerprint after the prefix is removed = %q, want the first fingerprint %q", restored, withoutPrefix)
	}
}

// TestUpsertCollectionItemsRejectsInvalidRows sends rows that break the saved
// declaration. Each stream fails with InvalidArgument, reports the rejected
// column in ErrorInfo when there is one, and queues no job. An unregistered
// collection fails with NotFound at the header.
func TestUpsertCollectionItemsRejectsInvalidRows(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)
	registered, err := daemon.registerCollection("docs-invalid-rows", "docId", documentScalars())
	if err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}
	header := collectionHeader("docs-invalid-rows", pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_UNSPECIFIED, false, false)
	withScalars := func(scalars ...*pb.CollectionScalarValue) *pb.CollectionRow {
		return &pb.CollectionRow{RowKey: "k/0", ItemId: "doc-a", Text: "text", Scalars: scalars}
	}
	cases := []struct {
		name       string
		rows       []*pb.CollectionRow
		wantColumn string
	}{
		{name: "undeclared column", rows: []*pb.CollectionRow{withScalars(int64Scalar("rank", 1), stringScalar("author", "x"))}, wantColumn: "author"},
		{name: "wrong type", rows: []*pb.CollectionRow{withScalars(stringScalar("rank", "1"))}, wantColumn: "rank"},
		{name: "null in a column that is not nullable", rows: []*pb.CollectionRow{withScalars(nullScalar("rank"))}, wantColumn: "rank"},
		{name: "missing column that is not nullable", rows: []*pb.CollectionRow{withScalars(stringScalar("title", "t"))}, wantColumn: "rank"},
		{name: "item id column conflict", rows: []*pb.CollectionRow{withScalars(int64Scalar("rank", 1), stringScalar("docId", "doc-z"))}, wantColumn: "docId"},
		{name: "string over max length", rows: []*pb.CollectionRow{withScalars(int64Scalar("rank", 1), stringScalar("title", strings.Repeat("t", 513)))}, wantColumn: "title"},
		{name: "column set twice", rows: []*pb.CollectionRow{withScalars(int64Scalar("rank", 1), int64Scalar("rank", 2))}, wantColumn: "rank"},
	}
	for _, testCase := range cases {
		_, err := daemon.sendCollectionStream(collectionFrames(header, testCase.rows, map[string]string{"doc-a": "fp"}))
		requireColumnError(t, err, codes.InvalidArgument, "invalid_argument", testCase.wantColumn)
	}
	duplicateKey := []*pb.CollectionRow{documentRow("k/0", "doc-a", "one", 1), documentRow("k/0", "doc-a", "two", 2)}
	emptyItem := []*pb.CollectionRow{documentRow("k/0", " ", "one", 1)}
	emptyKey := []*pb.CollectionRow{documentRow("", "doc-a", "one", 1)}
	for name, rows := range map[string][]*pb.CollectionRow{"duplicate row key": duplicateKey, "empty item id": emptyItem, "empty row key": emptyKey} {
		if _, err := daemon.sendCollectionStream(collectionFrames(header, rows, map[string]string{"doc-a": "fp"})); status.Code(err) != codes.InvalidArgument {
			t.Fatalf("%s returned %v, want InvalidArgument", name, err)
		}
	}
	if rows := daemon.localRows(registered.GetCollectionName()); len(rows) != 0 {
		t.Fatalf("rejected streams stored %d rows, want 0", len(rows))
	}
	jobs, err := daemon.client.ListJobs(grpcutil.WithCorrelation(context.Background()), &pb.ListJobsRequest{CodebaseId: registered.GetCodebaseId()})
	if err != nil || len(jobs.GetJobs()) != 0 {
		t.Fatalf("rejected streams queued jobs = %v (err %v), want none", jobs.GetJobs(), err)
	}

	_, err = daemon.sendCollectionStream(collectionFrames(collectionHeader("docs-unregistered", pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_UNSPECIFIED, false, false), nil, nil))
	if status.Code(err) != codes.NotFound {
		t.Fatalf("upsert into an unregistered collection returned %v, want NotFound", err)
	}
	if _, err := daemon.client.SyncCollectionManifest(grpcutil.WithCorrelation(context.Background()), &pb.SyncCollectionManifestRequest{CollectionId: "docs-unregistered"}); status.Code(err) != codes.NotFound {
		t.Fatalf("manifest sync for an unregistered collection returned %v, want NotFound", err)
	}
}

// TestUpsertCollectionItemsRejectsLongContinuationPrefix sends a row with a
// continuation prefix one byte over the 1024-byte limit. The stream fails with
// InvalidArgument, the error message includes the limit, and the stream stores
// no row and queues no job. The test accepts a row with a prefix of exactly
// 1024 bytes and waits for the ingest job to complete.
func TestUpsertCollectionItemsRejectsLongContinuationPrefix(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)
	registered, err := daemon.registerCollection("docs-long-prefix", "docId", documentScalars())
	if err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}
	header := collectionHeader("docs-long-prefix", pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_UNSPECIFIED, false, false)
	manifest := map[string]string{"doc-a": "fp-a1"}
	row := documentRow("doc-a/0", "doc-a", "row with a long continuation prefix", 1)

	row.ContinuationPrefix = strings.Repeat("p", 1025)
	_, err = daemon.sendCollectionStream(collectionFrames(header, []*pb.CollectionRow{row}, manifest))
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("row with a 1025-byte continuation prefix returned %v, want InvalidArgument", err)
	}
	if message := status.Convert(err).Message(); !strings.Contains(message, "1024 bytes") {
		t.Fatalf("the error message %q does not include the 1024-byte limit", message)
	}
	if rows := daemon.localRows(registered.GetCollectionName()); len(rows) != 0 {
		t.Fatalf("rejected stream stored %d rows, want 0", len(rows))
	}
	jobs, err := daemon.client.ListJobs(grpcutil.WithCorrelation(context.Background()), &pb.ListJobsRequest{CodebaseId: registered.GetCodebaseId()})
	if err != nil || len(jobs.GetJobs()) != 0 {
		t.Fatalf("rejected stream queued jobs = %d (err %v), want none", len(jobs.GetJobs()), err)
	}

	row.ContinuationPrefix = strings.Repeat("p", 1024)
	daemon.upsertItems(header, []*pb.CollectionRow{row}, manifest)
	if got := rowPaths(daemon.localRows(registered.GetCollectionName())); !slices.Equal(got, []string{"doc-a/0"}) {
		t.Fatalf("rows after an upsert with a 1024-byte continuation prefix = %v, want [doc-a/0]", got)
	}
}

// TestUpsertCollectionItemsStreamOrderAndLimits breaks the frame order and the
// per-frame row bound. Each stream fails with InvalidArgument and stores
// nothing.
func TestUpsertCollectionItemsStreamOrderAndLimits(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)
	registered, err := daemon.registerCollection("docs-limits", "docId", documentScalars())
	if err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}
	header := &pb.UpsertCollectionItemsStreamRequest{Chunk: &pb.UpsertCollectionItemsStreamRequest_Header{Header: collectionHeader("docs-limits", pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_UNSPECIFIED, false, false)}}
	rowsFrame := func(rows ...*pb.CollectionRow) *pb.UpsertCollectionItemsStreamRequest {
		return &pb.UpsertCollectionItemsStreamRequest{Chunk: &pb.UpsertCollectionItemsStreamRequest_Rows{Rows: &pb.UpsertCollectionItemsRows{Rows: rows}}}
	}
	manifestFrame := &pb.UpsertCollectionItemsStreamRequest{Chunk: &pb.UpsertCollectionItemsStreamRequest_Manifest{Manifest: &pb.UpsertCollectionItemsManifest{Manifest: collectionFingerprints(map[string]string{"doc-a": "fp"})}}}
	oneRow := documentRow("k/0", "doc-a", "text", 1)

	tooManyRows := make([]*pb.CollectionRow, 0, maxCollectionRowsPerFrame+1)
	for index := range maxCollectionRowsPerFrame + 1 {
		tooManyRows = append(tooManyRows, documentRow("k/"+strconv.Itoa(index), "doc-a", "t", 1))
	}

	cases := map[string][]*pb.UpsertCollectionItemsStreamRequest{
		"rows before header":       {rowsFrame(oneRow), header},
		"duplicate header":         {header, header},
		"rows after manifest":      {header, manifestFrame, rowsFrame(oneRow)},
		"duplicate manifest":       {header, rowsFrame(oneRow), manifestFrame, manifestFrame},
		"too many rows in a frame": {header, rowsFrame(tooManyRows...)},
		"no header":                nil,
	}
	for name, frames := range cases {
		if _, err := daemon.sendCollectionStream(frames); status.Code(err) != codes.InvalidArgument {
			t.Fatalf("%s returned %v, want InvalidArgument", name, err)
		}
	}
	if rows := daemon.localRows(registered.GetCollectionName()); len(rows) != 0 {
		t.Fatalf("rejected streams stored %d rows, want 0", len(rows))
	}
}

// TestUpsertCollectionItemsStreamRejectsStreamOverByteLimit sends 16 MiB row
// frames until the stream exceeds the 512 MiB row byte limit. The stream fails
// with InvalidArgument, the error message includes the limit, and the stream
// stores no row and queues no job. The test does not run in parallel with
// other tests, because the daemon buffers about 512 MiB before it rejects the
// stream.
func TestUpsertCollectionItemsStreamRejectsStreamOverByteLimit(t *testing.T) {
	daemon := newOfflineCollectionDaemon(t)
	registered, err := daemon.registerCollection("docs-over-limit", "docId", documentScalars())
	if err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}
	frameText := strings.Repeat("b", 16<<20)
	frames := []*pb.UpsertCollectionItemsStreamRequest{{Chunk: &pb.UpsertCollectionItemsStreamRequest_Header{Header: collectionHeader("docs-over-limit", pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_UNSPECIFIED, false, false)}}}
	for index := range maxCollectionStreamBytes/len(frameText) + 1 {
		row := documentRow("big/"+strconv.Itoa(index), "doc-a", frameText, 1)
		frames = append(frames, &pb.UpsertCollectionItemsStreamRequest{Chunk: &pb.UpsertCollectionItemsStreamRequest_Rows{Rows: &pb.UpsertCollectionItemsRows{Rows: []*pb.CollectionRow{row}}}})
	}
	_, err = daemon.sendCollectionStream(frames)
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("stream over the row byte limit returned %v, want InvalidArgument", status.Code(err))
	}
	if message := status.Convert(err).Message(); !strings.Contains(message, "512 MiB") {
		t.Fatal("the error message of a stream over the row byte limit does not include the 512 MiB limit")
	}
	if rows := daemon.localRows(registered.GetCollectionName()); len(rows) != 0 {
		t.Fatalf("rejected stream stored %d rows, want 0", len(rows))
	}
	jobs, err := daemon.client.ListJobs(grpcutil.WithCorrelation(context.Background()), &pb.ListJobsRequest{CodebaseId: registered.GetCodebaseId()})
	if err != nil || len(jobs.GetJobs()) != 0 {
		t.Fatalf("rejected stream queued jobs = %d (err %v), want none", len(jobs.GetJobs()), err)
	}
}

// TestUpsertCollectionItemsStreamAcceptsLargeStream sends 20 row frames of
// 3.5 MB of text each, 70 MB of row text in one stream, below the 512 MiB row
// byte limit. The stream queues a job, and the job completes. The rows belong
// to an item with an unchanged fingerprint. The job embeds nothing, and the
// collection keeps only the item's first row.
func TestUpsertCollectionItemsStreamAcceptsLargeStream(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)
	registered, err := daemon.registerCollection("docs-large", "docId", documentScalars())
	if err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}
	unspecified := pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_UNSPECIFIED
	manifest := map[string]string{"doc-a": "fp-a1"}
	daemon.upsertItems(collectionHeader("docs-large", unspecified, false, false), []*pb.CollectionRow{documentRow("a/0", "doc-a", "alpha", 1)}, manifest)

	const largeFrameCount = 20
	largeText := strings.Repeat("b", 3_500_000)
	frames := []*pb.UpsertCollectionItemsStreamRequest{{Chunk: &pb.UpsertCollectionItemsStreamRequest_Header{Header: collectionHeader("docs-large", unspecified, false, false)}}}
	for index := range largeFrameCount {
		row := documentRow("big/"+strconv.Itoa(index), "doc-a", largeText, 1)
		frames = append(frames, &pb.UpsertCollectionItemsStreamRequest{Chunk: &pb.UpsertCollectionItemsStreamRequest_Rows{Rows: &pb.UpsertCollectionItemsRows{Rows: []*pb.CollectionRow{row}}}})
	}
	frames = append(frames, &pb.UpsertCollectionItemsStreamRequest{Chunk: &pb.UpsertCollectionItemsStreamRequest_Manifest{Manifest: &pb.UpsertCollectionItemsManifest{Manifest: collectionFingerprints(manifest)}}})
	response, err := daemon.sendCollectionStream(frames)
	if err != nil {
		t.Fatalf("UpsertCollectionItemsStream with %d bytes of row text returned error: %v", largeFrameCount*len(largeText), err)
	}
	if job := waitForRPCJobTerminal(t, daemon.client, response.GetJobId()); job.GetState() != string(model.JobStateCompleted) {
		t.Fatalf("large stream job state = %q, want completed: %+v", job.GetState(), job.GetError())
	}
	if got := rowPaths(daemon.localRows(registered.GetCollectionName())); !slices.Equal(got, []string{"a/0"}) {
		t.Fatalf("rows after a large delivery with an unchanged fingerprint = %v, want [a/0]", got)
	}
}

// TestCollectionAndConversationStreamsStoreEqualRows submits the same
// transcript through the conversation stream and the generic stream into two
// collections with the conversation declaration. Both collections must store
// byte-identical rows and equal checkpoints after every step. Each subtest
// creates an independent daemon. A failure in one subtest does not affect the other.
func TestCollectionAndConversationStreamsStoreEqualRows(t *testing.T) {
	t.Parallel()
	t.Run("ingest", testConversationIngestParity)
	t.Run("backfill over a blank stored text row", testConversationBackfillParity)
}

// testConversationIngestParity ingests one transcript with two tool call rows
// longer than the split budget. The generic client sends the trimmed tool name
// as the continuation prefix of the named tool row. That row splits at the
// budget less the name length plus one, and each part after the first starts
// with the name line. The nameless tool row has no prefix. It splits at the
// full budget and stores its text unchanged. The generic manifest then needs
// nothing. A provider that disagrees with the item id is rejected, and so is a
// row key outside the conversation row key layout.
func testConversationIngestParity(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)
	conversation, err := daemon.registerConversationCollection("conv-parity-old")
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	generic, err := daemon.registerCollection("conv-parity-generic", "conversationId", conversationScalarsPB())
	if err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}
	conversationID := "claude:parity-1"
	longText := strings.Repeat("assistant answer sentence. ", 3000)
	budget := daemon.manager.conversationChunkByteBudget
	// Each long tool display spans about two split budgets. Each long tool row
	// then stores three parts. The nameless tool call has no name line, and the
	// first line of its row is a display line.
	longToolDisplay := strings.TrimSpace(strings.Repeat("write the parity fixture line. ", 2*budget/31+1))
	namelessToolText := "untitled tool notes\n" + strings.TrimSpace(strings.Repeat("nameless tool output line. ", 2*budget/27+1))
	documents := []*pb.ConversationDocument{
		{ConversationId: conversationID, MessageIndex: 0, Role: "user", TimestampUnix: 1712345678, Text: "how do generic rows match", WorkspaceRoot: "/work", LoadRules: "rules-v1"},
		{
			ConversationId: conversationID, MessageIndex: 1, Role: "assistant", TimestampUnix: 1712345679, Text: longText, WorkspaceRoot: "/work", LoadRules: "rules-v1",
			Tools: []*pb.ConversationToolCall{
				{Name: "Read", Display: "file.go", LangHint: "go"},
				{Name: "Write", Display: longToolDisplay, LangHint: "text"},
				{Name: "", Display: namelessToolText, LangHint: "text"},
			},
			Thinking: "private reasoning",
		},
	}
	manifest := map[string]string{conversationID: "fp-parity"}
	daemon.upsertConversation("conv-parity-old", documents, manifest, false)

	rows := []*pb.CollectionRow{
		{RowKey: "conv/" + conversationID + "/0", ItemId: conversationID, Text: "how do generic rows match", Scalars: conversationRowScalars(0, "user", 1712345678)},
		{RowKey: "conv/" + conversationID + "/1", ItemId: conversationID, Text: longText, Scalars: conversationRowScalars(1, "assistant", 1712345679)},
		{RowKey: "convtool/" + conversationID + "/1/0", ItemId: conversationID, Text: "Read\nfile.go", Scalars: conversationRowScalars(1, "assistant", 1712345679)},
		{RowKey: "convtool/" + conversationID + "/1/1", ItemId: conversationID, Text: "Write\n" + longToolDisplay, Scalars: conversationRowScalars(1, "assistant", 1712345679), ContinuationPrefix: "Write"},
		{RowKey: "convtool/" + conversationID + "/1/2", ItemId: conversationID, Text: namelessToolText, Scalars: conversationRowScalars(1, "assistant", 1712345679)},
		{RowKey: "convthink/" + conversationID + "/1", ItemId: conversationID, Text: "private reasoning", Scalars: conversationRowScalars(1, "assistant", 1712345679)},
	}
	daemon.upsertItems(collectionHeader("conv-parity-generic", pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_UNSPECIFIED, false, false), rows, manifest)

	conversationRows := daemon.requireEqualConversationRows("ingest", conversation, generic)
	namedTool := "convtool/" + conversationID + "/1/1"
	namelessTool := "convtool/" + conversationID + "/1/2"
	if paths := distinctRowPaths(conversationRows); len(paths) < 5 || !slices.Contains(paths, "conv/"+conversationID+"/1/1") || !slices.Contains(paths, "convthink/"+conversationID+"/1") || !slices.Contains(paths, "convtool/"+conversationID+"/1/0") || !slices.Contains(paths, namedTool+"/1") || !slices.Contains(paths, namelessTool+"/1") {
		t.Fatalf("The conversation stream stored rows %v. The expected rows include a split message, one tool row, two split tool rows, and one thinking row.", paths)
	}
	if got := len(storedPartContent(conversationRows, namedTool+"/0")); got != budget-len("Write")-1 {
		t.Fatalf("named tool row first part has %d bytes, want the budget less the name line, %d", got, budget-len("Write")-1)
	}
	if content := storedPartContent(conversationRows, namedTool+"/1"); !strings.HasPrefix(content, "Write\n") {
		t.Fatalf("named tool row part 1 starts with %.20q, want the tool name line", content)
	}
	if got := len(storedPartContent(conversationRows, namelessTool+"/0")); got != budget {
		t.Fatalf("nameless tool row first part has %d bytes, want the full budget, %d", got, budget)
	}
	if assembleParts(conversationRows, namelessTool+"/") != namelessToolText {
		t.Fatal("The nameless tool parts do not reassemble the original text.")
	}
	if needed := daemon.syncItems("conv-parity-generic", manifest); len(needed) != 0 {
		t.Fatalf("generic needed after ingest = %v, want none", needed)
	}

	badProvider := []*pb.CollectionRow{{RowKey: "conv/" + conversationID + "/2", ItemId: conversationID, Text: "x", Scalars: []*pb.CollectionScalarValue{stringScalar("provider", "codex")}}}
	_, err = daemon.sendCollectionStream(collectionFrames(collectionHeader("conv-parity-generic", pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_UNSPECIFIED, false, false), badProvider, manifest))
	requireColumnError(t, err, codes.InvalidArgument, "invalid_argument", "provider")

	for _, rowKey := range []string{"notes/1", "conv/" + conversationID + "/1", "conv/claude:other/0", "convtool/" + conversationID + "/0"} {
		misplaced := []*pb.CollectionRow{{RowKey: rowKey, ItemId: conversationID, Text: "x", Scalars: []*pb.CollectionScalarValue{int64Scalar("messageIndex", 0)}}}
		if _, err := daemon.sendCollectionStream(collectionFrames(collectionHeader("conv-parity-generic", pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_UNSPECIFIED, false, false), misplaced, manifest)); status.Code(err) != codes.InvalidArgument {
			t.Fatalf("row key %q in a conversation collection returned %v, want InvalidArgument", rowKey, err)
		}
	}
}

// testConversationBackfillParity ingests one transcript, then stores a blank
// text row for message 2 in both collections. A backfill then delivers text for
// message 2 under the unchanged fingerprint. A conversation backfill checks
// only tool call and thinking families, and every one of them is present.
// Both backfills retain the blank message row as the only row for message 2.
// The test compares stored row lines and checkpoints.
func testConversationBackfillParity(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)
	conversation, err := daemon.registerConversationCollection("conv-backfill-old")
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	generic, err := daemon.registerCollection("conv-backfill-generic", "conversationId", conversationScalarsPB())
	if err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}
	conversationID := "claude:backfill-1"
	documents := []*pb.ConversationDocument{
		{ConversationId: conversationID, MessageIndex: 0, Role: "user", TimestampUnix: 1712345678, Text: "which rows does a backfill check", WorkspaceRoot: "/work", LoadRules: "rules-v1"},
		{
			ConversationId: conversationID, MessageIndex: 1, Role: "assistant", TimestampUnix: 1712345679, Text: "tool call and thinking rows", WorkspaceRoot: "/work", LoadRules: "rules-v1",
			Tools: []*pb.ConversationToolCall{{Name: "Read", Display: "file.go", LangHint: "go"}}, Thinking: "private reasoning",
		},
	}
	rows := []*pb.CollectionRow{
		{RowKey: "conv/" + conversationID + "/0", ItemId: conversationID, Text: "which rows does a backfill check", Scalars: conversationRowScalars(0, "user", 1712345678)},
		{RowKey: "conv/" + conversationID + "/1", ItemId: conversationID, Text: "tool call and thinking rows", Scalars: conversationRowScalars(1, "assistant", 1712345679)},
		{RowKey: "convtool/" + conversationID + "/1/0", ItemId: conversationID, Text: "Read\nfile.go", Scalars: conversationRowScalars(1, "assistant", 1712345679)},
		{RowKey: "convthink/" + conversationID + "/1", ItemId: conversationID, Text: "private reasoning", Scalars: conversationRowScalars(1, "assistant", 1712345679)},
	}
	manifest := map[string]string{conversationID: "fp-backfill"}
	unspecified := pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_UNSPECIFIED
	daemon.upsertConversation("conv-backfill-old", documents, manifest, false)
	daemon.upsertItems(collectionHeader("conv-backfill-generic", unspecified, false, false), rows, manifest)
	daemon.requireEqualConversationRows("ingest", conversation, generic)

	daemon.storeBlankTextRow("conv-backfill-old", conversationID, 2)
	daemon.storeBlankTextRow("conv-backfill-generic", conversationID, 2)
	laterText := "message text an older pipeline stored blank"
	withText := append(slices.Clone(documents), &pb.ConversationDocument{ConversationId: conversationID, MessageIndex: 2, Role: "user", TimestampUnix: 1712345680, Text: laterText, WorkspaceRoot: "/work", LoadRules: "rules-v1"})
	withTextRows := append(slices.Clone(rows), &pb.CollectionRow{RowKey: "conv/" + conversationID + "/2", ItemId: conversationID, Text: laterText, Scalars: conversationRowScalars(2, "user", 1712345680)})
	daemon.upsertConversation("conv-backfill-old", withText, manifest, true)
	daemon.upsertItems(collectionHeader("conv-backfill-generic", unspecified, true, false), withTextRows, manifest)

	stored := daemon.requireEqualConversationRows("backfill", conversation, generic)
	messageRows := 0
	for _, row := range stored {
		if row.RelativePath != "conv/"+conversationID+"/2" {
			continue
		}
		messageRows++
		if strings.TrimSpace(row.Content) != "" {
			t.Fatalf("Backfill stored message 2 text %q, but the expected state contains only the blank row", row.Content)
		}
	}
	if messageRows != 1 {
		t.Fatalf("Message 2 has %d stored rows after the backfill, but the expected state contains one blank row", messageRows)
	}
}

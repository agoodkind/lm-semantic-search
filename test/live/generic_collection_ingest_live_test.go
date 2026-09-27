//go:build live

package live

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
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
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// parityLongTextBytes exceeds the 60000-byte conversation split budget. The
// long assistant message then stores several parts.
const parityLongTextBytes = 70_000

// parityLongToolBytes exceeds twice the conversation split budget. Each long
// tool call then stores at least three parts.
const parityLongToolBytes = 130_000

// parityScalarColumns are the conversation scalar columns a parity row compares.
var parityScalarColumns = []string{
	semantic.ConversationIDColumn,
	semantic.ConversationParentColumn,
	semantic.ConversationRoleColumn,
	semantic.ConversationProviderColumn,
	semantic.ConversationWorkspaceRootColumn,
	semantic.ConversationArchivedColumn,
	semantic.ConversationTimestampColumn,
	semantic.ConversationMessageIndexColumn,
	semantic.ConversationLoadRulesColumn,
	"splitPart",
}

// parityRow is one stored Milvus row reduced to comparable values. Content is
// stored as a hash, and a failure message never prints transcript content.
type parityRow struct {
	ID             string
	RelativePath   string
	ContentHash    string
	Metadata       string
	EmbeddingModel string
	VectorChecksum string
	Scalars        map[string]string
}

// TestGenericCollectionIngestParity submits the same synthetic transcript
// through the conversation RPCs and the generic item RPCs into two isolated
// collections with the conversation declaration in a real temporary Milvus
// database. After each step (first ingest, a backfill over a blank stored text
// row, append, backfill, force, and an authoritative removal) both collections
// store equal row keys, content, scalar values, vectors, and checkpoint
// fingerprints, and both manifest RPCs return the same needed set. The
// transcript includes a named and a nameless tool call longer than twice the
// split budget. The generic rows send the trimmed tool name as the continuation
// prefix, and an empty prefix for the nameless tool call. A backfill that
// delivers text for a message stored only as a blank row selects the
// conversation in neither collection, because a conversation backfill checks
// only tool call and thinking families. A provider that disagrees with the item
// id is rejected.
func TestGenericCollectionIngestParity(t *testing.T) {
	h := newHarness(t)
	genericCollectionID := "live-generic-" + randomID()
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
	if registration.GetCollectionName() == h.collectionName {
		t.Fatal("generic collection shares the conversation collection name")
	}

	first := parityConversationID("claude", "a")
	second := parityConversationID("codex", "b")
	convs := map[string][]*pb.ConversationDocument{
		first:  parityTranscript(first, ""),
		second: parityTranscript(second, first),
	}
	retain := pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN
	genericRetain := pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_RETAIN

	requireCompleted(t, h.upsert(convs, retain, false, false), "conversation ingest")
	requireCompleted(t, h.upsertGeneric(genericCollectionID, convs, parityManifest(convs), genericRetain, false, false), "generic ingest")
	h.requireParity(registration, "first ingest")
	if h.countRowsWithPrefix(convToolPrefix(first)+"1/1/1") == 0 {
		t.Fatal("the long tool call stored no second part")
	}
	if h.countRowsWithPrefix(convToolPrefix(first)+"1/2/1") == 0 {
		t.Fatal("the nameless long tool call stored no second part")
	}
	h.requireManifestParity(genericCollectionID, parityManifest(convs), nil)

	blankRowID := "blank-text-" + randomID()
	blankTextPath := convBasePrefix(second) + "2"
	h.insertBlankTextRow(h.collectionName, blankRowID, blankTextPath)
	h.insertBlankTextRow(registration.GetCollectionName(), blankRowID, blankTextPath)
	withLaterMessage := map[string][]*pb.ConversationDocument{second: parityWithLaterMessage(convs[second], second)}
	secondFingerprint := map[string]string{second: fingerprint(convs[second])}
	requireCompleted(t, h.upsertWithManifest(withLaterMessage, secondFingerprint, retain, true, false), "conversation backfill over a blank text row")
	requireCompleted(t, h.upsertGeneric(genericCollectionID, withLaterMessage, secondFingerprint, genericRetain, true, false), "generic backfill over a blank text row")
	h.requireParity(registration, "backfill over a blank text row")
	if count := h.countRowsWithPrefix(blankTextPath); count != 1 || strings.TrimSpace(h.contentForRelativePath(blankTextPath)) != "" {
		t.Fatalf("backfill over a blank text row left %d rows at %s, want only the blank row", count, blankTextPath)
	}

	changed := map[string][]*pb.ConversationDocument{first: appendMessage(convs[first], first), second: convs[second]}
	changedManifest := parityManifest(changed)
	unsent := parityConversationID("claude", "c")
	changedManifest[unsent] = "fingerprint-of-an-unsent-conversation"
	h.requireManifestParity(genericCollectionID, changedManifest, []string{first, unsent})

	appended := map[string][]*pb.ConversationDocument{first: changed[first]}
	requireCompleted(t, h.upsert(appended, retain, false, false), "conversation append")
	requireCompleted(t, h.upsertGeneric(genericCollectionID, appended, parityManifest(appended), genericRetain, false, false), "generic append")
	h.requireParity(registration, "append")

	backfilled := map[string][]*pb.ConversationDocument{first: parityWithExtraTool(changed[first], first)}
	unchangedFingerprint := map[string]string{first: fingerprint(changed[first])}
	requireCompleted(t, h.upsertWithManifest(backfilled, unchangedFingerprint, retain, true, false), "conversation backfill")
	requireCompleted(t, h.upsertGeneric(genericCollectionID, backfilled, unchangedFingerprint, genericRetain, true, false), "generic backfill")
	h.requireParity(registration, "backfill")
	if h.countRowsWithPrefix(convToolPrefix(first)+"1/3") == 0 {
		t.Fatal("backfill stored no row for the added tool call")
	}

	requireCompleted(t, h.upsertWithManifest(backfilled, unchangedFingerprint, retain, false, true), "conversation force")
	requireCompleted(t, h.upsertGeneric(genericCollectionID, backfilled, unchangedFingerprint, genericRetain, false, true), "generic force")
	h.requireParity(registration, "force")

	onlyFirst := map[string]string{first: fingerprint(changed[first])}
	noDelivery := map[string][]*pb.ConversationDocument{}
	requireCompleted(t, h.upsertWithManifest(noDelivery, onlyFirst, pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_AUTHORITATIVE, false, false), "conversation authoritative")
	requireCompleted(t, h.upsertGeneric(genericCollectionID, noDelivery, onlyFirst, pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_AUTHORITATIVE, false, false), "generic authoritative")
	h.requireParity(registration, "authoritative")
	if h.countRowsWithPrefix(convBasePrefix(second)) != 0 {
		t.Fatal("authoritative removal kept rows of the omitted conversation")
	}

	badProvider := parityRows(map[string][]*pb.ConversationDocument{first: convs[first][:1]})
	badProvider[0].Scalars = append(badProvider[0].Scalars[:2], badProvider[0].Scalars[3:]...)
	badProvider[0].Scalars = append(badProvider[0].Scalars, &pb.CollectionScalarValue{Column: semantic.ConversationProviderColumn, Value: &pb.CollectionScalarValue_StringValue{StringValue: "codex"}})
	_, err = h.sendGeneric(genericCollectionID, badProvider[:1], onlyFirst, genericRetain, false, false)
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("generic upsert with a mismatched provider returned %v, want InvalidArgument", err)
	}
}

// TestGenericCollectionTypedScalarsLive ingests rows of a generic declaration
// into a real temporary Milvus database. The created collection declares only
// the declared scalar columns, the rows store typed and null values with the
// item id column set from item_id, and the schema stays free of conversation
// columns across a restart and a second ingest. Backfill adds an absent row,
// force replaces an item's rows, and an authoritative manifest removes an
// omitted item by its item id column.
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
	h.requireNoConversationColumns(registration.GetCollectionName(), "first ingest")
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
	h.requireNoConversationColumns(registration.GetCollectionName(), "backfill after restart")
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

// typedRow builds one row of typedDeclarationPB. Its scalars are title,
// pinned, and rank in that order.
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

// requireNoConversationColumns fails when the stored collection declares any
// conversation scalar column.
func (h *harness) requireNoConversationColumns(collectionName string, step string) {
	h.t.Helper()
	collection, err := h.milvus.DescribeCollection(correlatedContext(), milvusclient.NewDescribeCollectionOption(collectionName))
	if err != nil {
		h.t.Fatalf("%s: describe %s: %v", step, collectionName, err)
	}
	for _, field := range collection.Schema.Fields {
		for _, conversationColumn := range semantic.ConversationDeclaration().Scalars {
			if field.Name == conversationColumn.Name {
				h.t.Fatalf("%s: generic collection declares conversation column %s", step, field.Name)
			}
		}
	}
}

// typedRows maps each stored row key of a generic collection to its rendered
// declared scalar values.
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

func parityConversationID(provider string, suffix string) string {
	return provider + ":live-parity-" + suffix
}

func parityDeclarationPB() []*pb.ScalarColumnDeclaration {
	declaration := semantic.ConversationDeclaration()
	scalars := make([]*pb.ScalarColumnDeclaration, 0, len(declaration.Scalars))
	for _, scalar := range declaration.Scalars {
		scalars = append(scalars, &pb.ScalarColumnDeclaration{Column: scalar.Name, Type: liveScalarType(scalar.Type), Nullable: scalar.Nullable, MaxLength: scalar.MaxLength})
	}
	return scalars
}

// parityTranscript is a two-message synthetic transcript. The assistant turn
// has text longer than the split budget, a short tool call, a named and a
// nameless tool call longer than twice the split budget, and thinking text.
// The first line of the nameless tool call is a display line.
func parityTranscript(conversationID string, parentID string) []*pb.ConversationDocument {
	return []*pb.ConversationDocument{
		{
			ConversationId: conversationID, ParentConversationId: parentID, MessageIndex: 0, Role: "user", TimestampUnix: 1712346000,
			Text: "summarize the design notes in " + conversationID, WorkspaceRoot: "/work/parity", LoadRules: "rules-v1",
		},
		{
			ConversationId: conversationID, ParentConversationId: parentID, MessageIndex: 1, Role: "assistant", TimestampUnix: 1712346001,
			Text:     strings.Repeat("The design note describes one ingestion step. ", parityLongTextBytes/47+1),
			Thinking: "reading the notes file for " + conversationID + " before the summary",
			Tools: []*pb.ConversationToolCall{
				{Name: "Read", Display: "/work/parity/notes.md", LangHint: "markdown"},
				{Name: "Write", Display: strings.TrimSpace(strings.Repeat("write the parity notes line. ", parityLongToolBytes/29+1)), LangHint: "markdown"},
				{Name: "", Display: "untitled parity notes\n" + strings.TrimSpace(strings.Repeat("nameless parity output line. ", parityLongToolBytes/29+1)), LangHint: "markdown"},
			},
			WorkspaceRoot: "/work/parity", LoadRules: "rules-v1",
		},
	}
}

// parityWithExtraTool returns a copy of documents with one more tool call on
// the assistant message at index 1.
func parityWithExtraTool(documents []*pb.ConversationDocument, conversationID string) []*pb.ConversationDocument {
	extended := make([]*pb.ConversationDocument, 0, len(documents))
	for _, document := range documents {
		if document.GetMessageIndex() != 1 {
			extended = append(extended, document)
			continue
		}
		copied := &pb.ConversationDocument{
			ConversationId:       document.GetConversationId(),
			ParentConversationId: document.GetParentConversationId(),
			MessageIndex:         document.GetMessageIndex(),
			Role:                 document.GetRole(),
			TimestampUnix:        document.GetTimestampUnix(),
			Text:                 document.GetText(),
			Thinking:             document.GetThinking(),
			WorkspaceRoot:        document.GetWorkspaceRoot(),
			LoadRules:            document.GetLoadRules(),
			Tools:                append(slices.Clone(document.GetTools()), &pb.ConversationToolCall{Name: "Grep", Display: "ingestion " + conversationID, LangHint: "text"}),
		}
		extended = append(extended, copied)
	}
	return extended
}

// parityWithLaterMessage returns a copy of documents with one more user
// message at index 2. The test stores a blank text row for that message before
// it delivers the message.
func parityWithLaterMessage(documents []*pb.ConversationDocument, conversationID string) []*pb.ConversationDocument {
	return append(slices.Clone(documents), &pb.ConversationDocument{
		ConversationId: conversationID, ParentConversationId: documents[0].GetParentConversationId(), MessageIndex: 2, Role: "user", TimestampUnix: 1712346002,
		Text: "a message an older pipeline stored blank in " + conversationID, WorkspaceRoot: "/work/parity", LoadRules: "rules-v1",
	})
}

// insertBlankTextRow writes one message text row with a single space as its
// content straight into a collection, with no conversationId value. An older
// pipeline wrote such a row for message text it did not keep, and current
// ingest never writes one.
func (h *harness) insertBlankTextRow(collectionName string, rowID string, relativePath string) {
	h.t.Helper()
	vector := make([]float32, fakeEmbeddingDimension)
	vector[0] = 1
	result, err := h.milvus.Insert(
		context.Background(),
		milvusclient.NewColumnBasedInsertOption(collectionName).
			WithVarcharColumn("id", []string{rowID}).
			WithVarcharColumn("content", []string{" "}).
			WithVarcharColumn(relativePathField, []string{relativePath}).
			WithInt64Column("startLine", []int64{0}).
			WithInt64Column("endLine", []int64{0}).
			WithVarcharColumn("fileExtension", []string{""}).
			WithVarcharColumn("metadata", []string{"{}"}).
			WithFloatVectorColumn("vector", len(vector), [][]float32{vector}),
	)
	if err != nil {
		h.t.Fatalf("insert blank text row into %s: %v", collectionName, err)
	}
	if result.InsertCount != 1 {
		h.t.Fatalf("insert blank text row into %s count = %d, want 1", collectionName, result.InsertCount)
	}
	flushTask, err := h.milvus.Flush(context.Background(), milvusclient.NewFlushOption(collectionName))
	if err != nil {
		h.t.Fatalf("flush blank text row in %s: %v", collectionName, err)
	}
	if err := flushTask.Await(context.Background()); err != nil {
		h.t.Fatalf("await blank text row flush in %s: %v", collectionName, err)
	}
}

func parityManifest(convs map[string][]*pb.ConversationDocument) map[string]string {
	manifest := make(map[string]string, len(convs))
	for conversationID, documents := range convs {
		manifest[conversationID] = fingerprint(documents)
	}
	return manifest
}

// parityRows derives the generic rows the conversation stream stores for the
// fixture: one text row per message, one row per tool call, and one thinking
// row per message with thinking. The fixture's tool calls are not shell
// commands. Each tool row is the tool name line, when the tool call has a name,
// and the display text. Each tool row sends the trimmed tool name as its
// continuation prefix, which is empty for a nameless tool call.
func parityRows(convs map[string][]*pb.ConversationDocument) []*pb.CollectionRow {
	rows := make([]*pb.CollectionRow, 0)
	for _, conversationID := range sortedKeys(convs) {
		for _, document := range convs[conversationID] {
			scalars := parityRowScalars(document)
			rows = append(rows, &pb.CollectionRow{RowKey: fmt.Sprintf("conv/%s/%d", conversationID, document.GetMessageIndex()), ItemId: conversationID, Text: document.GetText(), Scalars: scalars})
			for toolIndex, tool := range document.GetTools() {
				toolName := strings.TrimSpace(tool.GetName())
				toolText := tool.GetDisplay()
				if toolName != "" {
					toolText = toolName + "\n" + tool.GetDisplay()
				}
				rows = append(rows, &pb.CollectionRow{
					RowKey:             fmt.Sprintf("convtool/%s/%d/%d", conversationID, document.GetMessageIndex(), toolIndex),
					ItemId:             conversationID,
					Text:               toolText,
					Scalars:            scalars,
					ContinuationPrefix: toolName,
				})
			}
			if document.GetThinking() != "" {
				rows = append(rows, &pb.CollectionRow{RowKey: fmt.Sprintf("convthink/%s/%d", conversationID, document.GetMessageIndex()), ItemId: conversationID, Text: document.GetThinking(), Scalars: scalars})
			}
		}
	}
	return rows
}

func parityRowScalars(document *pb.ConversationDocument) []*pb.CollectionScalarValue {
	stringValue := func(columnName string, value string) *pb.CollectionScalarValue {
		return &pb.CollectionScalarValue{Column: columnName, Value: &pb.CollectionScalarValue_StringValue{StringValue: value}}
	}
	int64Value := func(columnName string, value int64) *pb.CollectionScalarValue {
		return &pb.CollectionScalarValue{Column: columnName, Value: &pb.CollectionScalarValue_Int64Value{Int64Value: value}}
	}
	return []*pb.CollectionScalarValue{
		stringValue(semantic.ConversationParentColumn, document.GetParentConversationId()),
		stringValue(semantic.ConversationRoleColumn, document.GetRole()),
		stringValue(semantic.ConversationProviderColumn, semantic.ProviderFromConversationID(document.GetConversationId())),
		stringValue(semantic.ConversationWorkspaceRootColumn, document.GetWorkspaceRoot()),
		{Column: semantic.ConversationArchivedColumn, Value: &pb.CollectionScalarValue_BoolValue{BoolValue: document.GetArchived()}},
		int64Value(semantic.ConversationTimestampColumn, document.GetTimestampUnix()),
		int64Value(semantic.ConversationMessageIndexColumn, int64(document.GetMessageIndex())),
		stringValue(semantic.ConversationLoadRulesColumn, document.GetLoadRules()),
	}
}

// upsertWithManifest drives the conversation stream with an explicit manifest,
// which may list conversations the delivery omits.
func (h *harness) upsertWithManifest(convs map[string][]*pb.ConversationDocument, manifest map[string]string, reconcile pb.ConversationReconcileMode, backfill bool, force bool) model.Job {
	h.t.Helper()
	stream, err := h.client.UpsertConversationDocumentsStream(correlatedContext())
	if err != nil {
		h.t.Fatalf("open UpsertConversationDocumentsStream returned error: %v", err)
	}
	documents := make([]*pb.ConversationDocument, 0)
	for _, conversationID := range sortedKeys(convs) {
		documents = append(documents, convs[conversationID]...)
	}
	fingerprints := make([]*pb.ConversationFingerprint, 0, len(manifest))
	for conversationID, value := range manifest {
		fingerprints = append(fingerprints, &pb.ConversationFingerprint{ConversationId: conversationID, Fingerprint: value})
	}
	for _, chunk := range []*pb.UpsertConversationDocumentsChunk{
		{Chunk: &pb.UpsertConversationDocumentsChunk_Header{Header: &pb.UpsertConversationDocumentsHeader{
			CollectionId: h.collectionID, Client: &pb.ClientInfo{Name: "live-harness"}, ReconcileMode: reconcile, BackfillDelivered: backfill, ForceReexamine: force,
		}}},
		{Chunk: &pb.UpsertConversationDocumentsChunk_Documents{Documents: &pb.UpsertConversationDocumentsDocuments{Documents: documents}}},
		{Chunk: &pb.UpsertConversationDocumentsChunk_Manifest{Manifest: &pb.UpsertConversationDocumentsManifest{Manifest: fingerprints}}},
	} {
		if err := stream.Send(chunk); err != nil {
			h.t.Fatalf("send conversation chunk returned error: %v", err)
		}
	}
	response, err := stream.CloseAndRecv()
	if err != nil {
		h.t.Fatalf("conversation CloseAndRecv returned error: %v", err)
	}
	return h.waitJob(response.GetJobId())
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

// upsertGeneric drives the generic item stream with the rows parityRows derives
// from convs and waits for the job.
func (h *harness) upsertGeneric(collectionID string, convs map[string][]*pb.ConversationDocument, manifest map[string]string, reconcile pb.CollectionReconcileMode, backfill bool, force bool) model.Job {
	h.t.Helper()
	response, err := h.sendGeneric(collectionID, parityRows(convs), manifest, reconcile, backfill, force)
	if err != nil {
		h.t.Fatalf("UpsertCollectionItemsStream returned error: %v", err)
	}
	return h.waitJob(response.GetJobId())
}

// requireParity compares the stored rows and checkpoints of the conversation
// collection and the generic collection.
func (h *harness) requireParity(generic *pb.RegisterCollectionResponse, step string) {
	h.t.Helper()
	conversationRows := h.parityRows(h.collectionName)
	genericRows := h.parityRows(generic.GetCollectionName())
	if len(conversationRows) == 0 {
		h.t.Fatalf("%s: conversation collection stored no rows", step)
	}
	if len(conversationRows) != len(genericRows) {
		h.t.Fatalf("%s: conversation collection stored %d rows, generic stored %d", step, len(conversationRows), len(genericRows))
	}
	for index := range conversationRows {
		if !reflect.DeepEqual(conversationRows[index], genericRows[index]) {
			h.t.Fatalf("%s: row %s differs: conversation %s, generic %s", step, conversationRows[index].RelativePath, describeParityRow(conversationRows[index]), describeParityRow(genericRows[index]))
		}
	}
	conversationCheckpoint := h.parityCheckpoint(h.codebaseID)
	genericCheckpoint := h.parityCheckpoint(generic.GetCodebaseId())
	if !reflect.DeepEqual(conversationCheckpoint, genericCheckpoint) {
		h.t.Fatalf("%s: checkpoint fingerprints differ: conversation %v, generic %v", step, conversationCheckpoint, genericCheckpoint)
	}
}

func describeParityRow(row parityRow) string {
	return fmt.Sprintf("id=%s content=%s vector=%s model=%s metadata_bytes=%d scalars=%v", row.ID, row.ContentHash, row.VectorChecksum, row.EmbeddingModel, len(row.Metadata), row.Scalars)
}

// requireManifestParity sends one manifest through both manifest RPCs and
// requires equal needed sets, sorted, equal to want.
func (h *harness) requireManifestParity(genericCollectionID string, manifest map[string]string, want []string) {
	h.t.Helper()
	conversationManifest := make([]*pb.ConversationFingerprint, 0, len(manifest))
	genericManifest := make([]*pb.CollectionItemFingerprint, 0, len(manifest))
	for itemID, value := range manifest {
		conversationManifest = append(conversationManifest, &pb.ConversationFingerprint{ConversationId: itemID, Fingerprint: value})
		genericManifest = append(genericManifest, &pb.CollectionItemFingerprint{ItemId: itemID, Fingerprint: value})
	}
	conversationResponse, err := h.client.SyncConversationManifest(correlatedContext(), &pb.SyncConversationManifestRequest{CollectionId: h.collectionID, Manifest: conversationManifest})
	if err != nil {
		h.t.Fatalf("SyncConversationManifest returned error: %v", err)
	}
	genericResponse, err := h.client.SyncCollectionManifest(correlatedContext(), &pb.SyncCollectionManifestRequest{CollectionId: genericCollectionID, Manifest: genericManifest})
	if err != nil {
		h.t.Fatalf("SyncCollectionManifest returned error: %v", err)
	}
	if !slices.Equal(conversationResponse.GetNeededConversationIds(), genericResponse.GetNeededItemIds()) {
		h.t.Fatalf("needed sets differ: conversation %v, generic %v", conversationResponse.GetNeededConversationIds(), genericResponse.GetNeededItemIds())
	}
	wantSorted := slices.Clone(want)
	slices.Sort(wantSorted)
	if !slices.Equal(genericResponse.GetNeededItemIds(), wantSorted) {
		h.t.Fatalf("needed set = %v, want %v", genericResponse.GetNeededItemIds(), wantSorted)
	}
}

func (h *harness) parityCheckpoint(codebaseID string) map[string]string {
	h.t.Helper()
	snapshot, err := merkle.ReadSnapshot(filepath.Join(h.config.MerkleDir, codebaseID+".json"))
	if err != nil {
		h.t.Fatalf("read checkpoint for %s: %v", codebaseID, err)
	}
	return snapshot.Files
}

// parityRows reads every row of a collection at strong consistency.
func (h *harness) parityRows(collectionName string) []parityRow {
	h.t.Helper()
	outputFields := append([]string{"id", relativePathField, "content", "metadata", "embeddingModel", "vector"}, parityScalarColumns...)
	result, err := h.milvus.Query(context.Background(), milvusclient.NewQueryOption(collectionName).
		WithFilter(`id != ""`).
		WithOutputFields(outputFields...).
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		h.t.Fatalf("query parity rows from %s: %v", collectionName, err)
	}
	rows := make([]parityRow, 0, result.ResultCount)
	for rowIndex := range result.ResultCount {
		rows = append(rows, h.parityRowAt(result, rowIndex))
	}
	slices.SortFunc(rows, func(left parityRow, right parityRow) int {
		return strings.Compare(left.RelativePath+"\x00"+left.ID, right.RelativePath+"\x00"+right.ID)
	})
	return rows
}

func (h *harness) parityRowAt(result milvusclient.ResultSet, rowIndex int) parityRow {
	h.t.Helper()
	stringAt := func(columnName string) string {
		return parityColumnValue(h.t, result.GetColumn(columnName), columnName, rowIndex)
	}
	vectorValue, err := result.GetColumn("vector").Get(rowIndex)
	if err != nil {
		h.t.Fatalf("read vector at %d: %v", rowIndex, err)
	}
	vector, isVector := vectorValue.(entity.FloatVector)
	if !isVector {
		h.t.Fatalf("vector at %d has type %T", rowIndex, vectorValue)
	}
	contentHash := sha256.Sum256([]byte(stringAt("content")))
	row := parityRow{
		ID:             stringAt("id"),
		RelativePath:   stringAt(relativePathField),
		ContentHash:    hex.EncodeToString(contentHash[:8]),
		Metadata:       stringAt("metadata"),
		EmbeddingModel: stringAt("embeddingModel"),
		VectorChecksum: checksumVector(vector),
		Scalars:        make(map[string]string, len(parityScalarColumns)),
	}
	for _, columnName := range parityScalarColumns {
		row.Scalars[columnName] = stringAt(columnName)
	}
	return row
}

// parityColumnValue renders one column value, or "null" for a null value.
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

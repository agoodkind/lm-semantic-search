package semantic

import (
	"context"
	"testing"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

type conversationBatchTestRow struct {
	conversationID  string
	relativePath    string
	role            string
	content         string
	messageIndex    int64
	hasMessageIndex bool
	vector          []float32
}

func conversationBatchResultSet(t *testing.T, rows []conversationBatchTestRow) milvusclient.ResultSet {
	t.Helper()
	vectorDimension := 0
	conversationIDs := make([]string, 0, len(rows))
	relativePaths := make([]string, 0, len(rows))
	roles := make([]string, 0, len(rows))
	contents := make([]string, 0, len(rows))
	vectors := make([][]float32, 0, len(rows))
	messageIndexes := make([]int64, 0, len(rows))
	validData := make([]bool, 0, len(rows))
	for _, row := range rows {
		if vectorDimension == 0 {
			vectorDimension = len(row.vector)
		}
		conversationIDs = append(conversationIDs, row.conversationID)
		relativePaths = append(relativePaths, row.relativePath)
		roles = append(roles, row.role)
		contents = append(contents, row.content)
		vectors = append(vectors, row.vector)
		messageIndexes = append(messageIndexes, row.messageIndex)
		validData = append(validData, row.hasMessageIndex)
	}
	messageIndexColumn, err := column.NewNullableColumnInt64(messageIndexFieldName, messageIndexes, validData, column.WithSparseNullableMode[int64](true))
	if err != nil {
		t.Fatalf("NewNullableColumnInt64 returned error: %v", err)
	}
	fields := milvusclient.DataSet{
		column.NewColumnVarChar(conversationIDFieldName, conversationIDs),
		column.NewColumnVarChar(relativePathFieldName, relativePaths),
		column.NewColumnVarChar(roleFieldName, roles),
		column.NewColumnVarChar(contentFieldName, contents),
		column.NewColumnFloatVector(denseVectorFieldName, vectorDimension, vectors),
		messageIndexColumn,
	}
	return milvusclient.ResultSet{ResultCount: len(rows), Fields: fields}
}

func TestAppendConversationBatchRowsBucketsBaseAndDerived(t *testing.T) {
	rows := []conversationBatchTestRow{
		{conversationID: "claude:a", relativePath: "conv/claude:a/0", role: "user", content: "hello", messageIndex: 0, hasMessageIndex: true, vector: []float32{1}},
		{conversationID: "claude:a", relativePath: "convthink/claude:a/0", role: "assistant", content: "reasoning", messageIndex: 0, hasMessageIndex: true, vector: []float32{2}},
		{conversationID: "claude:b", relativePath: "conv/claude:b/1", role: "assistant", content: "answer", messageIndex: 1, hasMessageIndex: true, vector: []float32{3}},
	}
	assemblies := newConversationBatchAssemblies()
	reuse := map[string][]float32{}
	if err := appendConversationBatchRows(conversationBatchResultSet(t, rows), assemblies, reuse); err != nil {
		t.Fatalf("appendConversationBatchRows returned error: %v", err)
	}
	batchRows := assemblies.finalize()

	first, found := batchRows["claude:a"]
	if !found {
		t.Fatalf("claude:a missing from batch rows: %#v", batchRows)
	}
	if first.Messages[0].Text != "hello" || first.Messages[0].Role != "user" {
		t.Fatalf("claude:a message 0 = %#v, want role user text hello", first.Messages[0])
	}
	derivedHash, derivedFound := first.DerivedPaths["convthink/claude:a/0"]
	if !derivedFound || derivedHash != contentVectorKey("reasoning") {
		t.Fatalf("claude:a derived path hash = %q, want %q", derivedHash, contentVectorKey("reasoning"))
	}

	second, found := batchRows["claude:b"]
	if !found {
		t.Fatalf("claude:b missing from batch rows: %#v", batchRows)
	}
	if second.Messages[1].Text != "answer" {
		t.Fatalf("claude:b message 1 text = %q, want answer", second.Messages[1].Text)
	}
	if len(second.DerivedPaths) != 0 {
		t.Fatalf("claude:b derived paths = %v, want none", second.DerivedPaths)
	}

	for _, content := range []string{"hello", "reasoning", "answer"} {
		if reuse[contentVectorKey(content)] == nil {
			t.Fatalf("reuse missing content %q", content)
		}
	}
}

// TestDerivedRowsRegisterTheirMessageWithItsRole covers a message whose only
// stored rows are derived, which is what a turn carrying just a tool call or
// just reasoning leaves behind once no blank text row is written for it.
//
// The examination path treats a delivered message absent from Messages as new,
// re-sends it, and reads its existing derived rows as orphans to remove. So a
// message the store does hold would be deleted and embedded again on every sync,
// forever. Registering it from its derived rows is what lets the comparison find
// it and settle.
//
// The role has to come with it. The comparison rejects a message whose stored
// role differs from the delivered one, so registering with an empty role would
// mismatch every time and churn exactly as if the message were missing. Every
// stored row carries the role, including a derived one.
func TestDerivedRowsRegisterTheirMessageWithItsRole(t *testing.T) {
	rows := []conversationBatchTestRow{
		{conversationID: "claude:a", relativePath: "conv/claude:a/0", role: "user", content: "ask", messageIndex: 0, hasMessageIndex: true, vector: []float32{1}},
		{conversationID: "claude:a", relativePath: "convtool/claude:a/1/0/tok", role: "assistant", content: "Bash", messageIndex: 1, hasMessageIndex: true, vector: []float32{2}},
		{conversationID: "claude:a", relativePath: "convtool/claude:a/1/0/cmd", role: "assistant", content: "ls -la", messageIndex: 1, hasMessageIndex: true, vector: []float32{3}},
		{conversationID: "claude:a", relativePath: "convthink/claude:a/2", role: "assistant", content: "considering", messageIndex: 2, hasMessageIndex: true, vector: []float32{4}},
	}
	assemblies := newConversationBatchAssemblies()
	if err := appendConversationBatchRows(conversationBatchResultSet(t, rows), assemblies, map[string][]float32{}); err != nil {
		t.Fatalf("appendConversationBatchRows returned error: %v", err)
	}
	stored := assemblies.finalize()["claude:a"]

	toolOnly, found := stored.Messages[1]
	if !found {
		t.Fatalf("tool-only message 1 is absent from Messages: %#v", stored.Messages)
	}
	if toolOnly.Role != "assistant" {
		t.Fatalf("tool-only message 1 role = %q, want assistant from its derived rows", toolOnly.Role)
	}
	if toolOnly.Text != "" {
		t.Fatalf("tool-only message 1 text = %q, want empty because it stored no text row", toolOnly.Text)
	}
	if !toolOnly.HasDerivedContent {
		t.Fatal("tool-only message 1 does not report derived content")
	}

	reasoningOnly, found := stored.Messages[2]
	if !found {
		t.Fatalf("reasoning-only message 2 is absent from Messages: %#v", stored.Messages)
	}
	if reasoningOnly.Role != "assistant" {
		t.Fatalf("reasoning-only message 2 role = %q, want assistant", reasoningOnly.Role)
	}

	if stored.Messages[0].Text != "ask" || stored.Messages[0].Role != "user" {
		t.Fatalf("message 0 = %#v, want the unchanged base row", stored.Messages[0])
	}
}

// TestDerivedRowRegistrationKeepsTheBaseRole proves a base row's role wins over
// a derived row's, whatever order the rows arrive in. Both carry the same role
// in practice, so a disagreement means one of them is wrong, and the base row is
// the one the comparison is against.
func TestDerivedRowRegistrationKeepsTheBaseRole(t *testing.T) {
	derivedFirst := []conversationBatchTestRow{
		{conversationID: "claude:a", relativePath: "convtool/claude:a/0/0/tok", role: "assistant", content: "Bash", messageIndex: 0, hasMessageIndex: true, vector: []float32{1}},
		{conversationID: "claude:a", relativePath: "conv/claude:a/0", role: "user", content: "text", messageIndex: 0, hasMessageIndex: true, vector: []float32{2}},
	}
	assemblies := newConversationBatchAssemblies()
	if err := appendConversationBatchRows(conversationBatchResultSet(t, derivedFirst), assemblies, map[string][]float32{}); err != nil {
		t.Fatalf("appendConversationBatchRows returned error: %v", err)
	}
	stored := assemblies.finalize()["claude:a"]

	if stored.Messages[0].Role != "user" {
		t.Fatalf("message 0 role = %q, want user from the base row even though the derived row arrived first", stored.Messages[0].Role)
	}
	if stored.Messages[0].Text != "text" {
		t.Fatalf("message 0 text = %q, want text", stored.Messages[0].Text)
	}
}

func TestLoadConversationDerivedBatchUnavailableReturnsEmpty(t *testing.T) {
	service := &Service{}
	state, err := service.LoadConversationDerivedBatch(context.Background(), "conv_chunks_test", []string{"claude:a"})
	if err != nil {
		t.Fatalf("LoadConversationDerivedBatch returned error: %v", err)
	}
	if len(state.Rows) != 0 || len(state.Reuse) != 0 {
		t.Fatalf("state = %#v, want empty rows and reuse when unavailable", state)
	}
}

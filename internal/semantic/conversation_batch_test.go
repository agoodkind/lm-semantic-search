package semantic

import (
	"context"
	"testing"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

type conversationBatchTestRow struct {
	conversationID    string
	hasConversationID bool
	relativePath      string
	role              string
	missingRole       bool
	content           string
	embeddingModel    string
	hasEmbeddingModel bool
	messageIndex      int64
	hasMessageIndex   bool
	vector            []float32
}

func conversationBatchResultSet(t *testing.T, rows []conversationBatchTestRow) milvusclient.ResultSet {
	t.Helper()
	vectorDimension := 0
	conversationIDs := make([]string, 0, len(rows))
	conversationIDValidData := make([]bool, 0, len(rows))
	relativePaths := make([]string, 0, len(rows))
	roles := make([]string, 0, len(rows))
	roleValidData := make([]bool, 0, len(rows))
	contents := make([]string, 0, len(rows))
	embeddingModels := make([]string, 0, len(rows))
	embeddingModelValidData := make([]bool, 0, len(rows))
	vectors := make([][]float32, 0, len(rows))
	messageIndexes := make([]int64, 0, len(rows))
	validData := make([]bool, 0, len(rows))
	for _, row := range rows {
		if vectorDimension == 0 {
			vectorDimension = len(row.vector)
		}
		conversationIDs = append(conversationIDs, row.conversationID)
		conversationIDValidData = append(
			conversationIDValidData,
			row.hasConversationID || row.conversationID != "",
		)
		relativePaths = append(relativePaths, row.relativePath)
		roles = append(roles, row.role)
		roleValidData = append(roleValidData, !row.missingRole)
		contents = append(contents, row.content)
		embeddingModels = append(embeddingModels, row.embeddingModel)
		embeddingModelValidData = append(
			embeddingModelValidData,
			row.hasEmbeddingModel || row.embeddingModel != "",
		)
		vectors = append(vectors, row.vector)
		messageIndexes = append(messageIndexes, row.messageIndex)
		validData = append(validData, row.hasMessageIndex)
	}
	messageIndexColumn, err := column.NewNullableColumnInt64(messageIndexFieldName, messageIndexes, validData, column.WithSparseNullableMode[int64](true))
	if err != nil {
		t.Fatalf("NewNullableColumnInt64 returned error: %v", err)
	}
	conversationIDColumn, err := column.NewNullableColumnVarChar(
		conversationIDFieldName,
		conversationIDs,
		conversationIDValidData,
		column.WithSparseNullableMode[string](true),
	)
	if err != nil {
		t.Fatalf("NewNullableColumnVarChar returned error: %v", err)
	}
	roleColumn, err := column.NewNullableColumnVarChar(
		roleFieldName,
		roles,
		roleValidData,
		column.WithSparseNullableMode[string](true),
	)
	if err != nil {
		t.Fatalf("NewNullableColumnVarChar role returned error: %v", err)
	}
	embeddingModelColumn, err := column.NewNullableColumnVarChar(
		embeddingModelFieldName,
		embeddingModels,
		embeddingModelValidData,
		column.WithSparseNullableMode[string](true),
	)
	if err != nil {
		t.Fatalf("NewNullableColumnVarChar embedding model returned error: %v", err)
	}
	fields := milvusclient.DataSet{
		conversationIDColumn,
		column.NewColumnVarChar(relativePathFieldName, relativePaths),
		roleColumn,
		column.NewColumnVarChar(contentFieldName, contents),
		embeddingModelColumn,
		column.NewColumnFloatVector(denseVectorFieldName, vectorDimension, vectors),
		messageIndexColumn,
	}
	return milvusclient.ResultSet{ResultCount: len(rows), Fields: fields}
}

func TestAppendConversationBatchRowsRejectsOnlyKnownUnequalEmbeddingModels(t *testing.T) {
	currentModel := "model-b"
	rows := []conversationBatchTestRow{
		{conversationID: "claude:a", relativePath: "conv/claude:a/0", content: "known unequal", embeddingModel: "model-a", messageIndex: 0, hasMessageIndex: true, vector: []float32{1}},
		{conversationID: "claude:a", relativePath: "conv/claude:a/1", content: "legacy unknown", messageIndex: 1, hasMessageIndex: true, vector: []float32{2}},
		{conversationID: "claude:a", relativePath: "conv/claude:a/2", content: "known equal", embeddingModel: currentModel, messageIndex: 2, hasMessageIndex: true, vector: []float32{3}},
	}
	assemblies := newConversationBatchAssemblies()
	reuse := map[string][]float32{}
	if err := appendConversationBatchRows(
		conversationBatchResultSet(t, rows),
		[]string{"claude:a"},
		currentModel,
		assemblies,
		reuse,
	); err != nil {
		t.Fatalf("appendConversationBatchRows returned error: %v", err)
	}

	if vector := reuse[contentVectorKey("known unequal")]; vector != nil {
		t.Fatalf("known unequal model vector = %v under current model %q, want absent", vector, currentModel)
	}
	if vector := reuse[contentVectorKey("legacy unknown")]; vector == nil {
		t.Fatal("legacy row with no embedding model was not reusable")
	}
	if vector := reuse[contentVectorKey("known equal")]; vector == nil {
		t.Fatal("row with the current embedding model was not reusable")
	}
	if messages := assemblies.finalize()["claude:a"].Messages; len(messages) != len(rows) {
		t.Fatalf("stored messages = %d, want all %d rows retained", len(messages), len(rows))
	}
}

func TestAppendConversationBatchRowsBucketsBaseAndDerived(t *testing.T) {
	rows := []conversationBatchTestRow{
		{conversationID: "claude:a", relativePath: "conv/claude:a/0", role: "user", content: "hello", messageIndex: 0, hasMessageIndex: true, vector: []float32{1}},
		{conversationID: "claude:a", relativePath: "convthink/claude:a/0", role: "assistant", content: "reasoning", messageIndex: 0, hasMessageIndex: true, vector: []float32{2}},
		{conversationID: "claude:b", relativePath: "conv/claude:b/1", role: "assistant", content: "answer", messageIndex: 1, hasMessageIndex: true, vector: []float32{3}},
	}
	assemblies := newConversationBatchAssemblies()
	reuse := map[string][]float32{}
	if err := appendConversationBatchRows(conversationBatchResultSet(t, rows), []string{"claude:a", "claude:b"}, "", assemblies, reuse); err != nil {
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
// Registering derived-only messages preserves the stored message index and role
// even when no base row exists.
func TestDerivedRowsRegisterTheirMessageWithItsRole(t *testing.T) {
	rows := []conversationBatchTestRow{
		{conversationID: "claude:a", relativePath: "conv/claude:a/0", role: "user", content: "ask", messageIndex: 0, hasMessageIndex: true, vector: []float32{1}},
		{conversationID: "claude:a", relativePath: "convtool/claude:a/1/0/tok", role: "assistant", content: "Bash", messageIndex: 1, hasMessageIndex: true, vector: []float32{2}},
		{conversationID: "claude:a", relativePath: "convtool/claude:a/1/0/cmd", role: "assistant", content: "ls -la", messageIndex: 1, hasMessageIndex: true, vector: []float32{3}},
		{conversationID: "claude:a", relativePath: "convthink/claude:a/2", role: "assistant", content: "considering", messageIndex: 2, hasMessageIndex: true, vector: []float32{4}},
	}
	assemblies := newConversationBatchAssemblies()
	if err := appendConversationBatchRows(conversationBatchResultSet(t, rows), []string{"claude:a"}, "", assemblies, map[string][]float32{}); err != nil {
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
// in practice, and the base row owns the assembled base state.
func TestDerivedRowRegistrationKeepsTheBaseRole(t *testing.T) {
	derivedFirst := []conversationBatchTestRow{
		{conversationID: "claude:a", relativePath: "convtool/claude:a/0/0/tok", role: "assistant", content: "Bash", messageIndex: 0, hasMessageIndex: true, vector: []float32{1}},
		{conversationID: "claude:a", relativePath: "conv/claude:a/0", role: "user", content: "text", messageIndex: 0, hasMessageIndex: true, vector: []float32{2}},
	}
	assemblies := newConversationBatchAssemblies()
	if err := appendConversationBatchRows(conversationBatchResultSet(t, derivedFirst), []string{"claude:a"}, "", assemblies, map[string][]float32{}); err != nil {
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

func TestAppendConversationBatchRowsMarksOnlyUsableDerivedPaths(t *testing.T) {
	rows := []conversationBatchTestRow{
		{
			conversationID:  "claude:a",
			relativePath:    "convtool/claude:a/3/0/tok",
			role:            "assistant",
			content:         " \n",
			messageIndex:    3,
			hasMessageIndex: true,
			vector:          []float32{1},
		},
		{
			conversationID:  "claude:a",
			relativePath:    "convthink/claude:a/3",
			role:            "assistant",
			content:         "usable reasoning",
			messageIndex:    3,
			hasMessageIndex: true,
			vector:          []float32{2},
		},
	}
	assemblies := newConversationBatchAssemblies()
	if err := appendConversationBatchRows(
		conversationBatchResultSet(t, rows),
		[]string{"claude:a"},
		"",
		assemblies,
		map[string][]float32{},
	); err != nil {
		t.Fatalf("appendConversationBatchRows returned error: %v", err)
	}
	stored := assemblies.finalize()["claude:a"]

	if len(stored.DerivedPaths) != 2 {
		t.Fatalf("derived paths = %v, want both stored identities", stored.DerivedPaths)
	}
	if _, found := stored.UsableDerivedPaths["convtool/claude:a/3/0/tok"]; found {
		t.Fatalf("usable derived paths = %v, blank tool row must not satisfy a family", stored.UsableDerivedPaths)
	}
	if _, found := stored.UsableDerivedPaths["convthink/claude:a/3"]; !found {
		t.Fatalf("usable derived paths = %v, want usable thinking row", stored.UsableDerivedPaths)
	}
}

func TestAppendConversationBatchRowsResolvesScalarLessLegacyPaths(t *testing.T) {
	rows := []conversationBatchTestRow{
		{
			relativePath:    "conv/claude:legacy/4/0",
			missingRole:     true,
			content:         "stored answer",
			hasMessageIndex: false,
			vector:          []float32{1},
		},
		{
			relativePath:    "convtool/claude:legacy/4/0/tok",
			missingRole:     true,
			content:         "Bash",
			hasMessageIndex: false,
			vector:          []float32{2},
		},
	}
	assemblies := newConversationBatchAssemblies()
	if err := appendConversationBatchRows(
		conversationBatchResultSet(t, rows),
		[]string{"claude:legacy"},
		"",
		assemblies,
		map[string][]float32{},
	); err != nil {
		t.Fatalf("appendConversationBatchRows returned error: %v", err)
	}
	stored := assemblies.finalize()["claude:legacy"]

	if got := stored.Messages[4].Text; got != "stored answer" {
		t.Fatalf("legacy message text = %q, want stored answer", got)
	}
	if _, found := stored.UsableDerivedPaths["convtool/claude:legacy/4/0/tok"]; !found {
		t.Fatalf("usable derived paths = %v, want scalar-less tool row", stored.UsableDerivedPaths)
	}
}

func TestConversationBatchFilterExpressionIncludesLegacyFamilyPrefixes(t *testing.T) {
	got := conversationBatchFilterExpression([]string{"claude:a"})
	want := `(conversationId in ["claude:a"] or relativePath like "conv/claude:a/%" or relativePath like "convtool/claude:a/%" or relativePath like "convthink/claude:a/%")`
	if got != want {
		t.Fatalf("conversationBatchFilterExpression = %q, want %q", got, want)
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

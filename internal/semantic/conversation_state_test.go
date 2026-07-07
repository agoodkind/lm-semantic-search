package semantic

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/model"
)

type conversationStateTestIterator struct {
	pages []milvusclient.ResultSet
	index int
}

func (iterator *conversationStateTestIterator) Next(context.Context) (milvusclient.ResultSet, error) {
	if iterator.index >= len(iterator.pages) {
		return milvusclient.ResultSet{}, io.EOF
	}
	page := iterator.pages[iterator.index]
	iterator.index++
	return page, nil
}

type conversationStateTestRow struct {
	relativePath    string
	role            string
	content         string
	messageIndex    int64
	hasMessageIndex bool
	vector          []float32
}

func TestLoadConversationMessageStateReturnsEmptyForEmptyPrefix(t *testing.T) {
	service := &Service{}
	service.available.Store(true)

	state, reuse, err := service.LoadConversationMessageState(context.Background(), "conv_chunks_test", "")
	if err != nil {
		t.Fatalf("LoadConversationMessageState returned error: %v", err)
	}
	if len(state) != 0 {
		t.Fatalf("state len = %d, want 0", len(state))
	}
	if len(reuse) != 0 {
		t.Fatalf("reuse len = %d, want 0", len(reuse))
	}
}

func TestLoadConversationMessageStateFromIteratorAssemblesSinglePart(t *testing.T) {
	rows := []conversationStateTestRow{
		{
			relativePath:    "conv/claude/thread-1/0",
			role:            "user",
			content:         "hello",
			messageIndex:    0,
			hasMessageIndex: true,
			vector:          []float32{1, 2},
		},
	}
	iterator := &conversationStateTestIterator{
		pages: []milvusclient.ResultSet{conversationStateResultSet(t, rows, true)},
	}

	state, reuse, err := loadConversationMessageStateFromIterator(context.Background(), "conv_chunks_test", "conv/claude/thread-1/", iterator)
	if err != nil {
		t.Fatalf("loadConversationMessageStateFromIterator returned error: %v", err)
	}

	wantState := map[int32]StoredMessageState{
		0: {Role: "user", Text: "hello"},
	}
	assertStoredMessageState(t, state, wantState)
	assertReuseVector(t, reuse, "hello", []float32{1, 2})
}

func TestLoadConversationMessageStateFromIteratorAcceptsNewlineConversationPrefix(t *testing.T) {
	prefix := "conv/cursor:task-call_0mtc\nfc_00729/"
	rows := []conversationStateTestRow{
		{
			relativePath:    prefix + "7/0",
			role:            "assistant",
			content:         "newline prefix content",
			messageIndex:    7,
			hasMessageIndex: true,
			vector:          []float32{7},
		},
	}
	iterator := &conversationStateTestIterator{
		pages: []milvusclient.ResultSet{conversationStateResultSet(t, rows, true)},
	}

	state, reuse, err := loadConversationMessageStateFromIterator(context.Background(), "conv_chunks_test", prefix, iterator)
	if err != nil {
		t.Fatalf("loadConversationMessageStateFromIterator returned error: %v", err)
	}

	wantState := map[int32]StoredMessageState{
		7: {Role: "assistant", Text: "newline prefix content"},
	}
	assertStoredMessageState(t, state, wantState)
	assertReuseVector(t, reuse, "newline prefix content", []float32{7})
}

func TestLoadConversationMessageStateFromIteratorAssemblesMultipartInPartOrder(t *testing.T) {
	prefix := "conv/codex/provider/thread/with/slash/"
	firstPage := conversationStateResultSet(t, []conversationStateTestRow{
		{
			relativePath:    prefix + "3/2",
			role:            "assistant",
			content:         "C",
			messageIndex:    3,
			hasMessageIndex: true,
			vector:          []float32{3},
		},
		{
			relativePath:    prefix + "3/0",
			role:            "assistant",
			content:         "A",
			messageIndex:    3,
			hasMessageIndex: true,
			vector:          []float32{1},
		},
	}, true)
	secondPage := conversationStateResultSet(t, []conversationStateTestRow{
		{
			relativePath:    prefix + "3/1",
			role:            "assistant",
			content:         "B",
			messageIndex:    3,
			hasMessageIndex: true,
			vector:          []float32{2},
		},
	}, true)
	iterator := &conversationStateTestIterator{pages: []milvusclient.ResultSet{firstPage, secondPage}}

	state, _, err := loadConversationMessageStateFromIterator(context.Background(), "conv_chunks_test", prefix, iterator)
	if err != nil {
		t.Fatalf("loadConversationMessageStateFromIterator returned error: %v", err)
	}

	wantState := map[int32]StoredMessageState{
		3: {Role: "assistant", Text: "ABC"},
	}
	assertStoredMessageState(t, state, wantState)
}

func TestLoadConversationMessageStateFromIteratorReturnsEmptyMapsForEmptyResult(t *testing.T) {
	iterator := &conversationStateTestIterator{}

	state, reuse, err := loadConversationMessageStateFromIterator(context.Background(), "conv_chunks_test", "conv/empty/", iterator)
	if err != nil {
		t.Fatalf("loadConversationMessageStateFromIterator returned error: %v", err)
	}
	if len(state) != 0 {
		t.Fatalf("state len = %d, want 0", len(state))
	}
	if len(reuse) != 0 {
		t.Fatalf("reuse len = %d, want 0", len(reuse))
	}
}

func TestLoadConversationMessageStateFromIteratorSkipsLegacyRowsAndKeepsReuse(t *testing.T) {
	t.Run("null messageIndex", func(t *testing.T) {
		logs := captureConversationStateLogs(t)
		rows := []conversationStateTestRow{
			{
				relativePath:    "conv/legacy/1",
				role:            "user",
				content:         "indexed",
				messageIndex:    1,
				hasMessageIndex: true,
				vector:          []float32{1},
			},
			{
				relativePath: "conv/legacy/2",
				role:         "assistant",
				content:      "legacy-null",
				vector:       []float32{2},
			},
		}
		iterator := &conversationStateTestIterator{
			pages: []milvusclient.ResultSet{conversationStateResultSet(t, rows, true)},
		}

		state, reuse, err := loadConversationMessageStateFromIterator(context.Background(), "conv_chunks_test", "conv/legacy/", iterator)
		if err != nil {
			t.Fatalf("loadConversationMessageStateFromIterator returned error: %v", err)
		}

		wantState := map[int32]StoredMessageState{
			1: {Role: "user", Text: "indexed"},
		}
		assertStoredMessageState(t, state, wantState)
		assertReuseVector(t, reuse, "indexed", []float32{1})
		assertReuseVector(t, reuse, "legacy-null", []float32{2})
		assertLegacyWarningCount(t, logs.String(), 1)
	})

	t.Run("missing messageIndex column", func(t *testing.T) {
		logs := captureConversationStateLogs(t)
		rows := []conversationStateTestRow{
			{
				relativePath: "conv/legacy/1",
				role:         "user",
				content:      "legacy-missing-a",
				vector:       []float32{1},
			},
			{
				relativePath: "conv/legacy/2",
				role:         "assistant",
				content:      "legacy-missing-b",
				vector:       []float32{2},
			},
		}
		iterator := &conversationStateTestIterator{
			pages: []milvusclient.ResultSet{conversationStateResultSet(t, rows, false)},
		}

		state, reuse, err := loadConversationMessageStateFromIterator(context.Background(), "conv_chunks_test", "conv/legacy/", iterator)
		if err != nil {
			t.Fatalf("loadConversationMessageStateFromIterator returned error: %v", err)
		}

		if len(state) != 0 {
			t.Fatalf("state len = %d, want 0", len(state))
		}
		assertReuseVector(t, reuse, "legacy-missing-a", []float32{1})
		assertReuseVector(t, reuse, "legacy-missing-b", []float32{2})
		assertLegacyWarningCount(t, logs.String(), 2)
	})
}

func TestLoadConversationMessageStateFromIteratorReuseMapMatchesContentKeysPerRow(t *testing.T) {
	rows := []conversationStateTestRow{
		{
			relativePath:    "conv/reuse/0",
			role:            "user",
			content:         "same message part zero",
			messageIndex:    0,
			hasMessageIndex: true,
			vector:          []float32{10},
		},
		{
			relativePath:    "conv/reuse/0/1",
			role:            "user",
			content:         "same message part one",
			messageIndex:    0,
			hasMessageIndex: true,
			vector:          []float32{11},
		},
	}
	iterator := &conversationStateTestIterator{
		pages: []milvusclient.ResultSet{conversationStateResultSet(t, rows, true)},
	}

	_, reuse, err := loadConversationMessageStateFromIterator(context.Background(), "conv_chunks_test", "conv/reuse/", iterator)
	if err != nil {
		t.Fatalf("loadConversationMessageStateFromIterator returned error: %v", err)
	}

	wantReuse := map[string][]float32{
		contentVectorKey("same message part zero"): {10},
		contentVectorKey("same message part one"):  {11},
	}
	assertReuseMap(t, reuse, wantReuse)
}

func TestConversationStateFilterExpressionIncludesDerivedPrefixes(t *testing.T) {
	got := conversationStateFilterExpression("conv/reuse/")
	want := `(conversationId in ["reuse"] or relativePath like "conv/reuse/%" or relativePath like "convtool/reuse/%" or relativePath like "convthink/reuse/%")`

	if got != want {
		t.Fatalf("conversation state filter = %q, want %q", got, want)
	}
}

func TestLoadConversationMessageStateFromIteratorSkipsDerivedConversationRows(t *testing.T) {
	rows := []conversationStateTestRow{
		{
			relativePath:    "conv/derived/5",
			role:            "assistant",
			content:         "visible text",
			messageIndex:    5,
			hasMessageIndex: true,
			vector:          []float32{1},
		},
		{
			relativePath:    "convtool/derived/5/0/tok",
			role:            "assistant",
			content:         "Bash cat /tmp/input.txt",
			messageIndex:    5,
			hasMessageIndex: true,
			vector:          []float32{2},
		},
		{
			relativePath:    "convthink/derived/5",
			role:            "assistant",
			content:         "private reasoning",
			messageIndex:    5,
			hasMessageIndex: true,
			vector:          []float32{3},
		},
	}
	iterator := &conversationStateTestIterator{
		pages: []milvusclient.ResultSet{conversationStateResultSet(t, rows, true)},
	}

	state, reuse, err := loadConversationMessageStateFromIterator(context.Background(), "conv_chunks_test", "conv/derived/", iterator)
	if err != nil {
		t.Fatalf("loadConversationMessageStateFromIterator returned error: %v", err)
	}

	wantState := map[int32]StoredMessageState{
		5: {Role: "assistant", Text: "visible text", HasDerivedContent: true},
	}
	assertStoredMessageState(t, state, wantState)
	assertReuseMap(t, reuse, map[string][]float32{
		contentVectorKey("visible text"):            {1},
		contentVectorKey("Bash cat /tmp/input.txt"): {2},
		contentVectorKey("private reasoning"):       {3},
	})
}

func TestLoadConversationMessageStateDerivedReuseSkipsEmbedder(t *testing.T) {
	rows := []conversationStateTestRow{
		{
			relativePath:    "conv/reuse-derived/5",
			role:            "assistant",
			content:         "visible text",
			messageIndex:    5,
			hasMessageIndex: true,
			vector:          []float32{1},
		},
		{
			relativePath:    "convtool/reuse-derived/5/0/tok",
			role:            "assistant",
			content:         "Bash cat /tmp/input.txt",
			messageIndex:    5,
			hasMessageIndex: true,
			vector:          []float32{2},
		},
		{
			relativePath:    "convthink/reuse-derived/5",
			role:            "assistant",
			content:         "private reasoning",
			messageIndex:    5,
			hasMessageIndex: true,
			vector:          []float32{3},
		},
	}
	iterator := &conversationStateTestIterator{
		pages: []milvusclient.ResultSet{conversationStateResultSet(t, rows, true)},
	}

	_, reuse, err := loadConversationMessageStateFromIterator(context.Background(), "conv_chunks_test", "conv/reuse-derived/", iterator)
	if err != nil {
		t.Fatalf("loadConversationMessageStateFromIterator returned error: %v", err)
	}

	embedder := &countingEmbedder{}
	service := &Service{embedder: embedder}
	chunks := []model.StoredChunk{
		{Content: "Bash cat /tmp/input.txt"},
		{Content: "private reasoning"},
		{Content: "new appended message"},
	}
	_, reused, err := service.embedChunkBatch(context.Background(), chunks, reuse)
	if err != nil {
		t.Fatalf("embedChunkBatch returned error: %v", err)
	}
	if reused != 2 {
		t.Fatalf("reused = %d, want 2 derived chunks reused", reused)
	}
	if len(embedder.batches) != 1 {
		t.Fatalf("embedder called %d time(s), want one call for the new chunk", len(embedder.batches))
	}
	if want := []string{"new appended message"}; !slices.Equal(embedder.batches[0], want) {
		t.Fatalf("embedded batch = %v, want %v", embedder.batches[0], want)
	}
}

func TestLoadConversationMessageStateRejectsNegativePathIndexes(t *testing.T) {
	tests := []struct {
		name         string
		relativePath string
	}{
		{
			name:         "message index",
			relativePath: "conv/reject/-1",
		},
		{
			name:         "part index",
			relativePath: "conv/reject/1/-2",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := conversationMessagePartIndex(test.relativePath, "conv/reject/")
			if err == nil {
				t.Fatalf("conversationMessagePartIndex(%q) returned nil error, want error", test.relativePath)
			}
		})
	}
}

func conversationStateResultSet(t *testing.T, rows []conversationStateTestRow, includeMessageIndex bool) milvusclient.ResultSet {
	t.Helper()
	vectorDimension := conversationStateVectorDimension(t, rows)
	relativePaths := make([]string, 0, len(rows))
	roles := make([]string, 0, len(rows))
	contents := make([]string, 0, len(rows))
	vectors := make([][]float32, 0, len(rows))
	for _, row := range rows {
		relativePaths = append(relativePaths, row.relativePath)
		roles = append(roles, row.role)
		contents = append(contents, row.content)
		vectors = append(vectors, row.vector)
	}

	fields := milvusclient.DataSet{
		column.NewColumnVarChar(relativePathFieldName, relativePaths),
		column.NewColumnVarChar(roleFieldName, roles),
		column.NewColumnVarChar(contentFieldName, contents),
		column.NewColumnFloatVector(denseVectorFieldName, vectorDimension, vectors),
	}
	if includeMessageIndex {
		values := make([]int64, 0, len(rows))
		validData := make([]bool, 0, len(rows))
		for _, row := range rows {
			validData = append(validData, row.hasMessageIndex)
			values = append(values, row.messageIndex)
		}
		messageIndexes, err := column.NewNullableColumnInt64(messageIndexFieldName, values, validData, column.WithSparseNullableMode[int64](true))
		if err != nil {
			t.Fatalf("NewNullableColumnInt64 returned error: %v", err)
		}
		fields = append(fields, messageIndexes)
	}
	return milvusclient.ResultSet{ResultCount: len(rows), Fields: fields}
}

func conversationStateVectorDimension(t *testing.T, rows []conversationStateTestRow) int {
	t.Helper()
	vectorDimension := 0
	for rowIndex, row := range rows {
		rowDimension := len(row.vector)
		if rowDimension == 0 {
			t.Fatalf("row %d vector dimension = 0, want positive dimension", rowIndex)
		}
		if vectorDimension == 0 {
			vectorDimension = rowDimension
			continue
		}
		if rowDimension != vectorDimension {
			t.Fatalf("row %d vector dimension = %d, want %d", rowIndex, rowDimension, vectorDimension)
		}
	}
	return vectorDimension
}

func captureConversationStateLogs(t *testing.T) *bytes.Buffer {
	t.Helper()
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() {
		slog.SetDefault(previous)
	})
	return &logs
}

func assertStoredMessageState(t *testing.T, got map[int32]StoredMessageState, want map[int32]StoredMessageState) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("state len = %d, want %d; got %#v", len(got), len(want), got)
	}
	for messageIndex, wantState := range want {
		gotState, ok := got[messageIndex]
		if !ok {
			t.Fatalf("state missing message %d; got %#v", messageIndex, got)
		}
		if gotState != wantState {
			t.Fatalf("state[%d] = %#v, want %#v", messageIndex, gotState, wantState)
		}
	}
}

func assertReuseMap(t *testing.T, got map[string][]float32, want map[string][]float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("reuse len = %d, want %d; got %#v", len(got), len(want), got)
	}
	for key, wantVector := range want {
		gotVector, ok := got[key]
		if !ok {
			t.Fatalf("reuse missing key %s; got %#v", key, got)
		}
		if !slices.Equal(gotVector, wantVector) {
			t.Fatalf("reuse[%s] = %v, want %v", key, gotVector, wantVector)
		}
	}
}

func assertReuseVector(t *testing.T, got map[string][]float32, content string, want []float32) {
	t.Helper()
	key := contentVectorKey(content)
	vector, ok := got[key]
	if !ok {
		t.Fatalf("reuse missing content %q with key %s", content, key)
	}
	if !slices.Equal(vector, want) {
		t.Fatalf("reuse vector for %q = %v, want %v", content, vector, want)
	}
}

func assertLegacyWarningCount(t *testing.T, logs string, want int) {
	t.Helper()
	if !strings.Contains(logs, "semantic.conversation_message_state_legacy_rows") {
		t.Fatalf("logs missing legacy warning: %s", logs)
	}
	wantFragment := "legacy_rows=" + strconv.Itoa(want)
	if !strings.Contains(logs, wantFragment) {
		t.Fatalf("logs = %s, want %s", logs, wantFragment)
	}
}

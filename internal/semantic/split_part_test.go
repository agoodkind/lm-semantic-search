package semantic

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"slices"
	"testing"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/model"
)

func TestResultSetsToChunksRestoresSplitPartAndIdentity(t *testing.T) {
	t.Parallel()

	input := []model.StoredChunk{
		{
			Content:       "same",
			RelativePath:  "src/big.go",
			StartLine:     1,
			EndLine:       400,
			FileExtension: ".go",
			SplitPart:     1,
		},
		{
			Content:       "same",
			RelativePath:  "src/big.go",
			StartLine:     1,
			EndLine:       400,
			FileExtension: ".go",
			SplitPart:     513,
		},
	}
	insertColumns := buildInsertBatchColumns(context.Background(), input, false)
	splitParts, err := newSplitPartColumn(
		"test_collection",
		insertColumns.splitParts,
		[]bool{true, true},
	)
	if err != nil {
		t.Fatalf("newSplitPartColumn returned error: %v", err)
	}
	resultSet := milvusclient.ResultSet{
		ResultCount: 2,
		Fields: milvusclient.DataSet{
			column.NewColumnVarChar(contentFieldName, insertColumns.contents),
			column.NewColumnVarChar(relativePathFieldName, insertColumns.relativePaths),
			column.NewColumnInt64(startLineFieldName, insertColumns.startLines),
			column.NewColumnInt64(endLineFieldName, insertColumns.endLines),
			column.NewColumnVarChar(fileExtensionFieldName, insertColumns.fileExtensions),
			column.NewColumnVarChar(metadataFieldName, insertColumns.metadataValues),
			splitParts,
		},
	}

	chunks, err := resultSetsToChunks([]milvusclient.ResultSet{resultSet})
	if err != nil {
		t.Fatalf("resultSetsToChunks returned error: %v", err)
	}
	if got := []int32{chunks[0].SplitPart, chunks[1].SplitPart}; !slices.Equal(got, []int32{1, 513}) {
		t.Fatalf("restored split parts = %v, want [1 513]", got)
	}
	if got := []bool{chunks[0].SplitPartRecorded, chunks[1].SplitPartRecorded}; !slices.Equal(got, []bool{true, true}) {
		t.Fatalf("restored recorded flags = %v, want [true true]", got)
	}
	if generateID(chunks[0], 0) == generateID(chunks[1], 1) {
		t.Fatal("round-tripped split pieces share a primary key")
	}
}

func TestResultSetsToChunksDistinguishesLegacyNullFromRecordedZero(t *testing.T) {
	t.Parallel()

	splitParts, err := newSplitPartColumn(
		"test_collection",
		[]int64{0, 0},
		[]bool{false, true},
	)
	if err != nil {
		t.Fatalf("newSplitPartColumn returned error: %v", err)
	}
	resultSet := milvusclient.ResultSet{
		ResultCount: 2,
		Fields: milvusclient.DataSet{
			column.NewColumnVarChar(contentFieldName, []string{"legacy", "unsplit"}),
			column.NewColumnVarChar(relativePathFieldName, []string{"old.go", "new.go"}),
			column.NewColumnInt64(startLineFieldName, []int64{1, 1}),
			column.NewColumnInt64(endLineFieldName, []int64{2, 2}),
			column.NewColumnVarChar(fileExtensionFieldName, []string{".go", ".go"}),
			column.NewColumnVarChar(metadataFieldName, []string{"{}", "{}"}),
			splitParts,
		},
	}

	chunks, err := resultSetsToChunks([]milvusclient.ResultSet{resultSet})
	if err != nil {
		t.Fatalf("resultSetsToChunks returned error: %v", err)
	}
	if chunks[0].SplitPartRecorded {
		t.Fatal("legacy null split part was treated as recorded zero")
	}
	if !chunks[1].SplitPartRecorded {
		t.Fatal("recorded unsplit row was treated as legacy null")
	}
}

func TestRewriteCopiedRowsPreservesSplitPartsAndDistinctDestinationIDs(t *testing.T) {
	t.Parallel()

	first := model.StoredChunk{
		Content:      "same",
		RelativePath: "src/big.go",
		StartLine:    1,
		EndLine:      400,
		SplitPart:    1,
	}
	second := first
	second.SplitPart = 513
	source := copiedRows{
		chunks:             []model.StoredChunk{first, second},
		ids:                []string{generateID(first, 0), generateID(second, 1)},
		vectors:            [][]float32{{1}, {2}},
		splitPartsRecorded: []bool{true, true},
	}

	rewritten, destinationIDs, recorded := rewriteCopiedRows(source, "src/renamed.go")

	if got := []int32{rewritten[0].SplitPart, rewritten[1].SplitPart}; !slices.Equal(got, []int32{1, 513}) {
		t.Fatalf("copied split parts = %v, want [1 513]", got)
	}
	if destinationIDs[0] == destinationIDs[1] {
		t.Fatalf("copied rows share destination primary key %q", destinationIDs[0])
	}
	if !rewritten[0].SplitPartRecorded || !rewritten[1].SplitPartRecorded {
		t.Fatal("copied split rows lost their recorded flags")
	}
	if !slices.Equal(recorded, []bool{true, true}) {
		t.Fatalf("copied recorded flags = %v, want [true true]", recorded)
	}
}

func TestRewriteCopiedRowsKeepsLegacyNullPartsDistinct(t *testing.T) {
	t.Parallel()

	chunk := model.StoredChunk{
		Content:      "same",
		RelativePath: "src/big.go",
		StartLine:    1,
		EndLine:      400,
	}
	source := copiedRows{
		chunks:             []model.StoredChunk{chunk, chunk},
		ids:                []string{"chunk_1111111111111111", "chunk_2222222222222222"},
		vectors:            [][]float32{{1}, {2}},
		splitPartsRecorded: []bool{false, false},
	}

	rewritten, destinationIDs, recorded := rewriteCopiedRows(source, "src/renamed.go")

	if destinationIDs[0] == destinationIDs[1] {
		t.Fatalf("legacy rows share destination primary key %q", destinationIDs[0])
	}
	if rewritten[0].SplitPartRecorded || rewritten[1].SplitPartRecorded {
		t.Fatal("legacy null rows were treated as recorded zero")
	}
	if !slices.Equal(recorded, []bool{false, false}) {
		t.Fatalf("legacy recorded flags = %v, want [false false]", recorded)
	}
}

func TestGenerateIDKeepsUnsplitIdentityByteIdentical(t *testing.T) {
	t.Parallel()

	chunk := model.StoredChunk{
		Content:       "hello",
		RelativePath:  "src/file.go",
		StartLine:     2,
		EndLine:       4,
		FileExtension: ".go",
	}
	legacyInput := fmt.Sprintf(
		"%s:%d:%d:%s",
		chunk.RelativePath,
		chunk.StartLine,
		chunk.EndLine,
		chunk.Content,
	)
	sum := sha256.Sum256([]byte(legacyInput))
	want := "chunk_" + hex.EncodeToString(sum[:])[:16]

	if got := generateID(chunk, 0); got != want {
		t.Fatalf("generateID for unsplit chunk = %q, want legacy %q", got, want)
	}
}

func TestSplitPartMigrationDecisionIsIdempotent(t *testing.T) {
	t.Parallel()

	schema := entity.NewSchema().
		WithField(entity.NewField().WithName(idFieldName).WithDataType(entity.FieldTypeVarChar))
	first := splitPartFieldsToAdd(schema)
	if len(first) != 1 || first[0].Name != splitPartFieldName {
		t.Fatalf("first splitPartFieldsToAdd = %#v, want splitPart", first)
	}
	if first[0].DataType != entity.FieldTypeInt64 || !first[0].Nullable {
		t.Fatalf("splitPart field = %#v, want nullable int64", first[0])
	}

	schema = schema.WithField(first[0])
	second := splitPartFieldsToAdd(schema)
	if len(second) != 0 {
		t.Fatalf("second splitPartFieldsToAdd = %#v, want no fields", second)
	}
}

func TestConversationAssemblyOrdersRowsBySplitPart(t *testing.T) {
	t.Parallel()

	splitParts, err := column.NewNullableColumnInt64(
		splitPartFieldName,
		[]int64{7, 1},
		[]bool{true, true},
		column.WithSparseNullableMode[int64](true),
	)
	if err != nil {
		t.Fatalf("NewNullableColumnInt64 returned error: %v", err)
	}
	resultSet := milvusclient.ResultSet{
		ResultCount: 2,
		Fields: milvusclient.DataSet{
			column.NewColumnVarChar(relativePathFieldName, []string{"conv/example/0", "conv/example/0"}),
			column.NewColumnVarChar(roleFieldName, []string{"user", "user"}),
			column.NewColumnVarChar(contentFieldName, []string{"second", "first"}),
			column.NewColumnInt64(messageIndexFieldName, []int64{0, 0}),
			column.NewColumnFloatVector(denseVectorFieldName, 1, [][]float32{{2}, {1}}),
			splitParts,
		},
	}
	assemblies := make(map[int32]*storedMessageAssembly)
	reuse := make(map[string][]float32)

	legacyRows, err := appendConversationMessageStateRows(
		resultSet,
		"conv/example/",
		assemblies,
		reuse,
	)
	if err != nil {
		t.Fatalf("appendConversationMessageStateRows returned error: %v", err)
	}
	if legacyRows != 0 {
		t.Fatalf("legacy rows = %d, want 0", legacyRows)
	}
	state := assembleStoredMessageState(assemblies)
	if got := state[0].Text; got != "firstsecond" {
		t.Fatalf("assembled text = %q, want firstsecond", got)
	}
}

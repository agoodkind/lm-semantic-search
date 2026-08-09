package semantic

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"slices"
	"sync"
	"testing"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
)

func TestInsertBatchRoundTripRestoresSplitPartAndIdentity(t *testing.T) {
	t.Parallel()

	const collectionName = "test_collection"
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
	var capturedOption milvusclient.InsertOption
	service := &Service{
		cfg: config.Config{EmbeddingModel: "model-a"},
		insertRows: func(
			_ context.Context,
			option milvusclient.InsertOption,
		) (milvusclient.InsertResult, error) {
			capturedOption = option
			return milvusclient.InsertResult{InsertCount: int64(len(input))}, nil
		},
	}
	migration := &splitPartMigration{}
	migration.once.Do(func() {})
	service.ensuredSplitPartColumns.Store(collectionName, migration)
	reuseIdentityMigration := &reuseIdentityMigration{}
	reuseIdentityMigration.once.Do(func() {})
	service.ensuredReuseIdentityColumns.Store(collectionName, reuseIdentityMigration)

	err := service.insertBatch(
		context.Background(),
		collectionName,
		input,
		[][]float32{{1}, {2}},
		StoreColumnSetCode,
	)
	if err != nil {
		t.Fatalf("insertBatch returned error: %v", err)
	}
	if capturedOption == nil {
		t.Fatal("insertBatch did not send an insert option")
	}
	request, err := capturedOption.InsertRequest(testInsertCollection(collectionName, 1))
	if err != nil {
		t.Fatalf("captured insert option returned error: %v", err)
	}
	fields := make(milvusclient.DataSet, 0, len(request.GetFieldsData()))
	for _, fieldData := range request.GetFieldsData() {
		fieldColumn, columnErr := column.FieldDataColumn(fieldData, 0, -1)
		if columnErr != nil {
			t.Fatalf("convert inserted field %q: %v", fieldData.GetFieldName(), columnErr)
		}
		fields = append(fields, fieldColumn)
	}
	resultSet := milvusclient.ResultSet{
		ResultCount: int(request.GetNumRows()),
		Fields:      fields,
	}
	contentHashes := resultSet.GetColumn(contentHashFieldName)
	if contentHashes == nil {
		t.Fatal("inserted row has no content hash column")
	}
	embeddingModels := resultSet.GetColumn(embeddingModelFieldName)
	if embeddingModels == nil {
		t.Fatal("inserted row has no embedding model column")
	}
	for index, chunk := range input {
		got, keyErr := contentHashes.GetAsString(index)
		if keyErr != nil {
			t.Fatalf("read content hash %d: %v", index, keyErr)
		}
		want := contentHash(chunk.Content)
		if got != want {
			t.Fatalf("content hash %d = %q, want %q", index, got, want)
		}
		modelName, modelErr := embeddingModels.GetAsString(index)
		if modelErr != nil {
			t.Fatalf("read embedding model %d: %v", index, modelErr)
		}
		if modelName != "model-a" {
			t.Fatalf("embedding model %d = %q, want model-a", index, modelName)
		}
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

func TestContentHashContainsOnlyNormalizedExactContent(t *testing.T) {
	t.Parallel()

	want := contentHash("same content")
	if got := contentHash("same content"); got != want {
		t.Fatalf("content hash is unstable: got %q, want %q", got, want)
	}
	if want == contentHash("different content") {
		t.Fatal("different exact content shared a content storage key")
	}
	invalidUTF8 := string([]byte{'x', 0xff})
	if contentHash(invalidUTF8) != contentHash("x\uFFFD") {
		t.Fatal("invalid UTF-8 and its normalized exact content have different keys")
	}
}

func TestEmbeddingModelCompatibilityRejectsOnlyKnownUnequalNames(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name    string
		stored  string
		current string
		want    bool
	}{
		{name: "both absent", stored: "", current: "", want: true},
		{name: "stored absent", stored: "", current: "model-a", want: true},
		{name: "current absent", stored: "model-a", current: "", want: true},
		{name: "equal known", stored: "model-a", current: "model-a", want: true},
		{name: "unequal known", stored: "model-a", current: "model-b", want: false},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			if got := embeddingModelsCompatible(testCase.stored, testCase.current); got != testCase.want {
				t.Fatalf("embeddingModelsCompatible(%q, %q) = %t, want %t", testCase.stored, testCase.current, got, testCase.want)
			}
		})
	}
}

func testInsertCollection(collectionName string, dimension int64) *entity.Collection {
	schema := entity.NewSchema().
		WithName(collectionName).
		WithField(entity.NewField().
			WithName(idFieldName).
			WithDataType(entity.FieldTypeVarChar).
			WithMaxLength(512).
			WithIsPrimaryKey(true)).
		WithField(entity.NewField().
			WithName(contentFieldName).
			WithDataType(entity.FieldTypeVarChar).
			WithMaxLength(65535)).
		WithField(entity.NewField().
			WithName(relativePathFieldName).
			WithDataType(entity.FieldTypeVarChar).
			WithMaxLength(1024)).
		WithField(entity.NewField().
			WithName(startLineFieldName).
			WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().
			WithName(endLineFieldName).
			WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().
			WithName(fileExtensionFieldName).
			WithDataType(entity.FieldTypeVarChar).
			WithMaxLength(32)).
		WithField(entity.NewField().
			WithName(metadataFieldName).
			WithDataType(entity.FieldTypeVarChar).
			WithMaxLength(65535)).
		WithField(contentHashField()).
		WithField(embeddingModelField()).
		WithField(splitPartField()).
		WithField(entity.NewField().
			WithName(denseVectorFieldName).
			WithDataType(entity.FieldTypeFloatVector).
			WithDim(dimension))
	return &entity.Collection{Name: collectionName, Schema: schema}
}

func TestInsertBatchRejectsPartialInsertCount(t *testing.T) {
	t.Parallel()

	const collectionName = "test_collection"
	service := &Service{
		insertRows: func(
			_ context.Context,
			_ milvusclient.InsertOption,
		) (milvusclient.InsertResult, error) {
			return milvusclient.InsertResult{InsertCount: 1}, nil
		},
	}
	migration := &splitPartMigration{}
	migration.once.Do(func() {})
	service.ensuredSplitPartColumns.Store(collectionName, migration)
	reuseIdentityMigration := &reuseIdentityMigration{}
	reuseIdentityMigration.once.Do(func() {})
	service.ensuredReuseIdentityColumns.Store(collectionName, reuseIdentityMigration)
	chunks := []model.StoredChunk{
		{Content: "first", RelativePath: "first.go"},
		{Content: "second", RelativePath: "second.go"},
	}

	err := service.insertBatch(
		context.Background(),
		collectionName,
		chunks,
		[][]float32{{1}, {2}},
		StoreColumnSetCode,
	)
	if err == nil {
		t.Fatal("insertBatch returned nil for a partial insert count")
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

func TestInvalidateCollectionCachesClearsSchemaState(t *testing.T) {
	t.Parallel()

	const collectionName = "test_collection"
	service := &Service{}
	service.ensuredConvColumns.Store(collectionName, "conversation")
	service.ensuredSplitPartColumns.Store(collectionName, "split-part")
	service.ensuredReuseIdentityColumns.Store(collectionName, "reuse-identity")
	service.mmapPolicyVersions = map[string]int{collectionName: mmapPolicyVersion}
	service.ensuredBackfill.Store(collectionName, "backfill")

	service.invalidateCollectionCaches(collectionName)

	caches := []*sync.Map{
		&service.ensuredConvColumns,
		&service.ensuredSplitPartColumns,
		&service.ensuredReuseIdentityColumns,
		&service.ensuredBackfill,
	}
	for index, cache := range caches {
		if _, found := cache.Load(collectionName); found {
			t.Fatalf("cache %d retained collection state", index)
		}
	}
	if _, found := service.mmapPolicyVersions[collectionName]; found {
		t.Fatal("mmap policy cache retained collection state")
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

func TestConversationAssemblyOrdersMigratedRowsDeterministically(t *testing.T) {
	t.Parallel()

	splitParts, err := newSplitPartColumn(
		"test_collection",
		[]int64{0, 0},
		[]bool{false, false},
	)
	if err != nil {
		t.Fatalf("newSplitPartColumn returned error: %v", err)
	}
	resultSet := milvusclient.ResultSet{
		ResultCount: 2,
		Fields: milvusclient.DataSet{
			column.NewColumnVarChar(
				relativePathFieldName,
				[]string{"conv/example/0", "conv/example/0"},
			),
			column.NewColumnVarChar(roleFieldName, []string{"user", "user"}),
			column.NewColumnVarChar(contentFieldName, []string{"second", "first"}),
			column.NewColumnInt64(messageIndexFieldName, []int64{0, 0}),
			column.NewColumnFloatVector(
				denseVectorFieldName,
				1,
				[][]float32{{2}, {1}},
			),
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
		t.Fatalf("assembled migrated text = %q, want firstsecond", got)
	}
}

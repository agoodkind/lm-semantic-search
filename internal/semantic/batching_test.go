package semantic

import (
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/client/v2/column"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
	"google.golang.org/protobuf/proto"
)

func chunkOfBytes(n int) model.StoredChunk {
	return model.StoredChunk{Content: strings.Repeat("a", n)}
}

// conversationSizedChunks builds count chunks of distinct content at roughly the
// size of a real conversation chunk, 1,500 bytes, which the packer estimates at
// 376 tokens each. Distinct content matters because the reuse map is keyed by
// content hash, so identical chunks could not be selectively marked reused.
func conversationSizedChunks(count int) []model.StoredChunk {
	const conversationChunkBytes = 1500
	chunks := make([]model.StoredChunk, count)
	for index := range chunks {
		chunks[index] = model.StoredChunk{
			Content: strconv.Itoa(index) + ":" + strings.Repeat("a", conversationChunkBytes),
		}
	}
	return chunks
}

// reuseCoveringAllBut returns a reuse map holding a vector for every chunk whose
// index is not a multiple of embedEvery, standing in for a re-ingest where most
// content is unchanged.
func reuseCoveringAllBut(chunks []model.StoredChunk, embedEvery int) map[string][]float32 {
	reuse := make(map[string][]float32, len(chunks))
	for index, chunk := range chunks {
		if index%embedEvery == 0 {
			continue
		}
		reuse[contentVectorKey(chunk.Content)] = []float32{float32(index)}
	}
	return reuse
}

func assertGroupRowCounts(t *testing.T, groups [][]model.StoredChunk, want []int) {
	t.Helper()
	got := make([]int, 0, len(groups))
	for _, group := range groups {
		got = append(got, len(group))
	}
	if len(got) != len(want) {
		t.Fatalf("group row counts = %v, want %v", got, want)
	}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("group row counts = %v, want %v", got, want)
		}
	}
}

func TestPackChunksEmptyInputYieldsNoGroups(t *testing.T) {
	groups := packChunksByEstimatedTokens(nil, 32, 6000, nil)
	if len(groups) != 0 {
		t.Fatalf("groups = %d, want 0", len(groups))
	}
}

func TestPackChunksSingleOversizeChunkShipsAlone(t *testing.T) {
	chunks := []model.StoredChunk{chunkOfBytes(100_000), chunkOfBytes(4)}
	groups := packChunksByEstimatedTokens(chunks, 32, 6000, nil)
	if len(groups) != 2 {
		t.Fatalf("groups = %d, want 2", len(groups))
	}
	if len(groups[0]) != 1 {
		t.Fatalf("first group rows = %d, want 1 (oversize ships alone)", len(groups[0]))
	}
}

func TestPackChunksClosesOnTokenBudget(t *testing.T) {
	// 10 chunks of 400 bytes = 100 estimated tokens each; budget 250 packs 2 per group.
	chunks := make([]model.StoredChunk, 10)
	for i := range chunks {
		chunks[i] = chunkOfBytes(400)
	}
	groups := packChunksByEstimatedTokens(chunks, 32, 250, nil)
	if len(groups) != 5 {
		t.Fatalf("groups = %d, want 5", len(groups))
	}
}

func TestPackChunksClosesOnRowCap(t *testing.T) {
	chunks := make([]model.StoredChunk, 10)
	for i := range chunks {
		chunks[i] = chunkOfBytes(4)
	}
	groups := packChunksByEstimatedTokens(chunks, 4, 6000, nil)
	want := []int{4, 4, 2}
	if len(groups) != len(want) {
		t.Fatalf("groups = %d, want %d", len(groups), len(want))
	}
	for i, group := range groups {
		if len(group) != want[i] {
			t.Fatalf("group %d rows = %d, want %d", i, len(group), want[i])
		}
	}
}

func TestPackChunksPreservesOrderAndCoverage(t *testing.T) {
	chunks := make([]model.StoredChunk, 25)
	for i := range chunks {
		chunks[i] = chunkOfBytes((i*53)%900 + 1)
	}
	groups := packChunksByEstimatedTokens(chunks, 8, 300, nil)
	var flattened []model.StoredChunk
	for _, group := range groups {
		flattened = append(flattened, group...)
	}
	if len(flattened) != len(chunks) {
		t.Fatalf("flattened = %d chunks, want %d", len(flattened), len(chunks))
	}
	for i := range chunks {
		if flattened[i].Content != chunks[i].Content {
			t.Fatalf("chunk %d out of order", i)
		}
	}
}

func TestPackForEmbeddingClosesOnConfiguredTokenBudget(t *testing.T) {
	service := &Service{cfg: config.Config{
		EmbeddingBatchSize:        32,
		EmbeddingBatchTokenBudget: 250,
	}}
	chunks := []model.StoredChunk{
		chunkOfBytes(400),
		chunkOfBytes(400),
		chunkOfBytes(400),
	}

	groups := service.packForEmbedding(chunks, nil)
	want := []int{2, 1}
	if len(groups) != len(want) {
		t.Fatalf("groups = %d, want %d", len(groups), len(want))
	}
	for i, group := range groups {
		if len(group) != want[i] {
			t.Fatalf("group %d rows = %d, want %d", i, len(group), want[i])
		}
	}
}

// TestPackForEmbeddingFillsRowCapWhenReuseCoversMostChunks pins the capacity
// invariant: a batch fills to the configured budget rather than stopping early
// on tokens the request will never carry. Sixty-four conversation-sized chunks
// estimate at 376 tokens each, so charging all of them against a 6,000-token
// budget closes a group after 15 rows and never reaches the 64-row ceiling. Only
// eight of them actually reach the embedder here, which is 3,008 estimated
// tokens, so the whole stream belongs in one request.
func TestPackForEmbeddingFillsRowCapWhenReuseCoversMostChunks(t *testing.T) {
	service := &Service{cfg: config.Config{
		EmbeddingBatchSize:        64,
		EmbeddingBatchTokenBudget: 6000,
	}}
	chunks := conversationSizedChunks(64)
	reuse := reuseCoveringAllBut(chunks, 8)

	groups := service.packForEmbedding(chunks, reuse)

	assertGroupRowCounts(t, groups, []int{64})
	assertGroupsWithinTokenBudget(t, groups, reuse, 6000)
}

// TestReuseAwareEmbeddingAndInsertPackingUseIndependentBudgets pins the two
// request limits independently. Reused rows cost the embedder nothing, so all
// 11,000 rows belong in one embedding group. Every row still contributes its
// content and dense vector to the insert payload, so insert groups must close
// below 192 MiB. That ceiling reserves 64 MiB below the store's 256 MiB receive
// limit for the remaining columns and protobuf overhead.
func TestReuseAwareEmbeddingAndInsertPackingUseIndependentBudgets(t *testing.T) {
	const (
		rowCount          = 11_000
		contentBytes      = 9_216
		vectorDimension   = 4_096
		insertByteCeiling = 192 << 20
	)
	sharedContent := strings.Repeat("a", contentBytes)
	chunks := make([]model.StoredChunk, rowCount)
	for index := range chunks {
		chunks[index] = model.StoredChunk{
			Content:      sharedContent,
			RelativePath: strconv.Itoa(index),
		}
	}
	vector := make([]float32, vectorDimension)
	reuse := map[string][]float32{contentVectorKey(sharedContent): vector}
	service := &Service{cfg: config.Config{
		EmbeddingBatchSize:        rowCount,
		EmbeddingBatchTokenBudget: 6000,
	}}

	embeddingGroups := service.packForEmbedding(chunks, reuse)
	assertGroupRowCounts(t, embeddingGroups, []int{rowCount})

	insertGroups := packChunksByEstimatedInsertBytes(
		chunks,
		len(vector),
		insertByteCeiling,
		StoreColumnSetCode,
	)
	assertGroupsCoverChunksInOrder(t, insertGroups, chunks)
	if len(insertGroups) < 2 {
		t.Fatalf("insert groups = %d, want more than one", len(insertGroups))
	}
	for groupIndex, group := range insertGroups {
		estimatedBytes := 0
		for _, chunk := range group {
			estimatedBytes += estimatedInsertRowBytes(
				chunk,
				len(vector),
				StoreColumnSetCode,
			)
		}
		if estimatedBytes > insertByteCeiling {
			t.Fatalf(
				"insert group %d estimated bytes = %d, want at most %d",
				groupIndex,
				estimatedBytes,
				insertByteCeiling,
			)
		}
	}
}

func TestInsertPackingKeepsReviewerShapedRequestUnderTransportLimit(t *testing.T) {
	const (
		rowCount             = 256
		contentBytes         = 1
		relativePathBytes    = 1024
		fileExtensionBytes   = 32
		vectorDimension      = 384
		scaledInsertBudget   = 192 << 10
		scaledTransportLimit = 256 << 10
	)
	chunks := make([]model.StoredChunk, rowCount)
	for index := range chunks {
		chunks[index] = model.StoredChunk{
			Content:       strings.Repeat("c", contentBytes),
			RelativePath:  strings.Repeat("p", relativePathBytes),
			StartLine:     int32(index),
			EndLine:       int32(index + 1),
			FileExtension: strings.Repeat("e", fileExtensionBytes),
		}
	}

	conversationChunks := make([]model.StoredChunk, len(chunks))
	copy(conversationChunks, chunks)
	for index := range conversationChunks {
		conversationChunks[index].ConversationID = strings.Repeat("p", 32) +
			":" +
			strings.Repeat("c", 223)
		conversationChunks[index].ParentConversationID = strings.Repeat("q", 256)
		conversationChunks[index].Role = strings.Repeat("R", 64)
		conversationChunks[index].WorkspaceRoot = strings.Repeat("w", 1024)
		conversationChunks[index].TimestampUnix = int64(index)
		conversationChunks[index].MessageIndex = int32(index)
	}

	tests := []struct {
		name      string
		chunks    []model.StoredChunk
		columnSet StoreColumnSet
	}{
		{
			name:      "code columns",
			chunks:    chunks,
			columnSet: StoreColumnSetCode,
		},
		{
			name:      "conversation columns",
			chunks:    conversationChunks,
			columnSet: StoreColumnSetConversation,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			groups := packChunksByEstimatedInsertBytes(
				test.chunks,
				vectorDimension,
				scaledInsertBudget,
				test.columnSet,
			)

			for groupIndex, group := range groups {
				requestBytes := actualInsertRequestBytes(
					group,
					vectorDimension,
					test.columnSet,
				)
				if requestBytes > scaledTransportLimit {
					t.Fatalf(
						"insert group %d request bytes = %d, want at most %d",
						groupIndex,
						requestBytes,
						scaledTransportLimit,
					)
				}
			}
		})
	}
}

func actualInsertRequestBytes(
	chunks []model.StoredChunk,
	vectorDimension int,
	columnSet StoreColumnSet,
) int {
	ids := make([]string, 0, len(chunks))
	contents := make([]string, 0, len(chunks))
	contentVectorKeys := make([]string, 0, len(chunks))
	relativePaths := make([]string, 0, len(chunks))
	startLines := make([]int64, 0, len(chunks))
	endLines := make([]int64, 0, len(chunks))
	fileExtensions := make([]string, 0, len(chunks))
	metadataValues := make([]string, 0, len(chunks))
	vectors := make([][]float32, 0, len(chunks))
	scalars := newConversationScalarColumns(columnSet.ConversationScalars(), len(chunks))
	for index, chunk := range chunks {
		content, _ := sanitizeUTF8(chunk.Content)
		relativePath, _ := sanitizeUTF8(chunk.RelativePath)
		fileExtension, _ := sanitizeUTF8(chunk.FileExtension)
		metadataValue, _ := sanitizeUTF8(encodeMetadata(chunk))
		ids = append(ids, generateID(chunk, index))
		contents = append(contents, content)
		contentVectorKeys = append(contentVectorKeys, ContentVectorKey(content))
		relativePaths = append(relativePaths, relativePath)
		startLines = append(startLines, int64(chunk.StartLine))
		endLines = append(endLines, int64(chunk.EndLine))
		fileExtensions = append(fileExtensions, fileExtension)
		metadataValues = append(metadataValues, metadataValue)
		vectors = append(vectors, make([]float32, vectorDimension))
		scalars.append(chunk)
	}

	fieldsData := []*schemapb.FieldData{
		column.NewColumnVarChar(idFieldName, ids).FieldData(),
		column.NewColumnVarChar(contentFieldName, contents).FieldData(),
		column.NewColumnVarChar(contentVectorKeyFieldName, contentVectorKeys).FieldData(),
		column.NewColumnVarChar(relativePathFieldName, relativePaths).FieldData(),
		column.NewColumnInt64(startLineFieldName, startLines).FieldData(),
		column.NewColumnInt64(endLineFieldName, endLines).FieldData(),
		column.NewColumnVarChar(fileExtensionFieldName, fileExtensions).FieldData(),
		column.NewColumnVarChar(metadataFieldName, metadataValues).FieldData(),
		column.NewColumnFloatVector(
			denseVectorFieldName,
			vectorDimension,
			vectors,
		).FieldData(),
	}
	if columnSet.ConversationScalars() {
		fieldsData = append(
			fieldsData,
			column.NewColumnVarChar(
				conversationIDFieldName,
				scalars.conversationIDs,
			).FieldData(),
			column.NewColumnVarChar(
				parentConversationIDFieldName,
				scalars.parentConversationIDs,
			).FieldData(),
			column.NewColumnVarChar(roleFieldName, scalars.roles).FieldData(),
			column.NewColumnVarChar(providerFieldName, scalars.providers).FieldData(),
			column.NewColumnVarChar(
				workspaceRootFieldName,
				scalars.workspaceRoots,
			).FieldData(),
			column.NewColumnBool(archivedFieldName, scalars.archiveds).FieldData(),
			column.NewColumnInt64(timestampUnixFieldName, scalars.timestamps).FieldData(),
			column.NewColumnInt64(
				messageIndexFieldName,
				scalars.messageIndexes,
			).FieldData(),
		)
	}

	request := &milvuspb.InsertRequest{
		CollectionName: "test_collection",
		NumRows:        uint32(len(chunks)),
		FieldsData:     fieldsData,
	}
	return proto.Size(request)
}

// TestPackForEmbeddingAllReusedGroupsOnlyOnRowCap covers the extreme of the same
// invariant. When almost every chunk is served from the reuse map the token
// budget cannot close a group at all, so the row ceiling is the only bound and
// the groups are whole multiples of it.
func TestPackForEmbeddingAllReusedGroupsOnlyOnRowCap(t *testing.T) {
	service := &Service{cfg: config.Config{
		EmbeddingBatchSize:        16,
		EmbeddingBatchTokenBudget: 6000,
	}}
	chunks := conversationSizedChunks(40)
	reuse := reuseCoveringAllBut(chunks, 40)

	groups := service.packForEmbedding(chunks, reuse)

	assertGroupRowCounts(t, groups, []int{16, 16, 8})
	assertGroupsWithinTokenBudget(t, groups, reuse, 6000)
}

// TestPackForEmbeddingNeverExceedsTokenBudgetWithPartialReuse is the safety half
// of the capacity change. Filling to the row ceiling must not let the embedded
// share of a group grow past the configured budget, so a mixed stream under a
// row ceiling far above what the budget allows still closes on tokens.
func TestPackForEmbeddingNeverExceedsTokenBudgetWithPartialReuse(t *testing.T) {
	service := &Service{cfg: config.Config{
		EmbeddingBatchSize:        256,
		EmbeddingBatchTokenBudget: 6000,
	}}
	chunks := conversationSizedChunks(120)
	reuse := reuseCoveringAllBut(chunks, 2)

	groups := service.packForEmbedding(chunks, reuse)

	assertGroupsWithinTokenBudget(t, groups, reuse, 6000)
	assertGroupsCoverChunksInOrder(t, groups, chunks)
	if len(groups) < 2 {
		t.Fatalf("groups = %d, want more than one: 60 embedded chunks at 376 tokens exceed the 6000-token budget", len(groups))
	}
}

// assertGroupsWithinTokenBudget fails when any group's embedded share, which is
// what the embedding request actually carries, exceeds tokenBudget. A group
// holding one chunk is exempt because an oversize chunk ships alone.
func assertGroupsWithinTokenBudget(t *testing.T, groups [][]model.StoredChunk, reuse map[string][]float32, tokenBudget int) {
	t.Helper()
	for index, group := range groups {
		total := 0
		for _, chunk := range group {
			total += embeddedTokenCount(chunk, reuse)
		}
		if len(group) > 1 && total > tokenBudget {
			t.Fatalf("group %d carries %d embedded tokens, want at most %d", index, total, tokenBudget)
		}
	}
}

// assertGroupsCoverChunksInOrder fails unless every chunk lands in exactly one
// group, in the original order.
func assertGroupsCoverChunksInOrder(t *testing.T, groups [][]model.StoredChunk, chunks []model.StoredChunk) {
	t.Helper()
	flattened := make([]model.StoredChunk, 0, len(chunks))
	for _, group := range groups {
		flattened = append(flattened, group...)
	}
	if len(flattened) != len(chunks) {
		t.Fatalf("groups hold %d chunks, want %d", len(flattened), len(chunks))
	}
	for index := range chunks {
		if flattened[index] != chunks[index] {
			t.Fatalf("chunk %d is out of order", index)
		}
	}
}

func TestPackForEmbeddingClosesOnConfiguredRowCap(t *testing.T) {
	service := &Service{cfg: config.Config{
		EmbeddingBatchSize:        2,
		EmbeddingBatchTokenBudget: 6000,
	}}
	chunks := []model.StoredChunk{
		chunkOfBytes(4),
		chunkOfBytes(4),
		chunkOfBytes(4),
		chunkOfBytes(4),
		chunkOfBytes(4),
	}

	groups := service.packForEmbedding(chunks, nil)
	want := []int{2, 2, 1}
	if len(groups) != len(want) {
		t.Fatalf("groups = %d, want %d", len(groups), len(want))
	}
	for i, group := range groups {
		if len(group) != want[i] {
			t.Fatalf("group %d rows = %d, want %d", i, len(group), want[i])
		}
	}
}

// TestPackChunksLargeInputKeepsEveryChunkInOrder packs an input large enough to
// need many groups and asserts the caller-visible outcome: every chunk appears
// exactly once, in input order, and no group exceeds the token budget.
func TestPackChunksLargeInputKeepsEveryChunkInOrder(t *testing.T) {
	const (
		chunkCount  = 2000
		chunkBytes  = 1500
		tokenBudget = 6000
	)
	chunks := make([]model.StoredChunk, 0, chunkCount)
	for index := range chunkCount {
		chunks = append(chunks, model.StoredChunk{
			Content: strconv.Itoa(index) + strings.Repeat("x", chunkBytes),
		})
	}

	groups := packChunksByEstimatedTokens(chunks, math.MaxInt, tokenBudget, nil)

	flattened := make([]model.StoredChunk, 0, chunkCount)
	for _, group := range groups {
		if len(group) == 0 {
			t.Fatal("packer emitted an empty group")
		}
		groupTokens := 0
		for _, chunk := range group {
			groupTokens += estimatedTokenCount(chunk.Content)
		}
		if len(group) > 1 && groupTokens > tokenBudget {
			t.Fatalf("group of %d rows estimated %d tokens, over the %d budget", len(group), groupTokens, tokenBudget)
		}
		flattened = append(flattened, group...)
	}
	if len(flattened) != chunkCount {
		t.Fatalf("packed %d chunks, want %d", len(flattened), chunkCount)
	}
	for index := range flattened {
		if flattened[index].Content != chunks[index].Content {
			t.Fatalf("chunk %d out of order or altered", index)
		}
	}
}

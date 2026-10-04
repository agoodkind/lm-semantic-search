//go:build live

package live

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func TestDuplicateLegacyCorpusReuseImmutabilitySmoke(t *testing.T) {
	const (
		duplicateRowCount = 16384
		vectorDimension   = 4096
	)

	harness := newHarness(t)
	embeddingRecorder := &embeddingCallRecorder{}
	embedServer := newFakeEmbeddingServerWithRecorder(
		t,
		nil,
		vectorDimension,
		embeddingRecorder,
	)
	lookupConfig := harness.childConfig()
	lookupConfig.OpenAIBaseURL = embedServer.URL
	lookupConfig.EmbeddingDimension = vectorDimension
	service, err := semantic.NewService(harness.milvusContext, lookupConfig)
	if err != nil {
		t.Fatalf("open 4096-dimension semantic service: %v", err)
	}
	t.Cleanup(func() { _ = service.Close(context.Background()) })

	sourcePath := filepath.Join(harness.stateRoot, "duplicate-legacy-"+randomID())
	collectionName := service.CollectionName(sourcePath)
	catalogName := semantic.ReuseCatalogCollectionName(lookupConfig)
	harness.trackCollectionFamily(collectionName)
	harness.trackTemporaryCollection(catalogName)
	schemaSeedContent := "duplicate legacy schema seed"
	if err := service.StageReindex(
		context.Background(),
		sourcePath,
		[]model.StoredChunk{{Content: schemaSeedContent, RelativePath: "seed.txt"}},
		semantic.Removal{},
		nil,
		map[string][]float32{},
		semantic.CodeColumns(),
	); err != nil {
		t.Fatalf("stage duplicate legacy source: %v", err)
	}
	if err := service.PromoteStaging(context.Background(), sourcePath); err != nil {
		t.Fatalf("promote duplicate legacy source: %v", err)
	}

	duplicateContent := "duplicate legacy transport sentinel"
	firstControlContent := "unique legacy control one"
	secondControlContent := "unique legacy control two"
	duplicateVector := markedVector(vectorDimension, 1, 2)
	firstControlVector := markedVector(vectorDimension, 3, 4)
	secondControlVector := markedVector(vectorDimension, 5, 6)
	fixtureRows := makeDuplicateLegacyRows(
		duplicateRowCount,
		duplicateContent,
		duplicateVector,
		[]legacyReuseFixtureRow{
			{
				id:            "legacy-control-one-" + randomID(),
				content:       firstControlContent,
				relativePath:  "legacy/control/one",
				startLine:     1,
				endLine:       2,
				fileExtension: "txt",
				metadata:      `{"control":1}`,
				vector:        firstControlVector,
			},
			{
				id:            "legacy-control-two-" + randomID(),
				content:       secondControlContent,
				relativePath:  "legacy/control/two",
				startLine:     3,
				endLine:       4,
				fileExtension: "md",
				metadata:      `{"control":2}`,
				vector:        secondControlVector,
			},
		},
	)
	insertLegacyRows(t, harness, collectionName, fixtureRows)

	fixtureIDs := make([]string, 0, len(fixtureRows))
	for _, row := range fixtureRows {
		fixtureIDs = append(fixtureIDs, row.id)
	}
	rowCountBefore := collectionRowCount(t, harness, collectionName)
	wantRowCount := int64(len(fixtureIDs) + 1)
	if rowCountBefore != wantRowCount {
		t.Fatalf("source row count before lookup = %d, want %d", rowCountBefore, wantRowCount)
	}
	rowsBefore := snapshotsForIDs(t, harness, collectionName, fixtureIDs)
	rowsBefore = append(
		rowsBefore,
		snapshotsForContent(t, harness, collectionName, schemaSeedContent)...,
	)
	slices.SortFunc(rowsBefore, compareStoredRowSnapshots)
	if len(rowsBefore) != int(wantRowCount) {
		t.Fatalf("source snapshots before lookup = %d, want %d", len(rowsBefore), wantRowCount)
	}
	catalogBefore := reuseCatalogSnapshots(t, harness, catalogName)
	embeddingCallsBefore := len(embeddingRecorder.snapshot())
	harness.callRecorder.reset()

	requestedContents := []string{
		duplicateContent,
		firstControlContent,
		secondControlContent,
	}
	requestedChunks := make([]model.StoredChunk, 0, len(requestedContents))
	for _, content := range requestedContents {
		requestedChunks = append(requestedChunks, model.StoredChunk{Content: content})
	}
	expectedVectors := map[string][]float32{
		duplicateContent:     duplicateVector,
		firstControlContent:  firstControlVector,
		secondControlContent: secondControlVector,
	}
	for attempt := 1; attempt <= 2; attempt++ {
		reuse, loadErr := service.LoadReuseVectorsForContents(
			context.Background(),
			collectionName,
			requestedChunks,
		)
		if loadErr != nil {
			t.Fatalf("load duplicate legacy reuse attempt %d: %v", attempt, loadErr)
		}
		if len(reuse) != len(requestedContents) {
			t.Fatalf(
				"reuse vectors on attempt %d = %d, want %d",
				attempt,
				len(reuse),
				len(requestedContents),
			)
		}
		for content, expectedVector := range expectedVectors {
			actualVector, found := reuse[semantic.ContentVectorKey(content)]
			if !found {
				t.Fatalf("reuse attempt %d omitted %q", attempt, content)
			}
			if !slices.Equal(actualVector, expectedVector) {
				t.Fatalf(
					"reuse attempt %d vector for %q has SHA-256 %s, want %s",
					attempt,
					content,
					checksumVector(entity.FloatVector(actualVector)),
					checksumVector(entity.FloatVector(expectedVector)),
				)
			}
		}
	}

	rowCountAfter := collectionRowCount(t, harness, collectionName)
	rowsAfter := snapshotsForIDs(t, harness, collectionName, fixtureIDs)
	rowsAfter = append(
		rowsAfter,
		snapshotsForContent(t, harness, collectionName, schemaSeedContent)...,
	)
	slices.SortFunc(rowsAfter, compareStoredRowSnapshots)
	catalogAfter := reuseCatalogSnapshots(t, harness, catalogName)
	if rowCountAfter != rowCountBefore {
		t.Fatalf("source row count changed: before=%d after=%d", rowCountBefore, rowCountAfter)
	}
	if !slices.Equal(rowsAfter, rowsBefore) {
		t.Fatal("source scalar fields or vector SHA-256 hashes changed during reuse lookup")
	}
	if !slices.Equal(catalogAfter, catalogBefore) {
		t.Fatalf("reuse catalog changed: before=%+v after=%+v", catalogBefore, catalogAfter)
	}
	if embeddingCallsAfter := len(embeddingRecorder.snapshot()); embeddingCallsAfter != embeddingCallsBefore {
		t.Fatalf(
			"embedding calls changed during lookup: before=%d after=%d",
			embeddingCallsBefore,
			embeddingCallsAfter,
		)
	}
	assertNoMilvusWrites(t, harness.callRecorder.snapshot())
	t.Logf(
		"duplicate_rows=%d vector_dimension=%d source_rows=%d catalog_rows=%d duplicate_vector_sha256=%s",
		duplicateRowCount,
		vectorDimension,
		rowCountAfter,
		len(catalogAfter),
		checksumVector(entity.FloatVector(duplicateVector)),
	)
}

func TestUnknownConfiguredDimensionScopesCatalogByReturnedVectorWidth(t *testing.T) {
	const (
		initialDimension = 1536
		targetDimension  = 4096
	)

	harness := newHarness(t)
	initialServer := newFakeEmbeddingServerWithDimension(t, nil, initialDimension)
	targetServer := newFakeEmbeddingServerWithDimension(t, nil, targetDimension)

	initialConfig := harness.childConfig()
	initialConfig.OpenAIBaseURL = initialServer.URL
	initialConfig.EmbeddingDimension = 0
	targetConfig := initialConfig
	targetConfig.OpenAIBaseURL = targetServer.URL

	initialService, err := semantic.NewService(harness.milvusContext, initialConfig)
	if err != nil {
		t.Fatalf("open initial dimension service: %v", err)
	}
	t.Cleanup(func() { _ = initialService.Close(context.Background()) })
	targetService, err := semantic.NewService(harness.milvusContext, targetConfig)
	if err != nil {
		t.Fatalf("open target dimension service: %v", err)
	}
	t.Cleanup(func() { _ = targetService.Close(context.Background()) })

	configuredCatalogName := semantic.ReuseCatalogCollectionName(initialConfig)
	initialCatalogConfig := initialConfig
	initialCatalogConfig.EmbeddingDimension = initialDimension
	initialCatalogName := semantic.ReuseCatalogCollectionName(initialCatalogConfig)
	targetCatalogConfig := targetConfig
	targetCatalogConfig.EmbeddingDimension = targetDimension
	targetCatalogName := semantic.ReuseCatalogCollectionName(targetCatalogConfig)
	initialPath := filepath.Join(harness.stateRoot, "dimension-1536-"+randomID())
	targetPath := filepath.Join(harness.stateRoot, "dimension-4096-"+randomID())
	sharedContent := "dimension transition shared content"
	newContent := "dimension transition new content"
	for _, collectionName := range []string{
		configuredCatalogName,
		initialCatalogName,
		targetCatalogName,
	} {
		harness.trackTemporaryCollection(collectionName)
	}
	harness.trackCollectionFamily(initialService.CollectionName(initialPath))
	harness.trackCollectionFamily(targetService.CollectionName(targetPath))

	if err := initialService.StageReindex(
		context.Background(),
		initialPath,
		[]model.StoredChunk{{Content: sharedContent, RelativePath: "seed.txt"}},
		semantic.Removal{},
		nil,
		map[string][]float32{},
		semantic.CodeColumns(),
	); err != nil {
		t.Fatalf("stage initial dimension row: %v", err)
	}
	if err := initialService.PromoteStaging(context.Background(), initialPath); err != nil {
		t.Fatalf("promote initial dimension row: %v", err)
	}

	var targetProgress semantic.Progress
	if err := targetService.StageReindex(
		context.Background(),
		targetPath,
		[]model.StoredChunk{
			{Content: sharedContent, RelativePath: "shared.txt"},
			{Content: newContent, RelativePath: "new.txt"},
		},
		semantic.Removal{},
		func(progress semantic.Progress) { targetProgress = progress },
		map[string][]float32{},
		semantic.CodeColumns(),
	); err != nil {
		t.Fatalf("stage target dimension row: %v", err)
	}
	if err := targetService.PromoteStaging(context.Background(), targetPath); err != nil {
		t.Fatalf("promote target dimension row: %v", err)
	}
	if initialCatalogName == targetCatalogName {
		t.Fatal("initial and target dimensions share a reuse catalog")
	}
	for _, collectionName := range []string{initialCatalogName, targetCatalogName} {
		exists, existsErr := harness.milvus.HasCollection(
			context.Background(),
			milvusclient.NewHasCollectionOption(collectionName),
		)
		if existsErr != nil {
			t.Fatalf("check reuse catalog %s: %v", collectionName, existsErr)
		}
		if !exists {
			t.Fatalf("reuse catalog %s does not exist", collectionName)
		}
	}
	configuredCatalogExists, err := harness.milvus.HasCollection(
		context.Background(),
		milvusclient.NewHasCollectionOption(configuredCatalogName),
	)
	if err != nil {
		t.Fatalf("check configured-zero reuse catalog: %v", err)
	}
	if configuredCatalogExists {
		t.Fatalf("configured-zero reuse catalog %s exists", configuredCatalogName)
	}
	if targetProgress.ChunksEmbedded != 2 || targetProgress.ChunksReused != 0 {
		t.Fatalf(
			"target embedded/reused = %d/%d, want 2/0",
			targetProgress.ChunksEmbedded,
			targetProgress.ChunksReused,
		)
	}

	targetCollectionName := targetService.CollectionName(targetPath)
	targetRows := snapshotsForContent(t, harness, targetCollectionName, sharedContent)
	if len(targetRows) != 1 {
		t.Fatalf("target dimension rows = %d, want 1", len(targetRows))
	}
	result, err := harness.milvus.Query(
		context.Background(),
		milvusclient.NewQueryOption(targetCollectionName).
			WithFilter(fmt.Sprintf(`content == "%s"`, sharedContent)).
			WithOutputFields("vector").
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		t.Fatalf("query target dimension vector: %v", err)
	}
	vectorColumn := result.GetColumn("vector")
	if vectorColumn == nil || result.ResultCount != 1 {
		t.Fatalf("target dimension vector rows = %d, want 1", result.ResultCount)
	}
	vectorValue, err := vectorColumn.Get(0)
	if err != nil {
		t.Fatalf("read target dimension vector: %v", err)
	}
	vector, ok := vectorValue.(entity.FloatVector)
	if !ok {
		t.Fatalf("target dimension vector has type %T", vectorValue)
	}
	if len(vector) != targetDimension {
		t.Fatalf("target vector dimension = %d, want %d", len(vector), targetDimension)
	}
}

type legacyReuseFixtureRow struct {
	id            string
	content       string
	relativePath  string
	startLine     int64
	endLine       int64
	fileExtension string
	metadata      string
	vector        []float32
}

func markedVector(dimension int, first float32, last float32) []float32 {
	vector := make([]float32, dimension)
	vector[0] = first
	vector[len(vector)-1] = last
	return vector
}

func makeDuplicateLegacyRows(
	duplicateCount int,
	content string,
	vector []float32,
	controls []legacyReuseFixtureRow,
) []legacyReuseFixtureRow {
	rows := make([]legacyReuseFixtureRow, 0, duplicateCount+len(controls))
	for index := range duplicateCount {
		rows = append(rows, legacyReuseFixtureRow{
			id:            fmt.Sprintf("legacy-duplicate-%05d", index),
			content:       content,
			relativePath:  fmt.Sprintf("legacy/duplicate/%05d", index),
			startLine:     int64(index),
			endLine:       int64(index + 1),
			fileExtension: "txt",
			metadata:      fmt.Sprintf(`{"duplicate":%d}`, index),
			vector:        vector,
		})
	}
	return append(rows, controls...)
}

func insertLegacyRows(
	t *testing.T,
	harness *harness,
	collectionName string,
	rows []legacyReuseFixtureRow,
) {
	t.Helper()
	const insertBatchSize = 128
	for batchStart := 0; batchStart < len(rows); batchStart += insertBatchSize {
		batchEnd := min(batchStart+insertBatchSize, len(rows))
		batch := rows[batchStart:batchEnd]
		ids := make([]string, 0, len(batch))
		contents := make([]string, 0, len(batch))
		relativePaths := make([]string, 0, len(batch))
		startLines := make([]int64, 0, len(batch))
		endLines := make([]int64, 0, len(batch))
		fileExtensions := make([]string, 0, len(batch))
		metadata := make([]string, 0, len(batch))
		vectors := make([][]float32, 0, len(batch))
		for _, row := range batch {
			ids = append(ids, row.id)
			contents = append(contents, row.content)
			relativePaths = append(relativePaths, row.relativePath)
			startLines = append(startLines, row.startLine)
			endLines = append(endLines, row.endLine)
			fileExtensions = append(fileExtensions, row.fileExtension)
			metadata = append(metadata, row.metadata)
			vectors = append(vectors, row.vector)
		}
		result, err := harness.milvus.Insert(
			context.Background(),
			milvusclient.NewColumnBasedInsertOption(collectionName).
				WithVarcharColumn("id", ids).
				WithVarcharColumn("content", contents).
				WithVarcharColumn("relativePath", relativePaths).
				WithInt64Column("startLine", startLines).
				WithInt64Column("endLine", endLines).
				WithVarcharColumn("fileExtension", fileExtensions).
				WithVarcharColumn("metadata", metadata).
				WithFloatVectorColumn("vector", len(batch[0].vector), vectors),
		)
		if err != nil {
			t.Fatalf("insert legacy rows %d:%d: %v", batchStart, batchEnd, err)
		}
		if result.InsertCount != int64(len(batch)) {
			t.Fatalf(
				"insert legacy rows %d:%d count = %d, want %d",
				batchStart,
				batchEnd,
				result.InsertCount,
				len(batch),
			)
		}
	}
	flushTask, err := harness.milvus.Flush(
		context.Background(),
		milvusclient.NewFlushOption(collectionName),
	)
	if err != nil {
		t.Fatalf("flush duplicate legacy rows: %v", err)
	}
	if err := flushTask.Await(context.Background()); err != nil {
		t.Fatalf("await duplicate legacy row flush: %v", err)
	}
}

func insertLegacyRow(
	t *testing.T,
	harness *harness,
	content string,
	vector []float32,
) string {
	t.Helper()
	rowID := "legacy-" + randomID()
	result, err := harness.milvus.Insert(
		context.Background(),
		milvusclient.NewColumnBasedInsertOption(harness.collectionName).
			WithVarcharColumn("id", []string{rowID}).
			WithVarcharColumn("content", []string{content}).
			WithVarcharColumn("relativePath", []string{"conv/legacy/0/0"}).
			WithInt64Column("startLine", []int64{0}).
			WithInt64Column("endLine", []int64{0}).
			WithVarcharColumn("fileExtension", []string{"txt"}).
			WithVarcharColumn("metadata", []string{"{}"}).
			WithFloatVectorColumn("vector", len(vector), [][]float32{vector}),
	)
	if err != nil {
		t.Fatalf("insert legacy row: %v", err)
	}
	if result.InsertCount != 1 {
		t.Fatalf("insert legacy row count = %d, want 1", result.InsertCount)
	}
	flushTask, err := harness.milvus.Flush(
		context.Background(),
		milvusclient.NewFlushOption(harness.collectionName),
	)
	if err != nil {
		t.Fatalf("flush legacy row: %v", err)
	}
	if err := flushTask.Await(context.Background()); err != nil {
		t.Fatalf("await legacy row flush: %v", err)
	}
	return rowID
}

type storedRowSnapshot struct {
	id                  string
	content             string
	relativePath        string
	startLine           int64
	endLine             int64
	fileExtension       string
	metadata            string
	contentHash         string
	contentHashKnown    bool
	embeddingModel      string
	embeddingModelKnown bool
	vectorChecksum      string
}

func compareStoredRowSnapshots(left storedRowSnapshot, right storedRowSnapshot) int {
	return strings.Compare(left.id, right.id)
}

func collectionRowCount(t *testing.T, harness *harness, collectionName string) int64 {
	t.Helper()
	result, err := harness.milvus.Query(
		context.Background(),
		milvusclient.NewQueryOption(collectionName).
			WithOutputFields(countOutputField).
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		t.Fatalf("count rows in %s: %v", collectionName, err)
	}
	countColumn := result.GetColumn(countOutputField)
	if countColumn == nil {
		t.Fatalf("row count query for %s returned no count column", collectionName)
	}
	count, err := countColumn.GetAsInt64(0)
	if err != nil {
		t.Fatalf("read row count for %s: %v", collectionName, err)
	}
	return count
}

func snapshotsForIDs(
	t *testing.T,
	harness *harness,
	collectionName string,
	ids []string,
) []storedRowSnapshot {
	t.Helper()
	const snapshotBatchSize = 64
	rows := make([]storedRowSnapshot, 0, len(ids))
	for batchStart := 0; batchStart < len(ids); batchStart += snapshotBatchSize {
		batchEnd := min(batchStart+snapshotBatchSize, len(ids))
		quotedIDs := make([]string, 0, batchEnd-batchStart)
		for _, id := range ids[batchStart:batchEnd] {
			quotedIDs = append(quotedIDs, `"`+strings.ReplaceAll(id, `"`, `\"`)+`"`)
		}
		rows = append(rows, queryRowSnapshots(
			t,
			harness,
			collectionName,
			"id in ["+strings.Join(quotedIDs, ",")+"]",
		)...)
	}
	slices.SortFunc(rows, compareStoredRowSnapshots)
	return rows
}

func snapshotRow(
	t *testing.T,
	harness *harness,
	collectionName string,
	rowID string,
) storedRowSnapshot {
	t.Helper()
	rows := queryRowSnapshots(
		t,
		harness,
		collectionName,
		fmt.Sprintf(`id == "%s"`, rowID),
	)
	if len(rows) != 1 {
		t.Fatalf("row %s snapshots = %d, want 1", rowID, len(rows))
	}
	return rows[0]
}

func snapshotsForContent(
	t *testing.T,
	harness *harness,
	collectionName string,
	content string,
) []storedRowSnapshot {
	t.Helper()
	return queryRowSnapshots(
		t,
		harness,
		collectionName,
		fmt.Sprintf(`content == "%s"`, strings.ReplaceAll(content, `"`, `\"`)),
	)
}

func queryRowSnapshots(
	t *testing.T,
	harness *harness,
	collectionName string,
	filter string,
) []storedRowSnapshot {
	t.Helper()
	result, err := harness.milvus.Query(
		context.Background(),
		milvusclient.NewQueryOption(collectionName).
			WithFilter(filter).
			WithOutputFields(
				"id",
				"content",
				"relativePath",
				"startLine",
				"endLine",
				"fileExtension",
				"metadata",
				"contentHash",
				"embeddingModel",
				"vector",
			).
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		t.Fatalf("query row snapshots from %s: %v", collectionName, err)
	}
	idColumn := result.GetColumn("id")
	contentColumn := result.GetColumn("content")
	pathColumn := result.GetColumn("relativePath")
	startLineColumn := result.GetColumn("startLine")
	endLineColumn := result.GetColumn("endLine")
	fileExtensionColumn := result.GetColumn("fileExtension")
	metadataColumn := result.GetColumn("metadata")
	hashColumn := result.GetColumn("contentHash")
	modelColumn := result.GetColumn("embeddingModel")
	vectorColumn := result.GetColumn("vector")
	if idColumn == nil || contentColumn == nil || pathColumn == nil ||
		startLineColumn == nil || endLineColumn == nil || fileExtensionColumn == nil ||
		metadataColumn == nil || vectorColumn == nil {
		t.Fatal("row snapshot query omitted a required column")
	}
	rows := make([]storedRowSnapshot, 0, result.ResultCount)
	for rowIndex := range result.ResultCount {
		rowID, idErr := idColumn.GetAsString(rowIndex)
		if idErr != nil {
			t.Fatalf("read row id at %d: %v", rowIndex, idErr)
		}
		content, contentErr := contentColumn.GetAsString(rowIndex)
		if contentErr != nil {
			t.Fatalf("read row content at %d: %v", rowIndex, contentErr)
		}
		relativePath, pathErr := pathColumn.GetAsString(rowIndex)
		if pathErr != nil {
			t.Fatalf("read row path at %d: %v", rowIndex, pathErr)
		}
		startLine, startLineErr := startLineColumn.GetAsInt64(rowIndex)
		if startLineErr != nil {
			t.Fatalf("read row start line at %d: %v", rowIndex, startLineErr)
		}
		endLine, endLineErr := endLineColumn.GetAsInt64(rowIndex)
		if endLineErr != nil {
			t.Fatalf("read row end line at %d: %v", rowIndex, endLineErr)
		}
		fileExtension, fileExtensionErr := fileExtensionColumn.GetAsString(rowIndex)
		if fileExtensionErr != nil {
			t.Fatalf("read row file extension at %d: %v", rowIndex, fileExtensionErr)
		}
		metadata, metadataErr := metadataColumn.GetAsString(rowIndex)
		if metadataErr != nil {
			t.Fatalf("read row metadata at %d: %v", rowIndex, metadataErr)
		}
		contentHash, contentHashKnown := nullableSnapshotString(t, hashColumn, rowIndex)
		embeddingModel, embeddingModelKnown := nullableSnapshotString(t, modelColumn, rowIndex)
		vectorValue, vectorErr := vectorColumn.Get(rowIndex)
		if vectorErr != nil {
			t.Fatalf("read row vector at %d: %v", rowIndex, vectorErr)
		}
		vector, ok := vectorValue.(entity.FloatVector)
		if !ok {
			t.Fatalf("row vector at %d has type %T", rowIndex, vectorValue)
		}
		rows = append(rows, storedRowSnapshot{
			id:                  rowID,
			content:             content,
			relativePath:        relativePath,
			startLine:           startLine,
			endLine:             endLine,
			fileExtension:       fileExtension,
			metadata:            metadata,
			contentHash:         contentHash,
			contentHashKnown:    contentHashKnown,
			embeddingModel:      embeddingModel,
			embeddingModelKnown: embeddingModelKnown,
			vectorChecksum:      checksumVector(vector),
		})
	}
	slices.SortFunc(rows, compareStoredRowSnapshots)
	return rows
}

type reuseCatalogSnapshot struct {
	catalogKey          string
	contentHash         string
	embeddingModel      string
	embeddingModelKnown bool
	vectorChecksum      string
}

func reuseCatalogSnapshots(
	t *testing.T,
	harness *harness,
	collectionName string,
) []reuseCatalogSnapshot {
	t.Helper()
	result, err := harness.milvus.Query(
		context.Background(),
		milvusclient.NewQueryOption(collectionName).
			WithFilter(`catalogKey != ""`).
			WithOutputFields("catalogKey", "contentHash", "embeddingModel", "vector").
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		t.Fatalf("snapshot reuse catalog %s: %v", collectionName, err)
	}
	catalogKeyColumn := result.GetColumn("catalogKey")
	contentHashColumn := result.GetColumn("contentHash")
	embeddingModelColumn := result.GetColumn("embeddingModel")
	vectorColumn := result.GetColumn("vector")
	if catalogKeyColumn == nil || contentHashColumn == nil || vectorColumn == nil {
		t.Fatal("reuse catalog snapshot query omitted a required column")
	}
	rows := make([]reuseCatalogSnapshot, 0, result.ResultCount)
	for rowIndex := range result.ResultCount {
		catalogKey, catalogKeyErr := catalogKeyColumn.GetAsString(rowIndex)
		if catalogKeyErr != nil {
			t.Fatalf("read reuse catalog key at %d: %v", rowIndex, catalogKeyErr)
		}
		contentHash, contentHashErr := contentHashColumn.GetAsString(rowIndex)
		if contentHashErr != nil {
			t.Fatalf("read reuse catalog hash at %d: %v", rowIndex, contentHashErr)
		}
		embeddingModel, embeddingModelKnown := nullableSnapshotString(
			t,
			embeddingModelColumn,
			rowIndex,
		)
		vectorValue, vectorErr := vectorColumn.Get(rowIndex)
		if vectorErr != nil {
			t.Fatalf("read reuse catalog vector at %d: %v", rowIndex, vectorErr)
		}
		vector, ok := vectorValue.(entity.FloatVector)
		if !ok {
			t.Fatalf("reuse catalog vector at %d has type %T", rowIndex, vectorValue)
		}
		rows = append(rows, reuseCatalogSnapshot{
			catalogKey:          catalogKey,
			contentHash:         contentHash,
			embeddingModel:      embeddingModel,
			embeddingModelKnown: embeddingModelKnown,
			vectorChecksum:      checksumVector(vector),
		})
	}
	slices.SortFunc(rows, func(left reuseCatalogSnapshot, right reuseCatalogSnapshot) int {
		return strings.Compare(left.catalogKey, right.catalogKey)
	})
	return rows
}

func assertNoMilvusWrites(t *testing.T, calls []milvusCall) {
	t.Helper()
	for _, call := range calls {
		if !protectedMilvusCall(call.method) {
			continue
		}
		if call.method == "LoadCollection" || call.method == "ReleaseCollection" {
			continue
		}
		t.Fatalf(
			"reuse lookup issued Milvus write %s against %v from %s",
			call.method,
			call.collectionNames,
			call.caller,
		)
	}
}

func nullableSnapshotString(
	t *testing.T,
	field interface {
		IsNull(int) (bool, error)
		GetAsString(int) (string, error)
	},
	rowIndex int,
) (string, bool) {
	t.Helper()
	if field == nil {
		return "", false
	}
	isNull, err := field.IsNull(rowIndex)
	if err != nil {
		t.Fatalf("read nullable marker at %d: %v", rowIndex, err)
	}
	if isNull {
		return "", false
	}
	value, err := field.GetAsString(rowIndex)
	if err != nil {
		t.Fatalf("read nullable string at %d: %v", rowIndex, err)
	}
	return value, true
}

func checksumVector(vector entity.FloatVector) string {
	hash := sha256.New()
	buffer := make([]byte, 4)
	for _, value := range vector {
		binary.LittleEndian.PutUint32(buffer, math.Float32bits(value))
		_, _ = hash.Write(buffer)
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func reuseCatalogRowCount(t *testing.T, harness *harness) int64 {
	t.Helper()
	result, err := harness.milvus.Query(
		context.Background(),
		milvusclient.NewQueryOption(harness.reuseCatalogName).
			WithOutputFields(countOutputField).
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		t.Fatalf("count reuse catalog rows: %v", err)
	}
	countColumn := result.GetColumn(countOutputField)
	if countColumn == nil {
		t.Fatal("reuse catalog count query returned no count column")
	}
	count, err := countColumn.GetAsInt64(0)
	if err != nil {
		t.Fatalf("read reuse catalog count: %v", err)
	}
	return count
}

func reuseCatalogModels(t *testing.T, harness *harness, content string) []string {
	t.Helper()
	contentHash := semantic.ContentVectorKey(content)
	result, err := harness.milvus.Query(
		context.Background(),
		milvusclient.NewQueryOption(harness.reuseCatalogName).
			WithFilter(fmt.Sprintf(`contentHash == "%s"`, contentHash)).
			WithOutputFields("embeddingModel").
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		t.Fatalf("query reuse catalog models: %v", err)
	}
	modelColumn := result.GetColumn("embeddingModel")
	if modelColumn == nil && result.ResultCount > 0 {
		t.Fatal("reuse catalog model query returned no model column")
	}
	models := make([]string, 0, result.ResultCount)
	for rowIndex := range result.ResultCount {
		modelName, known := nullableSnapshotString(t, modelColumn, rowIndex)
		if !known {
			models = append(models, "")
			continue
		}
		models = append(models, modelName)
	}
	slices.Sort(models)
	return models
}

func insertEmptyModelCatalogRow(t *testing.T, harness *harness, content string) {
	t.Helper()
	contentHash := semantic.ContentVectorKey(content)
	rowKeySum := sha256.Sum256([]byte(contentHash + "\x00"))
	rowKey := hex.EncodeToString(rowKeySum[:])
	embeddingModelColumn, err := column.NewNullableColumnVarChar(
		"embeddingModel",
		[]string{""},
		[]bool{false},
		column.WithSparseNullableMode[string](true),
	)
	if err != nil {
		t.Fatalf("build empty model catalog column: %v", err)
	}
	vector := make([]float32, fakeEmbeddingDimension)
	vector[0] = 1
	result, err := harness.milvus.Insert(
		context.Background(),
		milvusclient.NewColumnBasedInsertOption(harness.reuseCatalogName).
			WithVarcharColumn("catalogKey", []string{rowKey}).
			WithVarcharColumn("contentHash", []string{contentHash}).
			WithColumns(embeddingModelColumn).
			WithFloatVectorColumn("vector", len(vector), [][]float32{vector}),
	)
	if err != nil {
		t.Fatalf("insert empty model catalog row: %v", err)
	}
	if result.InsertCount != 1 {
		t.Fatalf("empty model catalog insert count = %d, want 1", result.InsertCount)
	}
	flushTask, err := harness.milvus.Flush(
		context.Background(),
		milvusclient.NewFlushOption(harness.reuseCatalogName),
	)
	if err != nil {
		t.Fatalf("flush empty model catalog row: %v", err)
	}
	if err := flushTask.Await(context.Background()); err != nil {
		t.Fatalf("await empty model catalog flush: %v", err)
	}
}

func dropLiveCollection(t *testing.T, harness *harness, collectionName string) {
	t.Helper()
	dropCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := harness.milvus.DropCollection(
		dropCtx,
		milvusclient.NewDropCollectionOption(collectionName),
	); err != nil &&
		!strings.Contains(err.Error(), "not exist") &&
		!strings.Contains(err.Error(), "not found") {
		t.Errorf("DropCollection(%s) returned error: %v", collectionName, err)
	}
}

func durationPercentile(values []time.Duration, percentile int) time.Duration {
	sorted := append([]time.Duration(nil), values...)
	slices.Sort(sorted)
	if percentile <= 0 {
		return sorted[0]
	}
	index := (len(sorted)*percentile+99)/100 - 1
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return sorted[index]
}

func durationMilliseconds(duration time.Duration) float64 {
	return float64(duration) / float64(time.Millisecond)
}

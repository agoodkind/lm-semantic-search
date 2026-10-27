//go:build live

package live

import (
	"cmp"
	"context"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

const (
	copyBatchPathCount      = 3
	copyBatchRowsPerPath    = 2
	copyBatchCountTimeout   = 30 * time.Second
	copyBatchInsertMethod   = "Insert"
	copyBatchFlushMethod    = "Flush"
	copyBatchDeleteMethod   = "Delete"
	copyBatchSourceFormat   = "batch/source%d.txt"
	copyBatchDestFormat     = "batch/destination%d.txt"
	copyBatchContentFormat  = "batched copy content for path %d part %d %s"
	copyBatchCodebasePrefix = "copy-chunks-batch-"
	copyBatchContentField   = "content"
	copyBatchStartLineField = "startLine"
	copyBatchEndLineField   = "endLine"
)

type copyBatchRow struct {
	content   string
	startLine int64
	endLine   int64
}

func TestCopyChunksBatchFlushesOnceBeforeOneSourceDelete(t *testing.T) {
	harness := newONNXHarness(t)
	serviceConfig := harness.childConfig()
	service, err := semantic.NewService(harness.milvusContext, serviceConfig)
	if err != nil {
		t.Fatalf("Expected semantic.NewService to succeed, but it returned an error: %v.", err)
	}
	t.Cleanup(func() { _ = service.Close(context.Background()) })

	codebasePath := filepath.Join(harness.stateRoot, copyBatchCodebasePrefix+randomID())
	collectionName := service.CollectionName(codebasePath)
	harness.trackCollectionFamily(collectionName)
	harness.trackTemporaryCollection(semantic.ReuseCatalogCollectionName(serviceConfig))

	runID := randomID()
	copies := make([]semantic.ChunkCopy, 0, copyBatchPathCount)
	chunks := make([]model.StoredChunk, 0, copyBatchPathCount*copyBatchRowsPerPath)
	stagedRows := make(map[string][]copyBatchRow, copyBatchPathCount)
	contentSources := make(map[string]string, copyBatchPathCount*copyBatchRowsPerPath)
	for pathIndex := range copyBatchPathCount {
		chunkCopy := semantic.ChunkCopy{
			Source:      fmt.Sprintf(copyBatchSourceFormat, pathIndex),
			Destination: fmt.Sprintf(copyBatchDestFormat, pathIndex),
		}
		copies = append(copies, chunkCopy)
		for partIndex := range int32(copyBatchRowsPerPath) {
			chunk := model.StoredChunk{
				Content:      fmt.Sprintf(copyBatchContentFormat, pathIndex, partIndex, runID),
				RelativePath: chunkCopy.Source,
				StartLine:    partIndex + 1,
				EndLine:      partIndex + 1,
			}
			chunks = append(chunks, chunk)
			stagedRows[chunkCopy.Source] = append(stagedRows[chunkCopy.Source], copyBatchRow{
				content:   chunk.Content,
				startLine: int64(chunk.StartLine),
				endLine:   int64(chunk.EndLine),
			})
			contentSources[chunk.Content] = chunkCopy.Source
		}
		slices.SortFunc(stagedRows[chunkCopy.Source], compareCopyBatchRows)
	}
	if err := service.StageReindex(
		context.Background(),
		codebasePath,
		chunks,
		semantic.Removal{},
		nil,
		map[string]model.ReuseVector{},
		semantic.CodeColumns(),
	); err != nil {
		t.Fatalf("Expected StageReindex to stage source rows for %s, but it returned an error: %v.", codebasePath, err)
	}
	if err := service.PromoteStaging(context.Background(), codebasePath); err != nil {
		t.Fatalf("Expected PromoteStaging to promote source rows for %s, but it returned an error: %v.", codebasePath, err)
	}
	sourceRowCounts := make([]int64, 0, len(copies))
	for _, chunkCopy := range copies {
		sourceRows := copyBatchPathRowCount(t, harness, collectionName, chunkCopy.Source)
		if sourceRows == 0 {
			t.Fatalf("Expected source path %s to have rows in collection %s before the copy, but the count query returned zero.", chunkCopy.Source, collectionName)
		}
		sourceRowCounts = append(sourceRowCounts, sourceRows)
	}

	harness.callRecorder.reset()
	copiedRowCounts, err := service.CopyChunks(context.Background(), codebasePath, copies)
	if err != nil {
		t.Fatalf("Expected CopyChunks to copy %d paths in collection %s, but it returned an error: %v.", len(copies), collectionName, err)
	}
	calls := harness.callRecorder.snapshot()

	wantCopiedRowCounts := make([]int, 0, len(sourceRowCounts))
	for _, sourceRows := range sourceRowCounts {
		wantCopiedRowCounts = append(wantCopiedRowCounts, int(sourceRows))
	}
	if !slices.Equal(copiedRowCounts, wantCopiedRowCounts) {
		t.Fatalf("CopyChunks returned row counts %v; the test expected %v.", copiedRowCounts, wantCopiedRowCounts)
	}

	mutations := copyBatchMutationMethods(calls, collectionName)
	flushPositions := copyBatchMethodPositions(mutations, copyBatchFlushMethod)
	deletePositions := copyBatchMethodPositions(mutations, copyBatchDeleteMethod)
	insertPositions := copyBatchMethodPositions(mutations, copyBatchInsertMethod)
	if len(flushPositions) != 1 {
		t.Fatalf("Expected one Flush call on collection %s, but the copy made %d Flush calls. The ordered Insert, Flush, and Delete calls were %v.", collectionName, len(flushPositions), mutations)
	}
	if len(deletePositions) != 1 {
		t.Fatalf("Expected one Delete call on collection %s, but the copy made %d Delete calls. The ordered calls were %v.", collectionName, len(deletePositions), mutations)
	}
	if len(insertPositions) == 0 {
		t.Fatalf("Expected at least one Insert call on collection %s, but the copy made none. The ordered calls were %v.", collectionName, mutations)
	}
	flushPosition := flushPositions[0]
	deletePosition := deletePositions[0]
	for _, insertPosition := range insertPositions {
		if insertPosition > flushPosition {
			t.Fatalf("Insert at position %d occurred after Flush at position %d; the test expected every Insert before Flush. The ordered calls were %v.", insertPosition, flushPosition, mutations)
		}
	}
	if deletePosition < flushPosition {
		t.Fatalf("Delete at position %d occurred before Flush at position %d; the test expected Delete after Flush. The ordered calls were %v.", deletePosition, flushPosition, mutations)
	}

	for index, chunkCopy := range copies {
		destinationRows := copyBatchPathRowCount(t, harness, collectionName, chunkCopy.Destination)
		if destinationRows != sourceRowCounts[index] {
			t.Fatalf("Destination path %s has %d rows; the test expected %d rows to match its source before the copy.", chunkCopy.Destination, destinationRows, sourceRowCounts[index])
		}
		remainingSourceRows := copyBatchPathRowCount(t, harness, collectionName, chunkCopy.Source)
		if remainingSourceRows != 0 {
			t.Fatalf("Expected source path %s to have no rows after the copy, but the count query returned %d rows.", chunkCopy.Source, remainingSourceRows)
		}
		copiedRows := copyBatchPathRows(t, harness, collectionName, chunkCopy.Destination)
		for _, copiedRow := range copiedRows {
			stagedSource := contentSources[copiedRow.content]
			if stagedSource != chunkCopy.Source {
				t.Fatalf("Destination path %s contains unexpected content %q. The matching staged source path is %q, where an empty string means the test did not stage this content. The test expected content staged for source path %s.", chunkCopy.Destination, copiedRow.content, stagedSource, chunkCopy.Source)
			}
		}
		if !slices.Equal(copiedRows, stagedRows[chunkCopy.Source]) {
			t.Fatalf("Destination path %s has rows %v, but the test expected the rows staged for source path %s: %v.", chunkCopy.Destination, copiedRows, chunkCopy.Source, stagedRows[chunkCopy.Source])
		}
	}
}

func compareCopyBatchRows(left copyBatchRow, right copyBatchRow) int {
	return cmp.Or(
		strings.Compare(left.content, right.content),
		cmp.Compare(left.startLine, right.startLine),
		cmp.Compare(left.endLine, right.endLine),
	)
}

func copyBatchPathRows(t *testing.T, harness *harness, collectionName string, relativePath string) []copyBatchRow {
	t.Helper()
	queryContext, cancel := context.WithTimeout(context.Background(), copyBatchCountTimeout)
	defer cancel()
	filter := fmt.Sprintf("%s == %q", relativePathField, relativePath)
	result, err := harness.milvus.Query(
		queryContext,
		milvusclient.NewQueryOption(collectionName).
			WithFilter(filter).
			WithOutputFields(copyBatchContentField, copyBatchStartLineField, copyBatchEndLineField).
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		t.Fatalf("The test expected the row query for collection %s with filter %q to succeed, but the query returned an error: %v.", collectionName, filter, err)
	}
	contentColumn := result.GetColumn(copyBatchContentField)
	startLineColumn := result.GetColumn(copyBatchStartLineField)
	endLineColumn := result.GetColumn(copyBatchEndLineField)
	for _, field := range []string{copyBatchContentField, copyBatchStartLineField, copyBatchEndLineField} {
		if result.GetColumn(field) == nil {
			t.Fatalf("The test expected field %s in the row query result for collection %s with filter %q, but the result does not contain that field.", field, collectionName, filter)
		}
	}
	rows := make([]copyBatchRow, 0, result.ResultCount)
	for rowIndex := range result.ResultCount {
		content, contentErr := contentColumn.GetAsString(rowIndex)
		if contentErr != nil {
			t.Fatalf("The test expected to read field %s from result row %d for collection %s with filter %q, but reading the value returned an error: %v.", copyBatchContentField, rowIndex, collectionName, filter, contentErr)
		}
		startLine, startLineErr := startLineColumn.GetAsInt64(rowIndex)
		if startLineErr != nil {
			t.Fatalf("The test expected to read field %s from result row %d for collection %s with filter %q, but reading the value returned an error: %v.", copyBatchStartLineField, rowIndex, collectionName, filter, startLineErr)
		}
		endLine, endLineErr := endLineColumn.GetAsInt64(rowIndex)
		if endLineErr != nil {
			t.Fatalf("The test expected to read field %s from result row %d for collection %s with filter %q, but reading the value returned an error: %v.", copyBatchEndLineField, rowIndex, collectionName, filter, endLineErr)
		}
		rows = append(rows, copyBatchRow{content: content, startLine: startLine, endLine: endLine})
	}
	slices.SortFunc(rows, compareCopyBatchRows)
	return rows
}

func copyBatchMutationMethods(calls []milvusCall, collectionName string) []string {
	methods := make([]string, 0, len(calls))
	for _, call := range calls {
		if !slices.Contains(call.collectionNames, collectionName) {
			continue
		}
		switch call.method {
		case copyBatchInsertMethod, copyBatchFlushMethod, copyBatchDeleteMethod:
			methods = append(methods, call.method)
		}
	}
	return methods
}

func copyBatchMethodPositions(methods []string, method string) []int {
	positions := make([]int, 0, len(methods))
	for position, recorded := range methods {
		if recorded == method {
			positions = append(positions, position)
		}
	}
	return positions
}

func copyBatchPathRowCount(t *testing.T, harness *harness, collectionName string, relativePath string) int64 {
	t.Helper()
	queryContext, cancel := context.WithTimeout(context.Background(), copyBatchCountTimeout)
	defer cancel()
	filter := fmt.Sprintf("%s == %q", relativePathField, relativePath)
	result, err := harness.milvus.Query(
		queryContext,
		milvusclient.NewQueryOption(collectionName).
			WithFilter(filter).
			WithOutputFields(countOutputField).
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		t.Fatalf("Expected the count query with Strong consistency on collection %s with filter %q to succeed, but it returned an error: %v.", collectionName, filter, err)
	}
	countColumn := result.GetColumn(countOutputField)
	if countColumn == nil {
		t.Fatalf("Expected the count query on collection %s with filter %q to return a count(*) column, but the result has no such column.", collectionName, filter)
	}
	count, err := countColumn.GetAsInt64(0)
	if err != nil {
		t.Fatalf("Expected to read count(*) from collection %s with filter %q, but reading the value returned an error: %v.", collectionName, filter, err)
	}
	return count
}

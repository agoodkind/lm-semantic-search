package semantic

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log/slog"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/model"
)

// CopyChunks rewrites the relativePath column on every existing chunk row
// for srcRelativePath so the same vectors are addressable under
// dstRelativePath. Used by the converge decision table to handle a rename
// or hardlink without re-embedding.
//
// Milvus does not support an in-place column update on a primary-keyed
// row (the primary key in this schema is derived from relativePath), so
// CopyChunks queries the source rows, computes new IDs and rows for the
// destination path, inserts and flushes the new rows, then deletes and flushes
// the source rows. A failure after the destination flush keeps the destination
// durable even if the source deletion has an uncertain result. CopyChunks
// returns the failure without retrying, leaving later convergence to remove
// any remaining source rows.
// The dense vector is preserved across the copy; the sparse vector, when
// the collection is hybrid, is re-derived by the BM25 function from the
// preserved content so no embedding API call is issued.
func (service *Service) CopyChunks(ctx context.Context, codebasePath string, srcRelativePath string, dstRelativePath string) (int, error) {
	if !service.Available() {
		return 0, ErrUnavailable
	}
	if srcRelativePath == dstRelativePath {
		return 0, nil
	}
	collectionName := service.CollectionName(codebasePath)
	hasCollection, err := service.hasCollection(ctx, collectionName, "check Milvus collection "+collectionName)
	if err != nil {
		return 0, err
	}
	if !hasCollection {
		return 0, ErrCollectionMissing
	}
	if err := service.PrepareCollection(ctx, collectionName); err != nil {
		return 0, err
	}
	lease, err := service.AcquireCollection(ctx, collectionName)
	if err != nil {
		return 0, err
	}
	defer lease.Release()

	source, err := service.fetchChunksForPath(ctx, collectionName, srcRelativePath)
	if err != nil {
		return 0, err
	}
	if len(source.chunks) == 0 {
		return 0, nil
	}

	rewritten, destinationIDs, splitPartsRecorded := rewriteCopiedRows(
		source,
		dstRelativePath,
	)

	mutations := copyChunkMutations{
		insertDestination: func() error {
			// CopyChunks rewrites existing rows within one known collection and
			// has no item source to ask, so it classifies the column set from
			// that collection.
			return service.insertBatchWithIDs(
				ctx,
				collectionName,
				rewritten,
				destinationIDs,
				source.vectors,
				splitPartsRecorded,
				storeColumnSetForCollection(collectionName),
			)
		},
		persistDestination: func() error {
			return service.flushCollection(ctx, collectionName)
		},
		deleteSource: func() error {
			// The deleted-row count is not part of the copy's result, which
			// reports the rows written at the destination.
			_, deleteErr := service.deleteByRelativePaths(
				ctx,
				collectionName,
				[]string{srcRelativePath},
			)
			return deleteErr
		},
		persistSourceDelete: func() error {
			return service.flushCollection(ctx, collectionName)
		},
	}
	if err := runCopyChunkMutations(mutations); err != nil {
		return 0, err
	}
	slog.InfoContext(ctx, "semantic.copy_chunks", "collection", collectionName, "src", srcRelativePath, "dst", dstRelativePath, "rows", len(rewritten))
	return len(rewritten), nil
}

type copyChunkMutation func() error

type copyChunkMutations struct {
	insertDestination   copyChunkMutation
	persistDestination  copyChunkMutation
	deleteSource        copyChunkMutation
	persistSourceDelete copyChunkMutation
}

func runCopyChunkMutations(mutations copyChunkMutations) error {
	if err := mutations.insertDestination(); err != nil {
		return err
	}
	if err := mutations.persistDestination(); err != nil {
		return err
	}
	if err := mutations.deleteSource(); err != nil {
		return err
	}
	if err := mutations.persistSourceDelete(); err != nil {
		return err
	}
	return nil
}

func (service *Service) flushCollection(ctx context.Context, collectionName string) error {
	task, err := service.milvus.Flush(
		ctx,
		milvusclient.NewFlushOption(collectionName),
	)
	if err != nil {
		return wrapStoreError(ctx, err, "flush Milvus collection "+collectionName)
	}
	if err := task.Await(ctx); err != nil {
		return wrapStoreError(ctx, err, "await Milvus collection flush "+collectionName)
	}
	return nil
}

func copiedChunkID(sourceID string, dstRelativePath string) string {
	sum := sha256.Sum256([]byte(dstRelativePath + ":" + sourceID))
	return "chunk_" + hex.EncodeToString(sum[:])[:16]
}

type copiedRows struct {
	chunks             []model.StoredChunk
	ids                []string
	vectors            [][]float32
	splitPartsRecorded []bool
}

type copiedRowColumns struct {
	id            column.Column
	content       column.Column
	startLine     column.Column
	endLine       column.Column
	fileExtension column.Column
	metadata      column.Column
	vector        column.Column
	splitPart     column.Column
}

type copiedRow struct {
	chunk             model.StoredChunk
	id                string
	vector            []float32
	splitPartRecorded bool
}

func noCopiedRows() copiedRows {
	return copiedRows{
		chunks:             nil,
		ids:                nil,
		vectors:            nil,
		splitPartsRecorded: nil,
	}
}

func rewriteCopiedRows(
	source copiedRows,
	dstRelativePath string,
) ([]model.StoredChunk, []string, []bool) {
	rewritten := make([]model.StoredChunk, 0, len(source.chunks))
	destinationIDs := make([]string, 0, len(source.chunks))
	recorded := make([]bool, 0, len(source.chunks))
	for index, chunk := range source.chunks {
		splitPartRecorded := source.splitPartsRecorded[index]
		chunkCopy := chunk
		chunkCopy.RelativePath = dstRelativePath
		chunkCopy.SplitPartRecorded = splitPartRecorded
		rewritten = append(rewritten, chunkCopy)
		recorded = append(recorded, splitPartRecorded)
		if splitPartRecorded {
			destinationIDs = append(destinationIDs, generateID(chunkCopy, index))
			continue
		}
		destinationIDs = append(
			destinationIDs,
			copiedChunkID(source.ids[index], dstRelativePath),
		)
	}
	return rewritten, destinationIDs, recorded
}

// fetchChunksForPath retrieves every chunk row for relativePath including
// the dense vector so CopyChunks can reinsert under a new key without
// re-embedding the content.
func (service *Service) fetchChunksForPath(
	ctx context.Context,
	collectionName string,
	relativePath string,
) (copiedRows, error) {
	if err := service.ensureSplitPartColumnOnce(ctx, collectionName); err != nil {
		return noCopiedRows(), err
	}
	expression := relativePathExpression(relativePath)
	outputFields := []string{
		idFieldName,
		contentFieldName,
		relativePathFieldName,
		startLineFieldName,
		endLineFieldName,
		fileExtensionFieldName,
		metadataFieldName,
		denseVectorFieldName,
		splitPartFieldName,
	}
	queryOption := milvusclient.NewQueryOption(collectionName).WithFilter(expression).WithOutputFields(outputFields...)
	resultSet, err := service.milvus.Query(ctx, queryOption)
	if err != nil {
		slog.ErrorContext(ctx, "query chunks for copy failed", "collection", collectionName, "path", relativePath, "err", err)
		return noCopiedRows(), fmt.Errorf(
			"query chunks for %s in %s: %w",
			relativePath,
			collectionName,
			err,
		)
	}
	if resultSet.ResultCount == 0 {
		return noCopiedRows(), nil
	}

	columns, err := copiedColumns(resultSet)
	if err != nil {
		return noCopiedRows(), err
	}

	chunks := make([]model.StoredChunk, 0, resultSet.ResultCount)
	ids := make([]string, 0, resultSet.ResultCount)
	vectors := make([][]float32, 0, resultSet.ResultCount)
	splitPartsRecorded := make([]bool, 0, resultSet.ResultCount)
	for rowIndex := range resultSet.ResultCount {
		row, rowErr := copiedRowAt(
			ctx,
			collectionName,
			relativePath,
			columns,
			rowIndex,
		)
		if rowErr != nil {
			return noCopiedRows(), rowErr
		}
		chunks = append(chunks, row.chunk)
		ids = append(ids, row.id)
		vectors = append(vectors, row.vector)
		splitPartsRecorded = append(splitPartsRecorded, row.splitPartRecorded)
	}
	return copiedRows{
		chunks:             chunks,
		ids:                ids,
		vectors:            vectors,
		splitPartsRecorded: splitPartsRecorded,
	}, nil
}

func copiedColumns(resultSet milvusclient.ResultSet) (copiedRowColumns, error) {
	columns := copiedRowColumns{
		id:            resultSet.GetColumn(idFieldName),
		content:       resultSet.GetColumn(contentFieldName),
		startLine:     resultSet.GetColumn(startLineFieldName),
		endLine:       resultSet.GetColumn(endLineFieldName),
		fileExtension: resultSet.GetColumn(fileExtensionFieldName),
		metadata:      resultSet.GetColumn(metadataFieldName),
		vector:        resultSet.GetColumn(denseVectorFieldName),
		splitPart:     resultSet.GetColumn(splitPartFieldName),
	}
	if columns.id == nil || columns.content == nil || columns.startLine == nil ||
		columns.endLine == nil || columns.fileExtension == nil ||
		columns.vector == nil {
		return copiedRowColumns{}, ErrSearchResultIncomplete
	}
	return columns, nil
}

func copiedRowAt(
	ctx context.Context,
	collectionName string,
	relativePath string,
	columns copiedRowColumns,
	rowIndex int,
) (copiedRow, error) {
	idValue, err := columns.id.GetAsString(rowIndex)
	if err != nil {
		slog.ErrorContext(ctx, "read id column for copy failed", "collection", collectionName, "row", rowIndex, "err", err)
		return copiedRow{}, fmt.Errorf("read id column at %d: %w", rowIndex, err)
	}
	contentValue, err := columns.content.GetAsString(rowIndex)
	if err != nil {
		slog.ErrorContext(ctx, "read content column for copy failed", "row", rowIndex, "err", err)
		return copiedRow{}, fmt.Errorf("read content column at %d: %w", rowIndex, err)
	}
	startLineValue, err := columns.startLine.GetAsInt64(rowIndex)
	if err != nil {
		slog.ErrorContext(ctx, "read start line column for copy failed", "row", rowIndex, "err", err)
		return copiedRow{}, fmt.Errorf("read start_line column at %d: %w", rowIndex, err)
	}
	endLineValue, err := columns.endLine.GetAsInt64(rowIndex)
	if err != nil {
		slog.ErrorContext(ctx, "read end line column for copy failed", "row", rowIndex, "err", err)
		return copiedRow{}, fmt.Errorf("read end_line column at %d: %w", rowIndex, err)
	}
	fileExtensionValue, err := columns.fileExtension.GetAsString(rowIndex)
	if err != nil {
		slog.ErrorContext(ctx, "read file extension column for copy failed", "row", rowIndex, "err", err)
		return copiedRow{}, fmt.Errorf("read file_extension column at %d: %w", rowIndex, err)
	}
	languageValue := ""
	if columns.metadata != nil {
		metadataValue, metadataErr := columns.metadata.GetAsString(rowIndex)
		if metadataErr == nil {
			languageValue = decodeMetadataLanguage(metadataValue)
		}
	}
	vector, err := vectorAt(columns.vector, rowIndex)
	if err != nil {
		slog.ErrorContext(ctx, "read vector column for copy failed", "row", rowIndex, "err", err)
		return copiedRow{}, fmt.Errorf("read vector column at %d: %w", rowIndex, err)
	}
	splitPartValue, splitPartRecorded, err := splitPartAt(columns.splitPart, rowIndex)
	if err != nil {
		return copiedRow{}, err
	}
	return copiedRow{
		chunk: model.StoredChunk{
			Content:              contentValue,
			RelativePath:         relativePath,
			StartLine:            safeInt32FromInt64(startLineValue),
			EndLine:              safeInt32FromInt64(endLineValue),
			Language:             languageValue,
			FileExtension:        fileExtensionValue,
			ConversationID:       "",
			ParentConversationID: "",
			MessageIndex:         0,
			Role:                 "",
			TimestampUnix:        0,
			WorkspaceRoot:        "",
			Archived:             false,
			SplitPart:            splitPartValue,
			SplitPartRecorded:    splitPartRecorded,
			LoadRules:            "",
			Score:                0,
		},
		id:                idValue,
		vector:            vector,
		splitPartRecorded: splitPartRecorded,
	}, nil
}

func splitPartAt(splitPartColumn column.Column, rowIndex int) (int32, bool, error) {
	if splitPartColumn == nil {
		return 0, false, nil
	}
	isNull, nullErr := splitPartColumn.IsNull(rowIndex)
	if nullErr != nil {
		slog.Error("read split part null state failed", "row", rowIndex, "err", nullErr)
		return 0, false, fmt.Errorf("read split part null state at %d: %w", rowIndex, nullErr)
	}
	if isNull {
		return 0, false, nil
	}
	value, valueErr := splitPartColumn.GetAsInt64(rowIndex)
	if valueErr != nil {
		slog.Error("read split part column failed", "row", rowIndex, "err", valueErr)
		return 0, false, fmt.Errorf("read split part column at %d: %w", rowIndex, valueErr)
	}
	return safeInt32FromInt64(value), true, nil
}

// vectorAt extracts one float-vector row from a Milvus result column. The
// client's typed Column surface exposes the row through Get(int); for a
// dense FloatVector column the returned value is entity.FloatVector,
// which is just a []float32 with a named type.
func vectorAt(vectorColumn column.Column, rowIndex int) ([]float32, error) {
	raw, err := vectorColumn.Get(rowIndex)
	if err != nil {
		slog.Error("read vector row failed", "row", rowIndex, "err", err)
		return nil, fmt.Errorf("read vector row %d: %w", rowIndex, err)
	}
	switch typed := raw.(type) {
	case entity.FloatVector:
		out := make([]float32, len(typed))
		copy(out, typed)
		return out, nil
	case []float32:
		out := make([]float32, len(typed))
		copy(out, typed)
		return out, nil
	}
	err = fmt.Errorf("unexpected vector row type %T", raw)
	slog.Error("vector row type unexpected", "row", rowIndex, "err", err)
	return nil, err
}

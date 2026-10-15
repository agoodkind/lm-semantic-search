package semantic

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"time"

	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/model"
	"google.golang.org/grpc/peer"
)

// ChunkCopy uses source and destination paths relative to the codebase root.
type ChunkCopy struct {
	Source      string
	Destination string
}

// CopyChunks reads source rows for each copy with different source and
// destination paths and inserts them under the destination path. After all
// inserts, it flushes the collection once, deletes the rows of all copied source
// paths with one delete, and does not flush after that delete. The returned
// slice contains one row count per copy in the order of copies, with zero for a
// copy with equal source and destination paths or no source rows. A failure
// before the flush leaves the source rows in Milvus. After the flush, the
// destination rows stay in Milvus even when the delete fails or its result is
// uncertain.
func (service *Service) CopyChunks(ctx context.Context, codebasePath string, copies []ChunkCopy) ([]int, error) {
	peerInfo, _ := peer.FromContext(ctx)
	if !service.Available() {
		return nil, ErrUnavailable
	}
	copiedRowCounts := make([]int, len(copies))
	if !hasDistinctChunkCopy(copies) {
		return copiedRowCounts, nil
	}
	collectionName := service.CollectionName(codebasePath)
	hasCollection, err := service.hasCollection(ctx, collectionName, "check Milvus collection "+collectionName)
	if err != nil {
		return nil, err
	}
	if !hasCollection {
		return nil, ErrCollectionMissing
	}
	if err := service.PrepareCollection(ctx, collectionName); err != nil {
		return nil, err
	}
	lease, err := service.AcquireCollection(ctx, collectionName)
	if err != nil {
		return nil, err
	}
	defer lease.Release()

	copiedSources := make([]string, 0, len(copies))
	for index, chunkCopy := range copies {
		if chunkCopy.Source == chunkCopy.Destination {
			continue
		}
		inserted, insertErr := service.insertCopiedRows(ctx, collectionName, chunkCopy)
		if insertErr != nil {
			return nil, insertErr
		}
		copiedRowCounts[index] = inserted
		if inserted > 0 && !slices.Contains(copiedSources, chunkCopy.Source) {
			copiedSources = append(copiedSources, chunkCopy.Source)
		}
	}
	if len(copiedSources) == 0 {
		return copiedRowCounts, nil
	}
	// The flush makes all destination rows durable before any source row deletion.
	if err := service.flushCollection(ctx, collectionName); err != nil {
		return nil, err
	}
	// The delete applies to later queries without another flush.
	if _, err := service.deleteByRelativePaths(ctx, collectionName, copiedSources); err != nil {
		return nil, err
	}
	slog.InfoContext(ctx, "semantic.copy_chunks", "collection", collectionName, "copies", len(copies), "sources_deleted", len(copiedSources), "peer", peerInfo.String())
	return copiedRowCounts, nil
}

func hasDistinctChunkCopy(copies []ChunkCopy) bool {
	for _, chunkCopy := range copies {
		if chunkCopy.Source != chunkCopy.Destination {
			return true
		}
	}
	return false
}

// insertCopiedRows derives new row IDs from the destination path. Milvus cannot
// update a row's primary key in place.
func (service *Service) insertCopiedRows(ctx context.Context, collectionName string, chunkCopy ChunkCopy) (int, error) {
	source, err := service.fetchChunksForPath(ctx, collectionName, chunkCopy.Source)
	if err != nil {
		return 0, err
	}
	if len(source.chunks) == 0 {
		return 0, nil
	}
	rewritten, destinationIDs, splitPartsRecorded := rewriteCopiedRows(source, chunkCopy.Destination)
	// The insert reuses stored dense vectors and embedding models without calling
	// an embedding API. In a hybrid collection, Milvus derives sparse vectors again
	// from the copied content with its BM25 function.
	if err := service.insertBatchWithIDs(
		ctx,
		collectionName,
		rewritten,
		destinationIDs,
		source.vectors,
		source.embeddingModels,
		splitPartsRecorded,
		CodeColumns(),
	); err != nil {
		return 0, err
	}
	return len(rewritten), nil
}

// Milvus returns a rate limit error when a flush occurs less than 10 seconds
// after the previous flush of the same collection. An earlier CopyChunks call or
// another Milvus client, such as the upstream TypeScript adapter, can perform
// that previous flush. flushCollection retries only rate limit errors, with 8
// total attempts 2 seconds apart. Each flush can wait up to 14 seconds.
const (
	flushRateLimitRetryWait   = 2 * time.Second
	flushRateLimitMaxAttempts = 8
)

func (service *Service) flushCollection(ctx context.Context, collectionName string) error {
	peerInfo, _ := peer.FromContext(ctx)
	startedAt := clock.Now()
	task, attempts, err := service.requestFlush(ctx, collectionName)
	if attempts > 1 {
		slog.InfoContext(ctx, "milvus rate limited the flush request", "collection", collectionName, "attempts", attempts, "waited_ms", clock.Now().Sub(startedAt).Milliseconds(), "peer", peerInfo.String())
	}
	if err != nil {
		return err
	}
	if err := task.Await(ctx); err != nil {
		return wrapStoreError(ctx, err, "await Milvus collection flush "+collectionName)
	}
	return nil
}

func (service *Service) requestFlush(
	ctx context.Context,
	collectionName string,
) (*milvusclient.FlushTask, int, error) {
	for attempt := 1; ; attempt++ {
		task, err := service.milvus.Flush(
			ctx,
			milvusclient.NewFlushOption(collectionName),
		)
		if err == nil {
			return task, attempt, nil
		}
		flushErr := classifyFlushError(err)
		var rateLimited flushRateLimitError
		if !errors.As(flushErr, &rateLimited) || attempt >= flushRateLimitMaxAttempts {
			return nil, attempt, wrapStoreError(ctx, flushErr, "flush Milvus collection "+collectionName)
		}
		if !reconnectSleep(ctx, flushRateLimitRetryWait) {
			// When the context ends during the retry wait, the returned error preserves
			// both the context error and the last Milvus rate limit error. Callers use
			// errors.Is to distinguish cancellation from a Milvus failure.
			return nil, attempt, wrapStoreError(ctx, errors.Join(ctx.Err(), flushErr), "flush Milvus collection "+collectionName)
		}
	}
}

func copiedChunkID(sourceID string, dstRelativePath string) string {
	sum := sha256.Sum256([]byte(dstRelativePath + ":" + sourceID))
	return "chunk_" + hex.EncodeToString(sum[:])[:16]
}

type copiedRows struct {
	chunks             []model.StoredChunk
	ids                []string
	vectors            [][]float32
	embeddingModels    []string
	splitPartsRecorded []bool
}

type copiedRowColumns struct {
	id             column.Column
	content        column.Column
	startLine      column.Column
	endLine        column.Column
	fileExtension  column.Column
	metadata       column.Column
	vector         column.Column
	splitPart      column.Column
	embeddingModel column.Column
}

type copiedRow struct {
	chunk             model.StoredChunk
	id                string
	vector            []float32
	splitPartRecorded bool
	embeddingModel    string
}

func noCopiedRows() copiedRows {
	return copiedRows{
		chunks:             nil,
		ids:                nil,
		vectors:            nil,
		embeddingModels:    nil,
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
	peerInfo, _ := peer.FromContext(ctx)
	if err := service.ensureSplitPartColumnOnce(ctx, collectionName); err != nil {
		return noCopiedRows(), err
	}
	if err := service.ensureReuseIdentityColumnsOnce(ctx, collectionName); err != nil {
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
		embeddingModelFieldName,
	}
	queryOption := milvusclient.NewQueryOption(collectionName).WithFilter(expression).WithOutputFields(outputFields...)
	resultSet, err := service.milvus.Query(ctx, queryOption)
	if err != nil {
		slog.ErrorContext(ctx, "query chunks for copy failed", "collection", collectionName, "path", relativePath, "peer", peerInfo.String(), "err", err)
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
	embeddingModels := make([]string, 0, resultSet.ResultCount)
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
		embeddingModels = append(embeddingModels, row.embeddingModel)
		splitPartsRecorded = append(splitPartsRecorded, row.splitPartRecorded)
	}
	return copiedRows{
		chunks:             chunks,
		ids:                ids,
		vectors:            vectors,
		embeddingModels:    embeddingModels,
		splitPartsRecorded: splitPartsRecorded,
	}, nil
}

func copiedColumns(resultSet milvusclient.ResultSet) (copiedRowColumns, error) {
	columns := copiedRowColumns{
		id:             resultSet.GetColumn(idFieldName),
		content:        resultSet.GetColumn(contentFieldName),
		startLine:      resultSet.GetColumn(startLineFieldName),
		endLine:        resultSet.GetColumn(endLineFieldName),
		fileExtension:  resultSet.GetColumn(fileExtensionFieldName),
		metadata:       resultSet.GetColumn(metadataFieldName),
		vector:         resultSet.GetColumn(denseVectorFieldName),
		splitPart:      resultSet.GetColumn(splitPartFieldName),
		embeddingModel: resultSet.GetColumn(embeddingModelFieldName),
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
	vector, err := milvusstore.VectorAt(columns.vector, rowIndex)
	if err != nil {
		slog.ErrorContext(ctx, "read vector column for copy failed", "row", rowIndex, "err", err)
		return copiedRow{}, fmt.Errorf("read vector column at %d: %w", rowIndex, err)
	}
	splitPartValue, splitPartRecorded, err := milvusstore.SplitPartAt(columns.splitPart, rowIndex)
	if err != nil {
		return copiedRow{}, fmt.Errorf("read split part column for copy at %d: %w", rowIndex, err)
	}
	embeddingModel, err := nullableStringAt(columns.embeddingModel, rowIndex)
	if err != nil {
		return copiedRow{}, fmt.Errorf("read embedding model column of copied row %d: %w", rowIndex, err)
	}
	return copiedRow{
		chunk: model.StoredChunk{
			Content:           contentValue,
			RelativePath:      relativePath,
			StartLine:         milvusstore.SafeInt32(startLineValue),
			EndLine:           milvusstore.SafeInt32(endLineValue),
			Language:          languageValue,
			FileExtension:     fileExtensionValue,
			SplitPart:         splitPartValue,
			SplitPartRecorded: splitPartRecorded,
			Scalars:           nil,
			Score:             0,
		},
		id:                idValue,
		vector:            vector,
		splitPartRecorded: splitPartRecorded,
		embeddingModel:    embeddingModel,
	}, nil
}

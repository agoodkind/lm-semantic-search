package semantic

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/spans"
)

type insertBatchColumns struct {
	contents          []string
	contentVectorKeys []string
	relativePaths     []string
	startLines        []int64
	endLines          []int64
	fileExtensions    []string
	metadataValues    []string
	splitParts        []int64
	scalars           conversationScalarColumns
	sanitizedCount    int
}

func (service *Service) insertBatch(
	ctx context.Context,
	collectionName string,
	chunks []model.StoredChunk,
	vectors [][]float32,
	columnSet StoreColumnSet,
) error {
	ids := make([]string, 0, len(chunks))
	splitPartsRecorded := make([]bool, 0, len(chunks))
	for index, chunk := range chunks {
		ids = append(ids, generateID(chunk, index))
		splitPartsRecorded = append(splitPartsRecorded, true)
	}
	return service.insertBatchWithIDs(
		ctx,
		collectionName,
		chunks,
		ids,
		vectors,
		splitPartsRecorded,
		columnSet,
	)
}

func (service *Service) insertBatchWithIDs(
	ctx context.Context,
	collectionName string,
	chunks []model.StoredChunk,
	ids []string,
	vectors [][]float32,
	splitPartsRecorded []bool,
	columnSet StoreColumnSet,
) (err error) {
	ctx, done := spans.Open(ctx, "semantic.insertBatch")
	defer done(&err)

	if err := validateInsertBatchCounts(
		ctx,
		collectionName,
		len(chunks),
		len(ids),
		len(splitPartsRecorded),
	); err != nil {
		return err
	}
	conversationCollection := columnSet.ConversationScalars()
	if err := service.ensureInsertColumns(
		ctx,
		collectionName,
		conversationCollection,
	); err != nil {
		return err
	}

	columns := buildInsertBatchColumns(ctx, chunks, conversationCollection)
	columns.contentVectorKeys = contentVectorStorageKeys(service.cfg, columns.contents)
	if columns.sanitizedCount > 0 {
		slog.WarnContext(
			ctx,
			"semantic.insertBatch sanitized chunks before Milvus marshal",
			"collection",
			collectionName,
			"sanitized",
			columns.sanitizedCount,
			"batch_size",
			len(chunks),
		)
	}
	splitPartColumn, err := newSplitPartColumn(
		collectionName,
		columns.splitParts,
		splitPartsRecorded,
	)
	if err != nil {
		return err
	}
	insertOption := milvusclient.NewColumnBasedInsertOption(collectionName).
		WithVarcharColumn(idFieldName, ids).
		WithVarcharColumn(contentFieldName, columns.contents).
		WithVarcharColumn(contentVectorKeyFieldName, columns.contentVectorKeys).
		WithVarcharColumn(relativePathFieldName, columns.relativePaths).
		WithInt64Column(startLineFieldName, columns.startLines).
		WithInt64Column(endLineFieldName, columns.endLines).
		WithVarcharColumn(fileExtensionFieldName, columns.fileExtensions).
		WithVarcharColumn(metadataFieldName, columns.metadataValues).
		WithColumns(splitPartColumn).
		WithFloatVectorColumn(denseVectorFieldName, len(vectors[0]), vectors)
	if conversationCollection {
		insertOption = insertOption.
			WithVarcharColumn(conversationIDFieldName, columns.scalars.conversationIDs).
			WithVarcharColumn(parentConversationIDFieldName, columns.scalars.parentConversationIDs).
			WithVarcharColumn(roleFieldName, columns.scalars.roles).
			WithVarcharColumn(providerFieldName, columns.scalars.providers).
			WithVarcharColumn(workspaceRootFieldName, columns.scalars.workspaceRoots).
			WithBoolColumn(archivedFieldName, columns.scalars.archiveds).
			WithInt64Column(timestampUnixFieldName, columns.scalars.timestamps).
			WithInt64Column(messageIndexFieldName, columns.scalars.messageIndexes)
	}

	insertResult, err := service.executeInsert(ctx, insertOption)
	if err != nil {
		return err
	}
	if insertResult.InsertCount != int64(len(chunks)) {
		countErr := fmt.Errorf(
			"milvus acknowledged %d of %d rows",
			insertResult.InsertCount,
			len(chunks),
		)
		slog.ErrorContext(
			ctx,
			"insert Milvus batch returned unexpected row count",
			"collection",
			collectionName,
			"inserted",
			insertResult.InsertCount,
			"expected",
			len(chunks),
			"err",
			countErr,
		)
		return fmt.Errorf("insert Milvus batch into %s: %w", collectionName, countErr)
	}
	return nil
}

func (service *Service) ensureInsertColumns(
	ctx context.Context,
	collectionName string,
	conversationCollection bool,
) error {
	if err := service.ensureSplitPartColumnOnce(ctx, collectionName); err != nil {
		return err
	}
	if err := service.ensureContentVectorKeyColumnOnce(ctx, collectionName); err != nil {
		return err
	}
	if !conversationCollection {
		return nil
	}
	return service.ensureConversationScalarColumnsOnce(ctx, collectionName)
}

func (service *Service) executeInsert(
	ctx context.Context,
	option milvusclient.InsertOption,
) (milvusclient.InsertResult, error) {
	var result milvusclient.InsertResult
	var err error
	if service.insertRows != nil {
		result, err = service.insertRows(ctx, option)
	} else {
		result, err = service.milvus.Insert(ctx, option)
	}
	if err != nil {
		return result, wrapStoreError(
			ctx,
			err,
			"insert Milvus batch into "+option.CollectionName(),
		)
	}
	return result, nil
}

func validateInsertBatchCounts(
	ctx context.Context,
	collectionName string,
	chunkCount int,
	idCount int,
	splitPartCount int,
) error {
	if idCount == chunkCount && splitPartCount == chunkCount {
		return nil
	}
	countErr := errors.New("insert batch column count mismatch")
	slog.ErrorContext(
		ctx,
		"insert batch column count mismatch",
		"collection",
		collectionName,
		"ids",
		idCount,
		"split_parts_recorded",
		splitPartCount,
		"chunks",
		chunkCount,
		"err",
		countErr,
	)
	return fmt.Errorf("insert Milvus batch into %s: %w", collectionName, countErr)
}

func buildInsertBatchColumns(
	ctx context.Context,
	chunks []model.StoredChunk,
	conversationCollection bool,
) insertBatchColumns {
	columns := insertBatchColumns{
		contents:          make([]string, 0, len(chunks)),
		contentVectorKeys: make([]string, 0, len(chunks)),
		relativePaths:     make([]string, 0, len(chunks)),
		startLines:        make([]int64, 0, len(chunks)),
		endLines:          make([]int64, 0, len(chunks)),
		fileExtensions:    make([]string, 0, len(chunks)),
		metadataValues:    make([]string, 0, len(chunks)),
		splitParts:        make([]int64, 0, len(chunks)),
		scalars:           newConversationScalarColumns(conversationCollection, len(chunks)),
		sanitizedCount:    0,
	}
	for _, chunk := range chunks {
		content, contentChanged := sanitizeUTF8(chunk.Content)
		relativePath, pathChanged := sanitizeUTF8(chunk.RelativePath)
		fileExtension, extensionChanged := sanitizeUTF8(chunk.FileExtension)
		metadataValue, metadataChanged := sanitizeUTF8(encodeMetadata(chunk))
		if contentChanged || pathChanged || extensionChanged || metadataChanged {
			columns.sanitizedCount++
			slog.WarnContext(ctx, "semantic.sanitized_invalid_utf8", "relative_path", chunk.RelativePath, "start_line", chunk.StartLine, "end_line", chunk.EndLine, "content_changed", contentChanged, "path_changed", pathChanged, "extension_changed", extensionChanged, "metadata_changed", metadataChanged)
		}
		columns.contents = append(columns.contents, content)
		columns.relativePaths = append(columns.relativePaths, relativePath)
		columns.startLines = append(columns.startLines, int64(chunk.StartLine))
		columns.endLines = append(columns.endLines, int64(chunk.EndLine))
		columns.fileExtensions = append(columns.fileExtensions, fileExtension)
		columns.metadataValues = append(columns.metadataValues, metadataValue)
		columns.splitParts = append(columns.splitParts, int64(chunk.SplitPart))
		columns.scalars.append(chunk)
	}
	return columns
}

func newSplitPartColumn(
	collectionName string,
	splitParts []int64,
	splitPartsRecorded []bool,
) (column.Column, error) {
	splitPartColumn, err := column.NewNullableColumnInt64(
		splitPartFieldName,
		splitParts,
		splitPartsRecorded,
		column.WithSparseNullableMode[int64](true),
	)
	if err != nil {
		slog.Error(
			"build split part insert column failed",
			"collection",
			collectionName,
			"err",
			err,
		)
		return nil, fmt.Errorf("build split part column for %s: %w", collectionName, err)
	}
	return splitPartColumn, nil
}

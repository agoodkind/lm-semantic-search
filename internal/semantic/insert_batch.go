package semantic

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/spans"
)

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
	if err := service.ensureInsertColumns(
		ctx,
		collectionName,
		columnSet.ConversationScalars(),
	); err != nil {
		return err
	}

	rows := insertRowsFromChunks(chunks, ids, vectors, splitPartsRecorded, columnSet)
	insertOption, err := milvusstore.BuildInsertOption(
		ctx,
		collectionName,
		rows,
		columnSet.creationScalars(),
		service.cfg.EmbeddingModel,
	)
	if err != nil {
		return fmt.Errorf("build insert request for %s: %w", collectionName, err)
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

// insertRowsFromChunks builds one store row per chunk. The row stores the
// metadata JSON the chunk encodes and the scalar values the column set writes.
func insertRowsFromChunks(
	chunks []model.StoredChunk,
	ids []string,
	vectors [][]float32,
	splitPartsRecorded []bool,
	columnSet StoreColumnSet,
) []collection.Row {
	rows := make([]collection.Row, 0, len(chunks))
	for index, chunk := range chunks {
		rows = append(rows, collection.Row{
			ID:                ids[index],
			Content:           chunk.Content,
			RelativePath:      chunk.RelativePath,
			StartLine:         chunk.StartLine,
			EndLine:           chunk.EndLine,
			FileExtension:     chunk.FileExtension,
			Metadata:          encodeMetadata(chunk),
			SplitPart:         chunk.SplitPart,
			SplitPartRecorded: splitPartsRecorded[index],
			Vector:            vectors[index],
			Scalars:           columnSet.rowScalars(chunk),
		})
	}
	return rows
}

func (service *Service) ensureInsertColumns(
	ctx context.Context,
	collectionName string,
	conversationCollection bool,
) error {
	if err := service.ensureSplitPartColumnOnce(ctx, collectionName); err != nil {
		return err
	}
	if err := service.ensureReuseIdentityColumnsOnce(ctx, collectionName); err != nil {
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

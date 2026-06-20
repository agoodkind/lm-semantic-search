package semantic

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/model"
)

// conversationBackfillBatchSize bounds how many rows are read and upserted per
// backfill batch, keeping memory (each row carries a dense vector) bounded.
const conversationBackfillBatchSize = 500

// BackfillConversationScalarColumns populates the native scalar columns on every
// existing row of a conversation collection from the metadata already stored on
// the row, preserving the dense vector so no chunk is re-embedded. This is the
// no-reindex migration for rows written before the columns existed:
// conversation_id, parent, role, message_index, and timestamp come from the
// metadata JSON, and provider from the conversation-id prefix. workspace_root is
// not in stored metadata, so it stays null on old rows; workspace filtering uses
// the conversation_id column instead. Idempotent: re-running upserts the same
// values. Returns the number of rows rewritten.
func (service *Service) BackfillConversationScalarColumns(ctx context.Context, collectionName string) (int, error) {
	if !service.Available() {
		return 0, ErrUnavailable
	}
	if !isConversationCollection(collectionName) {
		return 0, fmt.Errorf("backfill: %s is not a conversation collection", collectionName)
	}
	// A dormant conversation collection that never went through the on-access
	// column migration lacks the scalar columns, so the upsert below would reject
	// every batch with a schema mismatch. Add the columns first; this is a no-op
	// when they already exist.
	if _, err := service.addMissingConversationScalarColumns(ctx, collectionName); err != nil {
		return 0, err
	}
	if err := service.loadCollectionForRead(ctx, collectionName); err != nil {
		return 0, err
	}
	iterator, err := service.milvus.QueryIterator(ctx, milvusclient.NewQueryIteratorOption(collectionName).
		WithBatchSize(conversationBackfillBatchSize).
		WithOutputFields(idFieldName, contentFieldName, relativePathFieldName, startLineFieldName, endLineFieldName, fileExtensionFieldName, metadataFieldName, denseVectorFieldName))
	if err != nil {
		slog.ErrorContext(ctx, "open conversation backfill iterator failed", "collection", collectionName, "err", err)
		return 0, fmt.Errorf("open backfill iterator for %s: %w", collectionName, err)
	}
	total := 0
	for {
		resultSet, nextErr := iterator.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			slog.ErrorContext(ctx, "conversation backfill iterator next failed", "collection", collectionName, "rows_done", total, "err", nextErr)
			return total, fmt.Errorf("iterate %s for backfill: %w", collectionName, nextErr)
		}
		ids, chunks, vectors, buildErr := readBackfillRows(resultSet)
		if buildErr != nil {
			return total, buildErr
		}
		if len(ids) == 0 {
			continue
		}
		if err := service.upsertConversationBackfillBatch(ctx, collectionName, ids, chunks, vectors); err != nil {
			return total, err
		}
		total += len(ids)
	}
	slog.InfoContext(ctx, "semantic.conversation_scalar_backfill_complete", "collection", collectionName, "rows", total)
	return total, nil
}

// BackfillConversationCollectionsOnce runs the metadata-only scalar-column
// backfill once per conversation collection per process. It populates the native
// scalar columns (provider, role, conversation lineage, message index, timestamp)
// on rows written before those columns existed, keeping each row's dense vector
// so nothing is re-embedded. The daemon's periodic sync drives it; the
// per-collection guard makes later ticks a no-op. It is fault-tolerant per
// collection: one failure is logged and the sweep continues. This is the
// deliberate run the on-column-add trigger cannot do once the columns already
// exist, which is the case after a backfill that failed before the collection
// could load (the pre-mmap 28 GB load error).
func (service *Service) BackfillConversationCollectionsOnce(ctx context.Context) {
	if !service.Available() {
		return
	}
	collections, err := service.ListCollections(ctx)
	if err != nil {
		slog.ErrorContext(ctx, "semantic.conversation_backfill_list_failed", "err", err)
		return
	}
	swept := 0
	totalRows := 0
	for _, collectionName := range collections {
		if !isConversationCollection(collectionName) {
			continue
		}
		if _, done := service.ensuredBackfill.Load(collectionName); done {
			continue
		}
		needs, err := service.conversationCollectionNeedsBackfill(ctx, collectionName)
		if err != nil {
			slog.ErrorContext(ctx, "semantic.conversation_backfill_check_failed", "collection", collectionName, "err", err)
			continue
		}
		if !needs {
			// Already fully backfilled (every row has provider). Mark done so this
			// process does not re-upsert ~600k rows and re-inflate the collection.
			service.ensuredBackfill.Store(collectionName, struct{}{})
			continue
		}
		rows, err := service.BackfillConversationScalarColumns(ctx, collectionName)
		if err != nil {
			slog.ErrorContext(ctx, "semantic.conversation_backfill_failed", "collection", collectionName, "rows", rows, "err", err)
			continue
		}
		service.ensuredBackfill.Store(collectionName, struct{}{})
		swept++
		totalRows += rows
	}
	// Per-collection detail is logged by BackfillConversationScalarColumns; this
	// is the once-per-sweep summary, kept out of the loop.
	if swept > 0 {
		slog.InfoContext(ctx, "semantic.conversation_backfill_sweep_done", "collections", swept, "rows", totalRows)
	}
}

// conversationCollectionNeedsBackfill reports whether collectionName still needs
// the scalar-column backfill. It returns true when the collection lacks the
// provider column entirely (a dormant collection that never went through the
// on-access column migration) or has the column with at least one row where
// provider is null. A fully-backfilled collection returns false, so the sweep
// skips re-upserting every row on a later process.
func (service *Service) conversationCollectionNeedsBackfill(ctx context.Context, collectionName string) (bool, error) {
	collection, err := service.milvus.DescribeCollection(ctx, milvusclient.NewDescribeCollectionOption(collectionName))
	if err != nil {
		slog.ErrorContext(ctx, "describe conversation collection for backfill check failed", "collection", collectionName, "err", err)
		return false, fmt.Errorf("describe %s for backfill check: %w", collectionName, err)
	}
	hasProvider := false
	if collection.Schema != nil {
		for _, field := range collection.Schema.Fields {
			if field.Name == providerFieldName {
				hasProvider = true
				break
			}
		}
	}
	if !hasProvider {
		return true, nil
	}
	if err := service.loadCollectionForRead(ctx, collectionName); err != nil {
		return false, err
	}
	iterator, err := service.milvus.QueryIterator(ctx, milvusclient.NewQueryIteratorOption(collectionName).
		WithBatchSize(1).
		WithFilter(providerFieldName+" is null").
		WithOutputFields(idFieldName))
	if err != nil {
		slog.ErrorContext(ctx, "open null-provider probe failed", "collection", collectionName, "err", err)
		return false, fmt.Errorf("open null-provider probe for %s: %w", collectionName, err)
	}
	resultSet, nextErr := iterator.Next(ctx)
	if errors.Is(nextErr, io.EOF) {
		return false, nil
	}
	if nextErr != nil {
		slog.ErrorContext(ctx, "null-provider probe failed", "collection", collectionName, "err", nextErr)
		return false, fmt.Errorf("probe null provider in %s: %w", collectionName, nextErr)
	}
	return resultSet.ResultCount > 0, nil
}

// resultSetToBackfillRows decodes one iterator page into the existing primary
// keys, the reconstructed chunks (conversation attributes recovered from the
// metadata JSON), and the preserved dense vectors. The existing ids are read,
// not re-derived, so the upsert overwrites each row in place rather than
// creating a duplicate.
func readBackfillRows(resultSet milvusclient.ResultSet) ([]string, []model.StoredChunk, [][]float32, error) {
	idColumn := resultSet.GetColumn(idFieldName)
	contentColumn := resultSet.GetColumn(contentFieldName)
	relativePathColumn := resultSet.GetColumn(relativePathFieldName)
	startLineColumn := resultSet.GetColumn(startLineFieldName)
	endLineColumn := resultSet.GetColumn(endLineFieldName)
	fileExtensionColumn := resultSet.GetColumn(fileExtensionFieldName)
	metadataColumn := resultSet.GetColumn(metadataFieldName)
	vectorColumn := resultSet.GetColumn(denseVectorFieldName)
	if idColumn == nil || contentColumn == nil || relativePathColumn == nil || vectorColumn == nil {
		return nil, nil, nil, ErrSearchResultIncomplete
	}

	ids := make([]string, 0, resultSet.ResultCount)
	chunks := make([]model.StoredChunk, 0, resultSet.ResultCount)
	vectors := make([][]float32, 0, resultSet.ResultCount)
	for rowIndex := range resultSet.ResultCount {
		id, idErr := idColumn.GetAsString(rowIndex)
		if idErr != nil {
			slog.Error("read id column failed", "index", rowIndex, "err", idErr)
			return nil, nil, nil, fmt.Errorf("read id column at %d: %w", rowIndex, idErr)
		}
		content, contentErr := contentColumn.GetAsString(rowIndex)
		if contentErr != nil {
			slog.Error("read content column failed", "index", rowIndex, "err", contentErr)
			return nil, nil, nil, fmt.Errorf("read content column at %d: %w", rowIndex, contentErr)
		}
		relativePath, relativePathErr := relativePathColumn.GetAsString(rowIndex)
		if relativePathErr != nil {
			slog.Error("read relative path column failed", "index", rowIndex, "err", relativePathErr)
			return nil, nil, nil, fmt.Errorf("read relative path column at %d: %w", rowIndex, relativePathErr)
		}
		vector, vectorErr := vectorAt(vectorColumn, rowIndex)
		if vectorErr != nil {
			slog.Error("read vector column failed", "index", rowIndex, "err", vectorErr)
			return nil, nil, nil, fmt.Errorf("read vector column at %d: %w", rowIndex, vectorErr)
		}
		startLine := backfillInt64(startLineColumn, rowIndex)
		endLine := backfillInt64(endLineColumn, rowIndex)
		fileExtension := backfillString(fileExtensionColumn, rowIndex)
		metadata := emptyChunkMetadata()
		if metadataColumn != nil {
			if rawMetadata, metaErr := metadataColumn.GetAsString(rowIndex); metaErr == nil {
				metadata = decodeMetadata(rawMetadata)
			}
		}

		ids = append(ids, id)
		vectors = append(vectors, vector)
		chunks = append(chunks, model.StoredChunk{
			Content:              content,
			RelativePath:         relativePath,
			StartLine:            safeInt32FromInt64(startLine),
			EndLine:              safeInt32FromInt64(endLine),
			Language:             metadata.Language,
			FileExtension:        fileExtension,
			ConversationID:       metadata.ConversationID,
			ParentConversationID: metadata.ParentConversationID,
			MessageIndex:         metadata.messageIndex(),
			Role:                 metadata.Role,
			TimestampUnix:        metadata.timestampUnix(),
			WorkspaceRoot:        "",
			Score:                0,
		})
	}
	return ids, chunks, vectors, nil
}

func backfillInt64(col column.Column, rowIndex int) int64 {
	if col == nil {
		return 0
	}
	value, err := col.GetAsInt64(rowIndex)
	if err != nil {
		return 0
	}
	return value
}

func backfillString(col column.Column, rowIndex int) string {
	if col == nil {
		return ""
	}
	value, err := col.GetAsString(rowIndex)
	if err != nil {
		return ""
	}
	return value
}

// upsertConversationBackfillBatch overwrites each row in place: it keeps the
// existing primary key, content, path, and dense vector, and adds the native
// scalar columns derived from the chunk. Upsert matches by primary key, so no
// row is duplicated and no vector is regenerated.
func (service *Service) upsertConversationBackfillBatch(ctx context.Context, collectionName string, ids []string, chunks []model.StoredChunk, vectors [][]float32) error {
	if len(ids) == 0 {
		return nil
	}
	scalars := newConversationScalarColumns(true, len(chunks))
	contents := make([]string, 0, len(chunks))
	relativePaths := make([]string, 0, len(chunks))
	startLines := make([]int64, 0, len(chunks))
	endLines := make([]int64, 0, len(chunks))
	fileExtensions := make([]string, 0, len(chunks))
	metadataValues := make([]string, 0, len(chunks))
	for _, chunk := range chunks {
		content, _ := sanitizeUTF8(chunk.Content)
		relativePath, _ := sanitizeUTF8(chunk.RelativePath)
		fileExtension, _ := sanitizeUTF8(chunk.FileExtension)
		metadataValue, _ := sanitizeUTF8(encodeMetadata(chunk))
		contents = append(contents, content)
		relativePaths = append(relativePaths, relativePath)
		startLines = append(startLines, int64(chunk.StartLine))
		endLines = append(endLines, int64(chunk.EndLine))
		fileExtensions = append(fileExtensions, fileExtension)
		metadataValues = append(metadataValues, metadataValue)
		scalars.append(chunk)
	}

	// workspaceRoot is intentionally omitted from the backfill upsert: it is not
	// carried in the stored metadata JSON, so a backfilled value would be the
	// empty string, which writes a non-NULL "" and contradicts the "stays null on
	// old rows" contract. Leaving the nullable column out of the upsert preserves
	// NULL, and workspace filtering falls back to the conversation_id column.
	option := milvusclient.NewColumnBasedInsertOption(collectionName).
		WithVarcharColumn(idFieldName, ids).
		WithVarcharColumn(contentFieldName, contents).
		WithVarcharColumn(relativePathFieldName, relativePaths).
		WithInt64Column(startLineFieldName, startLines).
		WithInt64Column(endLineFieldName, endLines).
		WithVarcharColumn(fileExtensionFieldName, fileExtensions).
		WithVarcharColumn(metadataFieldName, metadataValues).
		WithFloatVectorColumn(denseVectorFieldName, len(vectors[0]), vectors).
		WithVarcharColumn(conversationIDFieldName, scalars.conversationIDs).
		WithVarcharColumn(parentConversationIDFieldName, scalars.parentConversationIDs).
		WithVarcharColumn(roleFieldName, scalars.roles).
		WithVarcharColumn(providerFieldName, scalars.providers).
		WithInt64Column(timestampUnixFieldName, scalars.timestamps).
		WithInt64Column(messageIndexFieldName, scalars.messageIndexes)

	if _, err := service.milvus.Upsert(ctx, option); err != nil {
		slog.ErrorContext(ctx, "conversation backfill upsert failed", "collection", collectionName, "rows", len(ids), "err", err)
		return fmt.Errorf("upsert backfill batch into %s: %w", collectionName, err)
	}
	return nil
}

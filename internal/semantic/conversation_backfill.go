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
		if err := service.upsertConversationColumns(ctx, collectionName, ids, chunks, vectors, conversationUpsertOptions{WriteWorkspaceRoot: false, WriteArchived: false}); err != nil {
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
	workspaceRootColumn := resultSet.GetColumn(workspaceRootFieldName)
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
			WorkspaceRoot:        backfillString(workspaceRootColumn, rowIndex),
			Archived:             false,
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

// upsertConversationColumns overwrites each row in place: it keeps the existing
// primary key, content, path, and dense vector, and writes the native scalar
// columns derived from the chunk. Upsert matches by primary key, so no row is
// duplicated and no vector is regenerated. opts gates the enrichment-sourced
// columns: workspaceRoot is written only when opts.WriteWorkspaceRoot is true
// (the caller populated chunk.WorkspaceRoot from a clyde enrichment); otherwise
// it is omitted so the nullable column keeps its existing value.
func (service *Service) upsertConversationColumns(ctx context.Context, collectionName string, ids []string, chunks []model.StoredChunk, vectors [][]float32, opts conversationUpsertOptions) error {
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

	// workspaceRoot is written only when opts.WriteWorkspaceRoot is set. The
	// metadata-only sweep cannot source it (it is not carried in the stored
	// metadata JSON), so it leaves the nullable column out of the upsert to
	// preserve the row's existing value (NULL on old rows), and workspace
	// filtering falls back to the conversation_id column. The enrichment-driven
	// backfill sets the flag after populating chunk.WorkspaceRoot from clyde.
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
	if opts.WriteWorkspaceRoot {
		option = option.WithVarcharColumn(workspaceRootFieldName, scalars.workspaceRoots)
	}
	if opts.WriteArchived {
		option = option.WithBoolColumn(archivedFieldName, scalars.archiveds)
	}

	if _, err := service.milvus.Upsert(ctx, option); err != nil {
		slog.ErrorContext(ctx, "conversation backfill upsert failed", "collection", collectionName, "rows", len(ids), "err", err)
		return fmt.Errorf("upsert backfill batch into %s: %w", collectionName, err)
	}
	return nil
}

// partitionConversationEnrichment splits one backfill page into the rows whose
// conversation clyde still knows and the orphans whose conversation is absent
// from the enrichment (its artifact is gone). For a known conversation it sets
// chunk.Archived from the enrichment and fills chunk.WorkspaceRoot from the
// enrichment only when the row's own workspace is empty, so a row that already
// carries a workspace keeps it. The chunks slice is mutated in place for the
// resolvable rows; orphans are left untouched.
func partitionConversationEnrichment(ids []string, chunks []model.StoredChunk, vectors [][]float32, enrichment ConversationEnrichment) ([]string, []model.StoredChunk, [][]float32, int) {
	fillIDs := make([]string, 0, len(ids))
	fillChunks := make([]model.StoredChunk, 0, len(chunks))
	fillVectors := make([][]float32, 0, len(vectors))
	orphan := 0
	for index := range ids {
		value, ok := enrichment[chunks[index].ConversationID]
		if !ok {
			orphan++
			continue
		}
		if chunks[index].WorkspaceRoot == "" {
			chunks[index].WorkspaceRoot = value.WorkspaceRoot
		}
		chunks[index].Archived = value.Archived
		fillIDs = append(fillIDs, ids[index])
		fillChunks = append(fillChunks, chunks[index])
		fillVectors = append(fillVectors, vectors[index])
	}
	return fillIDs, fillChunks, fillVectors, orphan
}

// BackfillConversationEnrichment writes the clyde-supplied enrichment columns
// (workspaceRoot and archived) onto the rows missing either, from an enrichment
// keyed by conversation id, preserving each row's dense vector so nothing is
// re-embedded. It iterates only the rows whose workspaceRoot is empty or whose
// archived is still null (WithFilter), so it touches a row only while it needs
// enrichment, never a fully-populated row. It reads each row's existing
// workspaceRoot so filling archived on a row that already has a workspace keeps
// that workspace. A row whose conversation id is absent from the enrichment is
// an orphan (its artifact is gone) and is left untouched. When dryRun is true it
// counts the would-change and orphan rows and writes nothing. Returns
// (changed, orphan).
func (service *Service) BackfillConversationEnrichment(ctx context.Context, collectionName string, enrichment ConversationEnrichment, dryRun bool) (int, int, error) {
	if !service.Available() {
		return 0, 0, ErrUnavailable
	}
	if !isConversationCollection(collectionName) {
		return 0, 0, fmt.Errorf("workspace backfill: %s is not a conversation collection", collectionName)
	}
	if _, err := service.addMissingConversationScalarColumns(ctx, collectionName); err != nil {
		return 0, 0, err
	}
	if err := service.loadCollectionForRead(ctx, collectionName); err != nil {
		return 0, 0, err
	}
	iterator, err := service.milvus.QueryIterator(ctx, milvusclient.NewQueryIteratorOption(collectionName).
		WithBatchSize(conversationBackfillBatchSize).
		WithFilter(workspaceRootFieldName+` == "" or `+archivedFieldName+` is null`).
		WithOutputFields(idFieldName, contentFieldName, relativePathFieldName, startLineFieldName, endLineFieldName, fileExtensionFieldName, metadataFieldName, denseVectorFieldName, workspaceRootFieldName))
	if err != nil {
		slog.ErrorContext(ctx, "open conversation workspace backfill iterator failed", "collection", collectionName, "err", err)
		return 0, 0, fmt.Errorf("open workspace backfill iterator for %s: %w", collectionName, err)
	}
	changed := 0
	orphan := 0
	for {
		resultSet, nextErr := iterator.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			slog.ErrorContext(ctx, "conversation workspace backfill iterator next failed", "collection", collectionName, "changed", changed, "err", nextErr)
			return changed, orphan, fmt.Errorf("iterate %s for workspace backfill: %w", collectionName, nextErr)
		}
		ids, chunks, vectors, buildErr := readBackfillRows(resultSet)
		if buildErr != nil {
			return changed, orphan, buildErr
		}
		fillIDs, fillChunks, fillVectors, pageOrphans := partitionConversationEnrichment(ids, chunks, vectors, enrichment)
		orphan += pageOrphans
		changed += len(fillIDs)
		if dryRun || len(fillIDs) == 0 {
			continue
		}
		if err := service.upsertConversationColumns(ctx, collectionName, fillIDs, fillChunks, fillVectors, conversationUpsertOptions{WriteWorkspaceRoot: true, WriteArchived: true}); err != nil {
			return changed, orphan, err
		}
	}
	slog.InfoContext(ctx, "semantic.conversation_workspace_backfill_complete", "collection", collectionName, "changed", changed, "orphan", orphan, "dry_run", dryRun)
	return changed, orphan, nil
}

package semantic

import (
	"fmt"
	"log/slog"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/model"
)

func resultSetsToChunks(resultSets []milvusclient.ResultSet) ([]model.StoredChunk, error) {
	if len(resultSets) == 0 {
		return []model.StoredChunk{}, nil
	}

	resultSet := resultSets[0]
	if resultSet.ResultCount == 0 {
		return []model.StoredChunk{}, nil
	}
	contentColumn := resultSet.GetColumn(contentFieldName)
	relativePathColumn := resultSet.GetColumn(relativePathFieldName)
	startLineColumn := resultSet.GetColumn(startLineFieldName)
	endLineColumn := resultSet.GetColumn(endLineFieldName)
	fileExtensionColumn := resultSet.GetColumn(fileExtensionFieldName)
	metadataColumn := resultSet.GetColumn(metadataFieldName)
	splitPartColumn := resultSet.GetColumn(splitPartFieldName)
	// workspaceRoot is only present on conversation-collection result sets, where
	// the search requests the native scalar column. It is nil for code
	// collections and on rows that never carried a workspace root, so reads stay
	// optional and default to empty. loadRules follows the same contract.
	workspaceRootColumn := resultSet.GetColumn(workspaceRootFieldName)
	loadRulesColumn := resultSet.GetColumn(loadRulesFieldName)
	if contentColumn == nil || relativePathColumn == nil || startLineColumn == nil || endLineColumn == nil || fileExtensionColumn == nil {
		return nil, ErrSearchResultIncomplete
	}

	chunks := make([]model.StoredChunk, 0, resultSet.ResultCount)
	for index := range resultSet.ResultCount {
		contentValue, err := contentColumn.GetAsString(index)
		if err != nil {
			slog.Error("read content column failed", "index", index, "err", err)
			return nil, fmt.Errorf("read content column at %d: %w", index, err)
		}
		relativePathValue, err := relativePathColumn.GetAsString(index)
		if err != nil {
			slog.Error("read relative path column failed", "index", index, "err", err)
			return nil, fmt.Errorf("read relative path column at %d: %w", index, err)
		}
		startLineValue, err := startLineColumn.GetAsInt64(index)
		if err != nil {
			slog.Error("read start line column failed", "index", index, "err", err)
			return nil, fmt.Errorf("read start line column at %d: %w", index, err)
		}
		endLineValue, err := endLineColumn.GetAsInt64(index)
		if err != nil {
			slog.Error("read end line column failed", "index", index, "err", err)
			return nil, fmt.Errorf("read end line column at %d: %w", index, err)
		}
		fileExtensionValue, err := fileExtensionColumn.GetAsString(index)
		if err != nil {
			slog.Error("read file extension column failed", "index", index, "err", err)
			return nil, fmt.Errorf("read file extension column at %d: %w", index, err)
		}
		metadataValue := emptyChunkMetadata()
		if metadataColumn != nil {
			rawMetadata, metadataErr := metadataColumn.GetAsString(index)
			if metadataErr == nil {
				metadataValue = decodeMetadata(rawMetadata)
			}
		}

		workspaceRootValue := backfillString(workspaceRootColumn, index)
		loadRulesValue := backfillString(loadRulesColumn, index)
		splitPartValue, splitPartRecorded, splitPartErr := splitPartAt(
			splitPartColumn,
			index,
		)
		if splitPartErr != nil {
			return nil, splitPartErr
		}

		score := 0.0
		if index < len(resultSet.Scores) {
			score = float64(resultSet.Scores[index])
		}
		chunks = append(chunks, model.StoredChunk{
			Content:              contentValue,
			RelativePath:         relativePathValue,
			StartLine:            safeInt32FromInt64(startLineValue),
			EndLine:              safeInt32FromInt64(endLineValue),
			Language:             metadataValue.Language,
			FileExtension:        fileExtensionValue,
			ConversationID:       metadataValue.ConversationID,
			ParentConversationID: metadataValue.ParentConversationID,
			MessageIndex:         metadataValue.messageIndex(),
			Role:                 metadataValue.Role,
			TimestampUnix:        metadataValue.timestampUnix(),
			WorkspaceRoot:        workspaceRootValue,
			Archived:             false,
			SplitPart:            splitPartValue,
			SplitPartRecorded:    splitPartRecorded,
			LoadRules:            loadRulesValue,
			Score:                score,
		})
	}
	return chunks, nil
}

// generateID matches the TS chunk-ID format at packages/core/src/context.ts:1067
// for an unsplit chunk. A split child (SplitPart > 0) folds its split position
// into the hash so identical pieces of repeated oversized content get distinct
// primary keys; an unsplit chunk keeps the original identity so the normal
// single-chunk case is not re-embedded.

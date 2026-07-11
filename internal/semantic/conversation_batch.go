package semantic

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

// ConversationStoredRows is one conversation's stored rows as read from the live
// collection: the assembled per-message base state for the text delta, and the
// exact derived-row identities that prove a specific derived target row exists.
// DerivedPaths maps a derived relativePath to the hex content hash stored at that
// path, so the examination path can key reuse validity on the full identity
// (conversationID, relativePath, contentHash) rather than a content hash alone.
type ConversationStoredRows struct {
	Messages     map[int32]StoredMessageState
	DerivedPaths map[string]string
}

// ConversationBatchState is one batched read of the live conversation collection
// for a set of conversation ids. Rows maps each requested id to its stored rows;
// Reuse is the batch-wide content-hash -> dense-vector map. A missing target row
// can reuse a vector Reuse holds for identical content embedded anywhere in the
// batch, so the row is inserted without re-embedding. Every row is read from the
// one live collection, which is built with the current embedding model, so a
// reused vector is always valid for that model.
type ConversationBatchState struct {
	Rows  map[string]ConversationStoredRows
	Reuse map[string][]float32
}

// conversationBatchIDFilterSize bounds how many conversation ids go into one
// Milvus membership clause, mirroring conversationFilterIDBatchSize, so a large
// bootstrap scope splits across several queries instead of overflowing the
// expression-size limit. A normal ingest scope is one query.
const conversationBatchIDFilterSize = conversationFilterIDBatchSize

// LoadConversationDerivedBatch resolves, in one Milvus query per id batch, the
// stored rows for a set of conversations. It replaces the per-conversation
// message-state iterator in the examination path: the caller computes each
// conversation's expected chunks and diffs them against these stored rows, so a
// batch of conversations costs one query rather than one per conversation. Rows
// that carry no conversationID scalar (legacy pre-scalar rows) are not matched
// and read as absent, so their conversation re-embeds and the message-level
// removal reconciles the stale rows.
func (service *Service) LoadConversationDerivedBatch(ctx context.Context, collectionName string, conversationIDs []string) (ConversationBatchState, error) {
	state := ConversationBatchState{Rows: map[string]ConversationStoredRows{}, Reuse: map[string][]float32{}}
	uniqueIDs := dedupeConversationIDs(conversationIDs)
	if !service.Available() || collectionName == "" || len(uniqueIDs) == 0 {
		return state, nil
	}

	hasCollection, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		slog.ErrorContext(ctx, "check collection for conversation batch load failed", "collection", collectionName, "err", err)
		return ConversationBatchState{}, fmt.Errorf("check Milvus collection %s: %w", collectionName, err)
	}
	if !hasCollection {
		return state, nil
	}
	if err := service.ensureConversationScalarColumnsOnce(ctx, collectionName); err != nil {
		return ConversationBatchState{}, err
	}
	if err := service.loadCollectionForRead(ctx, collectionName); err != nil {
		return ConversationBatchState{}, err
	}

	assemblies := newConversationBatchAssemblies()
	for _, idBatch := range batchConversationIDs(uniqueIDs, conversationBatchIDFilterSize) {
		if err := service.loadConversationBatchGroup(ctx, collectionName, idBatch, assemblies, state.Reuse); err != nil {
			return ConversationBatchState{}, err
		}
	}
	state.Rows = assemblies.finalize()
	slog.DebugContext(
		ctx, "semantic.conversation_derived_batch_loaded",
		"collection", collectionName,
		"conversations", len(uniqueIDs),
		"resolved", len(state.Rows),
		"chunks", len(state.Reuse),
	)
	return state, nil
}

func (service *Service) loadConversationBatchGroup(ctx context.Context, collectionName string, conversationIDs []string, assemblies *conversationBatchAssemblies, reuse map[string][]float32) error {
	if len(conversationIDs) == 0 {
		return nil
	}
	iterator, err := service.milvus.QueryIterator(ctx, milvusclient.NewQueryIteratorOption(collectionName).
		WithBatchSize(reuseVectorBatchSize).
		WithFilter(inStringClause(conversationIDFieldName, conversationIDs)).
		WithOutputFields(conversationIDFieldName, relativePathFieldName, messageIndexFieldName, roleFieldName, contentFieldName, denseVectorFieldName))
	if err != nil {
		slog.ErrorContext(ctx, "open conversation batch query iterator failed", "collection", collectionName, "err", err)
		return fmt.Errorf("open conversation batch iterator for %s: %w", collectionName, err)
	}
	for {
		resultSet, nextErr := iterator.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			return nil
		}
		if nextErr != nil {
			slog.ErrorContext(ctx, "conversation batch query iterator next failed", "collection", collectionName, "err", nextErr)
			return fmt.Errorf("iterate %s for conversation batch: %w", collectionName, nextErr)
		}
		if err := appendConversationBatchRows(resultSet, assemblies, reuse); err != nil {
			return err
		}
	}
}

func appendConversationBatchRows(resultSet milvusclient.ResultSet, assemblies *conversationBatchAssemblies, reuse map[string][]float32) error {
	contentColumn := resultSet.GetColumn(contentFieldName)
	vectorColumn := resultSet.GetColumn(denseVectorFieldName)
	conversationIDColumn := resultSet.GetColumn(conversationIDFieldName)
	relativePathColumn := resultSet.GetColumn(relativePathFieldName)
	if contentColumn == nil || vectorColumn == nil || conversationIDColumn == nil || relativePathColumn == nil {
		return ErrSearchResultIncomplete
	}
	roleColumn := resultSet.GetColumn(roleFieldName)
	messageIndexColumn := resultSet.GetColumn(messageIndexFieldName)

	for rowIndex := range resultSet.ResultCount {
		contentValue, vector, contentErr := conversationContentVectorAt(contentColumn, vectorColumn, rowIndex)
		if contentErr != nil {
			return contentErr
		}
		contentHash := contentVectorKey(contentValue)
		reuse[contentHash] = vector

		conversationID, idErr := conversationIDColumn.GetAsString(rowIndex)
		if idErr != nil {
			slog.Error("read conversation batch id column failed", "index", rowIndex, "err", idErr)
			return fmt.Errorf("read conversation id column at %d: %w", rowIndex, idErr)
		}
		if conversationID == "" {
			continue
		}
		relativePath, relativePathErr := relativePathColumn.GetAsString(rowIndex)
		if relativePathErr != nil {
			slog.Error("read conversation batch relative path column failed", "index", rowIndex, "err", relativePathErr)
			return fmt.Errorf("read relative path column at %d: %w", rowIndex, relativePathErr)
		}
		if isDerivedConversationRelativePath(relativePath) {
			assemblies.addDerived(conversationID, relativePath, contentHash)
			continue
		}
		if err := appendConversationBatchBaseRow(assemblies, conversationID, relativePath, contentValue, roleColumn, messageIndexColumn, rowIndex); err != nil {
			return err
		}
	}
	return nil
}

func appendConversationBatchBaseRow(assemblies *conversationBatchAssemblies, conversationID string, relativePath string, content string, roleColumn column.Column, messageIndexColumn column.Column, rowIndex int) error {
	messageIndex, ok, messageIndexErr := messageIndexAt(messageIndexColumn, rowIndex)
	if messageIndexErr != nil {
		return messageIndexErr
	}
	if !ok {
		// A base row without a messageIndex is a legacy pre-scalar row. It cannot be
		// placed into assembled per-message state, so its conversation reads as
		// absent for that message and re-embeds, which reconciles the stale row.
		return nil
	}
	if roleColumn == nil {
		return ErrSearchResultIncomplete
	}
	role, roleErr := roleColumn.GetAsString(rowIndex)
	if roleErr != nil {
		slog.Error("read conversation batch role column failed", "index", rowIndex, "err", roleErr)
		return fmt.Errorf("read role column at %d: %w", rowIndex, roleErr)
	}
	conversationPrefix := "conv/" + conversationID + "/"
	partIndex, partErr := conversationMessagePartIndex(relativePath, conversationPrefix)
	if partErr != nil {
		slog.Error("read conversation batch part index failed", "index", rowIndex, "err", partErr)
		return fmt.Errorf("read conversation part index at %d: %w", rowIndex, partErr)
	}
	assemblies.addBasePart(conversationID, safeInt32FromInt64(messageIndex), role, partIndex, content)
	return nil
}

// conversationBatchAssemblies accumulates the per-conversation base-message
// assemblies and derived-path identities across every page of a batched read.
type conversationBatchAssemblies struct {
	messages map[string]map[int32]*storedMessageAssembly
	derived  map[string]map[string]string
}

func newConversationBatchAssemblies() *conversationBatchAssemblies {
	return &conversationBatchAssemblies{
		messages: map[string]map[int32]*storedMessageAssembly{},
		derived:  map[string]map[string]string{},
	}
}

func (assemblies *conversationBatchAssemblies) addBasePart(conversationID string, messageIndex int32, role string, partIndex int, content string) {
	conversationMessages := assemblies.messages[conversationID]
	if conversationMessages == nil {
		conversationMessages = map[int32]*storedMessageAssembly{}
		assemblies.messages[conversationID] = conversationMessages
	}
	appendStoredMessagePart(conversationMessages, messageIndex, role, partIndex, content)
}

func (assemblies *conversationBatchAssemblies) addDerived(conversationID string, relativePath string, contentHash string) {
	conversationDerived := assemblies.derived[conversationID]
	if conversationDerived == nil {
		conversationDerived = map[string]string{}
		assemblies.derived[conversationID] = conversationDerived
	}
	conversationDerived[relativePath] = contentHash
}

func (assemblies *conversationBatchAssemblies) finalize() map[string]ConversationStoredRows {
	rows := make(map[string]ConversationStoredRows, len(assemblies.messages))
	for conversationID, conversationMessages := range assemblies.messages {
		rows[conversationID] = ConversationStoredRows{
			Messages:     assembleStoredMessageState(conversationMessages),
			DerivedPaths: assemblies.derived[conversationID],
		}
	}
	for conversationID, conversationDerived := range assemblies.derived {
		if _, found := rows[conversationID]; found {
			continue
		}
		rows[conversationID] = ConversationStoredRows{
			Messages:     map[int32]StoredMessageState{},
			DerivedPaths: conversationDerived,
		}
	}
	return rows
}

func dedupeConversationIDs(conversationIDs []string) []string {
	seen := make(map[string]struct{}, len(conversationIDs))
	unique := make([]string, 0, len(conversationIDs))
	for _, conversationID := range conversationIDs {
		trimmed := strings.TrimSpace(conversationID)
		if trimmed == "" {
			continue
		}
		if _, found := seen[trimmed]; found {
			continue
		}
		seen[trimmed] = struct{}{}
		unique = append(unique, trimmed)
	}
	return unique
}

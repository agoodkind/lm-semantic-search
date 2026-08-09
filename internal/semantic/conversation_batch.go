package semantic

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"slices"
	"strconv"
	"strings"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

// ConversationStoredRows is one conversation's stored rows as read from the live
// collection. DerivedPaths retains every derived-row identity, while
// UsableDerivedPaths contains only paths whose content can satisfy a family.
type ConversationStoredRows struct {
	Messages           map[int32]StoredMessageState
	DerivedPaths       map[string]string
	UsableDerivedPaths map[string]struct{}
}

// ConversationBatchState is one batched read of the live conversation collection
// for a set of conversation ids. Rows maps each requested id to its stored rows;
// Reuse is the batch-wide content-hash -> dense-vector map. A missing target row
// can reuse a vector Reuse holds for identical content embedded anywhere in the
// batch, so the row is inserted without re-embedding. Rows with no recorded
// embedding model remain reusable, while known unequal model names are excluded.
type ConversationBatchState struct {
	Rows  map[string]ConversationStoredRows
	Reuse map[string][]float32
}

// conversationBatchIDFilterSize bounds how many conversation ids go into one
// Milvus membership clause, mirroring conversationFilterIDBatchSize, so a large
// bootstrap scope splits across several queries instead of overflowing the
// expression-size limit. A normal ingest scope is one query.
const conversationBatchIDFilterSize = conversationFilterIDBatchSize

// LoadConversationDerivedBatch resolves stored rows for a set of conversations.
// Each query matches both current conversation scalars and historical family
// paths, so rows written before the scalar columns remain visible.
func (service *Service) LoadConversationDerivedBatch(ctx context.Context, collectionName string, conversationIDs []string) (ConversationBatchState, error) {
	state := ConversationBatchState{Rows: map[string]ConversationStoredRows{}, Reuse: map[string][]float32{}}
	uniqueIDs := dedupeConversationIDs(conversationIDs)
	if !service.Available() || collectionName == "" || len(uniqueIDs) == 0 {
		return state, nil
	}

	hasCollection, err := service.hasCollection(ctx, collectionName, "check Milvus collection "+collectionName)
	if err != nil {
		return ConversationBatchState{}, err
	}
	if !hasCollection {
		return state, nil
	}
	if err := service.ensureConversationScalarColumnsOnce(ctx, collectionName); err != nil {
		return ConversationBatchState{}, err
	}
	if err := service.ensureSplitPartColumnOnce(ctx, collectionName); err != nil {
		return ConversationBatchState{}, err
	}
	if err := service.ensureReuseIdentityColumnsOnce(ctx, collectionName); err != nil {
		return ConversationBatchState{}, err
	}
	lease, err := service.AcquireCollection(ctx, collectionName)
	if err != nil {
		return ConversationBatchState{}, err
	}
	defer lease.Release()

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
		WithFilter(conversationBatchFilterExpression(conversationIDs)).
		WithOutputFields(
			conversationIDFieldName,
			relativePathFieldName,
			messageIndexFieldName,
			roleFieldName,
			contentFieldName,
			embeddingModelFieldName,
			denseVectorFieldName,
			splitPartFieldName,
		))
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
		if err := appendConversationBatchRows(resultSet, conversationIDs, service.cfg.EmbeddingModel, assemblies, reuse); err != nil {
			return err
		}
	}
}

func conversationBatchFilterExpression(conversationIDs []string) string {
	clauses := []string{inStringClause(conversationIDFieldName, conversationIDs)}
	for _, conversationID := range conversationIDs {
		clauses = append(
			clauses,
			relativePathPrefixExpression("conv/"+conversationID+"/"),
			relativePathPrefixExpression("convtool/"+conversationID+"/"),
			relativePathPrefixExpression("convthink/"+conversationID+"/"),
		)
	}
	return "(" + strings.Join(clauses, " or ") + ")"
}

func conversationBatchRowID(
	conversationIDColumn column.Column,
	rowIndex int,
	relativePath string,
	conversationIDs []string,
) (string, error) {
	conversationID, present, err := readOptionalStringAt(conversationIDColumn, rowIndex)
	if err != nil {
		slog.Error("read conversation batch id column failed", "index", rowIndex, "err", err)
		return "", fmt.Errorf("read conversation id column at %d: %w", rowIndex, err)
	}
	if present && conversationID != "" && slices.Contains(conversationIDs, conversationID) {
		return conversationID, nil
	}
	matchedID := ""
	matchedPrefixLength := 0
	for _, requestedID := range conversationIDs {
		prefixes := []string{
			"conv/" + requestedID + "/",
			"convtool/" + requestedID + "/",
			"convthink/" + requestedID + "/",
		}
		for _, prefix := range prefixes {
			if strings.HasPrefix(relativePath, prefix) && len(prefix) > matchedPrefixLength {
				matchedID = requestedID
				matchedPrefixLength = len(prefix)
			}
		}
	}
	return matchedID, nil
}

func conversationBatchMessageIndexAt(
	messageIndexColumn column.Column,
	rowIndex int,
	relativePath string,
	conversationID string,
) (int64, bool, error) {
	messageIndex, present, err := messageIndexAt(messageIndexColumn, rowIndex)
	if err != nil || present {
		return messageIndex, present, err
	}
	prefixes := []string{
		"conv/" + conversationID + "/",
		"convtool/" + conversationID + "/",
		"convthink/" + conversationID + "/",
	}
	for _, prefix := range prefixes {
		remainder, found := strings.CutPrefix(relativePath, prefix)
		if !found {
			continue
		}
		indexText, _, _ := strings.Cut(remainder, "/")
		parsed, parseErr := strconv.ParseInt(indexText, 10, 32)
		if parseErr == nil && parsed >= 0 {
			return parsed, true, nil
		}
		return 0, false, nil
	}
	return 0, false, nil
}

func readOptionalStringAt(valueColumn column.Column, rowIndex int) (string, bool, error) {
	if valueColumn == nil {
		return "", false, nil
	}
	isNull, nullErr := valueColumn.IsNull(rowIndex)
	if nullErr != nil {
		slog.Error("read optional string null state failed", "row", rowIndex, "err", nullErr)
		return "", false, fmt.Errorf("read null state at row %d: %w", rowIndex, nullErr)
	}
	if isNull {
		return "", false, nil
	}
	value, valueErr := valueColumn.GetAsString(rowIndex)
	if valueErr != nil {
		slog.Error("read optional string failed", "row", rowIndex, "err", valueErr)
		return "", false, fmt.Errorf("read string at row %d: %w", rowIndex, valueErr)
	}
	return value, true, nil
}

func appendConversationBatchRows(
	resultSet milvusclient.ResultSet,
	conversationIDs []string,
	currentEmbeddingModel string,
	assemblies *conversationBatchAssemblies,
	reuse map[string][]float32,
) error {
	contentColumn := resultSet.GetColumn(contentFieldName)
	vectorColumn := resultSet.GetColumn(denseVectorFieldName)
	conversationIDColumn := resultSet.GetColumn(conversationIDFieldName)
	relativePathColumn := resultSet.GetColumn(relativePathFieldName)
	if contentColumn == nil || vectorColumn == nil || conversationIDColumn == nil || relativePathColumn == nil {
		return ErrSearchResultIncomplete
	}
	roleColumn := resultSet.GetColumn(roleFieldName)
	messageIndexColumn := resultSet.GetColumn(messageIndexFieldName)
	splitPartColumn := resultSet.GetColumn(splitPartFieldName)
	embeddingModelColumn := resultSet.GetColumn(embeddingModelFieldName)

	for rowIndex := range resultSet.ResultCount {
		contentValue, vector, contentErr := conversationContentVectorAt(contentColumn, vectorColumn, rowIndex)
		if contentErr != nil {
			return contentErr
		}
		contentHash := contentVectorKey(contentValue)
		embeddingModel, modelErr := nullableStringAt(embeddingModelColumn, rowIndex)
		if modelErr != nil {
			return fmt.Errorf("read conversation batch embedding model at %d: %w", rowIndex, modelErr)
		}
		if embeddingModelsCompatible(embeddingModel, currentEmbeddingModel) {
			reuse[contentHash] = vector
		}

		relativePath, relativePathErr := relativePathColumn.GetAsString(rowIndex)
		if relativePathErr != nil {
			slog.Error("read conversation batch relative path column failed", "index", rowIndex, "err", relativePathErr)
			return fmt.Errorf("read relative path column at %d: %w", rowIndex, relativePathErr)
		}
		conversationID, idErr := conversationBatchRowID(
			conversationIDColumn,
			rowIndex,
			relativePath,
			conversationIDs,
		)
		if idErr != nil {
			return idErr
		}
		if conversationID == "" {
			continue
		}
		if isDerivedConversationRelativePath(relativePath) {
			usable := strings.TrimSpace(contentValue) != ""
			assemblies.addDerived(conversationID, relativePath, contentHash, usable)
			if !usable {
				continue
			}
			if err := registerConversationBatchDerivedMessage(
				assemblies,
				conversationID,
				relativePath,
				roleColumn,
				messageIndexColumn,
				rowIndex,
			); err != nil {
				return err
			}
			continue
		}
		if err := appendConversationBatchBaseRow(
			assemblies,
			conversationID,
			relativePath,
			contentValue,
			roleColumn,
			messageIndexColumn,
			splitPartColumn,
			rowIndex,
		); err != nil {
			return err
		}
	}
	return nil
}

// registerConversationBatchDerivedMessage records a usable derived-only message.
// Historical rows recover their message index from the family path.
func registerConversationBatchDerivedMessage(
	assemblies *conversationBatchAssemblies,
	conversationID string,
	relativePath string,
	roleColumn column.Column,
	messageIndexColumn column.Column,
	rowIndex int,
) error {
	messageIndex, ok, messageIndexErr := conversationBatchMessageIndexAt(
		messageIndexColumn,
		rowIndex,
		relativePath,
		conversationID,
	)
	if messageIndexErr != nil {
		return messageIndexErr
	}
	if !ok {
		return nil
	}
	if roleColumn == nil {
		return ErrSearchResultIncomplete
	}
	role, _, roleErr := readOptionalStringAt(roleColumn, rowIndex)
	if roleErr != nil {
		slog.Error("read conversation batch derived role column failed", "index", rowIndex, "err", roleErr)
		return fmt.Errorf("read role column at %d: %w", rowIndex, roleErr)
	}
	assemblies.addDerivedMessage(conversationID, safeInt32FromInt64(messageIndex), role)
	return nil
}

func appendConversationBatchBaseRow(
	assemblies *conversationBatchAssemblies,
	conversationID string,
	relativePath string,
	content string,
	roleColumn column.Column,
	messageIndexColumn column.Column,
	splitPartColumn column.Column,
	rowIndex int,
) error {
	messageIndex, ok, messageIndexErr := conversationBatchMessageIndexAt(
		messageIndexColumn,
		rowIndex,
		relativePath,
		conversationID,
	)
	if messageIndexErr != nil {
		return messageIndexErr
	}
	if !ok {
		return nil
	}
	role, _, roleErr := readOptionalStringAt(roleColumn, rowIndex)
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
	splitPart, splitPartRecorded, splitPartErr := splitPartAt(
		splitPartColumn,
		rowIndex,
	)
	if splitPartErr != nil {
		return splitPartErr
	}
	assemblies.addBasePart(
		conversationID,
		safeInt32FromInt64(messageIndex),
		role,
		partIndex,
		splitPart,
		splitPartRecorded,
		content,
	)
	return nil
}

// conversationBatchAssemblies accumulates the per-conversation base-message
// assemblies and derived-path identities across every page of a batched read.
type conversationBatchAssemblies struct {
	messages      map[string]map[int32]*storedMessageAssembly
	derived       map[string]map[string]string
	usableDerived map[string]map[string]struct{}
}

func newConversationBatchAssemblies() *conversationBatchAssemblies {
	return &conversationBatchAssemblies{
		messages:      map[string]map[int32]*storedMessageAssembly{},
		derived:       map[string]map[string]string{},
		usableDerived: map[string]map[string]struct{}{},
	}
}

func (assemblies *conversationBatchAssemblies) addBasePart(
	conversationID string,
	messageIndex int32,
	role string,
	partIndex int,
	splitPart int32,
	splitPartRecorded bool,
	content string,
) {
	conversationMessages := assemblies.messages[conversationID]
	if conversationMessages == nil {
		conversationMessages = map[int32]*storedMessageAssembly{}
		assemblies.messages[conversationID] = conversationMessages
	}
	appendStoredMessagePart(
		conversationMessages,
		messageIndex,
		role,
		partIndex,
		splitPart,
		splitPartRecorded,
		content,
	)
}

// addDerivedMessage records a message that exists because one of its derived
// rows was read, carrying the role that row holds. It adds no text part, so a
// message with no base row assembles an empty text, which is what the store
// holds for it.
//
// The role is filled only when the assembly has none, so a base row's role wins
// whatever order the rows arrive in.
func (assemblies *conversationBatchAssemblies) addDerivedMessage(
	conversationID string,
	messageIndex int32,
	role string,
) {
	conversationMessages := assemblies.messages[conversationID]
	if conversationMessages == nil {
		conversationMessages = map[int32]*storedMessageAssembly{}
		assemblies.messages[conversationID] = conversationMessages
	}
	markStoredMessageDerived(conversationMessages, messageIndex)
	if assembly := conversationMessages[messageIndex]; assembly != nil && !assembly.roleFromBase {
		assembly.role = role
	}
}

func (assemblies *conversationBatchAssemblies) addDerived(
	conversationID string,
	relativePath string,
	contentHash string,
	usable bool,
) {
	conversationDerived := assemblies.derived[conversationID]
	if conversationDerived == nil {
		conversationDerived = map[string]string{}
		assemblies.derived[conversationID] = conversationDerived
	}
	conversationDerived[relativePath] = contentHash
	if !usable {
		return
	}
	conversationUsable := assemblies.usableDerived[conversationID]
	if conversationUsable == nil {
		conversationUsable = map[string]struct{}{}
		assemblies.usableDerived[conversationID] = conversationUsable
	}
	conversationUsable[relativePath] = struct{}{}
}

func (assemblies *conversationBatchAssemblies) finalize() map[string]ConversationStoredRows {
	rows := make(map[string]ConversationStoredRows, len(assemblies.messages))
	for conversationID, conversationMessages := range assemblies.messages {
		usableDerived := assemblies.usableDerived[conversationID]
		if usableDerived == nil {
			usableDerived = map[string]struct{}{}
		}
		rows[conversationID] = ConversationStoredRows{
			Messages:           assembleStoredMessageState(conversationMessages),
			DerivedPaths:       assemblies.derived[conversationID],
			UsableDerivedPaths: usableDerived,
		}
	}
	for conversationID, conversationDerived := range assemblies.derived {
		if _, found := rows[conversationID]; found {
			continue
		}
		rows[conversationID] = ConversationStoredRows{
			Messages:           map[int32]StoredMessageState{},
			DerivedPaths:       conversationDerived,
			UsableDerivedPaths: map[string]struct{}{},
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

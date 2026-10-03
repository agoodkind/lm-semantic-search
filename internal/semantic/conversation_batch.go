package semantic

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"strconv"
	"strings"

	"goodkind.io/lm-semantic-search/collection"
	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"

	"github.com/milvus-io/milvus/client/v2/column"
	"google.golang.org/grpc/peer"
)

// ConversationStoredRows is one conversation's stored rows as read from the live
// collection. DerivedPaths retains every derived-row identity. UsableDerivedPaths
// contains only the paths with content that can satisfy a family.
type ConversationStoredRows struct {
	Messages           map[int32]StoredMessageState
	DerivedPaths       map[string]string
	UsableDerivedPaths map[string]struct{}
}

// ConversationBatchState is one batched read of the live conversation collection
// for a set of conversation ids. Rows maps each requested id to its stored rows;
// Reuse is the batch-wide content-hash -> dense-vector map. A missing target row
// can reuse a vector from Reuse for identical content embedded anywhere in the
// batch, and the row is inserted without re-embedding. Rows with no recorded
// embedding model remain reusable, while known unequal model names are excluded.
type ConversationBatchState struct {
	Rows  map[string]ConversationStoredRows
	Reuse map[string][]float32
}

// conversationBatchIDFilterSize bounds how many conversation ids go into one
// Milvus membership clause, mirroring conversationFilterIDBatchSize. A large
// bootstrap scope splits across several queries instead of overflowing the
// expression-size limit. A normal ingest scope is one query.
const conversationBatchIDFilterSize = conversationFilterIDBatchSize

// LoadConversationDerivedBatch resolves stored rows for a set of conversations.
// Each query matches both current conversation scalars and historical family
// paths, so rows written before the scalar columns remain visible.
func (service *Service) LoadConversationDerivedBatch(ctx context.Context, collectionName string, conversationIDs []string) (ConversationBatchState, error) {
	peerInfo, _ := peer.FromContext(ctx)
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
		"peer", peerInfo.String(),
	)
	return state, nil
}

// conversationBatchDeclaration is the item ID column and the scalar columns the
// batch read needs: the conversation ID, the role, and the message index.
func conversationBatchDeclaration() collection.Declaration {
	declaration := ConversationDeclaration()
	needed := map[string]struct{}{
		conversationIDFieldName: {},
		roleFieldName:           {},
		messageIndexFieldName:   {},
	}
	scalars := make([]collection.ScalarColumn, 0, len(needed))
	for _, scalar := range declaration.Scalars {
		if _, found := needed[scalar.Name]; found {
			scalars = append(scalars, scalar)
		}
	}
	return collection.Declaration{ItemIDColumn: declaration.ItemIDColumn, Scalars: scalars}
}

func (service *Service) loadConversationBatchGroup(ctx context.Context, collectionName string, conversationIDs []string, assemblies *conversationBatchAssemblies, reuse map[string][]float32) error {
	if len(conversationIDs) == 0 {
		return nil
	}
	prefixes := make([]string, 0, len(conversationIDs)*3)
	for _, conversationID := range conversationIDs {
		prefixes = append(
			prefixes,
			"conv/"+conversationID+"/",
			"convtool/"+conversationID+"/",
			"convthink/"+conversationID+"/",
		)
	}
	rows, err := service.collectionStore().QueryRows(ctx, collection.RowsRequest{
		Collection:    collectionName,
		Declaration:   conversationBatchDeclaration(),
		ItemIDs:       conversationIDs,
		PathPrefixes:  prefixes,
		IncludeVector: true,
	})
	if err != nil {
		slog.ErrorContext(ctx, "load conversation batch rows failed", "collection", collectionName, "err", err)
		return fmt.Errorf("load conversation batch rows from %s: %w", collectionName, err)
	}
	return appendConversationBatchRows(rows, conversationIDs, service.cfg.EmbeddingModel, assemblies, reuse)
}

// cellString returns the string value of a row's scalar cell and whether the
// cell has a value.
func cellString(row collection.StoredRow, columnName string) (string, bool) {
	cell, found := row.Scalars[columnName]
	if !found || cell.State != collection.ScalarCellValue {
		return "", false
	}
	return cell.Value.String, true
}

func conversationBatchRowID(row collection.StoredRow, conversationIDs []string) string {
	conversationID, present := cellString(row, conversationIDFieldName)
	if present && conversationID != "" && slices.Contains(conversationIDs, conversationID) {
		return conversationID
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
			if strings.HasPrefix(row.RelativePath, prefix) && len(prefix) > matchedPrefixLength {
				matchedID = requestedID
				matchedPrefixLength = len(prefix)
			}
		}
	}
	return matchedID
}

// conversationBatchMessageIndex returns the message index of a stored row: the
// messageIndex cell when it has a value, otherwise the index in the family path.
func conversationBatchMessageIndex(row collection.StoredRow, conversationID string) (int64, bool) {
	if cell, found := row.Scalars[messageIndexFieldName]; found && cell.State == collection.ScalarCellValue {
		return cell.Value.Int64, true
	}
	prefixes := []string{
		"conv/" + conversationID + "/",
		"convtool/" + conversationID + "/",
		"convthink/" + conversationID + "/",
	}
	for _, prefix := range prefixes {
		remainder, found := strings.CutPrefix(row.RelativePath, prefix)
		if !found {
			continue
		}
		indexText, _, _ := strings.Cut(remainder, "/")
		parsed, parseErr := strconv.ParseInt(indexText, 10, 32)
		if parseErr == nil && parsed >= 0 {
			return parsed, true
		}
		return 0, false
	}
	return 0, false
}

func appendConversationBatchRows(
	rows []collection.StoredRow,
	conversationIDs []string,
	currentEmbeddingModel string,
	assemblies *conversationBatchAssemblies,
	reuse map[string][]float32,
) error {
	for _, row := range rows {
		if row.Vector == nil {
			return ErrSearchResultIncomplete
		}
		contentHash := contentVectorKey(row.Content)
		if embeddingModelsCompatible(row.EmbeddingModel, currentEmbeddingModel) {
			reuse[contentHash] = row.Vector
		}
		conversationID := conversationBatchRowID(row, conversationIDs)
		if conversationID == "" {
			continue
		}
		if isDerivedConversationRelativePath(row.RelativePath) {
			usable := strings.TrimSpace(row.Content) != ""
			assemblies.addDerived(conversationID, row.RelativePath, contentHash, usable)
			if !usable {
				continue
			}
			registerConversationBatchDerivedMessage(assemblies, conversationID, row)
			continue
		}
		if err := appendConversationBatchBaseRow(assemblies, conversationID, row); err != nil {
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
	row collection.StoredRow,
) {
	messageIndex, ok := conversationBatchMessageIndex(row, conversationID)
	if !ok {
		return
	}
	role, _ := cellString(row, roleFieldName)
	assemblies.addDerivedMessage(conversationID, milvusstore.SafeInt32(messageIndex), role)
}

func appendConversationBatchBaseRow(
	assemblies *conversationBatchAssemblies,
	conversationID string,
	row collection.StoredRow,
) error {
	messageIndex, ok := conversationBatchMessageIndex(row, conversationID)
	if !ok {
		return nil
	}
	role, _ := cellString(row, roleFieldName)
	conversationPrefix := "conv/" + conversationID + "/"
	partIndex, partErr := conversationMessagePartIndex(row.RelativePath, conversationPrefix)
	if partErr != nil {
		slog.Error("read conversation batch part index failed", "relative_path", row.RelativePath, "err", partErr)
		return fmt.Errorf("read conversation part index of %s: %w", row.RelativePath, partErr)
	}
	assemblies.addBasePart(
		conversationID,
		milvusstore.SafeInt32(messageIndex),
		role,
		partIndex,
		row.SplitPart,
		row.SplitPartRecorded,
		row.Content,
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
// rows was read, with the role of that row. It adds no text part. A message with
// no base row assembles an empty text, matching the store.
//
// The role is filled only when the assembly has none. A base row's role wins
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

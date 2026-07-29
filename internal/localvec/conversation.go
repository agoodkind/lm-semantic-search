package localvec

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strconv"
	"strings"

	"goodkind.io/lm-semantic-search/internal/semantic"
)

type conversationPart struct {
	index int
	// splitPart is the piece's position when one stored path was cut again to
	// fit the embedding budget, so several pieces share a relative path and are
	// told apart only by this. Ordering by path alone would concatenate them in
	// whatever order the snapshot returned, and a rebuilt text that differs from
	// the delivered one makes an unchanged message read as changed on every sync.
	splitPart int32
	// splitPartRecorded marks a row written after the position was persisted, so
	// a legacy row that predates the field is not read back as position zero.
	splitPartRecorded bool
	content           string
}

type conversationAssembly struct {
	role string
	// roleFromBase records that role came from a base text row rather than a
	// derived one. Rows arrive in no guaranteed order, and the delta comparison
	// is against the base row, so a base row's role must win whenever the two
	// disagree however late it arrives.
	roleFromBase bool
	// hasDerivedContent records that at least one of this message's derived rows
	// was read, which is how the comparison tells a message that lost its tool
	// call from one that never had one.
	hasDerivedContent bool
	parts             []conversationPart
}

// LoadConversationDerivedBatch loads stored state for a batch of conversations.
func (store *Store) LoadConversationDerivedBatch(
	ctx context.Context,
	collectionName string,
	conversationIDs []string,
) (semantic.ConversationBatchState, error) {
	state := semantic.ConversationBatchState{
		Rows:  map[string]semantic.ConversationStoredRows{},
		Reuse: map[string][]float32{},
	}
	requested := make(map[string]struct{}, len(conversationIDs))
	for _, conversationID := range conversationIDs {
		trimmed := strings.TrimSpace(conversationID)
		if trimmed != "" {
			requested[trimmed] = struct{}{}
		}
	}
	if collectionName == "" || len(requested) == 0 {
		return state, nil
	}
	if err := operationContextError(ctx, "load local conversation batch"); err != nil {
		return semantic.ConversationBatchState{}, err
	}
	stored, err := store.collectionForName(collectionName, false)
	if err != nil {
		return semantic.ConversationBatchState{}, err
	}
	rows, exists, err := stored.snapshot()
	if err != nil {
		return semantic.ConversationBatchState{}, err
	}
	if !exists {
		return state, nil
	}

	return buildConversationBatchState(rows, requested)
}

func buildConversationBatchState(
	rows []row,
	requested map[string]struct{},
) (semantic.ConversationBatchState, error) {
	state := semantic.ConversationBatchState{
		Rows:  map[string]semantic.ConversationStoredRows{},
		Reuse: map[string][]float32{},
	}
	assemblies := make(map[string]map[int32]*conversationAssembly)
	derived := make(map[string]map[string]string)
	for _, candidate := range rows {
		if _, found := requested[candidate.ConversationID]; !found {
			continue
		}
		if err := addConversationBatchRow(
			candidate,
			assemblies,
			derived,
			state.Reuse,
		); err != nil {
			return semantic.ConversationBatchState{}, err
		}
	}
	state.Rows = finalizeConversationBatchRows(assemblies, derived)
	return state, nil
}

func addConversationBatchRow(
	candidate row,
	assemblies map[string]map[int32]*conversationAssembly,
	derived map[string]map[string]string,
	reuse map[string][]float32,
) error {
	contentKey := candidate.ContentVectorKey
	if contentKey == "" {
		contentKey = semantic.ContentVectorKey(candidate.Content)
	}
	reuse[contentKey] = append([]float32(nil), candidate.Vector...)
	if isDerivedConversationPath(candidate.RelativePath) {
		conversationDerived := derived[candidate.ConversationID]
		if conversationDerived == nil {
			conversationDerived = make(map[string]string)
			derived[candidate.ConversationID] = conversationDerived
		}
		conversationDerived[candidate.RelativePath] = contentKey
		// Register the message this derived row belongs to. A turn carrying just
		// a tool call or just reasoning stores no text row, so these rows are all
		// the store holds for it, and a message missing from the assembled state
		// reads as new: the examination path re-sends it and removes these same
		// rows as orphans, on every sync for as long as the conversation exists.
		//
		// The role comes from the row because the comparison rejects a message
		// whose stored role differs from the delivered one.
		assembly := conversationAssemblyFor(assemblies, candidate)
		assembly.hasDerivedContent = true
		if !assembly.roleFromBase {
			assembly.role = candidate.Role
		}
		return nil
	}
	partIndex, err := conversationPartIndex(
		candidate.RelativePath,
		candidate.ConversationID,
	)
	if err != nil {
		return err
	}
	message := conversationAssemblyFor(assemblies, candidate)
	if !message.roleFromBase {
		message.role = candidate.Role
		message.roleFromBase = true
	}
	message.parts = append(
		message.parts,
		conversationPart{
			index:             partIndex,
			splitPart:         candidate.SplitPart,
			splitPartRecorded: candidate.SplitPartRecorded,
			content:           candidate.Content,
		},
	)
	return nil
}

// conversationAssemblyFor returns the assembly for a row's message, creating it
// when this is the first row seen for that message. Both the base and derived
// paths go through it so a message exists in the assembled state whichever kind
// of row named it first.
func conversationAssemblyFor(
	assemblies map[string]map[int32]*conversationAssembly,
	candidate row,
) *conversationAssembly {
	conversationMessages := assemblies[candidate.ConversationID]
	if conversationMessages == nil {
		conversationMessages = make(map[int32]*conversationAssembly)
		assemblies[candidate.ConversationID] = conversationMessages
	}
	message := conversationMessages[candidate.MessageIndex]
	if message == nil {
		message = &conversationAssembly{
			role:              "",
			roleFromBase:      false,
			hasDerivedContent: false,
			parts:             nil,
		}
		conversationMessages[candidate.MessageIndex] = message
	}
	return message
}

func finalizeConversationBatchRows(
	assemblies map[string]map[int32]*conversationAssembly,
	derived map[string]map[string]string,
) map[string]semantic.ConversationStoredRows {
	rows := make(map[string]semantic.ConversationStoredRows)
	for conversationID, messages := range assemblies {
		rows[conversationID] = semantic.ConversationStoredRows{
			Messages:     assembleConversationMessages(messages),
			DerivedPaths: derived[conversationID],
		}
	}
	for conversationID, derivedPaths := range derived {
		if _, found := rows[conversationID]; found {
			continue
		}
		rows[conversationID] = semantic.ConversationStoredRows{
			Messages:     map[int32]semantic.StoredMessageState{},
			DerivedPaths: derivedPaths,
		}
	}
	return rows
}

// DeleteConversation removes one conversation from a collection.
func (store *Store) DeleteConversation(
	ctx context.Context,
	collectionName string,
	conversationID string,
) error {
	if err := operationContextError(ctx, "delete local conversation"); err != nil {
		return err
	}
	trimmedCollectionName := strings.TrimSpace(collectionName)
	if trimmedCollectionName == "" {
		return errors.New("conversation collection name is required")
	}
	trimmedConversationID := strings.TrimSpace(conversationID)
	if trimmedConversationID == "" {
		return errors.New("conversation id is required")
	}
	prefixes := []string{
		"conv/" + trimmedConversationID + "/",
		"convtool/" + trimmedConversationID + "/",
		"convthink/" + trimmedConversationID + "/",
	}
	stored, err := store.collectionForName(trimmedCollectionName, false)
	if err != nil {
		return err
	}
	return stored.rewrite(false, func(rows []row) ([]row, error) {
		kept := make([]row, 0, len(rows))
		for _, candidate := range rows {
			if matchesAnyPrefix(candidate.RelativePath, prefixes) {
				continue
			}
			kept = append(kept, candidate)
		}
		return kept, nil
	})
}

// BackfillConversationEnrichment updates stored conversation enrichment.
func (store *Store) BackfillConversationEnrichment(
	ctx context.Context,
	collectionName string,
	enrichment semantic.ConversationEnrichment,
	dryRun bool,
) (int, int, error) {
	if err := operationContextError(ctx, "backfill local conversation enrichment"); err != nil {
		return 0, 0, err
	}
	if !strings.HasPrefix(collectionName, "conv_chunks_") {
		return 0, 0, fmt.Errorf(
			"workspace backfill: %s is not a conversation collection",
			collectionName,
		)
	}
	stored, err := store.collectionForName(collectionName, false)
	if err != nil {
		return 0, 0, err
	}
	if dryRun {
		rows, exists, snapshotErr := stored.snapshot()
		if snapshotErr != nil {
			return 0, 0, snapshotErr
		}
		if !exists {
			return 0, 0, semantic.ErrCollectionMissing
		}
		changed, orphan := countConversationEnrichment(rows, enrichment)
		return changed, orphan, nil
	}

	changed := 0
	orphan := 0
	err = stored.rewrite(true, func(rows []row) ([]row, error) {
		for index := range rows {
			if rows[index].WorkspaceRoot != "" {
				continue
			}
			value, found := enrichment[rows[index].ConversationID]
			if !found {
				orphan++
				continue
			}
			rows[index].WorkspaceRoot = value.WorkspaceRoot
			rows[index].Archived = value.Archived
			changed++
		}
		return rows, nil
	})
	return changed, orphan, err
}

func countConversationEnrichment(
	rows []row,
	enrichment semantic.ConversationEnrichment,
) (int, int) {
	changed := 0
	orphan := 0
	for _, candidate := range rows {
		if candidate.WorkspaceRoot != "" {
			continue
		}
		if _, found := enrichment[candidate.ConversationID]; found {
			changed++
		} else {
			orphan++
		}
	}
	return changed, orphan
}

func isDerivedConversationPath(relativePath string) bool {
	return strings.HasPrefix(relativePath, "convtool/") ||
		strings.HasPrefix(relativePath, "convthink/")
}

func conversationPartIndex(relativePath string, conversationID string) (int, error) {
	prefix := "conv/" + conversationID + "/"
	remainder, found := strings.CutPrefix(relativePath, prefix)
	if !found {
		return 0, fmt.Errorf(
			"relative path %q is outside conversation %q",
			relativePath,
			conversationID,
		)
	}
	parts := strings.Split(remainder, "/")
	if len(parts) == 0 || parts[0] == "" {
		return 0, fmt.Errorf("relative path %q has no message index", relativePath)
	}
	messageIndex, err := strconv.ParseInt(parts[0], 10, 32)
	if err != nil {
		slog.Error(
			"parse local conversation message index failed",
			"relative_path",
			relativePath,
			"err",
			err,
		)
		return 0, fmt.Errorf("parse message index from %q: %w", remainder, err)
	}
	if messageIndex < 0 {
		return 0, fmt.Errorf("negative message index %d", messageIndex)
	}
	if len(parts) == 1 {
		return 0, nil
	}
	if len(parts) != 2 || parts[1] == "" {
		return 0, fmt.Errorf(
			"relative path %q has an invalid message part",
			relativePath,
		)
	}
	partIndex, err := strconv.Atoi(parts[1])
	if err != nil || partIndex < 0 {
		return 0, fmt.Errorf("parse message part index from %q", remainder)
	}
	return partIndex, nil
}

// conversationPartPrecedes orders two pieces of one message's stored text. It
// must be a total order, and it must match the Milvus loader's exactly: both
// rebuild a message's text by concatenating its pieces, and a text that rebuilds
// differently from the one delivered makes an unchanged message read as changed
// on every sync for as long as the conversation exists.
//
// Every key is consulted before falling through to the next, so two pieces are
// treated as equal only when they are indistinguishable. Stopping at the split
// position would leave two pieces sharing one position ordered by whatever order
// the store returned them in, and the two stores do not return rows in the same
// order.
func conversationPartPrecedes(left conversationPart, right conversationPart) bool {
	if left.index != right.index {
		return left.index < right.index
	}
	if left.splitPartRecorded != right.splitPartRecorded {
		return left.splitPartRecorded
	}
	if left.splitPartRecorded && left.splitPart != right.splitPart {
		return left.splitPart < right.splitPart
	}
	return left.content < right.content
}

func assembleConversationMessages(
	assemblies map[int32]*conversationAssembly,
) map[int32]semantic.StoredMessageState {
	messages := make(map[int32]semantic.StoredMessageState, len(assemblies))
	for messageIndex, assembly := range assemblies {
		// This ordering matches the Milvus loader's exactly. Both rebuild a
		// message's text by concatenating its parts, and a text that rebuilds
		// differently from the one delivered makes an unchanged message read as
		// changed on every sync for as long as the conversation exists, so the
		// two loaders cannot order parts differently.
		sort.SliceStable(assembly.parts, func(left int, right int) bool {
			return conversationPartPrecedes(assembly.parts[left], assembly.parts[right])
		})
		var text strings.Builder
		for _, part := range assembly.parts {
			text.WriteString(part.content)
		}
		messages[messageIndex] = semantic.StoredMessageState{
			Role:              assembly.role,
			Text:              text.String(),
			HasDerivedContent: assembly.hasDerivedContent,
		}
	}
	return messages
}

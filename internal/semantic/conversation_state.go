package semantic

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"sort"
	"strconv"
	"strings"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

// StoredMessageState is the stored text and role for one delivered conversation
// message as currently represented in Milvus.
type StoredMessageState struct {
	Role string
	Text string
}

type storedMessagePart struct {
	index   int
	content string
}

type storedMessageAssembly struct {
	role  string
	parts []storedMessagePart
}

// LoadConversationMessageState reads rows for one conversation prefix and
// returns both assembled per-message state and row-granular reuse vectors.
func (service *Service) LoadConversationMessageState(ctx context.Context, collectionName string, conversationPrefix string) (map[int32]StoredMessageState, map[string][]float32, error) {
	state := make(map[int32]StoredMessageState)
	reuse := make(map[string][]float32)
	if !service.Available() || collectionName == "" || conversationPrefix == "" {
		return state, reuse, nil
	}

	hasCollection, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		slog.ErrorContext(ctx, "check collection for conversation state load failed", "collection", collectionName, "err", err)
		return nil, nil, fmt.Errorf("check Milvus collection %s: %w", collectionName, err)
	}
	if !hasCollection {
		return state, reuse, nil
	}
	if err := service.ensureConversationScalarColumnsOnce(ctx, collectionName); err != nil {
		return nil, nil, err
	}
	if err := service.loadCollectionForRead(ctx, collectionName); err != nil {
		return nil, nil, err
	}

	iterator, err := service.milvus.QueryIterator(ctx, milvusclient.NewQueryIteratorOption(collectionName).
		WithBatchSize(reuseVectorBatchSize).
		WithFilter(relativePathPrefixExpression(conversationPrefix)).
		WithOutputFields(relativePathFieldName, messageIndexFieldName, roleFieldName, contentFieldName, denseVectorFieldName))
	if err != nil {
		slog.ErrorContext(ctx, "open conversation state query iterator failed", "collection", collectionName, "err", err)
		return nil, nil, fmt.Errorf("open conversation state iterator for %s: %w", collectionName, err)
	}

	state, reuse, err = loadConversationMessageStateFromIterator(ctx, collectionName, conversationPrefix, iterator)
	if err != nil {
		return nil, nil, err
	}
	slog.DebugContext(
		ctx, "semantic.conversation_message_state_loaded",
		"collection", collectionName,
		"prefix", conversationPrefix,
		"messages", len(state),
		"chunks", len(reuse),
	)
	return state, reuse, nil
}

func loadConversationMessageStateFromIterator(ctx context.Context, collectionName string, conversationPrefix string, iterator milvusclient.QueryIterator) (map[int32]StoredMessageState, map[string][]float32, error) {
	assemblies := make(map[int32]*storedMessageAssembly)
	reuse := make(map[string][]float32)
	legacyRows := 0

	for {
		resultSet, nextErr := iterator.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			slog.ErrorContext(ctx, "conversation state query iterator next failed", "collection", collectionName, "err", nextErr)
			return nil, nil, fmt.Errorf("iterate %s for conversation state: %w", collectionName, nextErr)
		}
		pageLegacyRows, err := appendConversationMessageStateRows(resultSet, conversationPrefix, assemblies, reuse)
		if err != nil {
			return nil, nil, err
		}
		legacyRows += pageLegacyRows
	}

	if legacyRows > 0 {
		slog.WarnContext(
			ctx, "semantic.conversation_message_state_legacy_rows",
			"collection", collectionName,
			"prefix", conversationPrefix,
			"legacy_rows", legacyRows,
		)
	}
	return assembleStoredMessageState(assemblies), reuse, nil
}

func appendConversationMessageStateRows(resultSet milvusclient.ResultSet, conversationPrefix string, assemblies map[int32]*storedMessageAssembly, reuse map[string][]float32) (int, error) {
	contentColumn := resultSet.GetColumn(contentFieldName)
	vectorColumn := resultSet.GetColumn(denseVectorFieldName)
	if contentColumn == nil || vectorColumn == nil {
		return 0, ErrSearchResultIncomplete
	}

	relativePathColumn := resultSet.GetColumn(relativePathFieldName)
	roleColumn := resultSet.GetColumn(roleFieldName)
	messageIndexColumn := resultSet.GetColumn(messageIndexFieldName)
	legacyRows := 0
	for rowIndex := range resultSet.ResultCount {
		contentValue, contentErr := contentColumn.GetAsString(rowIndex)
		if contentErr != nil {
			slog.Error("read conversation state content column failed", "index", rowIndex, "err", contentErr)
			return legacyRows, fmt.Errorf("read content column at %d: %w", rowIndex, contentErr)
		}
		vector, vectorErr := vectorAt(vectorColumn, rowIndex)
		if vectorErr != nil {
			slog.Error("read conversation state vector column failed", "index", rowIndex, "err", vectorErr)
			return legacyRows, fmt.Errorf("read vector column at %d: %w", rowIndex, vectorErr)
		}
		reuse[contentVectorKey(contentValue)] = vector

		messageIndex, ok, messageIndexErr := messageIndexAt(messageIndexColumn, rowIndex)
		if messageIndexErr != nil {
			return legacyRows, messageIndexErr
		}
		if !ok {
			legacyRows++
			continue
		}
		if relativePathColumn == nil || roleColumn == nil {
			return legacyRows, ErrSearchResultIncomplete
		}
		relativePath, relativePathErr := relativePathColumn.GetAsString(rowIndex)
		if relativePathErr != nil {
			slog.Error("read conversation state relative path column failed", "index", rowIndex, "err", relativePathErr)
			return legacyRows, fmt.Errorf("read relative path column at %d: %w", rowIndex, relativePathErr)
		}
		role, roleErr := roleColumn.GetAsString(rowIndex)
		if roleErr != nil {
			slog.Error("read conversation state role column failed", "index", rowIndex, "err", roleErr)
			return legacyRows, fmt.Errorf("read role column at %d: %w", rowIndex, roleErr)
		}
		partIndex, partErr := conversationMessagePartIndex(relativePath, conversationPrefix)
		if partErr != nil {
			slog.Error("read conversation state part index failed", "index", rowIndex, "err", partErr)
			return legacyRows, fmt.Errorf("read conversation part index at %d: %w", rowIndex, partErr)
		}
		appendStoredMessagePart(assemblies, safeInt32FromInt64(messageIndex), role, partIndex, contentValue)
	}
	return legacyRows, nil
}

func messageIndexAt(messageIndexColumn column.Column, rowIndex int) (int64, bool, error) {
	if messageIndexColumn == nil {
		return 0, false, nil
	}
	isNull, nullErr := messageIndexColumn.IsNull(rowIndex)
	if nullErr != nil {
		slog.Error("read conversation state messageIndex null state failed", "index", rowIndex, "err", nullErr)
		return 0, false, fmt.Errorf("read messageIndex null state at %d: %w", rowIndex, nullErr)
	}
	if isNull {
		return 0, false, nil
	}
	messageIndex, messageIndexErr := messageIndexColumn.GetAsInt64(rowIndex)
	if messageIndexErr != nil {
		slog.Error("read conversation state messageIndex column failed", "index", rowIndex, "err", messageIndexErr)
		return 0, false, fmt.Errorf("read messageIndex column at %d: %w", rowIndex, messageIndexErr)
	}
	return messageIndex, true, nil
}

func appendStoredMessagePart(assemblies map[int32]*storedMessageAssembly, messageIndex int32, role string, partIndex int, content string) {
	assembly := assemblies[messageIndex]
	if assembly == nil {
		assembly = &storedMessageAssembly{role: "", parts: nil}
		assemblies[messageIndex] = assembly
	}
	if assembly.role == "" {
		assembly.role = role
	}
	assembly.parts = append(assembly.parts, storedMessagePart{index: partIndex, content: content})
}

func assembleStoredMessageState(assemblies map[int32]*storedMessageAssembly) map[int32]StoredMessageState {
	state := make(map[int32]StoredMessageState, len(assemblies))
	for messageIndex, assembly := range assemblies {
		sort.SliceStable(assembly.parts, func(left int, right int) bool {
			return assembly.parts[left].index < assembly.parts[right].index
		})
		var text strings.Builder
		for _, part := range assembly.parts {
			text.WriteString(part.content)
		}
		state[messageIndex] = StoredMessageState{Role: assembly.role, Text: text.String()}
	}
	return state
}

func conversationMessagePartIndex(relativePath string, conversationPrefix string) (int, error) {
	prefix := strings.TrimRight(conversationPrefix, "/")
	remainder, ok := strings.CutPrefix(relativePath, prefix+"/")
	if !ok {
		err := fmt.Errorf("relative path %q is outside prefix %q", relativePath, conversationPrefix)
		slog.Error("conversation state relative path outside prefix", "relative_path", relativePath, "prefix", conversationPrefix, "err", err)
		return 0, err
	}
	parts := strings.Split(remainder, "/")
	if len(parts) == 0 || parts[0] == "" {
		err := fmt.Errorf("relative path %q has no message index after prefix %q", relativePath, conversationPrefix)
		slog.Error("conversation state relative path missing message index", "relative_path", relativePath, "prefix", conversationPrefix, "err", err)
		return 0, err
	}
	messageIndex, err := strconv.ParseInt(parts[0], 10, 32)
	if err != nil {
		slog.Error("conversation state relative path message index invalid", "relative_path", relativePath, "remainder", remainder, "err", err)
		return 0, fmt.Errorf("parse message index from %q: %w", remainder, err)
	}
	if messageIndex < 0 {
		err := fmt.Errorf("negative message index %d", messageIndex)
		slog.Error("conversation state relative path message index invalid", "relative_path", relativePath, "remainder", remainder, "err", err)
		return 0, fmt.Errorf("parse message index from %q: %w", remainder, err)
	}
	if len(parts) == 1 {
		return 0, nil
	}
	if len(parts) != 2 || parts[1] == "" {
		err := fmt.Errorf("relative path %q has invalid message part path %q", relativePath, remainder)
		slog.Error("conversation state relative path part path invalid", "relative_path", relativePath, "remainder", remainder, "err", err)
		return 0, err
	}
	partIndex, err := strconv.Atoi(parts[1])
	if err != nil {
		slog.Error("conversation state relative path part index invalid", "relative_path", relativePath, "remainder", remainder, "err", err)
		return 0, fmt.Errorf("parse message part index from %q: %w", remainder, err)
	}
	if partIndex < 0 {
		err := fmt.Errorf("negative message part index %d", partIndex)
		slog.Error("conversation state relative path part index invalid", "relative_path", relativePath, "remainder", remainder, "err", err)
		return 0, fmt.Errorf("parse message part index from %q: %w", remainder, err)
	}
	return partIndex, nil
}

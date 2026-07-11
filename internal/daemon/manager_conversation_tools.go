package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"unicode/utf8"

	"goodkind.io/gksyntax/chunk"
	"goodkind.io/gksyntax/shelldecomp"
	"goodkind.io/lm-semantic-search/internal/model"
)

// derivedPipelineVersion must be bumped whenever conversation tool or thinking
// chunking changes, so a reexamine run rebuilds content derived by older logic.
const derivedPipelineVersion = "1"

var conversationToolExtensions = map[string]string{
	"bash":     ".bash",
	"json":     ".json",
	"markdown": ".md",
}

func newConversationToolDispatcher() *chunk.Dispatcher {
	return chunk.NewDispatcher()
}

func newConversationStoredChunk(document model.ConversationDocument, conversationID string, parentConversationID string, relativePath string, content string, language string, startLine int32, endLine int32) model.StoredChunk {
	return model.StoredChunk{
		Content:              content,
		RelativePath:         relativePath,
		StartLine:            startLine,
		EndLine:              endLine,
		Language:             language,
		FileExtension:        "",
		ConversationID:       conversationID,
		ParentConversationID: parentConversationID,
		MessageIndex:         document.MessageIndex,
		Role:                 document.Role,
		TimestampUnix:        document.TimestampUnix,
		WorkspaceRoot:        document.WorkspaceRoot,
		Archived:             document.Archived,
		Score:                0,
	}
}

func splitConversationToolPayload(ctx context.Context, dispatcher *chunk.Dispatcher, document model.ConversationDocument, conversationID string, parentConversationID string, relativePathPrefix string, splitPath string, content string) ([]model.StoredChunk, error) {
	splitResult, err := dispatcher.SplitFileWithType(ctx, splitPath, []byte(content), "")
	if err != nil {
		slog.ErrorContext(ctx, "split conversation tool payload failed", "relative_path_prefix", relativePathPrefix, "err", err)
		return nil, fmt.Errorf("split conversation tool payload %s: %w", relativePathPrefix, err)
	}
	chunks := make([]model.StoredChunk, 0, len(splitResult.Chunks))
	for partIndex, splitChunk := range splitResult.Chunks {
		chunks = append(chunks, newConversationStoredChunk(
			document,
			conversationID,
			parentConversationID,
			fmt.Sprintf("%s/%d", relativePathPrefix, partIndex),
			splitChunk.Content,
			splitChunk.Language,
			safeInt32(splitChunk.StartLine),
			safeInt32(splitChunk.EndLine),
		))
	}
	return chunks, nil
}

func splitConversationDerivedContent(document model.ConversationDocument, conversationID string, parentConversationID string, relativePath string, content string) []model.StoredChunk {
	pieces := splitConversationText(content)
	chunks := make([]model.StoredChunk, 0, len(pieces))
	multipart := len(pieces) > 1
	for partIndex, piece := range pieces {
		chunkRelativePath := relativePath
		if multipart {
			chunkRelativePath = fmt.Sprintf("%s/%d", relativePath, partIndex)
		}
		chunks = append(chunks, newConversationStoredChunk(
			document,
			conversationID,
			parentConversationID,
			chunkRelativePath,
			piece,
			"",
			0,
			0,
		))
	}
	return chunks
}

func conversationToolTokenContent(toolCall model.ConversationToolCall) string {
	tokens := make([]string, 0)
	appendConversationToken(&tokens, toolCall.Name)
	command := strings.TrimSpace(toolCall.Command)
	if command != "" {
		appendConversationShellTokens(&tokens, command)
	}
	if toolCall.InputJSON != "" {
		appendConversationToken(&tokens, truncateConversationToolSummary(toolCall.InputJSON))
	}
	return strings.Join(tokens, "\n")
}

// appendConversationShellTokens parses a shell command with gksyntax shelldecomp
// and appends its program names and read/write file targets as searchable tokens.
// A command shelldecomp cannot parse (opaque) falls back to the raw command text,
// and a parse that yields no tokens keeps the raw command so the tool call stays
// searchable.
func appendConversationShellTokens(tokens *[]string, command string) {
	decomposition := shelldecomp.Parse(command, "/", "")
	if decomposition == nil || decomposition.IsOpaque() {
		appendConversationToken(tokens, truncateConversationToolSummary(command))
		return
	}
	tokenCount := len(*tokens)
	for _, shellCommand := range decomposition.Commands() {
		appendConversationToken(tokens, shellCommand.Argv0)
	}
	for _, readTarget := range decomposition.ReadTargets() {
		appendConversationShellTarget(tokens, readTarget.Resolvable, readTarget.Path, readTarget.Raw)
	}
	for _, writeTarget := range decomposition.WriteTargets() {
		appendConversationShellTarget(tokens, writeTarget.Resolvable, writeTarget.Path, writeTarget.Raw)
	}
	if len(*tokens) == tokenCount {
		appendConversationToken(tokens, truncateConversationToolSummary(command))
	}
}

// appendConversationShellTarget keeps the resolved absolute path and the raw
// token when they differ, since commands are decomposed from cwd "/".
func appendConversationShellTarget(tokens *[]string, resolvable bool, path string, raw string) {
	if resolvable {
		appendConversationToken(tokens, path)
		if strings.TrimSpace(raw) != "" && strings.TrimSpace(raw) != strings.TrimSpace(path) {
			appendConversationToken(tokens, raw)
		}
		return
	}
	appendConversationToken(tokens, raw)
}

func appendConversationToken(tokens *[]string, value string) {
	trimmedValue := strings.TrimSpace(value)
	if trimmedValue == "" {
		return
	}
	*tokens = append(*tokens, trimmedValue)
}

func truncateConversationToolSummary(value string) string {
	return truncateUTF8Bytes(value, conversationToolSummaryMaxBytes)
}

func truncateUTF8Bytes(value string, maxBytes int) string {
	if len(value) <= maxBytes {
		return value
	}
	end := maxBytes
	for end > 0 && !utf8.RuneStart(value[end]) {
		end--
	}
	if end == 0 {
		return ""
	}
	return value[:end]
}

func conversationFullRemovalPrefixes(conversationID string) []string {
	return []string{
		conversationRelativePathPrefix(conversationID),
		conversationToolRelativePathPrefix(conversationID),
		conversationThinkingRelativePathPrefix(conversationID),
	}
}

func conversationToolRelativePathPrefix(conversationID string) string {
	return "convtool/" + conversationID + "/"
}

func conversationThinkingRelativePathPrefix(conversationID string) string {
	return "convthink/" + conversationID + "/"
}

func conversationToolMessagePath(conversationID string, messageIndex int32) string {
	return fmt.Sprintf("convtool/%s/%d", conversationID, messageIndex)
}

func conversationToolCallPath(conversationID string, messageIndex int32, toolIndex int) string {
	return fmt.Sprintf("%s/%d", conversationToolMessagePath(conversationID, messageIndex), toolIndex)
}

func conversationThinkingPath(conversationID string, messageIndex int32) string {
	return fmt.Sprintf("convthink/%s/%d", conversationID, messageIndex)
}

func conversationToolExtension(langHint string) string {
	extension, found := conversationToolExtensions[strings.ToLower(strings.TrimSpace(langHint))]
	if !found {
		return ""
	}
	return extension
}

package daemon

import (
	"fmt"
	"slices"
	"strings"

	"goodkind.io/gksyntax/shelldecomp"
	"goodkind.io/lm-semantic-search/internal/model"
)

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
		SplitPart:            0,
		SplitPartRecorded:    true,
		LoadRules:            document.LoadRules,
		Score:                0,
	}
}

func splitConversationDerivedContent(document model.ConversationDocument, conversationID string, parentConversationID string, relativePath string, content string, chunkByteBudget ...int) []model.StoredChunk {
	return appendStorableConversationField(
		nil,
		content,
		resolveConversationChunkBudget(chunkByteBudget),
		func(piece string, partIndex int, multipart bool) model.StoredChunk {
			chunkRelativePath := relativePath
			if multipart {
				chunkRelativePath = fmt.Sprintf("%s/%d", relativePath, partIndex)
			}
			return newConversationStoredChunk(
				document,
				conversationID,
				parentConversationID,
				chunkRelativePath,
				piece,
				"",
				0,
				0,
			)
		},
	)
}

func conversationToolContent(toolCall model.ConversationToolCall) string {
	tokens := make([]string, 0)
	appendConversationToken(&tokens, toolCall.Name)
	display := strings.TrimSpace(toolCall.Display)
	if display != "" && toolCall.LangHint == "bash" {
		appendConversationShellTokens(&tokens, display)
	}
	appendConversationToken(&tokens, toolCall.Display)
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
		appendConversationToken(tokens, command)
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
		appendConversationToken(tokens, command)
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

// appendConversationToken adds one searchable token to a tool row, skipping a
// value the row already holds.
//
// A shell decomposition and the display text beside it produce the same string
// whenever the command is one word, since its program name is the whole command,
// and whenever the parser reports the command opaque, since the decomposition
// then falls back to the command text. One row holding that string twice is the
// duplication a tool call's single row exists to remove.
func appendConversationToken(tokens *[]string, value string) {
	trimmedValue := strings.TrimSpace(value)
	if trimmedValue == "" {
		return
	}
	if slices.Contains(*tokens, trimmedValue) {
		return
	}
	*tokens = append(*tokens, trimmedValue)
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

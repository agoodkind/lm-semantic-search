package daemon

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"goodkind.io/gksyntax/shelldecomp"
	"goodkind.io/lm-semantic-search/internal/model"
)

// conversationPartNaming selects how one split names the rows it stores.
type conversationPartNaming int

const (
	// conversationPartNamingBareWhenSingle keeps an unsplit piece at the base
	// relative path and appends /<part> only once the content actually splits.
	// Message text, reasoning, and the derived token and command rows use it.
	conversationPartNamingBareWhenSingle conversationPartNaming = iota
	// conversationPartNamingAlwaysIndexed appends /<part> to every piece,
	// including an unsplit one, so tool input and output rows keep the stored
	// path scheme the collection already holds.
	conversationPartNamingAlwaysIndexed
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
		Score:                0,
	}
}

// splitConversationContent is the one path from a conversation string to stored
// rows. Every conversation family divides the same way: message text, reasoning,
// the derived tool token and command rows, and the tool input and output
// payloads all cut on the manager's configured byte budget, so a tool payload is
// bounded by the same embedder-derived cap as the message text beside it.
//
// A tool payload is data a provider serialized, not a source file, so it gets no
// parse tree. Dividing a serialized payload on syntax boundaries produces pieces
// that carry only the punctuation between two values, because a structural
// boundary is not a boundary between two retrievable things the way a function
// or a class is. Cutting on the byte budget cannot produce such a piece, and it
// needs no knowledge of any provider's payload format.
func splitConversationContent(document model.ConversationDocument, conversationID string, parentConversationID string, relativePath string, content string, naming conversationPartNaming, chunkByteBudget ...int) []model.StoredChunk {
	pieces := splitConversationText(content, chunkByteBudget...)
	indexed := naming == conversationPartNamingAlwaysIndexed || len(pieces) > 1
	chunks := make([]model.StoredChunk, 0, len(pieces))
	for partIndex, piece := range pieces {
		chunkRelativePath := relativePath
		if indexed {
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

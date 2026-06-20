package semantic

import (
	"encoding/json"

	"goodkind.io/lm-semantic-search/internal/model"
)

// chunkMetadata mirrors the JSON shape the TS adapter writes into the
// Milvus `metadata` field. The Go daemon adds a language hint so search
// results can resurface the splitter-derived language without a dedicated
// column. Conversation collections also carry these attributes as native
// scalar columns for filtering; the JSON copy stays for backward
// compatibility with rows written before those columns existed.
type chunkMetadata struct {
	Language             string `json:"language,omitempty"`
	ConversationID       string `json:"conversation_id,omitempty"`
	ParentConversationID string `json:"parent_conversation_id,omitempty"`
	MessageIndex         *int32 `json:"message_index,omitempty"`
	Role                 string `json:"role,omitempty"`
	TimestampUnix        *int64 `json:"timestamp_unix,omitempty"`
}

func encodeMetadata(chunk model.StoredChunk) string {
	if chunk.Language == "" &&
		chunk.ConversationID == "" &&
		chunk.ParentConversationID == "" &&
		chunk.MessageIndex == 0 &&
		chunk.Role == "" &&
		chunk.TimestampUnix == 0 {
		return "{}"
	}
	metadata := chunkMetadata{
		Language:             chunk.Language,
		ConversationID:       chunk.ConversationID,
		ParentConversationID: chunk.ParentConversationID,
		MessageIndex:         nil,
		Role:                 chunk.Role,
		TimestampUnix:        nil,
	}
	if hasConversationMetadata(chunk) {
		messageIndex := chunk.MessageIndex
		timestampUnix := chunk.TimestampUnix
		metadata.MessageIndex = &messageIndex
		metadata.TimestampUnix = &timestampUnix
	}
	encoded, err := json.Marshal(metadata)
	if err != nil {
		return "{}"
	}
	return string(encoded)
}

func hasConversationMetadata(chunk model.StoredChunk) bool {
	return chunk.ConversationID != "" ||
		chunk.MessageIndex != 0 ||
		chunk.Role != "" ||
		chunk.TimestampUnix != 0
}

func decodeMetadataLanguage(metadata string) string {
	return decodeMetadata(metadata).Language
}

func decodeMetadata(metadata string) chunkMetadata {
	if metadata == "" {
		return emptyChunkMetadata()
	}
	var parsed chunkMetadata
	if err := json.Unmarshal([]byte(metadata), &parsed); err != nil {
		return emptyChunkMetadata()
	}
	return parsed
}

func emptyChunkMetadata() chunkMetadata {
	return chunkMetadata{
		Language:             "",
		ConversationID:       "",
		ParentConversationID: "",
		MessageIndex:         nil,
		Role:                 "",
		TimestampUnix:        nil,
	}
}

func (metadata chunkMetadata) messageIndex() int32 {
	if metadata.MessageIndex == nil {
		return 0
	}
	return *metadata.MessageIndex
}

func (metadata chunkMetadata) timestampUnix() int64 {
	if metadata.TimestampUnix == nil {
		return 0
	}
	return *metadata.TimestampUnix
}

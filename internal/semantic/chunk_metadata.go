package semantic

import (
	"encoding/json"

	"goodkind.io/lm-semantic-search/internal/model"
)

type chunkMetadata struct {
	Language string `json:"language,omitempty"`
}

func encodeMetadata(chunk model.StoredChunk) string {
	if chunk.Language == "" {
		return "{}"
	}
	metadata := chunkMetadata{Language: chunk.Language}
	encoded, err := json.Marshal(metadata)
	if err != nil {
		return "{}"
	}
	return string(encoded)
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
		Language: "",
	}
}

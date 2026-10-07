package semantic

import (
	"encoding/json"

	"goodkind.io/lm-semantic-search/internal/model"
)

// chunkMetadata is the JSON the TypeScript adapter writes to the Milvus
// `metadata` field, plus a language hint from the Go daemon. Decoding ignores
// keys that earlier versions wrote.
type chunkMetadata struct {
	Language string `json:"language,omitempty"`
}

func encodeMetadata(chunk model.StoredChunk) string {
	if chunk.Language == "" {
		return "{}"
	}
	encoded, err := json.Marshal(chunkMetadata{Language: chunk.Language})
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

package semantic

import (
	"encoding/json"

	"goodkind.io/lm-semantic-search/internal/model"
)

// chunkMetadata mirrors the JSON shape the TS adapter writes into the
// Milvus `metadata` field. The Go daemon adds a language hint so search
// results can resurface the splitter-derived language without a dedicated
// column. Rows written by earlier versions can carry more keys, and decoding
// ignores them.
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

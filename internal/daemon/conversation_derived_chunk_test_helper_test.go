package daemon

import (
	"strings"

	"goodkind.io/lm-semantic-search/internal/model"
)

// isDerivedConversationChunk reports whether a chunk is a tool or thinking row
// rather than a message's own text row. Production no longer asks this: a
// message's derived rows are matched by presence, not by shape. The tests still
// need it to assemble a stored-row fixture from freshly generated chunks.
func isDerivedConversationChunk(chunk model.StoredChunk) bool {
	return strings.HasPrefix(chunk.RelativePath, "convtool/") || strings.HasPrefix(chunk.RelativePath, "convthink/")
}

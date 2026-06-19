package semantic

import (
	"strings"

	"goodkind.io/lm-semantic-search/internal/model"
)

// conversationScalarColumns accumulates the native scalar column values for one
// insert batch into a conversation collection, so Milvus can pre-filter a
// search by them. It is disabled for code collections, where the columns do not
// exist, in which case append and the collected slices are no-ops. workspaceRoot
// is populated once clyde sends it on each document; until then the column reads
// null on freshly inserted rows.
type conversationScalarColumns struct {
	enabled               bool
	conversationIDs       []string
	parentConversationIDs []string
	roles                 []string
	providers             []string
	workspaceRoots        []string
	timestamps            []int64
	messageIndexes        []int64
}

func newConversationScalarColumns(enabled bool, capacity int) conversationScalarColumns {
	if !enabled {
		return conversationScalarColumns{
			enabled:               false,
			conversationIDs:       nil,
			parentConversationIDs: nil,
			roles:                 nil,
			providers:             nil,
			workspaceRoots:        nil,
			timestamps:            nil,
			messageIndexes:        nil,
		}
	}
	return conversationScalarColumns{
		enabled:               true,
		conversationIDs:       make([]string, 0, capacity),
		parentConversationIDs: make([]string, 0, capacity),
		roles:                 make([]string, 0, capacity),
		providers:             make([]string, 0, capacity),
		workspaceRoots:        make([]string, 0, capacity),
		timestamps:            make([]int64, 0, capacity),
		messageIndexes:        make([]int64, 0, capacity),
	}
}

func (columns *conversationScalarColumns) append(chunk model.StoredChunk) {
	if !columns.enabled {
		return
	}
	columns.conversationIDs = append(columns.conversationIDs, chunk.ConversationID)
	columns.parentConversationIDs = append(columns.parentConversationIDs, chunk.ParentConversationID)
	columns.roles = append(columns.roles, strings.ToLower(chunk.Role))
	columns.providers = append(columns.providers, providerFromConversationID(chunk.ConversationID))
	columns.workspaceRoots = append(columns.workspaceRoots, chunk.WorkspaceRoot)
	columns.timestamps = append(columns.timestamps, chunk.TimestampUnix)
	columns.messageIndexes = append(columns.messageIndexes, int64(chunk.MessageIndex))
}

// providerFromConversationID returns the provider encoded as the prefix of a
// clyde conversation id (claude:<id> -> "claude", codex:<id> -> "codex"). An id
// with no provider separator yields the empty string, which no provider filter
// matches.
func providerFromConversationID(conversationID string) string {
	separator := strings.IndexByte(conversationID, ':')
	if separator <= 0 {
		return ""
	}
	return conversationID[:separator]
}

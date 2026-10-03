package semantic

import (
	"strings"

	"goodkind.io/lm-semantic-search/collection"
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
	archiveds             []bool
	timestamps            []int64
	messageIndexes        []int64
	loadRules             []string
}

// Column names of [ConversationDeclaration].
const (
	ConversationIDColumn            = conversationIDFieldName
	ConversationParentColumn        = parentConversationIDFieldName
	ConversationRoleColumn          = roleFieldName
	ConversationProviderColumn      = providerFieldName
	ConversationWorkspaceRootColumn = workspaceRootFieldName
	ConversationArchivedColumn      = archivedFieldName
	ConversationTimestampColumn     = timestampUnixFieldName
	ConversationMessageIndexColumn  = messageIndexFieldName
	ConversationLoadRulesColumn     = loadRulesFieldName
)

// ProviderFromConversationID returns the provider a conversation collection
// stores for a conversation id: the prefix before the first colon, or empty.
func ProviderFromConversationID(conversationID string) string {
	return providerFromConversationID(conversationID)
}

// ConversationDeclaration returns the scalar declaration of every conversation
// collection. conversationId stores the item id. Every column is nullable, so
// the same declaration defines a freshly created collection and the
// AddCollectionField migration onto a collection with existing rows. The
// column order and string maximum lengths define the stored Milvus schema.
func ConversationDeclaration() collection.Declaration {
	return collection.Declaration{
		ItemIDColumn: conversationIDFieldName,
		Scalars: []collection.ScalarColumn{
			nullableStringColumn(conversationIDFieldName, conversationIDFieldMaxLength),
			nullableStringColumn(parentConversationIDFieldName, conversationIDFieldMaxLength),
			nullableStringColumn(roleFieldName, conversationRoleFieldMaxLength),
			nullableStringColumn(providerFieldName, conversationProviderMaxLength),
			nullableStringColumn(workspaceRootFieldName, conversationWorkspaceMaxLength),
			{Name: archivedFieldName, Type: collection.ScalarTypeBool, Nullable: true, MaxLength: 0},
			{Name: timestampUnixFieldName, Type: collection.ScalarTypeInt64, Nullable: true, MaxLength: 0},
			{Name: messageIndexFieldName, Type: collection.ScalarTypeInt64, Nullable: true, MaxLength: 0},
			nullableStringColumn(loadRulesFieldName, conversationLoadRulesMaxLength),
		},
	}
}

func nullableStringColumn(name string, maxLength int32) collection.ScalarColumn {
	return collection.ScalarColumn{
		Name:      name,
		Type:      collection.ScalarTypeString,
		Nullable:  true,
		MaxLength: maxLength,
	}
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
			archiveds:             nil,
			timestamps:            nil,
			messageIndexes:        nil,
			loadRules:             nil,
		}
	}
	return conversationScalarColumns{
		enabled:               true,
		conversationIDs:       make([]string, 0, capacity),
		parentConversationIDs: make([]string, 0, capacity),
		roles:                 make([]string, 0, capacity),
		providers:             make([]string, 0, capacity),
		workspaceRoots:        make([]string, 0, capacity),
		archiveds:             make([]bool, 0, capacity),
		timestamps:            make([]int64, 0, capacity),
		messageIndexes:        make([]int64, 0, capacity),
		loadRules:             make([]string, 0, capacity),
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
	columns.archiveds = append(columns.archiveds, chunk.Archived)
	columns.timestamps = append(columns.timestamps, chunk.TimestampUnix)
	columns.messageIndexes = append(columns.messageIndexes, int64(chunk.MessageIndex))
	columns.loadRules = append(columns.loadRules, chunk.LoadRules)
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

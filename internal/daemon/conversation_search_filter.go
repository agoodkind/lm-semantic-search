package daemon

import (
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// conversationSearchFilter narrows conversation retrieval by row attributes.
// On the Milvus path every dimension except MinScore is pushed down as a native
// scalar-column expression (see toSemanticFilter), so the vector search returns
// the true top-K among matching rows. MinScore is a post-filter because it is
// the retrieval score, not stored data.
type conversationSearchFilter struct {
	Providers            []string
	WorkspaceRoots       []string
	Roles                []string
	FromUnix             int64
	UntilUnix            int64
	ConversationIDs      []string
	ParentConversationID string
	MinScore             float64
	MessageIndexFrom     int32
	MessageIndexUntil    int32
	Archived             *bool
}

// toSemanticFilter maps the request filter onto the engine's native scalar
// filter. MinScore is intentionally excluded: it is applied as a post-filter on
// the returned score, not as a column expression.
func (filter conversationSearchFilter) toSemanticFilter() semantic.ConversationFilter {
	return semantic.ConversationFilter{
		Providers:            filter.Providers,
		WorkspaceRoots:       filter.WorkspaceRoots,
		Roles:                filter.Roles,
		ConversationIDs:      filter.ConversationIDs,
		ParentConversationID: filter.ParentConversationID,
		FromUnix:             filter.FromUnix,
		UntilUnix:            filter.UntilUnix,
		MessageIndexFrom:     filter.MessageIndexFrom,
		MessageIndexUntil:    filter.MessageIndexUntil,
		Archived:             filter.Archived,
	}
}

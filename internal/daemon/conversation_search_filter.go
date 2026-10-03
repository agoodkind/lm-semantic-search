package daemon

import (
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// conversationSearchFilter narrows conversation retrieval by row attributes.
// Every dimension except MinScore converts to the typed collection filter tree
// (see collectionSearchRequest), and the store applies the tree natively. The
// vector search therefore returns the true top-K among matching rows. MinScore
// is the generic search's score floor because it is the retrieval score, not
// stored data.
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

// toSemanticFilter maps the request filter onto the engine's conversation
// filter. MinScore is intentionally excluded: the search applies it to the
// returned score, not as a column expression.
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

// collectionSearchRequest converts a conversation search to the generic
// collection search. An empty filter converts to no filter tree, which matches
// every row. A positive perConversationLimit becomes a per-group cap on the
// conversationId column.
func (filter conversationSearchFilter) collectionSearchRequest(collectionID string, query string, limit int32, perConversationLimit int32) CollectionSearchRequest {
	groupBy := ""
	perGroupLimit := int32(0)
	if perConversationLimit > 0 {
		groupBy = semantic.ConversationDeclaration().ItemIDColumn
		perGroupLimit = perConversationLimit
	}
	return CollectionSearchRequest{
		CollectionID:  collectionID,
		Query:         query,
		Limit:         limit,
		MinScore:      filter.MinScore,
		Filter:        filter.toSemanticFilter().CollectionFilter(),
		GroupBy:       groupBy,
		PerGroupLimit: perGroupLimit,
	}
}

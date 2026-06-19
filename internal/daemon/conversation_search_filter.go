package daemon

import (
	"slices"
	"strings"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// conversationSearchFilter narrows conversation retrieval by row attributes.
// On the Milvus path every dimension except MinScore is pushed down as a native
// scalar-column expression (see toSemanticFilter), so the vector search returns
// the true top-K among matching rows. On the literal chunk-cache fallback, where
// there is no pushdown, matches applies the same conditions per row before
// ranking. MinScore is always a post-filter because it is the retrieval score,
// not stored data.
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
	}
}

// matchesScope reports whether one chunk satisfies every set condition except
// MinScore. The literal cache fallback applies it BEFORE ranking, so the scope
// pre-filters the candidate set rather than truncating an unfiltered ranking.
// MinScore is excluded here because a chunk's Score is only set by ranking.
func (filter conversationSearchFilter) matchesScope(chunk model.StoredChunk) bool {
	if len(filter.Providers) > 0 && !slices.Contains(filter.Providers, providerFromConversationID(chunk.ConversationID)) {
		return false
	}
	if len(filter.WorkspaceRoots) > 0 && !slices.Contains(filter.WorkspaceRoots, chunk.WorkspaceRoot) {
		return false
	}
	if len(filter.ConversationIDs) > 0 && !slices.Contains(filter.ConversationIDs, chunk.ConversationID) {
		return false
	}
	if len(filter.Roles) > 0 && !containsRole(filter.Roles, chunk.Role) {
		return false
	}
	if filter.FromUnix > 0 && chunk.TimestampUnix < filter.FromUnix {
		return false
	}
	if filter.UntilUnix > 0 && chunk.TimestampUnix >= filter.UntilUnix {
		return false
	}
	if filter.ParentConversationID != "" && chunk.ParentConversationID != filter.ParentConversationID {
		return false
	}
	if filter.MessageIndexFrom > 0 && chunk.MessageIndex < filter.MessageIndexFrom {
		return false
	}
	if filter.MessageIndexUntil > 0 && chunk.MessageIndex >= filter.MessageIndexUntil {
		return false
	}
	return true
}

// matches reports whether one retrieved chunk satisfies every set condition,
// including MinScore. Applied after ranking has set the score.
func (filter conversationSearchFilter) matches(chunk model.StoredChunk) bool {
	if !filter.matchesScope(chunk) {
		return false
	}
	if filter.MinScore > 0 && chunk.Score < filter.MinScore {
		return false
	}
	return true
}

// providerFromConversationID returns the provider encoded as the prefix of a
// clyde conversation id (claude:<id> -> "claude"). Empty when the id has no
// provider separator.
func providerFromConversationID(conversationID string) string {
	separator := strings.IndexByte(conversationID, ':')
	if separator <= 0 {
		return ""
	}
	return conversationID[:separator]
}

// containsRole matches a chunk role against the filter set case-insensitively,
// since providers differ in role casing.
func containsRole(roles []string, role string) bool {
	for _, candidate := range roles {
		if strings.EqualFold(candidate, role) {
			return true
		}
	}
	return false
}

// applyConversationSearchFilter keeps matching chunks in retrieval order, caps
// hits per conversation when perConversationLimit is positive, and truncates
// to limit. Retrieval order is relevance order, so the cap keeps each
// conversation's best hits.
func applyConversationSearchFilter(chunks []model.StoredChunk, filter conversationSearchFilter, perConversationLimit int32, limit int32) []model.StoredChunk {
	kept := make([]model.StoredChunk, 0, len(chunks))
	perConversation := make(map[string]int32)
	for _, chunk := range chunks {
		if !filter.matches(chunk) {
			continue
		}
		if perConversationLimit > 0 {
			if perConversation[chunk.ConversationID] >= perConversationLimit {
				continue
			}
			perConversation[chunk.ConversationID]++
		}
		kept = append(kept, chunk)
		if limit > 0 && len(kept) >= int(limit) {
			break
		}
	}
	return kept
}

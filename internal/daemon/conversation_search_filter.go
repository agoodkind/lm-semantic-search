package daemon

import (
	"slices"
	"strings"

	"goodkind.io/lm-semantic-search/internal/model"
)

// conversationSearchFilter narrows conversation retrieval by row attributes.
// ConversationIDs additionally pushes into the vector store as path-prefix
// scopes; every condition also applies to the decoded chunk after retrieval,
// because role, timestamp, and message index live in the JSON metadata column
// and are not filterable in Milvus, and the literal chunk-cache fallback has
// no pushdown at all.
type conversationSearchFilter struct {
	Roles                []string
	FromUnix             int64
	UntilUnix            int64
	ConversationIDs      []string
	ParentConversationID string
	MinScore             float64
	MessageIndexFrom     int32
	MessageIndexUntil    int32
}

// hasRowConditions reports whether any condition applies per row after
// retrieval, which is what makes fetching more than the requested limit
// worthwhile before filtering.
func (filter conversationSearchFilter) hasRowConditions() bool {
	return len(filter.Roles) > 0 ||
		filter.FromUnix > 0 ||
		filter.UntilUnix > 0 ||
		len(filter.ConversationIDs) > 0 ||
		filter.ParentConversationID != "" ||
		filter.MinScore > 0 ||
		filter.MessageIndexFrom > 0 ||
		filter.MessageIndexUntil > 0
}

// matches reports whether one retrieved chunk satisfies every set condition.
// Time and index bounds are inclusive at from and exclusive at until.
func (filter conversationSearchFilter) matches(chunk model.StoredChunk) bool {
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
	if filter.MinScore > 0 && chunk.Score < filter.MinScore {
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

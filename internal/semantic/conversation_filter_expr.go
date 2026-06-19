package semantic

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"goodkind.io/lm-semantic-search/internal/model"
)

// conversationFilterIDBatchSize bounds how many conversation ids go into one
// Milvus `in [...]` membership clause, so a very large explicit conversation
// scope never overflows the expression-size limit. Larger scopes are split
// across several searches whose results are merged by score.
const conversationFilterIDBatchSize = 256

// ConversationFilter carries the native-filterable attributes of a conversation
// search. The daemon maps its request filter onto this, and buildExpr renders a
// Milvus boolean expression over the conversation scalar columns so the vector
// search pre-filters by every dimension before ranking, instead of the engine
// over-fetching and post-filtering. min_score is intentionally absent: it is the
// retrieval score, not stored data, so the caller applies it as a post-filter.
type ConversationFilter struct {
	Providers            []string
	WorkspaceRoots       []string
	Roles                []string
	ConversationIDs      []string
	ParentConversationID string
	FromUnix             int64
	UntilUnix            int64
	MessageIndexFrom     int32
	MessageIndexUntil    int32
}

// HasConversationScope reports whether the filter restricts retrieval to a
// specific set of conversation ids, which the caller uses to decide whether a
// large id set needs batching across several searches.
func (filter ConversationFilter) HasConversationScope() bool {
	return len(filter.ConversationIDs) > 0
}

// buildExpr renders the Milvus boolean expression for every native dimension,
// ANDing whichever clauses are present. An empty result searches the whole
// collection. Role values are lowercased to match the lowercased role column,
// so role filtering is case-insensitive across providers.
func (filter ConversationFilter) buildExpr() string {
	clauses := make([]string, 0, 9)
	if clause := inStringClause(providerFieldName, filter.Providers); clause != "" {
		clauses = append(clauses, clause)
	}
	if clause := inStringClause(workspaceRootFieldName, filter.WorkspaceRoots); clause != "" {
		clauses = append(clauses, clause)
	}
	if clause := inStringClause(roleFieldName, lowercaseAll(filter.Roles)); clause != "" {
		clauses = append(clauses, clause)
	}
	if clause := inStringClause(conversationIDFieldName, filter.ConversationIDs); clause != "" {
		clauses = append(clauses, clause)
	}
	if filter.ParentConversationID != "" {
		clauses = append(clauses, fmt.Sprintf(`%s == "%s"`, parentConversationIDFieldName, escapeMilvusString(filter.ParentConversationID)))
	}
	if filter.FromUnix > 0 {
		clauses = append(clauses, fmt.Sprintf("%s >= %d", timestampUnixFieldName, filter.FromUnix))
	}
	if filter.UntilUnix > 0 {
		clauses = append(clauses, fmt.Sprintf("%s < %d", timestampUnixFieldName, filter.UntilUnix))
	}
	if filter.MessageIndexFrom > 0 {
		clauses = append(clauses, fmt.Sprintf("%s >= %d", messageIndexFieldName, filter.MessageIndexFrom))
	}
	if filter.MessageIndexUntil > 0 {
		clauses = append(clauses, fmt.Sprintf("%s < %d", messageIndexFieldName, filter.MessageIndexUntil))
	}
	return strings.Join(clauses, " and ")
}

// inStringClause renders a Milvus `field in ["a", "b"]` membership clause, each
// value escaped for use inside a double-quoted Milvus string literal. An empty
// value set contributes no clause.
func inStringClause(field string, values []string) string {
	if len(values) == 0 {
		return ""
	}
	quoted := make([]string, 0, len(values))
	for _, value := range values {
		quoted = append(quoted, `"`+escapeMilvusString(value)+`"`)
	}
	return field + " in [" + strings.Join(quoted, ", ") + "]"
}

func lowercaseAll(values []string) []string {
	if len(values) == 0 {
		return values
	}
	lowered := make([]string, 0, len(values))
	for _, value := range values {
		lowered = append(lowered, strings.ToLower(value))
	}
	return lowered
}

// searchConversationBatched runs the native-filtered vector search. When the
// explicit conversation scope is larger than one membership clause can hold, it
// splits the ids across several searches and merges the hits by score, so a
// large scope never overflows the Milvus expression-size limit. The common case
// (no or small scope) is a single search.
func (service *Service) searchConversationBatched(ctx context.Context, collectionName string, query string, limit int32, filter ConversationFilter) ([]model.StoredChunk, error) {
	batches := batchConversationIDs(filter.ConversationIDs, conversationFilterIDBatchSize)
	if len(batches) <= 1 {
		return service.searchCollection(ctx, collectionName, query, limit, filter.buildExpr())
	}
	merged := make([]model.StoredChunk, 0, len(batches)*int(maxInt32(limit, 10)))
	for _, batch := range batches {
		batchFilter := filter
		batchFilter.ConversationIDs = batch
		chunks, err := service.searchCollection(ctx, collectionName, query, limit, batchFilter.buildExpr())
		if err != nil {
			return nil, err
		}
		merged = append(merged, chunks...)
	}
	sort.SliceStable(merged, func(first int, second int) bool {
		return merged[first].Score > merged[second].Score
	})
	if limit > 0 && len(merged) > int(limit) {
		merged = merged[:limit]
	}
	return merged, nil
}

// batchConversationIDs splits ids into chunks of at most size, returning a
// single empty batch when ids is empty so callers run exactly one unscoped
// search.
func batchConversationIDs(ids []string, size int) [][]string {
	if len(ids) == 0 {
		return [][]string{nil}
	}
	if size <= 0 || len(ids) <= size {
		return [][]string{ids}
	}
	batches := make([][]string, 0, (len(ids)+size-1)/size)
	for start := 0; start < len(ids); start += size {
		end := min(start+size, len(ids))
		batches = append(batches, ids[start:end])
	}
	return batches
}

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

// conversationSearchWindowMax bounds the paged cap-fill to the Milvus
// single-search ceiling, where offset + limit must stay under 16384.
// Conversation limits never approach this, so the bound is reached only when a
// corpus cannot supply enough per-conversation-distinct matches, in which case
// the honest partial result is returned.
const conversationSearchWindowMax = 16384

type conversationSearchFunc func(ctx context.Context, collectionName string, query string, limit int32, expr string) ([]model.StoredChunk, error)

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
	return searchConversationBatchedWith(ctx, collectionName, query, limit, filter, conversationFilterIDBatchSize, service.searchCollection)
}

func searchConversationBatchedWith(ctx context.Context, collectionName string, query string, limit int32, filter ConversationFilter, batchSize int, search conversationSearchFunc) ([]model.StoredChunk, error) {
	batches := batchConversationIDs(filter.ConversationIDs, batchSize)
	if len(batches) <= 1 {
		return search(ctx, collectionName, query, limit, filter.buildExpr())
	}
	merged := make([]model.StoredChunk, 0, len(batches)*int(maxInt32(limit, 10)))
	for _, batch := range batches {
		batchFilter := filter
		batchFilter.ConversationIDs = batch
		chunks, err := search(ctx, collectionName, query, limit, batchFilter.buildExpr())
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

// conversationPageSearch fetches one ranked page of at most pageLimit rows
// starting at offset, in descending score order. fillCappedConversationSearchWith
// drives it until the per-conversation cap yields the requested limit.
type conversationPageSearch func(ctx context.Context, offset int, pageLimit int) ([]model.StoredChunk, error)

// fillCappedConversationSearch pages the ranked search by offset, reusing one
// precomputed query vector across pages, and reduces each accumulated window
// until the per-conversation cap yields limit survivors. The common case fills
// on the first page and runs exactly one search.
func (service *Service) fillCappedConversationSearch(ctx context.Context, collectionName string, queryVector []float32, rawQuery string, filterExpr string, limit int32, perConversationLimit int32, minScore float64) ([]model.StoredChunk, error) {
	return fillCappedConversationSearchWith(ctx, limit, perConversationLimit, minScore, func(ctx context.Context, offset int, pageLimit int) ([]model.StoredChunk, error) {
		return service.searchCollectionWithVector(ctx, collectionName, queryVector, rawQuery, pageLimit, offset, filterExpr)
	})
}

// fillCappedConversationSearchWith pages pageSearch by offset and reduces each
// accumulated window with capConversationChunks until limit survivors are
// collected, the score frontier drops below minScore, the search is exhausted
// (an empty or short page), or the 16384 window is reached. It returns the honest
// partial result when the corpus cannot supply enough per-conversation-distinct
// matches.
func fillCappedConversationSearchWith(ctx context.Context, limit int32, perConversationLimit int32, minScore float64, pageSearch conversationPageSearch) ([]model.StoredChunk, error) {
	pageSize := int(limit)
	if pageSize <= 0 {
		pageSize = 10
	}
	buffer := make([]model.StoredChunk, 0, pageSize*2)
	survivors := make([]model.StoredChunk, 0, limit)
	for offset := 0; offset < conversationSearchWindowMax; offset += pageSize {
		pageLimit := pageSize
		if offset+pageLimit > conversationSearchWindowMax {
			pageLimit = conversationSearchWindowMax - offset
		}
		if pageLimit <= 0 {
			break
		}
		page, err := pageSearch(ctx, offset, pageLimit)
		if err != nil {
			return nil, err
		}
		if len(page) == 0 {
			break
		}
		buffer = append(buffer, page...)
		survivors = capConversationChunks(buffer, perConversationLimit, minScore, limit)
		if len(survivors) >= int(limit) {
			break
		}
		// Results descend by score, so once a page ends below the floor no later
		// page can clear it.
		if minScore > 0 && page[len(page)-1].Score < minScore {
			break
		}
		// A short page means the ranked results are exhausted.
		if len(page) < pageLimit {
			break
		}
	}
	return survivors, nil
}

// capConversationChunks keeps score-ordered chunks above minScore, at most
// perConversationLimit per conversation, up to limit total. It is the store-up
// reduction: every scope dimension is already enforced natively by the Milvus
// filter expression, so only the cap and the score floor apply here. It mirrors
// the daemon's applyConversationSearchFilter, which still reduces the literal
// cache fallback where there is no native pushdown.
func capConversationChunks(chunks []model.StoredChunk, perConversationLimit int32, minScore float64, limit int32) []model.StoredChunk {
	kept := make([]model.StoredChunk, 0, len(chunks))
	perConversation := make(map[string]int32)
	for _, chunk := range chunks {
		if minScore > 0 && chunk.Score < minScore {
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

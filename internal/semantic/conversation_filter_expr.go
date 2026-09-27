package semantic

import (
	"fmt"
	"sort"
	"strings"
)

// conversationFilterIDBatchSize bounds how many conversation ids go into one
// Milvus `in [...]` membership clause on the stored-row load path. A larger id
// set on that path runs one query per batch. The search path does not batch.
const conversationFilterIDBatchSize = 256

// conversationRankingDepth is the Milvus topK ceiling. Both hybrid legs, the
// fused hybrid limit, and the dense search rank this many candidates, which
// keeps the ranking independent of the requested limit and cap.
const conversationRankingDepth = 16384

// conversationIDsTemplateParam is the Milvus expression template parameter
// that binds the conversation id scope as a typed array. The array avoids the
// expression-text size limit.
const conversationIDsTemplateParam = "conversation_ids"

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
	// Archived, when non-nil, keeps only rows whose archived column equals the
	// pointed-to value. It filters on the nullable archived scalar, so a row
	// whose archived is still NULL (an old row not yet reached by the enrichment
	// backfill) is excluded by either value; callers should send it only once
	// the backfill has populated archived across the corpus.
	Archived *bool
}

// HasConversationScope reports whether the filter restricts retrieval to a
// specific set of conversation ids.
func (filter ConversationFilter) HasConversationScope() bool {
	return len(filter.ConversationIDs) > 0
}

// buildExpr renders the Milvus boolean expression for every native dimension,
// ANDing whichever clauses are present. An empty result searches the whole
// collection. Role values are lowercased to match the lowercased role column.
// Role filtering is therefore case-insensitive across providers. The
// conversation id scope renders as the conversationIDsTemplateParam
// placeholder, and the search supplies the ids as that template parameter.
func (filter ConversationFilter) buildExpr() string {
	clauses := make([]string, 0, 10)
	if clause := inStringClause(providerFieldName, filter.Providers); clause != "" {
		clauses = append(clauses, clause)
	}
	if clause := inStringClause(workspaceRootFieldName, filter.WorkspaceRoots); clause != "" {
		clauses = append(clauses, clause)
	}
	if clause := inStringClause(roleFieldName, lowercaseAll(filter.Roles)); clause != "" {
		clauses = append(clauses, clause)
	}
	if filter.HasConversationScope() {
		clauses = append(clauses, conversationIDFieldName+" in {"+conversationIDsTemplateParam+"}")
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
	if filter.Archived != nil {
		clauses = append(clauses, fmt.Sprintf("%s == %t", archivedFieldName, *filter.Archived))
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

// rankedCandidate is one row of a conversation search's fused ranking. It
// stores only the row identity, the cap group, and the score.
type rankedCandidate struct {
	PrimaryKey     string
	RelativePath   string
	ConversationID string
	// ConversationIDNull is true when the stored conversationId column is null.
	// ConversationID must be resolved from the row's metadata JSON before the
	// per-conversation cap applies.
	ConversationIDNull bool
	Score              float64
}

// sortRankedCandidates orders candidates by descending score, then ascending
// relativePath, then ascending primary key. The order is total.
func sortRankedCandidates(candidates []rankedCandidate) {
	sort.Slice(candidates, func(first int, second int) bool {
		left := candidates[first]
		right := candidates[second]
		if left.Score != right.Score {
			return left.Score > right.Score
		}
		if left.RelativePath != right.RelativePath {
			return left.RelativePath < right.RelativePath
		}
		return left.PrimaryKey < right.PrimaryKey
	})
}

// selectRankedCandidates walks sorted candidates once. It drops a candidate
// scoring below minScore, keeps at most perConversationLimit candidates per
// conversation, and stops at limit. A zero perConversationLimit is uncapped,
// and a zero minScore is no floor. A smaller limit returns a prefix of a
// larger limit's result, which search paging relies on.
func selectRankedCandidates(candidates []rankedCandidate, perConversationLimit int32, minScore float64, limit int32) []rankedCandidate {
	kept := make([]rankedCandidate, 0, min(len(candidates), int(max(limit, 0))))
	perConversation := make(map[string]int32)
	for _, candidate := range candidates {
		if limit > 0 && len(kept) >= int(limit) {
			break
		}
		if minScore > 0 && candidate.Score < minScore {
			continue
		}
		if perConversationLimit > 0 {
			if perConversation[candidate.ConversationID] >= perConversationLimit {
				continue
			}
			perConversation[candidate.ConversationID]++
		}
		kept = append(kept, candidate)
	}
	return kept
}

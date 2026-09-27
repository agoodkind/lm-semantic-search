package semantic

import (
	"strings"
)

// conversationFilterIDBatchSize bounds how many conversation ids go into one
// Milvus `in [...]` membership clause on the stored-row load path. A larger id
// set there runs one query per batch. Search does not batch: it binds every
// membership set as one expression template parameter.
const conversationFilterIDBatchSize = 256

// conversationFilterDimensionCount is the number of conversation filter
// dimensions CollectionFilter can convert: providers, workspace roots, roles,
// conversation ids, parent, both timestamp bounds, both message index bounds,
// and archived.
const conversationFilterDimensionCount = 10

// ConversationFilter is the native-filterable attributes of a conversation
// search. The daemon maps its request filter onto this, and CollectionFilter
// converts it to the typed filter tree over the conversation scalar columns.
// The vector search pre-filters by every dimension before ranking, instead of
// the engine over-fetching and post-filtering. min_score is intentionally
// absent. It is the retrieval score, not stored data, and the search applies it
// after ranking.
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
	// Archived, when non-nil, keeps only rows with an archived column equal to
	// the pointed-to value. It filters on the nullable archived scalar. A row
	// with a NULL archived value (an old row the enrichment backfill has not
	// updated) is excluded by either value. Callers should send it only once the
	// backfill has populated archived across the corpus.
	Archived *bool
}

// HasConversationScope reports whether the filter restricts retrieval to a
// specific set of conversation ids, which the caller uses to decide whether a
// large id set needs batching across several searches.
func (filter ConversationFilter) HasConversationScope() bool {
	return len(filter.ConversationIDs) > 0
}

// CollectionFilter converts the conversation filter to the typed filter tree
// over the conversation scalar columns. It returns nil when no dimension is
// set, which searches the whole collection. Every set dimension becomes one
// child of a top-level all node in a fixed order: providers, workspace roots,
// roles, conversation ids, parent, timestamp bounds, message index bounds, and
// archived. Role values are lowercased to match the lowercased role column, so
// role filtering is case-insensitive across providers. From bounds are
// inclusive and until bounds are exclusive. An archived value compares the
// nullable archived column, which excludes a row with a null archived value
// for either value.
func (filter ConversationFilter) CollectionFilter() *CollectionFilter {
	children := make([]CollectionFilter, 0, conversationFilterDimensionCount)
	if len(filter.Providers) > 0 {
		children = append(children, ColumnIn(providerFieldName, StringValues(filter.Providers)))
	}
	if len(filter.WorkspaceRoots) > 0 {
		children = append(children, ColumnIn(workspaceRootFieldName, StringValues(filter.WorkspaceRoots)))
	}
	if len(filter.Roles) > 0 {
		children = append(children, ColumnIn(roleFieldName, StringValues(lowercaseAll(filter.Roles))))
	}
	if len(filter.ConversationIDs) > 0 {
		children = append(children, ColumnIn(conversationIDFieldName, StringValues(filter.ConversationIDs)))
	}
	if filter.ParentConversationID != "" {
		children = append(children, ColumnEquals(parentConversationIDFieldName, StringScalar(filter.ParentConversationID)))
	}
	if filter.FromUnix > 0 {
		lower := filter.FromUnix
		children = append(children, ColumnRange(timestampUnixFieldName, &lower, nil))
	}
	if filter.UntilUnix > 0 {
		upper := filter.UntilUnix
		children = append(children, ColumnRange(timestampUnixFieldName, nil, &upper))
	}
	if filter.MessageIndexFrom > 0 {
		lower := int64(filter.MessageIndexFrom)
		children = append(children, ColumnRange(messageIndexFieldName, &lower, nil))
	}
	if filter.MessageIndexUntil > 0 {
		upper := int64(filter.MessageIndexUntil)
		children = append(children, ColumnRange(messageIndexFieldName, nil, &upper))
	}
	if filter.Archived != nil {
		children = append(children, ColumnEquals(archivedFieldName, BoolScalar(*filter.Archived)))
	}
	if len(children) == 0 {
		return nil
	}
	tree := AllOf(children...)
	return &tree
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

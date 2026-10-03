package collection

import (
	"context"

	"goodkind.io/lm-semantic-search/internal/adapterr"
)

// ErrCollectionMissing reports that the semantic collection does not exist yet.
var ErrCollectionMissing error = &adapterr.AdapterError{
	Class:         adapterr.ClassCollectionMissing,
	Message:       "semantic collection is missing",
	Code:          "collection_missing",
	Hint:          "re-run index_codebase to recreate the collection",
	Cause:         nil,
	SafeForClient: true,
}

// ErrCollectionNotReady reports that the semantic collection exists but cannot be searched yet.
var ErrCollectionNotReady error = &adapterr.AdapterError{
	Class:         adapterr.ClassCollectionNotReady,
	Message:       "semantic collection is not ready",
	Code:          "collection_not_ready",
	Hint:          "retry in a few seconds; the background collection load continues",
	Cause:         nil,
	SafeForClient: true,
}

// ErrSearchResultIncomplete reports that the backend returned a result set without the requested fields.
var ErrSearchResultIncomplete error = &adapterr.AdapterError{
	Class:         adapterr.ClassSearchResultIncomplete,
	Message:       "semantic search result is incomplete",
	Code:          "search_result_incomplete",
	Hint:          "retry the query; the daemon will refetch missing fields",
	Cause:         nil,
	SafeForClient: true,
}

// Row is one item row written to a collection. ID is the primary key. Metadata
// is the stored metadata JSON. Scalars maps each declared scalar column name
// to the row's value. Vector is the dense embedding.
type Row struct {
	ID                string
	Content           string
	RelativePath      string
	StartLine         int32
	EndLine           int32
	FileExtension     string
	Metadata          string
	SplitPart         int32
	SplitPartRecorded bool
	Vector            []float32
	Scalars           map[string]ScalarValue
}

// QueryRequest selects rows of a collection by filter, without ranking. A nil
// Filter selects every row. A positive Limit caps the rows returned.
type QueryRequest struct {
	Collection  string
	Declaration Declaration
	Filter      *Filter
	Limit       int
}

// EnsureRequest is the input of [Store.EnsureCollection]. Dimension is the
// dense vector width of a collection the store creates.
type EnsureRequest struct {
	Collection  string
	Declaration Declaration
	Dimension   int
}

// Store is a vector backend for collections of items.
type Store interface {
	// Search returns at most request.Limit hits ranked by dense similarity, fused
	// with lexical relevance when the backend runs a hybrid index.
	Search(ctx context.Context, request SearchRequest) ([]Hit, error)
	// Upsert writes rows to a collection. A row replaces any stored row with the
	// same ID.
	Upsert(ctx context.Context, collection string, declaration Declaration, rows []Row) error
	// Delete removes the rows a filter matches and returns the deleted count.
	Delete(ctx context.Context, collection string, filter Filter) (int64, error)
	// Query returns the rows a filter matches without ranking.
	Query(ctx context.Context, request QueryRequest) ([]Hit, error)
	// EnsureCollection creates the collection when it is absent and adds any
	// declared scalar column it lacks.
	EnsureCollection(ctx context.Context, request EnsureRequest) error
}

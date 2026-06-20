package daemon

import (
	"context"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// semanticReader is the read-only slice of the embedding-and-vector-store
// surface the manager queries.
type semanticReader interface {
	Available() bool
	CollectionName(codebasePath string) string
	ConversationCollectionName(collectionID string) string
	Search(ctx context.Context, codebasePath string, query string, limit int32, extensionFilter []string, relativePathPrefix string) ([]model.StoredChunk, error)
	SearchConversationCollection(ctx context.Context, collectionName string, query string, limit int32, filter semantic.ConversationFilter) ([]model.StoredChunk, error)
	Count(ctx context.Context, codebasePath string) (int32, error)
	ListCollections(ctx context.Context) ([]string, error)
	HasCollectionForPath(ctx context.Context, codebasePath string) (bool, error)
	HasStaging(ctx context.Context, codebasePath string) (bool, error)
}

// semanticHealthReader probes whether search can serve a query, both globally
// (store reachability) and per-path (the queried collection's load state).
type semanticHealthReader interface {
	// ProbeHealth actively checks that the store is reachable now, returning an
	// adapterr-classified error when it is not. It is the global shared-dependency
	// probe for surfaces without a single path.
	ProbeHealth(ctx context.Context) error
	// CollectionSearchable reports whether the collection serving codebasePath is
	// loaded into query nodes now, the deterministic per-path precondition for a
	// real search. The bool is false (with a classified error) when the store
	// cannot answer.
	CollectionSearchable(ctx context.Context, codebasePath string) (bool, error)
}

// semanticReuseLoader is the slice that reads already-embedded vectors back
// out of collections so a build or reindex can skip the embedder for chunks
// whose content is unchanged.
type semanticReuseLoader interface {
	LoadReuseVectors(ctx context.Context, collectionNames []string) (map[string][]float32, error)
	LoadReuseVectorsForPrefix(ctx context.Context, collectionName string, relativePathPrefix string) (map[string][]float32, error)
	LoadReuseVectorsForPath(ctx context.Context, collectionName string, relativePath string) (map[string][]float32, error)
}

// semanticWriter is the slice that mutates the live or staging collection.
type semanticWriter interface {
	Reindex(ctx context.Context, codebasePath string, addedOrModifiedChunks []model.StoredChunk, removal semantic.Removal, progress func(semantic.Progress), reuse map[string][]float32) error
	StageReindex(ctx context.Context, codebasePath string, chunks []model.StoredChunk, removal semantic.Removal, progress func(semantic.Progress), reuse map[string][]float32) error
	PromoteStaging(ctx context.Context, codebasePath string) error
	DeleteConversation(ctx context.Context, collectionName string, conversationID string) error
	CopyChunks(ctx context.Context, codebasePath string, srcRelativePath string, dstRelativePath string) (int, error)
	PruneToCurrent(ctx context.Context, codebasePath string, currentRelativePaths []string) error
}

// semanticDropper is the slice that removes collections.
type semanticDropper interface {
	Drop(ctx context.Context, codebasePath string) error
	DropStaging(ctx context.Context, codebasePath string) error
}

// semanticIndex is the full embedding-and-vector-store surface the manager
// depends on. It exists so tests can substitute a fake for the Milvus-backed
// [semantic.Service]; the concrete service satisfies it. The method set is
// exactly what the daemon calls, no more.
type semanticIndex interface {
	semanticReader
	semanticHealthReader
	semanticReuseLoader
	semanticWriter
	semanticDropper
}

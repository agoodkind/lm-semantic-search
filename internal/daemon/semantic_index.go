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
	SearchConversationCollectionCapped(ctx context.Context, collectionName string, query string, limit int32, perConversationLimit int32, minScore float64, filter semantic.ConversationFilter) ([]model.StoredChunk, error)
	Count(ctx context.Context, codebasePath string) (int32, error)
	ListCollections(ctx context.Context) ([]string, error)
	InspectCollection(ctx context.Context, collectionName string) (semantic.CollectionFacts, error)
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
	// CollectionState reports the per-path collection facts the daemon maps to a
	// status.CollectionReadiness: whether the collection exists and whether it is
	// loaded into query nodes now. A store that cannot answer returns a classified
	// error; the daemon treats that as unknown readiness without raising the global
	// banner (the global ProbeHealth covers a real outage). It returns
	// (false, false, nil) when semantic is not configured.
	CollectionState(ctx context.Context, codebasePath string) (exists bool, loaded bool, err error)
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
	BackfillConversationEnrichment(ctx context.Context, collectionName string, enrichment semantic.ConversationEnrichment, dryRun bool) (int, int, error)
	CopyChunks(ctx context.Context, codebasePath string, srcRelativePath string, dstRelativePath string) (int, error)
	PruneToCurrent(ctx context.Context, codebasePath string, currentRelativePaths []string) error
}

// semanticDropper is the slice that removes collections.
type semanticDropper interface {
	Drop(ctx context.Context, codebasePath string) error
	DropStaging(ctx context.Context, codebasePath string) error
}

// semanticMaintainer runs background store-maintenance migrations that change
// collection storage layout in place without re-embedding. The daemon's periodic
// sync loop drives these; they are idempotent and safe to call every tick.
type semanticMaintainer interface {
	// EnsureMmapEnabledAllCollections enables dense-vector mmap on every
	// collection, converging across ticks and skipping already-migrated ones.
	EnsureMmapEnabledAllCollections(ctx context.Context)
	// BackfillConversationCollectionsOnce populates the native scalar columns on
	// pre-existing conversation rows from stored metadata, preserving each dense
	// vector, at most once per collection per process.
	BackfillConversationCollectionsOnce(ctx context.Context)
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
	semanticMaintainer
}

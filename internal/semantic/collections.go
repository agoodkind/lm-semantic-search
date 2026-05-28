package semantic

import (
	"context"
	"strings"
)

// Collection name prefixes the daemon may write. The reverse-pass
// reconciler filters Milvus collection names by these prefixes so it never
// touches collections owned by another application sharing the same
// Milvus instance.
const (
	// CollectionPrefixDense is the prefix the daemon uses for dense
	// (non-hybrid) embedding collections.
	CollectionPrefixDense = "code_chunks_"
	// CollectionPrefixHybrid is the prefix the daemon uses for hybrid
	// (dense + sparse BM25) embedding collections.
	CollectionPrefixHybrid = "hybrid_code_chunks_"
)

// IsDaemonOwnedCollection reports whether collectionName belongs to one of
// the daemon's collection-name families. Callers use this to filter the
// raw collection list before deciding which collections are orphans of
// the daemon.
func IsDaemonOwnedCollection(collectionName string) bool {
	return strings.HasPrefix(collectionName, CollectionPrefixDense) || strings.HasPrefix(collectionName, CollectionPrefixHybrid)
}

// DropCollection removes one Milvus collection by name without consulting
// the codebase-path mapping. Used by the reverse-pass reconciler to drop
// orphan collections. Safe when the collection is missing.
func (service *Service) DropCollection(ctx context.Context, collectionName string) error {
	if !service.Available() {
		return nil
	}
	return service.dropIfExists(ctx, collectionName)
}

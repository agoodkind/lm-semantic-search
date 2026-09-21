package semantic

import "goodkind.io/lm-semantic-search/internal/adapterr"

// ErrUnavailable reports that the semantic backend is not configured.
var ErrUnavailable error = newSentinel(
	adapterr.ClassMilvusUnavailable,
	"semantic backend is unavailable",
	"milvus_unavailable",
	"verify MILVUS_ADDRESS and that the milvus process is reachable",
)

// ErrCollectionMissing reports that the semantic collection does not exist yet.
var ErrCollectionMissing error = newSentinel(
	adapterr.ClassCollectionMissing,
	"semantic collection is missing",
	"collection_missing",
	"re-run index_codebase to recreate the collection",
)

// ErrCollectionNotReady reports that the semantic collection exists but cannot be searched yet.
var ErrCollectionNotReady error = newSentinel(
	adapterr.ClassCollectionNotReady,
	"semantic collection is not ready",
	"collection_not_ready",
	"retry in a few seconds; the background collection load continues",
)

// ErrCollectionLoadDeferred reports that the daemon refused to start a
// collection load because Milvus recently ran out of memory loading one, and
// new loads are paused for a bounded interval. It shares the not-ready class so
// every caller that already retries on ErrCollectionNotReady treats it the same
// way, while its own code and hint say why the load did not start.
var ErrCollectionLoadDeferred error = newSentinel(
	adapterr.ClassCollectionNotReady,
	"semantic collection load is paused after Milvus ran out of memory",
	"collection_load_deferred",
	"retry later; new collection loads resume once the pause elapses",
)

// ErrMaintenance reports that the daemon refused to load a collection because
// an operator put it in maintenance mode.
var ErrMaintenance error = newSentinel(
	adapterr.ClassMaintenance,
	"semantic collection loads are paused for maintenance",
	"maintenance",
	"retry after the operator turns maintenance mode off",
)

// ErrSearchResultIncomplete reports that Milvus returned a result set without the requested fields.
var ErrSearchResultIncomplete error = newSentinel(
	adapterr.ClassSearchResultIncomplete,
	"semantic search result is incomplete",
	"search_result_incomplete",
	"retry the query; the daemon will refetch missing fields",
)

func newSentinel(class adapterr.Class, message, code, hint string) error {
	return &adapterr.AdapterError{
		Class:         class,
		Message:       message,
		Code:          code,
		Hint:          hint,
		Cause:         nil,
		SafeForClient: true,
	}
}

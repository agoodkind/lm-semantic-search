package semantic

import (
	"strings"

	"goodkind.io/lm-semantic-search/internal/adapterr"
)

// storeUnavailable reports whether a milvus client error means the store cannot
// serve a request now. The store's authoritative readiness signal is the
// [GetLoadState] enum the readiness probe reads, not error text; this classifies
// only the error path, when a milvus call fails outright, by its gRPC transport
// status (Unavailable or DeadlineExceeded), which a down or unreachable milvus
// returns before any server status. It deliberately does not parse milvus
// application error strings: the load-state enum, not a substring list, decides
// whether a collection can serve a search.
func storeUnavailable(err error) bool {
	return adapterr.IsGRPCUnavailable(err)
}

// collectionNotLoadedMessage is the stable milvus error text for a collection
// that exists but is not loaded into query nodes. It is matched
// case-insensitively (against a lowercased error string) so a capitalized
// milvus variant still maps to the retry hint rather than falling through to a
// generic internal error. The readiness probe decides load state
// deterministically from [GetLoadState]; this single message match exists only
// so a user-facing search that races a just-unloaded collection returns the
// ErrCollectionNotReady retry hint instead of an opaque internal error. It is
// not part of the searchable gating decision. The typed milvus sentinel
// (merr.ErrCollectionNotLoaded) would be exact, but importing milvus's merr
// package crashes the gate's govulncheck on its generics, so this keeps the
// single stable message instead.
const collectionNotLoadedMessage = "collection not loaded"

// storeSearchSentinel maps a milvus search failure to a typed error the daemon
// must react to, or nil when the caller should wrap the error generically. A
// still-loading collection keeps its retry hint (ErrCollectionNotReady); an
// unreachable store becomes a ClassMilvusUnavailable outage so the health record
// degrades and search gating fails open. The classified error carries the raw
// cause without re-wrapping, so the call site keeps ownership of logging and
// generic wrapping.
func storeSearchSentinel(err error) error {
	if err == nil {
		return nil
	}
	if strings.Contains(strings.ToLower(err.Error()), collectionNotLoadedMessage) {
		return ErrCollectionNotReady
	}
	if storeUnavailable(err) {
		return adapterr.NewMilvusUnavailable(err)
	}
	return nil
}

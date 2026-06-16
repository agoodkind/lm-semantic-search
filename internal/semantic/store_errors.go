package semantic

import (
	"strings"

	"goodkind.io/lm-semantic-search/internal/adapterr"
)

// storeUnavailableSubstrings are milvus client error fragments that mean the
// vector store's query plane cannot serve a request now: the server is
// unreachable, an RPC timed out, or no query node can answer. They are matched
// case-insensitively against the error text, including the gRPC status string
// ("rpc error: code = Unavailable ...") so this package needs no direct
// google.golang.org/grpc import; that import would make the grpc-handler lint
// heuristic treat every *Service method as a gRPC handler. "collection not
// loaded" is deliberately excluded: it has its own sentinel (ErrCollectionNotReady)
// so a still-loading collection keeps its retry hint rather than reading as a
// hard outage.
var storeUnavailableSubstrings = []string{
	"unavailable",
	"deadline exceeded",
	"deadlineexceeded",
	"connection refused",
	"connection reset",
	"transport is closing",
	"no such host",
	"no available querynode",
	"no available shard",
	"not enough nodes",
}

// storeUnavailable reports whether a milvus client error means the store is
// unreachable or its query plane cannot serve a request right now. A true result
// is the signal to surface a ClassMilvusUnavailable outage so search gating
// degrades and fails open instead of forcing the agent onto a search path that
// cannot answer.
func storeUnavailable(err error) bool {
	if err == nil {
		return false
	}
	lowered := strings.ToLower(err.Error())
	for _, fragment := range storeUnavailableSubstrings {
		if strings.Contains(lowered, fragment) {
			return true
		}
	}
	return false
}

// storeSearchSentinel maps a milvus search failure to a typed error the daemon
// must react to, or nil when the caller should wrap the error generically. A
// still-loading collection keeps its retry hint (ErrCollectionNotReady); an
// unreachable store or down query node becomes a ClassMilvusUnavailable outage so
// the health record degrades and search gating fails open. The classified error
// carries the raw cause without re-wrapping, so the call site keeps ownership of
// logging and generic wrapping.
func storeSearchSentinel(err error) error {
	if strings.Contains(err.Error(), "collection not loaded") {
		return ErrCollectionNotReady
	}
	if storeUnavailable(err) {
		return adapterr.NewMilvusUnavailable(err)
	}
	return nil
}

package milvus

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"goodkind.io/lm-semantic-search/collection"
	"goodkind.io/lm-semantic-search/internal/adapterr"
)

// collectionNotLoadedMessage is the stable milvus error text for a collection
// that exists but is not loaded into query nodes. The match is
// case-insensitive against a lowercased error string. The readiness probe
// decides load state from the load state enum. This message match only maps a
// user-facing search that races a just-unloaded collection to the
// collection.ErrCollectionNotReady retry hint instead of an opaque internal
// error. The typed milvus sentinel (merr.ErrCollectionNotLoaded) would be exact,
// but importing the merr package crashes the gate's govulncheck on its generics.
const collectionNotLoadedMessage = "collection not loaded"

// Unavailable reports whether a milvus client error means the store cannot
// serve a request now. The store's authoritative readiness signal is the
// load state enum the readiness probe reads, not error text. This classifies
// only the error path, when a milvus call fails outright, by its gRPC transport
// status (Unavailable or DeadlineExceeded), which a down or unreachable milvus
// returns before any server status. It does not parse milvus application error
// strings.
func Unavailable(err error) bool {
	return adapterr.IsGRPCUnavailable(err)
}

// SearchSentinel maps a milvus search failure to a typed error the daemon
// must react to, or nil when the caller should wrap the error generically. A
// still-loading collection returns [collection.ErrCollectionNotReady]. An
// unreachable store becomes a ClassMilvusUnavailable outage, which degrades the
// health record and fails search gating open. The classified error wraps the
// raw cause without another layer. The call site owns logging and generic
// wrapping.
func SearchSentinel(err error) error {
	if err == nil {
		return nil
	}
	if strings.Contains(strings.ToLower(err.Error()), collectionNotLoadedMessage) {
		return collection.ErrCollectionNotReady
	}
	if Unavailable(err) {
		return adapterr.NewMilvusUnavailable(err)
	}
	return nil
}

// WrapError logs and classifies a milvus call error from the write, index,
// or read/check paths. The operation argument is the failing call, for example
// "create Milvus collection <name>". A transport-unavailable store outage
// becomes a ClassMilvusUnavailable adapter error. The daemon treats it as a
// shared-infrastructure outage and keeps the codebase resumable behind the
// health banner instead of marking it failed. Every other error is wrapped
// plainly and keeps its existing classification, a real per-collection fault.
// The cause is appended with %w, and [errors.Is] and [errors.As] still match it.
// WrapError logs the failure once, and a nil err returns nil.
func WrapError(ctx context.Context, err error, operation string) error {
	if err == nil {
		return nil
	}
	slog.ErrorContext(ctx, "milvus operation failed", "operation", operation, "err", err)
	if Unavailable(err) {
		return adapterr.NewMilvusUnavailable(fmt.Errorf("%s: %w", operation, err))
	}
	return fmt.Errorf("%s: %w", operation, err)
}

// SearchError logs a Milvus search failure and maps it to a typed store
// sentinel when one applies. Otherwise it wraps the failure with the operation
// and collection.
func SearchError(ctx context.Context, operation string, collectionName string, err error) error {
	slog.ErrorContext(ctx, operation+" failed", "collection", collectionName, "err", err)
	if sentinel := SearchSentinel(err); sentinel != nil {
		return sentinel
	}
	return fmt.Errorf("%s collection %s: %w", operation, collectionName, err)
}

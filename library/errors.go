package library

import "errors"

// The library wraps each failure around one of these sentinels. A caller
// classifies a failure with [errors.Is].
var (
	// ErrStoreMismatch reports a catalog or vector pool that is bound to a
	// different descriptor or catalog UUID.
	ErrStoreMismatch = errors.New("library: store descriptor mismatch")
	// ErrInvalidRequest reports a configuration, declaration, or request that
	// fails validation.
	ErrInvalidRequest = errors.New("library: invalid request")
	// ErrAppendConflict reports a write that reuses a generation order with a
	// different token or content, or rewrites an appended occurrence.
	ErrAppendConflict = errors.New("library: append conflict")
	// ErrStaleGeneration reports an unknown generation order below the
	// committed one.
	ErrStaleGeneration = errors.New("library: stale generation")
	// ErrVectorCorrupt reports a backend vector with a mismatched identity
	// digest or checksum.
	ErrVectorCorrupt = errors.New("library: vector corrupt")
	// ErrVectorMissing reports a referenced vector the backend does not return.
	ErrVectorMissing = errors.New("library: vector missing")
	// ErrCursorExpired reports a cursor for an expired or removed result
	// snapshot.
	ErrCursorExpired = errors.New("library: cursor expired")
	// ErrCursorMismatch reports a cursor issued for a different request or
	// ranking configuration.
	ErrCursorMismatch = errors.New("library: cursor mismatch")
	// ErrDeadline reports an operation that could not finish before its
	// deadline. The library returns no partial result with it.
	ErrDeadline = errors.New("library: deadline exceeded")
	// ErrResourceLimit reports an operation that exceeded a configured resource
	// budget. The library returns no partial result with it.
	ErrResourceLimit = errors.New("library: resource limit exceeded")
)

package semantic

import (
	"context"
	"errors"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/adapterr"
)

// storeSearchSentinel keeps the retry hint for a still-loading collection and
// returns nil for an ordinary error so the caller wraps it generically. The
// gRPC-transport store-unavailable path is covered by
// adapterr.TestIsGRPCUnavailable so this package stays free of a grpc import.
func TestStoreSearchSentinel(t *testing.T) {
	t.Parallel()

	if got := storeSearchSentinel(errors.New("hybrid search collection foo: collection not loaded")); !errors.Is(got, ErrCollectionNotReady) {
		t.Fatalf("collection-not-loaded: storeSearchSentinel = %v, want ErrCollectionNotReady", got)
	}
	if got := storeSearchSentinel(errors.New("invalid filter expression")); got != nil {
		t.Fatalf("ordinary: storeSearchSentinel = %v, want nil", got)
	}
	if got := storeSearchSentinel(nil); got != nil {
		t.Fatalf("nil: storeSearchSentinel = %v, want nil", got)
	}
}

// storeUnavailable classifies only the gRPC transport status, so an ordinary
// error (including a milvus application message) does not read as a hard outage;
// the readiness probe decides load state from the GetLoadState enum instead.
func TestStoreUnavailableIgnoresPlainErrors(t *testing.T) {
	t.Parallel()

	if storeUnavailable(errors.New("collection not loaded")) {
		t.Fatal("a plain non-gRPC error must not read as store-unavailable")
	}
	if storeUnavailable(nil) {
		t.Fatal("nil must not read as store-unavailable")
	}
}

// wrapStoreError must NOT label a real per-collection fault (for example a schema
// mismatch) as a shared-infrastructure outage, so a write-path failure that is
// genuinely the collection's fault still marks the codebase failed. It also
// attaches the operation context and preserves the cause chain. The
// store-unavailable branch needs a real gRPC error and is covered externally in
// TestWrapStoreErrorClassifiesTransportOutage so this package stays grpc-free.
func TestWrapStoreErrorPlainError(t *testing.T) {
	t.Parallel()

	cause := errors.New("collection schema mismatch[field conversationId does not exist]")
	wrapped := wrapStoreError(context.Background(), cause, "insert Milvus batch into conv_chunks_x")
	if adapterr.IsInfraFailure(wrapped) {
		t.Fatalf("a non-transport error must not be classified as an infra outage: %v", wrapped)
	}
	if !errors.Is(wrapped, cause) {
		t.Fatal("wrapStoreError must preserve the cause chain")
	}
	if !strings.Contains(wrapped.Error(), "conv_chunks_x") {
		t.Fatalf("wrapStoreError must include the operation context, got %q", wrapped.Error())
	}
	if wrapStoreError(context.Background(), nil, "insert into x") != nil {
		t.Fatal("nil error must wrap to nil")
	}
}

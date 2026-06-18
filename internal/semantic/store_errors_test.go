package semantic

import (
	"errors"
	"testing"
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

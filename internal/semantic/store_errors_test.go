package semantic

import (
	"errors"
	"fmt"
	"testing"

	"goodkind.io/lm-semantic-search/internal/adapterr"
)

// storeUnavailable classifies a milvus client error as a query-plane outage from
// its message text (which includes the gRPC status string for an Unavailable or
// DeadlineExceeded RPC), and leaves an ordinary error alone so only a real outage
// degrades the health record. The errors use plain strings shaped like the gRPC
// status text so this test needs no direct google.golang.org/grpc import.
func TestStoreUnavailable(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"grpc unavailable", errors.New("rpc error: code = Unavailable desc = connection error"), true},
		{"grpc deadline", errors.New("rpc error: code = DeadlineExceeded desc = context deadline exceeded"), true},
		{"grpc not found", errors.New("rpc error: code = NotFound desc = collection not found"), false},
		{"connection refused text", errors.New("dial tcp 127.0.0.1:19530: connect: connection refused"), true},
		{"no querynode text", errors.New("no available querynode to serve the request"), true},
		{"wrapped unavailable", fmt.Errorf("search: %w", errors.New("rpc error: code = Unavailable desc = down")), true},
		{"ordinary error", errors.New("invalid argument: bad filter"), false},
	}
	for _, testCase := range cases {
		if got := storeUnavailable(testCase.err); got != testCase.want {
			t.Fatalf("%s: storeUnavailable = %v, want %v", testCase.name, got, testCase.want)
		}
	}
}

// storeSearchSentinel keeps the retry hint for a still-loading collection, raises
// a ClassMilvusUnavailable outage for an unreachable store so search gating fails
// open, and returns nil for an ordinary error so the caller wraps it generically.
func TestStoreSearchSentinel(t *testing.T) {
	t.Parallel()

	if got := storeSearchSentinel(errors.New("collection not loaded: hybrid_code_chunks_abc")); !errors.Is(got, ErrCollectionNotReady) {
		t.Fatalf("not-loaded error: storeSearchSentinel = %v, want ErrCollectionNotReady", got)
	}

	unavailable := storeSearchSentinel(errors.New("rpc error: code = Unavailable desc = milvus is down"))
	var adapterErr *adapterr.AdapterError
	if !errors.As(unavailable, &adapterErr) || adapterErr.Class != adapterr.ClassMilvusUnavailable {
		t.Fatalf("unavailable error: storeSearchSentinel = %v, want ClassMilvusUnavailable", unavailable)
	}

	if got := storeSearchSentinel(errors.New("invalid filter expression")); got != nil {
		t.Fatalf("ordinary error: storeSearchSentinel = %v, want nil", got)
	}
}

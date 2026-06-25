package semantic_test

import (
	"context"
	"errors"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// wrapStoreError must label a gRPC transport outage (the signal a down or
// unreachable Milvus returns) as a shared-infrastructure failure, so a store
// outage during indexing keeps the codebase resumable behind the health banner
// instead of marking it failed with "Re-run index_codebase". This lives in the
// external package because constructing a real gRPC status requires importing
// google.golang.org/grpc, which the production semantic package must not do.
func TestWrapStoreErrorClassifiesTransportOutage(t *testing.T) {
	t.Parallel()

	for _, code := range []codes.Code{codes.Unavailable, codes.DeadlineExceeded} {
		cause := status.Error(code, `connection error: desc = "transport: Error while dialing: dial tcp [::1]:19530: connect: connection refused"`)
		wrapped := semantic.WrapStoreErrorForTest(context.Background(), cause, "insert Milvus batch into hybrid_code_chunks_x")

		if !adapterr.IsInfraFailure(wrapped) {
			t.Fatalf("code %s: store outage must be an infra failure, got %v", code, wrapped)
		}
		var adapterErr *adapterr.AdapterError
		if !errors.As(wrapped, &adapterErr) || adapterErr.Class != adapterr.ClassMilvusUnavailable {
			t.Fatalf("code %s: want ClassMilvusUnavailable, got %#v", code, wrapped)
		}
		if !errors.Is(wrapped, cause) {
			t.Fatalf("code %s: wrapped error must preserve the cause chain", code)
		}
	}
}

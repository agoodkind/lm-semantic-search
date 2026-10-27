package milvus_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/collection"
	"goodkind.io/lm-semantic-search/collection/milvus"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestWrapErrorPreservesMilvusOutageCause(t *testing.T) {
	t.Parallel()
	for _, code := range []codes.Code{codes.Unavailable, codes.DeadlineExceeded} {
		cause := status.Error(code, "vector store connection failed")
		wrapped := milvus.WrapError(context.Background(), cause, "upsert collection source_chunks")
		var classified *adapterr.AdapterError
		if !errors.As(wrapped, &classified) || classified.Class != adapterr.ClassMilvusUnavailable {
			t.Fatalf("upsert error = %v, want MilvusUnavailable", wrapped)
		}
		if !errors.Is(wrapped, cause) || !strings.HasPrefix(wrapped.Error(), "upsert collection source_chunks:") {
			t.Fatalf("upsert error = %v, want operation context and original cause", wrapped)
		}
	}
	if wrapped := milvus.WrapError(context.Background(), nil, "upsert collection source_chunks"); wrapped != nil {
		t.Fatalf("successful upsert error = %v, want nil", wrapped)
	}
}

func TestSearchSentinelPreservesCollectionReadiness(t *testing.T) {
	t.Parallel()
	if got := milvus.SearchSentinel(errors.New("collection not loaded")); !errors.Is(got, collection.ErrCollectionNotReady) {
		t.Fatalf("search classification = %v, want CollectionNotReady", got)
	}
	if got := milvus.SearchSentinel(errors.New("invalid filter")); got != nil {
		t.Fatalf("ordinary search failure classification = %v, want nil", got)
	}
	if got := milvus.SearchSentinel(nil); got != nil {
		t.Fatalf("successful search classification = %v, want nil", got)
	}
}

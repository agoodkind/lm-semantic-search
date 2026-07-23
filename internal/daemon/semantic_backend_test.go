package daemon

import (
	"context"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/localvec"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func TestNewSemanticIndexSelectsLocalBackend(t *testing.T) {
	t.Parallel()
	idx, err := newSemanticIndex(context.Background(), config.Config{IndexBackend: config.IndexBackendLocal, StateRoot: t.TempDir()})
	if err != nil {
		t.Fatalf("newSemanticIndex(local) error: %v", err)
	}
	if _, ok := idx.(*localvec.Store); !ok {
		t.Fatalf("local backend type = %T, want *localvec.Store", idx)
	}
}

func TestNewSemanticIndexSelectsMilvusBackend(t *testing.T) {
	t.Parallel()
	idx, err := newSemanticIndex(context.Background(), config.Config{IndexBackend: config.IndexBackendMilvus})
	if err != nil {
		t.Fatalf("newSemanticIndex(milvus) error: %v", err)
	}
	if _, ok := idx.(*semantic.Service); !ok {
		t.Fatalf("milvus backend type = %T, want *semantic.Service", idx)
	}
}

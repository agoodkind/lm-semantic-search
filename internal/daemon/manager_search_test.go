package daemon

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func TestSearchCodeReturnsTypedErrorWhenSemanticStoreReportsUnavailable(t *testing.T) {
	t.Parallel()

	manager, repoPath := newIndexedSearchManager(t)
	manager.semantic = &fakeSemantic{unavailable: true}

	outcome, err := manager.SearchCode(context.Background(), repoPath, "needle", 5, nil)
	assertMilvusUnavailable(t, outcome, err)
}

func TestSearchCodeReturnsTypedErrorWhenSemanticSearchReturnsUnavailable(t *testing.T) {
	t.Parallel()

	manager, repoPath := newIndexedSearchManager(t)
	manager.semantic = &fakeSemantic{
		search: func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error) {
			return nil, semantic.ErrUnavailable
		},
	}

	outcome, err := manager.SearchCode(context.Background(), repoPath, "needle", 5, nil)
	assertMilvusUnavailable(t, outcome, err)
}

func newIndexedSearchManager(t *testing.T) (*Manager, string) {
	t.Helper()

	manager, _, repoPath := newTestManager(t)
	canonicalPath, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonicalPath)
	codebase.Status = model.CodebaseStatusIndexed
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	return manager, repoPath
}

func assertMilvusUnavailable(t *testing.T, outcome SearchOutcome, err error) {
	t.Helper()

	if len(outcome.Results) != 0 {
		t.Fatalf("SearchCode returned %d results, want none", len(outcome.Results))
	}
	if err == nil {
		t.Fatal("SearchCode returned nil error, want typed vector-store unavailable error")
	}
	var adapterError *adapterr.AdapterError
	if !errors.As(err, &adapterError) {
		t.Fatalf("SearchCode error type = %T, want *adapterr.AdapterError", err)
	}
	if adapterError.Class != adapterr.ClassMilvusUnavailable {
		t.Fatalf("SearchCode error class = %q, want %q", adapterError.Class, adapterr.ClassMilvusUnavailable)
	}
}

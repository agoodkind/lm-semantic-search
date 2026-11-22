package daemon

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestSearchRPCWithIngestionDisabled(t *testing.T) {
	t.Parallel()

	manager, repoPath := newIndexedSearchManager(t)
	manager.config.IngestionDisabled = true
	manager.semantic = &fakeSemantic{
		prepareCollection: func(context.Context, string) error {
			return errors.New("search attempted a schema migration")
		},
		search: func(context.Context, string, string, int32, []string, string) (
			[]model.StoredChunk,
			error,
		) {
			return []model.StoredChunk{{Content: "needle", RelativePath: "fixture.go"}}, nil
		},
	}
	server := NewGRPCServer(manager, nil)
	ctx := context.Background()
	request := &pb.SearchCodeRequest{Path: repoPath, Query: "needle"}
	response, err := server.SearchCode(ctx, request)
	if err != nil || len(response.GetResults()) != 1 {
		t.Fatalf(
			"SearchCode with ingestion disabled returned %d hits, error = %v",
			len(response.GetResults()),
			err,
		)
	}

	_, err = server.StartIndex(ctx, &pb.StartIndexRequest{Path: repoPath})
	if status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("StartIndex returned %v, want FailedPrecondition", err)
	}

	_, err = server.SyncIndex(ctx, &pb.SyncIndexRequest{Path: repoPath})
	if status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("SyncIndex returned %v, want FailedPrecondition", err)
	}

	_, err = server.ClearIndex(ctx, &pb.ClearIndexRequest{Path: repoPath})
	if status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("ClearIndex returned %v, want FailedPrecondition", err)
	}

	manager.config.SearchDisabled = true
	_, err = server.SearchCode(ctx, request)
	if status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("SearchCode returned %v, want FailedPrecondition", err)
	}
}

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
		search: func(context.Context, string, string, int32, []string, string) (
			[]model.StoredChunk,
			error,
		) {
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
		t.Fatalf(
			"SearchCode error class = %q, want %q",
			adapterError.Class,
			adapterr.ClassMilvusUnavailable,
		)
	}
}

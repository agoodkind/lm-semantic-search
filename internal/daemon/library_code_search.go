package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"strings"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/library"
)

// defaultCodeSearchLimit is the page size of a code search without a limit,
// the same default the Milvus service applies.
const defaultCodeSearchLimit = 10

// libraryCodeLease is the collection lease of a codebase namespace. The
// library reads its catalog and vector pool without a residency load.
type libraryCodeLease struct{}

// Release completes the lease contract.
func (libraryCodeLease) Release() {}

// PrepareCollection returns nil for a codebase namespace and passes any other
// collection name to the wrapped index.
func (index *libraryCodeIndex) PrepareCollection(ctx context.Context, collectionName string) error {
	if index.isCodeNamespace(collectionName) {
		return nil
	}
	return wrapDelegated(ctx, "prepare collection", index.semanticIndex.PrepareCollection(ctx, collectionName))
}

// AcquireCollection returns a no-op lease for a codebase namespace and passes
// any other collection name to the wrapped index.
func (index *libraryCodeIndex) AcquireCollection(ctx context.Context, collectionName string) (semantic.CollectionLease, error) {
	if index.isCodeNamespace(collectionName) {
		return libraryCodeLease{}, nil
	}
	value, err := index.semanticIndex.AcquireCollection(ctx, collectionName)
	return value, wrapDelegated(ctx, "acquire collection", err)
}

func (index *libraryCodeIndex) Search(
	ctx context.Context,
	codebasePath string,
	query string,
	limit int32,
	extensionFilter []string,
	relativePathPrefix string,
) ([]model.StoredChunk, error) {
	if semantic.IsDocumentPath(codebasePath) {
		value, err := index.semanticIndex.Search(ctx, codebasePath, query, limit, extensionFilter, relativePathPrefix)
		return value, wrapDelegated(ctx, "search", err)
	}
	namespace := index.codebaseNamespace(codebasePath)
	_, registered, err := index.namespaceStats(ctx, namespace)
	if err != nil {
		return nil, err
	}
	if !registered {
		return nil, semantic.ErrCollectionMissing
	}
	pageSize := int(limit)
	if pageSize <= 0 {
		pageSize = defaultCodeSearchLimit
	}
	page, err := index.store.Search(ctx, library.SearchRequest{
		Namespace:     namespace,
		Query:         strings.ReplaceAll(query, "\x00", " "),
		Filter:        codeSearchFilter(extensionFilter, relativePathPrefix),
		GroupBy:       "",
		PerGroupLimit: 0,
		MinScore:      0,
		PageSize:      pageSize,
		Cursor:        "",
	})
	if err != nil {
		slog.ErrorContext(ctx, "library code search failed", "namespace", namespace, "err", err)
		return nil, fmt.Errorf("library code search in %s: %w", namespace, err)
	}
	chunks := make([]model.StoredChunk, 0, len(page.Hits))
	for _, hit := range page.Hits {
		chunks = append(chunks, codeHitChunk(hit))
	}
	return chunks, nil
}

// codeSearchFilter returns nil when neither the extensions nor the prefix
// restrict the search.
func codeSearchFilter(extensionFilter []string, relativePathPrefix string) *library.Filter {
	children := make([]library.Filter, 0, 2)
	if len(extensionFilter) > 0 {
		values := make([]library.ScalarValue, 0, len(extensionFilter))
		for _, extension := range extensionFilter {
			values = append(values, libraryStringScalar(extension))
		}
		children = append(children, library.Filter{Op: library.In, Column: codeColumnFileExtension, Values: values})
	}
	if relativePathPrefix != "" {
		children = append(children, library.Filter{Op: library.Prefix, Column: codeColumnRelativePath, Prefix: relativePathPrefix})
	}
	switch len(children) {
	case 0:
		return nil
	case 1:
		return &children[0]
	default:
		return &library.Filter{Op: library.All, Children: children}
	}
}

// codeHitChunk converts one library hit to the chunk shape the search
// surfaces render.
func codeHitChunk(hit library.SearchHit) model.StoredChunk {
	var chunk model.StoredChunk
	chunk.Content = hit.SourceText
	chunk.Score = hit.Score
	chunk.RelativePath = hit.Scalars[codeColumnRelativePath].String
	chunk.Language = hit.Scalars[codeColumnLanguage].String
	chunk.FileExtension = hit.Scalars[codeColumnFileExtension].String
	chunk.StartLine = int32Scalar(hit.Scalars[codeColumnStartLine].Int64)
	chunk.EndLine = int32Scalar(hit.Scalars[codeColumnEndLine].Int64)
	chunk.SplitPart = int32Scalar(hit.Scalars[codeColumnSplitPart].Int64)
	chunk.SplitPartRecorded = true
	return chunk
}

// int32Scalar returns an int64 scalar that the adapter wrote from an int32
// field, capped at the int32 range.
func int32Scalar(value int64) int32 {
	if value > math.MaxInt32 {
		return math.MaxInt32
	}
	if value < math.MinInt32 {
		return math.MinInt32
	}
	return int32(value)
}

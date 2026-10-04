package localvec

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"strings"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

const (
	defaultSearchLimit = 10
	// exactSearchThreshold keeps small collections deterministic and avoids
	// graph traversal overhead. Larger collections use the usearch HNSW index.
	exactSearchThreshold         = 4096
	initialSearchOverfetchFactor = 8
	searchOverfetchGrowthFactor  = 2
)

type scoredRow struct {
	stored row
	score  float64
}

type scoredRowFilter func([]scoredRow, int) []scoredRow

// Search searches a codebase collection.
func (store *Store) Search(
	ctx context.Context,
	codebasePath string,
	query string,
	limit int32,
	extensionFilter []string,
	relativePathPrefix string,
) ([]model.StoredChunk, error) {
	extensions := normalizeExtensions(extensionFilter)
	prefix := normalizeSearchPrefix(relativePathPrefix)
	scored, err := store.searchRows(
		ctx,
		store.CollectionName(codebasePath),
		query,
		limit,
		func(scored []scoredRow, resultLimit int) []scoredRow {
			return limitScoredRows(
				scored,
				resultLimit,
				func(candidate scoredRow) bool {
					if len(extensions) > 0 {
						if _, found := extensions[candidate.stored.FileExtension]; !found {
							return false
						}
					}
					return matchesSearchPrefix(candidate.stored.RelativePath, prefix)
				},
			)
		},
	)
	if err != nil {
		return nil, err
	}
	results := make([]model.StoredChunk, 0, len(scored))
	for _, candidate := range scored {
		results = append(results, candidate.stored.chunk(candidate.score))
	}
	return results, nil
}

// searchRows embeds the query, scores the collection's rows, sorts them with
// sortScoredRows, and returns the rows filter keeps. Above
// exactSearchThreshold rows it grows the HNSW candidate count with the limit
// until filter keeps enough rows. Code search uses it. Collection search ranks
// a fixed candidate set instead (see Store.SearchCollection).
func (store *Store) searchRows(
	ctx context.Context,
	collectionName string,
	query string,
	limit int32,
	filter scoredRowFilter,
) ([]scoredRow, error) {
	if err := operationContextError(ctx, "search local vectors"); err != nil {
		return nil, err
	}
	stored, err := store.collectionForName(collectionName, false)
	if err != nil {
		return nil, err
	}
	collectionSize, exists, err := stored.vectorCount()
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, semantic.ErrCollectionMissing
	}
	normalizedQuery, err := store.embedQuery(ctx, collectionName, query)
	if err != nil {
		return nil, err
	}

	resultLimit := effectiveLimit(limit)
	var scored []scoredRow
	if collectionSize <= exactSearchThreshold {
		rows, _, snapshotErr := stored.snapshot()
		if snapshotErr != nil {
			return nil, snapshotErr
		}
		scored, err = scoreExactRows(ctx, collectionName, rows, normalizedQuery)
	} else {
		scored, err = scoreApproximateRowsAdaptive(
			stored,
			normalizedQuery,
			collectionSize,
			resultLimit,
			filter,
		)
	}
	if err != nil {
		return nil, err
	}

	if collectionSize <= exactSearchThreshold {
		sortScoredRows(scored)
		scored = filter(scored, resultLimit)
	}
	return scored, nil
}

// embedQuery embeds query with the configured query instruction prefix and
// normalizes the vector for cosine scoring.
func (store *Store) embedQuery(
	ctx context.Context,
	collectionName string,
	query string,
) ([]float32, error) {
	provider, err := store.embeddingProvider()
	if err != nil {
		return nil, err
	}
	queryText := store.cfg.QueryInstructionPrefix + query
	queryVector, err := provider.Embed(ctx, queryText)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"embed local vector query failed",
			"collection",
			collectionName,
			"err",
			err,
		)
		return nil, fmt.Errorf("embed local vector query: %w", err)
	}
	normalizedQuery, err := normalizeVector(queryVector)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"normalize local vector query failed",
			"collection",
			collectionName,
			"err",
			err,
		)
		return nil, fmt.Errorf("normalize local vector query: %w", err)
	}
	return normalizedQuery, nil
}

func scoreApproximateRowsAdaptive(
	stored *collection,
	query []float32,
	collectionSize int,
	resultLimit int,
	filter scoredRowFilter,
) ([]scoredRow, error) {
	candidateCount := int(min(
		int64(resultLimit)*int64(initialSearchOverfetchFactor),
		int64(collectionSize),
	))
	for {
		scored, err := scoreApproximateRows(stored, query, candidateCount)
		if err != nil {
			return nil, err
		}
		sortScoredRows(scored)
		filtered := filter(scored, resultLimit)
		if len(filtered) >= resultLimit ||
			candidateCount >= collectionSize ||
			len(scored) < candidateCount {
			return filtered, nil
		}
		candidateCount = int(min(
			int64(candidateCount)*int64(searchOverfetchGrowthFactor),
			int64(collectionSize),
		))
	}
}

func limitScoredRows(
	scored []scoredRow,
	resultLimit int,
	keep func(scoredRow) bool,
) []scoredRow {
	filtered := make([]scoredRow, 0, min(len(scored), resultLimit))
	for _, candidate := range scored {
		if keep != nil && !keep(candidate) {
			continue
		}
		filtered = append(filtered, candidate)
		if len(filtered) >= resultLimit {
			break
		}
	}
	return filtered
}

func scoreExactRows(
	ctx context.Context,
	collectionName string,
	rows []row,
	query []float32,
) ([]scoredRow, error) {
	scored := make([]scoredRow, 0, len(rows))
	for _, candidate := range rows {
		if err := operationContextError(ctx, "score local vectors"); err != nil {
			return nil, err
		}
		score, err := dotProduct(query, candidate.Vector)
		if err != nil {
			slog.ErrorContext(
				ctx,
				"score local vector row failed",
				"collection",
				collectionName,
				"row_id",
				candidate.ID,
				"err",
				err,
			)
			return nil, fmt.Errorf("score local vector row %s: %w", candidate.ID, err)
		}
		scored = append(scored, scoredRow{stored: candidate, score: score})
	}
	return scored, nil
}

func scoreApproximateRows(
	stored *collection,
	query []float32,
	candidateCount int,
) ([]scoredRow, error) {
	rows, distances, exists, err := stored.nearest(query, candidateCount)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, semantic.ErrCollectionMissing
	}
	scored := make([]scoredRow, 0, len(rows))
	for index, candidate := range rows {
		scored = append(scored, scoredRow{
			stored: candidate,
			score:  1 - float64(distances[index]),
		})
	}
	return scored, nil
}

// sortScoredRows orders rows by descending score, then ascending relativePath,
// then ascending row id, the same total order the Milvus collection search
// applies to its ranking.
func sortScoredRows(scored []scoredRow) {
	sort.SliceStable(scored, func(left int, right int) bool {
		if scored[left].score != scored[right].score {
			return scored[left].score > scored[right].score
		}
		if scored[left].stored.RelativePath != scored[right].stored.RelativePath {
			return scored[left].stored.RelativePath < scored[right].stored.RelativePath
		}
		return scored[left].stored.ID < scored[right].stored.ID
	})
}

func dotProduct(left []float32, right []float32) (float64, error) {
	if len(left) != len(right) {
		return 0, fmt.Errorf(
			"vector dimension mismatch: query has %d values, row has %d",
			len(left),
			len(right),
		)
	}
	var product float64
	for index := range left {
		product += float64(left[index]) * float64(right[index])
	}
	return product, nil
}

func normalizeExtensions(extensionFilter []string) map[string]struct{} {
	extensions := make(map[string]struct{}, len(extensionFilter))
	for _, extension := range extensionFilter {
		normalized := strings.TrimSpace(extension)
		if normalized == "" {
			continue
		}
		if !strings.HasPrefix(normalized, ".") {
			normalized = "." + normalized
		}
		extensions[normalized] = struct{}{}
	}
	return extensions
}

func normalizeSearchPrefix(relativePathPrefix string) string {
	return strings.Trim(strings.TrimSpace(relativePathPrefix), "/")
}

func matchesSearchPrefix(relativePath string, prefix string) bool {
	if prefix == "" || prefix == "." {
		return true
	}
	return relativePath == prefix || strings.HasPrefix(relativePath, prefix+"/")
}

func effectiveLimit(limit int32) int {
	if limit <= 0 {
		return defaultSearchLimit
	}
	return int(limit)
}

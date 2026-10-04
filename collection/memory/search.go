package memory

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"sort"

	"goodkind.io/lm-semantic-search/collection"
)

// defaultSearchLimit is the hit count a search returns when the request sets no
// positive limit.
const defaultSearchLimit = 10

// cancellationCheckInterval is the number of rows a scan reads between two
// checks of the request context.
const cancellationCheckInterval = 4096

// filterTruth is the three-valued result of a filter node on one row. A
// comparison on a null or absent value is unknown, and a row matches only when
// the whole tree is true. The Milvus expression evaluator gives the same
// results for the same tree.
type filterTruth int

const (
	truthFalse filterTruth = iota
	truthTrue
	truthUnknown
)

type candidate struct {
	stored *storedRow
	score  float64
}

// Search scores every row the filter matches by its cosine similarity with
// request.Vector. It orders the rows by descending score, then ascending
// relativePath, then ascending ID. It returns at most Limit rows, at most
// PerGroupLimit per GroupBy value, none scoring below a positive MinScore. A
// smaller limit returns a prefix of a larger limit's result. Search ignores
// request.Query.
func (store *Store) Search(ctx context.Context, request collection.SearchRequest) ([]collection.Hit, error) {
	name, err := requireName(request.Collection)
	if err != nil {
		return nil, err
	}
	if _, err := collection.Compile(request.Filter); err != nil {
		slog.ErrorContext(ctx, "validate collection filter failed", "collection", name, "err", err)
		return nil, fmt.Errorf("validate filter for %s: %w", name, err)
	}
	queryNorm, err := vectorNorm(request.Vector)
	if err != nil {
		slog.ErrorContext(ctx, "read query vector failed", "collection", name, "err", err)
		return nil, fmt.Errorf("search %s: query %w", name, err)
	}
	store.mutex.RLock()
	defer store.mutex.RUnlock()
	stored, exists := store.collections[name]
	if !exists {
		return nil, collection.ErrCollectionMissing
	}
	if len(request.Vector) != stored.dimension {
		return nil, fmt.Errorf("search %s: query vector has width %d, want %d", name, len(request.Vector), stored.dimension)
	}
	if queryNorm == 0 {
		return nil, fmt.Errorf("search %s: query vector has zero length", name)
	}
	candidates, err := stored.rank(ctx, name, request, queryNorm)
	if err != nil {
		return nil, err
	}
	return stored.selectHits(candidates, request), nil
}

// rank scores every row the request filter matches and returns the rows in
// ranking order.
func (stored *storedCollection) rank(ctx context.Context, name string, request collection.SearchRequest, queryNorm float64) ([]candidate, error) {
	declared := request.Declaration.Scalars
	candidates := make([]candidate, 0, len(stored.rows))
	scanned := 0
	for _, row := range stored.rows {
		if scanned%cancellationCheckInterval == 0 {
			if err := ctx.Err(); err != nil {
				slog.WarnContext(ctx, "memory collection search cancelled", "collection", name, "err", err)
				return nil, fmt.Errorf("search %s: %w", name, err)
			}
		}
		scanned++
		if request.Filter != nil && stored.evaluate(*request.Filter, row, declared) != truthTrue {
			continue
		}
		candidates = append(candidates, candidate{stored: row, score: cosine(request.Vector, queryNorm, row)})
	}
	sort.Slice(candidates, func(first int, second int) bool {
		left := candidates[first]
		right := candidates[second]
		if left.score != right.score {
			return left.score > right.score
		}
		if left.stored.row.RelativePath != right.stored.row.RelativePath {
			return left.stored.row.RelativePath < right.stored.row.RelativePath
		}
		return left.stored.row.ID < right.stored.row.ID
	})
	return candidates, nil
}

// selectHits walks ranked candidates once and applies the score floor, the
// per-group cap, and the limit.
func (stored *storedCollection) selectHits(candidates []candidate, request collection.SearchRequest) []collection.Hit {
	declared := request.Declaration.Scalars
	limit := int(request.Limit)
	if limit <= 0 {
		limit = defaultSearchLimit
	}
	grouped := request.GroupBy != "" && request.PerGroupLimit > 0
	perGroup := make(map[string]int32)
	hits := make([]collection.Hit, 0, min(limit, len(candidates)))
	for _, ranked := range candidates {
		if len(hits) >= limit {
			break
		}
		if request.MinScore > 0 && ranked.score < request.MinScore {
			break
		}
		if grouped {
			key := stored.groupCell(ranked.stored, request.GroupBy, declared).GroupKey()
			if perGroup[key] >= request.PerGroupLimit {
				continue
			}
			perGroup[key]++
		}
		hits = append(hits, stored.hit(ranked.stored, ranked.score, declared))
	}
	return hits
}

// cosine returns the cosine similarity of the query and a stored row. A stored
// zero vector scores zero.
func cosine(query []float32, queryNorm float64, row *storedRow) float64 {
	if row.norm == 0 {
		return 0
	}
	var dot float64
	for index, component := range query {
		dot += float64(component) * float64(row.row.Vector[index])
	}
	return dot / (queryNorm * row.norm)
}

// Query returns the rows a filter matches in ascending ID order, without
// ranking. A nil filter needs a positive Limit, which is the rule the Milvus
// store enforces.
func (store *Store) Query(ctx context.Context, request collection.QueryRequest) ([]collection.Hit, error) {
	name, err := requireName(request.Collection)
	if err != nil {
		return nil, err
	}
	if _, err := collection.Compile(request.Filter); err != nil {
		slog.ErrorContext(ctx, "validate collection filter failed", "collection", name, "err", err)
		return nil, fmt.Errorf("validate filter for %s: %w", name, err)
	}
	if request.Filter == nil && request.Limit <= 0 {
		return nil, fmt.Errorf("query %s: a query without a filter needs a positive limit", name)
	}
	store.mutex.RLock()
	defer store.mutex.RUnlock()
	stored, exists := store.collections[name]
	if !exists {
		return nil, collection.ErrCollectionMissing
	}
	declared := request.Declaration.Scalars
	hits := make([]collection.Hit, 0)
	for _, id := range stored.sortedIDs() {
		if request.Limit > 0 && len(hits) >= request.Limit {
			break
		}
		row := stored.rows[id]
		if request.Filter != nil && stored.evaluate(*request.Filter, row, declared) != truthTrue {
			continue
		}
		hits = append(hits, stored.hit(row, 0, declared))
	}
	return hits, nil
}

// Delete removes the rows a filter matches and returns the deleted count.
func (store *Store) Delete(ctx context.Context, collectionName string, filter collection.Filter) (int64, error) {
	name, err := requireName(collectionName)
	if err != nil {
		return 0, err
	}
	if _, err := collection.CompileInline(&filter); err != nil {
		slog.ErrorContext(ctx, "validate collection filter failed", "collection", name, "err", err)
		return 0, fmt.Errorf("validate filter for %s: %w", name, err)
	}
	store.mutex.Lock()
	defer store.mutex.Unlock()
	stored, exists := store.collections[name]
	if !exists {
		return 0, collection.ErrCollectionMissing
	}
	var removed int64
	for id, row := range stored.rows {
		if stored.evaluate(filter, row, stored.scalars) != truthTrue {
			continue
		}
		delete(stored.rows, id)
		removed++
	}
	return removed, nil
}

func (stored *storedCollection) sortedIDs() []string {
	ids := make([]string, 0, len(stored.rows))
	for id := range stored.rows {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	return ids
}

func (stored *storedCollection) hit(row *storedRow, score float64, declared []collection.ScalarColumn) collection.Hit {
	cells := make(map[string]collection.ScalarCell, len(declared))
	for _, column := range declared {
		cells[column.Name] = stored.cell(row, column.Name, declared)
	}
	return collection.Hit{
		ID:                row.row.ID,
		Content:           row.row.Content,
		Score:             score,
		RelativePath:      row.row.RelativePath,
		StartLine:         row.row.StartLine,
		EndLine:           row.row.EndLine,
		FileExtension:     row.row.FileExtension,
		Metadata:          row.row.Metadata,
		SplitPart:         row.row.SplitPart,
		SplitPartRecorded: row.row.SplitPartRecorded,
		Scalars:           cells,
	}
}

// cell returns the row's cell for a column of the request declaration. The
// cell is absent when the request does not declare the column, when the
// collection has no such column, or when the request declares another type
// than the collection stores. The cell is null when the collection has the
// column and the row stores no value for it.
func (stored *storedCollection) cell(row *storedRow, columnName string, declared []collection.ScalarColumn) collection.ScalarCell {
	position := slices.IndexFunc(declared, func(column collection.ScalarColumn) bool {
		return column.Name == columnName
	})
	if position < 0 {
		return collection.AbsentCell(columnName)
	}
	schemaColumn, inSchema := stored.column(columnName)
	if !inSchema || schemaColumn.Type != declared[position].Type {
		return collection.AbsentCell(columnName)
	}
	value, stores := row.row.Scalars[columnName]
	if !stores {
		return collection.NullCell(columnName)
	}
	return collection.ValueCell(columnName, value)
}

// groupCell returns the cell the per-group cap reads. A group column the
// request does not declare reads with the type the collection stores.
func (stored *storedCollection) groupCell(row *storedRow, columnName string, declared []collection.ScalarColumn) collection.ScalarCell {
	for _, column := range declared {
		if column.Name == columnName {
			return stored.cell(row, columnName, declared)
		}
	}
	return stored.cell(row, columnName, stored.scalars)
}

// evaluate evaluates a validated filter tree on one row with three-valued
// logic. A not node inverts true and false and keeps unknown. An all node is
// false when any child is false, else unknown when any child is unknown. An any
// node is true when any child is true, else unknown when any child is unknown.
func (stored *storedCollection) evaluate(filter collection.Filter, row *storedRow, declared []collection.ScalarColumn) filterTruth {
	switch filter.Kind {
	case collection.FilterAll:
		result := truthTrue
		for _, child := range filter.Children {
			switch stored.evaluate(child, row, declared) {
			case truthFalse:
				return truthFalse
			case truthUnknown:
				result = truthUnknown
			case truthTrue:
			}
		}
		return result
	case collection.FilterAny:
		result := truthFalse
		for _, child := range filter.Children {
			switch stored.evaluate(child, row, declared) {
			case truthTrue:
				return truthTrue
			case truthUnknown:
				result = truthUnknown
			case truthFalse:
			}
		}
		return result
	case collection.FilterNot:
		if len(filter.Children) != 1 {
			return truthFalse
		}
		switch stored.evaluate(filter.Children[0], row, declared) {
		case truthTrue:
			return truthFalse
		case truthFalse:
			return truthTrue
		case truthUnknown:
			return truthUnknown
		default:
			return truthUnknown
		}
	case collection.FilterIsNull:
		return truthOf(stored.cell(row, filter.Column, declared).State != collection.ScalarCellValue)
	case collection.FilterIsPresent:
		return truthOf(stored.cell(row, filter.Column, declared).State == collection.ScalarCellValue)
	case collection.FilterEquals, collection.FilterIn, collection.FilterRange:
		cell := stored.cell(row, filter.Column, declared)
		if cell.State != collection.ScalarCellValue {
			return truthUnknown
		}
		return truthOf(comparisonMatches(filter, cell.Value))
	default:
		return truthFalse
	}
}

func comparisonMatches(filter collection.Filter, value collection.ScalarValue) bool {
	if filter.Kind == collection.FilterRange {
		if filter.Lower != nil && value.Int64 < *filter.Lower {
			return false
		}
		if filter.Upper != nil && value.Int64 >= *filter.Upper {
			return false
		}
		return true
	}
	return slices.Contains(filter.Values, value)
}

func truthOf(matches bool) filterTruth {
	if matches {
		return truthTrue
	}
	return truthFalse
}

var errNoItemSelection = errors.New("request selects no item and no path prefix")

package localvec

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"strings"

	lmcollection "goodkind.io/lm-semantic-search/collection"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// filterTruth is the three-valued result of a filter node on one row. A
// comparison on a null or absent value is unknown, and a row matches only
// when the whole tree is true. The Milvus expression evaluator gives the same
// results for the same tree.
type filterTruth int

const (
	truthFalse filterTruth = iota
	truthTrue
	truthUnknown
)

// SearchCollection runs a typed search of a local collection. It ranks a
// fixed candidate set that never depends on the limit, the group cap, or the
// score floor, at the depth the Milvus collection search ranks. When at most
// lmcollection.RankingDepth rows match the filter tree, the candidates
// are every matching row, scored exactly. Otherwise the candidates are the
// lmcollection.RankingDepth nearest rows from the HNSW index. It sorts
// the candidates with sortScoredRows and walks them once to keep the rows
// that match the filter tree and score at or above MinScore, at most
// PerGroupLimit per GroupBy value, up to Limit rows. A smaller limit therefore
// returns a prefix of a larger one at any collection size. A local row stores
// the conversation scalar fields, so each hit decodes a declared conversation
// column from the row and reports every other declared column as absent.
func (store *Store) SearchCollection(
	ctx context.Context,
	search semantic.CollectionSearch,
) ([]semantic.CollectionHit, error) {
	if err := operationContextError(ctx, "search local collection"); err != nil {
		return nil, err
	}
	collectionName := strings.TrimSpace(search.CollectionName)
	stored, err := store.collectionForName(collectionName, false)
	if err != nil {
		return nil, err
	}
	_, exists, err := stored.vectorCount()
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, semantic.ErrCollectionMissing
	}
	query, err := store.embedQuery(ctx, collectionName, search.Query)
	if err != nil {
		return nil, err
	}
	declared := search.Declaration.Scalars
	matchesFilter := func(candidate row) bool {
		return search.Filter == nil || evaluateFilter(*search.Filter, candidate, declared) == truthTrue
	}
	candidates, err := stored.rankCandidates(ctx, query, matchesFilter, lmcollection.RankingDepth)
	if err != nil {
		return nil, err
	}
	sortScoredRows(candidates)
	groupLimit := int32(0)
	if search.GroupBy != "" && search.PerGroupLimit > 0 {
		groupLimit = search.PerGroupLimit
	}
	perGroup := make(map[string]int32)
	scored := limitScoredRows(
		candidates,
		effectiveLimit(search.Limit),
		func(candidate scoredRow) bool {
			if search.MinScore > 0 && candidate.score < search.MinScore {
				return false
			}
			if !matchesFilter(candidate.stored) {
				return false
			}
			if groupLimit <= 0 {
				return true
			}
			key := rowScalarCell(candidate.stored, search.GroupBy, declared).GroupKey()
			if perGroup[key] >= groupLimit {
				return false
			}
			perGroup[key]++
			return true
		},
	)
	hits := make([]semantic.CollectionHit, 0, len(scored))
	for _, candidate := range scored {
		cells := make([]lmcollection.ScalarCell, 0, len(declared))
		for _, column := range declared {
			cells = append(cells, rowScalarCell(candidate.stored, column.Name, declared))
		}
		hits = append(hits, semantic.CollectionHit{
			Chunk:   candidate.stored.chunk(candidate.score),
			Scalars: cells,
		})
	}
	return hits, nil
}

// rankCandidates returns the ranking candidates for query. It evaluates keep
// on every stored row under the collection lock. When at most depth rows
// match, the candidates are exactly the matching rows, each scored by its dot
// product with query. Otherwise the candidates are the depth nearest rows from
// the HNSW index, matching or not, and the caller applies keep again while it
// walks them. The candidate set depends only on query, keep, depth, and the
// stored rows. The returned rows omit their vectors.
func (stored *collection) rankCandidates(
	ctx context.Context,
	query []float32,
	keep func(row) bool,
	depth int,
) ([]scoredRow, error) {
	stored.mutex.Lock()
	defer stored.mutex.Unlock()
	if err := stored.loadLocked(); err != nil {
		return nil, err
	}
	if !stored.exists {
		return nil, semantic.ErrCollectionMissing
	}
	matching := make([]int, 0, min(len(stored.rows), depth))
	for rowIndex := range stored.rows {
		if !keep(stored.rows[rowIndex]) {
			continue
		}
		if len(matching) == depth {
			return stored.nearestCandidatesLocked(query, depth)
		}
		matching = append(matching, rowIndex)
	}
	return stored.scoreRowsLocked(ctx, query, matching)
}

// scoreRowsLocked scores the rows at rowIndexes by their dot product with
// query. Caller must hold stored.mutex.
func (stored *collection) scoreRowsLocked(
	ctx context.Context,
	query []float32,
	rowIndexes []int,
) ([]scoredRow, error) {
	scored := make([]scoredRow, 0, len(rowIndexes))
	for _, rowIndex := range rowIndexes {
		if err := operationContextError(ctx, "score local collection rows"); err != nil {
			return nil, err
		}
		candidate := stored.rows[rowIndex]
		score, err := dotProduct(query, candidate.Vector)
		if err != nil {
			slog.ErrorContext(ctx, "score local collection row failed", "collection", stored.name, "row_id", candidate.ID, "err", err)
			return nil, fmt.Errorf("score local collection row %s: %w", candidate.ID, err)
		}
		candidate.Vector = nil
		scored = append(scored, scoredRow{stored: candidate, score: score})
	}
	return scored, nil
}

// nearestCandidatesLocked returns the depth nearest rows from the HNSW index,
// scored as one minus their cosine distance. Caller must hold stored.mutex.
func (stored *collection) nearestCandidatesLocked(query []float32, depth int) ([]scoredRow, error) {
	keys, distances, err := stored.index.Search(query, depth)
	if err != nil {
		slog.Error("search local collection index failed", "collection", stored.name, "depth", depth, "err", err)
		return nil, fmt.Errorf("search usearch index for local collection %s: %w", stored.name, err)
	}
	rowsByLabel := make(map[uint64]int, len(stored.rows))
	for rowIndex, candidate := range stored.rows {
		rowsByLabel[candidate.Label] = rowIndex
	}
	scored := make([]scoredRow, 0, len(keys))
	for index, key := range keys {
		rowIndex, found := rowsByLabel[key]
		if !found {
			return nil, fmt.Errorf("usearch returned unknown label %d for %s", key, stored.name)
		}
		candidate := stored.rows[rowIndex]
		candidate.Vector = nil
		scored = append(scored, scoredRow{stored: candidate, score: 1 - float64(distances[index])})
	}
	return scored, nil
}

// rowScalarCell returns the row's cell for a declared column. The row stores
// the conversation scalar fields concretely. provider comes from the
// conversation id prefix and role is lowercased, as the Milvus insert writes
// them. A column the declaration omits, a column the row format lacks, and a
// column declared with a different type are absent.
func rowScalarCell(stored row, columnName string, declared []lmcollection.ScalarColumn) lmcollection.ScalarCell {
	declaredType, found := declaredColumnType(declared, columnName)
	if !found {
		return lmcollection.AbsentCell(columnName)
	}
	value, stores := conversationRowValue(stored, columnName)
	if !stores || value.Type != declaredType {
		return lmcollection.AbsentCell(columnName)
	}
	return lmcollection.ValueCell(columnName, value)
}

func declaredColumnType(declared []lmcollection.ScalarColumn, columnName string) (lmcollection.ScalarType, bool) {
	for _, column := range declared {
		if column.Name == columnName {
			return column.Type, true
		}
	}
	return "", false
}

// conversationRowColumn is the closed set of conversation scalar columns a
// local row stores. The names match the conversation declaration.
type conversationRowColumn string

const (
	rowColumnConversationID       conversationRowColumn = "conversationId"
	rowColumnParentConversationID conversationRowColumn = "parentConversationId"
	rowColumnRole                 conversationRowColumn = "role"
	rowColumnProvider             conversationRowColumn = "provider"
	rowColumnWorkspaceRoot        conversationRowColumn = "workspaceRoot"
	rowColumnArchived             conversationRowColumn = "archived"
	rowColumnTimestampUnix        conversationRowColumn = "timestampUnix"
	rowColumnMessageIndex         conversationRowColumn = "messageIndex"
	rowColumnLoadRules            conversationRowColumn = "loadRules"
)

func conversationRowValue(stored row, columnName string) (lmcollection.ScalarValue, bool) {
	switch conversationRowColumn(columnName) {
	case rowColumnConversationID:
		return lmcollection.StringScalar(stored.ConversationID), true
	case rowColumnParentConversationID:
		return lmcollection.StringScalar(stored.ParentConversationID), true
	case rowColumnRole:
		return lmcollection.StringScalar(strings.ToLower(stored.Role)), true
	case rowColumnProvider:
		return lmcollection.StringScalar(conversationProvider(stored.ConversationID)), true
	case rowColumnWorkspaceRoot:
		return lmcollection.StringScalar(stored.WorkspaceRoot), true
	case rowColumnArchived:
		return lmcollection.BoolScalar(stored.Archived), true
	case rowColumnTimestampUnix:
		return lmcollection.Int64Scalar(stored.TimestampUnix), true
	case rowColumnMessageIndex:
		return lmcollection.Int64Scalar(int64(stored.MessageIndex)), true
	case rowColumnLoadRules:
		return lmcollection.StringScalar(stored.LoadRules), true
	default:
		return lmcollection.ScalarValue{Type: "", String: "", Bool: false, Int64: 0}, false
	}
}

// evaluateFilter evaluates a validated filter tree on one row with three-valued
// logic. A not node inverts true and false and keeps unknown. An all node is
// false when any child is false, else unknown when any child is unknown. An any
// node is true when any child is true, else unknown when any child is unknown.
func evaluateFilter(filter lmcollection.Filter, stored row, declared []lmcollection.ScalarColumn) filterTruth {
	switch filter.Kind {
	case lmcollection.FilterAll:
		result := truthTrue
		for _, child := range filter.Children {
			switch evaluateFilter(child, stored, declared) {
			case truthFalse:
				return truthFalse
			case truthUnknown:
				result = truthUnknown
			case truthTrue:
			}
		}
		return result
	case lmcollection.FilterAny:
		result := truthFalse
		for _, child := range filter.Children {
			switch evaluateFilter(child, stored, declared) {
			case truthTrue:
				return truthTrue
			case truthUnknown:
				result = truthUnknown
			case truthFalse:
			}
		}
		return result
	case lmcollection.FilterNot:
		if len(filter.Children) != 1 {
			return truthFalse
		}
		switch evaluateFilter(filter.Children[0], stored, declared) {
		case truthTrue:
			return truthFalse
		case truthFalse:
			return truthTrue
		case truthUnknown:
			return truthUnknown
		default:
			return truthUnknown
		}
	case lmcollection.FilterIsNull:
		return truthOf(rowScalarCell(stored, filter.Column, declared).State != lmcollection.ScalarCellValue)
	case lmcollection.FilterIsPresent:
		return truthOf(rowScalarCell(stored, filter.Column, declared).State == lmcollection.ScalarCellValue)
	case lmcollection.FilterEquals, lmcollection.FilterIn, lmcollection.FilterRange:
		cell := rowScalarCell(stored, filter.Column, declared)
		if cell.State != lmcollection.ScalarCellValue {
			return truthUnknown
		}
		return truthOf(comparisonMatches(filter, cell.Value))
	default:
		return truthFalse
	}
}

func comparisonMatches(filter lmcollection.Filter, value lmcollection.ScalarValue) bool {
	if filter.Kind == lmcollection.FilterRange {
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

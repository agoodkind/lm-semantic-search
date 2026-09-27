package localvec

import (
	"context"
	"slices"
	"strings"

	"goodkind.io/lm-semantic-search/internal/model"
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

// SearchCollection runs a typed search of a local collection. It keeps the
// rows that the filter tree matches, scoring at or above MinScore, at most
// PerGroupLimit per GroupBy value, up to Limit rows. A local row stores the
// conversation scalar fields, so each hit decodes a declared conversation
// column from the row and reports every other declared column as absent.
func (store *Store) SearchCollection(
	ctx context.Context,
	search semantic.CollectionSearch,
) ([]semantic.CollectionHit, error) {
	declared := search.Declaration.Scalars
	groupLimit := int32(0)
	if search.GroupBy != "" && search.PerGroupLimit > 0 {
		groupLimit = search.PerGroupLimit
	}
	scored, err := store.searchRows(
		ctx,
		strings.TrimSpace(search.CollectionName),
		search.Query,
		search.Limit,
		func(scored []scoredRow, resultLimit int) []scoredRow {
			perGroup := make(map[string]int32)
			return limitScoredRows(
				scored,
				resultLimit,
				func(candidate scoredRow) bool {
					if search.MinScore > 0 && candidate.score < search.MinScore {
						return false
					}
					if search.Filter != nil && evaluateFilter(*search.Filter, candidate.stored, declared) != truthTrue {
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
		},
	)
	if err != nil {
		return nil, err
	}
	hits := make([]semantic.CollectionHit, 0, len(scored))
	for _, candidate := range scored {
		cells := make([]semantic.ScalarCell, 0, len(declared))
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

// rowScalarCell returns the row's cell for a declared column. The row stores
// the conversation scalar fields concretely. provider comes from the
// conversation id prefix and role is lowercased, as the Milvus insert writes
// them. A column the declaration omits, a column the row format lacks, and a
// column declared with a different type are absent.
func rowScalarCell(stored row, columnName string, declared []model.ScalarColumn) semantic.ScalarCell {
	declaredType, found := declaredColumnType(declared, columnName)
	if !found {
		return semantic.AbsentCell(columnName)
	}
	value, stores := conversationRowValue(stored, columnName)
	if !stores || value.Type != declaredType {
		return semantic.AbsentCell(columnName)
	}
	return semantic.ValueCell(columnName, value)
}

func declaredColumnType(declared []model.ScalarColumn, columnName string) (model.ScalarType, bool) {
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

func conversationRowValue(stored row, columnName string) (semantic.ScalarValue, bool) {
	switch conversationRowColumn(columnName) {
	case rowColumnConversationID:
		return semantic.StringScalar(stored.ConversationID), true
	case rowColumnParentConversationID:
		return semantic.StringScalar(stored.ParentConversationID), true
	case rowColumnRole:
		return semantic.StringScalar(strings.ToLower(stored.Role)), true
	case rowColumnProvider:
		return semantic.StringScalar(conversationProvider(stored.ConversationID)), true
	case rowColumnWorkspaceRoot:
		return semantic.StringScalar(stored.WorkspaceRoot), true
	case rowColumnArchived:
		return semantic.BoolScalar(stored.Archived), true
	case rowColumnTimestampUnix:
		return semantic.Int64Scalar(stored.TimestampUnix), true
	case rowColumnMessageIndex:
		return semantic.Int64Scalar(int64(stored.MessageIndex)), true
	case rowColumnLoadRules:
		return semantic.StringScalar(stored.LoadRules), true
	default:
		return semantic.ScalarValue{Type: "", String: "", Bool: false, Int64: 0}, false
	}
}

// evaluateFilter evaluates a validated filter tree on one row with three-valued
// logic. A not node inverts true and false and keeps unknown. An all node is
// false when any child is false, else unknown when any child is unknown. An any
// node is true when any child is true, else unknown when any child is unknown.
func evaluateFilter(filter semantic.CollectionFilter, stored row, declared []model.ScalarColumn) filterTruth {
	switch filter.Kind {
	case semantic.CollectionFilterAll:
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
	case semantic.CollectionFilterAny:
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
	case semantic.CollectionFilterNot:
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
	case semantic.CollectionFilterIsNull:
		return truthOf(rowScalarCell(stored, filter.Column, declared).State != semantic.ScalarCellValue)
	case semantic.CollectionFilterIsPresent:
		return truthOf(rowScalarCell(stored, filter.Column, declared).State == semantic.ScalarCellValue)
	case semantic.CollectionFilterEquals, semantic.CollectionFilterIn, semantic.CollectionFilterRange:
		cell := rowScalarCell(stored, filter.Column, declared)
		if cell.State != semantic.ScalarCellValue {
			return truthUnknown
		}
		return truthOf(comparisonMatches(filter, cell.Value))
	default:
		return truthFalse
	}
}

func comparisonMatches(filter semantic.CollectionFilter, value semantic.ScalarValue) bool {
	if filter.Kind == semantic.CollectionFilterRange {
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

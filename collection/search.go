package collection

import "strconv"

// RankingDepth is the number of candidates one collection search ranks: the
// topK of each hybrid leg, the fused hybrid limit, and the dense topK. It is
// the Milvus single-search ceiling. It never depends on the requested limit,
// the group cap, or the score floor. Every request for one query and filter
// therefore ranks the same candidate list. The offline store ranks at the
// same depth.
const RankingDepth = 16384

// nullGroupKey is the group key of every hit with a null or absent group
// column value. Those hits share one group.
const nullGroupKey = "null"

// ScalarCellState is the closed set of states of one declared scalar column on
// a stored row.
type ScalarCellState string

const (
	// ScalarCellAbsent means the stored row has no such column.
	ScalarCellAbsent ScalarCellState = "absent"
	// ScalarCellNull means the stored row has the column and its value is null.
	ScalarCellNull ScalarCellState = "null"
	// ScalarCellValue means the stored row has a concrete value in Value.
	ScalarCellValue ScalarCellState = "value"
)

// ScalarCell is one declared scalar column value on a search hit.
type ScalarCell struct {
	Column string
	State  ScalarCellState
	Value  ScalarValue
}

// AbsentCell returns the cell of a column the stored row does not have.
func AbsentCell(column string) ScalarCell {
	return ScalarCell{Column: column, State: ScalarCellAbsent, Value: EmptyScalar()}
}

// NullCell returns the cell of a column with a null value.
func NullCell(column string) ScalarCell {
	return ScalarCell{Column: column, State: ScalarCellNull, Value: EmptyScalar()}
}

// ValueCell returns the cell of a column with a concrete value.
func ValueCell(column string, value ScalarValue) ScalarCell {
	return ScalarCell{Column: column, State: ScalarCellValue, Value: value}
}

// GroupKey returns the key that groups hits by this cell's value for a
// per-group cap. Every null or absent cell returns one shared key. Values of
// different types never share a key.
func (cell ScalarCell) GroupKey() string {
	if cell.State != ScalarCellValue {
		return nullGroupKey
	}
	switch cell.Value.Type {
	case ScalarTypeBool:
		return "bool:" + strconv.FormatBool(cell.Value.Bool)
	case ScalarTypeInt64:
		return "int64:" + strconv.FormatInt(cell.Value.Int64, 10)
	case ScalarTypeString:
		return "string:" + cell.Value.String
	default:
		return "string:" + cell.Value.String
	}
}

// SearchRequest is one typed search of a collection. Vector is the dense query
// embedding and Query is the raw text of the lexical (BM25) leg. Filter is nil
// to match every row. PerGroupLimit caps the hits that share one GroupBy
// value, and zero means uncapped. Declaration is the collection's declared
// scalar schema.
type SearchRequest struct {
	Collection    string
	Query         string
	Vector        []float32
	Limit         int32
	MinScore      float64
	Filter        *Filter
	GroupBy       string
	PerGroupLimit int32
	Declaration   Declaration
}

// Hit is one stored row, with Score set on a ranked search hit. Metadata is the
// stored metadata JSON. Scalars maps each declared scalar column name to the
// row's cell for it.
type Hit struct {
	ID                string
	Content           string
	Score             float64
	RelativePath      string
	StartLine         int32
	EndLine           int32
	FileExtension     string
	Metadata          string
	SplitPart         int32
	SplitPartRecorded bool
	Scalars           map[string]ScalarCell
}

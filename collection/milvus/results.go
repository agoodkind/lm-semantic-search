package milvus

import (
	"fmt"
	"log/slog"
	"math"
	"strings"
	"unicode/utf8"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
)

// SafeInt32 converts value to int32 and clamps it to the int32 range.
func SafeInt32(value int64) int32 {
	if value > math.MaxInt32 {
		return math.MaxInt32
	}
	if value < math.MinInt32 {
		return math.MinInt32
	}
	return int32(value)
}

// SanitizeUTF8 returns a copy of value with invalid UTF-8 byte sequences
// replaced by the Unicode replacement character. Milvus rejects a VarChar
// payload with invalid UTF-8. Chunk content can cut a multi-byte codepoint, for
// example at a tree-sitter byte offset. The second return value reports whether
// the input needed repair.
func SanitizeUTF8(value string) (string, bool) {
	if utf8.ValidString(value) {
		return value, false
	}
	return strings.ToValidUTF8(value, "�"), true
}

// SplitPartAt reads the nullable splitPart value at one row. A missing column
// and a null value both report not recorded.
func SplitPartAt(splitPartColumn column.Column, rowIndex int) (int32, bool, error) {
	if splitPartColumn == nil {
		return 0, false, nil
	}
	isNull, nullErr := splitPartColumn.IsNull(rowIndex)
	if nullErr != nil {
		slog.Error("read split part null state failed", "row", rowIndex, "err", nullErr)
		return 0, false, fmt.Errorf("read split part null state at %d: %w", rowIndex, nullErr)
	}
	if isNull {
		return 0, false, nil
	}
	value, valueErr := splitPartColumn.GetAsInt64(rowIndex)
	if valueErr != nil {
		slog.Error("read split part column failed", "row", rowIndex, "err", valueErr)
		return 0, false, fmt.Errorf("read split part column at %d: %w", rowIndex, valueErr)
	}
	return SafeInt32(value), true, nil
}

// VectorAt extracts one float-vector row from a Milvus result column. The
// client's typed Column surface exposes the row through Get(int). For a
// dense FloatVector column the returned value is entity.FloatVector,
// which is a []float32 with a named type.
func VectorAt(vectorColumn column.Column, rowIndex int) ([]float32, error) {
	raw, err := vectorColumn.Get(rowIndex)
	if err != nil {
		slog.Error("read vector row failed", "row", rowIndex, "err", err)
		return nil, fmt.Errorf("read vector row %d: %w", rowIndex, err)
	}
	switch typed := raw.(type) {
	case entity.FloatVector:
		out := make([]float32, len(typed))
		copy(out, typed)
		return out, nil
	case []float32:
		out := make([]float32, len(typed))
		copy(out, typed)
		return out, nil
	}
	err = fmt.Errorf("unexpected vector row type %T", raw)
	slog.Error("vector row type unexpected", "row", rowIndex, "err", err)
	return nil, err
}

// scalarCellsAt decodes one row's cell for every declared scalar column. A
// column missing from the result set is absent. A null value is null.
func scalarCellsAt(resultSet milvusclient.ResultSet, scalarColumns []collection.ScalarColumn, rowIndex int) ([]collection.ScalarCell, error) {
	cells := make([]collection.ScalarCell, 0, len(scalarColumns))
	for _, declared := range scalarColumns {
		cell, err := ScalarCellAt(resultSet.GetColumn(declared.Name), declared, rowIndex)
		if err != nil {
			return nil, err
		}
		cells = append(cells, cell)
	}
	return cells, nil
}

// ScalarCellAt decodes one row's cell for a declared scalar column. A column
// missing from the result set is absent. A null value is null.
func ScalarCellAt(valueColumn column.Column, declared collection.ScalarColumn, rowIndex int) (collection.ScalarCell, error) {
	if valueColumn == nil {
		return collection.AbsentCell(declared.Name), nil
	}
	absent := collection.AbsentCell(declared.Name)
	isNull, err := valueColumn.IsNull(rowIndex)
	if err != nil {
		return absent, scalarReadError(declared, rowIndex, "null state", err)
	}
	if isNull {
		return collection.NullCell(declared.Name), nil
	}
	switch declared.Type {
	case collection.ScalarTypeString:
		value, valueErr := valueColumn.GetAsString(rowIndex)
		if valueErr != nil {
			return absent, scalarReadError(declared, rowIndex, "string value", valueErr)
		}
		return collection.ValueCell(declared.Name, collection.StringScalar(value)), nil
	case collection.ScalarTypeBool:
		value, valueErr := valueColumn.GetAsBool(rowIndex)
		if valueErr != nil {
			return absent, scalarReadError(declared, rowIndex, "bool value", valueErr)
		}
		return collection.ValueCell(declared.Name, collection.BoolScalar(value)), nil
	case collection.ScalarTypeInt64:
		value, valueErr := valueColumn.GetAsInt64(rowIndex)
		if valueErr != nil {
			return absent, scalarReadError(declared, rowIndex, "int64 value", valueErr)
		}
		return collection.ValueCell(declared.Name, collection.Int64Scalar(value)), nil
	default:
		return absent, scalarReadError(declared, rowIndex, "value", fmt.Errorf("unsupported declared type %q", declared.Type))
	}
}

func scalarReadError(declared collection.ScalarColumn, rowIndex int, part string, err error) error {
	slog.Error("read declared scalar column failed", "column", declared.Name, "index", rowIndex, "part", part, "err", err)
	return fmt.Errorf("read %s of scalar column %s at %d: %w", part, declared.Name, rowIndex, err)
}

// ScalarValueAt reads the value of one declared column at a row of a query
// result, with its null state.
func ScalarValueAt(valueColumn column.Column, declaration collection.ScalarColumn, rowIndex int) (collection.ScalarValue, error) {
	value := collection.ScalarValue{Type: declaration.Type, Null: false, String: "", Bool: false, Int64: 0}
	if valueColumn == nil {
		return value, collection.ErrSearchResultIncomplete
	}
	isNull, err := valueColumn.IsNull(rowIndex)
	if err != nil {
		slog.Error("read declared column null state failed", "column", declaration.Name, "index", rowIndex, "err", err)
		return value, fmt.Errorf("read null state of %s at %d: %w", declaration.Name, rowIndex, err)
	}
	if isNull {
		value.Null = true
		return value, nil
	}
	switch declaration.Type {
	case collection.ScalarTypeString:
		value.String, err = valueColumn.GetAsString(rowIndex)
	case collection.ScalarTypeBool:
		value.Bool, err = valueColumn.GetAsBool(rowIndex)
	case collection.ScalarTypeInt64:
		value.Int64, err = valueColumn.GetAsInt64(rowIndex)
	default:
		err = fmt.Errorf("unsupported declared column type %q", declaration.Type)
	}
	if err != nil {
		slog.Error("read declared column value failed", "column", declaration.Name, "index", rowIndex, "err", err)
		return value, fmt.Errorf("read %s at %d: %w", declaration.Name, rowIndex, err)
	}
	return value, nil
}

// HitsFromResultSet decodes every row of one result set into hits. Content,
// relativePath, startLine, endLine, and fileExtension are required. The id,
// metadata, and splitPart columns are optional. Scores comes from the result
// set when present. Each declared scalar column becomes one cell, absent when
// the result set lacks the column.
func HitsFromResultSet(resultSet milvusclient.ResultSet, scalarColumns []collection.ScalarColumn) ([]collection.Hit, error) {
	if resultSet.ResultCount == 0 {
		return []collection.Hit{}, nil
	}
	contentColumn := resultSet.GetColumn(ContentField)
	relativePathColumn := resultSet.GetColumn(RelativePathField)
	startLineColumn := resultSet.GetColumn(StartLineField)
	endLineColumn := resultSet.GetColumn(EndLineField)
	fileExtensionColumn := resultSet.GetColumn(FileExtensionField)
	metadataColumn := resultSet.GetColumn(MetadataField)
	splitPartColumn := resultSet.GetColumn(SplitPartField)
	idColumn := resultSet.GetColumn(IDField)
	if contentColumn == nil || relativePathColumn == nil || startLineColumn == nil || endLineColumn == nil || fileExtensionColumn == nil {
		return nil, collection.ErrSearchResultIncomplete
	}

	hits := make([]collection.Hit, 0, resultSet.ResultCount)
	for index := range resultSet.ResultCount {
		hit, err := hitAt(resultSet, index, scalarColumns, hitColumns{
			id:            idColumn,
			content:       contentColumn,
			relativePath:  relativePathColumn,
			startLine:     startLineColumn,
			endLine:       endLineColumn,
			fileExtension: fileExtensionColumn,
			metadata:      metadataColumn,
			splitPart:     splitPartColumn,
		})
		if err != nil {
			return nil, err
		}
		hits = append(hits, hit)
	}
	return hits, nil
}

// hitColumns groups the columns one hit decodes from. The id, metadata, and
// splitPart columns may be nil.
type hitColumns struct {
	id            column.Column
	content       column.Column
	relativePath  column.Column
	startLine     column.Column
	endLine       column.Column
	fileExtension column.Column
	metadata      column.Column
	splitPart     column.Column
}

func hitAt(resultSet milvusclient.ResultSet, index int, scalarColumns []collection.ScalarColumn, columns hitColumns) (collection.Hit, error) {
	contentValue, err := columns.content.GetAsString(index)
	if err != nil {
		slog.Error("read content column failed", "index", index, "err", err)
		return collection.Hit{}, fmt.Errorf("read content column at %d: %w", index, err)
	}
	relativePathValue, err := columns.relativePath.GetAsString(index)
	if err != nil {
		slog.Error("read relative path column failed", "index", index, "err", err)
		return collection.Hit{}, fmt.Errorf("read relative path column at %d: %w", index, err)
	}
	startLineValue, err := columns.startLine.GetAsInt64(index)
	if err != nil {
		slog.Error("read start line column failed", "index", index, "err", err)
		return collection.Hit{}, fmt.Errorf("read start line column at %d: %w", index, err)
	}
	endLineValue, err := columns.endLine.GetAsInt64(index)
	if err != nil {
		slog.Error("read end line column failed", "index", index, "err", err)
		return collection.Hit{}, fmt.Errorf("read end line column at %d: %w", index, err)
	}
	fileExtensionValue, err := columns.fileExtension.GetAsString(index)
	if err != nil {
		slog.Error("read file extension column failed", "index", index, "err", err)
		return collection.Hit{}, fmt.Errorf("read file extension column at %d: %w", index, err)
	}
	metadataValue := ""
	if columns.metadata != nil {
		rawMetadata, metadataErr := columns.metadata.GetAsString(index)
		if metadataErr == nil {
			metadataValue = rawMetadata
		}
	}
	splitPartValue, splitPartRecorded, splitPartErr := SplitPartAt(columns.splitPart, index)
	if splitPartErr != nil {
		return collection.Hit{}, splitPartErr
	}
	idValue := ""
	if columns.id != nil {
		idValue, err = columns.id.GetAsString(index)
		if err != nil {
			slog.Error("read id column failed", "index", index, "err", err)
			return collection.Hit{}, fmt.Errorf("read id column at %d: %w", index, err)
		}
	}
	score := 0.0
	if index < len(resultSet.Scores) {
		score = float64(resultSet.Scores[index])
	}
	cells, cellErr := scalarCellsAt(resultSet, scalarColumns, index)
	if cellErr != nil {
		return collection.Hit{}, cellErr
	}
	scalars := make(map[string]collection.ScalarCell, len(cells))
	for _, cell := range cells {
		scalars[cell.Column] = cell
	}
	return collection.Hit{
		ID:                idValue,
		Content:           contentValue,
		Score:             score,
		RelativePath:      relativePathValue,
		StartLine:         SafeInt32(startLineValue),
		EndLine:           SafeInt32(endLineValue),
		FileExtension:     fileExtensionValue,
		Metadata:          metadataValue,
		SplitPart:         splitPartValue,
		SplitPartRecorded: splitPartRecorded,
		Scalars:           scalars,
	}, nil
}

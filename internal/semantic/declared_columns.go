package semantic

import (
	"fmt"
	"log/slog"

	"github.com/milvus-io/milvus/client/v2/column"
	"goodkind.io/lm-semantic-search/internal/model"
)

// declaredScalarInsertColumns builds one Milvus insert column per declared
// scalar column from the Scalars map of each chunk. A nullable column marks a
// missing or null value invalid. A column that is not nullable rejects a
// missing or null value, and every column rejects a value of another type.
func declaredScalarInsertColumns(
	collectionName string,
	declared []model.ScalarColumn,
	chunks []model.StoredChunk,
) ([]column.Column, error) {
	columns := make([]column.Column, 0, len(declared))
	for _, declaration := range declared {
		built, err := declaredScalarInsertColumn(collectionName, declaration, chunks)
		if err != nil {
			return nil, err
		}
		columns = append(columns, built)
	}
	return columns, nil
}

func declaredScalarInsertColumn(
	collectionName string,
	declaration model.ScalarColumn,
	chunks []model.StoredChunk,
) (column.Column, error) {
	validData := make([]bool, 0, len(chunks))
	stringValues := make([]string, 0, len(chunks))
	boolValues := make([]bool, 0, len(chunks))
	int64Values := make([]int64, 0, len(chunks))
	for _, chunk := range chunks {
		value, present := chunk.Scalars[declaration.Name]
		valid := present && !value.Null
		if !valid && !declaration.Nullable {
			return nil, fmt.Errorf("insert into %s: row %s has no value for column %s, which is not nullable", collectionName, chunk.RelativePath, declaration.Name)
		}
		if valid && value.Type != declaration.Type {
			return nil, fmt.Errorf("insert into %s: row %s has a %s value for %s column %s", collectionName, chunk.RelativePath, value.Type, declaration.Type, declaration.Name)
		}
		validData = append(validData, valid)
		stringValue, _ := sanitizeUTF8(value.String)
		stringValues = append(stringValues, stringValue)
		boolValues = append(boolValues, value.Bool)
		int64Values = append(int64Values, value.Int64)
	}
	return newDeclaredColumn(collectionName, declaration, validData, stringValues, boolValues, int64Values)
}

func newDeclaredColumn(
	collectionName string,
	declaration model.ScalarColumn,
	validData []bool,
	stringValues []string,
	boolValues []bool,
	int64Values []int64,
) (column.Column, error) {
	var built column.Column
	var err error
	switch declaration.Type {
	case model.ScalarTypeString:
		if !declaration.Nullable {
			return column.NewColumnVarChar(declaration.Name, stringValues), nil
		}
		built, err = column.NewNullableColumnVarChar(declaration.Name, stringValues, validData, column.WithSparseNullableMode[string](true))
	case model.ScalarTypeBool:
		if !declaration.Nullable {
			return column.NewColumnBool(declaration.Name, boolValues), nil
		}
		built, err = column.NewNullableColumnBool(declaration.Name, boolValues, validData, column.WithSparseNullableMode[bool](true))
	case model.ScalarTypeInt64:
		if !declaration.Nullable {
			return column.NewColumnInt64(declaration.Name, int64Values), nil
		}
		built, err = column.NewNullableColumnInt64(declaration.Name, int64Values, validData, column.WithSparseNullableMode[int64](true))
	default:
		return nil, fmt.Errorf("unsupported declared column type %q", declaration.Type)
	}
	if err != nil {
		slog.Error("build declared scalar insert column failed", "collection", collectionName, "column", declaration.Name, "err", err)
		return nil, fmt.Errorf("build declared column %s for %s: %w", declaration.Name, collectionName, err)
	}
	return built, nil
}

// declaredScalarValueAt reads the value of one declared column at a row of a
// query result, with its null state.
func declaredScalarValueAt(valueColumn column.Column, declaration model.ScalarColumn, rowIndex int) (model.ScalarValue, error) {
	value := model.ScalarValue{Type: declaration.Type, Null: false, String: "", Bool: false, Int64: 0}
	if valueColumn == nil {
		return value, ErrSearchResultIncomplete
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
	case model.ScalarTypeString:
		value.String, err = valueColumn.GetAsString(rowIndex)
	case model.ScalarTypeBool:
		value.Bool, err = valueColumn.GetAsBool(rowIndex)
	case model.ScalarTypeInt64:
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

// declaredScalarRowBytes estimates the raw bytes the declared scalar values of
// one row add to an insert request, plus a small per-column framing allowance.
func declaredScalarRowBytes(declared []model.ScalarColumn, chunk model.StoredChunk) int {
	const perColumnFraming = 6
	total := 0
	for _, declaration := range declared {
		total += perColumnFraming
		value := chunk.Scalars[declaration.Name]
		switch declaration.Type {
		case model.ScalarTypeString:
			total += len(value.String)
		case model.ScalarTypeBool:
			total += boolBytes
		case model.ScalarTypeInt64:
			total += int64Bytes
		default:
		}
	}
	return total
}

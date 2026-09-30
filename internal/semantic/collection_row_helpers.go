package semantic

import (
	"fmt"
	"log/slog"
	"strings"

	"github.com/milvus-io/milvus/client/v2/column"
)

// inStringClause renders a Milvus `field in ["a", "b"]` membership clause, each
// value escaped for use inside a double-quoted Milvus string literal. An empty
// value set contributes no clause.
func inStringClause(field string, values []string) string {
	if len(values) == 0 {
		return ""
	}
	quoted := make([]string, 0, len(values))
	for _, value := range values {
		quoted = append(quoted, `"`+escapeMilvusString(value)+`"`)
	}
	return field + " in [" + strings.Join(quoted, ", ") + "]"
}

// batchConversationIDs splits ids into chunks of at most size, returning a
// single empty batch when ids is empty so callers run exactly one unscoped
// search.
func batchConversationIDs(ids []string, size int) [][]string {
	if len(ids) == 0 {
		return [][]string{nil}
	}
	if size <= 0 || len(ids) <= size {
		return [][]string{ids}
	}
	batches := make([][]string, 0, (len(ids)+size-1)/size)
	for start := 0; start < len(ids); start += size {
		end := min(start+size, len(ids))
		batches = append(batches, ids[start:end])
	}
	return batches
}

func readOptionalStringAt(valueColumn column.Column, rowIndex int) (string, bool, error) {
	if valueColumn == nil {
		return "", false, nil
	}
	isNull, nullErr := valueColumn.IsNull(rowIndex)
	if nullErr != nil {
		slog.Error("read optional string null state failed", "row", rowIndex, "err", nullErr)
		return "", false, fmt.Errorf("read null state at row %d: %w", rowIndex, nullErr)
	}
	if isNull {
		return "", false, nil
	}
	value, valueErr := valueColumn.GetAsString(rowIndex)
	if valueErr != nil {
		slog.Error("read optional string failed", "row", rowIndex, "err", valueErr)
		return "", false, fmt.Errorf("read string at row %d: %w", rowIndex, valueErr)
	}
	return value, true, nil
}

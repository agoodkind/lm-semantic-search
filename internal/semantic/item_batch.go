package semantic

import (
	"fmt"
	"log/slog"
	"strings"

	"github.com/milvus-io/milvus/client/v2/column"
	"goodkind.io/lm-semantic-search/collection"
	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"
)

// itemIDBatchSize bounds how many item ids go into one Milvus `in [...]`
// membership clause on the stored-row load path. A larger id set runs one query
// per batch.
const itemIDBatchSize = 256

// dedupeItemIDs trims every id, drops empty ids, and drops repeated ids while
// it keeps the first occurrence order.
func dedupeItemIDs(itemIDs []string) []string {
	seen := make(map[string]struct{}, len(itemIDs))
	unique := make([]string, 0, len(itemIDs))
	for _, itemID := range itemIDs {
		trimmed := strings.TrimSpace(itemID)
		if trimmed == "" {
			continue
		}
		if _, found := seen[trimmed]; found {
			continue
		}
		seen[trimmed] = struct{}{}
		unique = append(unique, trimmed)
	}
	return unique
}

// batchItemIDs splits ids into chunks of at most size, returning a single empty
// batch when ids is empty.
func batchItemIDs(ids []string, size int) [][]string {
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

// inStringClause renders a Milvus `field in ["a", "b"]` membership clause, each
// value escaped for use inside a double-quoted Milvus string literal. An empty
// value set contributes no clause.
func inStringClause(field string, values []string) string {
	if len(values) == 0 {
		return ""
	}
	quoted := make([]string, 0, len(values))
	for _, value := range values {
		quoted = append(quoted, `"`+collection.EscapeString(value)+`"`)
	}
	return field + " in [" + strings.Join(quoted, ", ") + "]"
}

// readOptionalStringAt reads a nullable string cell. The second result is false
// for an absent column or a null cell.
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

func contentVectorAt(contentColumn column.Column, vectorColumn column.Column, rowIndex int) (string, []float32, error) {
	contentValue, contentErr := contentColumn.GetAsString(rowIndex)
	if contentErr != nil {
		slog.Error("read content column failed", "index", rowIndex, "err", contentErr)
		return "", nil, fmt.Errorf("read content column at %d: %w", rowIndex, contentErr)
	}
	vector, vectorErr := milvusstore.VectorAt(vectorColumn, rowIndex)
	if vectorErr != nil {
		slog.Error("read vector column failed", "index", rowIndex, "err", vectorErr)
		return "", nil, fmt.Errorf("read vector column at %d: %w", rowIndex, vectorErr)
	}
	return contentValue, vector, nil
}

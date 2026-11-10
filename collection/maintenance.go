package collection

import "strings"

// DeleteItemsRequest selects the rows of a set of items for deletion. A row
// matches when the value of the declaration's item ID column is in ItemIDs, or
// when its relativePath starts with one of PathPrefixes. The prefixes select
// legacy rows with a null item ID column. The request deletes no other row.
type DeleteItemsRequest struct {
	Collection   string
	Declaration  Declaration
	ItemIDs      []string
	PathPrefixes []string
}

// ScalarBackfill is one backfill of declared scalar columns in a collection. A
// row needs the backfill when one of Columns is null, or an empty string, on the
// row. ItemColumn is the declared item ID column, and Values maps an item ID to
// the value of every column in Columns. A row without an item ID belongs to the
// item ID that follows one of LegacyPathFamilies in its relativePath, which
// covers rows written before the item ID column existed. DryRun counts rows and
// writes nothing.
type ScalarBackfill struct {
	ItemColumn         string
	Columns            []ScalarColumn
	Values             map[string]map[string]ScalarValue
	LegacyPathFamilies []string
	DryRun             bool
}

// ScalarValueMissing reports whether a backfill fills a stored value: a null
// value or an empty string.
func ScalarValueMissing(value ScalarValue) bool {
	return value.Null || (value.Type == ScalarTypeString && value.String == "")
}

// Needs reports whether a row with the stored backfill column values needs the
// backfill, because one of them is null or an empty string.
func (backfill ScalarBackfill) Needs(stored map[string]ScalarValue) bool {
	for _, column := range backfill.Columns {
		if ScalarValueMissing(stored[column.Name]) {
			return true
		}
	}
	return false
}

// ItemValues returns the backfill values of the item that owns a row. itemID is
// the row's item ID column value, and it is empty when the column is null. A row
// without an item ID belongs to the longest streamed item ID that follows one of
// the legacy path families in its relativePath.
func (backfill ScalarBackfill) ItemValues(itemID string, relativePath string) (map[string]ScalarValue, bool) {
	if itemID == "" && len(backfill.LegacyPathFamilies) > 0 {
		itemID = backfill.legacyPathItem(relativePath)
	}
	if itemID == "" {
		return nil, false
	}
	values, streamed := backfill.Values[itemID]
	return values, streamed
}

// legacyPathItem returns the longest streamed item ID that follows the family
// prefix of a row path, or an empty string.
func (backfill ScalarBackfill) legacyPathItem(relativePath string) string {
	for _, family := range backfill.LegacyPathFamilies {
		remainder, found := strings.CutPrefix(relativePath, family)
		if !found {
			continue
		}
		for end := strings.LastIndexByte(remainder, '/'); end > 0; end = strings.LastIndexByte(remainder[:end], '/') {
			if _, streamed := backfill.Values[remainder[:end]]; streamed {
				return remainder[:end]
			}
		}
		return ""
	}
	return ""
}

// Filled returns the backfill column values a row stores after the backfill. A
// missing stored value takes the item's value, and every other stored value
// stays. changed reports whether any value differs from the stored one.
func (backfill ScalarBackfill) Filled(stored map[string]ScalarValue, values map[string]ScalarValue) (map[string]ScalarValue, bool) {
	filled := make(map[string]ScalarValue, len(backfill.Columns))
	changed := false
	for _, column := range backfill.Columns {
		value := stored[column.Name]
		if ScalarValueMissing(value) && values[column.Name] != value {
			value = values[column.Name]
			changed = true
		}
		filled[column.Name] = value
	}
	return filled, changed
}

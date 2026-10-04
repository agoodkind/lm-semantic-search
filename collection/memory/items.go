package memory

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"

	"goodkind.io/lm-semantic-search/collection"
)

// itemSelection matches a row by its item ID column value or by a prefix of
// its relativePath.
type itemSelection struct {
	itemColumn   string
	itemIDs      map[string]struct{}
	pathPrefixes []string
}

func newItemSelection(declaration collection.Declaration, itemIDs []string, pathPrefixes []string) (itemSelection, error) {
	if len(itemIDs) == 0 && len(pathPrefixes) == 0 {
		return itemSelection{}, errNoItemSelection
	}
	if len(itemIDs) > 0 && declaration.ItemIDColumn == "" {
		return itemSelection{}, errors.New("declaration has no item ID column")
	}
	selected := make(map[string]struct{}, len(itemIDs))
	for _, itemID := range itemIDs {
		selected[itemID] = struct{}{}
	}
	return itemSelection{
		itemColumn:   declaration.ItemIDColumn,
		itemIDs:      selected,
		pathPrefixes: pathPrefixes,
	}, nil
}

func (selection itemSelection) matches(row *storedRow) bool {
	if len(selection.itemIDs) > 0 {
		value, stores := row.row.Scalars[selection.itemColumn]
		if stores && value.Type == collection.ScalarTypeString {
			if _, selected := selection.itemIDs[value.String]; selected {
				return true
			}
		}
	}
	for _, prefix := range selection.pathPrefixes {
		if strings.HasPrefix(row.row.RelativePath, prefix) {
			return true
		}
	}
	return false
}

// QueryRows returns every stored row of the requested items in ascending ID
// order. A row matches by item ID column value or by a relativePath prefix.
func (store *Store) QueryRows(ctx context.Context, request collection.RowsRequest) ([]collection.StoredRow, error) {
	name, err := requireName(request.Collection)
	if err != nil {
		return nil, err
	}
	selection, err := newItemSelection(request.Declaration, request.ItemIDs, request.PathPrefixes)
	if err != nil {
		slog.ErrorContext(ctx, "build stored row selection failed", "collection", name, "err", err)
		return nil, fmt.Errorf("build stored row selection for %s: %w", name, err)
	}
	store.mutex.RLock()
	defer store.mutex.RUnlock()
	stored, exists := store.collections[name]
	if !exists {
		return nil, collection.ErrCollectionMissing
	}
	declared := request.Declaration.Scalars
	rows := make([]collection.StoredRow, 0)
	for _, id := range stored.sortedIDs() {
		row := stored.rows[id]
		if !selection.matches(row) {
			continue
		}
		cells := make(map[string]collection.ScalarCell, len(declared))
		for _, column := range declared {
			cells[column.Name] = stored.cell(row, column.Name, declared)
		}
		var vector []float32
		if request.IncludeVector {
			vector = slices.Clone(row.row.Vector)
		}
		rows = append(rows, collection.StoredRow{
			ID:                row.row.ID,
			RelativePath:      row.row.RelativePath,
			Content:           row.row.Content,
			SplitPart:         row.row.SplitPart,
			SplitPartRecorded: row.row.SplitPartRecorded,
			EmbeddingModel:    store.options.EmbeddingModel,
			ContentHash:       row.contentHash,
			Vector:            vector,
			Scalars:           cells,
		})
	}
	return rows, nil
}

// DeleteItems removes every row with an item ID column value in ItemIDs and
// every row with a relativePath that starts with one of PathPrefixes. An empty
// prefix selects no row. It deletes no other row and returns the deleted count.
func (store *Store) DeleteItems(ctx context.Context, request collection.DeleteItemsRequest) (int64, error) {
	name, err := requireName(request.Collection)
	if err != nil {
		return 0, err
	}
	prefixes := make([]string, 0, len(request.PathPrefixes))
	for _, prefix := range request.PathPrefixes {
		if prefix != "" {
			prefixes = append(prefixes, prefix)
		}
	}
	if len(request.ItemIDs) == 0 && len(request.PathPrefixes) == 0 {
		return 0, errors.New("delete items selects no item and no path prefix")
	}
	if len(request.ItemIDs) == 0 && len(prefixes) == 0 {
		return 0, nil
	}
	selection, err := newItemSelection(request.Declaration, request.ItemIDs, prefixes)
	if err != nil {
		slog.ErrorContext(ctx, "build delete selection failed", "collection", name, "err", err)
		return 0, fmt.Errorf("delete items from %s: %w", name, err)
	}
	store.mutex.Lock()
	defer store.mutex.Unlock()
	stored, exists := store.collections[name]
	if !exists {
		return 0, collection.ErrCollectionMissing
	}
	var removed int64
	for id, row := range stored.rows {
		if !selection.matches(row) {
			continue
		}
		delete(stored.rows, id)
		removed++
	}
	return removed, nil
}

// BackfillScalars writes backfill values into the backfill columns that are
// null or an empty string on the rows of streamed items, and keeps every other
// stored value. It returns the rows that need the backfill: changed counts the
// rows of streamed items, and orphan counts the rest, which it leaves
// unchanged. A dry run counts and writes nothing.
func (store *Store) BackfillScalars(ctx context.Context, collectionName string, backfill collection.ScalarBackfill) (int, int, error) {
	name, err := requireName(collectionName)
	if err != nil {
		return 0, 0, err
	}
	if len(backfill.Columns) == 0 {
		return 0, 0, errors.New("scalar backfill lists no column")
	}
	store.mutex.Lock()
	defer store.mutex.Unlock()
	stored, exists := store.collections[name]
	if !exists {
		return 0, 0, collection.ErrCollectionMissing
	}
	if err := stored.validateBackfill(backfill); err != nil {
		slog.ErrorContext(ctx, "validate scalar backfill failed", "collection", name, "err", err)
		return 0, 0, fmt.Errorf("backfill scalars in %s: %w", name, err)
	}
	changed := 0
	orphan := 0
	for id, row := range stored.rows {
		current := make(map[string]collection.ScalarValue, len(backfill.Columns))
		for _, column := range backfill.Columns {
			value, stores := row.row.Scalars[column.Name]
			if !stores {
				value = collection.ScalarValue{Type: column.Type, Null: true, String: "", Bool: false, Int64: 0}
			}
			current[column.Name] = value
		}
		if !backfill.Needs(current) {
			continue
		}
		itemID := ""
		if value, stores := row.row.Scalars[backfill.ItemColumn]; stores {
			itemID = value.String
		}
		values, streamed := backfill.ItemValues(itemID, row.row.RelativePath)
		if !streamed {
			orphan++
			continue
		}
		changed++
		filled, differs := backfill.Filled(current, values)
		if !differs || backfill.DryRun {
			continue
		}
		for _, column := range backfill.Columns {
			value := filled[column.Name]
			if !value.Null && value.Type != column.Type {
				typeErr := fmt.Errorf("row %s has a %s value for %s column %s", row.row.RelativePath, value.Type, column.Type, column.Name)
				slog.ErrorContext(ctx, "write scalar backfill failed", "collection", name, "err", typeErr)
				return changed, orphan, fmt.Errorf("backfill scalars in %s: %w", name, typeErr)
			}
		}
		stored.rows[id] = row.withScalars(filled)
	}
	return changed, orphan, nil
}

func (stored *storedCollection) validateBackfill(backfill collection.ScalarBackfill) error {
	if _, inSchema := stored.column(backfill.ItemColumn); !inSchema {
		return fmt.Errorf("the collection has no item column %q", backfill.ItemColumn)
	}
	for _, column := range backfill.Columns {
		schemaColumn, inSchema := stored.column(column.Name)
		if !inSchema {
			return fmt.Errorf("the collection has no column %s", column.Name)
		}
		if schemaColumn.Type != column.Type {
			return fmt.Errorf("column %s is declared %s and stored as %s", column.Name, column.Type, schemaColumn.Type)
		}
	}
	return nil
}

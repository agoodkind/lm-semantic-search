package localvec

import (
	"context"
	"errors"
	"maps"
	"strings"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// DeleteItemRows deletes the rows removal selects from a local document
// collection. It serves an explicit item delete, and a missing collection
// deletes nothing.
func (store *Store) DeleteItemRows(ctx context.Context, collectionName string, removal semantic.Removal) error {
	if err := operationContextError(ctx, "delete local item rows"); err != nil {
		return err
	}
	trimmedCollectionName := strings.TrimSpace(collectionName)
	if trimmedCollectionName == "" {
		return errors.New("document collection name is required")
	}
	if removal.Empty() {
		return errors.New("item removal selects no rows")
	}
	stored, err := store.collectionForName(trimmedCollectionName, false)
	if err != nil {
		return err
	}
	return stored.mutate(removal, nil, false)
}

// BackfillCollectionScalars writes backfill values into the declared columns
// that are null or an empty string on the rows of streamed items. Every other
// row field, the vector, and the row key stay as stored. It returns the rows
// that need the backfill: changed counts the rows of streamed items, and
// orphan counts the rest, which it leaves unchanged. A dry run counts and
// writes nothing. A missing collection returns semantic.ErrCollectionMissing.
func (store *Store) BackfillCollectionScalars(ctx context.Context, collectionName string, backfill semantic.ScalarBackfill) (int, int, error) {
	if err := operationContextError(ctx, "backfill local collection scalars"); err != nil {
		return 0, 0, err
	}
	stored, err := store.collectionForName(collectionName, false)
	if err != nil {
		return 0, 0, err
	}
	if backfill.DryRun {
		rows, exists, snapshotErr := stored.snapshot()
		if snapshotErr != nil {
			return 0, 0, snapshotErr
		}
		if !exists {
			return 0, 0, semantic.ErrCollectionMissing
		}
		return backfillRows(rows, backfill)
	}
	changed := 0
	orphan := 0
	err = stored.rewrite(true, func(rows []row) ([]row, error) {
		var fillErr error
		changed, orphan, fillErr = backfillRows(rows, backfill)
		return rows, fillErr
	})
	return changed, orphan, err
}

// backfillRows counts the rows that need the backfill. Unless the backfill is a
// dry run, it fills the missing backfill columns of streamed items' rows in
// place.
func backfillRows(rows []row, backfill semantic.ScalarBackfill) (int, int, error) {
	changed := 0
	orphan := 0
	for index := range rows {
		stored, err := rows[index].backfillValues(backfill)
		if err != nil {
			return changed, orphan, err
		}
		if !backfill.Needs(stored) {
			continue
		}
		itemID, _ := rows[index].itemID(backfill.ItemColumn)
		values, streamed := backfill.ItemValues(itemID, rows[index].RelativePath)
		if !streamed {
			orphan++
			continue
		}
		changed++
		if backfill.DryRun {
			continue
		}
		filled, _ := backfill.Filled(stored, values)
		for _, column := range backfill.Columns {
			if filled[column.Name] == stored[column.Name] {
				continue
			}
			rows[index], err = rows[index].withScalarValue(column, filled[column.Name])
			if err != nil {
				return changed, orphan, err
			}
		}
	}
	return changed, orphan, nil
}

// backfillValues returns the row's stored value of every backfill column.
func (stored row) backfillValues(backfill semantic.ScalarBackfill) (map[string]model.ScalarValue, error) {
	values := make(map[string]model.ScalarValue, len(backfill.Columns))
	for _, column := range backfill.Columns {
		value, err := stored.scalarValue(column)
		if err != nil {
			return nil, err
		}
		values[column.Name] = value
	}
	return values, nil
}

func (stored row) scalarValue(column model.ScalarColumn) (model.ScalarValue, error) {
	value, found := stored.Scalars[column.Name]
	if !found {
		return model.ScalarValue{Type: column.Type, Null: true}, nil
	}
	return value, nil
}
func (stored row) withScalarValue(column model.ScalarColumn, value model.ScalarValue) (row, error) {
	scalars := maps.Clone(stored.Scalars)
	if scalars == nil {
		scalars = make(map[string]model.ScalarValue, 1)
	}
	scalars[column.Name] = value
	stored.Scalars = scalars
	return stored, nil
}

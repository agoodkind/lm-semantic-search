package localvec

import (
	"context"
	"errors"
	"fmt"
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
			rows[index], err = rows[index].withScalarValue(column, filled[column.Name], backfill.Conversation)
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
		value, err := stored.scalarValue(column, backfill.Conversation)
		if err != nil {
			return nil, err
		}
		values[column.Name] = value
	}
	return values, nil
}

// scalarValue returns the row's value of one declared column. A generic row
// keeps declared values in Scalars, and a column the row lacks is null. A
// conversation row keeps the conversation columns in its conversation fields.
// An unset string field there is an empty string, and archived and
// timestampUnix are never null.
func (stored row) scalarValue(column model.ScalarColumn, conversation bool) (model.ScalarValue, error) {
	if !conversation {
		value, found := stored.Scalars[column.Name]
		if !found {
			return model.ScalarValue{Type: column.Type, Null: true, String: "", Bool: false, Int64: 0}, nil
		}
		return value, nil
	}
	switch column.Name {
	case semantic.ConversationParentColumn:
		return stringScalarValue(stored.ParentConversationID), nil
	case semantic.ConversationRoleColumn:
		return stringScalarValue(stored.Role), nil
	case semantic.ConversationWorkspaceRootColumn:
		return stringScalarValue(stored.WorkspaceRoot), nil
	case semantic.ConversationLoadRulesColumn:
		return stringScalarValue(stored.LoadRules), nil
	case semantic.ConversationArchivedColumn:
		return model.ScalarValue{Type: model.ScalarTypeBool, Null: false, String: "", Bool: stored.Archived, Int64: 0}, nil
	case semantic.ConversationTimestampColumn:
		return model.ScalarValue{Type: model.ScalarTypeInt64, Null: false, String: "", Bool: false, Int64: stored.TimestampUnix}, nil
	default:
		return model.ScalarValue{}, fmt.Errorf("a local conversation row does not store column %s", column.Name)
	}
}

// withScalarValue returns a copy of the row that stores value in one declared
// column. The copy owns its Scalars map, and the stored row keeps its own.
func (stored row) withScalarValue(column model.ScalarColumn, value model.ScalarValue, conversation bool) (row, error) {
	if !conversation {
		scalars := maps.Clone(stored.Scalars)
		if scalars == nil {
			scalars = make(map[string]model.ScalarValue, 1)
		}
		scalars[column.Name] = value
		stored.Scalars = scalars
		return stored, nil
	}
	switch column.Name {
	case semantic.ConversationParentColumn:
		stored.ParentConversationID = value.String
	case semantic.ConversationRoleColumn:
		stored.Role = value.String
	case semantic.ConversationWorkspaceRootColumn:
		stored.WorkspaceRoot = value.String
	case semantic.ConversationLoadRulesColumn:
		stored.LoadRules = value.String
	case semantic.ConversationArchivedColumn:
		stored.Archived = value.Bool
	case semantic.ConversationTimestampColumn:
		stored.TimestampUnix = value.Int64
	default:
		return stored, fmt.Errorf("a local conversation row does not store column %s", column.Name)
	}
	return stored, nil
}

// stringScalarValue returns a string value that a conversation row field
// stores.
func stringScalarValue(value string) model.ScalarValue {
	return model.ScalarValue{Type: model.ScalarTypeString, Null: false, String: value, Bool: false, Int64: 0}
}

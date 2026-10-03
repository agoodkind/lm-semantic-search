package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"strings"

	"goodkind.io/lm-semantic-search/collection"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// conversationDerivedColumns are the conversation declaration columns a scalar
// backfill cannot fill. The engine derives provider from the item id and
// messageIndex from the row key.
var conversationDerivedColumns = []string{semantic.ConversationProviderColumn, semantic.ConversationMessageIndexColumn}

// collectionBackfillItemInput is one streamed item of a generic scalar backfill
// before validation.
type collectionBackfillItemInput struct {
	ItemID  string
	Scalars []collectionScalarInput
}

// collectionBackfillRequest is one generic scalar backfill before validation
// against the saved declaration. Columns lists the header columns the backfill
// fills.
type collectionBackfillRequest struct {
	CollectionID string
	Client       model.ClientInfo
	Columns      []string
	Items        []collectionBackfillItemInput
	DryRun       bool
}

// backfillCollectionItems validates a scalar backfill of a registered document
// collection against its saved declaration and runs it. An unregistered
// collection id fails. The generic RPCs never register a collection
// implicitly.
func (manager *Manager) backfillCollectionItems(ctx context.Context, request collectionBackfillRequest) (int, int, error) {
	codebase, err := manager.registeredCollection(request.CollectionID)
	if err != nil {
		return 0, 0, err
	}
	backfill, err := validateCollectionBackfill(savedCollectionDeclaration(codebase), request)
	if err != nil {
		return 0, 0, err
	}
	return manager.runScalarBackfill(ctx, codebase, backfill)
}

// runScalarBackfill runs one scalar backfill against the stored rows of a
// document collection. Both backfill RPCs run their backfills here.
func (manager *Manager) runScalarBackfill(ctx context.Context, codebase model.Codebase, backfill semantic.ScalarBackfill) (int, int, error) {
	if manager.semantic == nil {
		return 0, 0, semantic.ErrUnavailable
	}
	changed, orphan, err := manager.semantic.BackfillCollectionScalars(ctx, codebase.CollectionName, backfill)
	if err != nil {
		slog.ErrorContext(ctx, "backfill collection scalars failed", "codebase_id", codebase.ID, "collection", codebase.CollectionName, "changed", changed, "orphan", orphan, "err", err)
		return changed, orphan, fmt.Errorf("backfill scalars for %s: %w", codebase.CanonicalPath, err)
	}
	return changed, orphan, nil
}

// deleteCollectionItem queues the removal of one item's rows from a registered
// document collection. An unregistered collection id fails.
func (manager *Manager) deleteCollectionItem(ctx context.Context, collectionID string, itemID string, client model.ClientInfo) (model.Job, error) {
	trimmedItemID := strings.TrimSpace(itemID)
	if trimmedItemID == "" {
		return model.Job{}, adapterr.NewMissingArgument("item_id")
	}
	codebase, err := manager.registeredCollection(collectionID)
	if err != nil {
		return model.Job{}, err
	}
	return manager.queueItemDelete(ctx, codebase, trimmedItemID, client)
}

// queueItemDelete queues the asynchronous removal of one item's rows. Both
// delete RPCs queue their deletes here.
func (manager *Manager) queueItemDelete(ctx context.Context, codebase model.Codebase, itemID string, client model.ClientInfo) (model.Job, error) {
	return manager.queueConversationJob(ctx, codebase, client, conversationJobPayload{
		Kind:           conversationJobKindDelete,
		CollectionName: codebase.CollectionName,
		Manifest:       nil,
		Documents:      nil,
		Rows:           nil,
		ItemID:         itemID,
		// A delete removes exactly one item and never runs the manifest-absence
		// branch. Absence stays unused, and exhaustruct requires it set.
		Absence: absenceRetain,
		// A delete never backfills or force-rebuilds rows.
		Backfill: false,
		Force:    false,
	})
}

// itemSelector returns the stored-row selector of a document collection's saved
// declaration.
func (manager *Manager) itemSelector(codebaseID string) collectionItemSelector {
	manager.mu.Lock()
	codebase := manager.codebases[codebaseID]
	manager.mu.Unlock()
	return newCollectionItemSelector(savedCollectionDeclaration(codebase))
}

// declaredColumnsNamed returns the declared columns of declaration with the
// given names, in the order of names. It skips a name the declaration lacks.
func declaredColumnsNamed(declaration collection.Declaration, names ...string) []collection.ScalarColumn {
	columns := make([]collection.ScalarColumn, 0, len(names))
	for _, name := range names {
		index := slices.IndexFunc(declaration.Scalars, func(column collection.ScalarColumn) bool {
			return column.Name == name
		})
		if index >= 0 {
			columns = append(columns, declaration.Scalars[index])
		}
	}
	return columns
}

// validateCollectionBackfill checks a generic scalar backfill against the saved
// declaration and returns the backfill to run. It rejects an empty or repeated
// item id. validateBackfillColumns and validateBackfillItem list the column and
// value cases it rejects.
func validateCollectionBackfill(declaration collection.Declaration, request collectionBackfillRequest) (semantic.ScalarBackfill, error) {
	conversation := semantic.IsConversationDeclaration(declaration)
	declared := make(map[string]collection.ScalarColumn, len(declaration.Scalars))
	for _, column := range declaration.Scalars {
		declared[column.Name] = column
	}
	columns, err := validateBackfillColumns(declaration.ItemIDColumn, declared, conversation, request.Columns)
	if err != nil {
		return semantic.ScalarBackfill{}, err
	}
	values := make(map[string]map[string]collection.ScalarValue, len(request.Items))
	for _, item := range request.Items {
		itemID := strings.TrimSpace(item.ItemID)
		if itemID == "" {
			return semantic.ScalarBackfill{}, adapterr.NewMissingArgument("item_id")
		}
		if _, duplicate := values[itemID]; duplicate {
			return semantic.ScalarBackfill{}, adapterr.NewInvalidArgument(fmt.Sprintf("item_id %q appears more than once", itemID))
		}
		itemValues, err := validateBackfillItem(itemID, declared, columns, item.Scalars)
		if err != nil {
			return semantic.ScalarBackfill{}, err
		}
		values[itemID] = itemValues
	}
	return semantic.ScalarBackfill{
		ItemColumn:   declaration.ItemIDColumn,
		Columns:      columns,
		Values:       values,
		Conversation: conversation,
		DryRun:       request.DryRun,
	}, nil
}

// validateBackfillColumns resolves the header columns of a scalar backfill to
// their declarations. It rejects an empty list, an undeclared or repeated
// column, the item id column, a derived conversation column, and a column that
// is never null or empty: a bool or int64 column that is not nullable.
func validateBackfillColumns(itemColumn string, declared map[string]collection.ScalarColumn, conversation bool, names []string) ([]collection.ScalarColumn, error) {
	if len(names) == 0 {
		return nil, adapterr.NewMissingArgument("columns")
	}
	columns := make([]collection.ScalarColumn, 0, len(names))
	for _, name := range names {
		column, found := declared[name]
		if !found {
			return nil, adapterr.NewInvalidColumnValue(name, fmt.Sprintf("backfill column %q is not declared", name))
		}
		if slices.Contains(columns, column) {
			return nil, adapterr.NewInvalidColumnValue(name, fmt.Sprintf("backfill column %q is listed more than once", name))
		}
		if name == itemColumn {
			return nil, adapterr.NewInvalidColumnValue(name, fmt.Sprintf("backfill column %q is the item id column, which selects the rows of an item", name))
		}
		if conversation && slices.Contains(conversationDerivedColumns, name) {
			return nil, adapterr.NewInvalidColumnValue(name, fmt.Sprintf("backfill column %q is derived: the conversation declaration derives provider from the item id and messageIndex from the row key", name))
		}
		if !column.Nullable && column.Type != collection.ScalarTypeString {
			return nil, adapterr.NewInvalidColumnValue(name, fmt.Sprintf("backfill column %q is a %s column that is not nullable, and a backfill fills only null or empty values", name, column.Type))
		}
		columns = append(columns, column)
	}
	return columns, nil
}

// validateBackfillItem checks the values of one backfill item and returns them
// by column. It rejects a value that is undeclared, mistyped, oversized, null,
// set more than once, or outside the header columns, and it rejects an item
// that sets no value for a header column.
func validateBackfillItem(itemID string, declared map[string]collection.ScalarColumn, columns []collection.ScalarColumn, scalars []collectionScalarInput) (map[string]collection.ScalarValue, error) {
	subject := fmt.Sprintf("item %q", itemID)
	values := make(map[string]collection.ScalarValue, len(columns))
	for _, scalar := range scalars {
		if _, duplicate := values[scalar.Column]; duplicate {
			return nil, adapterr.NewInvalidColumnValue(scalar.Column, fmt.Sprintf("%s sets column %q more than once", subject, scalar.Column))
		}
		if err := validateCollectionScalar(subject, declared, scalar); err != nil {
			return nil, err
		}
		if !slices.Contains(columns, declared[scalar.Column]) {
			return nil, adapterr.NewInvalidColumnValue(scalar.Column, fmt.Sprintf("%s sets column %q, which the backfill header does not list", subject, scalar.Column))
		}
		if scalar.Value.Null {
			return nil, adapterr.NewInvalidColumnValue(scalar.Column, fmt.Sprintf("%s sets column %q to null, and a backfill value must not be null", subject, scalar.Column))
		}
		values[scalar.Column] = scalar.Value
	}
	for _, column := range columns {
		if _, present := values[column.Name]; !present {
			return nil, adapterr.NewInvalidColumnValue(column.Name, fmt.Sprintf("%s sets no value for backfill column %q", subject, column.Name))
		}
	}
	return values, nil
}

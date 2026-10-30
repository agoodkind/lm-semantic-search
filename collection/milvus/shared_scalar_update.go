package milvus

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
)

// UpdateRowScalars uses a Milvus partial update to write the listed scalar
// columns and metadata of existing occurrence rows by ID. It then recomputes
// the column ranges for every referenced vector from all of its occurrences.
// Content, vectors, vector keys, and all other columns retain their stored
// values. It returns the number of updated rows. It fails before any write if
// a row ID is empty or repeated, metadata is empty, a listed column is absent
// from the occurrence collection or has a different type, or an ID has no
// stored occurrence. A SharedStore serializes this update with its other writes.
func (store *SharedStore) UpdateRowScalars(ctx context.Context, collectionName string, columns []collection.ScalarColumn, rows []collection.Row) (int, error) {
	occurrences := strings.TrimSpace(collectionName)
	if occurrences == "" {
		return 0, errors.New("collection name is required")
	}
	if len(rows) == 0 {
		return 0, nil
	}
	if err := validateScalarUpdate(occurrences, columns, rows); err != nil {
		slog.ErrorContext(ctx, "row scalar update request is invalid", "collection", occurrences, "err", err)
		return 0, err
	}
	ids := make([]string, 0, len(rows))
	for _, row := range rows {
		ids = append(ids, row.ID)
	}
	store.write.Lock()
	defer store.write.Unlock()
	keyByID, err := store.occurrenceKeyMap(ctx, occurrences, ids)
	if err != nil {
		return 0, err
	}
	missing := make([]string, 0)
	for _, id := range ids {
		if _, found := keyByID[id]; !found {
			missing = append(missing, id)
		}
	}
	if len(missing) > 0 {
		missingErr := fmt.Errorf("occurrence collection %s does not contain rows for %d requested ids; first missing id: %s", occurrences, len(missing), missing[0])
		slog.ErrorContext(ctx, "row scalar update includes ids without stored occurrences", "collection", occurrences, "missing", len(missing), "err", missingErr)
		return 0, missingErr
	}
	declaration, err := store.storedDeclaration(ctx, occurrences, "")
	if err != nil {
		return 0, err
	}
	if err := requireStoredColumns(occurrences, declaration.Scalars, columns); err != nil {
		slog.ErrorContext(ctx, "row scalar update includes a column absent from the occurrence collection", "collection", occurrences, "err", err)
		return 0, err
	}
	for start := 0; start < len(rows); start += sharedWriteBatchSize {
		batch := rows[start:min(start+sharedWriteBatchSize, len(rows))]
		if err := store.writeRowScalars(ctx, occurrences, columns, batch); err != nil {
			return 0, err
		}
	}
	keys := make(map[string]struct{}, len(keyByID))
	for _, key := range keyByID {
		keys[key] = struct{}{}
	}
	if err := store.rebuildSupersets(ctx, occurrences, declaration.Scalars, sortedKeys(keys)); err != nil {
		return 0, err
	}
	slog.InfoContext(ctx, "collection.shared_row_scalars_updated", "collection", occurrences, "rows", len(rows), "vectors", len(keys))
	return len(rows), nil
}

// writeRowScalars sends one Milvus partial update containing each row's ID,
// metadata, and listed scalar columns.
func (store *SharedStore) writeRowScalars(ctx context.Context, occurrences string, columns []collection.ScalarColumn, rows []collection.Row) error {
	ids := make([]string, 0, len(rows))
	metadata := make([]string, 0, len(rows))
	for _, row := range rows {
		ids = append(ids, row.ID)
		metadata = append(metadata, row.Metadata)
	}
	scalarColumns, err := DeclaredScalarInsertColumns(occurrences, columns, rows)
	if err != nil {
		slog.ErrorContext(ctx, "failed to build row scalar update columns", "collection", occurrences, "err", err)
		return fmt.Errorf("build row scalar update columns for %s: %w", occurrences, err)
	}
	option := milvusclient.NewColumnBasedInsertOption(occurrences).
		WithVarcharColumn(IDField, ids).
		WithVarcharColumn(MetadataField, metadata).
		WithColumns(scalarColumns...).
		WithPartialUpdate(true)
	if _, err := store.inner.client.Upsert(ctx, option); err != nil {
		return WrapError(ctx, err, "update row scalars in "+occurrences)
	}
	return nil
}

func validateScalarUpdate(occurrences string, columns []collection.ScalarColumn, rows []collection.Row) error {
	if len(columns) == 0 {
		return fmt.Errorf("row scalar update does not list any columns for %s", occurrences)
	}
	seen := make(map[string]struct{}, len(rows))
	for _, row := range rows {
		if strings.TrimSpace(row.ID) == "" {
			return fmt.Errorf("row scalar update includes a row without an id for %s", occurrences)
		}
		if _, duplicate := seen[row.ID]; duplicate {
			return fmt.Errorf("row scalar update for collection %s lists id %s more than once", occurrences, row.ID)
		}
		seen[row.ID] = struct{}{}
		if strings.TrimSpace(row.Metadata) == "" {
			return fmt.Errorf("row scalar update for collection %s gives row %s empty metadata", occurrences, row.ID)
		}
	}
	return nil
}

func requireStoredColumns(occurrences string, stored []collection.ScalarColumn, columns []collection.ScalarColumn) error {
	storedTypes := make(map[string]collection.ScalarType, len(stored))
	for _, column := range stored {
		storedTypes[column.Name] = column.Type
	}
	for _, column := range columns {
		storedType, found := storedTypes[column.Name]
		if !found || storedType != column.Type {
			return fmt.Errorf("occurrence collection %s does not store column %s with type %s", occurrences, column.Name, column.Type)
		}
	}
	return nil
}

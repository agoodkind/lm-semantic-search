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

// UpdateRowScalars writes the listed scalar columns and metadata of existing
// occurrence rows by ID using Milvus partial updates. It preserves content,
// vectors, vector keys, and all other stored column values.
//
// Before any write, it rejects an empty or repeated ID, empty metadata, an ID
// without a stored occurrence, and a column that the occurrence collection
// does not store with the same type. It also rejects a missing scalar value,
// a null value for a column that is not nullable, or a value of another type
// than its column.
//
// UpdateRowScalars does not check metadata length. Milvus checks metadata
// length when it receives each batch and can reject a later batch after earlier
// batches committed.
//
// UpdateRowScalars adds the new values to the ranges of every referenced vector,
// then writes the occurrence rows in batches of 256. After every batch succeeds,
// it recomputes the ranges of the referenced vectors from all their occurrences.
//
// If a batch write fails, UpdateRowScalars keeps the widened ranges and returns
// the error and the number of rows in batches that Milvus confirmed. The server
// can still commit the failed batch, but the returned count does not include it.
//
// One SharedStore serializes this update with its other writes.
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
	options := make([]rowScalarWrite, 0, (len(rows)+sharedWriteBatchSize-1)/sharedWriteBatchSize)
	for start := 0; start < len(rows); start += sharedWriteBatchSize {
		batch := rows[start:min(start+sharedWriteBatchSize, len(rows))]
		option, err := rowScalarOption(ctx, occurrences, columns, batch)
		if err != nil {
			return 0, err
		}
		options = append(options, rowScalarWrite{option: option, rows: batch})
	}
	if err := store.widenSupersets(ctx, occurrences, declaration.Scalars, keyByID, rows); err != nil {
		return 0, err
	}
	written := 0
	for _, write := range options {
		if _, err := store.inner.client.Upsert(ctx, write.option); err != nil {
			writeErr := WrapError(ctx, err, "update row scalars in "+occurrences)
			// Milvus can commit the failed batch after this call returns. Keep the widened
			// ranges without recomputing them; they cover both the old and new values.
			slog.ErrorContext(ctx, "row scalar update stopped after a batch failed", "collection", occurrences, "written", written, "requested", len(rows), "err", writeErr)
			return written, writeErr
		}
		written += len(write.rows)
	}
	keys := make(map[string]struct{}, len(keyByID))
	for _, key := range keyByID {
		keys[key] = struct{}{}
	}
	if err := store.rebuildSupersets(ctx, occurrences, declaration.Scalars, sortedKeys(keys)); err != nil {
		return written, err
	}
	slog.InfoContext(ctx, "collection.shared_row_scalars_updated", "collection", occurrences, "rows", len(rows), "vectors", len(keys))
	return len(rows), nil
}

// widenSupersets adds the new scalar values of the requested rows to the stored
// ranges of each referenced vector. It rewrites only vector rows with changed
// ranges, using their stored vectors. If a later occurrence write fails, each
// vector range still covers both the old and new values.
func (store *SharedStore) widenSupersets(ctx context.Context, occurrences string, scalars []collection.ScalarColumn, keyByID map[string]string, rows []collection.Row) error {
	vectors := VectorCollectionName(occurrences)
	keySet := make(map[string]struct{}, len(rows))
	for _, row := range rows {
		keySet[keyByID[row.ID]] = struct{}{}
	}
	vectorByKey, err := store.vectorsByKey(ctx, vectors, sortedKeys(keySet))
	if err != nil {
		return err
	}
	records := make(map[string]*vectorRecord, len(keySet))
	dimension := 0
	for _, row := range rows {
		key := keyByID[row.ID]
		record, found := records[key]
		if !found {
			vector, stored := vectorByKey[key]
			if !stored {
				missingErr := fmt.Errorf("vector collection %s does not contain vector key %s referenced by occurrence %s", vectors, key, row.ID)
				slog.ErrorContext(ctx, "row scalar update found an occurrence that references a missing vector", "collection", vectors, "vector_key", key, "err", missingErr)
				return missingErr
			}
			dimension = len(vector)
			record = newVectorRecord(key, vector, scalars)
			records[key] = record
		}
		record.addRow(scalars, row.Scalars)
	}
	unchanged, err := store.mergeStoredSupersets(ctx, vectors, scalars, records)
	if err != nil {
		return err
	}
	ordered := make([]*vectorRecord, 0, len(records))
	for _, key := range sortedRecordKeys(records) {
		if _, skip := unchanged[key]; !skip {
			ordered = append(ordered, records[key])
		}
	}
	for start := 0; start < len(ordered); start += sharedWriteBatchSize {
		option, err := vectorUpsertOption(vectors, dimension, scalars, ordered[start:min(start+sharedWriteBatchSize, len(ordered))])
		if err != nil {
			return err
		}
		if _, err := store.inner.client.Upsert(ctx, option); err != nil {
			return WrapError(ctx, err, "widen Milvus vector ranges in "+vectors)
		}
	}
	return nil
}

// rowScalarWrite contains one prepared partial update and the rows it writes.
type rowScalarWrite struct {
	option WriteOption
	rows   []collection.Row
}

// rowScalarOption builds a Milvus partial update with each row's ID, metadata,
// and listed scalar columns. It returns an error if a value is missing, is null
// for a column that does not allow null, or has a different type than its column.
func rowScalarOption(ctx context.Context, occurrences string, columns []collection.ScalarColumn, rows []collection.Row) (WriteOption, error) {
	ids := make([]string, 0, len(rows))
	metadata := make([]string, 0, len(rows))
	for _, row := range rows {
		ids = append(ids, row.ID)
		metadata = append(metadata, row.Metadata)
	}
	scalarColumns, err := DeclaredScalarInsertColumns(occurrences, columns, rows)
	if err != nil {
		slog.ErrorContext(ctx, "failed to build row scalar update columns", "collection", occurrences, "err", err)
		return nil, fmt.Errorf("build row scalar update columns for %s: %w", occurrences, err)
	}
	return milvusclient.NewColumnBasedInsertOption(occurrences).
		WithVarcharColumn(IDField, ids).
		WithVarcharColumn(MetadataField, metadata).
		WithColumns(scalarColumns...).
		WithPartialUpdate(true), nil
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

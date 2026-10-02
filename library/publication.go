package library

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log/slog"
)

// stagedRow is one staged occurrence of a generation.
type stagedRow struct {
	occurrence     Occurrence
	occurrenceHash string
	vectorID       string
}

// CommitGeneration publishes one staged owner generation in one SQLite
// transaction. The staged row count and manifest hash must equal seal, and
// every staged vector must pass a strong backend read first. Append inserts
// the new rows and accepts identical repeats. Replace removes the owner's
// previous rows. A committed token with the committed row count and manifest
// returns its saved receipt, and any other seal for that token returns an
// error that wraps [ErrAppendConflict]. A seal that differs from the staged
// rows returns an error that wraps [ErrInvalidRequest] and keeps the staged
// rows.
func (library *Library) CommitGeneration(ctx context.Context, key GenerationKey, seal GenerationSeal) (_ ApplyReceipt, err error) {
	if err := validateGenerationKey(key); err != nil {
		return ApplyReceipt{}, err
	}
	release, err := library.acquireWriter(ctx)
	if err != nil {
		return ApplyReceipt{}, err
	}
	defer func() {
		err = errors.Join(err, release())
	}()
	if err := library.replayOutbox(ctx); err != nil {
		return ApplyReceipt{}, err
	}

	var receipt *ApplyReceipt
	var mode BatchMode
	var rows []stagedRow
	if err := library.read(ctx, func(tx *sql.Tx) error {
		var readErr error
		receipt, readErr = checkGeneration(ctx, tx, key)
		if readErr != nil {
			return readErr
		}
		if receipt != nil {
			return checkCommittedSeal(ctx, tx, key, seal)
		}
		mode, rows, readErr = readStagedGeneration(ctx, tx, key)
		return readErr
	}); err != nil {
		return ApplyReceipt{}, err
	}
	if receipt != nil {
		return *receipt, nil
	}
	manifest, err := checkSeal(key, rows, seal)
	if err != nil {
		return ApplyReceipt{}, err
	}
	if err := library.verifyStagedVectors(ctx, rows); err != nil {
		return ApplyReceipt{}, err
	}
	prepared, documents := preparePublication(rows)
	if err := library.read(ctx, func(tx *sql.Tx) error {
		var err error
		documents, err = prepareLexicalContent(ctx, tx, library.config.AnalyzerIdentity, documents)
		return err
	}); err != nil {
		return ApplyReceipt{}, err
	}

	published := ApplyReceipt{
		Namespace:       key.Namespace,
		OwnerID:         key.OwnerID,
		GenerationOrder: key.GenerationOrder,
		Fingerprint:     generationFingerprint(key, manifest),
	}
	err = library.write(ctx, func(tx *sql.Tx) error {
		saved, err := checkGeneration(ctx, tx, key)
		if err != nil {
			return err
		}
		if saved != nil {
			published = *saved
			return checkCommittedSeal(ctx, tx, key, seal)
		}
		return publishGeneration(ctx, tx, library.config.AnalyzerIdentity, key, mode, prepared, documents, manifest, published.Fingerprint)
	})
	if err != nil {
		return ApplyReceipt{}, err
	}
	return published, nil
}

func readStagedGeneration(ctx context.Context, tx *sql.Tx, key GenerationKey) (mode BatchMode, rows []stagedRow, err error) {
	scanErr := tx.QueryRowContext(
		ctx,
		`SELECT mode FROM staged_generations WHERE namespace = ? AND owner_id = ? AND generation_order = ? AND generation_token = ?`,
		key.Namespace, key.OwnerID, key.GenerationOrder, key.IdempotencyToken,
	).Scan(&mode)
	if errors.Is(scanErr, sql.ErrNoRows) {
		return 0, nil, invalidRequest(fmt.Sprintf("commit: owner %q order %d has no staged generation", key.OwnerID, key.GenerationOrder))
	}
	if scanErr != nil {
		slog.ErrorContext(ctx, "read staged generation failed", "err", scanErr)
		return 0, nil, fmt.Errorf("read staged generation: %w", scanErr)
	}
	result, queryErr := tx.QueryContext(
		ctx,
		`SELECT payload, occurrence_hash, vector_id FROM staged_occurrences
		WHERE namespace = ? AND owner_id = ? AND generation_order = ? AND generation_token = ? ORDER BY row_key`,
		key.Namespace, key.OwnerID, key.GenerationOrder, key.IdempotencyToken,
	)
	if queryErr != nil {
		slog.ErrorContext(ctx, "read staged rows failed", "err", queryErr)
		return 0, nil, fmt.Errorf("read staged rows: %w", queryErr)
	}
	defer func() {
		err = errors.Join(err, closeRows(ctx, result))
	}()
	for result.Next() {
		var payload []byte
		var row stagedRow
		if err := result.Scan(&payload, &row.occurrenceHash, &row.vectorID); err != nil {
			slog.ErrorContext(ctx, "scan staged row failed", "err", err)
			return 0, nil, fmt.Errorf("scan staged row: %w", err)
		}
		row.occurrence, err = decodeOccurrence(payload)
		if err != nil {
			return 0, nil, err
		}
		rows = append(rows, row)
	}
	if err := result.Err(); err != nil {
		slog.ErrorContext(ctx, "read staged rows failed", "err", err)
		return 0, nil, fmt.Errorf("read staged rows: %w", err)
	}
	return mode, rows, nil
}

// checkSeal compares the staged rows with seal and returns the manifest hash.
func checkSeal(key GenerationKey, rows []stagedRow, seal GenerationSeal) (string, error) {
	entries := make([]manifestEntry, 0, len(rows))
	for _, row := range rows {
		entries = append(entries, manifestEntry{rowKey: row.occurrence.RowKey, occurrenceHash: row.occurrenceHash})
	}
	manifest := manifestHash(entries)
	if uint64(len(rows)) != seal.RowCount || manifest != seal.ManifestHash {
		return "", invalidRequest(fmt.Sprintf(
			"commit: owner %q order %d staged %d rows with manifest %s, the seal states %d rows with manifest %s",
			key.OwnerID, key.GenerationOrder, len(rows), manifest, seal.RowCount, seal.ManifestHash,
		))
	}
	return manifest, nil
}

// verifyStagedVectors requires a verified catalog vector and a matching
// strong backend read for every vector the staged rows reference.
func (library *Library) verifyStagedVectors(ctx context.Context, rows []stagedRow) error {
	seen := make(map[string]bool, len(rows))
	ids := make([]string, 0, len(rows))
	for _, row := range rows {
		if !seen[row.vectorID] {
			seen[row.vectorID] = true
			ids = append(ids, row.vectorID)
		}
	}
	var stored map[string]storedVector
	if err := library.read(ctx, func(tx *sql.Tx) error {
		var readErr error
		stored, readErr = lookupVectors(ctx, tx, ids)
		return readErr
	}); err != nil {
		return err
	}
	identities := make([]VectorIdentity, 0, len(ids))
	for _, id := range ids {
		vector, found := stored[id]
		if !found || vector.state != vectorStateVerified {
			err := fmt.Errorf("%w: staged vector %s is not verified in the catalog", ErrVectorMissing, id)
			slog.ErrorContext(ctx, "staged vector not verified", "vector_id", id, "err", err)
			return err
		}
		identities = append(identities, VectorIdentity{ID: id, IdentityDigest: vector.digest, Checksum: vector.checksum})
	}
	return library.verifyStrong(ctx, identities)
}

// publishGeneration writes the published rows, their lexical index entries,
// owner state, and receipt, and removes the staged generation, inside tx. A
// Replace passes the owner's previous rows to the lexical index as removed
// and every new row as added. An Append passes only the rows it inserted.
func publishGeneration(
	ctx context.Context,
	tx *sql.Tx,
	analyzer string,
	key GenerationKey,
	mode BatchMode,
	rows []preparedPublicationRow,
	documents map[string]lexicalDocument,
	manifest string,
	fingerprint string,
) error {
	var removed []lexicalOccurrence
	if mode == Replace {
		var err error
		removed, err = readOwnerLexicalRows(ctx, tx, key.Namespace, key.OwnerID)
		if err != nil {
			return err
		}
		if err := deleteOwnerOccurrences(ctx, tx, key.Namespace, key.OwnerID); err != nil {
			return err
		}
	}
	added, err := publishRows(ctx, tx, key, rows)
	if err != nil {
		return err
	}
	if err := publishPreparedLexical(ctx, tx, analyzer, key.Namespace, added, removed, documents); err != nil {
		return err
	}
	if _, err := tx.ExecContext(
		ctx,
		`INSERT INTO owners (namespace, owner_id, generation_order, generation_token, fingerprint) VALUES (?, ?, ?, ?, ?)
		ON CONFLICT (namespace, owner_id) DO UPDATE SET generation_order = excluded.generation_order,
		generation_token = excluded.generation_token, fingerprint = excluded.fingerprint`,
		key.Namespace, key.OwnerID, key.GenerationOrder, key.IdempotencyToken, fingerprint,
	); err != nil {
		slog.ErrorContext(ctx, "save owner state failed", "namespace", key.Namespace, "err", err)
		return fmt.Errorf("save owner state: %w", err)
	}
	if _, err := tx.ExecContext(
		ctx,
		`INSERT INTO batch_receipts (namespace, owner_id, generation_order, generation_token, batch_hash, row_count, mode, fingerprint)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		key.Namespace, key.OwnerID, key.GenerationOrder, key.IdempotencyToken, manifest, len(rows), mode, fingerprint,
	); err != nil {
		slog.ErrorContext(ctx, "save generation receipt failed", "namespace", key.Namespace, "err", err)
		return fmt.Errorf("save generation receipt: %w", err)
	}
	if err := saveReceiptRows(ctx, tx, key); err != nil {
		return err
	}
	if err := deleteStagedGeneration(ctx, tx, key); err != nil {
		return err
	}
	_, err = incrementRevision(ctx, tx, identityKeyVisibilityRevision)
	return err
}

// readOwnerLexicalRows returns the row key and search hash of every published
// occurrence of one owner, ordered by row key.
func readOwnerLexicalRows(ctx context.Context, tx *sql.Tx, namespace string, ownerID string) (rows []lexicalOccurrence, err error) {
	result, queryErr := tx.QueryContext(
		ctx,
		`SELECT row_key, search_hash FROM occurrences WHERE namespace = ? AND owner_id = ? ORDER BY row_key`,
		namespace, ownerID,
	)
	if queryErr != nil {
		slog.ErrorContext(ctx, "read owner rows failed", "namespace", namespace, "err", queryErr)
		return nil, fmt.Errorf("read owner %q rows: %w", ownerID, queryErr)
	}
	defer func() {
		err = errors.Join(err, closeRows(ctx, result))
	}()
	for result.Next() {
		row := lexicalOccurrence{OwnerID: ownerID, RowKey: "", SearchHash: "", SearchText: ""}
		if err := result.Scan(&row.RowKey, &row.SearchHash); err != nil {
			slog.ErrorContext(ctx, "scan owner row failed", "namespace", namespace, "err", err)
			return nil, fmt.Errorf("scan owner %q row: %w", ownerID, err)
		}
		rows = append(rows, row)
	}
	if err := result.Err(); err != nil {
		slog.ErrorContext(ctx, "read owner rows failed", "namespace", namespace, "err", err)
		return nil, fmt.Errorf("read owner %q rows: %w", ownerID, err)
	}
	return rows, nil
}

// typedScalar is the column form of one [ScalarValue]. Only the column of the
// value's type is valid, and a null value leaves every column invalid.
type typedScalar struct {
	stringValue sql.NullString
	int64Value  sql.NullInt64
	boolValue   sql.NullBool
}

func newTypedScalar(value ScalarValue) typedScalar {
	return typedScalar{
		stringValue: sql.NullString{String: value.String, Valid: !value.Null && value.Type == String},
		int64Value:  sql.NullInt64{Int64: value.Int64, Valid: !value.Null && value.Type == Int64},
		boolValue:   sql.NullBool{Bool: value.Bool, Valid: !value.Null && value.Type == Bool},
	}
}

func upsertEffectiveScalar(
	ctx context.Context,
	tx *sql.Tx,
	namespace string,
	ownerID string,
	rowKey string,
	name string,
	value ScalarValue,
	projectionOrder uint64,
) error {
	typed := newTypedScalar(value)
	if _, err := tx.ExecContext(
		ctx,
		`INSERT INTO effective_scalars (namespace, owner_id, row_key, column_name, type, string_value, int64_value, bool_value, is_null, projection_order)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT (namespace, owner_id, row_key, column_name) DO UPDATE SET type = excluded.type,
		string_value = excluded.string_value, int64_value = excluded.int64_value, bool_value = excluded.bool_value,
		is_null = excluded.is_null, projection_order = excluded.projection_order`,
		namespace, ownerID, rowKey, name, value.Type, typed.stringValue, typed.int64Value, typed.boolValue, value.Null, projectionOrder,
	); err != nil {
		slog.ErrorContext(ctx, "save effective scalar failed", "column", name, "err", err)
		return fmt.Errorf("save effective scalar %s for %q: %w", name, rowKey, err)
	}
	return nil
}

// ownerDeleteStatements remove every published row of one owner with its
// scalars and effective scalars.
var ownerDeleteStatements = []string{
	`DELETE FROM occurrences WHERE namespace = ? AND owner_id = ?`,
	`DELETE FROM occurrence_scalars WHERE namespace = ? AND owner_id = ?`,
	`DELETE FROM effective_scalars WHERE namespace = ? AND owner_id = ?`,
}

func deleteOwnerOccurrences(ctx context.Context, tx *sql.Tx, namespace string, ownerID string) error {
	for _, statement := range ownerDeleteStatements {
		if _, err := tx.ExecContext(ctx, statement, namespace, ownerID); err != nil {
			slog.ErrorContext(ctx, "remove owner rows failed", "err", err)
			return fmt.Errorf("remove owner rows: %w", err)
		}
	}
	return nil
}

// Delete removes the exact occurrence IDs from ReplaceAllowed namespaces with
// their scalars and lexical index entries. An absent ID removes nothing. An ID
// in an AppendOnly namespace returns an error that wraps [ErrInvalidRequest]
// and removes no row. Canonical vectors stay in the pool.
func (library *Library) Delete(ctx context.Context, ids []OccurrenceID) (err error) {
	if len(ids) == 0 {
		return nil
	}
	release, err := library.acquireWriter(ctx)
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(err, release())
	}()
	return library.write(ctx, func(tx *sql.Tx) error {
		checked := make(map[string]bool)
		removed := make(map[string][]lexicalOccurrence)
		var namespaces []string
		for _, id := range ids {
			if !checked[id.Namespace] {
				spec, err := loadNamespace(ctx, tx, id.Namespace)
				if err != nil {
					return err
				}
				if spec.Policy != ReplaceAllowed {
					return invalidRequest(fmt.Sprintf("delete: namespace %q is AppendOnly", id.Namespace))
				}
				checked[id.Namespace] = true
				namespaces = append(namespaces, id.Namespace)
			}
			searchHash, deleted, err := deleteOccurrence(ctx, tx, id)
			if err != nil {
				return err
			}
			if deleted {
				removed[id.Namespace] = append(removed[id.Namespace], lexicalOccurrence{
					OwnerID:    id.OwnerID,
					RowKey:     id.RowKey,
					SearchHash: searchHash,
					SearchText: "",
				})
			}
		}
		for _, namespace := range namespaces {
			if err := publishLexical(ctx, tx, library.config.AnalyzerIdentity, namespace, nil, removed[namespace]); err != nil {
				return err
			}
		}
		_, err := incrementRevision(ctx, tx, identityKeyVisibilityRevision)
		return err
	})
}

// occurrenceScalarDeleteStatements remove the scalars and effective scalars of
// one published row.
var occurrenceScalarDeleteStatements = []string{
	`DELETE FROM occurrence_scalars WHERE namespace = ? AND owner_id = ? AND row_key = ?`,
	`DELETE FROM effective_scalars WHERE namespace = ? AND owner_id = ? AND row_key = ?`,
}

// deleteOccurrence removes one published row with its scalars and effective
// scalars. It returns the search hash of the deleted row and whether a row
// existed.
func deleteOccurrence(ctx context.Context, tx *sql.Tx, id OccurrenceID) (string, bool, error) {
	var searchHash string
	scanErr := tx.QueryRowContext(
		ctx,
		`DELETE FROM occurrences WHERE namespace = ? AND owner_id = ? AND row_key = ? RETURNING search_hash`,
		id.Namespace, id.OwnerID, id.RowKey,
	).Scan(&searchHash)
	deleted := true
	if errors.Is(scanErr, sql.ErrNoRows) {
		deleted = false
	} else if scanErr != nil {
		slog.ErrorContext(ctx, "delete occurrence failed", "err", scanErr)
		return "", false, fmt.Errorf("delete occurrence %s/%s/%s: %w", id.Namespace, id.OwnerID, id.RowKey, scanErr)
	}
	for _, statement := range occurrenceScalarDeleteStatements {
		if _, err := tx.ExecContext(ctx, statement, id.Namespace, id.OwnerID, id.RowKey); err != nil {
			slog.ErrorContext(ctx, "delete occurrence failed", "err", err)
			return "", false, fmt.Errorf("delete occurrence %s/%s/%s: %w", id.Namespace, id.OwnerID, id.RowKey, err)
		}
	}
	return searchHash, deleted, nil
}

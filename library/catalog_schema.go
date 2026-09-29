package library

import (
	"context"
	"database/sql"
	"fmt"
	"log/slog"
	"strconv"
)

// catalogSchemaVersion is the schema version that this build creates and
// reads. Version 2 adds the lexical index tables. createCatalogSchema upgrades
// a version 1 catalog without occurrences, and a catalog saved with any other
// version fails to open.
const catalogSchemaVersion = 2

// lexicalSchemaFromVersion is the saved schema version that the lexical index
// tables upgrade.
const lexicalSchemaFromVersion = 1

// Keys of the store_identity table.
const (
	identityKeySchemaVersion      = "schema_version"
	identityKeyCatalogUUID        = "catalog_uuid"
	identityKeyDescriptor         = "descriptor"
	identityKeyAnalyzer           = "analyzer_identity"
	identityKeyVisibilityRevision = "visibility_revision"
	identityKeyProjectionRevision = "projection_revision"
)

// catalogSchemaStatements creates every catalog table and index. Each
// statement is idempotent.
var catalogSchemaStatements = []string{
	`CREATE TABLE IF NOT EXISTS store_identity (
		key TEXT PRIMARY KEY,
		value TEXT NOT NULL
	)`,
	`CREATE TABLE IF NOT EXISTS namespaces (
		id TEXT PRIMARY KEY,
		declaration TEXT NOT NULL
	)`,
	`CREATE TABLE IF NOT EXISTS vectors (
		vector_id TEXT PRIMARY KEY,
		identity_digest TEXT NOT NULL,
		input_hash TEXT NOT NULL,
		input_bytes BLOB NOT NULL,
		model_identity TEXT NOT NULL,
		dimension INTEGER NOT NULL,
		normalization TEXT NOT NULL,
		vector_checksum TEXT NOT NULL,
		state TEXT NOT NULL,
		generation INTEGER NOT NULL
	)`,
	`CREATE TABLE IF NOT EXISTS vector_outbox (
		vector_id TEXT PRIMARY KEY,
		vector_payload BLOB NOT NULL,
		payload_hash TEXT NOT NULL,
		write_generation INTEGER NOT NULL,
		state TEXT NOT NULL
	)`,
	`CREATE TABLE IF NOT EXISTS owners (
		namespace TEXT NOT NULL,
		owner_id TEXT NOT NULL,
		generation_order INTEGER NOT NULL,
		generation_token TEXT NOT NULL,
		fingerprint TEXT NOT NULL,
		PRIMARY KEY (namespace, owner_id)
	)`,
	`CREATE TABLE IF NOT EXISTS batch_receipts (
		namespace TEXT NOT NULL,
		owner_id TEXT NOT NULL,
		generation_order INTEGER NOT NULL,
		generation_token TEXT NOT NULL,
		batch_hash TEXT NOT NULL,
		row_count INTEGER NOT NULL,
		mode INTEGER NOT NULL,
		fingerprint TEXT NOT NULL,
		PRIMARY KEY (namespace, owner_id, generation_order, generation_token)
	)`,
	`CREATE TABLE IF NOT EXISTS receipt_rows (
		namespace TEXT NOT NULL,
		owner_id TEXT NOT NULL,
		generation_order INTEGER NOT NULL,
		generation_token TEXT NOT NULL,
		row_key TEXT NOT NULL,
		occurrence_hash TEXT NOT NULL,
		PRIMARY KEY (namespace, owner_id, generation_order, generation_token, row_key)
	)`,
	`CREATE TABLE IF NOT EXISTS staged_generations (
		namespace TEXT NOT NULL,
		owner_id TEXT NOT NULL,
		generation_order INTEGER NOT NULL,
		generation_token TEXT NOT NULL,
		mode INTEGER NOT NULL,
		state TEXT NOT NULL,
		PRIMARY KEY (namespace, owner_id, generation_order, generation_token)
	)`,
	`CREATE TABLE IF NOT EXISTS staged_occurrences (
		namespace TEXT NOT NULL,
		owner_id TEXT NOT NULL,
		generation_order INTEGER NOT NULL,
		generation_token TEXT NOT NULL,
		row_key TEXT NOT NULL,
		payload BLOB NOT NULL,
		occurrence_hash TEXT NOT NULL,
		vector_id TEXT NOT NULL,
		PRIMARY KEY (namespace, owner_id, generation_order, generation_token, row_key)
	)`,
	`CREATE TABLE IF NOT EXISTS occurrences (
		namespace TEXT NOT NULL,
		owner_id TEXT NOT NULL,
		row_key TEXT NOT NULL,
		sort_key TEXT NOT NULL,
		vector_id TEXT NOT NULL,
		source_blob_id TEXT NOT NULL,
		search_hash TEXT NOT NULL,
		source_length INTEGER NOT NULL,
		generation_order INTEGER NOT NULL,
		occurrence_hash TEXT NOT NULL,
		PRIMARY KEY (namespace, owner_id, row_key)
	)`,
	`CREATE INDEX IF NOT EXISTS occurrences_vector_id ON occurrences (vector_id)`,
	`CREATE INDEX IF NOT EXISTS occurrences_owner ON occurrences (namespace, owner_id)`,
	`CREATE TABLE IF NOT EXISTS source_blobs (
		blob_id TEXT PRIMARY KEY,
		content TEXT NOT NULL
	)`,
	`CREATE TABLE IF NOT EXISTS occurrence_scalars (
		namespace TEXT NOT NULL,
		owner_id TEXT NOT NULL,
		row_key TEXT NOT NULL,
		column_name TEXT NOT NULL,
		type INTEGER NOT NULL,
		string_value TEXT,
		int64_value INTEGER,
		bool_value INTEGER,
		is_null INTEGER NOT NULL,
		PRIMARY KEY (namespace, owner_id, row_key, column_name)
	)`,
	`CREATE INDEX IF NOT EXISTS occurrence_scalars_string ON occurrence_scalars (namespace, column_name, string_value)`,
	`CREATE INDEX IF NOT EXISTS occurrence_scalars_int64 ON occurrence_scalars (namespace, column_name, int64_value)`,
	`CREATE INDEX IF NOT EXISTS occurrence_scalars_bool ON occurrence_scalars (namespace, column_name, bool_value)`,
	`CREATE TABLE IF NOT EXISTS projection_events (
		namespace TEXT NOT NULL,
		owner_id TEXT NOT NULL,
		projection_order INTEGER NOT NULL,
		token TEXT NOT NULL,
		payload_hash TEXT NOT NULL,
		payload BLOB NOT NULL,
		fingerprint TEXT NOT NULL,
		PRIMARY KEY (namespace, owner_id, projection_order)
	)`,
	`CREATE TABLE IF NOT EXISTS effective_scalars (
		namespace TEXT NOT NULL,
		owner_id TEXT NOT NULL,
		row_key TEXT NOT NULL,
		column_name TEXT NOT NULL,
		type INTEGER NOT NULL,
		string_value TEXT,
		int64_value INTEGER,
		bool_value INTEGER,
		is_null INTEGER NOT NULL,
		projection_order INTEGER NOT NULL,
		PRIMARY KEY (namespace, owner_id, row_key, column_name)
	)`,
	`CREATE INDEX IF NOT EXISTS effective_scalars_string ON effective_scalars (namespace, column_name, string_value)`,
	`CREATE INDEX IF NOT EXISTS effective_scalars_int64 ON effective_scalars (namespace, column_name, int64_value)`,
	`CREATE INDEX IF NOT EXISTS effective_scalars_bool ON effective_scalars (namespace, column_name, bool_value)`,
	`CREATE TABLE IF NOT EXISTS search_snapshots (
		snapshot_id TEXT PRIMARY KEY,
		namespace TEXT NOT NULL,
		request_hash TEXT NOT NULL,
		visibility_revision INTEGER NOT NULL,
		projection_revision INTEGER NOT NULL,
		rank_config TEXT NOT NULL,
		expires_at INTEGER NOT NULL
	)`,
	`CREATE INDEX IF NOT EXISTS search_snapshots_expiry ON search_snapshots (expires_at)`,
	`CREATE TABLE IF NOT EXISTS search_results (
		snapshot_id TEXT NOT NULL,
		ordinal INTEGER NOT NULL,
		namespace TEXT NOT NULL,
		owner_id TEXT NOT NULL,
		row_key TEXT NOT NULL,
		source_blob_id TEXT NOT NULL,
		vector_id TEXT NOT NULL,
		effective_scalars TEXT NOT NULL,
		score REAL NOT NULL,
		PRIMARY KEY (snapshot_id, ordinal)
	)`,
	`CREATE INDEX IF NOT EXISTS search_results_vector_id ON search_results (vector_id)`,
	`CREATE INDEX IF NOT EXISTS search_results_source_blob_id ON search_results (source_blob_id)`,
}

// createCatalogSchema creates every table and index, including the lexical
// index tables, and records the schema version. A version 1 catalog without
// occurrences is upgraded to the current version. A version 1 catalog with
// occurrences returns an error that wraps [ErrStoreMismatch], because version
// 1 saves no SearchText to build the lexical index from. Any other saved
// version returns an error that wraps [ErrStoreMismatch].
func createCatalogSchema(ctx context.Context, tx *sql.Tx) error {
	statements := make([]string, 0, len(catalogSchemaStatements)+len(lexicalSchemaStatements))
	statements = append(statements, catalogSchemaStatements...)
	statements = append(statements, lexicalSchemaStatements...)
	for _, statement := range statements {
		if _, err := tx.ExecContext(ctx, statement); err != nil {
			slog.ErrorContext(ctx, "create catalog schema failed", "err", err)
			return fmt.Errorf("create catalog schema: %w", err)
		}
	}
	saved, found, err := readIdentityValue(ctx, tx, identityKeySchemaVersion)
	if err != nil {
		return err
	}
	want := strconv.Itoa(catalogSchemaVersion)
	switch {
	case !found:
		return writeIdentityValue(ctx, tx, identityKeySchemaVersion, want)
	case saved == want:
		return nil
	case saved == strconv.Itoa(lexicalSchemaFromVersion):
		return upgradeToLexicalSchema(ctx, tx, want)
	default:
		err := fmt.Errorf("%w: catalog schema version %s, this build reads %s", ErrStoreMismatch, saved, want)
		slog.ErrorContext(ctx, "catalog schema version mismatch", "err", err)
		return err
	}
}

// upgradeToLexicalSchema records the current schema version for a version 1
// catalog without occurrences, after createCatalogSchema created the empty
// lexical tables. It refuses a catalog with occurrences.
func upgradeToLexicalSchema(ctx context.Context, tx *sql.Tx, want string) error {
	var occurrences int64
	if err := tx.QueryRowContext(ctx, `SELECT COUNT(*) FROM occurrences`).Scan(&occurrences); err != nil {
		slog.ErrorContext(ctx, "count occurrences for the schema upgrade failed", "err", err)
		return fmt.Errorf("count occurrences for the schema upgrade: %w", err)
	}
	if occurrences > 0 {
		err := fmt.Errorf(
			"%w: catalog schema version %d has %d occurrences and saves no SearchText to build the lexical index; rebuild the catalog",
			ErrStoreMismatch, lexicalSchemaFromVersion, occurrences,
		)
		slog.ErrorContext(ctx, "catalog schema upgrade refused", "err", err)
		return err
	}
	slog.InfoContext(ctx, "catalog schema upgraded", "from", lexicalSchemaFromVersion, "to", want)
	return writeIdentityValue(ctx, tx, identityKeySchemaVersion, want)
}

// readIdentityValue returns the store_identity value for key and whether the
// row exists.
func readIdentityValue(ctx context.Context, tx *sql.Tx, key string) (string, bool, error) {
	var value string
	err := tx.QueryRowContext(ctx, `SELECT value FROM store_identity WHERE key = ?`, key).Scan(&value)
	if err == sql.ErrNoRows {
		return "", false, nil
	}
	if err != nil {
		slog.ErrorContext(ctx, "read catalog identity failed", "key", key, "err", err)
		return "", false, fmt.Errorf("read catalog identity %s: %w", key, err)
	}
	return value, true, nil
}

// writeIdentityValue inserts or replaces one store_identity value.
func writeIdentityValue(ctx context.Context, tx *sql.Tx, key string, value string) error {
	if _, err := tx.ExecContext(
		ctx,
		`INSERT INTO store_identity (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value`,
		key,
		value,
	); err != nil {
		slog.ErrorContext(ctx, "write catalog identity failed", "key", key, "err", err)
		return fmt.Errorf("write catalog identity %s: %w", key, err)
	}
	return nil
}

// incrementRevision adds one to a numeric store_identity counter and returns
// the new value.
func incrementRevision(ctx context.Context, tx *sql.Tx, key string) (int64, error) {
	saved, found, err := readIdentityValue(ctx, tx, key)
	if err != nil {
		return 0, err
	}
	var current int64
	if found {
		current, err = strconv.ParseInt(saved, 10, 64)
		if err != nil {
			slog.ErrorContext(ctx, "parse catalog revision failed", "key", key, "err", err)
			return 0, fmt.Errorf("parse catalog revision %s: %w", key, err)
		}
	}
	next := current + 1
	if err := writeIdentityValue(ctx, tx, key, strconv.FormatInt(next, 10)); err != nil {
		return 0, err
	}
	return next, nil
}

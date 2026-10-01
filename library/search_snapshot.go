package library

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	sqlite3 "github.com/mattn/go-sqlite3"
	"goodkind.io/lm-semantic-search/internal/clock"
)

// queryDatabasePageBytes is the page size of a query database. MaxTemporaryBytes
// divided by it is the page limit that SQLite enforces.
const queryDatabasePageBytes = 4096

// snapshotIDBytes is the random byte length of a snapshot ID.
const snapshotIDBytes = 16

// cursorVersion identifies the cursor encoding.
const cursorVersion = 1

// queryDatabaseSchema creates the tables of one query database. filter_sets
// stores one occurrence set per filter node. candidates stores every eligible
// occurrence copied from the catalog read transaction. query_vectors stores
// each distinct eligible vector with its copied identity and its exact score.
// postings and lexical_scores store the hybrid leg. dense_order,
// lexical_order, and final_order are keyed in rank order, and dense_ranks and
// lexical_ranks number their rows by rowid. Each ordering is a b-tree in the
// query database file, and the page limit of that file bounds it. ranked
// stores the final ordered result.
var queryDatabaseSchema = []string{
	`CREATE TABLE filter_sets (
		node INTEGER NOT NULL,
		owner_id TEXT NOT NULL,
		row_key TEXT NOT NULL,
		PRIMARY KEY (node, owner_id, row_key)
	) WITHOUT ROWID`,
	`CREATE TABLE candidates (
		owner_id TEXT NOT NULL,
		row_key TEXT NOT NULL,
		sort_key TEXT NOT NULL,
		vector_id TEXT NOT NULL,
		source_blob_id TEXT NOT NULL,
		search_hash TEXT NOT NULL,
		group_key TEXT NOT NULL,
		scalars TEXT NOT NULL,
		PRIMARY KEY (owner_id, row_key)
	) WITHOUT ROWID`,
	`CREATE TABLE query_vectors (
		vector_id TEXT PRIMARY KEY,
		identity_digest TEXT NOT NULL,
		vector_checksum TEXT NOT NULL,
		score REAL
	) WITHOUT ROWID`,
	`CREATE TABLE postings (
		search_hash TEXT NOT NULL,
		term_hash INTEGER NOT NULL,
		tf INTEGER NOT NULL,
		document_length INTEGER NOT NULL,
		PRIMARY KEY (search_hash, term_hash)
	) WITHOUT ROWID`,
	`CREATE TABLE lexical_scores (
		search_hash TEXT PRIMARY KEY,
		score REAL NOT NULL
	) WITHOUT ROWID`,
	`CREATE INDEX candidates_search_hash ON candidates (search_hash)`,
	`CREATE TABLE dense_order (
		negated_score REAL NOT NULL,
		owner_id TEXT NOT NULL,
		row_key TEXT NOT NULL,
		PRIMARY KEY (negated_score, owner_id, row_key)
	) WITHOUT ROWID`,
	`CREATE TABLE dense_ranks (
		rank INTEGER PRIMARY KEY,
		owner_id TEXT NOT NULL,
		row_key TEXT NOT NULL
	)`,
	`CREATE UNIQUE INDEX dense_ranks_key ON dense_ranks (owner_id, row_key)`,
	`CREATE TABLE lexical_order (
		negated_score REAL NOT NULL,
		owner_id TEXT NOT NULL,
		row_key TEXT NOT NULL,
		PRIMARY KEY (negated_score, owner_id, row_key)
	) WITHOUT ROWID`,
	`CREATE TABLE lexical_ranks (
		rank INTEGER PRIMARY KEY,
		owner_id TEXT NOT NULL,
		row_key TEXT NOT NULL
	)`,
	`CREATE UNIQUE INDEX lexical_ranks_key ON lexical_ranks (owner_id, row_key)`,
	`CREATE TABLE final_order (
		negated_score REAL NOT NULL,
		sort_key TEXT NOT NULL,
		owner_id TEXT NOT NULL,
		row_key TEXT NOT NULL,
		source_blob_id TEXT NOT NULL,
		vector_id TEXT NOT NULL,
		scalars TEXT NOT NULL,
		group_key TEXT NOT NULL,
		PRIMARY KEY (negated_score, sort_key, owner_id, row_key)
	) WITHOUT ROWID`,
	`CREATE TABLE group_counts (
		group_key TEXT PRIMARY KEY,
		count INTEGER NOT NULL
	) WITHOUT ROWID`,
	`CREATE TABLE ranked (
		ordinal INTEGER PRIMARY KEY,
		owner_id TEXT NOT NULL,
		row_key TEXT NOT NULL,
		source_blob_id TEXT NOT NULL,
		vector_id TEXT NOT NULL,
		scalars TEXT NOT NULL,
		score REAL NOT NULL
	)`,
}

// queryDatabasePattern is the [os.CreateTemp] pattern of a query database file.
// queryDatabasePrefix is its fixed part.
const (
	queryDatabasePattern = ".lms-query-*.sqlite"
	queryDatabasePrefix  = ".lms-query-"
)

// removeStaleQueryDatabases deletes query database files in directory that
// were last written more than staleAfter ago. A process that ended during a
// search leaves its file behind. A running search writes its file within its
// QueryTimeout. A failure to list or delete a file is logged and does not fail
// the search.
func removeStaleQueryDatabases(ctx context.Context, directory string, staleAfter time.Duration) {
	entries, err := os.ReadDir(directory)
	if err != nil {
		slog.WarnContext(ctx, "list query databases failed", "directory", directory, "err", err)
		return
	}
	cutoff := clock.Now().Add(-staleAfter)
	var removed []string
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasPrefix(entry.Name(), queryDatabasePrefix) {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			slog.WarnContext(ctx, "read query database file information failed", "file", entry.Name(), "err", err)
			continue
		}
		if !info.ModTime().Before(cutoff) {
			continue
		}
		path := filepath.Join(directory, entry.Name())
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			slog.WarnContext(ctx, "remove stale query database failed", "path", path, "err", err)
			continue
		}
		removed = append(removed, entry.Name())
	}
	if len(removed) > 0 {
		slog.InfoContext(ctx, "removed stale query databases", "directory", directory, "files", removed)
	}
}

// queryDatabase is one per-query SQLite file in the catalog directory. Every
// statement runs on one connection. That connection sets the page limit that
// bounds the file size.
type queryDatabase struct {
	path     string
	database *sql.DB
	conn     *sql.Conn
}

// openQueryDatabase creates an empty query database limited to maxBytes.
func openQueryDatabase(ctx context.Context, catalogPath string, maxBytes int64, staleAfter time.Duration) (*queryDatabase, error) {
	removeStaleQueryDatabases(ctx, filepath.Dir(catalogPath), staleAfter)
	file, err := os.CreateTemp(filepath.Dir(catalogPath), queryDatabasePattern)
	if err != nil {
		slog.ErrorContext(ctx, "create query database failed", "err", err)
		return nil, fmt.Errorf("create query database: %w", err)
	}
	path := file.Name()
	if err := file.Close(); err != nil {
		slog.ErrorContext(ctx, "create query database failed", "path", path, "err", err)
		return nil, errors.Join(fmt.Errorf("close new query database %s: %w", path, err), os.Remove(path))
	}
	parameters := url.Values{}
	parameters.Set("_journal_mode", "OFF")
	parameters.Set("_synchronous", "OFF")
	database, err := sql.Open("sqlite3", "file:"+path+"?"+parameters.Encode())
	if err != nil {
		slog.ErrorContext(ctx, "open query database failed", "path", path, "err", err)
		return nil, errors.Join(fmt.Errorf("open query database %s: %w", path, err), os.Remove(path))
	}
	database.SetMaxOpenConns(1)
	query := &queryDatabase{path: path, database: database, conn: nil}
	if err := query.initialize(ctx, maxBytes); err != nil {
		return nil, errors.Join(err, query.close())
	}
	return query, nil
}

func (query *queryDatabase) initialize(ctx context.Context, maxBytes int64) error {
	conn, err := query.database.Conn(ctx)
	if err != nil {
		slog.ErrorContext(ctx, "open query database connection failed", "err", err)
		return fmt.Errorf("open query database connection: %w", err)
	}
	query.conn = conn
	// The query database statements open no temporary b-tree or sorter;
	// TestQueryDatabaseStatementsUseNoTemporaryStore checks their bytecode.
	// temp_store FILE keeps any unexpected temporary table off the process heap.
	statements := []string{
		"PRAGMA page_size = " + strconv.Itoa(queryDatabasePageBytes),
		"PRAGMA max_page_count = " + strconv.FormatInt(max(maxBytes/queryDatabasePageBytes, 1), 10),
		"PRAGMA temp_store = FILE",
	}
	statements = append(statements, queryDatabaseSchema...)
	for _, statement := range statements {
		if _, err := conn.ExecContext(ctx, statement); err != nil {
			return queryDatabaseError(ctx, "create query database schema", err)
		}
	}
	return nil
}

// close closes the query database and deletes its file.
func (query *queryDatabase) close() error {
	var closeErr error
	if query.conn != nil {
		closeErr = query.conn.Close()
	}
	closeErr = errors.Join(closeErr, query.database.Close())
	if err := os.Remove(query.path); err != nil && !errors.Is(err, os.ErrNotExist) {
		closeErr = errors.Join(closeErr, err)
	}
	if closeErr != nil {
		slog.Error("close query database failed", "path", query.path, "err", closeErr)
		return fmt.Errorf("close query database %s: %w", query.path, closeErr)
	}
	return nil
}

// queryDatabaseError wraps a query database failure. MaxTemporaryBytes sets
// the page limit, and a full database wraps [ErrResourceLimit].
func queryDatabaseError(ctx context.Context, operation string, err error) error {
	var sqliteErr sqlite3.Error
	if errors.As(err, &sqliteErr) && sqliteErr.Code == sqlite3.ErrFull {
		limited := fmt.Errorf("%w: %s: the query database is at its MaxTemporaryBytes page limit: %w", ErrResourceLimit, operation, err)
		slog.WarnContext(ctx, "search exceeded temporary disk budget", "err", limited)
		return limited
	}
	slog.ErrorContext(ctx, "query database operation failed", "operation", operation, "err", err)
	return fmt.Errorf("%s: %w", operation, err)
}

// snapshotRevisions are the catalog revisions that one result snapshot fixes.
type snapshotRevisions struct {
	Visibility int64  `json:"visibility"`
	Projection int64  `json:"projection"`
	Statistics uint64 `json:"statistics"`
}

func readRevisions(ctx context.Context, tx *sql.Tx) (snapshotRevisions, error) {
	var revisions snapshotRevisions
	for _, target := range []struct {
		key   string
		value *int64
	}{
		{key: identityKeyVisibilityRevision, value: &revisions.Visibility},
		{key: identityKeyProjectionRevision, value: &revisions.Projection},
	} {
		saved, found, err := readIdentityValue(ctx, tx, target.key)
		if err != nil {
			return snapshotRevisions{}, err
		}
		if !found {
			continue
		}
		parsed, err := strconv.ParseInt(saved, 10, 64)
		if err != nil {
			slog.ErrorContext(ctx, "parse catalog revision failed", "key", target.key, "err", err)
			return snapshotRevisions{}, fmt.Errorf("parse catalog revision %s: %w", target.key, err)
		}
		*target.value = parsed
	}
	return revisions, nil
}

// candidateColumns selects the ranking inputs of occurrences of :namespace:
// the key, SortKey, vector identity, source blob, search hash, the group key
// of the column bound to :group_column, and the effective scalars as one JSON
// object. The group key is 'a' for an absent column, 'n' for null, and 'v'
// followed by the value otherwise. Each JSON member maps a column name to
// [type, is_null, string_value, int64_value, bool_value], and a projected
// value replaces the published value of the same column.
const candidateColumns = `SELECT o.owner_id, o.row_key, o.sort_key, o.vector_id, v.identity_digest, v.vector_checksum,
	o.source_blob_id, o.search_hash,
	CASE WHEN ge.column_name IS NULL AND gs.column_name IS NULL THEN 'a'
		WHEN (CASE WHEN ge.column_name IS NOT NULL THEN ge.is_null ELSE gs.is_null END) = 1 THEN 'n'
		ELSE 'v' || CAST(CASE WHEN ge.column_name IS NOT NULL THEN COALESCE(ge.string_value, ge.int64_value, ge.bool_value)
			ELSE COALESCE(gs.string_value, gs.int64_value, gs.bool_value) END AS TEXT) END,
	json_patch(
		(SELECT json_group_object(s.column_name, json_array(s.type, s.is_null, s.string_value, s.int64_value, s.bool_value))
			FROM occurrence_scalars s WHERE s.namespace = o.namespace AND s.owner_id = o.owner_id AND s.row_key = o.row_key),
		(SELECT json_group_object(e.column_name, json_array(e.type, e.is_null, e.string_value, e.int64_value, e.bool_value))
			FROM effective_scalars e WHERE e.namespace = o.namespace AND e.owner_id = o.owner_id AND e.row_key = o.row_key))
FROM occurrences o
LEFT JOIN vectors v ON v.vector_id = o.vector_id
LEFT JOIN effective_scalars ge ON ge.namespace = o.namespace AND ge.owner_id = o.owner_id
	AND ge.row_key = o.row_key AND ge.column_name = :group_column
LEFT JOIN occurrence_scalars gs ON gs.namespace = o.namespace AND gs.owner_id = o.owner_id
	AND gs.row_key = o.row_key AND gs.column_name = :group_column
WHERE o.namespace = :namespace`

// Candidate statements. allCandidatesStatement reads every occurrence of the
// namespace. oneCandidateStatement reads one occurrence that the filter
// selected. eligibleKeysStatement reads the true rows of the filter root node
// from the query database.
const (
	allCandidatesStatement = candidateColumns
	oneCandidateStatement  = candidateColumns + ` AND o.owner_id = :owner_id AND o.row_key = :row_key`
	eligibleKeysStatement  = `SELECT owner_id, row_key FROM filter_sets WHERE node = :node`
)

// copyCandidates copies every eligible occurrence of the plan and the identity
// of each distinct eligible vector from the catalog read transaction into the
// query database. Without a filter it reads every occurrence of the
// namespace. With a filter it evaluates the filter and reads each occurrence
// with a true root value. An occurrence that references a vector without a
// catalog row fails the copy.
func copyCandidates(ctx context.Context, tx *sql.Tx, query *queryDatabase, plan searchPlan) (count int64, err error) {
	writer, err := query.conn.BeginTx(ctx, nil)
	if err != nil {
		return 0, queryDatabaseError(ctx, "begin candidate copy", err)
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, writer.Rollback())
		}
	}()
	namespace := sql.Named("namespace", plan.request.Namespace)
	groupColumn := sql.Named("group_column", plan.request.GroupBy)
	if plan.request.Filter == nil {
		rows, queryErr := tx.QueryContext(ctx, allCandidatesStatement, namespace, groupColumn)
		if queryErr != nil {
			slog.ErrorContext(ctx, "select occurrences failed", "namespace", plan.request.Namespace, "err", queryErr)
			return 0, fmt.Errorf("select occurrences of %s: %w", plan.request.Namespace, queryErr)
		}
		count, err = insertCandidateRows(ctx, writer, rows)
	} else {
		evaluator := filterEvaluator{catalog: tx, writer: writer, namespace: plan.request.Namespace, columns: declaredColumns(plan.spec), nextNode: 0}
		root, evaluateErr := evaluator.evaluate(ctx, *plan.request.Filter, false)
		if evaluateErr != nil {
			return 0, evaluateErr
		}
		count, err = copyEligibleCandidates(ctx, tx, writer, root, namespace, groupColumn)
	}
	if err != nil {
		return 0, err
	}
	if err := writer.Commit(); err != nil {
		return 0, queryDatabaseError(ctx, "commit candidate copy", err)
	}
	return count, nil
}

// copyEligibleCandidates reads the ranking inputs of every occurrence with a
// true value at the filter root node.
func copyEligibleCandidates(
	ctx context.Context,
	tx *sql.Tx,
	writer *sql.Tx,
	root int,
	namespace sql.NamedArg,
	groupColumn sql.NamedArg,
) (count int64, err error) {
	lookup, err := tx.PrepareContext(ctx, oneCandidateStatement)
	if err != nil {
		slog.ErrorContext(ctx, "prepare occurrence lookup failed", "err", err)
		return 0, fmt.Errorf("prepare occurrence lookup: %w", err)
	}
	defer func() {
		err = errors.Join(err, closeStatement(ctx, lookup))
	}()
	keys, err := writer.QueryContext(ctx, eligibleKeysStatement, sql.Named("node", root))
	if err != nil {
		return 0, queryDatabaseError(ctx, "read filtered occurrences", err)
	}
	defer func() {
		err = errors.Join(err, closeRows(ctx, keys))
	}()
	for keys.Next() {
		var ownerID, rowKey string
		if err := keys.Scan(&ownerID, &rowKey); err != nil {
			return 0, queryDatabaseError(ctx, "scan filtered occurrence", err)
		}
		rows, err := lookup.QueryContext(ctx, namespace, groupColumn, sql.Named("owner_id", ownerID), sql.Named("row_key", rowKey))
		if err != nil {
			slog.ErrorContext(ctx, "read filtered occurrence failed", "err", err)
			return 0, fmt.Errorf("read occurrence %s/%s: %w", ownerID, rowKey, err)
		}
		copied, err := insertCandidateRows(ctx, writer, rows)
		if err != nil {
			return 0, err
		}
		if copied != 1 {
			err := fmt.Errorf("filtered occurrence %s/%s has %d catalog rows in the read snapshot", ownerID, rowKey, copied)
			slog.ErrorContext(ctx, "filtered occurrence lookup failed", "err", err)
			return 0, err
		}
		count++
	}
	if err := keys.Err(); err != nil {
		return 0, queryDatabaseError(ctx, "read filtered occurrences", err)
	}
	return count, nil
}

// eligibleRow is one eligible occurrence as the catalog copy reads it.
type eligibleRow struct {
	ownerID, rowKey, sortKey, vectorID, blobID, searchHash, groupKey, scalars string
	digest, checksum                                                          sql.NullString
}

// insertCandidateRows copies every row of a candidate statement into the
// query database and closes rows.
func insertCandidateRows(ctx context.Context, writer *sql.Tx, rows *sql.Rows) (count int64, err error) {
	defer func() {
		err = errors.Join(err, closeRows(ctx, rows))
	}()
	for rows.Next() {
		var row eligibleRow
		if err := rows.Scan(
			&row.ownerID, &row.rowKey, &row.sortKey, &row.vectorID, &row.digest,
			&row.checksum, &row.blobID, &row.searchHash, &row.groupKey, &row.scalars,
		); err != nil {
			slog.ErrorContext(ctx, "scan eligible occurrence failed", "err", err)
			return 0, fmt.Errorf("scan eligible occurrence: %w", err)
		}
		if err := insertCandidate(ctx, writer, row); err != nil {
			return 0, err
		}
		count++
	}
	if err := rows.Err(); err != nil {
		slog.ErrorContext(ctx, "read eligible occurrences failed", "err", err)
		return 0, fmt.Errorf("read eligible occurrences: %w", err)
	}
	return count, nil
}

func insertCandidate(ctx context.Context, writer *sql.Tx, row eligibleRow) error {
	if !row.digest.Valid || !row.checksum.Valid {
		missing := fmt.Errorf("%w: occurrence %s/%s references vector %s without a catalog row",
			ErrVectorMissing, row.ownerID, row.rowKey, row.vectorID)
		slog.ErrorContext(ctx, "eligible occurrence has no catalog vector", "err", missing)
		return missing
	}
	if _, err := writer.ExecContext(ctx, insertCandidateStatement,
		row.ownerID, row.rowKey, row.sortKey, row.vectorID, row.blobID, row.searchHash, row.groupKey, row.scalars,
	); err != nil {
		return queryDatabaseError(ctx, "copy eligible occurrence", err)
	}
	if _, err := writer.ExecContext(ctx, insertQueryVectorStatement,
		row.vectorID, row.digest.String, row.checksum.String,
	); err != nil {
		return queryDatabaseError(ctx, "copy eligible vector identity", err)
	}
	return nil
}

// rankedRow is one row of a ranked result.
type rankedRow struct {
	ordinal      int64
	ownerID      string
	rowKey       string
	sourceBlobID string
	vectorID     string
	scalars      string
	score        float64
}

// readRankedRows returns up to limit rows of the query database ranking from
// ordinal first.
func readRankedRows(ctx context.Context, query *queryDatabase, first int64, limit int) (_ []rankedRow, err error) {
	rows, err := query.conn.QueryContext(ctx, rankedPageStatement, first, limit)
	if err != nil {
		return nil, queryDatabaseError(ctx, "read ranked rows", err)
	}
	defer func() {
		err = errors.Join(err, closeRows(ctx, rows))
	}()
	var ranked []rankedRow
	for rows.Next() {
		var row rankedRow
		if err := rows.Scan(&row.ordinal, &row.ownerID, &row.rowKey, &row.sourceBlobID, &row.vectorID, &row.scalars, &row.score); err != nil {
			return nil, queryDatabaseError(ctx, "scan ranked row", err)
		}
		ranked = append(ranked, row)
	}
	if err := rows.Err(); err != nil {
		return nil, queryDatabaseError(ctx, "read ranked rows", err)
	}
	return ranked, nil
}

// snapshotRecord is the rank_config JSON of one search_snapshots row.
type snapshotRecord struct {
	Rank        rankConfig        `json:"rank"`
	RankHash    string            `json:"rank_hash"`
	Revisions   snapshotRevisions `json:"revisions"`
	ResultBytes int64             `json:"result_bytes"`
}

// persistSnapshot deletes expired snapshots, checks the snapshot disk budget,
// and writes the full ranking of the query database as one snapshot in one
// catalog write transaction. Every result row records its source blob ID and
// its vector ID. A reference audit reads both IDs from unexpired snapshots.
func (library *Library) persistSnapshot(
	ctx context.Context,
	query *queryDatabase,
	plan searchPlan,
	revisions snapshotRevisions,
	phases *searchPhases,
) (string, error) {
	resultBytes, err := rankedResultBytes(ctx, query, plan.request.Namespace)
	if err != nil {
		return "", err
	}
	snapshotID, err := newSnapshotID()
	if err != nil {
		return "", err
	}
	record, err := json.Marshal(snapshotRecord{
		Rank:        plan.rank,
		RankHash:    plan.rankHash,
		Revisions:   revisions,
		ResultBytes: resultBytes,
	})
	if err != nil {
		slog.ErrorContext(ctx, "encode snapshot record failed", "err", err)
		return "", fmt.Errorf("encode snapshot record: %w", err)
	}
	now := clock.Now()
	expiresAt := now.Add(library.config.SnapshotTTL).UnixMilli()
	// Library.write opens the transaction with BEGIN IMMEDIATE, which waits for
	// any other SQLite writer before the closure runs.
	requested := clock.Now()
	var writing time.Time
	err = library.write(ctx, func(tx *sql.Tx) error {
		writing = phases.mark(&phases.writeWait, requested)
		if err := deleteExpiredSnapshots(ctx, tx, now.UnixMilli()); err != nil {
			return err
		}
		if err := checkSnapshotBudget(ctx, tx, now.UnixMilli(), resultBytes, library.config.MaxSnapshotBytes); err != nil {
			return err
		}
		if _, err := tx.ExecContext(ctx, insertSnapshotStatement,
			snapshotID, plan.request.Namespace, plan.requestHash, revisions.Visibility, revisions.Projection, string(record), expiresAt,
		); err != nil {
			slog.ErrorContext(ctx, "save search snapshot failed", "err", err)
			return fmt.Errorf("save search snapshot: %w", err)
		}
		return copyRankedResults(ctx, tx, query, snapshotID, plan.request.Namespace)
	})
	if err != nil {
		return "", err
	}
	phases.mark(&phases.write, writing)
	return snapshotID, nil
}

// rankedResultBytes returns the logical bytes that the ranking adds to
// search_results: the byte length of every text column plus 8 bytes for each
// numeric column.
func rankedResultBytes(ctx context.Context, query *queryDatabase, namespace string) (int64, error) {
	const numericColumnBytes = 8
	const numericColumns = 2
	perRow := int64(snapshotIDBytes*2+len(namespace)) + numericColumns*numericColumnBytes
	var total sql.NullInt64
	if err := query.conn.QueryRowContext(ctx, rankedBytesStatement, perRow).Scan(&total); err != nil {
		return 0, queryDatabaseError(ctx, "measure ranked result bytes", err)
	}
	return total.Int64, nil
}

// deleteExpiredSnapshots reads the IDs of expired snapshots through the
// expiry index and deletes each snapshot's result rows and snapshot row by
// primary key.
func deleteExpiredSnapshots(ctx context.Context, tx *sql.Tx, nowMilli int64) error {
	expired, err := expiredSnapshotIDs(ctx, tx, nowMilli)
	if err != nil {
		return err
	}
	for _, snapshotID := range expired {
		for _, statement := range []string{deleteSnapshotResultsStatement, deleteSnapshotStatement} {
			if _, err := tx.ExecContext(ctx, statement, snapshotID); err != nil {
				slog.ErrorContext(ctx, "delete expired search snapshot failed", "snapshot", snapshotID, "err", err)
				return fmt.Errorf("delete expired search snapshot %s: %w", snapshotID, err)
			}
		}
	}
	return nil
}

func expiredSnapshotIDs(ctx context.Context, tx *sql.Tx, nowMilli int64) (_ []string, err error) {
	rows, err := tx.QueryContext(ctx, expiredSnapshotsStatement, nowMilli)
	if err != nil {
		slog.ErrorContext(ctx, "read expired search snapshots failed", "err", err)
		return nil, fmt.Errorf("read expired search snapshots: %w", err)
	}
	defer func() {
		err = errors.Join(err, closeRows(ctx, rows))
	}()
	var expired []string
	for rows.Next() {
		var snapshotID string
		if err := rows.Scan(&snapshotID); err != nil {
			slog.ErrorContext(ctx, "scan expired search snapshot failed", "err", err)
			return nil, fmt.Errorf("scan expired search snapshot: %w", err)
		}
		expired = append(expired, snapshotID)
	}
	if err := rows.Err(); err != nil {
		slog.ErrorContext(ctx, "read expired search snapshots failed", "err", err)
		return nil, fmt.Errorf("read expired search snapshots: %w", err)
	}
	return expired, nil
}

// checkSnapshotBudget fails with [ErrResourceLimit] when the unexpired
// snapshots and the new snapshot exceed maxBytes of logical result bytes.
func checkSnapshotBudget(ctx context.Context, tx *sql.Tx, nowMilli int64, newBytes int64, maxBytes int64) error {
	var existing sql.NullInt64
	if err := tx.QueryRowContext(ctx, snapshotBytesStatement, nowMilli).Scan(&existing); err != nil {
		slog.ErrorContext(ctx, "measure search snapshot bytes failed", "err", err)
		return fmt.Errorf("measure search snapshot bytes: %w", err)
	}
	if existing.Int64+newBytes > maxBytes {
		limited := fmt.Errorf(
			"%w: the snapshot needs %d result bytes, unexpired snapshots use %d, and MaxSnapshotBytes is %d",
			ErrResourceLimit, newBytes, existing.Int64, maxBytes,
		)
		slog.WarnContext(ctx, "search exceeded snapshot disk budget", "err", limited)
		return limited
	}
	return nil
}

func copyRankedResults(ctx context.Context, tx *sql.Tx, query *queryDatabase, snapshotID string, namespace string) (err error) {
	rows, err := query.conn.QueryContext(ctx, rankedRowsStatement)
	if err != nil {
		return queryDatabaseError(ctx, "read ranking for the snapshot", err)
	}
	defer func() {
		err = errors.Join(err, closeRows(ctx, rows))
	}()
	insert, err := tx.PrepareContext(ctx, insertSearchResultStatement)
	if err != nil {
		slog.ErrorContext(ctx, "prepare search result insert failed", "err", err)
		return fmt.Errorf("prepare search result insert: %w", err)
	}
	defer func() {
		err = errors.Join(err, closeStatement(ctx, insert))
	}()
	for rows.Next() {
		var row rankedRow
		if err := rows.Scan(&row.ordinal, &row.ownerID, &row.rowKey, &row.sourceBlobID, &row.vectorID, &row.scalars, &row.score); err != nil {
			return queryDatabaseError(ctx, "scan ranking for the snapshot", err)
		}
		if _, err := insert.ExecContext(ctx,
			snapshotID, row.ordinal, namespace, row.ownerID, row.rowKey, row.sourceBlobID, row.vectorID, row.scalars, row.score,
		); err != nil {
			slog.ErrorContext(ctx, "save search result failed", "ordinal", row.ordinal, "err", err)
			return fmt.Errorf("save search result %d: %w", row.ordinal, err)
		}
	}
	if err := rows.Err(); err != nil {
		return queryDatabaseError(ctx, "read ranking for the snapshot", err)
	}
	return nil
}

func newSnapshotID() (string, error) {
	random := make([]byte, snapshotIDBytes)
	if _, err := rand.Read(random); err != nil {
		slog.Error("generate snapshot ID failed", "err", err)
		return "", fmt.Errorf("generate snapshot ID: %w", err)
	}
	return hex.EncodeToString(random), nil
}

// searchCursor is the decoded form of an opaque cursor. It stores the snapshot
// ID, the next ordinal, the request hash, the rank configuration hash, and the
// revisions that page one fixed.
type searchCursor struct {
	Version     int               `json:"v"`
	SnapshotID  string            `json:"snapshot"`
	NextOrdinal int64             `json:"ordinal"`
	RequestHash string            `json:"request"`
	RankHash    string            `json:"rank"`
	Revisions   snapshotRevisions `json:"revisions"`
}

func encodeCursor(cursor searchCursor) (string, error) {
	encoded, err := json.Marshal(cursor)
	if err != nil {
		slog.Error("encode search cursor failed", "err", err)
		return "", fmt.Errorf("encode search cursor: %w", err)
	}
	return base64.RawURLEncoding.EncodeToString(encoded), nil
}

// decodeCursor parses a cursor. A cursor that is not the output of
// [encodeCursor] returns an error that wraps [ErrInvalidRequest].
func decodeCursor(text string) (searchCursor, error) {
	encoded, err := base64.RawURLEncoding.DecodeString(text)
	if err != nil {
		return searchCursor{}, invalidRequest("search cursor is not valid base64url")
	}
	var cursor searchCursor
	if err := json.Unmarshal(encoded, &cursor); err != nil {
		return searchCursor{}, invalidRequest("search cursor is not a library cursor")
	}
	if cursor.Version != cursorVersion || cursor.SnapshotID == "" || cursor.NextOrdinal < 1 {
		return searchCursor{}, invalidRequest("search cursor is not a library cursor")
	}
	return cursor, nil
}

// loadSnapshotForCursor checks that the cursor's snapshot exists, has not
// expired, and matches the cursor and the request.
func loadSnapshotForCursor(ctx context.Context, tx *sql.Tx, cursor searchCursor, plan searchPlan, nowMilli int64) error {
	var namespace, requestHash, record string
	var visibility, projection, expiresAt int64
	err := tx.QueryRowContext(ctx, cursorSnapshotStatement, cursor.SnapshotID).Scan(&namespace, &requestHash, &visibility, &projection, &record, &expiresAt)
	if errors.Is(err, sql.ErrNoRows) {
		expired := fmt.Errorf("%w: snapshot %s does not exist", ErrCursorExpired, cursor.SnapshotID)
		slog.WarnContext(ctx, "search cursor rejected", "err", expired)
		return expired
	}
	if err != nil {
		slog.ErrorContext(ctx, "read search snapshot failed", "err", err)
		return fmt.Errorf("read search snapshot %s: %w", cursor.SnapshotID, err)
	}
	if expiresAt <= nowMilli {
		expired := fmt.Errorf("%w: snapshot %s expired", ErrCursorExpired, cursor.SnapshotID)
		slog.WarnContext(ctx, "search cursor rejected", "err", expired)
		return expired
	}
	var saved snapshotRecord
	if err := json.Unmarshal([]byte(record), &saved); err != nil {
		slog.ErrorContext(ctx, "decode search snapshot failed", "err", err)
		return fmt.Errorf("decode search snapshot %s: %w", cursor.SnapshotID, err)
	}
	matches := namespace == plan.request.Namespace &&
		requestHash == plan.requestHash &&
		saved.RankHash == plan.rankHash &&
		saved.RankHash == cursor.RankHash &&
		visibility == cursor.Revisions.Visibility &&
		projection == cursor.Revisions.Projection &&
		saved.Revisions == cursor.Revisions
	if !matches {
		mismatch := fmt.Errorf("%w: snapshot %s was created for another request or rank configuration", ErrCursorMismatch, cursor.SnapshotID)
		slog.WarnContext(ctx, "search cursor rejected", "err", mismatch)
		return mismatch
	}
	return nil
}

// readSnapshotRows returns up to limit persisted rows of a snapshot from
// ordinal first.
func readSnapshotRows(ctx context.Context, tx *sql.Tx, snapshotID string, first int64, limit int) (_ []rankedRow, err error) {
	rows, err := tx.QueryContext(ctx, snapshotPageStatement, snapshotID, first, limit)
	if err != nil {
		slog.ErrorContext(ctx, "read search results failed", "err", err)
		return nil, fmt.Errorf("read search results of %s: %w", snapshotID, err)
	}
	defer func() {
		err = errors.Join(err, closeRows(ctx, rows))
	}()
	var ranked []rankedRow
	for rows.Next() {
		var row rankedRow
		if err := rows.Scan(&row.ordinal, &row.ownerID, &row.rowKey, &row.sourceBlobID, &row.vectorID, &row.scalars, &row.score); err != nil {
			slog.ErrorContext(ctx, "scan search result failed", "err", err)
			return nil, fmt.Errorf("scan search result: %w", err)
		}
		ranked = append(ranked, row)
	}
	if err := rows.Err(); err != nil {
		slog.ErrorContext(ctx, "read search results failed", "err", err)
		return nil, fmt.Errorf("read search results of %s: %w", snapshotID, err)
	}
	return ranked, nil
}

// buildHits reads the source text of each row from its immutable source blob
// and decodes its effective scalars.
func buildHits(ctx context.Context, tx *sql.Tx, namespace string, rows []rankedRow) ([]SearchHit, error) {
	hits := make([]SearchHit, 0, len(rows))
	for _, row := range rows {
		var content string
		err := tx.QueryRowContext(ctx, sourceBlobStatement, row.sourceBlobID).Scan(&content)
		if err != nil {
			slog.ErrorContext(ctx, "read source blob failed", "blob_id", row.sourceBlobID, "err", err)
			return nil, fmt.Errorf("read source blob %s of %s/%s: %w", row.sourceBlobID, row.ownerID, row.rowKey, err)
		}
		scalars, err := decodeEffectiveScalars(row.scalars)
		if err != nil {
			return nil, err
		}
		hits = append(hits, SearchHit{
			ID:         OccurrenceID{Namespace: namespace, OwnerID: row.ownerID, RowKey: row.rowKey},
			SourceText: content,
			Scalars:    scalars,
			Score:      row.score,
		})
	}
	return hits, nil
}

// effectiveScalarFields is the member length of one [candidateColumns]
// column entry.
const effectiveScalarFields = 5

// decodeEffectiveScalars parses the object that [candidateColumns]
// produces.
func decodeEffectiveScalars(encoded string) (map[string]ScalarValue, error) {
	var members map[string][]json.RawMessage
	if err := json.Unmarshal([]byte(encoded), &members); err != nil {
		slog.Error("decode effective scalars failed", "err", err)
		return nil, fmt.Errorf("decode effective scalars: %w", err)
	}
	scalars := make(map[string]ScalarValue, len(members))
	for name, fields := range members {
		value, err := decodeEffectiveScalar(name, fields)
		if err != nil {
			return nil, err
		}
		scalars[name] = value
	}
	return scalars, nil
}

func decodeEffectiveScalar(name string, fields []json.RawMessage) (ScalarValue, error) {
	if len(fields) != effectiveScalarFields {
		return ScalarValue{}, fmt.Errorf("decode effective scalar %s: %d fields, want %d", name, len(fields), effectiveScalarFields)
	}
	var value ScalarValue
	var isNull int64
	var stringValue *string
	var int64Value *int64
	var boolValue *int64
	err := errors.Join(
		json.Unmarshal(fields[0], &value.Type),
		json.Unmarshal(fields[1], &isNull),
		json.Unmarshal(fields[2], &stringValue),
		json.Unmarshal(fields[3], &int64Value),
		json.Unmarshal(fields[4], &boolValue),
	)
	if err != nil {
		slog.Error("decode effective scalar failed", "column", name, "err", err)
		return ScalarValue{}, fmt.Errorf("decode effective scalar %s: %w", name, err)
	}
	value.Null = isNull != 0
	if stringValue != nil {
		value.String = *stringValue
	}
	if int64Value != nil {
		value.Int64 = *int64Value
	}
	if boolValue != nil {
		value.Bool = *boolValue != 0
	}
	return value, nil
}

// Query database statements of the candidate copy and the ranked result.
const (
	insertCandidateStatement = `INSERT INTO candidates (owner_id, row_key, sort_key, vector_id, source_blob_id, search_hash, group_key, scalars)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
	insertQueryVectorStatement = `INSERT OR IGNORE INTO query_vectors (vector_id, identity_digest, vector_checksum) VALUES (?, ?, ?)`
	rankedPageStatement        = `SELECT ordinal, owner_id, row_key, source_blob_id, vector_id, scalars, score FROM ranked
		WHERE ordinal >= ? ORDER BY ordinal LIMIT ?`
	rankedRowsStatement  = `SELECT ordinal, owner_id, row_key, source_blob_id, vector_id, scalars, score FROM ranked ORDER BY ordinal`
	rankedBytesStatement = `SELECT SUM(? + length(CAST(owner_id AS BLOB)) + length(CAST(row_key AS BLOB)) + length(CAST(source_blob_id AS BLOB))
			+ length(CAST(vector_id AS BLOB)) + length(CAST(scalars AS BLOB))) FROM ranked`
)

// Catalog statements of the snapshot, the cursor pages, and the hits.
const (
	insertSnapshotStatement = `INSERT INTO search_snapshots
		(snapshot_id, namespace, request_hash, visibility_revision, projection_revision, rank_config, expires_at)
		VALUES (?, ?, ?, ?, ?, ?, ?)`
	expiredSnapshotsStatement      = `SELECT snapshot_id FROM search_snapshots WHERE expires_at <= ?`
	deleteSnapshotResultsStatement = `DELETE FROM search_results WHERE snapshot_id = ?`
	deleteSnapshotStatement        = `DELETE FROM search_snapshots WHERE snapshot_id = ?`
	snapshotBytesStatement         = `SELECT SUM(json_extract(rank_config, '$.result_bytes')) FROM search_snapshots WHERE expires_at > ?`
	insertSearchResultStatement    = `INSERT INTO search_results
		(snapshot_id, ordinal, namespace, owner_id, row_key, source_blob_id, vector_id, effective_scalars, score)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
	cursorSnapshotStatement = `SELECT namespace, request_hash, visibility_revision, projection_revision, rank_config, expires_at
		FROM search_snapshots WHERE snapshot_id = ?`
	renewSnapshotStatement = `UPDATE search_snapshots SET expires_at = MAX(expires_at, ?) WHERE snapshot_id = ?`
	snapshotPageStatement  = `SELECT ordinal, owner_id, row_key, source_blob_id, vector_id, effective_scalars, score FROM search_results
		WHERE snapshot_id = ? AND ordinal >= ? ORDER BY ordinal LIMIT ?`
	sourceBlobStatement = `SELECT content FROM source_blobs WHERE blob_id = ?`
)

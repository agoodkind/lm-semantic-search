package library

import (
	"context"
	"database/sql"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/mattn/go-sqlite3"
	"goodkind.io/lm-semantic-search/internal/clock"
)

//go:embed search_key_batch.sql
var searchKeyBatchStatement string

//go:embed search_key_next_batch.sql
var searchKeyNextBatchStatement string

//go:embed search_key_encoded.sql
var searchKeyEncodedStatement string

type candidateKeyReader struct {
	first, next *sql.Stmt
	after       [2]string
	started     bool
}

func (reader *candidateKeyReader) read(ctx context.Context, writer *sql.Tx, root int) ([][2]string, error) {
	statement, prepared := searchKeyBatchStatement, &reader.first
	if reader.started {
		statement, prepared = searchKeyNextBatchStatement, &reader.next
	}
	if *prepared == nil {
		encodedStatement := strings.ReplaceAll(searchKeyEncodedStatement, "{{keys}}", statement)
		value, err := writer.PrepareContext(ctx, encodedStatement)
		if err != nil {
			return nil, queryDatabaseError(ctx, "prepare filtered occurrence keys", err)
		}
		if reader.started {
			reader.next = value
		} else {
			reader.first = value
		}
	}
	var encoded string
	var err error
	if reader.started {
		err = (*prepared).QueryRowContext(ctx, sql.Named("node", root), sql.Named("batch_size", publicationInsertRows),
			sql.Named("after_owner", reader.after[0]), sql.Named("after_row", reader.after[1])).Scan(&encoded)
	} else {
		err = (*prepared).QueryRowContext(ctx, sql.Named("node", root), sql.Named("batch_size", publicationInsertRows)).Scan(&encoded)
	}
	var keys [][2]string
	if isSQLiteLengthLimit(err) {
		keys, err = reader.readRows(ctx, writer, root, statement)
	} else if err == nil {
		keys, err = decodeCandidateKeys(ctx, encoded)
	}
	if err != nil {
		return nil, queryDatabaseError(ctx, "read filtered occurrence keys", err)
	}
	if len(keys) > 0 {
		reader.after = keys[len(keys)-1]
		reader.started = true
	}
	return keys, nil
}

// JSON expansion can exceed SQLite's value limit even when each row fits.
func isSQLiteLengthLimit(err error) bool {
	var sqliteErr sqlite3.Error
	return errors.As(err, &sqliteErr) && sqliteErr.Code == sqlite3.ErrTooBig
}

func (reader *candidateKeyReader) readRows(ctx context.Context, writer *sql.Tx, root int, statement string) (keys [][2]string, err error) {
	var rows *sql.Rows
	if reader.started {
		rows, err = writer.QueryContext(ctx, statement, sql.Named("node", root), sql.Named("batch_size", publicationInsertRows),
			sql.Named("after_owner", reader.after[0]), sql.Named("after_row", reader.after[1]))
	} else {
		rows, err = writer.QueryContext(ctx, statement, sql.Named("node", root), sql.Named("batch_size", publicationInsertRows))
	}
	if err != nil {
		return nil, queryDatabaseError(ctx, "read filtered occurrence key rows", err)
	}
	defer func() { err = errors.Join(err, closeRows(ctx, rows)) }()
	for rows.Next() {
		var key [2]string
		if err := rows.Scan(&key[0], &key[1]); err != nil {
			return nil, queryDatabaseError(ctx, "scan filtered occurrence key", err)
		}
		keys = append(keys, key)
	}
	if err := rows.Err(); err != nil {
		return nil, queryDatabaseError(ctx, "read filtered occurrence key rows", err)
	}
	return keys, nil
}

func decodeCandidateKeys(ctx context.Context, encoded string) ([][2]string, error) {
	var rows [][]string
	if err := json.Unmarshal([]byte(encoded), &rows); err != nil {
		return nil, queryDatabaseError(ctx, "decode filtered occurrence keys", err)
	}
	keys := make([][2]string, 0, len(rows))
	for _, row := range rows {
		if len(row) != 2 {
			return nil, queryDatabaseError(ctx, "decode filtered occurrence keys", errors.New("filtered occurrence key does not have two fields"))
		}
		owner, err := hex.DecodeString(row[0])
		if err != nil {
			return nil, queryDatabaseError(ctx, "decode filtered occurrence owner", err)
		}
		keys = append(keys, [2]string{string(owner), row[1]})
	}
	return keys, nil
}

func (reader *candidateKeyReader) close(ctx context.Context) (err error) {
	for _, statement := range []*sql.Stmt{reader.first, reader.next} {
		if statement != nil {
			err = errors.Join(err, closeStatement(ctx, statement))
		}
	}
	return err
}

// Owners permit arbitrary bytes. Hex encoding avoids JSON's UTF-8 replacement.
// Scalars remain an encoded string rather than a reserialized JSON object.
type candidateBatchRow struct {
	OwnerHex       string  `json:"owner_hex"`
	RowKey         string  `json:"row_key"`
	SortKey        string  `json:"sort_key"`
	VectorID       string  `json:"vector_id"`
	IdentityDigest *string `json:"identity_digest"`
	VectorChecksum *string `json:"vector_checksum"`
	SourceBlobID   string  `json:"source_blob_id"`
	SearchHash     string  `json:"search_hash"`
	GroupKey       string  `json:"group_key"`
	Scalars        string  `json:"scalars"`
}

func insertSelectedCandidateBatch(ctx context.Context, writer *sql.Tx, encoded string, selected map[[2]string]bool, inserts *candidateInserts, phases *searchPhases) (int64, error) {
	var rows []candidateBatchRow
	if err := json.Unmarshal([]byte(encoded), &rows); err != nil {
		return 0, queryDatabaseError(ctx, "decode filtered occurrence batch", err)
	}
	for _, item := range rows {
		owner, err := hex.DecodeString(item.OwnerHex)
		if err != nil {
			return 0, queryDatabaseError(ctx, "decode filtered occurrence owner", err)
		}
		row := eligibleRow{
			ownerID: string(owner), rowKey: item.RowKey, sortKey: item.SortKey,
			vectorID: item.VectorID, blobID: item.SourceBlobID, searchHash: item.SearchHash,
			groupKey: item.GroupKey, scalars: item.Scalars,
		}
		if item.IdentityDigest != nil {
			row.digest = sql.NullString{String: *item.IdentityDigest, Valid: true}
		}
		if item.VectorChecksum != nil {
			row.checksum = sql.NullString{String: *item.VectorChecksum, Valid: true}
		}
		if err := appendEligibleCandidate(ctx, writer, row, selected, inserts, phases); err != nil {
			return 0, err
		}
	}
	if err := finishEligibleCandidates(ctx, writer, selected, inserts, phases); err != nil {
		return 0, err
	}
	return int64(len(rows)), nil
}

func appendEligibleCandidate(ctx context.Context, writer *sql.Tx, row eligibleRow, selected map[[2]string]bool, inserts *candidateInserts, phases *searchPhases) error {
	if selected != nil {
		key := [2]string{row.ownerID, row.rowKey}
		seen, found := selected[key]
		if !found || seen {
			return fmt.Errorf("filtered occurrence %s/%s has unexpected catalog rows in the read snapshot", row.ownerID, row.rowKey)
		}
		selected[key] = true
	}
	if !row.digest.Valid || !row.checksum.Valid {
		missing := fmt.Errorf("%w: occurrence %s/%s references vector %s without a catalog row",
			ErrVectorMissing, row.ownerID, row.rowKey, row.vectorID)
		slog.ErrorContext(ctx, "eligible occurrence has no catalog vector", "err", missing)
		return missing
	}
	started := clock.Now()
	defer func() { phases.candidateInsert += clock.Now().Sub(started) }()
	if err := inserts.candidates.append(ctx, writer,
		publicationString(row.ownerID), publicationString(row.rowKey), publicationString(row.sortKey), publicationString(row.vectorID),
		publicationString(row.blobID), publicationString(row.searchHash), publicationString(row.groupKey), publicationString(row.scalars)); err != nil {
		return queryDatabaseError(ctx, "copy eligible occurrences", err)
	}
	if err := inserts.vectors.append(ctx, writer, VectorIdentity{ID: row.vectorID, IdentityDigest: row.digest.String, Checksum: row.checksum.String}); err != nil {
		return queryDatabaseError(ctx, "copy eligible vector identities", err)
	}
	phases.acceptedCandidates++
	return nil
}

func finishEligibleCandidates(ctx context.Context, writer *sql.Tx, selected map[[2]string]bool, inserts *candidateInserts, phases *searchPhases) error {
	for key, seen := range selected {
		if !seen {
			return fmt.Errorf("filtered occurrence %s/%s has 0 catalog rows in the read snapshot", key[0], key[1])
		}
	}
	started := clock.Now()
	defer func() { phases.candidateInsert += clock.Now().Sub(started) }()
	if err := inserts.candidates.flush(ctx, writer); err != nil {
		return queryDatabaseError(ctx, "copy eligible occurrences", err)
	}
	if err := inserts.vectors.flush(ctx, writer); err != nil {
		return queryDatabaseError(ctx, "copy eligible vector identities", err)
	}
	return nil
}

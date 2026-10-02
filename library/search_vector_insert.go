package library

import (
	"context"
	"database/sql"
	_ "embed"
	"errors"
	"fmt"
	"log/slog"
	"strings"
)

//go:embed search_returning.sql
var searchReturningStatement string

type queryVectorInsert struct {
	rows     []VectorIdentity
	prepared *sql.Stmt
	scoring  *searchScoring
}

func (insert *queryVectorInsert) append(ctx context.Context, writer *sql.Tx, identity VectorIdentity) error {
	insert.rows = append(insert.rows, identity)
	if len(insert.rows) == publicationInsertRows {
		return insert.flush(ctx, writer)
	}
	return nil
}

func (insert *queryVectorInsert) flush(ctx context.Context, writer *sql.Tx) (err error) {
	defer func() {
		if err != nil {
			slog.ErrorContext(ctx, "insert candidate vectors failed", "err", err)
		}
	}()
	if len(insert.rows) == 0 {
		return nil
	}
	values := strings.TrimSuffix(strings.Repeat("(?,?,?),", len(insert.rows)), ",")
	statement := strings.ReplaceAll(searchVectorsStatement, "{{rows}}", values)
	statement = strings.ReplaceAll(searchReturningStatement, "{{insert}}", statement)
	if len(insert.rows) == publicationInsertRows && insert.prepared == nil {
		prepared, err := writer.PrepareContext(ctx, statement)
		if err != nil {
			return queryDatabaseError(ctx, "prepare candidate vectors", err)
		}
		insert.prepared = prepared
	}
	arguments := make([]any, 0, len(insert.rows)*3)
	expected := make(map[string]VectorIdentity, len(insert.rows))
	for _, identity := range insert.rows {
		arguments = append(arguments, identity.ID, identity.IdentityDigest, identity.Checksum)
		if previous, found := expected[identity.ID]; found && previous != identity {
			return fmt.Errorf("%w: candidate vector %s has conflicting identities", ErrVectorCorrupt, identity.ID)
		}
		expected[identity.ID] = identity
	}
	var rows *sql.Rows
	if len(insert.rows) == publicationInsertRows {
		rows, err = insert.prepared.QueryContext(ctx, arguments...)
	} else {
		rows, err = writer.QueryContext(ctx, statement, arguments...)
	}
	if err != nil {
		return queryDatabaseError(ctx, "copy candidate vectors", err)
	}
	identities, err := returnedVectorIdentities(ctx, rows, expected)
	if err != nil {
		return err
	}
	clear(insert.rows)
	insert.rows = insert.rows[:0]
	return insert.scoring.append(ctx, writer, identities)
}

func returnedVectorIdentities(ctx context.Context, rows *sql.Rows, expected map[string]VectorIdentity) (_ []VectorIdentity, err error) {
	defer func() {
		if err != nil {
			slog.ErrorContext(ctx, "read inserted vector identities failed", "err", err)
		}
	}()
	defer func() { err = errors.Join(err, closeRows(ctx, rows)) }()
	var identities []VectorIdentity
	for rows.Next() {
		var identity VectorIdentity
		if err := rows.Scan(&identity.ID, &identity.IdentityDigest, &identity.Checksum); err != nil {
			return nil, queryDatabaseError(ctx, "read candidate vector identity", err)
		}
		wanted, found := expected[identity.ID]
		if !found || wanted != identity {
			return nil, fmt.Errorf("%w: copied vector %s does not match its candidate identity", ErrVectorCorrupt, identity.ID)
		}
		delete(expected, identity.ID)
		identities = append(identities, identity)
	}
	if err := rows.Err(); err != nil {
		return nil, queryDatabaseError(ctx, "copy candidate vector identities", err)
	}
	return identities, nil
}

func (insert *queryVectorInsert) close(ctx context.Context) error {
	if insert.prepared == nil {
		return nil
	}
	return closeStatement(ctx, insert.prepared)
}

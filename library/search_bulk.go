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

//go:embed search_ranked.sql
var searchRankedStatement string

//go:embed search_results.sql
var searchResultsStatement string

type candidateLookup struct {
	prepared *sql.Stmt
}

func (lookup *candidateLookup) query(ctx context.Context, tx *sql.Tx, bindings []sql.NamedArg, placeholders []string) (*sql.Rows, error) {
	arguments := make([]any, len(bindings))
	for index, binding := range bindings {
		arguments[index] = binding
	}
	full := len(placeholders) == publicationInsertRows
	statement := ""
	if !full || lookup.prepared == nil {
		statement = strings.NewReplacer("{{columns}}", candidateColumns, "{{keys}}", strings.Join(placeholders, ",")).Replace(searchSelectedStatement)
	}
	if full && lookup.prepared == nil {
		prepared, err := tx.PrepareContext(ctx, statement)
		if err != nil {
			slog.ErrorContext(ctx, "prepare filtered occurrence batch failed", "err", err)
			return nil, fmt.Errorf("prepare filtered occurrence batch: %w", err)
		}
		lookup.prepared = prepared
	}
	var rows *sql.Rows
	var err error
	if full {
		rows, err = lookup.prepared.QueryContext(ctx, arguments...)
	} else {
		rows, err = tx.QueryContext(ctx, statement, arguments...)
	}
	if err != nil {
		slog.ErrorContext(ctx, "read filtered occurrence failed", "err", err)
		return nil, fmt.Errorf("read filtered occurrence batch: %w", err)
	}
	return rows, nil
}

func (lookup *candidateLookup) close(ctx context.Context) error {
	if lookup.prepared == nil {
		return nil
	}
	return closeStatement(ctx, lookup.prepared)
}

type candidateInserts struct {
	candidates publicationInsert
	vectors    publicationInsert
}

func (inserts *candidateInserts) close(ctx context.Context) (err error) {
	for _, insert := range []*publicationInsert{&inserts.candidates, &inserts.vectors} {
		if insert.prepared != nil {
			err = errors.Join(err, closeStatement(ctx, insert.prepared))
		}
	}
	return err
}

// rankedInsert buffers at most 64 rows. Scores retain their float64 SQL binding.
type rankedInsert struct {
	rows       []rankedRow
	prepared   *sql.Stmt
	snapshotID string
	namespace  string
}

func (insert *rankedInsert) append(ctx context.Context, tx *sql.Tx, row rankedRow) error {
	if insert.rows == nil {
		insert.rows = make([]rankedRow, 0, publicationInsertRows)
	}
	insert.rows = append(insert.rows, row)
	if len(insert.rows) == publicationInsertRows {
		return insert.flush(ctx, tx)
	}
	return nil
}

func (insert *rankedInsert) flush(ctx context.Context, tx *sql.Tx) error {
	if len(insert.rows) == 0 {
		return nil
	}
	columns := 7
	statement := searchRankedStatement
	if insert.snapshotID != "" {
		columns = 9
		statement = searchResultsStatement
	}
	group := "(" + strings.TrimSuffix(strings.Repeat("?,", columns), ",") + ")"
	values := strings.TrimSuffix(strings.Repeat(group+",", len(insert.rows)), ",")
	statement = strings.ReplaceAll(statement, "{{rows}}", values)
	if len(insert.rows) == publicationInsertRows && insert.prepared == nil {
		prepared, err := tx.PrepareContext(ctx, statement)
		if err != nil {
			return fmt.Errorf("prepare ranked rows: %w", err)
		}
		insert.prepared = prepared
	}
	arguments := make([]any, 0, len(insert.rows)*columns)
	for _, row := range insert.rows {
		if insert.snapshotID != "" {
			arguments = append(arguments, insert.snapshotID, row.ordinal, insert.namespace,
				row.ownerID, row.rowKey, row.sourceBlobID, row.vectorID, row.scalars, row.score)
		} else {
			arguments = append(arguments, row.ordinal, row.ownerID, row.rowKey,
				row.sourceBlobID, row.vectorID, row.scalars, row.score)
		}
	}
	var err error
	if len(insert.rows) == publicationInsertRows {
		_, err = insert.prepared.ExecContext(ctx, arguments...)
	} else {
		_, err = tx.ExecContext(ctx, statement, arguments...)
	}
	if err != nil {
		slog.ErrorContext(ctx, "insert ranked rows failed", "err", err)
		return fmt.Errorf("insert ranked rows: %w", err)
	}
	clear(insert.rows)
	insert.rows = insert.rows[:0]
	return nil
}

func (insert *rankedInsert) close(ctx context.Context) error {
	if insert.prepared == nil {
		return nil
	}
	return closeStatement(ctx, insert.prepared)
}

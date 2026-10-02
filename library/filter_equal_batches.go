package library

import (
	"context"
	"database/sql"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"

	"goodkind.io/lm-semantic-search/internal/clock"
)

//go:embed filter_equal_effective.sql
var filterEqualEffectiveStatement string

//go:embed filter_equal_published.sql
var filterEqualPublishedStatement string

//go:embed filter_equal_cursor.sql
var filterEqualCursorStatement string

//go:embed filter_equal_encoded.sql
var filterEqualEncodedStatement string

type equalFilterRow struct {
	RowID    int64  `json:"row_id"`
	OwnerHex string `json:"owner_hex"`
	RowKey   string `json:"row_key"`
}

type equalFilterReader struct {
	first, next *sql.Stmt
	after       int64
	started     bool
}

func (evaluator *filterEvaluator) equalLeaf(ctx context.Context, filter Filter, target int) (err error) {
	started := clock.Now()
	var scanned, accepted int64
	defer func() {
		evaluator.phases.recordFilterNode(filterNodeTiming{
			node: target, operation: "leaf", column: filter.Column,
			scanned: scanned, accepted: accepted, elapsed: clock.Now().Sub(started), complete: err == nil,
		})
	}()
	insert := publicationInsert{statement: searchFilterRowsStatement, columns: 3}
	defer func() {
		if insert.prepared != nil {
			err = errors.Join(err, closeStatement(ctx, insert.prepared))
		}
	}()
	predicate, err := typedStatement(evaluator.columns[filter.Column].Type, false,
		[2]string{stringEqual, stringNotEqual}, [2]string{int64Equal, int64NotEqual}, [2]string{boolEqual, boolNotEqual})
	if err != nil {
		return err
	}
	for _, source := range []string{filterEqualEffectiveStatement, filterEqualPublishedStatement} {
		reader := &equalFilterReader{}
		defer func() { err = errors.Join(err, reader.close(ctx)) }()
		for {
			cursor := ""
			arguments := []sql.NamedArg{
				sql.Named("namespace", evaluator.namespace), sql.Named("column", filter.Column),
				scalarParameter("value", filter.Values[0]), sql.Named("batch_size", publicationInsertRows),
			}
			if reader.started {
				cursor = filterEqualCursorStatement
				arguments = append(arguments, sql.Named("after_rowid", reader.after))
			}
			statement := strings.NewReplacer("{{predicate}}", predicate, "{{cursor}}", cursor).Replace(source)
			rows, err := reader.read(ctx, evaluator, statement, arguments)
			if err != nil {
				return err
			}
			if len(rows) == 0 {
				break
			}
			read, saved, err := evaluator.saveEqualBatch(ctx, target, reader, rows, &insert)
			scanned += read
			accepted += saved
			if err != nil {
				return err
			}
		}
	}
	if err := insert.flush(ctx, evaluator.writer); err != nil {
		return queryDatabaseError(ctx, "save equal filter rows", err)
	}
	return nil
}

func (evaluator *filterEvaluator) saveEqualBatch(ctx context.Context, target int, reader *equalFilterReader, rows []equalFilterRow, insert *publicationInsert) (scanned, accepted int64, err error) {
	for _, row := range rows {
		if reader.started && row.RowID <= reader.after {
			return scanned, accepted, queryDatabaseError(ctx, "read equal filter batch", errors.New("equal filter row cursor did not increase"))
		}
		owner, err := hex.DecodeString(row.OwnerHex)
		if err != nil {
			return scanned, accepted, queryDatabaseError(ctx, "decode equal filter owner", err)
		}
		scanned++
		if err := insert.append(ctx, evaluator.writer, publicationInteger(int64(target)), publicationString(string(owner)), publicationString(row.RowKey)); err != nil {
			return scanned, accepted, queryDatabaseError(ctx, "save equal filter row", err)
		}
		accepted++
		reader.after, reader.started = row.RowID, true
	}
	return scanned, accepted, nil
}

func (reader *equalFilterReader) read(ctx context.Context, evaluator *filterEvaluator, statement string, arguments []sql.NamedArg) ([]equalFilterRow, error) {
	prepared := &reader.first
	if reader.started {
		prepared = &reader.next
	}
	if *prepared == nil {
		encodedStatement := strings.ReplaceAll(filterEqualEncodedStatement, "{{rows}}", statement)
		value, err := evaluator.catalog.PrepareContext(ctx, encodedStatement)
		if err != nil {
			return nil, queryDatabaseError(ctx, "prepare equal filter batch", err)
		}
		if reader.started {
			reader.next = value
		} else {
			reader.first = value
		}
	}
	values := make([]any, len(arguments))
	for index, argument := range arguments {
		values[index] = argument
	}
	var encoded string
	err := (*prepared).QueryRowContext(ctx, values...).Scan(&encoded)
	if isSQLiteLengthLimit(err) {
		return evaluator.readEqualRows(ctx, statement, arguments)
	}
	if err != nil {
		return nil, queryDatabaseError(ctx, "read equal filter batch", err)
	}
	var rows []equalFilterRow
	if err := json.Unmarshal([]byte(encoded), &rows); err != nil {
		return nil, queryDatabaseError(ctx, "decode equal filter batch", err)
	}
	return rows, nil
}

func (reader *equalFilterReader) close(ctx context.Context) (err error) {
	for _, statement := range []*sql.Stmt{reader.first, reader.next} {
		if statement != nil {
			err = errors.Join(err, closeStatement(ctx, statement))
		}
	}
	return err
}

func (evaluator *filterEvaluator) readEqualRows(ctx context.Context, statement string, arguments []sql.NamedArg) (batch []equalFilterRow, err error) {
	values := make([]any, len(arguments))
	for index, argument := range arguments {
		values[index] = argument
	}
	rows, err := evaluator.catalog.QueryContext(ctx, statement, values...)
	if err != nil {
		return nil, queryDatabaseError(ctx, "read equal filter rows", err)
	}
	defer func() { err = errors.Join(err, closeRows(ctx, rows)) }()
	for rows.Next() {
		var row equalFilterRow
		var owner string
		if err := rows.Scan(&row.RowID, &owner, &row.RowKey); err != nil {
			return nil, queryDatabaseError(ctx, "scan equal filter row", err)
		}
		row.OwnerHex = hex.EncodeToString([]byte(owner))
		batch = append(batch, row)
	}
	if err := rows.Err(); err != nil {
		return nil, queryDatabaseError(ctx, "read equal filter rows", err)
	}
	return batch, nil
}

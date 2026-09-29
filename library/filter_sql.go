package library

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"unicode/utf8"
)

// The filter statements store the truth value of one node for one occurrence
// as 2 for true and 1 for unknown. The query database stores only true and
// unknown rows; an occurrence without a row is false.

// effectiveColumnValues selects, for every occurrence of :namespace, whether
// the column bound to :column is present, whether it is null, and its typed
// effective value. A ReprojectScalars row in effective_scalars replaces the
// published row. Exactly one typed field of a non-null row is set, and COALESCE
// returns it. The value is SQL NULL for an explicit null and for an absent
// column.
const effectiveColumnValues = `WITH effective AS (
	SELECT o.owner_id, o.row_key,
		(e.column_name IS NOT NULL OR s.column_name IS NOT NULL) AS present,
		CASE WHEN e.column_name IS NOT NULL THEN e.is_null
			WHEN s.column_name IS NOT NULL THEN s.is_null ELSE 0 END AS null_value,
		CASE WHEN e.column_name IS NOT NULL THEN COALESCE(e.string_value, e.int64_value, e.bool_value)
			ELSE COALESCE(s.string_value, s.int64_value, s.bool_value) END AS value
	FROM occurrences o
	LEFT JOIN effective_scalars e ON e.namespace = o.namespace AND e.owner_id = o.owner_id
		AND e.row_key = o.row_key AND e.column_name = :column
	LEFT JOIN occurrence_scalars s ON s.namespace = o.namespace AND s.owner_id = o.owner_id
		AND s.row_key = o.row_key AND s.column_name = :column
	WHERE o.namespace = :namespace
)
SELECT owner_id, row_key, `

// comparisonTruth returns unknown for a null or absent value and true for a
// value that satisfies the comparison. A false comparison returns no row.
const comparisonTruth = `CASE WHEN value IS NULL THEN 1 ELSE 2 END FROM effective WHERE value IS NULL OR `

// Leaf statements. Each one reads the catalog and returns owner_id, row_key,
// and the truth value of one leaf for every occurrence that is not false.
// Every compared value is a bound parameter.
const (
	equalLeafStatement     = effectiveColumnValues + comparisonTruth + `value = :value`
	inLeafStatement        = effectiveColumnValues + comparisonTruth + `value IN (SELECT j.value FROM json_each(:values) AS j)`
	rangeLeafStatement     = effectiveColumnValues + comparisonTruth + `((:lower IS NULL OR value >= :lower) AND (:upper IS NULL OR value < :upper))`
	prefixLeafStatement    = effectiveColumnValues + comparisonTruth + `substr(value, 1, :length) = :prefix`
	isNullLeafStatement    = effectiveColumnValues + `2 FROM effective WHERE present AND null_value = 1`
	isPresentLeafStatement = effectiveColumnValues + `2 FROM effective WHERE present`
)

// Combination statements. Each one writes the rows of one node from the rows
// of its children in the query database with SQL three-valued logic: All
// keeps the minimum truth of an occurrence that every child keeps, Any keeps
// the maximum truth, and Not maps false to true and keeps unknown.
const (
	allCombineStatement = `INSERT INTO filter_sets (node, owner_id, row_key, truth)
		SELECT :node, owner_id, row_key, MIN(truth) FROM filter_sets
		WHERE node IN (SELECT j.value FROM json_each(:children) AS j)
		GROUP BY owner_id, row_key HAVING COUNT(*) = :count`
	anyCombineStatement = `INSERT INTO filter_sets (node, owner_id, row_key, truth)
		SELECT :node, owner_id, row_key, MAX(truth) FROM filter_sets
		WHERE node IN (SELECT j.value FROM json_each(:children) AS j)
		GROUP BY owner_id, row_key`
	notCombineStatement = `INSERT INTO filter_sets (node, owner_id, row_key, truth)
		SELECT :node, u.owner_id, u.row_key, 2 - COALESCE(f.truth, 0) FROM universe u
		LEFT JOIN filter_sets f ON f.node = :child AND f.owner_id = u.owner_id AND f.row_key = u.row_key
		WHERE COALESCE(f.truth, 0) < 2`
	insertFilterRowStatement = `INSERT INTO filter_sets (node, owner_id, row_key, truth) VALUES (:node, :owner_id, :row_key, :truth)`
	universeStatement        = `SELECT owner_id, row_key FROM occurrences WHERE namespace = :namespace`
	insertUniverseStatement  = `INSERT INTO universe (owner_id, row_key) VALUES (:owner_id, :row_key)`
)

// filterEvaluator evaluates one filter tree that [Config.ValidateSearchRequest]
// accepted. It reads leaf truth values from the catalog read transaction and
// combines nodes in the query database writer transaction.
type filterEvaluator struct {
	catalog   *sql.Tx
	writer    *sql.Tx
	namespace string
	nextNode  int
	universe  bool
}

// evaluate writes the true and unknown rows of filter as one node and returns
// the node number.
func (evaluator *filterEvaluator) evaluate(ctx context.Context, filter Filter) (int, error) {
	switch filter.Op {
	case All, Any:
		children := make([]int, 0, len(filter.Children))
		for _, child := range filter.Children {
			node, err := evaluator.evaluate(ctx, child)
			if err != nil {
				return 0, err
			}
			children = append(children, node)
		}
		return evaluator.combine(ctx, filter.Op, children)
	case Not:
		child, err := evaluator.evaluate(ctx, filter.Children[0])
		if err != nil {
			return 0, err
		}
		if err := evaluator.copyUniverse(ctx); err != nil {
			return 0, err
		}
		node := evaluator.newNode()
		_, err = evaluator.writer.ExecContext(ctx, notCombineStatement, sql.Named("node", node), sql.Named("child", child))
		if err != nil {
			return 0, queryDatabaseError(ctx, "evaluate Not filter", err)
		}
		return node, nil
	case Equal, In, Range, Prefix, IsNull, IsPresent:
		return evaluator.leaf(ctx, filter)
	default:
		return 0, invalidRequest(fmt.Sprintf("filter operator %d is not a FilterOp", filter.Op))
	}
}

func (evaluator *filterEvaluator) newNode() int {
	node := evaluator.nextNode
	evaluator.nextNode++
	return node
}

func (evaluator *filterEvaluator) combine(ctx context.Context, op FilterOp, children []int) (int, error) {
	encoded, err := json.Marshal(children)
	if err != nil {
		slog.ErrorContext(ctx, "encode filter children failed", "err", err)
		return 0, fmt.Errorf("encode filter children: %w", err)
	}
	node := evaluator.newNode()
	if op == All {
		_, err = evaluator.writer.ExecContext(ctx, allCombineStatement,
			sql.Named("node", node), sql.Named("children", string(encoded)), sql.Named("count", len(children)))
	} else {
		_, err = evaluator.writer.ExecContext(ctx, anyCombineStatement,
			sql.Named("node", node), sql.Named("children", string(encoded)))
	}
	if err != nil {
		return 0, queryDatabaseError(ctx, "combine filter children", err)
	}
	return node, nil
}

// copyUniverse copies every occurrence key of the namespace once. Not uses it
// as the complement of its child.
func (evaluator *filterEvaluator) copyUniverse(ctx context.Context) (err error) {
	if evaluator.universe {
		return nil
	}
	rows, err := evaluator.catalog.QueryContext(ctx, universeStatement, sql.Named("namespace", evaluator.namespace))
	if err != nil {
		slog.ErrorContext(ctx, "read namespace occurrences failed", "err", err)
		return fmt.Errorf("read occurrences of %s: %w", evaluator.namespace, err)
	}
	defer func() {
		err = errors.Join(err, closeRows(ctx, rows))
	}()
	for rows.Next() {
		var ownerID, rowKey string
		if err := rows.Scan(&ownerID, &rowKey); err != nil {
			slog.ErrorContext(ctx, "scan namespace occurrence failed", "err", err)
			return fmt.Errorf("scan occurrence of %s: %w", evaluator.namespace, err)
		}
		if _, err := evaluator.writer.ExecContext(ctx, insertUniverseStatement,
			sql.Named("owner_id", ownerID), sql.Named("row_key", rowKey)); err != nil {
			return queryDatabaseError(ctx, "copy namespace occurrence", err)
		}
	}
	if err := rows.Err(); err != nil {
		slog.ErrorContext(ctx, "read namespace occurrences failed", "err", err)
		return fmt.Errorf("read occurrences of %s: %w", evaluator.namespace, err)
	}
	evaluator.universe = true
	return nil
}

// leaf runs the leaf statement of filter against the catalog and writes its
// true and unknown rows as one node.
func (evaluator *filterEvaluator) leaf(ctx context.Context, filter Filter) (_ int, err error) {
	rows, err := evaluator.queryLeaf(ctx, filter)
	if err != nil {
		return 0, err
	}
	defer func() {
		err = errors.Join(err, closeRows(ctx, rows))
	}()
	node := evaluator.newNode()
	for rows.Next() {
		var ownerID, rowKey string
		var truth int
		if err := rows.Scan(&ownerID, &rowKey, &truth); err != nil {
			slog.ErrorContext(ctx, "scan filter leaf row failed", "err", err)
			return 0, fmt.Errorf("scan filter row: %w", err)
		}
		if _, err := evaluator.writer.ExecContext(ctx, insertFilterRowStatement,
			sql.Named("node", node), sql.Named("owner_id", ownerID), sql.Named("row_key", rowKey), sql.Named("truth", truth)); err != nil {
			return 0, queryDatabaseError(ctx, "save filter row", err)
		}
	}
	if err := rows.Err(); err != nil {
		slog.ErrorContext(ctx, "evaluate filter leaf failed", "err", err)
		return 0, fmt.Errorf("evaluate filter on column %s: %w", filter.Column, err)
	}
	return node, nil
}

// queryLeaf runs the constant leaf statement of filter with the namespace,
// the column, and the operator values as named parameters.
func (evaluator *filterEvaluator) queryLeaf(ctx context.Context, filter Filter) (*sql.Rows, error) {
	namespace := sql.Named("namespace", evaluator.namespace)
	column := sql.Named("column", filter.Column)
	var rows *sql.Rows
	var err error
	switch filter.Op {
	case Equal:
		rows, err = evaluator.catalog.QueryContext(ctx, equalLeafStatement, namespace, column, scalarParameter("value", filter.Values[0]))
	case In:
		encoded, encodeErr := encodeScalarList(filter.Values)
		if encodeErr != nil {
			return nil, encodeErr
		}
		rows, err = evaluator.catalog.QueryContext(ctx, inLeafStatement, namespace, column, sql.Named("values", encoded))
	case Range:
		rows, err = evaluator.catalog.QueryContext(ctx, rangeLeafStatement, namespace, column,
			boundParameter("lower", filter.Lower), boundParameter("upper", filter.Upper))
	case Prefix:
		rows, err = evaluator.catalog.QueryContext(ctx, prefixLeafStatement, namespace, column,
			sql.Named("length", utf8.RuneCountInString(filter.Prefix)), sql.Named("prefix", filter.Prefix))
	case IsNull:
		rows, err = evaluator.catalog.QueryContext(ctx, isNullLeafStatement, namespace, column)
	case IsPresent:
		rows, err = evaluator.catalog.QueryContext(ctx, isPresentLeafStatement, namespace, column)
	case All, Any, Not:
		return nil, invalidRequest(fmt.Sprintf("filter operator %d is not a leaf", filter.Op))
	default:
		return nil, invalidRequest(fmt.Sprintf("filter operator %d is not a FilterOp", filter.Op))
	}
	if err != nil {
		slog.ErrorContext(ctx, "evaluate filter leaf failed", "column", filter.Column, "err", err)
		return nil, fmt.Errorf("evaluate filter on column %s: %w", filter.Column, err)
	}
	return rows, nil
}

// scalarParameter binds a non-null comparison value in the storage form of
// its typed column.
func scalarParameter(name string, value ScalarValue) sql.NamedArg {
	switch value.Type {
	case Bool:
		return sql.Named(name, value.Bool)
	case Int64:
		return sql.Named(name, value.Int64)
	case String:
		return sql.Named(name, value.String)
	default:
		return sql.Named(name, value.String)
	}
}

// boundParameter binds an optional int64 range bound. A nil bound binds SQL
// NULL, which the range statement reads as unbounded.
func boundParameter(name string, bound *ScalarValue) sql.NamedArg {
	if bound == nil {
		return sql.Named(name, sql.NullInt64{Int64: 0, Valid: false})
	}
	return sql.Named(name, sql.NullInt64{Int64: bound.Int64, Valid: true})
}

// jsonScalar encodes a non-null comparison value as its JSON string, number,
// or boolean. SQLite json_each returns each one in the storage form of the
// typed column: text, integer, or 1 and 0.
type jsonScalar ScalarValue

// MarshalJSON encodes the typed field of the value.
func (value jsonScalar) MarshalJSON() ([]byte, error) {
	var encoded []byte
	var err error
	switch value.Type {
	case Bool:
		encoded, err = json.Marshal(value.Bool)
	case Int64:
		encoded, err = json.Marshal(value.Int64)
	case String:
		encoded, err = json.Marshal(value.String)
	default:
		encoded, err = json.Marshal(value.String)
	}
	if err != nil {
		slog.Error("encode filter value failed", "err", err)
		return nil, fmt.Errorf("encode filter value: %w", err)
	}
	return encoded, nil
}

func encodeScalarList(values []ScalarValue) (string, error) {
	encodable := make([]jsonScalar, 0, len(values))
	for _, value := range values {
		encodable = append(encodable, jsonScalar(value))
	}
	encoded, err := json.Marshal(encodable)
	if err != nil {
		slog.Error("encode filter values failed", "err", err)
		return "", fmt.Errorf("encode filter values: %w", err)
	}
	return string(encoded), nil
}

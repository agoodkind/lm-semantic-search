package library

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log/slog"
	"math"

	"goodkind.io/lm-semantic-search/internal/clock"
)

// A filter node evaluates to true, false, or unknown for each occurrence, with
// SQL three-valued logic. The evaluator stores one set per node in the query
// database: the true set, or the false set of a node below an odd number of
// Not nodes. An occurrence in neither set is unknown for that node. Not swaps
// the true and false sets of its child. All keeps the occurrences in every
// child's true set and in any child's false set. Any keeps the occurrences in
// any child's true set and in every child's false set.

// Leaf statements read the typed scalar indexes (namespace, column_name,
// typed value) of effective_scalars and occurrence_scalars. The first part
// reads projected values. The second part reads published values of rows
// without a projected value for the column. Each part applies the same
// predicate to the typed value, which is SQL NULL for an explicit null; a
// comparison with a null value selects no row in either set. Every compared
// value is a bound parameter.
const (
	effectiveLeafHead = `SELECT v.owner_id, v.row_key FROM effective_scalars v
	WHERE v.namespace = :namespace AND v.column_name = :column AND `
	publishedLeafHead = `
UNION ALL SELECT v.owner_id, v.row_key FROM occurrence_scalars v
	WHERE v.namespace = :namespace AND v.column_name = :column
	AND NOT EXISTS (SELECT 1 FROM effective_scalars e WHERE e.namespace = v.namespace
		AND e.owner_id = v.owner_id AND e.row_key = v.row_key AND e.column_name = v.column_name)
	AND `
)

// Leaf predicates over the alias v. Each pair selects the true set and the
// false set of one operator on one typed value.
const (
	stringEqual         = `v.string_value = :value`
	stringNotEqual      = `v.string_value <> :value`
	int64Equal          = `v.int64_value = :value`
	int64NotEqual       = `v.int64_value <> :value`
	boolEqual           = `v.bool_value = :value`
	boolNotEqual        = `v.bool_value <> :value`
	int64InRange        = `v.int64_value BETWEEN :lower AND :upper`
	int64OutsideRange   = `(v.int64_value < :lower OR v.int64_value > :upper)`
	stringInPrefix      = `v.string_value >= :prefix AND v.string_value < :successor`
	stringOutsidePrefix = `(v.string_value < :prefix OR v.string_value >= :successor)`
	stringFromPrefix    = `v.string_value >= :prefix`
	stringBeforePrefix  = `v.string_value < :prefix`
	nullValue           = `v.is_null = 1`
	nonNullValue        = `v.is_null = 0`
	anyValue            = `v.is_null IN (0, 1)`
)

// Leaf statements, one per operator, typed value, and set.
const (
	stringEqualLeaf         = effectiveLeafHead + stringEqual + publishedLeafHead + stringEqual
	stringNotEqualLeaf      = effectiveLeafHead + stringNotEqual + publishedLeafHead + stringNotEqual
	int64EqualLeaf          = effectiveLeafHead + int64Equal + publishedLeafHead + int64Equal
	int64NotEqualLeaf       = effectiveLeafHead + int64NotEqual + publishedLeafHead + int64NotEqual
	boolEqualLeaf           = effectiveLeafHead + boolEqual + publishedLeafHead + boolEqual
	boolNotEqualLeaf        = effectiveLeafHead + boolNotEqual + publishedLeafHead + boolNotEqual
	int64InRangeLeaf        = effectiveLeafHead + int64InRange + publishedLeafHead + int64InRange
	int64OutsideRangeLeaf   = effectiveLeafHead + int64OutsideRange + publishedLeafHead + int64OutsideRange
	stringInPrefixLeaf      = effectiveLeafHead + stringInPrefix + publishedLeafHead + stringInPrefix
	stringOutsidePrefixLeaf = effectiveLeafHead + stringOutsidePrefix + publishedLeafHead + stringOutsidePrefix
	stringFromPrefixLeaf    = effectiveLeafHead + stringFromPrefix + publishedLeafHead + stringFromPrefix
	stringBeforePrefixLeaf  = effectiveLeafHead + stringBeforePrefix + publishedLeafHead + stringBeforePrefix
	nullValueLeaf           = effectiveLeafHead + nullValue + publishedLeafHead + nullValue
	nonNullValueLeaf        = effectiveLeafHead + nonNullValue + publishedLeafHead + nonNullValue
	anyValueLeaf            = effectiveLeafHead + anyValue + publishedLeafHead + anyValue
)

// absentLeaf selects the occurrences of :namespace with no published and no
// projected row for :column. It reads every occurrence of the namespace. Only
// the false sets of IsNull and IsPresent use it.
const absentLeaf = `SELECT o.owner_id, o.row_key FROM occurrences o WHERE o.namespace = :namespace
	AND NOT EXISTS (SELECT 1 FROM effective_scalars e WHERE e.namespace = o.namespace
		AND e.owner_id = o.owner_id AND e.row_key = o.row_key AND e.column_name = :column)
	AND NOT EXISTS (SELECT 1 FROM occurrence_scalars s WHERE s.namespace = o.namespace
		AND s.owner_id = o.owner_id AND s.row_key = o.row_key AND s.column_name = :column)`

// Query database statements of the filter sets. The combinations read one set
// and write another through Go. A single INSERT that selects from filter_sets
// would copy its input into a temporary table outside the query database.
const (
	filterNodeRowsStatement = `SELECT owner_id, row_key FROM filter_sets WHERE node = :node`
	filterRowStatement      = `SELECT 1 FROM filter_sets WHERE node = :node AND owner_id = :owner_id AND row_key = :row_key`
	filterBothRowsStatement = `SELECT f.owner_id, f.row_key FROM filter_sets f WHERE f.node = :left
		AND EXISTS (SELECT 1 FROM filter_sets g WHERE g.node = :right AND g.owner_id = f.owner_id AND g.row_key = f.row_key)`
)

// filterEvaluator evaluates one filter tree that [Config.ValidateSearchRequest]
// accepted. It reads leaf sets from the catalog read transaction and writes
// every node set in the query database writer transaction.
type filterEvaluator struct {
	catalog   *sql.Tx
	writer    *sql.Tx
	namespace string
	columns   map[string]ScalarColumn
	nextNode  int
	phases    *searchPhases
}

// evaluate writes the true set of filter, or its false set when negated, as
// one node and returns the node number.
func (evaluator *filterEvaluator) evaluate(ctx context.Context, filter Filter, negated bool) (int, error) {
	switch filter.Op {
	case Not:
		return evaluator.evaluate(ctx, filter.Children[0], !negated)
	case All, Any:
		children := make([]int, 0, len(filter.Children))
		for _, child := range filter.Children {
			node, err := evaluator.evaluate(ctx, child, negated)
			if err != nil {
				return 0, err
			}
			children = append(children, node)
		}
		if (filter.Op == All) != negated {
			return evaluator.intersect(ctx, children)
		}
		return evaluator.union(ctx, children)
	case In:
		return evaluator.inLeaf(ctx, filter, negated)
	case Equal, Range, Prefix, IsNull, IsPresent:
		return evaluator.leaf(ctx, filter, negated)
	default:
		return 0, invalidRequest(fmt.Sprintf("filter operator %d is not a FilterOp", filter.Op))
	}
}

func (evaluator *filterEvaluator) newNode() int {
	node := evaluator.nextNode
	evaluator.nextNode++
	return node
}

// intersect writes the occurrences in every child set as one node.
func (evaluator *filterEvaluator) intersect(ctx context.Context, children []int) (int, error) {
	node := children[0]
	for _, right := range children[1:] {
		target := evaluator.newNode()
		rows, err := evaluator.writer.QueryContext(ctx, filterBothRowsStatement, sql.Named("left", node), sql.Named("right", right))
		if err != nil {
			return 0, queryDatabaseError(ctx, "read filter intersection", err)
		}
		if err := evaluator.saveRows(ctx, rows, target, noExcludedNode, "intersection", ""); err != nil {
			return 0, err
		}
		node = target
	}
	return node, nil
}

// union writes the occurrences in any child set as one node.
func (evaluator *filterEvaluator) union(ctx context.Context, children []int) (int, error) {
	if len(children) == 1 {
		return children[0], nil
	}
	target := evaluator.newNode()
	for _, child := range children {
		rows, err := evaluator.writer.QueryContext(ctx, filterNodeRowsStatement, sql.Named("node", child))
		if err != nil {
			return 0, queryDatabaseError(ctx, "read filter union", err)
		}
		if err := evaluator.saveRows(ctx, rows, target, noExcludedNode, "union", ""); err != nil {
			return 0, err
		}
	}
	return target, nil
}

// leaf writes the true set of a leaf, or its false set when negated, as one
// node. The false sets of IsNull and IsPresent include the occurrences with
// an absent column.
func (evaluator *filterEvaluator) leaf(ctx context.Context, filter Filter, negated bool) (int, error) {
	target := evaluator.newNode()
	statement, err := leafStatement(filter.Op, evaluator.columns[filter.Column].Type, filter.Prefix, negated)
	if err != nil {
		return 0, err
	}
	if statement != "" {
		rows, err := evaluator.queryLeaf(ctx, statement, filter)
		if err != nil {
			return 0, err
		}
		if err := evaluator.saveRows(ctx, rows, target, noExcludedNode, "leaf", filter.Column); err != nil {
			return 0, err
		}
	}
	if negated && (filter.Op == IsNull || filter.Op == IsPresent) {
		rows, err := evaluator.catalog.QueryContext(ctx, absentLeaf,
			sql.Named("namespace", evaluator.namespace), sql.Named("column", filter.Column))
		if err != nil {
			slog.ErrorContext(ctx, "read absent filter column failed", "column", filter.Column, "err", err)
			return 0, fmt.Errorf("read occurrences without column %s: %w", filter.Column, err)
		}
		if err := evaluator.saveRows(ctx, rows, target, noExcludedNode, "leaf", filter.Column); err != nil {
			return 0, err
		}
	}
	return target, nil
}

// inLeaf writes the true set of an In leaf, or its false set when negated,
// as one node. The true set runs the Equal statement once per listed value.
// The false set is the non-null set of the column without the true set.
func (evaluator *filterEvaluator) inLeaf(ctx context.Context, filter Filter, negated bool) (int, error) {
	scalarType := evaluator.columns[filter.Column].Type
	equal, err := leafStatement(Equal, scalarType, "", false)
	if err != nil {
		return 0, err
	}
	matched := evaluator.newNode()
	for _, value := range filter.Values {
		rows, err := evaluator.catalog.QueryContext(ctx, equal,
			sql.Named("namespace", evaluator.namespace), sql.Named("column", filter.Column), scalarParameter("value", value))
		if err != nil {
			slog.ErrorContext(ctx, "evaluate In filter value failed", "column", filter.Column, "err", err)
			return 0, fmt.Errorf("evaluate In filter on column %s: %w", filter.Column, err)
		}
		if err := evaluator.saveRows(ctx, rows, matched, noExcludedNode, "in_match", filter.Column); err != nil {
			return 0, err
		}
	}
	if !negated {
		return matched, nil
	}
	unmatched := evaluator.newNode()
	rows, err := evaluator.catalog.QueryContext(ctx, nonNullValueLeaf,
		sql.Named("namespace", evaluator.namespace), sql.Named("column", filter.Column))
	if err != nil {
		slog.ErrorContext(ctx, "read non-null filter column failed", "column", filter.Column, "err", err)
		return 0, fmt.Errorf("read non-null values of column %s: %w", filter.Column, err)
	}
	if err := evaluator.saveRows(ctx, rows, unmatched, matched, "in_exclude", filter.Column); err != nil {
		return 0, err
	}
	return unmatched, nil
}

// leafStatement returns the statement of one leaf set. The false set of
// IsPresent has no statement; it is the absent set alone.
func leafStatement(op FilterOp, scalarType ScalarType, prefix string, negated bool) (string, error) {
	switch op {
	case Equal:
		return typedStatement(scalarType, negated,
			[2]string{stringEqualLeaf, stringNotEqualLeaf},
			[2]string{int64EqualLeaf, int64NotEqualLeaf},
			[2]string{boolEqualLeaf, boolNotEqualLeaf})
	case Range:
		return chooseSet(negated, int64InRangeLeaf, int64OutsideRangeLeaf), nil
	case Prefix:
		if _, bounded := prefixSuccessor(prefix); !bounded {
			return chooseSet(negated, stringFromPrefixLeaf, stringBeforePrefixLeaf), nil
		}
		return chooseSet(negated, stringInPrefixLeaf, stringOutsidePrefixLeaf), nil
	case IsNull:
		return chooseSet(negated, nullValueLeaf, nonNullValueLeaf), nil
	case IsPresent:
		return chooseSet(negated, anyValueLeaf, ""), nil
	case In, All, Any, Not:
		return "", invalidRequest(fmt.Sprintf("filter operator %d has no single leaf statement", op))
	default:
		return "", invalidRequest(fmt.Sprintf("filter operator %d is not a FilterOp", op))
	}
}

// typedStatement returns the true or false statement for the typed value of
// the column. Each pair lists the true statement, then the false statement.
func typedStatement(scalarType ScalarType, negated bool, stringPair [2]string, int64Pair [2]string, boolPair [2]string) (string, error) {
	switch scalarType {
	case String:
		return chooseSet(negated, stringPair[0], stringPair[1]), nil
	case Int64:
		return chooseSet(negated, int64Pair[0], int64Pair[1]), nil
	case Bool:
		return chooseSet(negated, boolPair[0], boolPair[1]), nil
	default:
		return "", invalidRequest(fmt.Sprintf("scalar type %d is not a ScalarType", scalarType))
	}
}

func chooseSet(negated bool, trueSet string, falseSet string) string {
	if negated {
		return falseSet
	}
	return trueSet
}

// queryLeaf runs a leaf statement with the namespace, the column, and the
// operator values as named parameters.
func (evaluator *filterEvaluator) queryLeaf(ctx context.Context, statement string, filter Filter) (*sql.Rows, error) {
	namespace := sql.Named("namespace", evaluator.namespace)
	column := sql.Named("column", filter.Column)
	var rows *sql.Rows
	var err error
	switch filter.Op {
	case Equal:
		rows, err = evaluator.catalog.QueryContext(ctx, statement, namespace, column, scalarParameter("value", filter.Values[0]))
	case Range:
		lower, upper := inclusiveRange(filter)
		rows, err = evaluator.catalog.QueryContext(ctx, statement, namespace, column, sql.Named("lower", lower), sql.Named("upper", upper))
	case Prefix:
		successor, bounded := prefixSuccessor(filter.Prefix)
		if bounded {
			rows, err = evaluator.catalog.QueryContext(ctx, statement, namespace, column,
				sql.Named("prefix", filter.Prefix), sql.Named("successor", successor))
		} else {
			rows, err = evaluator.catalog.QueryContext(ctx, statement, namespace, column, sql.Named("prefix", filter.Prefix))
		}
	case IsNull, IsPresent:
		rows, err = evaluator.catalog.QueryContext(ctx, statement, namespace, column)
	case In, All, Any, Not:
		return nil, invalidRequest(fmt.Sprintf("filter operator %d has no single leaf statement", filter.Op))
	default:
		return nil, invalidRequest(fmt.Sprintf("filter operator %d is not a FilterOp", filter.Op))
	}
	if err != nil {
		slog.ErrorContext(ctx, "evaluate filter leaf failed", "column", filter.Column, "err", err)
		return nil, fmt.Errorf("evaluate filter on column %s: %w", filter.Column, err)
	}
	return rows, nil
}

// noExcludedNode tells saveRows to keep every row.
const noExcludedNode = -1

// saveRows inserts every owner_id and row_key of rows into the set of node,
// except the rows in the set of excluded, and closes rows.
func (evaluator *filterEvaluator) saveRows(ctx context.Context, rows *sql.Rows, node int, excluded int, operation string, column string) (err error) {
	started := clock.Now()
	var scanned, accepted int64
	defer func() {
		evaluator.phases.recordFilterNode(filterNodeTiming{
			node: node, operation: operation, column: column,
			scanned: scanned, accepted: accepted, elapsed: clock.Now().Sub(started),
			complete: err == nil,
		})
	}()
	defer func() {
		err = errors.Join(err, closeRows(ctx, rows))
	}()
	insert := publicationInsert{statement: searchFilterRowsStatement, columns: 3}
	defer func() {
		if insert.prepared != nil {
			err = errors.Join(err, closeStatement(ctx, insert.prepared))
		}
	}()
	for rows.Next() {
		var ownerID, rowKey string
		if err := rows.Scan(&ownerID, &rowKey); err != nil {
			slog.ErrorContext(ctx, "scan filter row failed", "err", err)
			return fmt.Errorf("scan filter row: %w", err)
		}
		scanned++
		if excluded != noExcludedNode {
			present, err := evaluator.inSet(ctx, excluded, ownerID, rowKey)
			if err != nil {
				return err
			}
			if present {
				continue
			}
		}
		if err := insert.append(ctx, evaluator.writer, publicationInteger(int64(node)), publicationString(ownerID), publicationString(rowKey)); err != nil {
			return queryDatabaseError(ctx, "save filter row", err)
		}
		accepted++
	}
	if err := rows.Err(); err != nil {
		slog.ErrorContext(ctx, "read filter rows failed", "err", err)
		return fmt.Errorf("read filter rows: %w", err)
	}
	if err := insert.flush(ctx, evaluator.writer); err != nil {
		return queryDatabaseError(ctx, "save filter rows", err)
	}
	return nil
}

// inclusiveRange returns the inclusive bounds of a Range, which includes
// Lower and excludes Upper. A missing bound is the int64 limit. An Upper of
// [math.MinInt64] returns bounds that no value satisfies.
func inclusiveRange(filter Filter) (int64, int64) {
	lower := int64(math.MinInt64)
	upper := int64(math.MaxInt64)
	if filter.Lower != nil {
		lower = filter.Lower.Int64
	}
	if filter.Upper != nil {
		if filter.Upper.Int64 == math.MinInt64 {
			return math.MaxInt64, math.MinInt64
		}
		upper = filter.Upper.Int64 - 1
	}
	return lower, upper
}

// prefixSuccessor returns the least byte string above every string that
// starts with prefix: prefix without its trailing 0xFF bytes, with the last
// remaining byte increased by one. SQLite compares TEXT byte by byte, and a
// valid UTF-8 value starts with the prefix characters exactly when it starts
// with the prefix bytes. It reports false when every byte is 0xFF; the Prefix
// leaf then has no upper bound. Valid UTF-8 contains no 0xFF byte, and
// ValidateSearchRequest rejects a prefix that is not valid UTF-8.
func prefixSuccessor(prefix string) (string, bool) {
	successor := []byte(prefix)
	for len(successor) > 0 && successor[len(successor)-1] == math.MaxUint8 {
		successor = successor[:len(successor)-1]
	}
	if len(successor) == 0 {
		return "", false
	}
	successor[len(successor)-1]++
	return string(successor), true
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

// inSet reports whether the set of node contains the occurrence.
func (evaluator *filterEvaluator) inSet(ctx context.Context, node int, ownerID string, rowKey string) (bool, error) {
	var present int
	err := evaluator.writer.QueryRowContext(ctx, filterRowStatement,
		sql.Named("node", node), sql.Named("owner_id", ownerID), sql.Named("row_key", rowKey)).Scan(&present)
	if errors.Is(err, sql.ErrNoRows) {
		return false, nil
	}
	if err != nil {
		return false, queryDatabaseError(ctx, "read filter row", err)
	}
	return true, nil
}

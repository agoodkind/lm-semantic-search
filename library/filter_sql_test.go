package library

import (
	"context"
	"database/sql"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

// filterSetFixture is a catalog with scalar rows written directly and an
// empty query database.
type filterSetFixture struct {
	catalog *sql.DB
	query   *queryDatabase
}

func newFilterSetFixture(t *testing.T) *filterSetFixture {
	t.Helper()
	ctx := context.Background()
	root := t.TempDir()
	catalog, err := sql.Open("sqlite3", "file:"+filepath.Join(root, "catalog.sqlite"))
	if err != nil {
		t.Fatalf("open catalog: %v", err)
	}
	t.Cleanup(func() { _ = catalog.Close() })
	tx, err := catalog.BeginTx(ctx, nil)
	if err != nil {
		t.Fatalf("begin schema transaction: %v", err)
	}
	if err := createCatalogSchema(ctx, tx); err != nil {
		t.Fatalf("create catalog schema: %v", err)
	}
	if err := tx.Commit(); err != nil {
		t.Fatalf("commit catalog schema: %v", err)
	}
	query, err := openQueryDatabase(ctx, filepath.Join(root, "catalog.sqlite"), 1<<30, time.Hour)
	if err != nil {
		t.Fatalf("open query database: %v", err)
	}
	t.Cleanup(func() { _ = query.close() })
	return &filterSetFixture{catalog: catalog, query: query}
}

// writeScalar writes one string scalar of column c for row key in table.
func (fixture *filterSetFixture) writeScalar(t *testing.T, table string, rowKey string, value string) {
	t.Helper()
	statements := map[string]string{
		"occurrence_scalars": `INSERT INTO occurrence_scalars
			(namespace, owner_id, row_key, column_name, type, string_value, int64_value, bool_value, is_null)
			VALUES ('n', 'o', ?, 'c', ?, ?, NULL, NULL, 0)`,
		"effective_scalars": `INSERT INTO effective_scalars
			(namespace, owner_id, row_key, column_name, type, string_value, int64_value, bool_value, is_null, projection_order)
			VALUES ('n', 'o', ?, 'c', ?, ?, NULL, NULL, 0, 1)`,
	}
	if _, err := fixture.catalog.ExecContext(context.Background(), statements[table], rowKey, String, value); err != nil {
		t.Fatalf("write %s %s: %v", table, rowKey, err)
	}
}

// evaluateSet returns the sorted row keys of the true set of filter, or its
// false set when negated.
func (fixture *filterSetFixture) evaluateSet(t *testing.T, filter Filter, negated bool) []string {
	t.Helper()
	ctx := context.Background()
	catalog, err := fixture.catalog.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelDefault, ReadOnly: true})
	if err != nil {
		t.Fatalf("begin catalog read: %v", err)
	}
	defer func() { _ = catalog.Rollback() }()
	writer, err := fixture.query.conn.BeginTx(ctx, nil)
	if err != nil {
		t.Fatalf("begin query database write: %v", err)
	}
	defer func() { _ = writer.Rollback() }()
	evaluator := filterEvaluator{
		catalog:   catalog,
		writer:    writer,
		namespace: "n",
		columns:   map[string]ScalarColumn{"c": {Name: "c", Type: String, Nullable: true, Mutable: true, MaxLength: 64}},
		nextNode:  0,
	}
	node, err := evaluator.evaluate(ctx, filter, negated)
	if err != nil {
		t.Fatalf("evaluate %+v: %v", filter, err)
	}
	rows, err := writer.QueryContext(ctx, filterNodeRowsStatement, sql.Named("node", node))
	if err != nil {
		t.Fatalf("read node %d: %v", node, err)
	}
	defer func() { _ = rows.Close() }()
	var keys []string
	for rows.Next() {
		var ownerID, rowKey string
		if err := rows.Scan(&ownerID, &rowKey); err != nil {
			t.Fatalf("scan node row: %v", err)
		}
		keys = append(keys, rowKey)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("read node rows: %v", err)
	}
	slices.Sort(keys)
	return keys
}

func equalString(value string) Filter {
	return Filter{Op: Equal, Column: "c", Values: []ScalarValue{{Type: String, Null: false, String: value, Bool: false, Int64: 0}}}
}

// TestFilterSetsUseTheProjectedValue writes a published and a projected value
// that differ, in both directions. Each true and false set follows the
// projected value.
func TestFilterSetsUseTheProjectedValue(t *testing.T) {
	fixture := newFilterSetFixture(t)
	fixture.writeScalar(t, "occurrence_scalars", "projected-matches", "/w/published")
	fixture.writeScalar(t, "effective_scalars", "projected-matches", "/w/projected")
	fixture.writeScalar(t, "occurrence_scalars", "published-matches", "/w/projected")
	fixture.writeScalar(t, "effective_scalars", "published-matches", "/w/published")
	cases := []struct {
		name    string
		filter  Filter
		negated bool
		want    []string
	}{
		{name: "Equal projected, true set", filter: equalString("/w/projected"), negated: false, want: []string{"projected-matches"}},
		{name: "Equal projected, false set", filter: equalString("/w/projected"), negated: true, want: []string{"published-matches"}},
		{name: "Equal published, true set", filter: equalString("/w/published"), negated: false, want: []string{"published-matches"}},
		{name: "Equal published, false set", filter: equalString("/w/published"), negated: true, want: []string{"projected-matches"}},
		{name: "Prefix projected, true set", filter: Filter{Op: Prefix, Column: "c", Prefix: "/w/pro"}, negated: false, want: []string{"projected-matches"}},
		{name: "Prefix projected, false set", filter: Filter{Op: Prefix, Column: "c", Prefix: "/w/pro"}, negated: true, want: []string{"published-matches"}},
	}
	for _, testCase := range cases {
		if got := fixture.evaluateSet(t, testCase.filter, testCase.negated); !slices.Equal(got, testCase.want) {
			t.Errorf("%s = %q, want %q", testCase.name, got, testCase.want)
		}
	}
}

// TestPrefixWithoutSuccessorHasNoUpperBound evaluates a Prefix made only of
// 0xFF bytes. No byte string follows every string that starts with it. The
// true set has no upper bound.
func TestPrefixWithoutSuccessorHasNoUpperBound(t *testing.T) {
	if successor, bounded := prefixSuccessor("a\xff"); !bounded || successor != "b" {
		t.Fatalf("prefixSuccessor(\"a\\xff\") = %q, %v; want \"b\", true", successor, bounded)
	}
	if _, bounded := prefixSuccessor("\xff\xff"); bounded {
		t.Fatal("prefixSuccessor(\"\\xff\\xff\") reports a successor")
	}
	fixture := newFilterSetFixture(t)
	fixture.writeScalar(t, "occurrence_scalars", "equal", "\xff\xff")
	fixture.writeScalar(t, "occurrence_scalars", "longer", "\xff\xff\xffz")
	fixture.writeScalar(t, "occurrence_scalars", "below", "\xff\xfe")
	fixture.writeScalar(t, "occurrence_scalars", "ascii", "a")
	prefix := Filter{Op: Prefix, Column: "c", Prefix: "\xff\xff"}
	if got := fixture.evaluateSet(t, prefix, false); !slices.Equal(got, []string{"equal", "longer"}) {
		t.Errorf("true set = %q, want [equal longer]", got)
	}
	if got := fixture.evaluateSet(t, prefix, true); !slices.Equal(got, []string{"ascii", "below"}) {
		t.Errorf("false set = %q, want [ascii below]", got)
	}
}

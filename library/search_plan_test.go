package library

import (
	"context"
	"database/sql"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

// temporaryStoreOpcodes are the SQLite opcodes that open a b-tree or sorter in
// the temporary store, outside the query database pages that
// MaxTemporaryBytes limits.
var temporaryStoreOpcodes = []string{"OpenEphemeral", "SorterOpen", "OpenAutoindex"}

// TestQueryDatabaseStatementsUseNoTemporaryStore compiles every constant
// statement that page one runs in the query database and fails on any opcode
// that opens a b-tree or sorter in SQLite's temporary store.
func TestQueryDatabaseStatementsUseNoTemporaryStore(t *testing.T) {
	ctx := context.Background()
	query, err := openQueryDatabase(ctx, filepath.Join(t.TempDir(), "catalog.sqlite"), 1<<30, time.Hour)
	if err != nil {
		t.Fatalf("open query database: %v", err)
	}
	t.Cleanup(func() {
		if err := query.close(); err != nil {
			t.Errorf("close query database: %v", err)
		}
	})
	statements := map[string]string{
		"denseOrderStatement":       denseOrderStatement,
		"denseRankStatement":        denseRankStatement,
		"lexicalOrderStatement":     lexicalOrderStatement,
		"lexicalRankStatement":      lexicalRankStatement,
		"finalOrderStatement":       finalOrderStatement,
		"finalRowsStatement":        finalRowsStatement,
		"insertRankedStatement":     insertRankedStatement,
		"eligiblePostingsStatement": eligiblePostingsStatement,
		"insertFilterRowStatement":  insertFilterRowStatement,
		"filterNodeRowsStatement":   filterNodeRowsStatement,
		"filterBothRowsStatement":   filterBothRowsStatement,
		"eligibleKeysStatement":     eligibleKeysStatement,
		"filterRowStatement":        filterRowStatement,
		"groupCountStatement":       groupCountStatement,
		"incrementGroupStatement":   incrementGroupStatement,
		"insertPostingStatement":    insertPostingStatement,
		"insertLexicalScore":        insertLexicalScoreStatement,
		"insertCandidateStatement":  insertCandidateStatement,
		"insertQueryVector":         insertQueryVectorStatement,
		"rankedPageStatement":       rankedPageStatement,
		"rankedRowsStatement":       rankedRowsStatement,
		"rankedBytesStatement":      rankedBytesStatement,
		"vectorBlockStatement":      vectorBlockStatement,
		"saveScoreStatement":        saveScoreStatement,
		"unscoredVectorsStatement":  unscoredVectorsStatement,
	}
	for name, statement := range statements {
		for _, opcode := range explainOpcodes(t, query.conn, statement) {
			for _, temporary := range temporaryStoreOpcodes {
				if opcode == temporary {
					t.Errorf("%s compiles to %s, which uses SQLite's temporary store", name, opcode)
				}
			}
		}
	}
}

// explainOpcodes returns the opcode column of EXPLAIN for statement.
func explainOpcodes(t *testing.T, conn *sql.Conn, statement string) []string {
	t.Helper()
	rows, err := conn.QueryContext(context.Background(), "EXPLAIN "+statement, nullParameters(statement)...)
	if err != nil {
		t.Fatalf("EXPLAIN %s: %v", statement, err)
	}
	defer func() { _ = rows.Close() }()
	var opcodes []string
	for rows.Next() {
		var address int
		var opcode string
		var p1, p2, p3, p4, p5, comment sql.NullString
		if err := rows.Scan(&address, &opcode, &p1, &p2, &p3, &p4, &p5, &comment); err != nil {
			t.Fatalf("scan EXPLAIN row: %v", err)
		}
		opcodes = append(opcodes, opcode)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("read EXPLAIN rows: %v", err)
	}
	return opcodes
}

// parameterName matches one SQLite named parameter.
var parameterName = regexp.MustCompile(`:([a-z_]+)`)

// nullParameters binds NULL to every named or positional parameter of
// statement. The driver requires a value for each parameter before it
// compiles the statement.
func nullParameters(statement string) []any {
	seen := map[string]bool{}
	var parameters []any
	for _, match := range parameterName.FindAllStringSubmatch(statement, -1) {
		if !seen[match[1]] {
			seen[match[1]] = true
			parameters = append(parameters, sql.Named(match[1], nil))
		}
	}
	for range strings.Count(statement, "?") {
		parameters = append(parameters, nil)
	}
	return parameters
}

// scalarIndexStep matches a query plan step that reads a scalar table through
// its typed index with the namespace and column as a key prefix.
var scalarIndexStep = regexp.MustCompile(`^SEARCH v USING (COVERING )?INDEX (effective|occurrence)_scalars_(string|int64|bool) \(namespace=\? AND column_name=\?`)

// TestFilterLeavesSearchTheTypedScalarIndexes plans every leaf statement over
// the catalog schema. Each read of effective_scalars or occurrence_scalars as
// v must search a typed index by namespace and column, and each comparison on
// a typed value must use that value's index. The plan text comes from the
// SQLite build of the pinned github.com/mattn/go-sqlite3 module. It then
// compiles every catalog statement of Search and fails on any opcode that
// uses SQLite's temporary store.
func TestFilterLeavesSearchTheTypedScalarIndexes(t *testing.T) {
	ctx := context.Background()
	database, err := sql.Open("sqlite3", "file:"+filepath.Join(t.TempDir(), "catalog.sqlite"))
	if err != nil {
		t.Fatalf("open catalog: %v", err)
	}
	t.Cleanup(func() { _ = database.Close() })
	tx, err := database.BeginTx(ctx, nil)
	if err != nil {
		t.Fatalf("begin schema transaction: %v", err)
	}
	if err := createCatalogSchema(ctx, tx); err != nil {
		t.Fatalf("create catalog schema: %v", err)
	}
	if err := tx.Commit(); err != nil {
		t.Fatalf("commit catalog schema: %v", err)
	}
	leaves := map[string]struct {
		statement string
		index     string
	}{
		"stringEqualLeaf":         {stringEqualLeaf, "string_value=?"},
		"stringNotEqualLeaf":      {stringNotEqualLeaf, ""},
		"int64EqualLeaf":          {int64EqualLeaf, "int64_value=?"},
		"int64NotEqualLeaf":       {int64NotEqualLeaf, ""},
		"boolEqualLeaf":           {boolEqualLeaf, "bool_value=?"},
		"boolNotEqualLeaf":        {boolNotEqualLeaf, ""},
		"int64InRangeLeaf":        {int64InRangeLeaf, "int64_value>? AND int64_value<?"},
		"int64OutsideRangeLeaf":   {int64OutsideRangeLeaf, ""},
		"stringInPrefixLeaf":      {stringInPrefixLeaf, "string_value>? AND string_value<?"},
		"stringOutsidePrefixLeaf": {stringOutsidePrefixLeaf, ""},
		"stringFromPrefixLeaf":    {stringFromPrefixLeaf, "string_value>?"},
		"stringBeforePrefixLeaf":  {stringBeforePrefixLeaf, ""},
		"nullValueLeaf":           {nullValueLeaf, ""},
		"nonNullValueLeaf":        {nonNullValueLeaf, ""},
		"anyValueLeaf":            {anyValueLeaf, ""},
	}
	for name, leaf := range leaves {
		steps := explainQueryPlan(t, database, leaf.statement)
		reads := 0
		for _, step := range steps {
			if !strings.HasPrefix(step, "SCAN v") && !strings.HasPrefix(step, "SEARCH v ") {
				continue
			}
			reads++
			if !scalarIndexStep.MatchString(step) {
				t.Errorf("%s reads a scalar table without its typed index: %q", name, step)
				continue
			}
			if leaf.index != "" && !strings.Contains(step, leaf.index) {
				t.Errorf("%s does not search the typed value %q: %q", name, leaf.index, step)
			}
		}
		if reads < 2 {
			t.Errorf("%s plan has %d reads of a scalar table, want one or more per part: %q", name, reads, steps)
		}
	}
	// Every catalog statement of Search, including absentLeaf, reads the
	// catalog without SQLite's temporary store.
	conn, err := database.Conn(ctx)
	if err != nil {
		t.Fatalf("open catalog connection: %v", err)
	}
	defer func() { _ = conn.Close() }()
	catalogStatements := map[string]string{
		"absentLeaf":                     absentLeaf,
		"allCandidatesStatement":         allCandidatesStatement,
		"oneCandidateStatement":          oneCandidateStatement,
		"termPostingsStatement":          termPostingsStatement,
		"insertSnapshotStatement":        insertSnapshotStatement,
		"expiredSnapshotsStatement":      expiredSnapshotsStatement,
		"deleteSnapshotResultsStatement": deleteSnapshotResultsStatement,
		"deleteSnapshotStatement":        deleteSnapshotStatement,
		"snapshotBytesStatement":         snapshotBytesStatement,
		"insertSearchResultStatement":    insertSearchResultStatement,
		"cursorSnapshotStatement":        cursorSnapshotStatement,
		"snapshotPageStatement":          snapshotPageStatement,
		"sourceBlobStatement":            sourceBlobStatement,
	}
	for name, leaf := range leaves {
		catalogStatements[name] = leaf.statement
	}
	for name, statement := range catalogStatements {
		for _, opcode := range explainOpcodes(t, conn, statement) {
			for _, temporary := range temporaryStoreOpcodes {
				if opcode == temporary {
					t.Errorf("%s compiles to %s, which uses SQLite's temporary store", name, opcode)
				}
			}
		}
	}
}

// explainQueryPlan returns the detail column of EXPLAIN QUERY PLAN for
// statement.
func explainQueryPlan(t *testing.T, database *sql.DB, statement string) []string {
	t.Helper()
	rows, err := database.QueryContext(context.Background(), "EXPLAIN QUERY PLAN "+statement, nullParameters(statement)...)
	if err != nil {
		t.Fatalf("EXPLAIN QUERY PLAN %s: %v", statement, err)
	}
	defer func() { _ = rows.Close() }()
	var steps []string
	for rows.Next() {
		var id, parent, unused int
		var detail string
		if err := rows.Scan(&id, &parent, &unused, &detail); err != nil {
			t.Fatalf("scan query plan row: %v", err)
		}
		steps = append(steps, detail)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("read query plan rows: %v", err)
	}
	return steps
}

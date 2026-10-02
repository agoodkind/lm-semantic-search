package library

import (
	"context"
	"database/sql"
	"fmt"
	"path/filepath"
	"regexp"
	"strconv"
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
		"bulkFilterRows":            strings.ReplaceAll(searchFilterRowsStatement, "{{rows}}", strings.TrimSuffix(strings.Repeat("(?,?,?),", publicationInsertRows), ",")),
		"bulkCandidates":            strings.ReplaceAll(searchCandidatesStatement, "{{rows}}", strings.TrimSuffix(strings.Repeat("(?,?,?,?,?,?,?,?),", publicationInsertRows), ",")),
		"bulkVectors":               strings.ReplaceAll(searchVectorsStatement, "{{rows}}", strings.TrimSuffix(strings.Repeat("(?,?,?),", publicationInsertRows), ",")),
		"denseOrderStatement":       denseOrderStatement,
		"denseRankStatement":        denseRankStatement,
		"lexicalOrderStatement":     lexicalOrderStatement,
		"lexicalRankStatement":      lexicalRankStatement,
		"finalOrderStatement":       finalOrderStatement,
		"finalRowsStatement":        finalRowsStatement,
		"insertRankedStatement":     insertRankedStatement,
		"eligiblePostingsStatement": eligiblePostingsStatement,
		"filterNodeRowsStatement":   filterNodeRowsStatement,
		"filterBothRowsStatement":   filterBothRowsStatement,
		"eligibleKeysStatement":     eligibleKeysStatement,
		"filterRowStatement":        filterRowStatement,
		"groupCountStatement":       groupCountStatement,
		"incrementGroupStatement":   incrementGroupStatement,
		"insertPostingStatement":    insertPostingStatement,
		"insertLexicalScore":        insertLexicalScoreStatement,
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

// scalarTables are the two scalar tables that every filter leaf reads.
var scalarTables = []string{"effective_scalars", "occurrence_scalars"}

// openOpcodes open a read cursor on the b-tree at root page P2 as cursor P1.
// A multi-index OR loop opens its index cursor with ReopenIdx.
var openOpcodes = map[string]bool{"OpenRead": true, "ReopenIdx": true}

// seekOpcodes position an index cursor on a key prefix of P4 columns.
var seekOpcodes = map[string]bool{"SeekGE": true, "SeekGT": true, "SeekLE": true, "SeekLT": true}

// scanOpcodes position a cursor at one end of a b-tree, which starts a scan
// of the whole b-tree.
var scanOpcodes = map[string]bool{"Rewind": true, "Last": true}

// bytecodeRow is one EXPLAIN row: the opcode and its P1, P2, and P4 operands.
type bytecodeRow struct {
	opcode string
	p1     int64
	p2     int64
	p4     string
}

// indexCursor is one cursor that the bytecode opens on a scalar b-tree, with
// the longest key that any seek on it uses.
type indexCursor struct {
	btree   string
	seekKey int
	scanned bool
}

// TestFilterLeavesSearchTheTypedScalarIndexes checks that every filter leaf
// statement reads effective_scalars and occurrence_scalars through a seek on
// a typed scalar index and never scans a scalar table or index. It is the only
// check of the index condition that main set for #318 review item 5; the
// oracle tests check filter results and pass with a full scan. It reads the
// compiled bytecode of the SQLite 3.53.0 build that github.com/mattn/go-sqlite3
// v1.14.44 bundles, which exposes no statement step counts. A SQLite or planner
// change that stops using a typed index fails this test by design. For each
// scalar table the bytecode must open a cursor on an index of that table with
// OpenRead and seek it with a key of at least minimumKey columns: namespace
// and column_name, and the typed value for equality, range, prefix, and OR
// leaves. A not-equal leaf seeks any typed index by namespace and column and
// compares the value from the table row.
// It then compiles every catalog statement of Search and fails on any opcode
// that uses SQLite's temporary store.
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
	conn, err := database.Conn(ctx)
	if err != nil {
		t.Fatalf("open catalog connection: %v", err)
	}
	defer func() { _ = conn.Close() }()
	btrees := scalarBtrees(t, conn)
	leaves := map[string]struct {
		statement  string
		valueIndex string
		minimumKey int
	}{
		"stringEqualLeaf":         {stringEqualLeaf, "string", 3},
		"stringNotEqualLeaf":      {stringNotEqualLeaf, "", 2},
		"int64EqualLeaf":          {int64EqualLeaf, "int64", 3},
		"int64NotEqualLeaf":       {int64NotEqualLeaf, "", 2},
		"boolEqualLeaf":           {boolEqualLeaf, "bool", 3},
		"boolNotEqualLeaf":        {boolNotEqualLeaf, "", 2},
		"int64InRangeLeaf":        {int64InRangeLeaf, "int64", 3},
		"int64OutsideRangeLeaf":   {int64OutsideRangeLeaf, "int64", 3},
		"stringInPrefixLeaf":      {stringInPrefixLeaf, "string", 3},
		"stringOutsidePrefixLeaf": {stringOutsidePrefixLeaf, "string", 3},
		"stringFromPrefixLeaf":    {stringFromPrefixLeaf, "string", 3},
		"stringBeforePrefixLeaf":  {stringBeforePrefixLeaf, "string", 2},
		"nullValueLeaf":           {nullValueLeaf, "", 2},
		"nonNullValueLeaf":        {nonNullValueLeaf, "", 2},
		"anyValueLeaf":            {anyValueLeaf, "", 2},
	}
	for name, leaf := range leaves {
		cursors := scalarCursors(explainBytecode(t, conn, leaf.statement), btrees)
		for _, cursor := range cursors {
			if cursor.scanned {
				t.Errorf("%s scans %s", name, cursor.btree)
			}
		}
		for _, table := range scalarTables {
			if !seeksTypedIndex(cursors, table, leaf.valueIndex, leaf.minimumKey) {
				t.Errorf("%s does not seek a typed %q index of %s with %d key columns; cursors %s",
					name, leaf.valueIndex, table, leaf.minimumKey, describeCursors(cursors))
			}
		}
	}
	// Every catalog statement of Search, including absentLeaf, reads the
	// catalog without SQLite's temporary store.
	catalogStatements := map[string]string{
		"absentLeaf":                     absentLeaf,
		"allCandidatesStatement":         allCandidatesStatement,
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

// scalarBtrees maps the root page of each scalar table and each index of a
// scalar table to its name.
func scalarBtrees(t *testing.T, conn *sql.Conn) map[int64]string {
	t.Helper()
	rows, err := conn.QueryContext(context.Background(),
		`SELECT rootpage, name FROM sqlite_schema WHERE tbl_name IN ('effective_scalars', 'occurrence_scalars') AND rootpage > 0`)
	if err != nil {
		t.Fatalf("read scalar root pages: %v", err)
	}
	defer func() { _ = rows.Close() }()
	btrees := map[int64]string{}
	for rows.Next() {
		var rootPage int64
		var name string
		if err := rows.Scan(&rootPage, &name); err != nil {
			t.Fatalf("scan scalar root page: %v", err)
		}
		btrees[rootPage] = name
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("read scalar root pages: %v", err)
	}
	return btrees
}

// scalarCursors returns every cursor that bytecode opens on a scalar b-tree,
// with the longest seek key and whether a scan starts on it.
func scalarCursors(bytecode []bytecodeRow, btrees map[int64]string) []*indexCursor {
	byCursor := map[int64]*indexCursor{}
	var cursors []*indexCursor
	for _, row := range bytecode {
		switch {
		case openOpcodes[row.opcode]:
			if btree, found := btrees[row.p2]; found {
				cursor := &indexCursor{btree: btree, seekKey: 0, scanned: false}
				byCursor[row.p1] = cursor
				cursors = append(cursors, cursor)
			}
		case seekOpcodes[row.opcode]:
			if cursor, found := byCursor[row.p1]; found {
				keyColumns, err := strconv.Atoi(row.p4)
				if err == nil {
					cursor.seekKey = max(cursor.seekKey, keyColumns)
				}
			}
		case scanOpcodes[row.opcode]:
			if cursor, found := byCursor[row.p1]; found {
				cursor.scanned = true
			}
		}
	}
	return cursors
}

// seeksTypedIndex reports whether a cursor seeks a typed index of table with
// at least minimumKey key columns. An empty valueIndex accepts any typed index.
func seeksTypedIndex(cursors []*indexCursor, table string, valueIndex string, minimumKey int) bool {
	for _, cursor := range cursors {
		suffix, typed := strings.CutPrefix(cursor.btree, table+"_")
		if !typed || (suffix != "string" && suffix != "int64" && suffix != "bool") {
			continue
		}
		if valueIndex != "" && suffix != valueIndex {
			continue
		}
		if cursor.seekKey >= minimumKey {
			return true
		}
	}
	return false
}

// describeCursors formats each cursor as its b-tree, seek key length, and
// scan flag.
func describeCursors(cursors []*indexCursor) string {
	parts := make([]string, 0, len(cursors))
	for _, cursor := range cursors {
		parts = append(parts, fmt.Sprintf("%s(seek %d, scanned %v)", cursor.btree, cursor.seekKey, cursor.scanned))
	}
	return strings.Join(parts, ", ")
}

// explainBytecode returns the opcode, P1, P2, and P4 of every EXPLAIN row of
// statement.
func explainBytecode(t *testing.T, conn *sql.Conn, statement string) []bytecodeRow {
	t.Helper()
	rows, err := conn.QueryContext(context.Background(), "EXPLAIN "+statement, nullParameters(statement)...)
	if err != nil {
		t.Fatalf("EXPLAIN %s: %v", statement, err)
	}
	defer func() { _ = rows.Close() }()
	var bytecode []bytecodeRow
	for rows.Next() {
		var address int
		var row bytecodeRow
		var p3, p5 int64
		var p4, comment sql.NullString
		if err := rows.Scan(&address, &row.opcode, &row.p1, &row.p2, &p3, &p4, &p5, &comment); err != nil {
			t.Fatalf("scan EXPLAIN row: %v", err)
		}
		row.p4 = p4.String
		bytecode = append(bytecode, row)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("read EXPLAIN rows: %v", err)
	}
	return bytecode
}

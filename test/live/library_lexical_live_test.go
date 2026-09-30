//go:build live

package live

import (
	"database/sql"
	"errors"
	"fmt"
	"hash/crc32"
	"maps"
	"math"
	"slices"
	"strings"
	"testing"
	"time"
	"unicode"

	"goodkind.io/lm-semantic-search/library"
)

// lexicalOracleRow is one occurrence that the test expects in the catalog.
type lexicalOracleRow struct {
	namespace, ownerID, rowKey, searchText string
}

// lexicalOracleStats is the recount of one namespace from the expected rows.
type lexicalOracleStats struct {
	corpusSize  int64
	totalTokens int64
	frequencies map[int64]int64
}

// recountLexical recomputes the occurrence-weighted statistics of namespace
// from rows. The test texts are ASCII, and each token is a maximal run of
// ASCII letters and digits, lowercased, hashed with CRC-32 IEEE modulo
// 2^32-1, as the Milvus default analyzer does for ASCII text.
func recountLexical(rows []lexicalOracleRow, namespace string) lexicalOracleStats {
	stats := lexicalOracleStats{corpusSize: 0, totalTokens: 0, frequencies: map[int64]int64{}}
	for _, row := range rows {
		if row.namespace != namespace {
			continue
		}
		stats.corpusSize++
		seen := map[int64]bool{}
		tokens := strings.FieldsFunc(row.searchText, func(character rune) bool {
			return character > unicode.MaxASCII || !(unicode.IsLetter(character) || unicode.IsDigit(character))
		})
		for _, token := range tokens {
			stats.totalTokens++
			hash := int64(crc32.ChecksumIEEE([]byte(strings.ToLower(token))) % math.MaxUint32)
			if !seen[hash] {
				seen[hash] = true
				stats.frequencies[hash]++
			}
		}
	}
	return stats
}

// assertLexicalIndex reads the lexical tables of the catalog with its own
// SQLite connection and compares them with a recount of rows. It also
// requires the lexical occurrences to equal the published occurrences and
// every stored lexical content to have an occurrence.
func assertLexicalIndex(t *testing.T, harness *libraryHarness, descriptor library.StoreDescriptor, rows []lexicalOracleRow, step string) {
	t.Helper()
	database, err := sql.Open("sqlite3", "file:"+descriptor.CatalogPath+"?mode=ro")
	if err != nil {
		t.Fatalf("%s: open catalog: %v", step, err)
	}
	defer func() {
		if err := database.Close(); err != nil {
			t.Errorf("%s: close catalog: %v", step, err)
		}
	}()
	namespaces := map[string]bool{}
	for _, row := range rows {
		namespaces[row.namespace] = true
	}
	for _, namespace := range slices.Sorted(maps.Keys(namespaces)) {
		want := recountLexical(rows, namespace)
		var corpusSize, totalTokens int64
		if err := database.QueryRowContext(harness.context(),
			`SELECT corpus_size, total_tokens FROM lexical_stats WHERE namespace = ?`, namespace,
		).Scan(&corpusSize, &totalTokens); err != nil {
			t.Fatalf("%s: read lexical_stats of %s: %v", step, namespace, err)
		}
		if corpusSize != want.corpusSize || totalTokens != want.totalTokens {
			t.Fatalf("%s: %s corpus size %d and %d tokens, recount %d and %d",
				step, namespace, corpusSize, totalTokens, want.corpusSize, want.totalTokens)
		}
		got := map[int64]int64{}
		result, err := database.QueryContext(harness.context(), `SELECT term_hash, df FROM lexical_df WHERE namespace = ?`, namespace)
		if err != nil {
			t.Fatalf("%s: read lexical_df of %s: %v", step, namespace, err)
		}
		for result.Next() {
			var hash, frequency int64
			if err := result.Scan(&hash, &frequency); err != nil {
				t.Fatalf("%s: scan lexical_df: %v", step, err)
			}
			got[hash] = frequency
		}
		if err := result.Close(); err != nil {
			t.Fatalf("%s: close lexical_df rows: %v", step, err)
		}
		if !maps.Equal(got, want.frequencies) {
			t.Fatalf("%s: %s document frequencies %v, recount %v", step, namespace, got, want.frequencies)
		}
	}
	checks := []struct {
		name  string
		query string
	}{
		{name: "occurrence without lexical row", query: `SELECT COUNT(*) FROM occurrences LEFT JOIN lexical_occurrences USING (namespace, owner_id, row_key)
			WHERE lexical_occurrences.search_hash IS NULL OR lexical_occurrences.search_hash != occurrences.search_hash`},
		{name: "lexical row without occurrence", query: `SELECT COUNT(*) FROM lexical_occurrences LEFT JOIN occurrences USING (namespace, owner_id, row_key)
			WHERE occurrences.row_key IS NULL`},
		{name: "unreferenced lexical content", query: `SELECT COUNT(*) FROM lexical_content
			WHERE NOT EXISTS (SELECT 1 FROM lexical_occurrences WHERE lexical_occurrences.search_hash = lexical_content.search_hash)`},
	}
	for _, check := range checks {
		var count int
		if err := database.QueryRowContext(harness.context(), check.query).Scan(&count); err != nil {
			t.Fatalf("%s: %s: %v", step, check.name, err)
		}
		if count != 0 {
			t.Fatalf("%s: %d rows with %s", step, count, check.name)
		}
	}
	var occurrences int
	if err := database.QueryRowContext(harness.context(), `SELECT COUNT(*) FROM occurrences`).Scan(&occurrences); err != nil {
		t.Fatalf("%s: count occurrences: %v", step, err)
	}
	if occurrences != len(rows) {
		t.Fatalf("%s: catalog has %d occurrences, the test expects %d", step, occurrences, len(rows))
	}
}

func lexicalMessage(key string, text string, index int64) library.Occurrence {
	row := messageRow(key, text, index)
	row.EmbeddingInput = "passage: " + text
	return row
}

// TestLibraryLexicalPublicationCountsEveryOccurrence publishes, replays,
// replaces, reprojects, and deletes occurrences through the public library
// and compares the lexical statistics in the catalog with a recount after
// every step.
func TestLibraryLexicalPublicationCountsEveryOccurrence(t *testing.T) {
	harness := newLibraryHarness(t)
	t.Logf("embedding window start %s", time.Now().UTC().Format(time.RFC3339))
	defer func() { t.Logf("embedding window end %s", time.Now().UTC().Format(time.RFC3339)) }()
	descriptor := harness.descriptor("lexical")
	opened := harness.open(descriptor, harness.vectorStore("lexical_pool"))
	for _, spec := range []library.NamespaceSpec{conversationSpec(), codeSpec()} {
		if err := opened.RegisterNamespace(harness.context(), spec); err != nil {
			t.Fatalf("register %s: %v", spec.ID, err)
		}
	}

	conversation := appendBatch("conversation-a", 1, "append-1",
		lexicalMessage("m1", "Alpha beta", 1),
		lexicalMessage("m2", "Alpha beta", 2),
		lexicalMessage("m3", "beta gamma gamma", 3),
	)
	mustApply(t, harness.context(), opened, conversation)
	rows := []lexicalOracleRow{
		{namespace: "conversations", ownerID: "conversation-a", rowKey: "m1", searchText: "Alpha beta"},
		{namespace: "conversations", ownerID: "conversation-a", rowKey: "m2", searchText: "Alpha beta"},
		{namespace: "conversations", ownerID: "conversation-a", rowKey: "m3", searchText: "beta gamma gamma"},
	}
	assertLexicalIndex(t, harness, descriptor, rows, "append")

	mustApply(t, harness.context(), opened, conversation)
	assertLexicalIndex(t, harness, descriptor, rows, "append replay")

	codeBatch := func(order uint64, occurrences ...library.Occurrence) library.Batch {
		return library.Batch{
			Namespace: "code", OwnerID: "parse.go", GenerationOrder: order,
			IdempotencyToken: fmt.Sprintf("code-%d", order), Mode: library.Replace, Rows: occurrences,
		}
	}
	mustApply(t, harness.context(), opened, codeBatch(1,
		codeRow("parse.go", 0, "alpha delta"),
		codeRow("parse.go", 1, "epsilon"),
	))
	rows = append(rows,
		lexicalOracleRow{namespace: "code", ownerID: "parse.go", rowKey: "parse.go#0", searchText: "alpha delta"},
		lexicalOracleRow{namespace: "code", ownerID: "parse.go", rowKey: "parse.go#1", searchText: "epsilon"},
	)
	assertLexicalIndex(t, harness, descriptor, rows, "first replace")

	mustApply(t, harness.context(), opened, codeBatch(2,
		codeRow("parse.go", 0, "alpha delta"),
		codeRow("parse.go", 2, "zeta alpha alpha"),
	))
	rows = slices.DeleteFunc(rows, func(row lexicalOracleRow) bool { return row.rowKey == "parse.go#1" })
	rows = append(rows, lexicalOracleRow{namespace: "code", ownerID: "parse.go", rowKey: "parse.go#2", searchText: "zeta alpha alpha"})
	assertLexicalIndex(t, harness, descriptor, rows, "second replace")

	generationsBefore := readLexicalGenerations(t, harness, descriptor)
	if _, err := opened.ReprojectScalars(harness.context(), library.ScalarProjection{
		Namespace: "conversations", OwnerID: "conversation-a", ProjectionOrder: 1, IdempotencyToken: "projection-1",
		Rows: map[string]map[string]library.ScalarValue{
			"m1": {"archived": {Type: library.Bool, Bool: true}},
		},
	}); err != nil {
		t.Fatalf("reproject scalars: %v", err)
	}
	assertLexicalIndex(t, harness, descriptor, rows, "reprojection")
	if generationsAfter := readLexicalGenerations(t, harness, descriptor); !maps.Equal(generationsAfter, generationsBefore) {
		t.Fatalf("reprojection changed the lexical statistics generations from %v to %v", generationsBefore, generationsAfter)
	}

	if err := opened.Delete(harness.context(), []library.OccurrenceID{
		{Namespace: "code", OwnerID: "parse.go", RowKey: "parse.go#2"},
		{Namespace: "code", OwnerID: "parse.go", RowKey: "absent"},
	}); err != nil {
		t.Fatalf("delete: %v", err)
	}
	rows = slices.DeleteFunc(rows, func(row lexicalOracleRow) bool { return row.rowKey == "parse.go#2" })
	assertLexicalIndex(t, harness, descriptor, rows, "delete")
}

// TestLibraryLexicalSchemaUpgradesEmptyVersionOneCatalogs reopens a catalog
// rewritten to schema version 1. An empty catalog opens and records version
// 2. A catalog with occurrences fails with ErrStoreMismatch, because version 1
// saves no SearchText.
func TestLibraryLexicalSchemaUpgradesEmptyVersionOneCatalogs(t *testing.T) {
	harness := newLibraryHarness(t)
	t.Logf("embedding window start %s", time.Now().UTC().Format(time.RFC3339))
	defer func() { t.Logf("embedding window end %s", time.Now().UTC().Format(time.RFC3339)) }()

	empty := harness.descriptor("empty-v1")
	emptyPool := harness.vectorStore("empty_v1_pool")
	first, err := library.Open(harness.context(), library.Config{Store: empty, Vectors: emptyPool, Embedder: harness.embedder})
	if err != nil {
		t.Fatalf("open empty catalog: %v", err)
	}
	if err := first.Close(); err != nil {
		t.Fatalf("close empty catalog: %v", err)
	}
	rewriteToVersionOne(t, harness, empty)
	upgraded := harness.open(empty, emptyPool)
	if err := upgraded.RegisterNamespace(harness.context(), conversationSpec()); err != nil {
		t.Fatalf("register after the upgrade: %v", err)
	}
	mustApply(t, harness.context(), upgraded, appendBatch("conversation-a", 1, "append-1", lexicalMessage("m1", "alpha", 1)))
	assertLexicalIndex(t, harness, empty, []lexicalOracleRow{
		{namespace: "conversations", ownerID: "conversation-a", rowKey: "m1", searchText: "alpha"},
	}, "upgraded empty catalog")
	if version := readSchemaVersion(t, harness, empty); version != "2" {
		t.Fatalf("upgraded catalog records schema version %s, want 2", version)
	}

	populated := harness.descriptor("populated-v1")
	populatedPool := harness.vectorStore("populated_v1_pool")
	writer, err := library.Open(harness.context(), library.Config{Store: populated, Vectors: populatedPool, Embedder: harness.embedder})
	if err != nil {
		t.Fatalf("open populated catalog: %v", err)
	}
	if err := writer.RegisterNamespace(harness.context(), conversationSpec()); err != nil {
		t.Fatalf("register populated namespace: %v", err)
	}
	mustApply(t, harness.context(), writer, appendBatch("conversation-a", 1, "append-1", lexicalMessage("m1", "alpha", 1)))
	if err := writer.Close(); err != nil {
		t.Fatalf("close populated catalog: %v", err)
	}
	rewriteToVersionOne(t, harness, populated)
	reopened, err := library.Open(harness.context(), library.Config{Store: populated, Vectors: populatedPool, Embedder: harness.embedder})
	if err == nil {
		_ = reopened.Close()
		t.Fatal("a version 1 catalog with occurrences opened without a lexical index")
	}
	requireError(t, err, library.ErrStoreMismatch)
}

// rewriteToVersionOne drops the lexical tables and records schema version 1,
// which recreates the layout that the version 1 build saved.
func rewriteToVersionOne(t *testing.T, harness *libraryHarness, descriptor library.StoreDescriptor) {
	t.Helper()
	database, err := sql.Open("sqlite3", "file:"+descriptor.CatalogPath)
	if err != nil {
		t.Fatalf("open catalog for rewrite: %v", err)
	}
	defer func() {
		if err := database.Close(); err != nil {
			t.Errorf("close rewritten catalog: %v", err)
		}
	}()
	statements := []string{
		`DROP TABLE lexical_terms`,
		`DROP TABLE lexical_occurrences`,
		`DROP TABLE lexical_content`,
		`DROP TABLE lexical_stats`,
		`DROP TABLE lexical_df`,
		`UPDATE store_identity SET value = '1' WHERE key = 'schema_version'`,
	}
	for _, statement := range statements {
		if _, err := database.ExecContext(harness.context(), statement); err != nil {
			t.Fatalf("rewrite catalog to version 1 with %q: %v", statement, err)
		}
	}
}

func readSchemaVersion(t *testing.T, harness *libraryHarness, descriptor library.StoreDescriptor) string {
	t.Helper()
	database, err := sql.Open("sqlite3", "file:"+descriptor.CatalogPath+"?mode=ro")
	if err != nil {
		t.Fatalf("open catalog: %v", err)
	}
	defer func() {
		if err := database.Close(); err != nil {
			t.Errorf("close catalog: %v", err)
		}
	}()
	var version string
	if err := database.QueryRowContext(harness.context(), `SELECT value FROM store_identity WHERE key = 'schema_version'`).Scan(&version); err != nil {
		t.Fatalf("read schema version: %v", err)
	}
	return version
}

// readLexicalGenerations returns the lexical statistics generation of every
// namespace in the catalog.
func readLexicalGenerations(t *testing.T, harness *libraryHarness, descriptor library.StoreDescriptor) map[string]int64 {
	t.Helper()
	database, err := sql.Open("sqlite3", "file:"+descriptor.CatalogPath+"?mode=ro")
	if err != nil {
		t.Fatalf("open catalog: %v", err)
	}
	defer func() {
		if err := database.Close(); err != nil {
			t.Errorf("close catalog: %v", err)
		}
	}()
	result, err := database.QueryContext(harness.context(), `SELECT namespace, generation FROM lexical_stats`)
	if err != nil {
		t.Fatalf("read lexical_stats generations: %v", err)
	}
	generations := map[string]int64{}
	for result.Next() {
		var namespace string
		var generation int64
		if err := result.Scan(&namespace, &generation); err != nil {
			t.Fatalf("scan lexical_stats generation: %v", err)
		}
		generations[namespace] = generation
	}
	if err := errors.Join(result.Err(), result.Close()); err != nil {
		t.Fatalf("read lexical_stats generations: %v", err)
	}
	return generations
}

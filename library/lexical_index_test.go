package library

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"path/filepath"
	"slices"
	"testing"

	_ "github.com/mattn/go-sqlite3"
)

// lexicalTestRow is one live occurrence of the independent statistics oracle.
type lexicalTestRow struct {
	namespace, ownerID, rowKey, text string
}

func openLexicalTestCatalog(t *testing.T) *sql.DB {
	t.Helper()
	database, err := sql.Open("sqlite3", filepath.Join(t.TempDir(), "catalog.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := database.Close(); err != nil {
			t.Errorf("close catalog: %v", err)
		}
	})
	database.SetMaxOpenConns(1)
	if _, err := database.Exec(`PRAGMA foreign_keys = ON`); err != nil {
		t.Fatal(err)
	}
	for _, statement := range lexicalSchemaStatements {
		if _, err := database.Exec(statement); err != nil {
			t.Fatalf("schema statement %q: %v", statement, err)
		}
	}
	return database
}

func testSearchHash(text string) string {
	digest := sha256.Sum256([]byte(text))
	return hex.EncodeToString(digest[:])
}

func testLexicalOccurrences(rows []lexicalTestRow) []lexicalOccurrence {
	occurrences := make([]lexicalOccurrence, 0, len(rows))
	for _, row := range rows {
		occurrences = append(occurrences, lexicalOccurrence{
			OwnerID:    row.ownerID,
			RowKey:     row.rowKey,
			SearchHash: testSearchHash(row.text),
			SearchText: row.text,
		})
	}
	return occurrences
}

func publishLexicalTest(t *testing.T, database *sql.DB, namespace string, added []lexicalTestRow, removed []lexicalTestRow) error {
	t.Helper()
	ctx := context.Background()
	tx, err := database.BeginTx(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := publishLexical(ctx, tx, StandardAnalyzer, namespace, testLexicalOccurrences(added), testLexicalOccurrences(removed)); err != nil {
		if rollbackErr := tx.Rollback(); rollbackErr != nil {
			t.Fatalf("rollback: %v", rollbackErr)
		}
		return err
	}
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}
	return nil
}

// assertLexicalStatistics recomputes the corpus statistics of namespace from
// the live rows with the analyzer and compares them with the catalog. It also
// requires every stored content to have a live occurrence and the analyzer's
// postings.
func assertLexicalStatistics(t *testing.T, database *sql.DB, namespace string, live []lexicalTestRow, wantGeneration uint64) {
	t.Helper()
	ctx := context.Background()
	var wantSize, wantTokens uint64
	wantFrequencies := make(map[uint32]uint64)
	var queryTerms []lexicalTerm
	for _, row := range live {
		if row.namespace != namespace {
			continue
		}
		document := analyzeLexical(row.text)
		wantSize++
		wantTokens += document.length
		for _, term := range document.terms {
			if wantFrequencies[term.hash] == 0 {
				queryTerms = append(queryTerms, term)
			}
			wantFrequencies[term.hash]++
		}
	}
	tx, err := database.BeginTx(ctx, &sql.TxOptions{ReadOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := tx.Rollback(); err != nil {
			t.Errorf("rollback: %v", err)
		}
	}()
	generation, corpus, frequencies, err := readLexicalCorpus(ctx, tx, namespace, queryTerms)
	if err != nil {
		t.Fatal(err)
	}
	if generation != wantGeneration || corpus.size != wantSize || corpus.totalTokens != wantTokens {
		t.Fatalf("%s statistics generation %d size %d tokens %d, want %d, %d, %d",
			namespace, generation, corpus.size, corpus.totalTokens, wantGeneration, wantSize, wantTokens)
	}
	var storedFrequencyRows int
	if err := tx.QueryRowContext(ctx, `SELECT count(*) FROM lexical_df WHERE namespace = ?`, namespace).Scan(&storedFrequencyRows); err != nil {
		t.Fatal(err)
	}
	if storedFrequencyRows != len(wantFrequencies) || len(frequencies) != len(wantFrequencies) {
		t.Fatalf("%s stores %d document frequency rows and read %d, want %d", namespace, storedFrequencyRows, len(frequencies), len(wantFrequencies))
	}
	for hash, want := range wantFrequencies {
		if frequencies[hash] != want {
			t.Fatalf("%s document frequency of %d = %d, want %d", namespace, hash, frequencies[hash], want)
		}
	}

	referenced := make(map[string]string)
	for _, row := range live {
		referenced[testSearchHash(row.text)] = row.text
	}
	contents := make(map[string]uint64)
	rows, err := tx.QueryContext(ctx, `SELECT search_hash, document_length FROM lexical_content`)
	if err != nil {
		t.Fatal(err)
	}
	for rows.Next() {
		var searchHash string
		var documentLength uint64
		if err := rows.Scan(&searchHash, &documentLength); err != nil {
			t.Fatal(err)
		}
		contents[searchHash] = documentLength
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if err := rows.Close(); err != nil {
		t.Fatal(err)
	}
	if len(contents) != len(referenced) {
		t.Fatalf("catalog stores %d contents, want %d", len(contents), len(referenced))
	}
	for searchHash, documentLength := range contents {
		text, found := referenced[searchHash]
		if !found {
			t.Fatalf("content %s has no live occurrence", searchHash)
		}
		document := analyzeLexical(text)
		if documentLength != document.length {
			t.Fatalf("content %s length %d, want %d", searchHash, documentLength, document.length)
		}
		if postings := storedPostings(t, tx, searchHash); !slices.Equal(postings, document.terms) {
			t.Fatalf("content %s postings %+v, want %+v", searchHash, postings, document.terms)
		}
	}
}

func storedPostings(t *testing.T, tx *sql.Tx, searchHash string) []lexicalTerm {
	t.Helper()
	rows, err := tx.QueryContext(context.Background(),
		`SELECT term_hash, tf FROM lexical_terms WHERE search_hash = ? ORDER BY term_hash`, searchHash)
	if err != nil {
		t.Fatal(err)
	}
	var postings []lexicalTerm
	for rows.Next() {
		var hash, frequency uint32
		if err := rows.Scan(&hash, &frequency); err != nil {
			t.Fatal(err)
		}
		postings = append(postings, lexicalTerm{hash: hash, frequency: frequency})
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if err := rows.Close(); err != nil {
		t.Fatal(err)
	}
	return postings
}

func TestPublishLexicalCountsEveryOccurrence(t *testing.T) {
	t.Parallel()
	database := openLexicalTestCatalog(t)
	first := lexicalTestRow{namespace: "code", ownerID: "file-a", rowKey: "1", text: "alpha beta"}
	duplicate := lexicalTestRow{namespace: "code", ownerID: "file-a", rowKey: "2", text: "alpha beta"}
	different := lexicalTestRow{namespace: "code", ownerID: "file-a", rowKey: "3", text: "beta Gamma gamma"}
	shared := lexicalTestRow{namespace: "chat", ownerID: "conversation", rowKey: "1", text: "alpha beta"}

	if err := publishLexicalTest(t, database, "code", []lexicalTestRow{first, duplicate, different}, nil); err != nil {
		t.Fatal(err)
	}
	if err := publishLexicalTest(t, database, "chat", []lexicalTestRow{shared}, nil); err != nil {
		t.Fatal(err)
	}
	live := []lexicalTestRow{first, duplicate, different, shared}
	assertLexicalStatistics(t, database, "code", live, 1)
	assertLexicalStatistics(t, database, "chat", live, 1)

	// The replacement republishes row 1 with unchanged text, drops rows 2 and
	// 3, and adds row 4.
	added := lexicalTestRow{namespace: "code", ownerID: "file-a", rowKey: "4", text: "delta"}
	if err := publishLexicalTest(t, database, "code", []lexicalTestRow{first, added}, []lexicalTestRow{first, duplicate, different}); err != nil {
		t.Fatal(err)
	}
	live = []lexicalTestRow{first, added, shared}
	assertLexicalStatistics(t, database, "code", live, 2)
	assertLexicalStatistics(t, database, "chat", live, 1)

	// The deletion removes the last code occurrence of the content that the
	// chat namespace also stores.
	if err := publishLexicalTest(t, database, "code", nil, []lexicalTestRow{first}); err != nil {
		t.Fatal(err)
	}
	live = []lexicalTestRow{added, shared}
	assertLexicalStatistics(t, database, "code", live, 3)
	assertLexicalStatistics(t, database, "chat", live, 1)
}

func TestPublishLexicalRejectsUnknownRemovalWithoutChanges(t *testing.T) {
	t.Parallel()
	database := openLexicalTestCatalog(t)
	row := lexicalTestRow{namespace: "code", ownerID: "file-a", rowKey: "1", text: "alpha"}
	if err := publishLexicalTest(t, database, "code", []lexicalTestRow{row}, nil); err != nil {
		t.Fatal(err)
	}
	missing := lexicalTestRow{namespace: "code", ownerID: "file-a", rowKey: "2", text: "alpha"}
	extra := lexicalTestRow{namespace: "code", ownerID: "file-a", rowKey: "3", text: "beta"}
	if err := publishLexicalTest(t, database, "code", []lexicalTestRow{extra}, []lexicalTestRow{missing}); err == nil {
		t.Fatal("publishLexical accepted the removal of an occurrence that was never published")
	}
	assertLexicalStatistics(t, database, "code", []lexicalTestRow{row}, 1)
}

func TestAccumulateLexicalScoresFromStoredPostings(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	database := openLexicalTestCatalog(t)
	texts := []string{"alpha beta", "alpha beta", "beta beta gamma", "delta", "Alpha alpha alpha epsilon zeta", "!!!"}
	var rows []lexicalTestRow
	for index, text := range texts {
		rows = append(rows, lexicalTestRow{namespace: "code", ownerID: "file", rowKey: string(rune('a' + index)), text: text})
	}
	if err := publishLexicalTest(t, database, "code", rows, nil); err != nil {
		t.Fatal(err)
	}
	parameters, err := newLexicalRankParameters(1.2, 0.75)
	if err != nil {
		t.Fatal(err)
	}
	query := analyzeLexical("alpha beta alpha").terms
	tx, err := database.BeginTx(ctx, &sql.TxOptions{ReadOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := tx.Rollback(); err != nil {
			t.Errorf("rollback: %v", err)
		}
	}()
	_, corpus, frequencies, err := readLexicalCorpus(ctx, tx, "code", query)
	if err != nil {
		t.Fatal(err)
	}
	scorer, ranked := newLexicalScorer(parameters, corpus, query, frequencies)
	if !ranked {
		t.Fatal("query was not ranked")
	}
	postings, err := tx.QueryContext(ctx,
		`SELECT lexical_terms.search_hash, lexical_terms.term_hash, lexical_terms.tf, lexical_content.document_length
		FROM lexical_terms JOIN lexical_content ON lexical_content.search_hash = lexical_terms.search_hash
		WHERE lexical_terms.term_hash IN (?, ?)
		ORDER BY lexical_terms.search_hash, lexical_terms.term_hash`,
		int64(query[0].hash), int64(query[1].hash),
	)
	if err != nil {
		t.Fatal(err)
	}
	got := make(map[string]float32)
	if err := accumulateLexicalScores(ctx, postings, scorer, func(searchHash string, score float32) error {
		got[searchHash] = score
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	want := make(map[string]float32)
	for _, text := range texts {
		document := analyzeLexical(text)
		if score := scorer.scoreDocument(document.terms, document.length); score > 0 {
			want[testSearchHash(text)] = score
		}
	}
	if len(got) != len(want) || len(want) != 3 {
		t.Fatalf("accumulated %d content scores, want %d and 3 matching contents", len(got), len(want))
	}
	for searchHash, score := range want {
		if got[searchHash] != score {
			t.Fatalf("content %s score %v, want %v", searchHash, got[searchHash], score)
		}
	}
}

func TestPublishLexicalKeepsGenerationWhenEveryContentNetsToZero(t *testing.T) {
	t.Parallel()
	database := openLexicalTestCatalog(t)
	rows := []lexicalTestRow{
		{namespace: "code", ownerID: "file-a", rowKey: "1", text: "alpha beta"},
		{namespace: "code", ownerID: "file-a", rowKey: "2", text: "gamma"},
	}
	if err := publishLexicalTest(t, database, "code", rows, nil); err != nil {
		t.Fatal(err)
	}
	if err := publishLexicalTest(t, database, "code", rows, rows); err != nil {
		t.Fatal(err)
	}
	assertLexicalStatistics(t, database, "code", rows, 1)
	renamed := []lexicalTestRow{
		{namespace: "code", ownerID: "file-a", rowKey: "1", text: "alpha beta"},
		{namespace: "code", ownerID: "file-a", rowKey: "3", text: "gamma"},
	}
	if err := publishLexicalTest(t, database, "code", renamed, rows); err != nil {
		t.Fatal(err)
	}
	assertLexicalStatistics(t, database, "code", renamed, 1)
}

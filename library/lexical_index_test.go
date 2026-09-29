package library

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"math"
	"path/filepath"
	"slices"
	"strings"
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
	empty := lexicalTestRow{namespace: "code", ownerID: "file-b", rowKey: "1", text: ""}
	punctuation := lexicalTestRow{namespace: "code", ownerID: "file-b", rowKey: "2", text: "!!! ---"}
	shared := lexicalTestRow{namespace: "chat", ownerID: "conversation", rowKey: "1", text: "alpha beta"}

	if err := publishLexicalTest(t, database, "code", []lexicalTestRow{first, duplicate, different, empty, punctuation}, nil); err != nil {
		t.Fatal(err)
	}
	if err := publishLexicalTest(t, database, "chat", []lexicalTestRow{shared}, nil); err != nil {
		t.Fatal(err)
	}
	// Milvus counts a row without tokens in the corpus size. The two token-free
	// occurrences count here too.
	live := []lexicalTestRow{first, duplicate, different, empty, punctuation, shared}
	assertLexicalStatistics(t, database, "code", live, 1)
	assertLexicalStatistics(t, database, "chat", live, 1)

	// The replacement republishes row 1 with unchanged text, drops rows 2 and
	// 3, and adds row 4.
	added := lexicalTestRow{namespace: "code", ownerID: "file-a", rowKey: "4", text: "delta"}
	if err := publishLexicalTest(t, database, "code", []lexicalTestRow{first, added}, []lexicalTestRow{first, duplicate, different}); err != nil {
		t.Fatal(err)
	}
	live = []lexicalTestRow{first, added, empty, punctuation, shared}
	assertLexicalStatistics(t, database, "code", live, 2)
	assertLexicalStatistics(t, database, "chat", live, 1)

	// The deletion removes the last code occurrence of the content that the
	// chat namespace also stores, and one token-free occurrence.
	if err := publishLexicalTest(t, database, "code", nil, []lexicalTestRow{first, empty}); err != nil {
		t.Fatal(err)
	}
	live = []lexicalTestRow{added, punctuation, shared}
	assertLexicalStatistics(t, database, "code", live, 3)
	assertLexicalStatistics(t, database, "chat", live, 1)
}

func TestPublishLexicalRejectsUnknownRemovalWithoutChanges(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	database := openLexicalTestCatalog(t)
	row := lexicalTestRow{namespace: "code", ownerID: "file-a", rowKey: "1", text: "alpha"}
	if err := publishLexicalTest(t, database, "code", []lexicalTestRow{row}, nil); err != nil {
		t.Fatal(err)
	}
	missing := lexicalTestRow{namespace: "code", ownerID: "file-a", rowKey: "2", text: "alpha"}
	extra := lexicalTestRow{namespace: "code", ownerID: "file-a", rowKey: "3", text: "beta"}
	tx, err := database.BeginTx(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := tx.Rollback(); err != nil {
			t.Errorf("rollback: %v", err)
		}
	}()
	err = publishLexical(ctx, tx, StandardAnalyzer, "code",
		testLexicalOccurrences([]lexicalTestRow{extra}), testLexicalOccurrences([]lexicalTestRow{missing}))
	if err == nil {
		t.Fatal("publishLexical accepted the removal of an occurrence that was never published")
	}
	// The failed call wrote nothing inside the transaction before it returned.
	var occurrences, contents int
	var generation int64
	if err := tx.QueryRowContext(ctx,
		`SELECT (SELECT COUNT(*) FROM lexical_occurrences), (SELECT COUNT(*) FROM lexical_content),
			(SELECT generation FROM lexical_stats WHERE namespace = 'code')`,
	).Scan(&occurrences, &contents, &generation); err != nil {
		t.Fatal(err)
	}
	if occurrences != 1 || contents != 1 || generation != 1 {
		t.Fatalf("after the rejected call the transaction contains %d occurrences, %d contents, generation %d; want 1, 1, 1",
			occurrences, contents, generation)
	}
}

// bm25Oracle computes the BM25 score of text for query in float64 from the
// specification formula, with ASCII word tokens, independently of the
// library analyzer and scorer.
func bm25Oracle(corpus []string, query string, text string, k1 float64, b float64) float64 {
	tokenize := func(value string) []string {
		return strings.FieldsFunc(strings.ToLower(value), func(character rune) bool {
			return (character < 'a' || character > 'z') && (character < '0' || character > '9')
		})
	}
	documentFrequency := map[string]float64{}
	var totalTokens float64
	for _, document := range corpus {
		tokens := tokenize(document)
		totalTokens += float64(len(tokens))
		seen := map[string]bool{}
		for _, token := range tokens {
			if !seen[token] {
				seen[token] = true
				documentFrequency[token]++
			}
		}
	}
	corpusSize := float64(len(corpus))
	averageLength := totalTokens / corpusSize
	queryFrequency := map[string]float64{}
	for _, token := range tokenize(query) {
		queryFrequency[token]++
	}
	termFrequency := map[string]float64{}
	tokens := tokenize(text)
	for _, token := range tokens {
		termFrequency[token]++
	}
	var score float64
	for token, count := range queryFrequency {
		tf := termFrequency[token]
		if tf == 0 {
			continue
		}
		df := documentFrequency[token]
		inverse := math.Log(1 + (corpusSize-df+0.5)/(df+0.5))
		score += count * inverse * tf * (k1 + 1) / (tf + k1*(1-b+b*float64(len(tokens))/averageLength))
	}
	return score
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
	assertLexicalStatistics(t, database, "code", rows, 1)
	const queryText = "alpha beta alpha"
	parameters, err := newLexicalRankParameters(1.2, 0.75)
	if err != nil {
		t.Fatal(err)
	}
	query := analyzeLexical(queryText).terms
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
	// The float32 scores stay within a relative 1e-5 of the float64 oracle for
	// these small inputs. A doubled or missing term is far outside it.
	const tolerance = 1e-5
	matched := map[string]bool{}
	for _, text := range texts {
		want := bm25Oracle(texts, queryText, text, 1.2, 0.75)
		score, found := got[testSearchHash(text)]
		if want == 0 {
			if found {
				t.Fatalf("content %q has score %v, want no score", text, score)
			}
			continue
		}
		matched[testSearchHash(text)] = true
		if !found || math.Abs(float64(score)-want) > tolerance*want {
			t.Fatalf("content %q score %v, oracle %v", text, score, want)
		}
	}
	if len(matched) != 3 || len(got) != 3 {
		t.Fatalf("accumulated %d content scores and the oracle scored %d contents, want 3", len(got), len(matched))
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

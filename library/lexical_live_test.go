//go:build live

package library

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/config"
)

const (
	lexicalLiveDatabasePrefix = "lms_lib_live_"
	lexicalLiveIDField        = "id"
	lexicalLiveTextField      = "text"
	lexicalLiveSparseField    = "sparse"
	lexicalLiveTextMaxLength  = 65535
	// lexicalLiveScanChunk is the number of code points in one RunAnalyzer
	// request of the exhaustive scan.
	lexicalLiveScanChunk = 16384
	lexicalLiveTimeout   = 10 * time.Minute
	// lexicalLiveMaxReportedDifferences bounds the differences one failure
	// prints. The failure always reports the total count.
	lexicalLiveMaxReportedDifferences = 40
)

// lexicalLiveMilvus is one isolated Milvus database that this test created.
type lexicalLiveMilvus struct {
	client   *milvusclient.Client
	database string
}

// openLexicalLiveMilvus creates a new database named lms_lib_live_ plus 32
// random hex characters on the configured Milvus, fails on a name collision,
// and drops only that database in cleanup.
func openLexicalLiveMilvus(t *testing.T, ctx context.Context) lexicalLiveMilvus {
	t.Helper()
	cfg, err := config.Default()
	if err != nil {
		t.Fatalf("read LMS configuration for the Milvus address: %v", err)
	}
	if cfg.MilvusAddress == "" {
		t.Fatal("the LMS configuration sets no Milvus address; the lexical live suite requires a real Milvus")
	}
	suffix := make([]byte, 16)
	if _, err := cryptorand.Read(suffix); err != nil {
		t.Fatalf("read random database suffix: %v", err)
	}
	database := lexicalLiveDatabasePrefix + hex.EncodeToString(suffix)

	admin, err := milvusclient.New(ctx, &milvusclient.ClientConfig{Address: cfg.MilvusAddress, APIKey: cfg.MilvusToken})
	if err != nil {
		t.Fatalf("connect to Milvus at %s: %v", cfg.MilvusAddress, err)
	}
	t.Cleanup(func() {
		if err := admin.Close(context.WithoutCancel(ctx)); err != nil {
			t.Errorf("close Milvus admin client: %v", err)
		}
	})
	existing, err := admin.ListDatabase(ctx, milvusclient.NewListDatabaseOption())
	if err != nil {
		t.Fatalf("list Milvus databases: %v", err)
	}
	if slices.Contains(existing, database) {
		t.Fatalf("Milvus database %s already exists; refusing to reuse it", database)
	}
	if err := admin.CreateDatabase(ctx, milvusclient.NewCreateDatabaseOption(database)); err != nil {
		t.Fatalf("create Milvus database %s: %v", database, err)
	}
	t.Logf("created Milvus database %s at %s", database, time.Now().UTC().Format(time.RFC3339))
	// Cleanups run in reverse order: the collection cleanup registered below
	// runs first, then this database drop, then the admin close.
	t.Cleanup(func() {
		if err := admin.DropDatabase(context.WithoutCancel(ctx), milvusclient.NewDropDatabaseOption(database)); err != nil {
			t.Errorf("drop Milvus database %s: %v", database, err)
			return
		}
		t.Logf("dropped Milvus database %s at %s", database, time.Now().UTC().Format(time.RFC3339))
	})

	client, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: cfg.MilvusAddress,
		APIKey:  cfg.MilvusToken,
		DBName:  database,
	})
	if err != nil {
		t.Fatalf("connect to Milvus database %s: %v", database, err)
	}
	t.Cleanup(func() {
		cleanupContext := context.WithoutCancel(ctx)
		collections, err := client.ListCollections(cleanupContext, milvusclient.NewListCollectionOption())
		if err != nil {
			t.Errorf("list collections in %s before drop: %v", database, err)
		}
		for _, collection := range collections {
			if err := client.DropCollection(cleanupContext, milvusclient.NewDropCollectionOption(collection)); err != nil {
				t.Errorf("drop collection %s in %s: %v", collection, database, err)
			}
		}
		if err := client.Close(cleanupContext); err != nil {
			t.Errorf("close Milvus client for %s: %v", database, err)
		}
	})
	return lexicalLiveMilvus{client: client, database: database}
}

// lexicalLiveCollection describes one BM25 collection with the default
// analyzer on its text field.
type lexicalLiveCollection struct {
	name      string
	k1        float64
	b         float64
	algorithm string
}

// createLexicalLiveCollection creates a collection with three fields: an
// int64 primary key, a text field that uses the Milvus default analyzer, and a
// sparse field that a BM25 function fills from the text field.
func createLexicalLiveCollection(t *testing.T, ctx context.Context, milvus lexicalLiveMilvus, collection lexicalLiveCollection) {
	t.Helper()
	schema := entity.NewSchema().
		WithField(entity.NewField().WithName(lexicalLiveIDField).WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
		WithField(entity.NewField().
			WithName(lexicalLiveTextField).
			WithDataType(entity.FieldTypeVarChar).
			WithMaxLength(lexicalLiveTextMaxLength).
			WithEnableAnalyzer(true)).
		WithField(entity.NewField().WithName(lexicalLiveSparseField).WithDataType(entity.FieldTypeSparseVector)).
		WithFunction(entity.NewFunction().
			WithName("text_bm25").
			WithType(entity.FunctionTypeBM25).
			WithInputFields(lexicalLiveTextField).
			WithOutputFields(lexicalLiveSparseField))
	indexParams := map[string]string{
		index.IndexTypeKey:    string(index.SparseInverted),
		index.MetricTypeKey:   string(entity.BM25),
		"bm25_k1":             strconv.FormatFloat(collection.k1, 'g', -1, 64),
		"bm25_b":              strconv.FormatFloat(collection.b, 'g', -1, 64),
		"inverted_index_algo": collection.algorithm,
	}
	createOption := milvusclient.NewCreateCollectionOption(collection.name, schema).
		WithIndexOptions(milvusclient.NewCreateIndexOption(
			collection.name,
			lexicalLiveSparseField,
			index.NewGenericIndex(lexicalLiveSparseField, indexParams),
		).WithIndexName(lexicalLiveSparseField))
	if err := milvus.client.CreateCollection(ctx, createOption); err != nil {
		t.Fatalf("create BM25 collection %s in %s: %v", collection.name, milvus.database, err)
	}
	loadTask, err := milvus.client.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(collection.name))
	if err != nil {
		t.Fatalf("load BM25 collection %s in %s: %v", collection.name, milvus.database, err)
	}
	if err := loadTask.Await(ctx); err != nil {
		t.Fatalf("wait for BM25 collection %s in %s to load: %v", collection.name, milvus.database, err)
	}
}

// TestLibraryLexicalAnalyzerParity compares the local StandardAnalyzer with the
// Milvus default analyzer of a BM25 text field through RunAnalyzer. It scans
// every Unicode scalar value as a separate token and a set of mixed texts, and
// compares every token and its hash.
func TestLibraryLexicalAnalyzerParity(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), lexicalLiveTimeout)
	defer cancel()
	milvus := openLexicalLiveMilvus(t, ctx)
	collection := lexicalLiveCollection{name: "analyzer_parity", k1: 1.2, b: 0.75, algorithm: "TAAT_NAIVE"}
	createLexicalLiveCollection(t, ctx, milvus, collection)

	var differences []scalarDifference
	scanned := 0
	for first := rune(0); first <= unicode.MaxRune; first += lexicalLiveScanChunk {
		last := min(first+lexicalLiveScanChunk-1, unicode.MaxRune)
		chunkDifferences, chunkScanned := compareScalarChunk(t, ctx, milvus, collection.name, first, last)
		differences = append(differences, chunkDifferences...)
		scanned += chunkScanned
	}
	t.Logf("scanned %d Unicode scalar values; %d differ", scanned, len(differences))
	reportLexicalDifferences(t, "scalar value range", summarizeScalarDifferences(differences))

	texts := lexicalParityTexts()
	results, err := milvus.client.RunAnalyzer(ctx, milvusclient.NewRunAnalyzerOption(texts...).
		WithField(collection.name, lexicalLiveTextField).
		WithHash())
	if err != nil {
		t.Fatalf("run Milvus analyzer on %d parity texts: %v", len(texts), err)
	}
	if len(results) != len(texts) {
		t.Fatalf("Milvus returned %d analyzer results for %d texts", len(results), len(texts))
	}
	var textDifferences []string
	for textIndex, text := range texts {
		expected := results[textIndex].Tokens
		actual := lexicalTokens(text)
		if len(expected) != len(actual) {
			textDifferences = append(textDifferences, fmt.Sprintf(
				"text %d %q: Milvus returned %d tokens, local returned %d (%q)",
				textIndex, truncateForReport(text), len(expected), len(actual), actual,
			))
			continue
		}
		for tokenIndex, token := range expected {
			localHash := lexicalTermHash(actual[tokenIndex])
			if token.Text != actual[tokenIndex] || token.Hash != localHash {
				textDifferences = append(textDifferences, fmt.Sprintf(
					"text %d token %d: Milvus %q hash %d, local %q hash %d",
					textIndex, tokenIndex, token.Text, token.Hash, actual[tokenIndex], localHash,
				))
			}
		}
	}
	reportLexicalDifferences(t, "parity text", textDifferences)
}

// compareScalarChunk sends every scalar value from first through last,
// separated by spaces, to RunAnalyzer and compares the token that each value
// produces with the local analyzer. It returns the differences and the number
// of scalar values compared.
func compareScalarChunk(
	t *testing.T,
	ctx context.Context,
	milvus lexicalLiveMilvus,
	collection string,
	first rune,
	last rune,
) ([]scalarDifference, int) {
	t.Helper()
	var text strings.Builder
	offsets := make(map[int64]rune)
	for value := first; value <= last; value++ {
		// The NUL parity text covers U+0000, which ends a C string.
		if !utf8.ValidRune(value) || value == 0 {
			continue
		}
		offsets[int64(text.Len())] = value
		text.WriteRune(value)
		text.WriteByte(' ')
	}
	if len(offsets) == 0 {
		return nil, 0
	}
	results, err := milvus.client.RunAnalyzer(ctx, milvusclient.NewRunAnalyzerOption(text.String()).
		WithField(collection, lexicalLiveTextField).
		WithDetail().
		WithHash())
	if err != nil {
		t.Fatalf("run Milvus analyzer on U+%04X through U+%04X: %v", first, last, err)
	}
	if len(results) != 1 {
		t.Fatalf("Milvus returned %d analyzer results for one text", len(results))
	}
	milvusTokens := make(map[rune]*entity.Token, len(results[0].Tokens))
	var differences []scalarDifference
	for _, token := range results[0].Tokens {
		value, found := offsets[token.StartOffset]
		if !found {
			t.Fatalf("Milvus token %q starts at byte %d, which is not a scanned value", token.Text, token.StartOffset)
		}
		milvusTokens[value] = token
	}
	for _, value := range offsets {
		local := lexicalTokens(string(value))
		token, milvusHasToken := milvusTokens[value]
		switch {
		case !milvusHasToken && len(local) == 0:
		case !milvusHasToken:
			differences = append(differences, scalarDifference{value: value, kind: "Milvus emits no token, local emits one"})
		case len(local) != 1:
			differences = append(differences, scalarDifference{value: value, kind: "Milvus emits a token, local emits none"})
		case token.Text != local[0]:
			differences = append(differences, scalarDifference{value: value, kind: fmt.Sprintf(
				"Milvus lowercases to %q, local to %q", token.Text, local[0],
			)})
		case token.Hash != lexicalTermHash(local[0]):
			differences = append(differences, scalarDifference{value: value, kind: "token hash differs"})
		}
	}
	return differences, len(offsets)
}

// scalarDifference is one scalar value on which Milvus and the local analyzer
// disagree.
type scalarDifference struct {
	value rune
	kind  string
}

// summarizeScalarDifferences merges consecutive scalar values with the same
// kind of difference into ranges.
func summarizeScalarDifferences(differences []scalarDifference) []string {
	slices.SortFunc(differences, func(left scalarDifference, right scalarDifference) int {
		return int(left.value - right.value)
	})
	var ranges []string
	for start := 0; start < len(differences); {
		end := start
		for end+1 < len(differences) &&
			differences[end+1].value == differences[end].value+1 &&
			differences[end+1].kind == differences[start].kind {
			end++
		}
		ranges = append(ranges, fmt.Sprintf(
			"U+%04X..U+%04X (%d): %s",
			differences[start].value, differences[end].value, end-start+1, differences[start].kind,
		))
		start = end + 1
	}
	return ranges
}

// lexicalParityTexts returns mixed texts that exercise multi-character tokens,
// ASCII and non-ASCII lowercasing, combining marks, digits, hash truncation at
// 100 bytes, and separators.
func lexicalParityTexts() []string {
	return []string{
		"Hello, happy tax payer!",
		"make lint-deadcode && go test ./library/... -run '^TestLibraryLexical'",
		"İstanbul İİ Iİ ΣΊΣΥΦΟΣ ǅemal ǄEMAL ẞtraße Ǉ ǈ",
		"Русский текст and 中文分词 without jieba, 日本語テキスト, 한국어 문장",
		"e\u0301te\u0301 cafe\u0301 na\u0308ive Å ﬁle ﬀ ½ ² ⅷ Ⅻ ①",
		"x86_64 v2.6.18 0xFFFF 3.14159 1e-9 foo_bar foo-bar foo.bar",
		strings.Repeat("a", 99) + "Z " + strings.Repeat("a", 100) + "Z " + strings.Repeat("a", 101) + "Z",
		strings.Repeat("é", 49) + "X " + strings.Repeat("é", 50) + "X " + strings.Repeat("é", 51) + "X",
		// Byte 100 falls inside the 34th three-byte character of each token, so
		// the hash covers a partial UTF-8 sequence. The tokens differ only after
		// byte 100.
		strings.Repeat("中", 40) + " " + strings.Repeat("中", 39) + "文" + " " + strings.Repeat("क", 60),
		"emoji 👍🏽 flags 🇺🇸 zwj 👨‍👩‍👧 math 𝐀𝐁𝐂 𝟙𝟚 ancient 𐌰𐌱",
		"tabs\tand\nnewlines\r\nand\u00a0nbsp\u2028line\u3000ideographic",
		"before\x00after nul",
		"",
		"   \t\n",
	}
}

func truncateForReport(text string) string {
	const limit = 60
	if len(text) <= limit {
		return text
	}
	cut := limit
	for cut > 0 && !utf8.RuneStart(text[cut]) {
		cut--
	}
	return text[:cut] + "..."
}

func reportLexicalDifferences(t *testing.T, kind string, differences []string) {
	t.Helper()
	if len(differences) == 0 {
		return
	}
	shown := differences[:min(len(differences), lexicalLiveMaxReportedDifferences)]
	t.Errorf("%d %s differences between Milvus and the library; first %d:\n%s",
		len(differences), kind, len(shown), strings.Join(shown, "\n"))
}

// The collision pair shares CRC-32 IEEE 1506756466. The long tokens share
// their first 100 bytes and therefore their Milvus hash.
const (
	lexicalCollisionLeft  = "hzswstzcso"
	lexicalCollisionRight = "bkqqpyieph"
	lexicalCollisionHash  = 1506756466
)

var (
	lexicalLongPrefix     = strings.Repeat("p", 100)
	lexicalLongTokenLeft  = lexicalLongPrefix + "alpha"
	lexicalLongTokenRight = lexicalLongPrefix + "omega"
)

// TestLibraryLexicalScoreParity compares local BM25 scores with Milvus BM25
// search scores on a duplicate-heavy corpus in which every source occurrence is
// a separate Milvus row. It covers repeated query terms, colliding token
// hashes, tokens that share 100 leading bytes, documents without terms, and
// several k1, b, and index algorithm settings, first on growing segments and
// then on flushed and indexed segments.
func TestLibraryLexicalScoreParity(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), lexicalLiveTimeout)
	defer cancel()
	if lexicalTermHash(lexicalCollisionLeft) != lexicalCollisionHash ||
		lexicalTermHash(lexicalCollisionRight) != lexicalCollisionHash {
		t.Fatalf("collision tokens hash to %d and %d, not %d",
			lexicalTermHash(lexicalCollisionLeft), lexicalTermHash(lexicalCollisionRight), lexicalCollisionHash)
	}
	milvus := openLexicalLiveMilvus(t, ctx)
	texts := lexicalScoreCorpus()
	queries := lexicalScoreQueries()
	collections := []lexicalLiveCollection{
		{name: "bm25_default_taat", k1: 1.2, b: 0.75, algorithm: "TAAT_NAIVE"},
		{name: "bm25_default_maxscore", k1: 1.2, b: 0.75, algorithm: "DAAT_MAXSCORE"},
		{name: "bm25_b0_k2", k1: 2.0, b: 0, algorithm: "TAAT_NAIVE"},
		{name: "bm25_b1_k05_wand", k1: 0.5, b: 1, algorithm: "DAAT_WAND"},
	}
	for _, collection := range collections {
		createLexicalLiveCollection(t, ctx, milvus, collection)
		insertLexicalCorpus(t, ctx, milvus, collection.name, texts)
		compareLexicalScores(t, ctx, milvus, collection, texts, queries, "growing")
		flushLexicalCollection(t, ctx, milvus, collection.name, len(texts))
		compareLexicalScores(t, ctx, milvus, collection, texts, queries, "indexed")
	}

	// In this corpus the third row scores 0.17426978 when b*ratio+(1-b) rounds
	// once and 0.1742698 when it rounds twice.
	normalization := lexicalLiveCollection{name: "bm25_normalization_rounding", k1: 1.2, b: 0.75, algorithm: "TAAT_NAIVE"}
	normalizationTexts := []string{
		"alpha" + strings.Repeat(" filler", 46),
		"alpha",
		"alpha" + strings.Repeat(" filler", 7),
	}
	createLexicalLiveCollection(t, ctx, milvus, normalization)
	insertLexicalCorpus(t, ctx, milvus, normalization.name, normalizationTexts)
	compareLexicalScores(t, ctx, milvus, normalization, normalizationTexts, []string{"alpha"}, "growing")
	flushLexicalCollection(t, ctx, milvus, normalization.name, len(normalizationTexts))
	compareLexicalScores(t, ctx, milvus, normalization, normalizationTexts, []string{"alpha"}, "indexed")
}

// lexicalScoreCorpus returns a deterministic corpus with repeated source texts,
// skewed term frequencies, mixed case, colliding hashes, long tokens, empty
// texts, and texts without tokens.
func lexicalScoreCorpus() []string {
	vocabulary := []string{
		"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
		"iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
		"rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
		"daemon", "milvus", "catalog", "search", "vector", "lexical",
		lexicalCollisionLeft, lexicalCollisionRight, lexicalLongTokenLeft, lexicalLongTokenRight,
	}
	random := rand.New(rand.NewPCG(20260929, 712))
	separators := []string{" ", ", ", ".", " -- ", "\n", "/", "_"}
	var texts []string
	for range 240 {
		length := 1 + random.IntN(60)
		var text strings.Builder
		for position := range length {
			// Squaring a uniform draw skews choices toward the start of the
			// vocabulary.
			draw := random.Float64()
			word := vocabulary[int(draw*draw*float64(len(vocabulary)))]
			if random.IntN(5) == 0 {
				word = strings.ToUpper(word[:1]) + word[1:]
			}
			if position > 0 {
				text.WriteString(separators[random.IntN(len(separators))])
			}
			text.WriteString(word)
		}
		texts = append(texts, text.String())
	}
	for duplicate := range 20 {
		for range 1 + duplicate%4 {
			texts = append(texts, texts[duplicate*7])
		}
	}
	texts = append(texts,
		"", "", "!!! --- ???", "... ,,,",
		strings.Repeat("alpha ", 3000),
		lexicalCollisionLeft+" "+lexicalCollisionRight+" "+lexicalCollisionLeft,
		lexicalLongTokenLeft+" "+lexicalLongTokenRight,
		"Alpha ALPHA alpha aLpHa",
	)
	return texts
}

// lexicalScoreQueries returns queries with single, repeated, colliding, long,
// absent, and mixed terms, and one query without tokens.
func lexicalScoreQueries() []string {
	return []string{
		"alpha",
		"alpha alpha",
		"alpha alpha alpha beta",
		"Beta GAMMA delta",
		"omega psi chi phi",
		lexicalCollisionLeft,
		lexicalCollisionRight + " " + lexicalCollisionLeft,
		lexicalLongTokenLeft,
		lexicalLongPrefix + "zzz",
		"absentterm alpha",
		"absentterm",
		"daemon milvus catalog search vector lexical alpha beta gamma",
		"!!! ???",
	}
}

func insertLexicalCorpus(t *testing.T, ctx context.Context, milvus lexicalLiveMilvus, collection string, texts []string) {
	t.Helper()
	ids := make([]int64, len(texts))
	for index := range texts {
		ids[index] = int64(index)
	}
	result, err := milvus.client.Insert(ctx, milvusclient.NewColumnBasedInsertOption(collection).
		WithInt64Column(lexicalLiveIDField, ids).
		WithVarcharColumn(lexicalLiveTextField, texts))
	if err != nil {
		t.Fatalf("insert %d rows into %s: %v", len(texts), collection, err)
	}
	if result.InsertCount != int64(len(texts)) {
		t.Fatalf("Milvus inserted %d of %d rows into %s", result.InsertCount, len(texts), collection)
	}
}

// flushLexicalCollection seals the inserted rows and waits until the BM25
// index covers every row.
func flushLexicalCollection(t *testing.T, ctx context.Context, milvus lexicalLiveMilvus, collection string, rows int) {
	t.Helper()
	flushTask, err := milvus.client.Flush(ctx, milvusclient.NewFlushOption(collection))
	if err != nil {
		t.Fatalf("flush %s: %v", collection, err)
	}
	if err := flushTask.Await(ctx); err != nil {
		t.Fatalf("wait for flush of %s: %v", collection, err)
	}
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for {
		description, err := milvus.client.DescribeIndex(ctx, milvusclient.NewDescribeIndexOption(collection, lexicalLiveSparseField))
		if err != nil {
			t.Fatalf("describe BM25 index of %s: %v", collection, err)
		}
		if description.IndexedRows >= int64(rows) && description.PendingIndexRows == 0 {
			return
		}
		select {
		case <-ctx.Done():
			t.Fatalf("BM25 index of %s indexed %d of %d rows before the deadline", collection, description.IndexedRows, rows)
		case <-ticker.C:
		}
	}
}

// compareLexicalScores searches every query through Milvus with a limit equal
// to the row count and compares the returned IDs and float32 scores with the
// local scorer over the same rows. Each row is one occurrence.
func compareLexicalScores(
	t *testing.T,
	ctx context.Context,
	milvus lexicalLiveMilvus,
	collection lexicalLiveCollection,
	texts []string,
	queries []string,
	phase string,
) {
	t.Helper()
	documents := make([]lexicalDocument, len(texts))
	corpus := lexicalCorpus{size: uint64(len(texts)), totalTokens: 0}
	documentFrequencies := make(map[uint32]uint64)
	for index, text := range texts {
		documents[index] = analyzeLexical(text)
		corpus.totalTokens += documents[index].length
		for _, term := range documents[index].terms {
			documentFrequencies[term.hash]++
		}
	}
	parameters, err := newLexicalRankParameters(collection.k1, collection.b)
	if err != nil {
		t.Fatalf("rank parameters k1 %v b %v: %v", collection.k1, collection.b, err)
	}

	var (
		maximumUnits uint32
		compared     int
		mismatches   []string
	)
	for _, query := range queries {
		expected := make(map[int64]float32)
		scorer, ranked := newLexicalScorer(parameters, corpus, analyzeLexical(query).terms, documentFrequencies)
		if ranked {
			for index, document := range documents {
				if score := scorer.scoreDocument(document.terms, document.length); score > 0 {
					expected[int64(index)] = score
				}
			}
		}
		results, err := milvus.client.Search(ctx, milvusclient.NewSearchOption(
			collection.name,
			len(texts),
			[]entity.Vector{entity.Text(query)},
		).WithANNSField(lexicalLiveSparseField).WithConsistencyLevel(entity.ClStrong))
		if err != nil {
			t.Fatalf("%s %s: search %q: %v", collection.name, phase, query, err)
		}
		if len(results) != 1 {
			t.Fatalf("%s %s: search %q returned %d result sets", collection.name, phase, query, len(results))
		}
		actual := make(map[int64]float32, results[0].ResultCount)
		for position := range results[0].ResultCount {
			id, err := results[0].IDs.GetAsInt64(position)
			if err != nil {
				t.Fatalf("%s %s: read result ID %d for %q: %v", collection.name, phase, position, query, err)
			}
			actual[id] = results[0].Scores[position]
		}
		if len(actual) != len(expected) {
			mismatches = append(mismatches, fmt.Sprintf(
				"query %q: Milvus returned %d rows, local scored %d", query, len(actual), len(expected),
			))
		}
		for id, expectedScore := range expected {
			actualScore, found := actual[id]
			if !found {
				mismatches = append(mismatches, fmt.Sprintf(
					"query %q: Milvus omitted row %d with local score %v", query, id, expectedScore,
				))
				continue
			}
			units := float32Units(expectedScore, actualScore)
			maximumUnits = max(maximumUnits, units)
			compared++
			if units > lexicalScoreToleranceUnits {
				mismatches = append(mismatches, fmt.Sprintf(
					"query %q row %d: Milvus %v (%08x), local %v (%08x), %d float32 units apart",
					query, id, actualScore, math.Float32bits(actualScore),
					expectedScore, math.Float32bits(expectedScore), units,
				))
			}
		}
	}
	t.Logf("%s %s: compared %d scores over %d queries; largest difference %d float32 units",
		collection.name, phase, compared, len(queries), maximumUnits)
	reportLexicalDifferences(t, collection.name+" "+phase+" score", mismatches)
}

// lexicalScoreToleranceUnits is the largest accepted distance between a local
// and a Milvus score, counted in adjacent float32 values. Every score of the
// pinned server matched bit for bit.
const lexicalScoreToleranceUnits = 0

// float32Units returns the number of float32 values between two positive
// scores.
func float32Units(left float32, right float32) uint32 {
	leftBits := math.Float32bits(left)
	rightBits := math.Float32bits(right)
	if leftBits > rightBits {
		return leftBits - rightBits
	}
	return rightBits - leftBits
}

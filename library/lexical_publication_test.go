package library_test

import (
	"context"
	"database/sql"
	"errors"
	"hash/crc32"
	"maps"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/library"
)

// lexicalRecount is the occurrence-weighted lexical statistics of one
// namespace, recounted from the texts the test published.
type lexicalRecount struct {
	generation  int64
	corpusSize  int64
	totalTokens int64
	frequencies map[int64]int64
}

// recountASCII counts ASCII texts the way the Milvus default analyzer does:
// maximal runs of ASCII letters and digits, lowercased, hashed with CRC-32
// IEEE modulo 2^32-1.
func recountASCII(generation int64, texts []string) lexicalRecount {
	recount := lexicalRecount{generation: generation, corpusSize: 0, totalTokens: 0, frequencies: map[int64]int64{}}
	for _, text := range texts {
		recount.corpusSize++
		seen := map[int64]bool{}
		tokens := strings.FieldsFunc(strings.ToLower(text), func(character rune) bool {
			return (character < 'a' || character > 'z') && (character < '0' || character > '9')
		})
		for _, token := range tokens {
			recount.totalTokens++
			hash := int64(crc32.ChecksumIEEE([]byte(token)) % math.MaxUint32)
			if !seen[hash] {
				seen[hash] = true
				recount.frequencies[hash]++
			}
		}
	}
	return recount
}

// readLexicalStatistics reads the lexical statistics of namespace from the
// catalog file with its own read-only SQLite connection.
func readLexicalStatistics(t *testing.T, catalogPath string, namespace string) lexicalRecount {
	t.Helper()
	database, err := sql.Open("sqlite3", "file:"+catalogPath+"?mode=ro")
	if err != nil {
		t.Fatalf("open catalog: %v", err)
	}
	defer func() {
		if err := database.Close(); err != nil {
			t.Errorf("close catalog: %v", err)
		}
	}()
	ctx := context.Background()
	stored := lexicalRecount{generation: 0, corpusSize: 0, totalTokens: 0, frequencies: map[int64]int64{}}
	err = database.QueryRowContext(ctx,
		`SELECT generation, corpus_size, total_tokens FROM lexical_stats WHERE namespace = ?`, namespace,
	).Scan(&stored.generation, &stored.corpusSize, &stored.totalTokens)
	if err != nil {
		t.Fatalf("read lexical_stats of %s: %v", namespace, err)
	}
	rows, err := database.QueryContext(ctx, `SELECT term_hash, df FROM lexical_df WHERE namespace = ?`, namespace)
	if err != nil {
		t.Fatalf("read lexical_df of %s: %v", namespace, err)
	}
	for rows.Next() {
		var hash, frequency int64
		if err := rows.Scan(&hash, &frequency); err != nil {
			t.Fatalf("scan lexical_df: %v", err)
		}
		stored.frequencies[hash] = frequency
	}
	if err := errors.Join(rows.Err(), rows.Close()); err != nil {
		t.Fatalf("read lexical_df of %s: %v", namespace, err)
	}
	return stored
}

func assertLexicalRecount(t *testing.T, catalogPath string, step string, want lexicalRecount) {
	t.Helper()
	got := readLexicalStatistics(t, catalogPath, "code")
	if got.generation != want.generation || got.corpusSize != want.corpusSize || got.totalTokens != want.totalTokens {
		t.Fatalf("%s: generation %d, corpus size %d, %d tokens; want %d, %d, %d",
			step, got.generation, got.corpusSize, got.totalTokens, want.generation, want.corpusSize, want.totalTokens)
	}
	if !maps.Equal(got.frequencies, want.frequencies) {
		t.Fatalf("%s: document frequencies %v, want %v", step, got.frequencies, want.frequencies)
	}
}

// TestLibraryPublicationMaintainsLexicalStatistics publishes, replaces, and
// deletes one owner through the public library over the embedded pool and
// compares the catalog lexical statistics with a recount after each step.
func TestLibraryPublicationMaintainsLexicalStatistics(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	descriptor := embeddedDescriptor(t, filepath.Join(root, "catalog"))
	opened, err := library.Open(ctx, library.Config{
		Store:    descriptor,
		Vectors:  embeddedPool(t, filepath.Join(root, "pool")),
		Embedder: offlineEmbedder(t),
	})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() {
		if err := opened.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
	if err := opened.RegisterNamespace(ctx, library.NamespaceSpec{ID: "code", Policy: library.ReplaceAllowed, Scalars: nil}); err != nil {
		t.Fatalf("RegisterNamespace: %v", err)
	}

	mustApplyBatch(t, opened, library.Batch{
		Namespace: "code", OwnerID: "main.go", GenerationOrder: 1, IdempotencyToken: "g1", Mode: library.Replace,
		Rows: []library.Occurrence{
			codePartRow("a", "func Alpha beta"),
			codePartRow("b", "func Alpha beta"),
			codePartRow("c", "gamma gamma"),
			codePartRow("d", "!!! ---"),
		},
	})
	assertLexicalRecount(t, descriptor.CatalogPath, "first replace",
		recountASCII(1, []string{"func Alpha beta", "func Alpha beta", "gamma gamma", "!!! ---"}))

	mustApplyBatch(t, opened, library.Batch{
		Namespace: "code", OwnerID: "main.go", GenerationOrder: 2, IdempotencyToken: "g2", Mode: library.Replace,
		Rows: []library.Occurrence{
			codePartRow("a", "func Alpha beta"),
			codePartRow("e", "delta alpha"),
		},
	})
	assertLexicalRecount(t, descriptor.CatalogPath, "second replace",
		recountASCII(2, []string{"func Alpha beta", "delta alpha"}))

	if err := opened.Delete(ctx, []library.OccurrenceID{
		{Namespace: "code", OwnerID: "main.go", RowKey: "e"},
		{Namespace: "code", OwnerID: "main.go", RowKey: "absent"},
	}); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	assertLexicalRecount(t, descriptor.CatalogPath, "delete",
		recountASCII(3, []string{"func Alpha beta"}))
}

package library

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
)

// lexicalOccurrence is one occurrence that a publication adds or removes. The
// catalog computes SearchHash from SearchText.
type lexicalOccurrence struct {
	OwnerID, RowKey, SearchHash, SearchText string
}

// lexicalContentDelta is the net number of occurrences that one publication
// adds to or removes from one lexical content.
type lexicalContentDelta struct {
	searchHash string
	delta      int64
}

// publishLexical updates the lexical index of one namespace inside the
// catalog transaction that publishes or deletes occurrences. It deletes the
// removed occurrences, stores the content of each added occurrence once per
// search hash, and inserts the added occurrences. It then applies the net
// occurrence change of each content to the corpus size, the token total, and
// the document frequency of that content's terms, and advances the namespace
// statistics generation by 1. A content with a net change of zero changes no
// document frequency, and a publication in which every content nets to zero
// leaves the statistics and their generation unchanged. Content that no
// occurrence references after the change is deleted.
func publishLexical(
	ctx context.Context,
	tx *sql.Tx,
	analyzer string,
	namespace string,
	added []lexicalOccurrence,
	removed []lexicalOccurrence,
) error {
	if err := validateLexicalAnalyzer(analyzer); err != nil {
		return err
	}
	if len(added) == 0 && len(removed) == 0 {
		return nil
	}
	deltas := make(map[string]int64, len(added)+len(removed))
	for _, occurrence := range removed {
		var storedHash string
		err := tx.QueryRowContext(ctx,
			`DELETE FROM lexical_occurrences WHERE namespace = ? AND owner_id = ? AND row_key = ? RETURNING search_hash`,
			namespace, occurrence.OwnerID, occurrence.RowKey,
		).Scan(&storedHash)
		if err != nil {
			return lexicalIndexError(ctx, fmt.Errorf(
				"delete lexical occurrence %s/%s/%s: %w", namespace, occurrence.OwnerID, occurrence.RowKey, err,
			))
		}
		if storedHash != occurrence.SearchHash {
			return lexicalIndexError(ctx, fmt.Errorf(
				"lexical occurrence %s/%s/%s stores search hash %s, and the catalog removed %s",
				namespace, occurrence.OwnerID, occurrence.RowKey, storedHash, occurrence.SearchHash,
			))
		}
		deltas[storedHash]--
	}
	for _, occurrence := range added {
		if err := storeLexicalContent(ctx, tx, analyzer, occurrence.SearchHash, occurrence.SearchText); err != nil {
			return err
		}
		if _, err := tx.ExecContext(ctx,
			`INSERT INTO lexical_occurrences (namespace, owner_id, row_key, search_hash) VALUES (?, ?, ?, ?)`,
			namespace, occurrence.OwnerID, occurrence.RowKey, occurrence.SearchHash,
		); err != nil {
			return lexicalIndexError(ctx, fmt.Errorf(
				"insert lexical occurrence %s/%s/%s: %w", namespace, occurrence.OwnerID, occurrence.RowKey, err,
			))
		}
		deltas[occurrence.SearchHash]++
	}

	var sizeDelta, tokenDelta int64
	changed := false
	for _, content := range sortedLexicalDeltas(deltas) {
		if content.delta == 0 {
			continue
		}
		changed = true
		var documentLength int64
		if err := tx.QueryRowContext(ctx,
			`SELECT document_length FROM lexical_content WHERE search_hash = ?`,
			content.searchHash,
		).Scan(&documentLength); err != nil {
			return lexicalIndexError(ctx, fmt.Errorf("read lexical content %s: %w", content.searchHash, err))
		}
		sizeDelta += content.delta
		tokenDelta += content.delta * documentLength
		if err := changeDocumentFrequencies(ctx, tx, namespace, content); err != nil {
			return err
		}
		if content.delta < 0 {
			if err := deleteUnreferencedContent(ctx, tx, content.searchHash); err != nil {
				return err
			}
		}
	}
	if !changed {
		return nil
	}
	return updateLexicalStats(ctx, tx, namespace, sizeDelta, tokenDelta)
}

// storeLexicalContent analyzes searchText and stores its document length and
// term frequencies under searchHash. For a content that is already stored, it
// compares the stored analyzer identity and document length and writes
// nothing.
func storeLexicalContent(ctx context.Context, tx *sql.Tx, analyzer string, searchHash string, searchText string) error {
	document := analyzeLexical(searchText)
	var storedAnalyzer string
	var storedLength uint64
	err := tx.QueryRowContext(ctx,
		`SELECT analyzer_identity, document_length FROM lexical_content WHERE search_hash = ?`,
		searchHash,
	).Scan(&storedAnalyzer, &storedLength)
	switch {
	case err == nil:
		if storedAnalyzer != analyzer || storedLength != document.length {
			return lexicalIndexError(ctx, fmt.Errorf(
				"%w: lexical content %s stores analyzer %q and length %d, and the analysis produced %q and %d",
				ErrStoreMismatch, searchHash, storedAnalyzer, storedLength, analyzer, document.length,
			))
		}
		return nil
	case !errors.Is(err, sql.ErrNoRows):
		return lexicalIndexError(ctx, fmt.Errorf("read lexical content %s: %w", searchHash, err))
	}
	if _, err := tx.ExecContext(ctx,
		`INSERT INTO lexical_content (search_hash, analyzer_identity, document_length) VALUES (?, ?, ?)`,
		searchHash, analyzer, document.length,
	); err != nil {
		return lexicalIndexError(ctx, fmt.Errorf("insert lexical content %s: %w", searchHash, err))
	}
	for _, term := range document.terms {
		if _, err := tx.ExecContext(ctx,
			`INSERT INTO lexical_terms (search_hash, term_hash, tf) VALUES (?, ?, ?)`,
			searchHash, int64(term.hash), int64(term.frequency),
		); err != nil {
			return lexicalIndexError(ctx, fmt.Errorf("insert lexical term %d of %s: %w", term.hash, searchHash, err))
		}
	}
	return nil
}

// changeDocumentFrequencies adds the content delta to the document frequency
// of each term of one content. For a negative delta it deletes each frequency
// row of those terms that equals the removed count and subtracts the count
// from the others. It reads only that content's terms. A negative delta fails
// when any term has no frequency row or a frequency below the removed count.
func changeDocumentFrequencies(ctx context.Context, tx *sql.Tx, namespace string, content lexicalContentDelta) error {
	switch {
	case content.delta > 0:
		if _, err := tx.ExecContext(ctx,
			`INSERT INTO lexical_df (namespace, term_hash, df)
			SELECT ?, term_hash, ? FROM lexical_terms WHERE search_hash = ?
			ON CONFLICT (namespace, term_hash) DO UPDATE SET df = df + excluded.df`,
			namespace, content.delta, content.searchHash,
		); err != nil {
			return lexicalIndexError(ctx, fmt.Errorf("add document frequencies of %s: %w", content.searchHash, err))
		}
		return nil
	case content.delta == 0:
		return nil
	}
	removedCount := -content.delta
	var terms, covered int64
	if err := tx.QueryRowContext(ctx,
		`SELECT count(*), count(lexical_df.term_hash) FROM lexical_terms
		LEFT JOIN lexical_df ON lexical_df.namespace = ? AND lexical_df.term_hash = lexical_terms.term_hash
			AND lexical_df.df >= ?
		WHERE lexical_terms.search_hash = ?`,
		namespace, removedCount, content.searchHash,
	).Scan(&terms, &covered); err != nil {
		return lexicalIndexError(ctx, fmt.Errorf("check document frequencies of %s: %w", content.searchHash, err))
	}
	if covered != terms {
		return lexicalIndexError(ctx, fmt.Errorf(
			"lexical statistics of %s cover %d of %d terms of %s at frequency %d or more",
			namespace, covered, terms, content.searchHash, removedCount,
		))
	}
	if _, err := tx.ExecContext(ctx,
		`DELETE FROM lexical_df
		WHERE namespace = ? AND df = ? AND term_hash IN (SELECT term_hash FROM lexical_terms WHERE search_hash = ?)`,
		namespace, removedCount, content.searchHash,
	); err != nil {
		return lexicalIndexError(ctx, fmt.Errorf("delete exhausted document frequencies of %s: %w", content.searchHash, err))
	}
	if _, err := tx.ExecContext(ctx,
		`UPDATE lexical_df SET df = df - ?
		WHERE namespace = ? AND term_hash IN (SELECT term_hash FROM lexical_terms WHERE search_hash = ?)`,
		removedCount, namespace, content.searchHash,
	); err != nil {
		return lexicalIndexError(ctx, fmt.Errorf("subtract document frequencies of %s: %w", content.searchHash, err))
	}
	return nil
}

// deleteUnreferencedContent deletes the terms and the content row of a search
// hash that no lexical occurrence in any namespace references. Publication
// stores the content again when an occurrence with that text returns.
func deleteUnreferencedContent(ctx context.Context, tx *sql.Tx, searchHash string) error {
	var referenced bool
	if err := tx.QueryRowContext(ctx,
		`SELECT EXISTS (SELECT 1 FROM lexical_occurrences WHERE search_hash = ?)`,
		searchHash,
	).Scan(&referenced); err != nil {
		return lexicalIndexError(ctx, fmt.Errorf("check references to lexical content %s: %w", searchHash, err))
	}
	if referenced {
		return nil
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM lexical_terms WHERE search_hash = ?`, searchHash); err != nil {
		return lexicalIndexError(ctx, fmt.Errorf("delete lexical terms of %s: %w", searchHash, err))
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM lexical_content WHERE search_hash = ?`, searchHash); err != nil {
		return lexicalIndexError(ctx, fmt.Errorf("delete lexical content %s: %w", searchHash, err))
	}
	return nil
}

// updateLexicalStats adds the deltas to the namespace statistics and advances
// the generation by 1. The first change of a namespace inserts its row at
// generation 1. The table constraints fail a total below zero.
func updateLexicalStats(ctx context.Context, tx *sql.Tx, namespace string, sizeDelta int64, tokenDelta int64) error {
	result, err := tx.ExecContext(ctx,
		`UPDATE lexical_stats SET
			generation = generation + 1,
			corpus_size = corpus_size + ?,
			total_tokens = total_tokens + ?
		WHERE namespace = ?`,
		sizeDelta, tokenDelta, namespace,
	)
	if err != nil {
		return lexicalIndexError(ctx, fmt.Errorf(
			"change lexical statistics of %s by %d occurrences and %d tokens: %w",
			namespace, sizeDelta, tokenDelta, err,
		))
	}
	updated, err := result.RowsAffected()
	if err != nil {
		return lexicalIndexError(ctx, fmt.Errorf("count updated lexical statistics of %s: %w", namespace, err))
	}
	if updated == 1 {
		return nil
	}
	if _, err := tx.ExecContext(ctx,
		`INSERT INTO lexical_stats (namespace, generation, corpus_size, total_tokens) VALUES (?, 1, ?, ?)`,
		namespace, sizeDelta, tokenDelta,
	); err != nil {
		return lexicalIndexError(ctx, fmt.Errorf(
			"insert lexical statistics of %s with %d occurrences and %d tokens: %w",
			namespace, sizeDelta, tokenDelta, err,
		))
	}
	return nil
}

// readLexicalCorpus reads the namespace statistics generation, corpus size,
// token total, and the document frequency of each requested term hash. A
// namespace without statistics has generation 0 and an empty corpus.
func readLexicalCorpus(
	ctx context.Context,
	tx *sql.Tx,
	namespace string,
	terms []lexicalTerm,
) (uint64, lexicalCorpus, map[uint32]uint64, error) {
	var generation uint64
	var corpus lexicalCorpus
	err := tx.QueryRowContext(ctx,
		`SELECT generation, corpus_size, total_tokens FROM lexical_stats WHERE namespace = ?`,
		namespace,
	).Scan(&generation, &corpus.size, &corpus.totalTokens)
	switch {
	case errors.Is(err, sql.ErrNoRows):
		return 0, lexicalCorpus{size: 0, totalTokens: 0}, map[uint32]uint64{}, nil
	case err != nil:
		return 0, lexicalCorpus{}, nil, lexicalIndexError(ctx, fmt.Errorf("read lexical statistics of %s: %w", namespace, err))
	}
	frequencies := make(map[uint32]uint64, len(terms))
	for _, term := range terms {
		var documentFrequency uint64
		err := tx.QueryRowContext(ctx,
			`SELECT df FROM lexical_df WHERE namespace = ? AND term_hash = ?`,
			namespace, int64(term.hash),
		).Scan(&documentFrequency)
		switch {
		case errors.Is(err, sql.ErrNoRows):
			continue
		case err != nil:
			return 0, lexicalCorpus{}, nil, lexicalIndexError(ctx, fmt.Errorf(
				"read document frequency of term %d in %s: %w", term.hash, namespace, err,
			))
		case documentFrequency > corpus.size:
			return 0, lexicalCorpus{}, nil, lexicalIndexError(ctx, fmt.Errorf(
				"lexical document frequency %d for term %d of %s exceeds the corpus size %d",
				documentFrequency, term.hash, namespace, corpus.size,
			))
		}
		frequencies[term.hash] = documentFrequency
	}
	return generation, corpus, frequencies, nil
}

// sortedLexicalDeltas returns the deltas in ascending search hash order, which
// fixes the statement order of one publication.
func sortedLexicalDeltas(deltas map[string]int64) []lexicalContentDelta {
	sorted := make([]lexicalContentDelta, 0, len(deltas))
	for searchHash, delta := range deltas {
		sorted = append(sorted, lexicalContentDelta{searchHash: searchHash, delta: delta})
	}
	slices.SortFunc(sorted, func(left lexicalContentDelta, right lexicalContentDelta) int {
		return strings.Compare(left.searchHash, right.searchHash)
	})
	return sorted
}

// lexicalIndexError logs a lexical index failure and returns it.
func lexicalIndexError(ctx context.Context, err error) error {
	slog.WarnContext(ctx, "lexical index operation failed", "err", err)
	return err
}

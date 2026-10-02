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
	documents := make(map[string]lexicalDocument)
	for _, occurrence := range added {
		if _, found := documents[occurrence.SearchHash]; !found {
			documents[occurrence.SearchHash] = analyzeLexical(occurrence.SearchText)
		}
	}
	missing, err := prepareLexicalContent(ctx, tx, analyzer, documents)
	if err != nil {
		return lexicalIndexError(ctx, err)
	}
	return publishPreparedLexical(ctx, tx, analyzer, namespace, added, removed, missing)
}

func publishPreparedLexical(
	ctx context.Context,
	tx *sql.Tx,
	analyzer string,
	namespace string,
	added []lexicalOccurrence,
	removed []lexicalOccurrence,
	documents map[string]lexicalDocument,
) error {
	if err := validateLexicalAnalyzer(analyzer); err != nil {
		return err
	}
	if len(added) == 0 && len(removed) == 0 {
		return nil
	}
	deltas := make(map[string]int64, len(added)+len(removed))
	if err := removeLexicalOccurrences(ctx, tx, namespace, removed, deltas); err != nil {
		return lexicalIndexError(ctx, err)
	}
	if err := storePreparedLexicalContent(ctx, tx, analyzer, documents); err != nil {
		return lexicalIndexError(ctx, err)
	}
	insert := publicationInsert{statement: publicationLexicalOccurrencesStatement, columns: 4}
	for _, occurrence := range added {
		if err := insert.append(ctx, tx, publicationString(namespace), publicationString(occurrence.OwnerID), publicationString(occurrence.RowKey), publicationString(occurrence.SearchHash)); err != nil {
			return lexicalIndexError(ctx, err)
		}
		deltas[occurrence.SearchHash]++
	}
	if err := insert.flush(ctx, tx); err != nil {
		return lexicalIndexError(ctx, err)
	}

	if err := publishLexicalDeltas(ctx, tx, namespace, deltas); err != nil {
		return lexicalIndexError(ctx, err)
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

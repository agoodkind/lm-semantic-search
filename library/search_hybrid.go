package library

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log/slog"
)

// lexicalLeg is the frozen lexical input of one hybrid search: the analyzed
// query terms, the corpus statistics generation, and the scorer. ranked is
// false when the query has no terms or the corpus average length is zero; the
// search then omits the lexical ranking.
type lexicalLeg struct {
	terms      []lexicalTerm
	generation uint64
	scorer     lexicalScorer
	ranked     bool
}

// copyLexical reads the namespace corpus statistics and the query terms'
// document frequencies, and copies the postings of the query terms for the
// namespace's lexical content into the query database, all inside the catalog
// read transaction. Publication deletes lexical content that no occurrence
// references. The copy runs before the read transaction ends.
func (library *Library) copyLexical(
	ctx context.Context,
	tx *sql.Tx,
	query *queryDatabase,
	namespace string,
	text string,
) (lexicalLeg, error) {
	if err := requireLexicalIndex(ctx, tx); err != nil {
		return lexicalLeg{}, err
	}
	terms := analyzeLexical(text).terms
	generation, corpus, frequencies, err := readLexicalCorpus(ctx, tx, namespace, terms)
	if err != nil {
		return lexicalLeg{}, err
	}
	parameters, err := newLexicalRankParameters(library.config.BM25K1, *library.config.BM25B)
	if err != nil {
		return lexicalLeg{}, err
	}
	scorer, ranked := newLexicalScorer(parameters, corpus, terms, frequencies)
	leg := lexicalLeg{terms: terms, generation: generation, scorer: scorer, ranked: ranked}
	if !ranked {
		return leg, nil
	}
	for _, term := range terms {
		if err := copyTermPostings(ctx, tx, query, namespace, term.hash); err != nil {
			return lexicalLeg{}, err
		}
	}
	return leg, nil
}

// requireLexicalIndex returns an error that wraps [ErrInvalidRequest] when the
// catalog schema has no lexical tables. The catalog schema at this commit does
// not create them; the IC-6 lexical integration adds them to every catalog.
func requireLexicalIndex(ctx context.Context, tx *sql.Tx) error {
	var present bool
	if err := tx.QueryRowContext(ctx,
		`SELECT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'lexical_stats')`,
	).Scan(&present); err != nil {
		slog.ErrorContext(ctx, "read catalog schema failed", "err", err)
		return fmt.Errorf("read catalog schema: %w", err)
	}
	if !present {
		return invalidRequest("hybrid search needs the lexical index, and this catalog has no lexical tables; use SearchMode Dense")
	}
	return nil
}

func copyTermPostings(ctx context.Context, tx *sql.Tx, query *queryDatabase, namespace string, termHash uint32) (err error) {
	rows, err := tx.QueryContext(ctx,
		`SELECT t.search_hash, t.term_hash, t.tf, c.document_length FROM lexical_terms t
		JOIN lexical_content c ON c.search_hash = t.search_hash
		WHERE t.term_hash = ? AND EXISTS (
			SELECT 1 FROM lexical_occurrences lo WHERE lo.search_hash = t.search_hash AND lo.namespace = ?)`,
		int64(termHash), namespace,
	)
	if err != nil {
		slog.ErrorContext(ctx, "read lexical postings failed", "term", termHash, "err", err)
		return fmt.Errorf("read lexical postings of term %d: %w", termHash, err)
	}
	defer func() {
		err = errors.Join(err, closeRows(ctx, rows))
	}()
	writer, err := query.conn.BeginTx(ctx, nil)
	if err != nil {
		return queryDatabaseError(ctx, "begin postings copy", err)
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, writer.Rollback())
		}
	}()
	for rows.Next() {
		var searchHash string
		var term, frequency, documentLength int64
		if err := rows.Scan(&searchHash, &term, &frequency, &documentLength); err != nil {
			slog.ErrorContext(ctx, "scan lexical posting failed", "err", err)
			return fmt.Errorf("scan lexical posting: %w", err)
		}
		if _, err := writer.ExecContext(ctx,
			`INSERT INTO postings (search_hash, term_hash, tf, document_length) VALUES (?, ?, ?, ?)`,
			searchHash, term, frequency, documentLength,
		); err != nil {
			return queryDatabaseError(ctx, "copy lexical posting", err)
		}
	}
	if err := rows.Err(); err != nil {
		slog.ErrorContext(ctx, "read lexical postings failed", "err", err)
		return fmt.Errorf("read lexical postings of term %d: %w", termHash, err)
	}
	if err := writer.Commit(); err != nil {
		return queryDatabaseError(ctx, "commit postings copy", err)
	}
	return nil
}

// scoreLexical computes one BM25 score for each eligible lexical content from
// the copied postings and saves it in lexical_scores. A content without a
// posting for a query term receives no score and no lexical rank.
func scoreLexical(ctx context.Context, query *queryDatabase, leg lexicalLeg) (err error) {
	if !leg.ranked {
		return nil
	}
	writer, err := query.conn.BeginTx(ctx, nil)
	if err != nil {
		return queryDatabaseError(ctx, "begin lexical scoring", err)
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, writer.Rollback())
		}
	}()
	insert, err := writer.PrepareContext(ctx, `INSERT INTO lexical_scores (search_hash, score) VALUES (?, ?)`)
	if err != nil {
		return queryDatabaseError(ctx, "prepare lexical score insert", err)
	}
	defer func() {
		err = errors.Join(err, closeStatement(ctx, insert))
	}()
	rows, err := writer.QueryContext(ctx,
		`SELECT search_hash, term_hash, tf, document_length FROM postings
		WHERE search_hash IN (SELECT search_hash FROM candidates) ORDER BY search_hash, term_hash`,
	)
	if err != nil {
		return queryDatabaseError(ctx, "read copied postings", err)
	}
	if err := accumulateLexicalScores(ctx, rows, leg.scorer, func(searchHash string, score float32) error {
		if _, err := insert.ExecContext(ctx, searchHash, float64(score)); err != nil {
			return queryDatabaseError(ctx, "save lexical score", err)
		}
		return nil
	}); err != nil {
		return err
	}
	if err := writer.Commit(); err != nil {
		return queryDatabaseError(ctx, "commit lexical scores", err)
	}
	return nil
}

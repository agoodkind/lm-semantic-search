package library

import (
	"context"
	"database/sql"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"math"
	"slices"
	"strings"
)

//go:embed publication_lexical_content.sql
var publicationLexicalContentStatement string

//go:embed publication_lexical_existing.sql
var publicationLexicalExistingStatement string

//go:embed publication_lexical_deltas.sql
var publicationLexicalDeltasStatement string

//go:embed publication_lexical_frequency_check.sql
var publicationLexicalFrequencyCheckStatement string

//go:embed publication_lexical_frequency_delete.sql
var publicationLexicalFrequencyDeleteStatement string

//go:embed publication_lexical_frequency_upsert.sql
var publicationLexicalFrequencyUpsertStatement string

//go:embed publication_lexical_totals.sql
var publicationLexicalTotalsStatement string

//go:embed publication_lexical_unused_terms.sql
var publicationLexicalUnusedTermsStatement string

//go:embed publication_lexical_unused_content.sql
var publicationLexicalUnusedContentStatement string

//go:embed publication_lexical_remove.sql
var publicationLexicalRemoveStatement string

func removeLexicalOccurrences(ctx context.Context, tx *sql.Tx, namespace string, removed []lexicalOccurrence, deltas map[string]int64) error {
	for start := 0; start < len(removed); start += publicationInsertRows {
		batch := removed[start:min(start+publicationInsertRows, len(removed))]
		arguments := make([]any, 1, 1+len(batch)*2)
		arguments[0] = namespace
		expected := make(map[[2]string]string, len(batch))
		for _, occurrence := range batch {
			arguments = append(arguments, occurrence.OwnerID, occurrence.RowKey)
			expected[[2]string{occurrence.OwnerID, occurrence.RowKey}] = occurrence.SearchHash
		}
		groups := strings.TrimSuffix(strings.Repeat("(?,?),", len(batch)), ",")
		rows, err := tx.QueryContext(ctx, strings.ReplaceAll(publicationLexicalRemoveStatement, "{{rows}}", groups), arguments...)
		if err != nil {
			slog.ErrorContext(ctx, "remove lexical occurrences failed", "namespace", namespace, "err", err)
			return fmt.Errorf("remove lexical occurrences: %w", err)
		}
		if err := checkRemovedLexicalOccurrences(ctx, rows, expected, deltas); err != nil {
			return err
		}
	}
	return nil
}

func checkRemovedLexicalOccurrences(ctx context.Context, rows *sql.Rows, expected map[[2]string]string, deltas map[string]int64) (err error) {
	defer func() { err = errors.Join(err, closeRows(ctx, rows)) }()
	for rows.Next() {
		var owner, key, hash string
		if err := rows.Scan(&owner, &key, &hash); err != nil {
			slog.ErrorContext(ctx, "read removed lexical occurrence failed", "err", err)
			return fmt.Errorf("read removed lexical occurrence: %w", err)
		}
		id := [2]string{owner, key}
		if saved, found := expected[id]; !found || saved != hash {
			return fmt.Errorf("removed lexical occurrence %s/%s has an unexpected search hash", owner, key)
		}
		delete(expected, id)
		deltas[hash]--
	}
	if err := rows.Err(); err != nil {
		slog.ErrorContext(ctx, "iterate removed lexical occurrences failed", "err", err)
		return fmt.Errorf("iterate removed lexical occurrences: %w", err)
	}
	if len(expected) != 0 {
		return fmt.Errorf("%d removed occurrences have no lexical occurrence", len(expected))
	}
	return nil
}

func prepareLexicalContent(ctx context.Context, tx *sql.Tx, analyzer string, documents map[string]lexicalDocument) (map[string]lexicalDocument, error) {
	missing := maps.Clone(documents)
	hashes := slices.Sorted(maps.Keys(documents))
	for start := 0; start < len(hashes); start += publicationInsertRows {
		batch := hashes[start:min(start+publicationInsertRows, len(hashes))]
		arguments := make([]any, len(batch))
		for index, hash := range batch {
			arguments[index] = hash
		}
		placeholders := strings.TrimSuffix(strings.Repeat("?,", len(batch)), ",")
		statement := strings.ReplaceAll(publicationLexicalExistingStatement, "{{hashes}}", placeholders)
		rows, err := tx.QueryContext(ctx, statement, arguments...)
		if err != nil {
			slog.ErrorContext(ctx, "read lexical content identities failed", "err", err)
			return nil, fmt.Errorf("read lexical content identities: %w", err)
		}
		if err := checkLexicalContent(ctx, rows, analyzer, documents, missing); err != nil {
			return nil, err
		}
	}
	return missing, nil
}

func checkLexicalContent(ctx context.Context, rows *sql.Rows, analyzer string, documents map[string]lexicalDocument, missing map[string]lexicalDocument) (err error) {
	defer func() { err = errors.Join(err, closeRows(ctx, rows)) }()
	for rows.Next() {
		var hash, storedAnalyzer string
		var length uint64
		if err := rows.Scan(&hash, &storedAnalyzer, &length); err != nil {
			slog.ErrorContext(ctx, "read lexical content identity failed", "err", err)
			return fmt.Errorf("read lexical content identity: %w", err)
		}
		if storedAnalyzer != analyzer || length != documents[hash].length {
			return fmt.Errorf("%w: lexical content %s has a different analyzer or length", ErrStoreMismatch, hash)
		}
		delete(missing, hash)
	}
	if err := rows.Err(); err != nil {
		slog.ErrorContext(ctx, "iterate lexical content identities failed", "err", err)
		return fmt.Errorf("iterate lexical content identities: %w", err)
	}
	return nil
}

func storePreparedLexicalContent(ctx context.Context, tx *sql.Tx, analyzer string, documents map[string]lexicalDocument) error {
	hashes := slices.Sorted(maps.Keys(documents))
	contents := publicationInsert{statement: publicationLexicalContentStatement, columns: 3}
	for _, hash := range hashes {
		length := documents[hash].length
		if length > math.MaxInt64 {
			return fmt.Errorf("lexical content %s length exceeds the SQL integer range", hash)
		}
		if err := contents.append(ctx, tx, publicationString(hash), publicationString(analyzer), publicationInteger(int64(length))); err != nil {
			return err
		}
	}
	if err := contents.flush(ctx, tx); err != nil {
		return err
	}
	terms := publicationInsert{statement: publicationLexicalTermsStatement, columns: 3}
	for _, hash := range hashes {
		for _, term := range documents[hash].terms {
			if err := terms.append(ctx, tx, publicationString(hash), publicationInteger(int64(term.hash)), publicationInteger(int64(term.frequency))); err != nil {
				return err
			}
		}
	}
	return terms.flush(ctx, tx)
}

type publicationLexicalDelta struct {
	Hash  string `json:"hash"`
	Delta int64  `json:"delta"`
}

func publishLexicalDeltas(ctx context.Context, tx *sql.Tx, namespace string, deltas map[string]int64) error {
	changes := make([]publicationLexicalDelta, 0, len(deltas))
	for _, content := range sortedLexicalDeltas(deltas) {
		if content.delta != 0 {
			changes = append(changes, publicationLexicalDelta{Hash: content.searchHash, Delta: content.delta})
		}
	}
	if len(changes) == 0 {
		return nil
	}
	payload, err := json.Marshal(changes)
	if err != nil {
		slog.ErrorContext(ctx, "encode lexical publication deltas failed", "err", err)
		return fmt.Errorf("encode lexical publication deltas: %w", err)
	}
	var matched, sizeDelta, tokenDelta int64
	if err := tx.QueryRowContext(ctx, publicationLexicalTotalsStatement, string(payload)).Scan(&matched, &sizeDelta, &tokenDelta); err != nil {
		slog.ErrorContext(ctx, "read lexical publication totals failed", "namespace", namespace, "err", err)
		return fmt.Errorf("read lexical publication totals: %w", err)
	}
	if matched != int64(len(changes)) {
		return fmt.Errorf("lexical publication references %d contents, but only %d exist", len(changes), matched)
	}
	var invalid int64
	if err := tx.QueryRowContext(ctx, publicationLexicalDeltasStatement+publicationLexicalFrequencyCheckStatement, string(payload), namespace).Scan(&invalid); err != nil {
		slog.ErrorContext(ctx, "check lexical document frequencies failed", "namespace", namespace, "err", err)
		return fmt.Errorf("check lexical document frequencies: %w", err)
	}
	if invalid != 0 {
		return fmt.Errorf("lexical publication would make %d document frequencies negative", invalid)
	}
	for _, statement := range []string{publicationLexicalFrequencyDeleteStatement, publicationLexicalFrequencyUpsertStatement} {
		if _, err := tx.ExecContext(ctx, publicationLexicalDeltasStatement+statement, string(payload), namespace, namespace); err != nil {
			slog.ErrorContext(ctx, "publish lexical document frequencies failed", "namespace", namespace, "err", err)
			return fmt.Errorf("publish lexical document frequencies: %w", err)
		}
	}
	for _, statement := range []string{publicationLexicalUnusedTermsStatement, publicationLexicalUnusedContentStatement} {
		if _, err := tx.ExecContext(ctx, statement, string(payload)); err != nil {
			slog.ErrorContext(ctx, "remove unused lexical content failed", "err", err)
			return fmt.Errorf("remove unused lexical content: %w", err)
		}
	}
	return updateLexicalStats(ctx, tx, namespace, sizeDelta, tokenDelta)
}

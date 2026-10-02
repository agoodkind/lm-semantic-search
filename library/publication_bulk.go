package library

import (
	"context"
	"database/sql"
	_ "embed"
	"errors"
	"fmt"
	"strings"
)

const publicationInsertRows = 64

//go:embed publication_blobs.sql
var publicationBlobsStatement string

//go:embed publication_occurrences.sql
var publicationOccurrencesStatement string

//go:embed publication_scalars.sql
var publicationScalarsStatement string

//go:embed publication_owner_hashes.sql
var publicationOwnerHashesStatement string

//go:embed publication_receipt_rows.sql
var publicationReceiptRowsStatement string

//go:embed publication_lexical_occurrences.sql
var publicationLexicalOccurrencesStatement string

//go:embed publication_lexical_terms.sql
var publicationLexicalTermsStatement string

type publicationInsert struct {
	statement string
	columns   int
	arguments []any
	rows      int
	prepared  *sql.Stmt
}

func (insert *publicationInsert) append(ctx context.Context, tx *sql.Tx, arguments ...any) error {
	insert.arguments = append(insert.arguments, arguments...)
	insert.rows++
	if insert.rows == publicationInsertRows {
		return insert.flush(ctx, tx)
	}
	return nil
}

func (insert *publicationInsert) flush(ctx context.Context, tx *sql.Tx) error {
	if insert.rows == 0 {
		return nil
	}
	group := "(" + strings.TrimSuffix(strings.Repeat("?,", insert.columns), ",") + ")"
	values := strings.TrimSuffix(strings.Repeat(group+",", insert.rows), ",")
	statement := strings.ReplaceAll(insert.statement, "{{rows}}", values)
	if insert.rows == publicationInsertRows && insert.prepared == nil {
		prepared, err := tx.PrepareContext(ctx, statement)
		if err != nil {
			return fmt.Errorf("prepare publication rows: %w", err)
		}
		insert.prepared = prepared
	}
	var err error
	if insert.rows == publicationInsertRows {
		_, err = insert.prepared.ExecContext(ctx, insert.arguments...)
	} else {
		_, err = tx.ExecContext(ctx, statement, insert.arguments...)
	}
	if err != nil {
		return fmt.Errorf("insert publication rows: %w", err)
	}
	insert.arguments = insert.arguments[:0]
	insert.rows = 0
	return nil
}

type preparedPublicationRow struct {
	stagedRow
	blobID      string
	searchHash  string
	scalarNames []string
}

func preparePublication(rows []stagedRow) ([]preparedPublicationRow, map[string]lexicalDocument) {
	prepared := make([]preparedPublicationRow, 0, len(rows))
	documents := make(map[string]lexicalDocument)
	for _, row := range rows {
		hash := searchTextHash(row.occurrence.SearchText)
		prepared = append(prepared, preparedPublicationRow{
			stagedRow: row, blobID: sourceBlobID(row.occurrence.SourceText), searchHash: hash,
			scalarNames: sortedScalarNames(row.occurrence.Scalars),
		})
		if _, found := documents[hash]; !found {
			documents[hash] = analyzeLexical(row.occurrence.SearchText)
		}
	}
	return prepared, documents
}

func publishRows(ctx context.Context, tx *sql.Tx, key GenerationKey, rows []preparedPublicationRow) (_ []lexicalOccurrence, err error) {
	result, err := tx.QueryContext(ctx, publicationOwnerHashesStatement, key.Namespace, key.OwnerID)
	if err != nil {
		return nil, fmt.Errorf("read published occurrence hashes: %w", err)
	}
	saved := make(map[string]string)
	for result.Next() {
		var rowKey, hash string
		if err := result.Scan(&rowKey, &hash); err != nil {
			return nil, errors.Join(err, closeRows(ctx, result))
		}
		saved[rowKey] = hash
	}
	if err := errors.Join(result.Err(), closeRows(ctx, result)); err != nil {
		return nil, err
	}
	fresh := make([]preparedPublicationRow, 0, len(rows))
	added := make([]lexicalOccurrence, 0, len(rows))
	for _, row := range rows {
		occurrence := row.occurrence
		if hash, found := saved[occurrence.RowKey]; found {
			if hash != row.occurrenceHash {
				return nil, fmt.Errorf("%w: owner %q row %q exists with different content", ErrAppendConflict, key.OwnerID, occurrence.RowKey)
			}
			continue
		}
		fresh = append(fresh, row)
		added = append(added, lexicalOccurrence{OwnerID: key.OwnerID, RowKey: occurrence.RowKey, SearchHash: row.searchHash, SearchText: occurrence.SearchText})
	}
	blobs := publicationInsert{statement: publicationBlobsStatement, columns: 2}
	for _, row := range fresh {
		occurrence := row.occurrence
		if err := blobs.append(ctx, tx, row.blobID, occurrence.SourceText); err != nil {
			return nil, err
		}
	}
	if err := blobs.flush(ctx, tx); err != nil {
		return nil, err
	}
	occurrences := publicationInsert{statement: publicationOccurrencesStatement, columns: 10}
	for _, row := range fresh {
		occurrence := row.occurrence
		if err := occurrences.append(ctx, tx, key.Namespace, key.OwnerID, occurrence.RowKey, occurrence.SortKey, row.vectorID, row.blobID, row.searchHash, len(occurrence.SourceText), key.GenerationOrder, row.occurrenceHash); err != nil {
			return nil, err
		}
	}
	if err := occurrences.flush(ctx, tx); err != nil {
		return nil, err
	}
	scalars := publicationInsert{statement: publicationScalarsStatement, columns: 9}
	for _, row := range fresh {
		occurrence := row.occurrence
		for _, name := range row.scalarNames {
			value := occurrence.Scalars[name]
			typed := newTypedScalar(value)
			if err := scalars.append(ctx, tx, key.Namespace, key.OwnerID, occurrence.RowKey, name, value.Type, typed.stringValue, typed.int64Value, typed.boolValue, value.Null); err != nil {
				return nil, err
			}
		}
	}
	if err := scalars.flush(ctx, tx); err != nil {
		return nil, err
	}
	return added, nil
}

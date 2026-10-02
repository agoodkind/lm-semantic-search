package library

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log/slog"
)

// committedGeneration is the saved content of one committed generation token.
type committedGeneration struct {
	manifestHash string
	rowCount     uint64
	mode         BatchMode
}

// readCommittedGeneration returns the saved manifest hash, row count, and mode
// of a committed key.
func readCommittedGeneration(ctx context.Context, tx *sql.Tx, key GenerationKey) (committedGeneration, error) {
	var committed committedGeneration
	if err := tx.QueryRowContext(
		ctx,
		`SELECT batch_hash, row_count, mode FROM batch_receipts
		WHERE namespace = ? AND owner_id = ? AND generation_order = ? AND generation_token = ?`,
		key.Namespace, key.OwnerID, key.GenerationOrder, key.IdempotencyToken,
	).Scan(&committed.manifestHash, &committed.rowCount, &committed.mode); err != nil {
		slog.ErrorContext(ctx, "read committed generation failed", "namespace", key.Namespace, "err", err)
		return committedGeneration{}, fmt.Errorf("read committed generation: %w", err)
	}
	return committed, nil
}

// checkCommittedSeal compares seal with the saved manifest of a committed key.
// A different row count or manifest hash returns an error that wraps
// [ErrAppendConflict].
func checkCommittedSeal(ctx context.Context, tx *sql.Tx, key GenerationKey, seal GenerationSeal) error {
	committed, err := readCommittedGeneration(ctx, tx, key)
	if err != nil {
		return err
	}
	if committed.rowCount == seal.RowCount && committed.manifestHash == seal.ManifestHash {
		return nil
	}
	conflict := fmt.Errorf(
		"%w: owner %q order %d token %q was committed with %d rows and manifest %s, the replay seals %d rows and manifest %s",
		ErrAppendConflict, key.OwnerID, key.GenerationOrder, key.IdempotencyToken,
		committed.rowCount, committed.manifestHash, seal.RowCount, seal.ManifestHash,
	)
	slog.WarnContext(ctx, "committed generation replay conflict", "namespace", key.Namespace, "err", conflict)
	return conflict
}

// checkCommittedRows compares a staged batch with the saved rows of a
// committed key. A different mode, a row key the generation did not commit, or
// a row with different content returns an error that wraps
// [ErrAppendConflict].
func checkCommittedRows(ctx context.Context, tx *sql.Tx, key GenerationKey, mode BatchMode, rows []Occurrence) error {
	committed, err := readCommittedGeneration(ctx, tx, key)
	if err != nil {
		return err
	}
	if committed.mode != mode {
		conflict := fmt.Errorf("%w: owner %q order %d was committed in mode %d, the replay uses mode %d", ErrAppendConflict, key.OwnerID, key.GenerationOrder, committed.mode, mode)
		slog.WarnContext(ctx, "committed generation replay conflict", "namespace", key.Namespace, "err", conflict)
		return conflict
	}
	for _, row := range rows {
		_, occurrenceHash, err := encodeOccurrence(row)
		if err != nil {
			return err
		}
		var savedHash string
		scanErr := tx.QueryRowContext(
			ctx,
			`SELECT occurrence_hash FROM receipt_rows
			WHERE namespace = ? AND owner_id = ? AND generation_order = ? AND generation_token = ? AND row_key = ?`,
			key.Namespace, key.OwnerID, key.GenerationOrder, key.IdempotencyToken, row.RowKey,
		).Scan(&savedHash)
		if scanErr != nil && !errors.Is(scanErr, sql.ErrNoRows) {
			slog.ErrorContext(ctx, "read committed row failed", "row_key", row.RowKey, "err", scanErr)
			return fmt.Errorf("read committed row %q: %w", row.RowKey, scanErr)
		}
		if errors.Is(scanErr, sql.ErrNoRows) || savedHash != occurrenceHash {
			conflict := fmt.Errorf("%w: owner %q order %d did not commit row %q with this content", ErrAppendConflict, key.OwnerID, key.GenerationOrder, row.RowKey)
			slog.WarnContext(ctx, "committed generation replay conflict", "namespace", key.Namespace, "err", conflict)
			return conflict
		}
	}
	return nil
}

// saveReceiptRows saves the row key and occurrence hash of every row of a
// committed generation.
func saveReceiptRows(ctx context.Context, tx *sql.Tx, key GenerationKey) error {
	if _, err := tx.ExecContext(ctx, publicationReceiptRowsStatement,
		key.Namespace, key.OwnerID, key.GenerationOrder, key.IdempotencyToken,
	); err != nil {
		return fmt.Errorf("save committed rows: %w", err)
	}
	return nil
}

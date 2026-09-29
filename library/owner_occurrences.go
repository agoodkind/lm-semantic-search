package library

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log/slog"
)

// ListOwnerOccurrences returns the committed state, the latest saved
// projection order, and the published row keys of one owner in one read
// transaction, without the writer lock. Staged rows are not listed. An unknown
// owner returns the zero value. An unregistered namespace returns an error
// that wraps [ErrInvalidRequest].
func (library *Library) ListOwnerOccurrences(ctx context.Context, namespace string, ownerID string) (OwnerOccurrences, error) {
	var listed OwnerOccurrences
	err := library.read(ctx, func(tx *sql.Tx) error {
		if _, err := loadNamespace(ctx, tx, namespace); err != nil {
			return err
		}
		state, _, err := readOwnerState(ctx, tx, namespace, ownerID)
		if err != nil {
			return err
		}
		projectionOrder, err := readProjectionOrder(ctx, tx, namespace, ownerID)
		if err != nil {
			return err
		}
		rows, err := readOwnerOccurrenceRows(ctx, tx, namespace, ownerID)
		if err != nil {
			return err
		}
		listed = OwnerOccurrences{State: state, ProjectionOrder: projectionOrder, Rows: rows}
		return nil
	})
	if err != nil {
		return OwnerOccurrences{State: OwnerState{GenerationOrder: 0, IdempotencyToken: "", Fingerprint: ""}, ProjectionOrder: 0, Rows: nil}, err
	}
	return listed, nil
}

// readProjectionOrder returns the highest saved projection order of one
// owner, or zero.
func readProjectionOrder(ctx context.Context, tx *sql.Tx, namespace string, ownerID string) (uint64, error) {
	var order uint64
	if err := tx.QueryRowContext(
		ctx,
		`SELECT COALESCE(MAX(projection_order), 0) FROM projection_events WHERE namespace = ? AND owner_id = ?`,
		namespace, ownerID,
	).Scan(&order); err != nil {
		slog.ErrorContext(ctx, "read owner projection order failed", "namespace", namespace, "err", err)
		return 0, fmt.Errorf("read owner %q projection order: %w", ownerID, err)
	}
	return order, nil
}

// readOwnerOccurrenceRows returns the published row keys of one owner with
// their generation orders, sorted by row key.
func readOwnerOccurrenceRows(ctx context.Context, tx *sql.Tx, namespace string, ownerID string) (rows []OwnerOccurrence, err error) {
	result, queryErr := tx.QueryContext(
		ctx,
		`SELECT row_key, generation_order FROM occurrences WHERE namespace = ? AND owner_id = ? ORDER BY row_key`,
		namespace, ownerID,
	)
	if queryErr != nil {
		slog.ErrorContext(ctx, "list owner occurrences failed", "namespace", namespace, "err", queryErr)
		return nil, fmt.Errorf("list owner %q occurrences: %w", ownerID, queryErr)
	}
	defer func() {
		err = errors.Join(err, closeRows(ctx, result))
	}()
	for result.Next() {
		row := OwnerOccurrence{RowKey: "", GenerationOrder: 0}
		if err := result.Scan(&row.RowKey, &row.GenerationOrder); err != nil {
			slog.ErrorContext(ctx, "scan owner occurrence failed", "namespace", namespace, "err", err)
			return nil, fmt.Errorf("scan owner %q occurrence: %w", ownerID, err)
		}
		rows = append(rows, row)
	}
	if err := result.Err(); err != nil {
		slog.ErrorContext(ctx, "list owner occurrences failed", "namespace", namespace, "err", err)
		return nil, fmt.Errorf("list owner %q occurrences: %w", ownerID, err)
	}
	return rows, nil
}

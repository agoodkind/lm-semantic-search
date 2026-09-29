package library

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log/slog"
)

// NamespaceStats counts the published content of one namespace.
type NamespaceStats struct {
	// Owners is the number of owners with at least one published occurrence.
	Owners int64
	// Occurrences is the number of published occurrences.
	Occurrences int64
}

// NamespaceStats returns the published owner and occurrence counts of
// namespace. An unregistered namespace returns an error that wraps
// [ErrInvalidRequest].
func (library *Library) NamespaceStats(ctx context.Context, namespace string) (NamespaceStats, error) {
	var stats NamespaceStats
	err := library.read(ctx, func(tx *sql.Tx) error {
		if _, err := loadNamespace(ctx, tx, namespace); err != nil {
			return err
		}
		if err := tx.QueryRowContext(
			ctx,
			`SELECT COUNT(DISTINCT owner_id), COUNT(*) FROM occurrences WHERE namespace = ?`,
			namespace,
		).Scan(&stats.Owners, &stats.Occurrences); err != nil {
			slog.ErrorContext(ctx, "count namespace occurrences failed", "namespace", namespace, "err", err)
			return fmt.Errorf("count namespace %q occurrences: %w", namespace, err)
		}
		return nil
	})
	return stats, err
}

// ListOwners returns the owner IDs of namespace with at least one published
// occurrence, sorted. An unregistered namespace returns an error that wraps
// [ErrInvalidRequest].
func (library *Library) ListOwners(ctx context.Context, namespace string) ([]string, error) {
	var owners []string
	err := library.read(ctx, func(tx *sql.Tx) (err error) {
		if _, err := loadNamespace(ctx, tx, namespace); err != nil {
			return err
		}
		rows, err := tx.QueryContext(
			ctx,
			`SELECT DISTINCT owner_id FROM occurrences WHERE namespace = ? ORDER BY owner_id`,
			namespace,
		)
		if err != nil {
			slog.ErrorContext(ctx, "list namespace owners failed", "namespace", namespace, "err", err)
			return fmt.Errorf("list namespace %q owners: %w", namespace, err)
		}
		defer func() {
			err = errors.Join(err, closeRows(ctx, rows))
		}()
		for rows.Next() {
			var owner string
			if err := rows.Scan(&owner); err != nil {
				slog.ErrorContext(ctx, "scan namespace owner failed", "namespace", namespace, "err", err)
				return fmt.Errorf("scan namespace %q owner: %w", namespace, err)
			}
			owners = append(owners, owner)
		}
		if err := rows.Err(); err != nil {
			slog.ErrorContext(ctx, "list namespace owners failed", "namespace", namespace, "err", err)
			return fmt.Errorf("list namespace %q owners: %w", namespace, err)
		}
		return nil
	})
	return owners, err
}

package library

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
)

// ReprojectScalars changes declared mutable columns for exact published row
// keys of one owner. It saves a projection event and the effective values in
// one transaction and writes no vector. A known projection token returns its
// saved receipt. An unknown order at or below the owner's latest projection
// returns an error that wraps [ErrStaleGeneration]. A reused order with
// another token or payload returns an error that wraps [ErrAppendConflict].
func (library *Library) ReprojectScalars(ctx context.Context, projection ScalarProjection) (_ ProjectionReceipt, err error) {
	if projection.Namespace == "" || projection.OwnerID == "" || projection.IdempotencyToken == "" || projection.ProjectionOrder == 0 {
		return ProjectionReceipt{}, invalidRequest("reproject: namespace, owner ID, idempotency token, and a positive order are required")
	}
	rows := make(map[string]map[string]scalarValue, len(projection.Rows))
	for rowKey, values := range projection.Rows {
		encoded := make(map[string]scalarValue, len(values))
		for name, value := range values {
			encoded[name] = scalarValue(value)
		}
		rows[rowKey] = encoded
	}
	payload, err := json.Marshal(rows)
	if err != nil {
		slog.ErrorContext(ctx, "encode scalar projection failed", "namespace", projection.Namespace, "err", err)
		return ProjectionReceipt{}, fmt.Errorf("encode scalar projection: %w", err)
	}
	payloadHash := hexSHA256(payload)
	receipt := ProjectionReceipt{
		Namespace:       projection.Namespace,
		OwnerID:         projection.OwnerID,
		ProjectionOrder: projection.ProjectionOrder,
		Fingerprint: hexSHA256([]byte(strings.Join([]string{
			"projection",
			projection.Namespace,
			projection.OwnerID,
			strconv.FormatUint(projection.ProjectionOrder, 10),
			projection.IdempotencyToken,
			payloadHash,
		}, "\x00"))),
	}

	release, err := library.lock.acquire(ctx)
	if err != nil {
		return ProjectionReceipt{}, err
	}
	defer func() {
		err = errors.Join(err, release())
	}()
	err = library.write(ctx, func(tx *sql.Tx) error {
		saved, err := checkProjection(ctx, tx, projection, payloadHash)
		if err != nil {
			return err
		}
		if saved != nil {
			receipt = *saved
			return nil
		}
		spec, err := loadNamespace(ctx, tx, projection.Namespace)
		if err != nil {
			return err
		}
		if err := validateProjectionRows(ctx, tx, spec, projection); err != nil {
			return err
		}
		return saveProjection(ctx, tx, projection, payload, payloadHash, receipt.Fingerprint)
	})
	if err != nil {
		return ProjectionReceipt{}, err
	}
	return receipt, nil
}

// checkProjection returns the saved receipt of a replayed projection.
func checkProjection(ctx context.Context, tx *sql.Tx, projection ScalarProjection, payloadHash string) (*ProjectionReceipt, error) {
	var token, savedHash, fingerprint string
	err := tx.QueryRowContext(
		ctx,
		`SELECT token, payload_hash, fingerprint FROM projection_events WHERE namespace = ? AND owner_id = ? AND projection_order = ?`,
		projection.Namespace, projection.OwnerID, projection.ProjectionOrder,
	).Scan(&token, &savedHash, &fingerprint)
	if err == nil {
		if token != projection.IdempotencyToken || savedHash != payloadHash {
			err := fmt.Errorf("%w: projection order %d is saved with another token or payload", ErrAppendConflict, projection.ProjectionOrder)
			slog.WarnContext(ctx, "projection conflict", "namespace", projection.Namespace, "err", err)
			return nil, err
		}
		return &ProjectionReceipt{
			Namespace:       projection.Namespace,
			OwnerID:         projection.OwnerID,
			ProjectionOrder: projection.ProjectionOrder,
			Fingerprint:     fingerprint,
		}, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		slog.ErrorContext(ctx, "read projection event failed", "namespace", projection.Namespace, "err", err)
		return nil, fmt.Errorf("read projection event: %w", err)
	}
	var newerExists bool
	if err := tx.QueryRowContext(
		ctx,
		`SELECT EXISTS (SELECT 1 FROM projection_events WHERE namespace = ? AND owner_id = ? AND projection_order > ?)`,
		projection.Namespace, projection.OwnerID, projection.ProjectionOrder,
	).Scan(&newerExists); err != nil {
		slog.ErrorContext(ctx, "read later projections failed", "namespace", projection.Namespace, "err", err)
		return nil, fmt.Errorf("read later projections: %w", err)
	}
	if newerExists {
		err := fmt.Errorf("%w: projection order %d is below a saved order", ErrStaleGeneration, projection.ProjectionOrder)
		slog.WarnContext(ctx, "stale projection rejected", "namespace", projection.Namespace, "err", err)
		return nil, err
	}
	return nil, nil
}

// validateProjectionRows requires declared mutable columns with valid values
// and a published occurrence for every row key.
func validateProjectionRows(ctx context.Context, tx *sql.Tx, spec NamespaceSpec, projection ScalarProjection) error {
	columns := make(map[string]ScalarColumn, len(spec.Scalars))
	for _, column := range spec.Scalars {
		columns[column.Name] = column
	}
	for rowKey, values := range projection.Rows {
		var exists int
		err := tx.QueryRowContext(
			ctx,
			`SELECT 1 FROM occurrences WHERE namespace = ? AND owner_id = ? AND row_key = ?`,
			projection.Namespace, projection.OwnerID, rowKey,
		).Scan(&exists)
		if errors.Is(err, sql.ErrNoRows) {
			return invalidRequest(fmt.Sprintf("reproject: owner %q has no published row %q", projection.OwnerID, rowKey))
		}
		if err != nil {
			slog.ErrorContext(ctx, "read projected occurrence failed", "row_key", rowKey, "err", err)
			return fmt.Errorf("read projected occurrence %q: %w", rowKey, err)
		}
		for name, value := range values {
			column, declared := columns[name]
			if !declared || !column.Mutable {
				return invalidRequest(fmt.Sprintf("reproject: column %q is not a declared mutable column of namespace %q", name, spec.ID))
			}
			if message := scalarValueViolation(column, value); message != "" {
				return invalidRequest("reproject: " + message)
			}
		}
	}
	return nil
}

func saveProjection(ctx context.Context, tx *sql.Tx, projection ScalarProjection, payload []byte, payloadHash string, fingerprint string) error {
	if _, err := tx.ExecContext(
		ctx,
		`INSERT INTO projection_events (namespace, owner_id, projection_order, token, payload_hash, payload, fingerprint) VALUES (?, ?, ?, ?, ?, ?, ?)`,
		projection.Namespace, projection.OwnerID, projection.ProjectionOrder, projection.IdempotencyToken, payloadHash, payload, fingerprint,
	); err != nil {
		slog.ErrorContext(ctx, "save projection event failed", "namespace", projection.Namespace, "err", err)
		return fmt.Errorf("save projection event: %w", err)
	}
	for rowKey, values := range projection.Rows {
		for _, name := range sortedScalarNames(values) {
			if err := upsertEffectiveScalar(
				ctx, tx, projection.Namespace, projection.OwnerID, rowKey, name, values[name], projection.ProjectionOrder,
			); err != nil {
				return err
			}
		}
	}
	_, err := incrementRevision(ctx, tx, identityKeyProjectionRevision)
	return err
}

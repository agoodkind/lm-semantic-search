package library

import (
	"context"
	"crypto/sha256"
	"database/sql"
	_ "embed"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
)

// CommittedOccurrence contains one published occurrence's content seals and
// effective scalars. Each seal is the lowercase SHA-256 of the exact original
// UTF-8 bytes saved at publication. The reader returns stored seals without
// rehashing content. Scalars include mutable projections with saved precedence.
type CommittedOccurrence struct {
	ID                   OccurrenceID
	GenerationOrder      uint64
	SortKey              string
	SourceSHA256         string
	SearchSHA256         string
	EmbeddingInputSHA256 string
	Scalars              map[string]ScalarValue
}

// CommittedOwnerSnapshot summarizes the same catalog snapshot as the visited
// occurrences. RowCount excludes staged rows.
type CommittedOwnerSnapshot struct {
	State           OwnerState
	ProjectionOrder uint64
	RowCount        uint64
}

//go:embed committed_owner.sql
var committedOwnerStatement string

// ReadCommittedOwner visits one owner's published occurrences in row-key order
// within one read transaction, without writer admission, embedding, or backend
// access. Each callback receives an independent scalar map. An unknown owner
// returns a zero snapshot. An unregistered namespace returns an error that wraps
// ErrInvalidRequest.
// The callback runs synchronously and must return promptly. Any failure returns
// a zero snapshot; callers must discard previously visited rows after an error.
func (library *Library) ReadCommittedOwner(ctx context.Context, namespace string, ownerID string, visit func(CommittedOccurrence) error) (CommittedOwnerSnapshot, error) {
	if namespace == "" || ownerID == "" || visit == nil {
		return CommittedOwnerSnapshot{}, invalidRequest("read committed owner requires a namespace, an owner ID, and a callback")
	}
	var snapshot CommittedOwnerSnapshot
	err := library.read(ctx, func(tx *sql.Tx) error {
		spec, err := loadNamespace(ctx, tx, namespace)
		if err != nil {
			return err
		}
		state, found, err := readOwnerState(ctx, tx, namespace, ownerID)
		if err != nil || !found {
			return err
		}
		projectionOrder, err := readProjectionOrder(ctx, tx, namespace, ownerID)
		if err != nil {
			return err
		}
		count, err := visitCommittedOwnerRows(ctx, tx, spec, ownerID, visit)
		if err != nil {
			return err
		}
		snapshot = CommittedOwnerSnapshot{State: state, ProjectionOrder: projectionOrder, RowCount: count}
		return nil
	})
	if err != nil {
		return CommittedOwnerSnapshot{}, err
	}
	if contextErr := ctx.Err(); contextErr != nil {
		slog.WarnContext(ctx, "committed owner read canceled", "err", contextErr)
		return CommittedOwnerSnapshot{}, fmt.Errorf("read committed owner: %w", contextErr)
	}
	return snapshot, nil
}

func visitCommittedOwnerRows(ctx context.Context, tx *sql.Tx, spec NamespaceSpec, ownerID string, visit func(CommittedOccurrence) error) (count uint64, err error) {
	rows, queryErr := tx.QueryContext(ctx, committedOwnerStatement, spec.ID, ownerID)
	if queryErr != nil {
		slog.ErrorContext(ctx, "read committed owner occurrences failed", "err", queryErr)
		return 0, fmt.Errorf("read committed owner occurrences: %w", queryErr)
	}
	defer func() { err = errors.Join(err, closeRows(ctx, rows)) }()
	for rows.Next() {
		row, decodeErr := scanCommittedOccurrence(ctx, rows, spec, ownerID)
		if decodeErr != nil {
			return 0, decodeErr
		}
		if contextErr := ctx.Err(); contextErr != nil {
			slog.WarnContext(ctx, "committed owner read canceled", "err", contextErr)
			return 0, fmt.Errorf("read committed owner: %w", contextErr)
		}
		if callbackErr := visit(row); callbackErr != nil {
			slog.WarnContext(ctx, "committed owner callback failed", "err", callbackErr)
			return 0, fmt.Errorf("visit committed occurrence: %w", callbackErr)
		}
		count++
	}
	if rowsErr := errors.Join(rows.Err(), ctx.Err()); rowsErr != nil {
		slog.ErrorContext(ctx, "committed owner iteration failed", "err", rowsErr)
		return 0, fmt.Errorf("iterate committed owner occurrences: %w", rowsErr)
	}
	return count, nil
}

func scanCommittedOccurrence(ctx context.Context, rows *sql.Rows, spec NamespaceSpec, ownerID string) (CommittedOccurrence, error) {
	row := CommittedOccurrence{ID: OccurrenceID{Namespace: spec.ID, OwnerID: ownerID, RowKey: ""}, GenerationOrder: 0, SortKey: "", SourceSHA256: "", SearchSHA256: "", EmbeddingInputSHA256: "", Scalars: nil}
	var encoded string
	if err := rows.Scan(&row.ID.RowKey, &row.GenerationOrder, &row.SortKey, &row.SourceSHA256, &row.SearchSHA256, &row.EmbeddingInputSHA256, &encoded); err != nil {
		slog.ErrorContext(ctx, "scan committed occurrence failed", "err", err)
		return CommittedOccurrence{}, fmt.Errorf("scan committed occurrence: %w", err)
	}
	for _, seal := range []string{row.SourceSHA256, row.SearchSHA256, row.EmbeddingInputSHA256} {
		decoded, err := hex.DecodeString(seal)
		if err != nil || len(decoded) != sha256.Size || hex.EncodeToString(decoded) != seal {
			return CommittedOccurrence{}, invalidRequest("committed occurrence has an invalid content seal")
		}
	}
	scalars, err := decodeEffectiveScalars(encoded)
	if err != nil {
		return CommittedOccurrence{}, err
	}
	columns := make(map[string]ScalarColumn, len(spec.Scalars))
	for _, column := range spec.Scalars {
		columns[column.Name] = column
	}
	for name, value := range scalars {
		column, found := columns[name]
		if !found {
			return CommittedOccurrence{}, invalidRequest("committed occurrence contains an undeclared scalar")
		}
		if violation := scalarValueViolation(column, value); violation != "" {
			return CommittedOccurrence{}, invalidRequest(violation)
		}
	}
	row.Scalars = scalars
	return row, nil
}

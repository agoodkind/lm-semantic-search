package library

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/library/internal/vectorcodec"
)

// outboxEntry is one vector write that a writer persisted before its backend
// write and verification finished.
type outboxEntry struct {
	vectorID string
	digest   string
	checksum string
	payload  []byte
}

// replayOutbox finishes every persisted vector write. For each entry it reads
// the backend vector with a strong read. A missing vector is written again
// with the exact saved bytes and read again. A verified vector is marked
// verified. A checksum or digest mismatch returns an error that wraps
// [ErrVectorCorrupt]. The caller owns the writer lock.
func (library *Library) replayOutbox(ctx context.Context) error {
	entries, err := library.readOutbox(ctx)
	if err != nil {
		return err
	}
	if len(entries) == 0 {
		return nil
	}
	slog.InfoContext(ctx, "replay vector outbox", "entries", len(entries))
	identities := make([]VectorIdentity, 0, len(entries))
	for _, entry := range entries {
		identity := VectorIdentity{ID: entry.vectorID, IdentityDigest: entry.digest, Checksum: entry.checksum}
		verifyErr := library.config.Vectors.VerifyStrong(ctx, []VectorIdentity{identity})
		if errors.Is(verifyErr, ErrVectorMissing) {
			if err := library.rewriteOutboxEntry(ctx, entry); err != nil {
				return err
			}
		} else if verifyErr != nil {
			slog.ErrorContext(ctx, "verify outbox vector failed", "vector_id", entry.vectorID, "err", verifyErr)
			return fmt.Errorf("verify outbox vector %s: %w", entry.vectorID, verifyErr)
		}
		identities = append(identities, identity)
	}
	return library.verifyAndMark(ctx, identities)
}

// rewriteOutboxEntry writes the saved payload bytes of one entry again.
func (library *Library) rewriteOutboxEntry(ctx context.Context, entry outboxEntry) error {
	values, err := vectorcodec.Decode(entry.payload)
	if err != nil {
		slog.ErrorContext(ctx, "decode outbox vector failed", "vector_id", entry.vectorID, "err", err)
		return fmt.Errorf("%w: decode outbox vector %s: %w", ErrVectorCorrupt, entry.vectorID, err)
	}
	if vectorcodec.Checksum(values) != entry.checksum {
		err := fmt.Errorf("%w: outbox vector %s bytes do not match the saved checksum", ErrVectorCorrupt, entry.vectorID)
		slog.ErrorContext(ctx, "outbox vector checksum mismatch", "vector_id", entry.vectorID, "err", err)
		return err
	}
	record := VectorRecord{ID: entry.vectorID, IdentityDigest: entry.digest, Checksum: entry.checksum, Values: values}
	if err := library.config.Vectors.PutCanonical(ctx, record); err != nil {
		slog.ErrorContext(ctx, "rewrite outbox vector failed", "vector_id", entry.vectorID, "err", err)
		return fmt.Errorf("rewrite outbox vector %s: %w", entry.vectorID, err)
	}
	return nil
}

func (library *Library) readOutbox(ctx context.Context) ([]outboxEntry, error) {
	var entries []outboxEntry
	err := library.write(ctx, func(tx *sql.Tx) (err error) {
		rows, err := tx.QueryContext(
			ctx,
			`SELECT outbox.vector_id, vectors.identity_digest, outbox.payload_hash, outbox.vector_payload
			FROM vector_outbox AS outbox JOIN vectors ON vectors.vector_id = outbox.vector_id
			ORDER BY outbox.write_generation, outbox.vector_id`,
		)
		if err != nil {
			slog.ErrorContext(ctx, "read vector outbox failed", "err", err)
			return fmt.Errorf("read vector outbox: %w", err)
		}
		defer func() {
			err = errors.Join(err, closeRows(ctx, rows))
		}()
		for rows.Next() {
			var entry outboxEntry
			if err := rows.Scan(&entry.vectorID, &entry.digest, &entry.checksum, &entry.payload); err != nil {
				slog.ErrorContext(ctx, "scan vector outbox failed", "err", err)
				return fmt.Errorf("scan vector outbox: %w", err)
			}
			entries = append(entries, entry)
		}
		if err := rows.Err(); err != nil {
			slog.ErrorContext(ctx, "read vector outbox failed", "err", err)
			return fmt.Errorf("read vector outbox: %w", err)
		}
		return nil
	})
	return entries, err
}

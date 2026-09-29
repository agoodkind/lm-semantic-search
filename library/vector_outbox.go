package library

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/library/internal/vectorcodec"
)

// Vector states in the vectors table.
const (
	vectorStatePending  = "pending"
	vectorStateVerified = "verified"
)

// identityKeyVectorWriteGeneration counts outbox write batches.
const identityKeyVectorWriteGeneration = "vector_write_generation"

// embeddedVector is one new vector from the embedder with its identity.
type embeddedVector struct {
	identity vectorIdentity
	values   []float32
	checksum string
}

// storedVector is the catalog row of one canonical vector.
type storedVector struct {
	id            string
	digest        string
	inputBytes    []byte
	modelIdentity string
	checksum      string
	state         string
}

// lookupVectors returns the catalog rows for the requested vector IDs.
func lookupVectors(ctx context.Context, tx *sql.Tx, ids []string) (found map[string]storedVector, err error) {
	statement, err := tx.PrepareContext(
		ctx,
		`SELECT vector_id, identity_digest, input_bytes, model_identity, vector_checksum, state FROM vectors WHERE vector_id = ?`,
	)
	if err != nil {
		slog.ErrorContext(ctx, "prepare catalog vector read failed", "err", err)
		return nil, fmt.Errorf("prepare catalog vector read: %w", err)
	}
	defer func() {
		err = errors.Join(err, closeStatement(ctx, statement))
	}()
	found = make(map[string]storedVector, len(ids))
	for _, id := range ids {
		var vector storedVector
		scanErr := statement.QueryRowContext(ctx, id).Scan(&vector.id, &vector.digest, &vector.inputBytes, &vector.modelIdentity, &vector.checksum, &vector.state)
		if errors.Is(scanErr, sql.ErrNoRows) {
			continue
		}
		if scanErr != nil {
			slog.ErrorContext(ctx, "read catalog vector failed", "vector_id", id, "err", scanErr)
			return nil, fmt.Errorf("read catalog vector %s: %w", id, scanErr)
		}
		found[vector.id] = vector
	}
	return found, nil
}

func closeStatement(ctx context.Context, statement *sql.Stmt) error {
	if err := statement.Close(); err != nil {
		slog.ErrorContext(ctx, "close catalog statement failed", "err", err)
		return fmt.Errorf("close catalog statement: %w", err)
	}
	return nil
}

// missingIdentities returns the identities without a catalog vector. An
// existing vector with different identity bytes returns an error that wraps
// [ErrVectorCorrupt].
func missingIdentities(ctx context.Context, tx *sql.Tx, identities []vectorIdentity) ([]vectorIdentity, error) {
	ids := make([]string, 0, len(identities))
	for _, identity := range identities {
		ids = append(ids, identity.id)
	}
	existing, err := lookupVectors(ctx, tx, ids)
	if err != nil {
		return nil, err
	}
	missing := make([]vectorIdentity, 0, len(identities))
	for _, identity := range identities {
		stored, found := existing[identity.id]
		if !found {
			missing = append(missing, identity)
			continue
		}
		if stored.digest != identity.digest || stored.modelIdentity != identity.modelIdentity || string(stored.inputBytes) != string(identity.inputBytes) {
			err := fmt.Errorf("%w: vector %s has a different saved identity", ErrVectorCorrupt, identity.id)
			slog.ErrorContext(ctx, "vector identity collision", "vector_id", identity.id, "err", err)
			return nil, err
		}
	}
	return missing, nil
}

// embedIdentities embeds the exact input of every identity in batches of at
// most MaxBatchRows inputs and validates every returned vector.
func (library *Library) embedIdentities(ctx context.Context, identities []vectorIdentity) ([]embeddedVector, error) {
	embedded := make([]embeddedVector, 0, len(identities))
	batchRows := library.config.MaxBatchRows
	for start := 0; start < len(identities); start += batchRows {
		end := min(start+batchRows, len(identities))
		batch := identities[start:end]
		inputs := make([]string, 0, len(batch))
		for _, identity := range batch {
			inputs = append(inputs, string(identity.inputBytes))
		}
		vectors, err := library.config.Embedder.EmbedBatch(ctx, inputs)
		if err != nil {
			slog.ErrorContext(ctx, "embed occurrence inputs failed", "inputs", len(inputs), "err", err)
			return nil, fmt.Errorf("embed %d occurrence inputs: %w", len(inputs), err)
		}
		if len(vectors) != len(batch) {
			err := fmt.Errorf("embedder returned %d vectors for %d inputs", len(vectors), len(batch))
			slog.ErrorContext(ctx, "embed occurrence inputs failed", "err", err)
			return nil, err
		}
		for index, identity := range batch {
			if err := vectorcodec.Validate(vectors[index], library.config.Store.Dimension); err != nil {
				slog.ErrorContext(ctx, "embedder returned an invalid vector", "vector_id", identity.id, "err", err)
				return nil, fmt.Errorf("embedder vector for %s: %w", identity.id, err)
			}
			embedded = append(embedded, embeddedVector{
				identity: identity,
				values:   vectors[index],
				checksum: vectorcodec.Checksum(vectors[index]),
			})
		}
	}
	return embedded, nil
}

// publishVectors persists every vector in the outbox, writes it to the vector
// backend, verifies it with a strong read, and marks it verified. The caller
// owns the writer lock. A vector that another writer saved after the caller
// embedded it is skipped.
func (library *Library) publishVectors(ctx context.Context, vectors []embeddedVector) error {
	if len(vectors) == 0 {
		return nil
	}
	var pending []embeddedVector
	err := library.write(ctx, func(tx *sql.Tx) error {
		identities := make([]vectorIdentity, 0, len(vectors))
		for _, vector := range vectors {
			identities = append(identities, vector.identity)
		}
		missing, err := missingIdentities(ctx, tx, identities)
		if err != nil {
			return err
		}
		missingIDs := make(map[string]bool, len(missing))
		for _, identity := range missing {
			missingIDs[identity.id] = true
		}
		generation, err := incrementRevision(ctx, tx, identityKeyVectorWriteGeneration)
		if err != nil {
			return err
		}
		pending = pending[:0]
		for _, vector := range vectors {
			if !missingIDs[vector.identity.id] {
				continue
			}
			if err := insertOutboxVector(ctx, tx, library.config.Store, vector, generation); err != nil {
				return err
			}
			pending = append(pending, vector)
		}
		return nil
	})
	if err != nil {
		return err
	}
	for _, vector := range pending {
		record := VectorRecord{
			ID:             vector.identity.id,
			IdentityDigest: vector.identity.digest,
			Checksum:       vector.checksum,
			Values:         vector.values,
		}
		if err := library.config.Vectors.PutCanonical(ctx, record); err != nil {
			slog.ErrorContext(ctx, "write canonical vector failed", "vector_id", record.ID, "err", err)
			return fmt.Errorf("write canonical vector %s: %w", record.ID, err)
		}
	}
	identities := make([]VectorIdentity, 0, len(pending))
	for _, vector := range pending {
		identities = append(identities, VectorIdentity{ID: vector.identity.id, IdentityDigest: vector.identity.digest, Checksum: vector.checksum})
	}
	return library.verifyAndMark(ctx, identities)
}

func insertOutboxVector(ctx context.Context, tx *sql.Tx, descriptor StoreDescriptor, vector embeddedVector, generation int64) error {
	identity := vector.identity
	if _, err := tx.ExecContext(
		ctx,
		`INSERT INTO vectors (vector_id, identity_digest, input_hash, input_bytes, model_identity, dimension, normalization, vector_checksum, state, generation)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		identity.id,
		identity.digest,
		identity.inputHash,
		identity.inputBytes,
		identity.modelIdentity,
		descriptor.Dimension,
		descriptor.Normalization,
		vector.checksum,
		vectorStatePending,
		generation,
	); err != nil {
		slog.ErrorContext(ctx, "save pending vector failed", "vector_id", identity.id, "err", err)
		return fmt.Errorf("save pending vector %s: %w", identity.id, err)
	}
	if _, err := tx.ExecContext(
		ctx,
		`INSERT INTO vector_outbox (vector_id, vector_payload, payload_hash, write_generation, state) VALUES (?, ?, ?, ?, ?)`,
		identity.id,
		vectorcodec.Encode(vector.values),
		vector.checksum,
		generation,
		vectorStatePending,
	); err != nil {
		slog.ErrorContext(ctx, "save vector outbox entry failed", "vector_id", identity.id, "err", err)
		return fmt.Errorf("save vector outbox entry %s: %w", identity.id, err)
	}
	return nil
}

// verifyAndMark reads every vector back with a strong read, then marks it
// verified and removes its outbox entry in one transaction.
func (library *Library) verifyAndMark(ctx context.Context, identities []VectorIdentity) error {
	if len(identities) == 0 {
		return nil
	}
	if err := library.verifyStrong(ctx, identities); err != nil {
		return err
	}
	return library.write(ctx, func(tx *sql.Tx) error {
		for _, identity := range identities {
			if err := markVectorVerified(ctx, tx, identity.ID); err != nil {
				return err
			}
		}
		return nil
	})
}

// verifyStrong verifies identities in blocks of at most QueryBlockSize.
func (library *Library) verifyStrong(ctx context.Context, identities []VectorIdentity) error {
	blockSize := library.config.QueryBlockSize
	for start := 0; start < len(identities); start += blockSize {
		end := min(start+blockSize, len(identities))
		if err := library.config.Vectors.VerifyStrong(ctx, identities[start:end]); err != nil {
			slog.ErrorContext(ctx, "verify canonical vectors failed", "vectors", end-start, "err", err)
			return fmt.Errorf("verify %d canonical vectors: %w", end-start, err)
		}
	}
	return nil
}

func markVectorVerified(ctx context.Context, tx *sql.Tx, vectorID string) error {
	if _, err := tx.ExecContext(ctx, `UPDATE vectors SET state = ? WHERE vector_id = ?`, vectorStateVerified, vectorID); err != nil {
		slog.ErrorContext(ctx, "mark vector verified failed", "vector_id", vectorID, "err", err)
		return fmt.Errorf("mark vector %s verified: %w", vectorID, err)
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM vector_outbox WHERE vector_id = ?`, vectorID); err != nil {
		slog.ErrorContext(ctx, "remove vector outbox entry failed", "vector_id", vectorID, "err", err)
		return fmt.Errorf("remove vector outbox entry %s: %w", vectorID, err)
	}
	return nil
}

func closeRows(ctx context.Context, rows *sql.Rows) error {
	if err := rows.Close(); err != nil {
		slog.ErrorContext(ctx, "close catalog rows failed", "err", err)
		return fmt.Errorf("close catalog rows: %w", err)
	}
	return nil
}

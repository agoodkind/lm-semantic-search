package library

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log/slog"
	"sort"
)

// Staged generation states.
const stagedStateOpen = "open"

// RegisterNamespace saves a namespace declaration. An identical declaration
// succeeds again. A changed declaration for a saved ID returns an error that
// wraps [ErrInvalidRequest].
func (library *Library) RegisterNamespace(ctx context.Context, spec NamespaceSpec) (err error) {
	if err := spec.Validate(); err != nil {
		return err
	}
	declaration, err := encodeNamespace(spec)
	if err != nil {
		return err
	}
	release, err := library.lock.acquire(ctx)
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(err, release())
	}()
	return library.write(ctx, func(tx *sql.Tx) error {
		var saved string
		scanErr := tx.QueryRowContext(ctx, `SELECT declaration FROM namespaces WHERE id = ?`, spec.ID).Scan(&saved)
		if errors.Is(scanErr, sql.ErrNoRows) {
			if _, err := tx.ExecContext(ctx, `INSERT INTO namespaces (id, declaration) VALUES (?, ?)`, spec.ID, declaration); err != nil {
				slog.ErrorContext(ctx, "save namespace declaration failed", "namespace", spec.ID, "err", err)
				return fmt.Errorf("save namespace %q declaration: %w", spec.ID, err)
			}
			return nil
		}
		if scanErr != nil {
			slog.ErrorContext(ctx, "read namespace declaration failed", "namespace", spec.ID, "err", scanErr)
			return fmt.Errorf("read namespace %q declaration: %w", spec.ID, scanErr)
		}
		if saved != declaration {
			return invalidRequest(fmt.Sprintf("namespace %q is registered with a different declaration", spec.ID))
		}
		return nil
	})
}

// loadNamespace returns the saved declaration of id. An unknown namespace
// returns an error that wraps [ErrInvalidRequest].
func loadNamespace(ctx context.Context, tx *sql.Tx, id string) (NamespaceSpec, error) {
	var declaration string
	err := tx.QueryRowContext(ctx, `SELECT declaration FROM namespaces WHERE id = ?`, id).Scan(&declaration)
	if errors.Is(err, sql.ErrNoRows) {
		return NamespaceSpec{}, invalidRequest(fmt.Sprintf("namespace %q is not registered", id))
	}
	if err != nil {
		slog.ErrorContext(ctx, "read namespace declaration failed", "namespace", id, "err", err)
		return NamespaceSpec{}, fmt.Errorf("read namespace %q declaration: %w", id, err)
	}
	return decodeNamespace(declaration)
}

// Stage embeds, persists, and verifies the vectors of one bounded batch and
// adds its rows to a staged generation. No staged row is searchable before
// CommitGeneration publishes the generation. A batch for a committed
// generation token succeeds without change when its mode and every row match
// the committed generation, and returns an error that wraps
// [ErrAppendConflict] otherwise.
func (library *Library) Stage(ctx context.Context, batch StageBatch) (err error) {
	if err := library.validateStageBatch(batch); err != nil {
		return err
	}
	identities, committed, err := library.prepareStage(ctx, batch)
	if err != nil || committed {
		return err
	}
	var missing []vectorIdentity
	if err := library.read(ctx, func(tx *sql.Tx) error {
		var readErr error
		missing, readErr = missingIdentities(ctx, tx, identities)
		return readErr
	}); err != nil {
		return err
	}
	embedded, err := library.embedIdentities(ctx, missing)
	if err != nil {
		return err
	}

	release, err := library.lock.acquire(ctx)
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(err, release())
	}()
	if err := library.replayOutbox(ctx); err != nil {
		return err
	}
	if err := library.publishVectors(ctx, embedded); err != nil {
		return err
	}
	return library.write(ctx, func(tx *sql.Tx) error {
		spec, err := loadNamespace(ctx, tx, batch.Key.Namespace)
		if err != nil {
			return err
		}
		if err := checkNamespaceMode(spec, batch.Mode); err != nil {
			return err
		}
		receipt, err := checkGeneration(ctx, tx, batch.Key)
		if err != nil {
			return err
		}
		if receipt != nil {
			return checkCommittedRows(ctx, tx, batch.Key, batch.Mode, batch.Rows)
		}
		return stageRows(ctx, tx, library.config.Store, batch)
	})
}

// prepareStage validates the rows against the namespace and returns the
// distinct vector identities of the batch. It reports whether the generation
// token is already committed.
func (library *Library) prepareStage(ctx context.Context, batch StageBatch) ([]vectorIdentity, bool, error) {
	var identities []vectorIdentity
	committed := false
	err := library.read(ctx, func(tx *sql.Tx) error {
		spec, err := loadNamespace(ctx, tx, batch.Key.Namespace)
		if err != nil {
			return err
		}
		if err := checkNamespaceMode(spec, batch.Mode); err != nil {
			return err
		}
		for _, row := range batch.Rows {
			if err := spec.ValidateOccurrence(row); err != nil {
				return err
			}
		}
		receipt, err := checkGeneration(ctx, tx, batch.Key)
		if err != nil {
			return err
		}
		committed = receipt != nil
		if committed {
			return checkCommittedRows(ctx, tx, batch.Key, batch.Mode, batch.Rows)
		}
		identities = distinctIdentities(library.config.Store, batch.Rows)
		return nil
	})
	return identities, committed, err
}

func (library *Library) validateStageBatch(batch StageBatch) error {
	if err := validateGenerationKey(batch.Key); err != nil {
		return err
	}
	if batch.Mode != Append && batch.Mode != Replace {
		return invalidRequest(fmt.Sprintf("stage: mode %d is neither Append nor Replace", batch.Mode))
	}
	if len(batch.Rows) > library.config.MaxBatchRows {
		return invalidRequest(fmt.Sprintf("stage: %d rows exceed MaxBatchRows %d", len(batch.Rows), library.config.MaxBatchRows))
	}
	var batchBytes int64
	seen := make(map[string]bool, len(batch.Rows))
	for _, row := range batch.Rows {
		if seen[row.RowKey] {
			return invalidRequest(fmt.Sprintf("stage: row key %q appears twice in one batch", row.RowKey))
		}
		seen[row.RowKey] = true
		batchBytes += int64(len(row.SourceText) + len(row.SearchText) + len(row.EmbeddingInput))
	}
	if batchBytes > library.config.MaxBatchBytes {
		return invalidRequest(fmt.Sprintf("stage: %d text bytes exceed MaxBatchBytes %d", batchBytes, library.config.MaxBatchBytes))
	}
	return nil
}

func validateGenerationKey(key GenerationKey) error {
	if key.Namespace == "" || key.OwnerID == "" || key.IdempotencyToken == "" {
		return invalidRequest("generation key requires a namespace, an owner ID, and an idempotency token")
	}
	if key.GenerationOrder == 0 {
		return invalidRequest("generation key order must be positive")
	}
	return nil
}

// checkNamespaceMode rejects a replacement in an AppendOnly namespace.
func checkNamespaceMode(spec NamespaceSpec, mode BatchMode) error {
	if spec.Policy == AppendOnly && mode == Replace {
		return invalidRequest(fmt.Sprintf("namespace %q is AppendOnly and rejects Replace", spec.ID))
	}
	return nil
}

func distinctIdentities(descriptor StoreDescriptor, rows []Occurrence) []vectorIdentity {
	seen := make(map[string]bool, len(rows))
	identities := make([]vectorIdentity, 0, len(rows))
	for _, row := range rows {
		identity := newVectorIdentity(descriptor, row.EmbeddingInput)
		if seen[identity.id] {
			continue
		}
		seen[identity.id] = true
		identities = append(identities, identity)
	}
	return identities
}

// checkGeneration returns the saved receipt when key is already committed. An
// unknown key below the owner's committed order returns an error that wraps
// [ErrStaleGeneration]. An unknown key at the committed order returns an error
// that wraps [ErrAppendConflict].
func checkGeneration(ctx context.Context, tx *sql.Tx, key GenerationKey) (*ApplyReceipt, error) {
	var fingerprint string
	err := tx.QueryRowContext(
		ctx,
		`SELECT fingerprint FROM batch_receipts WHERE namespace = ? AND owner_id = ? AND generation_order = ? AND generation_token = ?`,
		key.Namespace, key.OwnerID, key.GenerationOrder, key.IdempotencyToken,
	).Scan(&fingerprint)
	if err == nil {
		return &ApplyReceipt{Namespace: key.Namespace, OwnerID: key.OwnerID, GenerationOrder: key.GenerationOrder, Fingerprint: fingerprint}, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		slog.ErrorContext(ctx, "read generation receipt failed", "namespace", key.Namespace, "err", err)
		return nil, fmt.Errorf("read generation receipt: %w", err)
	}
	state, found, err := readOwnerState(ctx, tx, key.Namespace, key.OwnerID)
	if err != nil || !found {
		return nil, err
	}
	if key.GenerationOrder < state.GenerationOrder {
		err := fmt.Errorf("%w: owner %q committed order %d, request order %d", ErrStaleGeneration, key.OwnerID, state.GenerationOrder, key.GenerationOrder)
		slog.WarnContext(ctx, "stale generation rejected", "namespace", key.Namespace, "err", err)
		return nil, err
	}
	if key.GenerationOrder == state.GenerationOrder {
		err := fmt.Errorf("%w: owner %q order %d is committed with another token", ErrAppendConflict, key.OwnerID, key.GenerationOrder)
		slog.WarnContext(ctx, "generation token conflict", "namespace", key.Namespace, "err", err)
		return nil, err
	}
	return nil, nil
}

// stageRows saves the staged generation and its rows. A replayed row with
// identical content succeeds. A row key with different content, or a staged
// generation with another mode, returns an error that wraps
// [ErrAppendConflict].
func stageRows(ctx context.Context, tx *sql.Tx, descriptor StoreDescriptor, batch StageBatch) error {
	key := batch.Key
	var savedMode BatchMode
	err := tx.QueryRowContext(
		ctx,
		`SELECT mode FROM staged_generations WHERE namespace = ? AND owner_id = ? AND generation_order = ? AND generation_token = ?`,
		key.Namespace, key.OwnerID, key.GenerationOrder, key.IdempotencyToken,
	).Scan(&savedMode)
	switch {
	case errors.Is(err, sql.ErrNoRows):
		if _, err := tx.ExecContext(
			ctx,
			`INSERT INTO staged_generations (namespace, owner_id, generation_order, generation_token, mode, state) VALUES (?, ?, ?, ?, ?, ?)`,
			key.Namespace, key.OwnerID, key.GenerationOrder, key.IdempotencyToken, batch.Mode, stagedStateOpen,
		); err != nil {
			slog.ErrorContext(ctx, "save staged generation failed", "namespace", key.Namespace, "err", err)
			return fmt.Errorf("save staged generation: %w", err)
		}
	case err != nil:
		slog.ErrorContext(ctx, "read staged generation failed", "namespace", key.Namespace, "err", err)
		return fmt.Errorf("read staged generation: %w", err)
	case savedMode != batch.Mode:
		err := fmt.Errorf("%w: staged generation mode %d differs from batch mode %d", ErrAppendConflict, savedMode, batch.Mode)
		slog.WarnContext(ctx, "staged generation mode conflict", "namespace", key.Namespace, "err", err)
		return err
	}
	for _, row := range batch.Rows {
		if err := stageRow(ctx, tx, descriptor, key, row); err != nil {
			return err
		}
	}
	return nil
}

func stageRow(ctx context.Context, tx *sql.Tx, descriptor StoreDescriptor, key GenerationKey, row Occurrence) error {
	payload, occurrenceHash, err := encodeOccurrence(row)
	if err != nil {
		return err
	}
	var savedHash string
	scanErr := tx.QueryRowContext(
		ctx,
		`SELECT occurrence_hash FROM staged_occurrences
		WHERE namespace = ? AND owner_id = ? AND generation_order = ? AND generation_token = ? AND row_key = ?`,
		key.Namespace, key.OwnerID, key.GenerationOrder, key.IdempotencyToken, row.RowKey,
	).Scan(&savedHash)
	if scanErr == nil {
		if savedHash != occurrenceHash {
			err := fmt.Errorf("%w: staged row %q has different content", ErrAppendConflict, row.RowKey)
			slog.WarnContext(ctx, "staged row conflict", "namespace", key.Namespace, "err", err)
			return err
		}
		return nil
	}
	if !errors.Is(scanErr, sql.ErrNoRows) {
		slog.ErrorContext(ctx, "read staged row failed", "row_key", row.RowKey, "err", scanErr)
		return fmt.Errorf("read staged row %q: %w", row.RowKey, scanErr)
	}
	vectorID := newVectorIdentity(descriptor, row.EmbeddingInput).id
	if _, err := tx.ExecContext(
		ctx,
		`INSERT INTO staged_occurrences (namespace, owner_id, generation_order, generation_token, row_key, payload, occurrence_hash, vector_id)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		key.Namespace, key.OwnerID, key.GenerationOrder, key.IdempotencyToken, row.RowKey, payload, occurrenceHash, vectorID,
	); err != nil {
		slog.ErrorContext(ctx, "save staged row failed", "row_key", row.RowKey, "err", err)
		return fmt.Errorf("save staged row %q: %w", row.RowKey, err)
	}
	return nil
}

// AbortGeneration removes the unpublished staging rows of key. Vectors that
// the staged rows referenced stay in the pool.
func (library *Library) AbortGeneration(ctx context.Context, key GenerationKey) (err error) {
	if err := validateGenerationKey(key); err != nil {
		return err
	}
	release, err := library.lock.acquire(ctx)
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(err, release())
	}()
	return library.write(ctx, func(tx *sql.Tx) error {
		return deleteStagedGeneration(ctx, tx, key)
	})
}

// stagedGenerationDeleteStatements remove one staged generation.
var stagedGenerationDeleteStatements = []string{
	`DELETE FROM staged_occurrences WHERE namespace = ? AND owner_id = ? AND generation_order = ? AND generation_token = ?`,
	`DELETE FROM staged_generations WHERE namespace = ? AND owner_id = ? AND generation_order = ? AND generation_token = ?`,
}

func deleteStagedGeneration(ctx context.Context, tx *sql.Tx, key GenerationKey) error {
	for _, statement := range stagedGenerationDeleteStatements {
		if _, err := tx.ExecContext(ctx, statement, key.Namespace, key.OwnerID, key.GenerationOrder, key.IdempotencyToken); err != nil {
			slog.ErrorContext(ctx, "remove staged generation failed", "err", err)
			return fmt.Errorf("remove staged generation: %w", err)
		}
	}
	return nil
}

// Apply seals, stages, and commits one small owner generation. It returns the
// saved receipt for a replay of a committed generation token.
func (library *Library) Apply(ctx context.Context, batch Batch) (ApplyReceipt, error) {
	seal, err := SealRows(batch.Rows)
	if err != nil {
		return ApplyReceipt{}, err
	}
	key := GenerationKey{
		Namespace:        batch.Namespace,
		OwnerID:          batch.OwnerID,
		GenerationOrder:  batch.GenerationOrder,
		IdempotencyToken: batch.IdempotencyToken,
	}
	if err := library.Stage(ctx, StageBatch{Key: key, Mode: batch.Mode, Rows: batch.Rows}); err != nil {
		return ApplyReceipt{}, err
	}
	return library.CommitGeneration(ctx, key, seal)
}

// GetOwnerState returns the committed generation of one owner. An owner
// without a committed generation returns a zero state.
func (library *Library) GetOwnerState(ctx context.Context, namespace string, ownerID string) (OwnerState, error) {
	var state OwnerState
	err := library.read(ctx, func(tx *sql.Tx) error {
		var readErr error
		state, _, readErr = readOwnerState(ctx, tx, namespace, ownerID)
		return readErr
	})
	return state, err
}

func readOwnerState(ctx context.Context, tx *sql.Tx, namespace string, ownerID string) (OwnerState, bool, error) {
	var state OwnerState
	err := tx.QueryRowContext(
		ctx,
		`SELECT generation_order, generation_token, fingerprint FROM owners WHERE namespace = ? AND owner_id = ?`,
		namespace, ownerID,
	).Scan(&state.GenerationOrder, &state.IdempotencyToken, &state.Fingerprint)
	if errors.Is(err, sql.ErrNoRows) {
		return OwnerState{}, false, nil
	}
	if err != nil {
		slog.ErrorContext(ctx, "read owner state failed", "namespace", namespace, "err", err)
		return OwnerState{}, false, fmt.Errorf("read owner state: %w", err)
	}
	return state, true, nil
}

// read runs work in one read transaction. Readers do not take the writer
// lock.
func (library *Library) read(ctx context.Context, work func(*sql.Tx) error) error {
	tx, err := library.reader.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelDefault, ReadOnly: true})
	if err != nil {
		slog.ErrorContext(ctx, "begin catalog read failed", "err", err)
		return fmt.Errorf("begin catalog read: %w", err)
	}
	workErr := work(tx)
	if err := tx.Rollback(); err != nil {
		slog.ErrorContext(ctx, "end catalog read failed", "err", err)
		return errors.Join(workErr, fmt.Errorf("end catalog read: %w", err))
	}
	return workErr
}

// sortedScalarNames returns the column names of scalars in ascending order.
func sortedScalarNames(scalars map[string]ScalarValue) []string {
	names := make([]string, 0, len(scalars))
	for name := range scalars {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

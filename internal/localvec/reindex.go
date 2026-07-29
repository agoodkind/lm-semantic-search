package localvec

import (
	"context"
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/metrics"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

const localEmbeddingPhase = "Generating embeddings and writing local vectors..."

// Reindex applies changed chunks and removals to a live collection.
func (store *Store) Reindex(
	ctx context.Context,
	codebasePath string,
	addedOrModifiedChunks []model.StoredChunk,
	removal semantic.Removal,
	progress func(semantic.Progress),
	reuse map[string][]float32,
	_ semantic.StoreColumnSet,
) error {
	if err := operationContextError(ctx, "reindex local vectors"); err != nil {
		return err
	}
	stored, err := store.collectionForName(store.CollectionName(codebasePath), false)
	if err != nil {
		return err
	}
	if _, exists, err := stored.rowCount(); err != nil {
		return err
	} else if !exists {
		return semantic.ErrCollectionMissing
	}
	rows, reused, dropped, err := store.embedRows(ctx, addedOrModifiedChunks, reuse)
	if err != nil {
		return err
	}
	if err := stored.mutate(removal, rows, true); err != nil {
		return err
	}
	emitProgress(progress, len(rows), reused, dropped)
	return nil
}

// StageReindex applies changed chunks and removals to a staging collection.
func (store *Store) StageReindex(
	ctx context.Context,
	codebasePath string,
	chunks []model.StoredChunk,
	removal semantic.Removal,
	progress func(semantic.Progress),
	reuse map[string][]float32,
	_ semantic.StoreColumnSet,
) error {
	if err := operationContextError(ctx, "stage local vectors"); err != nil {
		return err
	}
	stored, err := store.collectionForName(store.CollectionName(codebasePath), true)
	if err != nil {
		return err
	}
	rows, reused, dropped, err := store.embedRows(ctx, chunks, reuse)
	if err != nil {
		return err
	}
	if err := stored.mutate(removal, rows, false); err != nil {
		return err
	}
	emitProgress(progress, len(rows), reused, dropped)
	return nil
}

// PromoteStaging promotes a staging collection to its live name.
func (store *Store) PromoteStaging(
	ctx context.Context,
	codebasePath string,
) error {
	if err := operationContextError(ctx, "promote local vector staging"); err != nil {
		return err
	}
	collectionName := store.CollectionName(codebasePath)
	live, err := store.collectionForName(collectionName, false)
	if err != nil {
		return err
	}
	staging, err := store.collectionForName(collectionName, true)
	if err != nil {
		return err
	}

	live.mutex.Lock()
	defer live.mutex.Unlock()
	staging.mutex.Lock()
	defer staging.mutex.Unlock()
	if err := staging.loadLocked(); err != nil {
		return err
	}
	if !staging.exists {
		return semantic.ErrCollectionMissing
	}
	if err := replaceCollectionDirectory(staging.path, live.path); err != nil {
		slog.ErrorContext(
			ctx,
			"promote local vector staging collection failed",
			"collection",
			collectionName,
			"err",
			err,
		)
		return fmt.Errorf(
			"promote local vector staging collection %s: %w",
			collectionName,
			err,
		)
	}
	if live.index != nil {
		live.index.Close()
	}
	live.rows = cloneRows(staging.rows)
	live.index = staging.index
	live.dimensions = staging.dimensions
	live.loaded = true
	live.exists = true
	staging.rows = nil
	staging.index = nil
	staging.dimensions = 0
	staging.loaded = true
	staging.exists = false
	return nil
}

// CopyChunks rewrites stored chunks from one relative path to another.
func (store *Store) CopyChunks(
	ctx context.Context,
	codebasePath string,
	srcRelativePath string,
	dstRelativePath string,
) (int, error) {
	if err := operationContextError(ctx, "copy local vector chunks"); err != nil {
		return 0, err
	}
	if srcRelativePath == dstRelativePath {
		return 0, nil
	}
	stored, err := store.collectionForName(store.CollectionName(codebasePath), false)
	if err != nil {
		return 0, err
	}
	copied := 0
	err = stored.rewrite(true, func(rows []row) ([]row, error) {
		for index := range rows {
			if rows[index].RelativePath != srcRelativePath {
				continue
			}
			destinationID := copiedRowID(rows[index], dstRelativePath)
			rows[index].RelativePath = dstRelativePath
			rows[index].ID = destinationID
			rows[index].Label = 0
			copied++
		}
		return rows, nil
	})
	return copied, err
}

// PruneToCurrent removes chunks whose paths are no longer current.
func (store *Store) PruneToCurrent(
	ctx context.Context,
	codebasePath string,
	currentRelativePaths []string,
) error {
	if err := operationContextError(ctx, "prune local vector chunks"); err != nil {
		return err
	}
	// An empty current set means the file walk found nothing, which is usually a
	// transient read rather than a real empty codebase. Return without pruning so
	// the index is not wiped, matching the Milvus backend's PruneToCurrent guard.
	if len(currentRelativePaths) == 0 {
		return nil
	}
	current := make(map[string]struct{}, len(currentRelativePaths))
	for _, relativePath := range currentRelativePaths {
		current[relativePath] = struct{}{}
	}
	stored, err := store.collectionForName(store.CollectionName(codebasePath), false)
	if err != nil {
		return err
	}
	return stored.rewrite(true, func(rows []row) ([]row, error) {
		kept := make([]row, 0, len(rows))
		for _, existing := range rows {
			if _, found := current[existing.RelativePath]; found {
				kept = append(kept, existing)
			}
		}
		return kept, nil
	})
}

func (store *Store) embedRows(
	ctx context.Context,
	chunks []model.StoredChunk,
	reuse map[string][]float32,
) ([]row, int, int, error) {
	if len(chunks) == 0 {
		return nil, 0, 0, nil
	}
	provider, err := store.embeddingProvider()
	if err != nil {
		return nil, 0, 0, err
	}

	// Pre-split oversized chunks at the active model's real input limit before the
	// embedder sees them. The offline ONNX tokenizer would otherwise truncate an
	// oversized input to its 2048/512-token maximum and silently lose the rest, so
	// the byte budget is derived from the preset limit, not the 4096
	// OpenAI-compatible limit.
	byteBudget := config.EmbedChunkByteBudgetForLimit(store.cfg.EmbeddingMaxTokens, config.ActiveEmbedTokenLimit(store.cfg))
	splitChunks, splitCount := semantic.SplitChunksToByteBudget(chunks, byteBudget)
	if splitCount > 0 {
		slog.InfoContext(ctx, "localvec.split_over_token_budget", "byte_budget", byteBudget, "chunks_split", splitCount, "expanded_from", len(chunks), "expanded_to", len(splitChunks))
	}

	rows := make([]row, 0, len(splitChunks))
	missChunks := make([]model.StoredChunk, 0, len(splitChunks))
	reused := 0
	dropped := 0
	for _, chunk := range splitChunks {
		if vector, found := reuse[semantic.ContentVectorKey(chunk.Content)]; found {
			stored, rowErr := newRow(chunk, append([]float32(nil), vector...))
			if rowErr != nil {
				return nil, 0, 0, rowErr
			}
			rows = append(rows, stored)
			reused++
			continue
		}
		missChunks = append(missChunks, chunk)
	}
	// Counted here as well as reported through progress, so the process counter
	// moves on this backend too. Without it a status read shows
	// embed_chunks_reused_total at zero while the job's own chunks_reused rises,
	// and the two numbers on one screen contradict each other.
	metrics.ChunksReused(reused)

	if len(missChunks) > 0 {
		// EmbedChunksSplittingOversize re-splits and retries any chunk the endpoint
		// rejects as oversized, so an OpenAI-compatible local embedder never drops a
		// dense chunk; the offline ONNX embedder never rejects because the pre-split
		// already kept every piece under the preset limit.
		pack := semantic.NewChunkPacker(store.cfg.EmbeddingBatchSize, store.cfg.EmbeddingBatchTokenBudget)
		embeddedChunks, vectors, droppedCount, embedErr := semantic.EmbedChunksSplittingOversize(ctx, missChunks, config.ActiveEmbedTokenLimit(store.cfg), pack, provider.EmbedBatch)
		if embedErr != nil {
			slog.ErrorContext(ctx, "embed local vector chunks failed", "chunks", len(missChunks), "err", embedErr)
			return nil, 0, 0, fmt.Errorf("embed local vector chunks: %w", embedErr)
		}
		dropped = droppedCount
		if droppedCount > 0 {
			slog.WarnContext(ctx, "local vector rows dropped as indivisible", "dropped", droppedCount, "chunks", len(missChunks))
		}
		for index := range embeddedChunks {
			stored, rowErr := newRow(embeddedChunks[index], vectors[index])
			if rowErr != nil {
				return nil, 0, 0, rowErr
			}
			rows = append(rows, stored)
		}
	}
	return rows, reused, dropped, nil
}

func emitProgress(
	progress func(semantic.Progress),
	rowCount int,
	reused int,
	dropped int,
) {
	if progress == nil || rowCount == 0 && dropped == 0 {
		return
	}
	rows := safeInt32(rowCount)
	progress(semantic.Progress{
		Phase:                     localEmbeddingPhase,
		OverallPercent:            100,
		EmbeddingBatchesTotal:     1,
		EmbeddingBatchesCompleted: 1,
		CollectionRowsWritten:     rows,
		ChunksProcessed:           rows,
		ChunksReused:              safeInt32(reused),
		ChunksEmbedded:            safeInt32(rowCount - reused),
		ChunksDropped:             safeInt32(dropped),
	})
}

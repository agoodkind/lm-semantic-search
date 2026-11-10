package localvec

import (
	"context"
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/embedding"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/metrics"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

const localEmbeddingPhase = "Generating embeddings and writing local vectors..."

type rowEmbedding struct {
	rows    []row
	reused  int
	dropped int
	batches int
}

// Reindex applies changed chunks and removals to a live collection.
func (store *Store) Reindex(
	ctx context.Context,
	codebasePath string,
	addedOrModifiedChunks []model.StoredChunk,
	removal semantic.Removal,
	progress func(semantic.Progress),
	reuse map[string]model.ReuseVector,
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
	embedded, err := store.embedRows(ctx, addedOrModifiedChunks, reuse)
	if err != nil {
		emitProgress(progress, 0, 0, 0, embedded.batches)
		return err
	}
	if err := stored.mutate(removal, embedded.rows, true); err != nil {
		emitProgress(progress, 0, 0, 0, embedded.batches)
		return err
	}
	emitProgress(progress, len(embedded.rows), embedded.reused, embedded.dropped, embedded.batches)
	return nil
}

// StageReindex applies changed chunks and removals to a staging collection.
func (store *Store) StageReindex(
	ctx context.Context,
	codebasePath string,
	chunks []model.StoredChunk,
	removal semantic.Removal,
	progress func(semantic.Progress),
	reuse map[string]model.ReuseVector,
	_ semantic.StoreColumnSet,
) error {
	if err := operationContextError(ctx, "stage local vectors"); err != nil {
		return err
	}
	stored, err := store.collectionForName(store.CollectionName(codebasePath), true)
	if err != nil {
		return err
	}
	embedded, err := store.embedRows(ctx, chunks, reuse)
	if err != nil {
		emitProgress(progress, 0, 0, 0, embedded.batches)
		return err
	}
	if err := stored.mutate(removal, embedded.rows, false); err != nil {
		emitProgress(progress, 0, 0, 0, embedded.batches)
		return err
	}
	emitProgress(progress, len(embedded.rows), embedded.reused, embedded.dropped, embedded.batches)
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
	live.reuseRows = buildReuseRowIndex(live.rows)
	live.index = staging.index
	live.dimensions = staging.dimensions
	live.loaded = true
	live.exists = true
	staging.rows = nil
	staging.reuseRows = nil
	staging.index = nil
	staging.dimensions = 0
	staging.loaded = true
	staging.exists = false
	return nil
}

// CopyChunks applies all copies in one rewrite of the stored collection. A later
// copy can rewrite rows that an earlier copy renamed.
func (store *Store) CopyChunks(
	ctx context.Context,
	codebasePath string,
	copies []semantic.ChunkCopy,
) ([]int, error) {
	if err := operationContextError(ctx, "copy local vector chunks"); err != nil {
		return nil, err
	}
	copiedRowCounts := make([]int, len(copies))
	stored, err := store.collectionForName(store.CollectionName(codebasePath), false)
	if err != nil {
		return nil, err
	}
	err = stored.rewrite(true, func(rows []row) ([]row, error) {
		for copyIndex, chunkCopy := range copies {
			if chunkCopy.Source == chunkCopy.Destination {
				continue
			}
			for index := range rows {
				if rows[index].RelativePath != chunkCopy.Source {
					continue
				}
				destinationID := copiedRowID(rows[index], chunkCopy.Destination)
				rows[index].RelativePath = chunkCopy.Destination
				rows[index].ID = destinationID
				rows[index].Label = 0
				copiedRowCounts[copyIndex]++
			}
		}
		return rows, nil
	})
	if err != nil {
		return nil, err
	}
	return copiedRowCounts, nil
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
	reuse map[string]model.ReuseVector,
) (rowEmbedding, error) {
	result := rowEmbedding{rows: nil, reused: 0, dropped: 0, batches: 0}
	if len(chunks) == 0 {
		return result, nil
	}
	provider, err := store.embeddingProvider(ctx)
	if err != nil {
		return result, err
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

	result.rows = make([]row, 0, len(splitChunks))
	missChunks := make([]model.StoredChunk, 0, len(splitChunks))
	for _, chunk := range splitChunks {
		if reusedVector, found := reuse[semantic.ContentVectorKey(chunk.Content)]; found {
			stored, rowErr := newRow(chunk, append([]float32(nil), reusedVector.Vector...))
			if rowErr != nil {
				return result, rowErr
			}
			result.rows = append(result.rows, stored)
			result.reused++
			continue
		}
		missChunks = append(missChunks, chunk)
	}
	// Counted here as well as reported through progress, so the process counter
	// moves on this backend too. Without it a status read shows
	// embed_chunks_reused_total at zero while the job's own chunks_reused rises,
	// and the two numbers on one screen contradict each other.
	metrics.ChunksReused(result.reused)

	if len(missChunks) > 0 {
		// EmbedChunksSplittingOversize re-splits and retries any chunk the endpoint
		// rejects as oversized, so an OpenAI-compatible local embedder never drops a
		// dense chunk; the offline ONNX embedder never rejects because the pre-split
		// already kept every piece under the preset limit.
		pack := semantic.NewChunkPacker(store.cfg.EmbeddingBatchSize, store.cfg.EmbeddingBatchTokenBudget)
		embedBatch := func(batchContext context.Context, texts []string) (embedding.BatchResult, error) {
			vectors, embedErr := provider.EmbedBatch(batchContext, texts)
			if embedErr != nil {
				return vectors, fmt.Errorf("embed local vector batch: %w", embedErr)
			}
			result.batches++
			return vectors, nil
		}
		embeddedChunks, vectors, droppedCount, embedErr := semantic.EmbedChunksSplittingOversize(ctx, missChunks, config.ActiveEmbedTokenLimit(store.cfg), pack, embedBatch)
		if embedErr != nil {
			slog.ErrorContext(ctx, "embed local vector chunks failed", "chunks", len(missChunks), "err", embedErr)
			return result, fmt.Errorf("embed local vector chunks: %w", embedErr)
		}
		result.dropped = droppedCount
		if droppedCount > 0 {
			slog.WarnContext(ctx, "local vector rows dropped as indivisible", "dropped", droppedCount, "chunks", len(missChunks))
		}
		for index := range embeddedChunks {
			stored, rowErr := newRow(embeddedChunks[index], vectors[index])
			if rowErr != nil {
				return result, rowErr
			}
			result.rows = append(result.rows, stored)
		}
	}
	return result, nil
}

func emitProgress(
	progress func(semantic.Progress),
	rowCount int,
	reused int,
	dropped int,
	batches int,
) {
	if progress == nil || rowCount == 0 && dropped == 0 && batches == 0 {
		return
	}
	rows := safeInt32(rowCount)
	progress(semantic.Progress{
		Phase:                     localEmbeddingPhase,
		OverallPercent:            100,
		EmbeddingBatchesTotal:     safeInt32(batches),
		EmbeddingBatchesCompleted: safeInt32(batches),
		EmbeddingCallsCompleted:   safeInt32(batches),
		CollectionRowsWritten:     rows,
		ChunksProcessed:           rows,
		ChunksReused:              safeInt32(reused),
		ChunksEmbedded:            safeInt32(rowCount - reused),
		ChunksDropped:             safeInt32(dropped),
	})
}

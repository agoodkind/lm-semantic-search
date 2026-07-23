package localvec

import (
	"context"
	"fmt"
	"log/slog"

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
	rows, reused, err := store.embedRows(ctx, addedOrModifiedChunks, reuse)
	if err != nil {
		return err
	}
	if err := stored.mutate(removal, rows, true); err != nil {
		return err
	}
	emitProgress(progress, len(rows), reused)
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
	rows, reused, err := store.embedRows(ctx, chunks, reuse)
	if err != nil {
		return err
	}
	if err := stored.mutate(removal, rows, false); err != nil {
		return err
	}
	emitProgress(progress, len(rows), reused)
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
			rows[index].RelativePath = dstRelativePath
			rows[index].ID = generateRowID(rows[index].chunk(0))
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
) ([]row, int, error) {
	if len(chunks) == 0 {
		return nil, 0, nil
	}
	provider, err := store.embeddingProvider()
	if err != nil {
		return nil, 0, err
	}
	vectors := make([][]float32, len(chunks))
	missingTexts := make([]string, 0, len(chunks))
	missingIndexes := make([]int, 0, len(chunks))
	for index, chunk := range chunks {
		if vector, found := reuse[semantic.ContentVectorKey(chunk.Content)]; found {
			vectors[index] = append([]float32(nil), vector...)
			continue
		}
		missingTexts = append(missingTexts, chunk.Content)
		missingIndexes = append(missingIndexes, index)
	}
	if len(missingTexts) > 0 {
		embedded, embedErr := provider.EmbedBatch(ctx, missingTexts)
		if embedErr != nil {
			slog.ErrorContext(
				ctx,
				"embed local vector chunks failed",
				"chunks",
				len(missingTexts),
				"err",
				embedErr,
			)
			return nil, 0, fmt.Errorf("embed local vector chunks: %w", embedErr)
		}
		if len(embedded) != len(missingTexts) {
			return nil, 0, fmt.Errorf(
				"embedding provider returned %d vectors for %d chunks",
				len(embedded),
				len(missingTexts),
			)
		}
		for position, chunkIndex := range missingIndexes {
			vectors[chunkIndex] = embedded[position]
		}
	}
	rows := make([]row, 0, len(chunks))
	for index, chunk := range chunks {
		stored, rowErr := newRow(chunk, vectors[index])
		if rowErr != nil {
			return nil, 0, rowErr
		}
		rows = append(rows, stored)
	}
	return rows, len(chunks) - len(missingTexts), nil
}

func emitProgress(
	progress func(semantic.Progress),
	rowCount int,
	reused int,
) {
	if progress == nil || rowCount == 0 {
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
	})
}

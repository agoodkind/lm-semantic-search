package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func (manager *Manager) reuseStateForChangedFile(
	ctx context.Context,
	state deltaState,
	relativePath string,
	fileResult indexer.OneFileResult,
	removal semantic.Removal,
	force bool,
) (deltaState, error) {
	contentReuse, err := manager.contentReuseForChangedFile(
		ctx,
		state,
		relativePath,
		fileResult.Chunks,
		force,
	)
	if err != nil {
		return state, err
	}
	state.reuse = mergedReuse(state.reuse, contentReuse)
	if state.chunkCounts != nil {
		state.chunkCounts.reuseVectorsLoaded += safeInt32(len(contentReuse))
	}
	if fileResult.ReuseVectors != nil {
		state.reuse = mergedReuse(state.reuse, fileResult.ReuseVectors)
		if state.chunkCounts != nil {
			state.chunkCounts.reuseVectorsLoaded += safeInt32(len(fileResult.ReuseVectors))
		}
		return state, nil
	}
	if len(fileResult.Chunks) == 0 && removal.Empty() {
		return state, nil
	}
	reuse, loaded, err := manager.itemReuse(ctx, state, relativePath)
	if err != nil {
		return state, err
	}
	state.reuse = reuse
	if state.chunkCounts != nil {
		state.chunkCounts.reuseVectorsLoaded += loaded
	}
	return state, nil
}

func (manager *Manager) contentReuseForChangedFile(
	ctx context.Context,
	state deltaState,
	relativePath string,
	chunks []model.StoredChunk,
	force bool,
) (map[string][]float32, error) {
	if force || len(chunks) == 0 {
		return nil, nil
	}
	source := state.source.reuseSource(relativePath)
	if source.Scope == itemReuseScopeNone {
		return nil, nil
	}
	var reuse map[string][]float32
	err := manager.runReleasingCapacityIfStalled(ctx, func() error {
		var loaded map[string][]float32
		var loadErr error
		if source.ContentReuse.explicit {
			loaded, loadErr = manager.semantic.LoadReuseVectorsForContentsFrom(
				ctx,
				[]string{source.CollectionName},
				chunks,
			)
		} else {
			loaded, loadErr = manager.semantic.LoadReuseVectorsForContents(
				ctx,
				source.CollectionName,
				chunks,
			)
		}
		reuse = loaded
		if loadErr != nil {
			slog.ErrorContext(
				ctx,
				"load corpus reuse failed",
				"path", relativePath,
				"collection", source.CollectionName,
				"err", loadErr,
			)
			return fmt.Errorf("load corpus reuse for %s: %w", relativePath, loadErr)
		}
		return nil
	})
	if err != nil || source.ContentReuse.source == nil {
		return reuse, err
	}
	return manager.declaredSourceContentReuse(ctx, source.ContentReuse.source, chunks, reuse)
}

func (manager *Manager) declaredSourceContentReuse(
	ctx context.Context,
	source *jobSourceReuse,
	chunks []model.StoredChunk,
	targetReuse map[string][]float32,
) (map[string][]float32, error) {
	if source.disabled.Load() {
		return targetReuse, nil
	}
	missing := make([]model.StoredChunk, 0, len(chunks))
	for _, chunk := range chunks {
		if _, found := targetReuse[semantic.ContentVectorKey(chunk.Content)]; !found {
			missing = append(missing, chunk)
		}
	}
	if len(missing) == 0 {
		return targetReuse, nil
	}
	var sourceReuse map[string][]float32
	var sourceErr error
	capacityErr := manager.runReleasingCapacityIfStalled(ctx, func() error {
		sourceReuse, sourceErr = manager.semantic.LoadReuseVectorsForContentsFrom(
			ctx,
			[]string{source.collection},
			missing,
		)
		return nil
	})
	if capacityErr != nil {
		return targetReuse, capacityErr
	}
	if sourceErr == nil {
		return mergedReuse(targetReuse, sourceReuse), nil
	}
	if errors.Is(sourceErr, context.Canceled) {
		slog.WarnContext(ctx, "declared source collection read canceled", "job_id", source.jobID, "source_collection", source.collection, "err", sourceErr)
		return targetReuse, fmt.Errorf("read reuse vectors from declared source collection %s canceled: %w", source.collection, sourceErr)
	}
	if source.disabled.CompareAndSwap(false, true) {
		slog.WarnContext(ctx, "declared source collection read failed and the job continues with only the target collection",
			"job_id", source.jobID,
			"source_path", source.sourcePath,
			"source_collection", source.collection,
			"err", sourceErr,
		)
	}
	return targetReuse, nil
}

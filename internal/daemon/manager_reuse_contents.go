package daemon

import (
	"context"
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
		loaded, loadErr := manager.semantic.LoadReuseVectorsForContents(
			ctx,
			source.CollectionName,
			chunks,
		)
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
	return reuse, err
}

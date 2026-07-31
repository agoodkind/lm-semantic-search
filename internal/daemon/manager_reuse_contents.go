package daemon

import (
	"context"
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/model"
)

func (manager *Manager) contentReuseForChangedFile(
	ctx context.Context,
	state deltaState,
	relativePath string,
	chunks []model.StoredChunk,
) (map[string][]float32, error) {
	if !state.itemReuseEnabled {
		return nil, nil
	}
	if len(chunks) == 0 {
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

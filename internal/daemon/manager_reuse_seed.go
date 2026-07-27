package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/model"
)

// resolveReuseSeed loads build-wide reuse vectors from indexed descendants and
// sibling worktrees that share the requested embedding model. It also returns
// the descendant candidates it scanned so a from-scratch build can absorb them
// without re-scanning the registry, keeping the reuse seed and the absorb list
// derived from one consistent snapshot.
func (manager *Manager) resolveReuseSeed(
	ctx context.Context,
	job model.Job,
) (map[string][]float32, int32, []model.Codebase, error) {
	descendants := manager.descendantReuseCandidates(job.CanonicalPath, job.Config)
	reuse := map[string][]float32{}
	if manager.semantic == nil || !manager.semantic.Available() {
		return reuse, 0, descendants, nil
	}
	reuseCollections := collectionNamesOf(descendants)
	reuseCollections = append(reuseCollections, manager.worktreeSiblingReuseCollections(job.CanonicalPath, job.Config)...)
	if len(reuseCollections) == 0 {
		return reuse, 0, descendants, nil
	}
	var loadedReuse map[string][]float32
	err := manager.runReleasingCapacityIfStalled(ctx, func() error {
		var loadErr error
		loadedReuse, loadErr = manager.semantic.LoadReuseVectors(ctx, reuseCollections)
		if loadErr == nil {
			return nil
		}
		slog.WarnContext(ctx, "load reuse vectors from semantic backend failed",
			"job_id", job.ID,
			"err", loadErr,
		)
		return fmt.Errorf("load reuse vectors: %w", loadErr)
	})
	if err != nil {
		var reacquireErr *jobCapacityReacquireError
		if errors.As(err, &reacquireErr) {
			return reuse, 0, descendants, reacquireErr
		}
		slog.WarnContext(ctx, "load reuse vectors failed; building without the reuse seed", "job_id", job.ID, "err", err)
		return reuse, 0, descendants, nil
	}
	seeded := safeInt32(len(loadedReuse))
	slog.InfoContext(ctx, "build.reuse_seeded", "job_id", job.ID, "reuse_collections", len(reuseCollections), "reuse_vectors", len(loadedReuse))
	return loadedReuse, seeded, descendants, nil
}

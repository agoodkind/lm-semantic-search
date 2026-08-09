package daemon

import (
	"context"
	"sort"

	"goodkind.io/lm-semantic-search/internal/model"
	statusresolver "goodkind.io/lm-semantic-search/internal/status"
)

// ListIndexes returns every tracked codebase in canonical path order.
func (manager *Manager) ListIndexes(ctx context.Context) []model.Codebase {
	_ = ctx
	manager.mu.Lock()
	defer manager.mu.Unlock()

	codebases := make([]model.Codebase, 0, len(manager.codebases))
	for _, codebase := range manager.codebases {
		codebases = append(codebases, codebase)
	}
	sort.Slice(codebases, func(i int, j int) bool {
		return codebases[i].CanonicalPath < codebases[j].CanonicalPath
	})
	return codebases
}

// CodebaseView pairs a codebase with its daemon-computed display status, so the
// presentation fold (live job phase) is decided once, under the lock, rather
// than recomputed at each rendering callsite.
type CodebaseView struct {
	Codebase model.Codebase
	Display  displayStatus
}

// ListIndexesView returns every tracked codebase in canonical path order, each
// paired with its single-source-of-truth display status. It folds the active
// job in under the lock so the list and detail surfaces agree by construction.
func (manager *Manager) ListIndexesView() []CodebaseView {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	return manager.codebaseViewsLocked()
}

func (manager *Manager) displayForCollectionReadiness(
	codebase model.Codebase,
	readiness statusresolver.CollectionReadiness,
) displayStatus {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	current, found := manager.codebases[codebase.ID]
	if found {
		codebase = current
	}
	return computeDisplayStatus(
		codebase,
		manager.activeJobSnapshotLocked(codebase),
		manager.health.Mode,
		readiness,
	)
}

// codebaseViewsLocked builds the view list. It is separate from ListIndexesView
// so a caller that already holds the lock for a wider snapshot reads the same
// codebases the rest of that snapshot describes. Caller holds manager.mu.
func (manager *Manager) codebaseViewsLocked() []CodebaseView {
	globalMode := manager.health.Mode
	views := make([]CodebaseView, 0, len(manager.codebases))
	for _, codebase := range manager.codebases {
		activeJob := manager.activeJobSnapshotLocked(codebase)
		views = append(views, CodebaseView{
			Codebase: codebase,
			Display:  computeDisplayStatus(codebase, activeJob, globalMode, collectionNotApplicable),
		})
	}
	sort.Slice(views, func(i int, j int) bool {
		return views[i].Codebase.CanonicalPath < views[j].Codebase.CanonicalPath
	})
	return views
}

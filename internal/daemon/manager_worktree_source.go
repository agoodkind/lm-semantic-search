package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/gitworktree"
	"goodkind.io/lm-semantic-search/internal/model"
)

type worktreeSourceIntent struct {
	sourcePath     string
	startingCommit string
}

func (manager *Manager) resolveWorktreeSourceIntent(
	ctx context.Context,
	canonicalPath string,
	request model.WorktreeSourceRequest,
) (worktreeSourceIntent, error) {
	emptyIntent := worktreeSourceIntent{sourcePath: "", startingCommit: ""}
	intent := worktreeSourceIntent{
		sourcePath:     "",
		startingCommit: strings.TrimSpace(request.StartingCommit),
	}
	requestedSource := strings.TrimSpace(request.SourcePath)
	if requestedSource == "" {
		if intent.startingCommit == "" {
			return intent, nil
		}
		manager.mu.Lock()
		target, _ := manager.findCodebaseByExactRoot(canonicalPath)
		manager.mu.Unlock()
		if target.WorktreeSourcePath == "" {
			return emptyIntent, adapterr.NewInvalidPath(fmt.Sprintf("target %q has no supplied or recorded source path for the requested starting commit", canonicalPath), nil)
		}
		return intent, nil
	}
	sourcePath, err := manager.resolveCanonicalPath(requestedSource)
	if err != nil {
		slog.WarnContext(ctx, "failed to resolve worktree source path", "path", canonicalPath, "source_path", requestedSource, "err", err)
		return emptyIntent, fmt.Errorf("failed to canonicalize worktree source path %s: %w", requestedSource, err)
	}
	if sourcePath == canonicalPath {
		return emptyIntent, adapterr.NewInvalidPath(fmt.Sprintf("worktree source path %q equals the target path", sourcePath), nil)
	}
	sourceInfo, statErr := os.Stat(sourcePath)
	if statErr != nil || !sourceInfo.IsDir() {
		return emptyIntent, adapterr.NewInvalidPath(fmt.Sprintf("worktree source path %q does not exist or is not a directory", sourcePath), statErr)
	}
	targetWorktree, targetInRepository := gitworktree.Resolve(canonicalPath)
	sourceWorktree, sourceInRepository := gitworktree.Resolve(sourcePath)
	if !targetInRepository || !sourceInRepository || targetWorktree.CommonDir != sourceWorktree.CommonDir {
		return emptyIntent, adapterr.NewInvalidPath(fmt.Sprintf("worktree source path %q and target path %q are not in the same Git repository", sourcePath, canonicalPath), nil)
	}
	manager.mu.Lock()
	_, sourceRegistered := manager.findCodebaseByExactRoot(sourcePath)
	cyclic := manager.worktreeSourceCycleLocked(canonicalPath, sourcePath)
	manager.mu.Unlock()
	if sourceWorktree.WorktreeRoot != sourcePath && !sourceRegistered {
		return emptyIntent, adapterr.NewInvalidPath(fmt.Sprintf("source path %q is neither a worktree root of the repository nor a registered codebase root", sourcePath), nil)
	}
	if cyclic {
		return emptyIntent, adapterr.NewInvalidPath(fmt.Sprintf("recording worktree source path %q would create a source cycle that includes target path %q", sourcePath, canonicalPath), nil)
	}
	intent.sourcePath = sourcePath
	return intent, nil
}

func (manager *Manager) worktreeSourceCycleLocked(targetPath string, sourcePath string) bool {
	visited := map[string]struct{}{}
	current := sourcePath
	for current != "" {
		if current == targetPath {
			return true
		}
		if _, seen := visited[current]; seen {
			return false
		}
		visited[current] = struct{}{}
		codebase, found := manager.findCodebaseByExactRoot(current)
		if !found {
			return false
		}
		current = codebase.WorktreeSourcePath
	}
	return false
}

// A request without source fields preserves the stored values; a new source path replaces the stored path and clears the stored starting commit unless the request supplies a commit.
// The caller must own manager.mu.
func (manager *Manager) applyWorktreeSourceLocked(
	ctx context.Context,
	codebase model.Codebase,
	intent worktreeSourceIntent,
) (model.Codebase, error) {
	var emptyCodebase model.Codebase
	if intent.sourcePath == "" && intent.startingCommit != "" && codebase.WorktreeSourcePath == "" {
		return emptyCodebase, adapterr.NewInvalidPath(fmt.Sprintf("target %q has no supplied or recorded source path for the requested starting commit", codebase.CanonicalPath), nil)
	}
	if intent.sourcePath != "" {
		if manager.worktreeSourceCycleLocked(codebase.CanonicalPath, intent.sourcePath) {
			return emptyCodebase, adapterr.NewInvalidPath(fmt.Sprintf("recording worktree source path %q would create a source cycle that includes target path %q", intent.sourcePath, codebase.CanonicalPath), nil)
		}
		if intent.sourcePath != codebase.WorktreeSourcePath {
			codebase.WorktreeStartingCommit = ""
		}
		codebase.WorktreeSourcePath = intent.sourcePath
	}
	if intent.startingCommit != "" {
		codebase.WorktreeStartingCommit = intent.startingCommit
	}
	codebase.WorktreeSourceID = ""
	if codebase.WorktreeSourcePath != "" {
		if sourceCodebase, found := manager.findCodebaseByExactRoot(codebase.WorktreeSourcePath); found {
			codebase.WorktreeSourceID = sourceCodebase.ID
		}
	}
	if intent.sourcePath != "" || intent.startingCommit != "" {
		slog.InfoContext(ctx, "request changed recorded worktree source", "codebase_id", codebase.ID, "path", codebase.CanonicalPath, "source_path", codebase.WorktreeSourcePath, "source_id", codebase.WorktreeSourceID, "starting_commit", codebase.WorktreeStartingCommit)
	}
	return codebase, nil
}

func (manager *Manager) recordWorktreeSource(
	ctx context.Context,
	codebaseID string,
	intent worktreeSourceIntent,
) error {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	codebase, found := manager.codebases[codebaseID]
	if !found {
		return fmt.Errorf("failed to find codebase %s in registry", codebaseID)
	}
	updated, err := manager.applyWorktreeSourceLocked(ctx, codebase, intent)
	if err != nil {
		return err
	}
	if worktreeSourceEqual(codebase, updated) {
		return nil
	}
	manager.codebases[updated.ID] = updated
	if err := manager.saveLocked(); err != nil {
		manager.codebases[codebase.ID] = codebase
		return err
	}
	return nil
}

func worktreeSourceEqual(first model.Codebase, second model.Codebase) bool {
	return first.WorktreeSourcePath == second.WorktreeSourcePath &&
		first.WorktreeSourceID == second.WorktreeSourceID &&
		first.WorktreeStartingCommit == second.WorktreeStartingCommit
}

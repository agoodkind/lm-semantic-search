// Package discovery lists the files of a codebase. It walks the tree once and
// defers every "should this path be indexed?" decision to the single source of
// truth in internal/indexability, so the walk owns no ignore rules, no size
// cap, and no binary denylist of its own.
package discovery

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"slices"

	"goodkind.io/lm-semantic-search/internal/indexability"
)

// Result is one discovery pass over a codebase root.
type Result struct {
	// Files holds the absolute paths the walk listed, sorted. Each path passed
	// the resolver's scope and ignore gates; the size and content gates are left
	// to the indexer and merkle capture, which apply them per file.
	Files []string
}

// Discover walks a codebase root once and lists the files the indexability
// resolver keeps in scope. The walk prunes a directory the resolver declines so
// it never descends into an ignored tree, a nested same-repo worktree, or the
// git directory, mirroring the gates Decide applies to a single path. The size
// and content gates are intentionally not applied here, so a listed file may
// still be skipped later by the indexer; the walk only resolves scope and
// ignore membership.
//
// The caller supplies the daemon's one shared resolver and the codebase id it
// keys rules under, so the walk honors the same custom overrides and cached
// matcher every other indexability decision sees.
func Discover(ctx context.Context, resolver *indexability.Resolver, codebaseID string, root string) (Result, error) {
	absoluteRoot, err := filepath.Abs(root)
	if err != nil {
		slog.ErrorContext(ctx, "resolve absolute root failed", "root", root, "err", err)
		return Result{}, fmt.Errorf("resolve absolute root %s: %w", root, err)
	}

	files := []string{}
	var walk func(relativeDir string) error
	walk = func(relativeDir string) error {
		if walkErr := ctx.Err(); walkErr != nil {
			slog.ErrorContext(ctx, "walk cancelled", "path", relativeDir, "err", walkErr)
			return fmt.Errorf("walk cancelled at %s: %w", relativeDir, walkErr)
		}
		currentDir := filepath.Join(absoluteRoot, filepath.FromSlash(relativeDir))
		entries, readErr := os.ReadDir(currentDir)
		if readErr != nil {
			slog.ErrorContext(ctx, "read directory failed", "path", currentDir, "err", readErr)
			return fmt.Errorf("read directory %s: %w", currentDir, readErr)
		}
		for _, entry := range entries {
			childRelative := entry.Name()
			if relativeDir != "" {
				childRelative = relativeDir + "/" + entry.Name()
			}
			if resolver.Ignored(ctx, codebaseID, absoluteRoot, childRelative, entry.IsDir()) {
				continue
			}
			if entry.IsDir() {
				if walkErr := walk(childRelative); walkErr != nil {
					return walkErr
				}
				continue
			}
			files = append(files, filepath.Join(absoluteRoot, filepath.FromSlash(childRelative)))
		}
		return nil
	}
	if walkErr := walk(""); walkErr != nil {
		return Result{}, walkErr
	}
	slices.Sort(files)

	return Result{Files: files}, nil
}

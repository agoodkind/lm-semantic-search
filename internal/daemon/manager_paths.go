package daemon

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
)

// findCodebaseByExactRoot returns the codebase whose CanonicalPath equals
// canonicalPath. Used by registration, dedup, and cancellation paths where
// two distinct codebases rooted at sibling directories must remain
// independent.
func (manager *Manager) findCodebaseByExactRoot(canonicalPath string) (model.Codebase, bool) {
	for _, codebase := range manager.codebases {
		if codebase.CanonicalPath == canonicalPath {
			return codebase, true
		}
	}
	var emptyCodebase model.Codebase
	return emptyCodebase, false
}

// findCodebasesByCoverage returns every codebase whose CanonicalPath is a
// prefix of canonicalPath, sorted longest-prefix first so the nearest
// covering codebase is index 0. Used by status, search, clear, and watcher
// dispatch where one path can fall under multiple overlapping codebases.
func (manager *Manager) findCodebasesByCoverage(canonicalPath string) []model.Codebase {
	matches := make([]model.Codebase, 0, len(manager.codebases))
	for _, codebase := range manager.codebases {
		if codebase.Kind == model.CodebaseKindDocument {
			continue
		}
		if pathCovers(codebase.CanonicalPath, canonicalPath) {
			matches = append(matches, codebase)
		}
	}
	sort.Slice(matches, func(first int, second int) bool {
		return len(matches[first].CanonicalPath) > len(matches[second].CanonicalPath)
	})
	return matches
}

// findStrictAncestor returns the longest codebase whose CanonicalPath
// strictly prefix-covers canonicalPath (so the codebase's root is not equal
// to canonicalPath). Empty result when there is no strict ancestor. Used at
// registration time to surface an overlap hint.
func (manager *Manager) findStrictAncestor(canonicalPath string) (model.Codebase, bool) {
	var best model.Codebase
	bestLength := -1
	for _, codebase := range manager.codebases {
		if codebase.Kind == model.CodebaseKindDocument {
			continue
		}
		if codebase.CanonicalPath == canonicalPath {
			continue
		}
		if !pathCovers(codebase.CanonicalPath, canonicalPath) {
			continue
		}
		if len(codebase.CanonicalPath) > bestLength {
			best = codebase
			bestLength = len(codebase.CanonicalPath)
		}
	}
	if bestLength < 0 {
		var empty model.Codebase
		return empty, false
	}
	return best, true
}

// findDescendants returns every codebase whose CanonicalPath is strictly
// inside canonicalPath (so canonicalPath covers the codebase root and is not
// equal to it), sorted longest-prefix first. It is the mirror of
// findStrictAncestor and drives the merge-down absorb path: when a new index
// roots above existing child codebases, their stored vectors are reused
// instead of being re-embedded.
func (manager *Manager) findDescendants(canonicalPath string) []model.Codebase {
	matches := make([]model.Codebase, 0, len(manager.codebases))
	for _, codebase := range manager.codebases {
		if codebase.Kind == model.CodebaseKindDocument {
			continue
		}
		if codebase.CanonicalPath == canonicalPath {
			continue
		}
		if !pathCovers(canonicalPath, codebase.CanonicalPath) {
			continue
		}
		matches = append(matches, codebase)
	}
	sort.Slice(matches, func(first int, second int) bool {
		return len(matches[first].CanonicalPath) > len(matches[second].CanonicalPath)
	})
	return matches
}

// looksLikeCodebaseID reports whether requestedPath has the shape of a codebase
// id rather than a filesystem path: the "cb_" prefix newID stamps and no path
// separator. It lets resolveCanonicalPath return a clear "unknown id" error for
// a mistyped id instead of silently resolving it as a relative path.
func looksLikeCodebaseID(requestedPath string) bool {
	return strings.HasPrefix(requestedPath, "cb_") && !strings.ContainsRune(requestedPath, filepath.Separator)
}

// resolveCanonicalPath maps a request argument to a codebase's canonical path,
// accepting three forms interchangeably: a registry codebase id, a filesystem
// path, or a symlink to one. An exact id match returns that codebase's stored
// canonical path; an id-shaped argument with no match returns an error naming
// the id; anything else resolves through canonicalizePath, which makes the path
// absolute and follows symlinks. Resolution lives here in the daemon so the CLI
// and MCP adapter pass their argument through unchanged.
func (manager *Manager) resolveCanonicalPath(requestedPath string) (string, error) {
	trimmed := strings.TrimSpace(requestedPath)
	manager.mu.Lock()
	codebase, found := manager.codebases[trimmed]
	manager.mu.Unlock()
	if found {
		return codebase.CanonicalPath, nil
	}
	if looksLikeCodebaseID(trimmed) {
		return "", adapterr.NewUnknownCodebaseID(trimmed)
	}
	return canonicalizePath(requestedPath)
}

// resolveRequestPath makes a relative request path absolute using the
// caller's working directory, carried in ClientInfo.caller_cwd. Absolute
// paths, codebase ids, and URI-shaped arguments pass through unchanged for
// the later resolution stages to classify. A relative path with no absolute
// caller cwd is rejected: the daemon's own working directory is never the
// caller's, so resolving against it silently is never correct.
func resolveRequestPath(requestedPath string, callerCwd string) (string, error) {
	trimmed := strings.TrimSpace(requestedPath)
	if trimmed == "" || looksLikeCodebaseID(trimmed) || strings.Contains(trimmed, "://") {
		return requestedPath, nil
	}
	if filepath.IsAbs(trimmed) {
		return trimmed, nil
	}
	cwd := strings.TrimSpace(callerCwd)
	if !filepath.IsAbs(cwd) {
		return "", fmt.Errorf("path %q is relative and the request carries no absolute caller working directory; pass an absolute path or upgrade the client", requestedPath)
	}
	return filepath.Join(cwd, trimmed), nil
}

func canonicalizePath(requestedPath string) (string, error) {
	// Reject an empty or whitespace path early; "" must never silently
	// resolve to any directory.
	if strings.TrimSpace(requestedPath) == "" {
		return "", errors.New("codebase path is required")
	}
	if strings.Contains(requestedPath, "://") {
		return "", fmt.Errorf("path %q looks like a URI; pass a filesystem directory instead", requestedPath)
	}
	// A relative path reaching the daemon is unresolvable here: the daemon's
	// working directory is never the caller's. resolveRequestPath at the gRPC
	// boundary joins relative paths against the caller's cwd before this point.
	if !filepath.IsAbs(requestedPath) {
		return "", fmt.Errorf("path %q is relative; pass an absolute path or send caller_cwd", requestedPath)
	}
	absolutePath := filepath.Clean(requestedPath)
	canonicalPath, err := filepath.EvalSymlinks(absolutePath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return absolutePath, nil
		}
		slog.Error("resolve symlinks failed", "path", absolutePath, "err", err)
		return "", fmt.Errorf("resolve symlinks for %s: %w", absolutePath, err)
	}
	return canonicalPath, nil
}

// subtreePrefix returns the covering-codebase-relative directory that
// requestedPath points at, in forward-slash form, or "" when the request
// targets the codebase root itself or cannot be resolved. Search uses it to
// scope a query aimed at a nested directory of a larger covering index to only
// that directory's chunks.
func subtreePrefix(requestedPath string, coveringRoot string) string {
	canonical, err := canonicalizePath(requestedPath)
	if err != nil {
		return ""
	}
	coveringRoot = filepath.Clean(coveringRoot)
	if canonical == coveringRoot || !pathCovers(coveringRoot, canonical) {
		return ""
	}
	relative, err := filepath.Rel(coveringRoot, canonical)
	if err != nil || relative == "" || relative == "." {
		return ""
	}
	return filepath.ToSlash(relative)
}

func pathCovers(rootPath string, targetPath string) bool {
	rootPath = filepath.Clean(rootPath)
	targetPath = filepath.Clean(targetPath)
	if rootPath == targetPath {
		return true
	}
	prefixWithSeparator := rootPath + string(filepath.Separator)
	return strings.HasPrefix(targetPath, prefixWithSeparator)
}

// Package gitworktree derives git worktree topology directly from the on-disk
// .git pointer files, without shelling out to git. It answers three questions
// the daemon needs: which worktree root contains a path, what repo group
// (shared common dir) a worktree belongs to, and which sibling worktree roots
// share that group. The .git layout is git's own source of truth, so reading
// it live always reflects the current set of worktrees without a daemon
// restart.
package gitworktree

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
)

const gitdirPrefix = "gitdir:"

const refPrefix = "ref:"

const headsPrefix = "refs/heads/"

// Info describes the worktree that contains a queried path.
type Info struct {
	// WorktreeRoot is the resolved absolute path of the worktree's top
	// directory (the directory that holds the .git entry).
	WorktreeRoot string
	// CommonDir is the resolved absolute path of the shared git common
	// directory. It is the stable identity of the repo group: every worktree
	// of the same repository resolves to the same CommonDir.
	CommonDir string
	// Branch is the short branch name the worktree has checked out, or empty
	// when the worktree is in detached-HEAD state.
	Branch string
	// Detached is true when the worktree's HEAD points at a commit rather than
	// a branch.
	Detached bool
	// Head is the raw HEAD target: the full ref for an attached worktree or the
	// commit object name for a detached one.
	Head string
	// Linked is true for a linked worktree (a .git file) and false for the
	// main worktree (a .git directory).
	Linked bool
}

// Resolve walks up from absPath to the nearest ancestor that has a .git entry
// and returns that worktree's Info. The second return is false when no .git is
// found on the way to the filesystem root, which is the non-git case the caller
// treats as "behave exactly as today".
func Resolve(absPath string) (Info, bool) {
	var empty Info
	root, found := worktreeRootOf(absPath)
	if !found {
		return empty, false
	}
	gitDirPath, linked, ok := gitDirForRoot(root)
	if !ok {
		return empty, false
	}
	commonDir, ok := commonDirFor(gitDirPath)
	if !ok {
		return empty, false
	}
	resolvedRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return empty, false
	}
	branch, detached, head := parseHead(gitDirPath)
	return Info{
		WorktreeRoot: resolvedRoot,
		CommonDir:    commonDir,
		Branch:       branch,
		Detached:     detached,
		Head:         head,
		Linked:       linked,
	}, true
}

// CommonDirAt returns the resolved common directory for the .git entry located
// directly in dir, without walking up. The second return is false when dir has
// no .git entry or the pointer cannot be parsed. The discovery boundary uses it
// to decide whether a nested directory is a worktree of the same repo as the
// codebase being indexed.
func CommonDirAt(dir string) (string, bool) {
	gitDirPath, _, ok := gitDirForRoot(dir)
	if !ok {
		return "", false
	}
	return commonDirFor(gitDirPath)
}

// WorktreeOfRepo reports whether dir is itself the root of a worktree belonging
// to the repository identified by commonDir. It reads dir's own .git entry (no
// upwalk) and compares its resolved common dir to commonDir, so a worktree of
// the same repo matches while a submodule (a different common dir) and a plain
// or non-git directory do not. This is the form a file scanner asks while
// descending a tree.
func WorktreeOfRepo(dir string, commonDir string) bool {
	if commonDir == "" {
		return false
	}
	dirCommon, ok := CommonDirAt(dir)
	return ok && dirCommon == commonDir
}

// PathInsideNestedWorktree reports whether path lies inside a worktree of the
// repository identified by rootCommonDir that is itself nested below rootPath.
// It walks the directories from path up toward rootPath and returns true at the
// first level that is a worktree of the same repo. rootPath itself is excluded
// from the walk, so the repository's own .git never matches; a submodule
// resolves to a different common dir and never matches. This is the form an
// event handler asks about an arbitrary file path, the watcher's boundary.
func PathInsideNestedWorktree(rootPath string, rootCommonDir string, path string) bool {
	if rootCommonDir == "" {
		return false
	}
	rootClean := filepath.Clean(rootPath)
	current := filepath.Clean(path)
	for current != rootClean && strings.HasPrefix(current, rootClean+string(os.PathSeparator)) {
		if WorktreeOfRepo(current, rootCommonDir) {
			return true
		}
		current = filepath.Dir(current)
	}
	return false
}

// PathInsideSubmodule reports whether path lies inside a child git repository
// below rootPath whose common dir differs from rootCommonDir. It returns the
// child repository root relative to rootPath, slash-separated. rootPath itself is
// excluded from the walk, so indexing a submodule as its own codebase still keeps
// that root in scope.
func PathInsideSubmodule(rootPath string, rootCommonDir string, path string) (string, bool) {
	if rootCommonDir == "" {
		return "", false
	}
	rootClean := filepath.Clean(rootPath)
	current := filepath.Clean(path)
	for current != rootClean && strings.HasPrefix(current, rootClean+string(os.PathSeparator)) {
		currentCommon, ok := CommonDirAt(current)
		if ok && currentCommon != rootCommonDir {
			relativeRoot, err := filepath.Rel(rootClean, current)
			if err != nil {
				return "", false
			}
			return filepath.ToSlash(relativeRoot), true
		}
		current = filepath.Dir(current)
	}
	return "", false
}

// WorktreeTracked reports whether commonDir still holds a linked-worktree admin
// entry for worktreeRoot. git records one pointer per linked worktree at
// commonDir/worktrees/<name>/gitdir, and `git worktree remove` deletes it, so a
// false result is the positive "this worktree was removed" signal. It inspects
// the admin entry, not the directory, so a transiently-missing live worktree
// (its directory unmounted while the admin entry stays) still reports tracked
// and is never mistaken for a removal.
func WorktreeTracked(commonDir string, worktreeRoot string) bool {
	if commonDir == "" {
		return false
	}
	entries, err := os.ReadDir(filepath.Join(commonDir, "worktrees"))
	if err != nil {
		return false
	}
	wantGitDir := filepath.Clean(filepath.Join(worktreeRoot, ".git"))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		target, ok := readTrimmed(filepath.Join(commonDir, "worktrees", entry.Name(), "gitdir"))
		if !ok {
			continue
		}
		if filepath.Clean(target) == wantGitDir {
			return true
		}
	}
	return false
}

// SiblingWorktreeRoots returns the resolved worktree roots that share
// commonDir, including the main worktree, sorted. Roots whose directory no
// longer exists on disk are omitted, so a stale pointer (a worktree moved
// without `git worktree repair`) contributes nothing rather than a bad path.
func SiblingWorktreeRoots(commonDir string) []string {
	roots := make([]string, 0)
	seen := make(map[string]struct{})

	addRoot := func(candidate string) {
		resolvedRoot, err := filepath.EvalSymlinks(candidate)
		if err != nil {
			return
		}
		if _, dup := seen[resolvedRoot]; dup {
			return
		}
		seen[resolvedRoot] = struct{}{}
		roots = append(roots, resolvedRoot)
	}

	// The main worktree root is the parent of a standard ".git" common dir.
	if filepath.Base(commonDir) == ".git" {
		addRoot(filepath.Dir(commonDir))
	}

	entries, err := os.ReadDir(filepath.Join(commonDir, "worktrees"))
	if err == nil {
		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}
			gitdirFile := filepath.Join(commonDir, "worktrees", entry.Name(), "gitdir")
			target, ok := readTrimmed(gitdirFile)
			if !ok || target == "" {
				continue
			}
			// gitdir points at the worktree's own .git pointer; its directory is
			// the worktree root.
			addRoot(filepath.Dir(target))
		}
	}

	slices.Sort(roots)
	return roots
}

// worktreeRootOf walks up from start to the nearest ancestor that holds a .git
// entry, returning that directory. It stops at the filesystem root.
func worktreeRootOf(start string) (string, bool) {
	current := filepath.Clean(start)
	for {
		if _, err := os.Lstat(filepath.Join(current, ".git")); err == nil {
			return current, true
		}
		parent := filepath.Dir(current)
		if parent == current {
			return "", false
		}
		current = parent
	}
}

// gitDirForRoot resolves the git directory referenced by the .git entry that
// lives directly in root. For the main worktree the .git entry is a directory
// and is itself the git directory; for a linked worktree or a submodule it is a
// file whose "gitdir:" line points elsewhere. linked is true only for the file
// form.
func gitDirForRoot(root string) (gitDirPath string, linked bool, ok bool) {
	entryPath := filepath.Join(root, ".git")
	info, err := os.Lstat(entryPath)
	if err != nil {
		return "", false, false
	}
	if info.IsDir() {
		return entryPath, false, true
	}
	target, ok := readTrimmed(entryPath)
	if !ok {
		return "", false, false
	}
	target = strings.TrimSpace(strings.TrimPrefix(target, gitdirPrefix))
	if target == "" {
		return "", false, false
	}
	if !filepath.IsAbs(target) {
		target = filepath.Join(root, target)
	}
	return filepath.Clean(target), true, true
}

// commonDirFor resolves the shared common directory for a git directory. A
// linked worktree's git directory carries a "commondir" file pointing back at
// the shared .git; a main worktree or a submodule git directory has none and is
// its own common dir.
func commonDirFor(gitDirPath string) (string, bool) {
	commonFile := filepath.Join(gitDirPath, "commondir")
	if target, ok := readTrimmed(commonFile); ok && target != "" {
		if !filepath.IsAbs(target) {
			target = filepath.Join(gitDirPath, target)
		}
		resolved, evalErr := filepath.EvalSymlinks(filepath.Clean(target))
		if evalErr != nil {
			return "", false
		}
		return resolved, true
	}
	resolved, err := filepath.EvalSymlinks(gitDirPath)
	if err != nil {
		return "", false
	}
	return resolved, true
}

// parseHead reads the HEAD file inside a git directory and classifies it as an
// attached branch or a detached commit.
func parseHead(gitDirPath string) (branch string, detached bool, head string) {
	content, ok := readTrimmed(filepath.Join(gitDirPath, "HEAD"))
	if !ok || content == "" {
		return "", false, ""
	}
	if refTarget, isRef := strings.CutPrefix(content, refPrefix); isRef {
		ref := strings.TrimSpace(refTarget)
		return strings.TrimPrefix(ref, headsPrefix), false, ref
	}
	return "", true, content
}

// readTrimmed reads a small pointer file and trims surrounding whitespace. The
// boolean reports whether the read succeeded; every caller treats a missing or
// unreadable pointer as "not a worktree here" rather than a hard error, so no
// error value is surfaced.
func readTrimmed(path string) (string, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	return strings.TrimSpace(string(data)), true
}

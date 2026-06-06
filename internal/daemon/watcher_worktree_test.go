package daemon

import "testing"

// TestCoveredByNestedWorktree proves the watcher dispatch boundary: a parent
// root yields a path to a nested same-repo worktree, but keeps indexing a plain
// nested subdir (intentional overlap), and a non-git parent never yields.
func TestCoveredByNestedWorktree(t *testing.T) {
	t.Parallel()
	parent := watchRoot{codebaseID: "parent", root: "/repo", commonDir: "/repo/.git"}
	nestedWorktree := watchRoot{codebaseID: "wt", root: "/repo/.claude/worktrees/wt", commonDir: "/repo/.git"}

	worktreeCovers := []watchRoot{nestedWorktree, parent}
	if !coveredByNestedWorktree(parent, worktreeCovers) {
		t.Error("parent should yield a worktree path to the nested same-repo worktree")
	}
	if coveredByNestedWorktree(nestedWorktree, worktreeCovers) {
		t.Error("the nested worktree owns its own files; it should not yield")
	}

	// A plain subdir indexed as its own codebase has no common dir, so the parent
	// must still index its files; overlap is intentional there.
	plainSub := watchRoot{codebaseID: "sub", root: "/repo/sub", commonDir: ""}
	if coveredByNestedWorktree(parent, []watchRoot{plainSub, parent}) {
		t.Error("a plain nested subdir is not a worktree boundary; parent must still index it")
	}

	// A non-git parent (no common dir) never yields.
	nonGitParent := watchRoot{codebaseID: "p2", root: "/plain", commonDir: ""}
	nonGitChild := watchRoot{codebaseID: "c", root: "/plain/child", commonDir: ""}
	if coveredByNestedWorktree(nonGitParent, []watchRoot{nonGitChild, nonGitParent}) {
		t.Error("a non-git parent should never yield")
	}
}

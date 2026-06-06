package daemon

import (
	"path/filepath"
	"strings"
	"testing"
)

// TestRenderWorktreeRelationLinkedWorktree proves the status renderer names the
// main checkout and branch for a linked worktree path.
func TestRenderWorktreeRelationLinkedWorktree(t *testing.T) {
	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(base, "feature")
	makeLinkedWorktree(t, mainRoot, "feature", worktreeDir, "feature")

	line := renderWorktreeRelation(worktreeDir)
	if line == "" {
		t.Fatal("renderWorktreeRelation returned empty for a linked worktree")
	}
	if !strings.Contains(line, "git worktree of") {
		t.Errorf("worktree line = %q, want it to name the relationship", line)
	}
	if !strings.Contains(line, evalSym(t, mainRoot)) {
		t.Errorf("worktree line = %q, want it to name the main checkout %q", line, evalSym(t, mainRoot))
	}
	if !strings.Contains(line, "feature") {
		t.Errorf("worktree line = %q, want it to name the branch", line)
	}
}

// TestRenderWorktreeRelationMainAndNonGit proves the renderer stays silent for a
// main worktree and for a non-git directory, so ordinary status output is
// unchanged.
func TestRenderWorktreeRelationMainAndNonGit(t *testing.T) {
	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	if line := renderWorktreeRelation(mainRoot); line != "" {
		t.Errorf("main worktree produced a relation line %q, want empty", line)
	}

	plain := t.TempDir()
	if line := renderWorktreeRelation(plain); line != "" {
		t.Errorf("non-git directory produced a relation line %q, want empty", line)
	}
}

// TestRenderWorktreeRelationDetached proves a detached-HEAD worktree renders its
// commit instead of a branch name.
func TestRenderWorktreeRelationDetached(t *testing.T) {
	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(base, "detached")
	perWorktree := filepath.Join(mainRoot, ".git", "worktrees", "detached")
	writeWorktreeFile(t, filepath.Join(perWorktree, "commondir"), "../..\n")
	writeWorktreeFile(t, filepath.Join(perWorktree, "gitdir"), filepath.Join(worktreeDir, ".git")+"\n")
	writeWorktreeFile(t, filepath.Join(perWorktree, "HEAD"), "0123456789abcdef0123456789abcdef01234567\n")
	writeWorktreeFile(t, filepath.Join(worktreeDir, ".git"), "gitdir: "+perWorktree+"\n")

	line := renderWorktreeRelation(worktreeDir)
	if !strings.Contains(line, "detached") {
		t.Errorf("detached worktree line = %q, want it to mark detached HEAD", line)
	}
	if !strings.Contains(line, "0123456789abcdef") {
		t.Errorf("detached worktree line = %q, want it to name the commit", line)
	}
}

package indexability

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// isolateHome points HOME and GIT_CONFIG_GLOBAL at empty temp locations so the
// global excludes, the git config, and ~/.context/.contextignore contribute
// nothing and the test only sees the rules it writes. It cannot run in parallel
// because it sets process environment.
func isolateHome(t *testing.T) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("GIT_CONFIG_GLOBAL", filepath.Join(home, "absent-gitconfig"))
}

func writeFile(t *testing.T, path string, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir for %s: %v", path, err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func statInfo(t *testing.T, path string) os.FileInfo {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
	return info
}

func assertDecision(t *testing.T, got Decision, wantIndexed bool, wantReason Reason) {
	t.Helper()
	if got.Indexed != wantIndexed || got.Reason != wantReason {
		t.Fatalf("decision = {Indexed:%t Reason:%q}, want {Indexed:%t Reason:%q}", got.Indexed, got.Reason, wantIndexed, wantReason)
	}
}

func TestDecideNestedGitignoreScopesToSubdir(t *testing.T) {
	isolateHome(t)
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "sub", ".gitignore"), "*.tmp\n")
	writeFile(t, filepath.Join(root, "sub", "a.tmp"), "x")
	writeFile(t, filepath.Join(root, "sub", "b.go"), "package b\n")
	writeFile(t, filepath.Join(root, "a.tmp"), "y")

	resolver := NewResolver()
	ctx := context.Background()

	assertDecision(t, resolver.Decide(ctx, "cb", root, "sub/a.tmp", statInfo(t, filepath.Join(root, "sub", "a.tmp"))), false, ReasonIgnored)
	assertDecision(t, resolver.Decide(ctx, "cb", root, "sub/b.go", statInfo(t, filepath.Join(root, "sub", "b.go"))), true, Keep)
	// The same name outside the subdir is unaffected by the subdir rule.
	assertDecision(t, resolver.Decide(ctx, "cb", root, "a.tmp", statInfo(t, filepath.Join(root, "a.tmp"))), true, Keep)
}

func TestDecideNegationReincludesFile(t *testing.T) {
	isolateHome(t)
	root := t.TempDir()
	writeFile(t, filepath.Join(root, ".gitignore"), "*.log\n!keep.log\n")
	writeFile(t, filepath.Join(root, "keep.log"), "x")
	writeFile(t, filepath.Join(root, "skip.log"), "y")

	resolver := NewResolver()
	ctx := context.Background()

	assertDecision(t, resolver.Decide(ctx, "cb", root, "keep.log", statInfo(t, filepath.Join(root, "keep.log"))), true, Keep)
	assertDecision(t, resolver.Decide(ctx, "cb", root, "skip.log", statInfo(t, filepath.Join(root, "skip.log"))), false, ReasonIgnored)
}

func TestDecideDirectoryExclusionWinsOverReinclude(t *testing.T) {
	isolateHome(t)
	root := t.TempDir()
	writeFile(t, filepath.Join(root, ".gitignore"), "build/\n!build/keep.txt\n")
	writeFile(t, filepath.Join(root, "build", "keep.txt"), "x")

	resolver := NewResolver()
	ctx := context.Background()

	// keep.txt cannot be re-included because its parent directory is excluded.
	assertDecision(t, resolver.Decide(ctx, "cb", root, "build/keep.txt", statInfo(t, filepath.Join(root, "build", "keep.txt"))), false, ReasonIgnored)
}

func TestDecideHonorsInfoExclude(t *testing.T) {
	isolateHome(t)
	root := t.TempDir()
	writeFile(t, filepath.Join(root, ".git", "HEAD"), "ref: refs/heads/main\n")
	writeFile(t, filepath.Join(root, ".git", "info", "exclude"), "secret.txt\n")
	writeFile(t, filepath.Join(root, "secret.txt"), "x")
	writeFile(t, filepath.Join(root, "main.go"), "package main\n")

	resolver := NewResolver()
	ctx := context.Background()

	assertDecision(t, resolver.Decide(ctx, "cb", root, "secret.txt", statInfo(t, filepath.Join(root, "secret.txt"))), false, ReasonIgnored)
	assertDecision(t, resolver.Decide(ctx, "cb", root, "main.go", statInfo(t, filepath.Join(root, "main.go"))), true, Keep)
}

// makeNestedWorktree writes a main repo with a shared info/exclude plus a linked
// worktree nested under it, returning the main root and the nested worktree dir.
func makeNestedWorktree(t *testing.T) (string, string) {
	t.Helper()
	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	writeFile(t, filepath.Join(mainRoot, ".git", "HEAD"), "ref: refs/heads/main\n")
	writeFile(t, filepath.Join(mainRoot, ".git", "info", "exclude"), "shared.txt\n")

	name := "wt"
	worktreeDir := filepath.Join(mainRoot, ".claude", "worktrees", name)
	perWorktree := filepath.Join(mainRoot, ".git", "worktrees", name)
	writeFile(t, filepath.Join(perWorktree, "commondir"), "../..\n")
	writeFile(t, filepath.Join(perWorktree, "gitdir"), filepath.Join(worktreeDir, ".git")+"\n")
	writeFile(t, filepath.Join(perWorktree, "HEAD"), "ref: refs/heads/"+name+"\n")
	writeFile(t, filepath.Join(worktreeDir, ".git"), "gitdir: "+perWorktree+"\n")
	writeFile(t, filepath.Join(worktreeDir, "feature.go"), "package feature\n")
	writeFile(t, filepath.Join(worktreeDir, "shared.txt"), "x")
	return mainRoot, worktreeDir
}

func TestDecideLinkedWorktreeResolvesInfoExcludeViaCommonDir(t *testing.T) {
	isolateHome(t)
	_, worktreeDir := makeNestedWorktree(t)

	resolver := NewResolver()
	ctx := context.Background()

	// The linked worktree's .git is a file, so info/exclude lives under the
	// shared common dir; shared.txt must resolve as ignored from there.
	assertDecision(t, resolver.Decide(ctx, "cb_wt", worktreeDir, "shared.txt", statInfo(t, filepath.Join(worktreeDir, "shared.txt"))), false, ReasonIgnored)
	assertDecision(t, resolver.Decide(ctx, "cb_wt", worktreeDir, "feature.go", statInfo(t, filepath.Join(worktreeDir, "feature.go"))), true, Keep)
}

func TestDecideNestedSameRepoWorktreeIsOutOfScope(t *testing.T) {
	isolateHome(t)
	mainRoot, worktreeDir := makeNestedWorktree(t)

	resolver := NewResolver()
	ctx := context.Background()

	relPath := ".claude/worktrees/wt/feature.go"
	assertDecision(t, resolver.Decide(ctx, "cb_main", mainRoot, relPath, statInfo(t, filepath.Join(worktreeDir, "feature.go"))), false, ReasonOutOfScope)
}

func TestDecideContentDenylistAndStatGates(t *testing.T) {
	isolateHome(t)
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "image.png"), "x")
	writeFile(t, filepath.Join(root, "main.go"), "package main\n")
	if err := os.MkdirAll(filepath.Join(root, "src"), 0o755); err != nil {
		t.Fatalf("mkdir src: %v", err)
	}

	resolver := NewResolver()
	ctx := context.Background()

	assertDecision(t, resolver.Decide(ctx, "cb", root, "image.png", statInfo(t, filepath.Join(root, "image.png"))), false, ReasonIgnored)
	assertDecision(t, resolver.Decide(ctx, "cb", root, "main.go", statInfo(t, filepath.Join(root, "main.go"))), true, Keep)
	assertDecision(t, resolver.Decide(ctx, "cb", root, "src", statInfo(t, filepath.Join(root, "src"))), false, ReasonNotRegular)
}

func TestDecideExcludesGitDirAsOutOfScope(t *testing.T) {
	isolateHome(t)
	root := t.TempDir()
	writeFile(t, filepath.Join(root, ".git", "index"), "x")
	writeFile(t, filepath.Join(root, ".gitignore"), "secret/\n")
	writeFile(t, filepath.Join(root, "pkg", "main.go"), "package pkg\n")

	resolver := NewResolver()
	ctx := context.Background()

	// .git content is git metadata and is never indexed, even though .gitignore
	// never lists .git (git excludes its own directory implicitly).
	assertDecision(t, resolver.Decide(ctx, "cb", root, ".git/index", statInfo(t, filepath.Join(root, ".git", "index"))), false, ReasonOutOfScope)
	// .gitignore itself is tracked content and stays indexable.
	assertDecision(t, resolver.Decide(ctx, "cb", root, ".gitignore", statInfo(t, filepath.Join(root, ".gitignore"))), true, Keep)
	assertDecision(t, resolver.Decide(ctx, "cb", root, "pkg/main.go", statInfo(t, filepath.Join(root, "pkg", "main.go"))), true, Keep)
}

func TestDecideOversizeRejectsLargeFile(t *testing.T) {
	isolateHome(t)
	t.Setenv("INDEX_MAX_FILE_BYTES", "8")
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "big.txt"), "0123456789")

	resolver := NewResolver()
	ctx := context.Background()

	assertDecision(t, resolver.Decide(ctx, "cb", root, "big.txt", statInfo(t, filepath.Join(root, "big.txt"))), false, ReasonOversize)
}

func TestInvalidateRulesPicksUpDiskChange(t *testing.T) {
	isolateHome(t)
	root := t.TempDir()
	writeFile(t, filepath.Join(root, ".gitignore"), "*.tmp\n")
	writeFile(t, filepath.Join(root, "x.tmp"), "x")

	resolver := NewResolver()
	ctx := context.Background()

	// First call resolves and caches the rule that ignores *.tmp.
	assertDecision(t, resolver.Decide(ctx, "cb", root, "x.tmp", statInfo(t, filepath.Join(root, "x.tmp"))), false, ReasonIgnored)

	// Remove the rule on disk; the cached matcher still ignores the file.
	writeFile(t, filepath.Join(root, ".gitignore"), "\n")
	assertDecision(t, resolver.Decide(ctx, "cb", root, "x.tmp", statInfo(t, filepath.Join(root, "x.tmp"))), false, ReasonIgnored)

	// After invalidation the rebuilt matcher reflects the emptied .gitignore.
	resolver.InvalidateRules("cb")
	assertDecision(t, resolver.Decide(ctx, "cb", root, "x.tmp", statInfo(t, filepath.Join(root, "x.tmp"))), true, Keep)
}

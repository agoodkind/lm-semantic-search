package discovery

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"goodkind.io/lm-semantic-search/internal/indexability"
)

func writeFile(t *testing.T, path string, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir parent for %s: %v", path, err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func mkdir(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", path, err)
	}
}

// isolateHome points HOME and GIT_CONFIG_GLOBAL at empty temp locations so the
// resolver's global excludes, git config, and ~/.context/.contextignore add
// nothing and the walk only sees the rules the test writes. It sets process
// environment, so a test that uses it must not run in parallel.
func isolateHome(t *testing.T) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("GIT_CONFIG_GLOBAL", filepath.Join(home, "absent-gitconfig"))
}

func relativeFiles(t *testing.T, root string, files []string) []string {
	t.Helper()
	out := make([]string, 0, len(files))
	for _, file := range files {
		rel, err := filepath.Rel(root, file)
		if err != nil {
			t.Fatalf("rel for %s: %v", file, err)
		}
		out = append(out, filepath.ToSlash(rel))
	}
	slices.Sort(out)
	return out
}

// TestDiscoverRoutesIgnoreDecisionsThroughResolver proves the walk lists only
// the files the indexability resolver keeps in scope: a directory ignore, a
// glob ignore, and a nested .gitignore all prune their matches, while the
// .gitignore files themselves and the tracked source files remain.
func TestDiscoverRoutesIgnoreDecisionsThroughResolver(t *testing.T) {
	isolateHome(t)
	tempDir := t.TempDir()
	writeFile(t, filepath.Join(tempDir, ".gitignore"), "ignored-dir/\n*.tmp\n")
	mkdir(t, filepath.Join(tempDir, "ignored-dir"))
	writeFile(t, filepath.Join(tempDir, "ignored-dir", "inside.go"), "package x\n")
	mkdir(t, filepath.Join(tempDir, "nested"))
	writeFile(t, filepath.Join(tempDir, "nested", ".gitignore"), "local-only.go\n")
	writeFile(t, filepath.Join(tempDir, "nested", "local-only.go"), "package x\n")
	writeFile(t, filepath.Join(tempDir, "nested", "kept.go"), "package x\n")
	writeFile(t, filepath.Join(tempDir, "kept.go"), "package x\n")
	writeFile(t, filepath.Join(tempDir, "scratch.tmp"), "x\n")

	result, err := Discover(context.Background(), indexability.NewResolver(nil, nil), "cb", tempDir)
	if err != nil {
		t.Fatalf("Discover returned error: %v", err)
	}

	got := relativeFiles(t, tempDir, result.Files)
	want := []string{".gitignore", "kept.go", "nested/.gitignore", "nested/kept.go"}
	if !slices.Equal(got, want) {
		t.Fatalf("Discover files = %v, want %v", got, want)
	}
}

// TestDiscoverDoesNotDescendIntoIgnoredDirectories proves the walk prunes:
// an unreadable directory that the rules exclude must not fail discovery.
func TestDiscoverDoesNotDescendIntoIgnoredDirectories(t *testing.T) {
	isolateHome(t)
	tempDir := t.TempDir()
	writeFile(t, filepath.Join(tempDir, ".gitignore"), "sealed/\n")
	mkdir(t, filepath.Join(tempDir, "sealed"))
	writeFile(t, filepath.Join(tempDir, "sealed", "file.go"), "package x\n")
	if err := os.Chmod(filepath.Join(tempDir, "sealed"), 0o000); err != nil {
		t.Fatalf("chmod: %v", err)
	}
	t.Cleanup(func() { _ = os.Chmod(filepath.Join(tempDir, "sealed"), 0o755) })

	if _, err := Discover(context.Background(), indexability.NewResolver(nil, nil), "cb", tempDir); err != nil {
		t.Fatalf("Discover failed on a pruned unreadable directory: %v", err)
	}
}

// TestDiscoverExcludesNestedSameRepoWorktree proves the walk stops at a nested
// directory that is a git worktree of the same repository as the codebase root,
// so worktree files never leak into the parent index.
func TestDiscoverExcludesNestedSameRepoWorktree(t *testing.T) {
	isolateHome(t)
	repo := filepath.Join(t.TempDir(), "repo")

	// main worktree
	writeFile(t, filepath.Join(repo, ".git", "HEAD"), "ref: refs/heads/main\n")
	writeFile(t, filepath.Join(repo, "main.go"), "package main\n")

	// nested linked worktree of the same repo at repo/wt
	perWorktree := filepath.Join(repo, ".git", "worktrees", "foo")
	writeFile(t, filepath.Join(perWorktree, "commondir"), "../..\n")
	writeFile(t, filepath.Join(perWorktree, "gitdir"), filepath.Join(repo, "wt", ".git")+"\n")
	writeFile(t, filepath.Join(perWorktree, "HEAD"), "ref: refs/heads/feature\n")
	writeFile(t, filepath.Join(repo, "wt", ".git"), "gitdir: "+perWorktree+"\n")
	writeFile(t, filepath.Join(repo, "wt", "feature.go"), "package feature\n")

	// nested submodule (a different common dir) at repo/extern/lib
	moduleGitDir := filepath.Join(repo, ".git", "modules", "lib")
	writeFile(t, filepath.Join(moduleGitDir, "HEAD"), "ref: refs/heads/main\n")
	writeFile(t, filepath.Join(repo, "extern", "lib", ".git"), "gitdir: "+moduleGitDir+"\n")
	writeFile(t, filepath.Join(repo, "extern", "lib", "sub.go"), "package lib\n")

	result, err := Discover(context.Background(), indexability.NewResolver(nil, nil), "cb", repo)
	if err != nil {
		t.Fatalf("Discover returned error: %v", err)
	}

	got := make(map[string]bool)
	for _, file := range result.Files {
		rel, relErr := filepath.Rel(repo, file)
		if relErr != nil {
			t.Fatalf("rel for %s: %v", file, relErr)
		}
		got[filepath.ToSlash(rel)] = true
	}

	if !got["main.go"] {
		t.Errorf("main.go missing from discovery; got %v", got)
	}
	if got["wt/feature.go"] {
		t.Errorf("nested worktree file wt/feature.go must be excluded; got %v", got)
	}
	if got["wt/.git"] {
		t.Errorf("nested worktree .git pointer must never be indexed; got %v", got)
	}
}

func TestDiscoverPrunesSubmoduleByDefault(t *testing.T) {
	isolateHome(t)
	repo := filepath.Join(t.TempDir(), "repo")
	writeFile(t, filepath.Join(repo, ".git", "HEAD"), "ref: refs/heads/main\n")
	writeFile(t, filepath.Join(repo, ".gitmodules"), "[submodule \"lib\"]\n\tpath = extern/lib\n\turl = ../lib.git\n")
	writeFile(t, filepath.Join(repo, "main.go"), "package main\n")

	moduleGitDir := filepath.Join(repo, ".git", "modules", "extern", "lib")
	writeFile(t, filepath.Join(moduleGitDir, "HEAD"), "ref: refs/heads/main\n")
	writeFile(t, filepath.Join(repo, "extern", "lib", ".git"), "gitdir: "+moduleGitDir+"\n")
	writeFile(t, filepath.Join(repo, "extern", "lib", ".gitignore"), "*.tmp\n")
	writeFile(t, filepath.Join(repo, "extern", "lib", "sub.go"), "package lib\n")

	result, err := Discover(context.Background(), indexability.NewResolver(nil, nil), "cb", repo)
	if err != nil {
		t.Fatalf("Discover returned error: %v", err)
	}

	got := relativeFiles(t, repo, result.Files)
	want := []string{".gitmodules", "main.go"}
	if !slices.Equal(got, want) {
		t.Fatalf("Discover files = %v, want %v", got, want)
	}
}

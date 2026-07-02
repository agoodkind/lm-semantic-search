package indexability

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// isolateHome points HOME, XDG_CONFIG_HOME, and GIT_CONFIG_GLOBAL at empty temp
// locations so the global excludes, the git config, and ~/.context/.contextignore
// contribute nothing and the test only sees the rules it writes. Pinning
// XDG_CONFIG_HOME under the temp home keeps the resolver's global-excludes path
// (which honors XDG_CONFIG_HOME, like git) HOME-derived, so a runner that exports
// XDG_CONFIG_HOME cannot leak its real ~/.config into the test. It cannot run in
// parallel because it sets process environment.
func isolateHome(t *testing.T) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(home, ".config"))
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

func makeGitRoot(t *testing.T, root string) {
	t.Helper()
	writeFile(t, filepath.Join(root, ".git", "HEAD"), "ref: refs/heads/main\n")
}

func makeSubmodule(t *testing.T, root string, relPath string, name string) {
	t.Helper()
	makeGitRoot(t, root)
	gitmodules := "[submodule \"" + name + "\"]\n\tpath = " + relPath + "\n\turl = ../" + name + ".git\n"
	writeFile(t, filepath.Join(root, ".gitmodules"), gitmodules)
	moduleGitDir := filepath.Join(root, ".git", "modules", filepath.FromSlash(relPath))
	writeFile(t, filepath.Join(moduleGitDir, "HEAD"), "ref: refs/heads/main\n")
	writeFile(t, filepath.Join(root, filepath.FromSlash(relPath), ".git"), "gitdir: "+moduleGitDir+"\n")
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

	resolver := NewResolver(nil, nil)
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

	resolver := NewResolver(nil, nil)
	ctx := context.Background()

	assertDecision(t, resolver.Decide(ctx, "cb", root, "keep.log", statInfo(t, filepath.Join(root, "keep.log"))), true, Keep)
	assertDecision(t, resolver.Decide(ctx, "cb", root, "skip.log", statInfo(t, filepath.Join(root, "skip.log"))), false, ReasonIgnored)
}

func TestDecideDirectoryExclusionWinsOverReinclude(t *testing.T) {
	isolateHome(t)
	root := t.TempDir()
	writeFile(t, filepath.Join(root, ".gitignore"), "build/\n!build/keep.txt\n")
	writeFile(t, filepath.Join(root, "build", "keep.txt"), "x")

	resolver := NewResolver(nil, nil)
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

	resolver := NewResolver(nil, nil)
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

	resolver := NewResolver(nil, nil)
	ctx := context.Background()

	// The linked worktree's .git is a file, so info/exclude lives under the
	// shared common dir; shared.txt must resolve as ignored from there.
	assertDecision(t, resolver.Decide(ctx, "cb_wt", worktreeDir, "shared.txt", statInfo(t, filepath.Join(worktreeDir, "shared.txt"))), false, ReasonIgnored)
	assertDecision(t, resolver.Decide(ctx, "cb_wt", worktreeDir, "feature.go", statInfo(t, filepath.Join(worktreeDir, "feature.go"))), true, Keep)
}

func TestDecideNestedSameRepoWorktreeIsOutOfScope(t *testing.T) {
	isolateHome(t)
	mainRoot, worktreeDir := makeNestedWorktree(t)

	resolver := NewResolver(nil, nil)
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

	resolver := NewResolver(nil, nil)
	ctx := context.Background()

	assertDecision(t, resolver.Decide(ctx, "cb", root, "image.png", statInfo(t, filepath.Join(root, "image.png"))), false, ReasonIgnored)
	assertDecision(t, resolver.Decide(ctx, "cb", root, "main.go", statInfo(t, filepath.Join(root, "main.go"))), true, Keep)
	assertDecision(t, resolver.Decide(ctx, "cb", root, "src", statInfo(t, filepath.Join(root, "src"))), false, ReasonNotRegular)
}

// TestDecideContentGate proves DecideContent wraps the post-read content gate:
// valid UTF-8 stays indexable while invalid UTF-8 is rejected with the content
// reason, so callers route the content stage through the resolver.
func TestDecideContentGate(t *testing.T) {
	t.Parallel()
	resolver := NewResolver(nil, nil)

	assertDecision(t, resolver.DecideContent([]byte("package main\n")), true, Keep)
	assertDecision(t, resolver.DecideContent([]byte{'g', 'o', 0xff}), false, ReasonNonUTF8)
}

// TestIgnoreSourcesListsReadOrderAndPrunes proves IgnoreSources reports the one
// ordered set of ignore-source files the resolver reads: the global excludes,
// the contextignore path, the repo info/exclude, then each nested .gitignore the
// pruned walk visits. A .gitignore inside a directory the rules already exclude
// is pruned, so it is absent from the set.
func TestIgnoreSourcesListsReadOrderAndPrunes(t *testing.T) {
	isolateHome(t)
	root := t.TempDir()
	writeFile(t, filepath.Join(root, ".git", "HEAD"), "ref: refs/heads/main\n")
	writeFile(t, filepath.Join(root, ".git", "info", "exclude"), "secret.txt\n")
	writeFile(t, filepath.Join(root, ".gitignore"), "ignored_dir/\n")
	writeFile(t, filepath.Join(root, "sub", ".gitignore"), "*.tmp\n")
	writeFile(t, filepath.Join(root, "sub", "keep.go"), "package sub\n")
	writeFile(t, filepath.Join(root, "ignored_dir", ".gitignore"), "*.x\n")

	resolver := NewResolver(nil, nil)
	sources := resolver.IgnoreSources(context.Background(), "cb", root)

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("UserHomeDir returned error: %v", err)
	}
	wantPresent := []string{
		filepath.Join(home, ".config", "git", "ignore"),
		filepath.Join(home, ".context", ".contextignore"),
		filepath.Join(root, ".gitignore"),
		filepath.Join(root, "sub", ".gitignore"),
	}
	for _, want := range wantPresent {
		if !slices.Contains(sources, want) {
			t.Fatalf("IgnoreSources = %v, want to contain %q", sources, want)
		}
	}

	pruned := filepath.Join(root, "ignored_dir", ".gitignore")
	if slices.Contains(sources, pruned) {
		t.Fatalf("IgnoreSources = %v, want pruned .gitignore %q absent", sources, pruned)
	}

	hasInfoExclude := false
	for _, source := range sources {
		if strings.HasSuffix(filepath.ToSlash(source), "info/exclude") {
			hasInfoExclude = true
		}
	}
	if !hasInfoExclude {
		t.Fatalf("IgnoreSources = %v, want to contain a repo info/exclude path", sources)
	}

	// The root .gitignore precedes the nested sub/.gitignore, mirroring the
	// top-down walk order buildRules relies on (last match wins).
	rootIndex := slices.Index(sources, filepath.Join(root, ".gitignore"))
	subIndex := slices.Index(sources, filepath.Join(root, "sub", ".gitignore"))
	if rootIndex < 0 || subIndex < 0 || rootIndex > subIndex {
		t.Fatalf("IgnoreSources order = %v, want root .gitignore before sub/.gitignore", sources)
	}
}

func TestDecideExcludesGitDirAsOutOfScope(t *testing.T) {
	isolateHome(t)
	root := t.TempDir()
	writeFile(t, filepath.Join(root, ".git", "index"), "x")
	writeFile(t, filepath.Join(root, ".gitignore"), "secret/\n")
	writeFile(t, filepath.Join(root, "pkg", "main.go"), "package pkg\n")

	resolver := NewResolver(nil, nil)
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

	resolver := NewResolver(nil, nil)
	ctx := context.Background()

	assertDecision(t, resolver.Decide(ctx, "cb", root, "big.txt", statInfo(t, filepath.Join(root, "big.txt"))), false, ReasonOversize)
}

func TestInvalidateRulesPicksUpDiskChange(t *testing.T) {
	isolateHome(t)
	root := t.TempDir()
	writeFile(t, filepath.Join(root, ".gitignore"), "*.tmp\n")
	writeFile(t, filepath.Join(root, "x.tmp"), "x")

	resolver := NewResolver(nil, nil)
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

func TestGlobalExcludePathsHonorsXDGConfigHome(t *testing.T) {
	t.Setenv("HOME", filepath.Join(t.TempDir(), "home"))
	t.Setenv("GIT_CONFIG_GLOBAL", filepath.Join(t.TempDir(), "absent-gitconfig"))
	t.Setenv("XDG_CONFIG_HOME", "/xdg")

	paths := globalExcludePaths(context.Background())
	if len(paths) == 0 || paths[0] != filepath.Join("/xdg", "git", "ignore") {
		t.Fatalf("globalExcludePaths = %v, want first %q", paths, "/xdg/git/ignore")
	}
}

func TestUnquoteConfigValueStripsMatchingQuotes(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{`"~/path"`, "~/path"},
		{`'~/path'`, "~/path"},
		{"~/path", "~/path"},
		{`"mismatch'`, `"mismatch'`},
		{`"`, `"`},
	}
	for _, testCase := range cases {
		if got := unquoteConfigValue(testCase.in); got != testCase.want {
			t.Errorf("unquoteConfigValue(%q) = %q, want %q", testCase.in, got, testCase.want)
		}
	}
}

// TestResolverAppliesCustomOverrides proves a pattern that exists only in the
// per-codebase override provider excludes a path no other ignore source touches,
// across Decide, Ignored, and IgnoreDetail, and that changing what the provider
// returns plus InvalidateRules makes the next decision reflect the new set.
func TestResolverAppliesCustomOverrides(t *testing.T) {
	isolateHome(t)
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "keep.go"), "package x\n")
	writeFile(t, filepath.Join(root, "custom.secret"), "x")

	var overrides []string
	resolver := NewResolver(func(codebaseID string) []string {
		if codebaseID != "cb" {
			return nil
		}
		return overrides
	}, nil)
	ctx := context.Background()
	secretInfo := statInfo(t, filepath.Join(root, "custom.secret"))

	// No override yet: nothing ignores *.secret, so the file is indexable.
	assertDecision(t, resolver.Decide(ctx, "cb", root, "custom.secret", secretInfo), true, Keep)
	if resolver.Ignored(ctx, "cb", root, "custom.secret", false) {
		t.Fatal("Ignored reported custom.secret excluded before any override was set")
	}

	// Add the override and invalidate; the rebuilt matcher now excludes it.
	overrides = []string{"*.secret"}
	resolver.InvalidateRules("cb")
	assertDecision(t, resolver.Decide(ctx, "cb", root, "custom.secret", secretInfo), false, ReasonIgnored)
	assertDecision(t, resolver.Decide(ctx, "cb", root, "keep.go", statInfo(t, filepath.Join(root, "keep.go"))), true, Keep)
	if !resolver.Ignored(ctx, "cb", root, "custom.secret", false) {
		t.Fatal("Ignored did not report custom.secret excluded after the override was set")
	}
	detail, excluded := resolver.IgnoreDetail(ctx, "cb", root, "custom.secret", false)
	if !excluded {
		t.Fatal("IgnoreDetail did not report custom.secret excluded under the override")
	}
	if detail.Pattern != "*.secret" {
		t.Fatalf("IgnoreDetail pattern = %q, want %q", detail.Pattern, "*.secret")
	}

	// Drop the override and invalidate; the file is indexable again.
	overrides = nil
	resolver.InvalidateRules("cb")
	assertDecision(t, resolver.Decide(ctx, "cb", root, "custom.secret", secretInfo), true, Keep)
}

// TestResolverCustomOverrideBeatsReinclude proves a custom override pattern is
// applied after the repository's .gitignore, so it wins over a re-include rule
// the repository declares for the same path.
func TestResolverCustomOverrideBeatsReinclude(t *testing.T) {
	isolateHome(t)
	root := t.TempDir()
	// The repo ignores *.log but re-includes keep.log; the override re-ignores it.
	writeFile(t, filepath.Join(root, ".gitignore"), "*.log\n!keep.log\n")
	writeFile(t, filepath.Join(root, "keep.log"), "x")

	resolver := NewResolver(func(string) []string { return []string{"keep.log"} }, nil)
	ctx := context.Background()

	// Without the override keep.log would be re-included; the override wins.
	assertDecision(t, resolver.Decide(ctx, "cb", root, "keep.log", statInfo(t, filepath.Join(root, "keep.log"))), false, ReasonIgnored)
}

func TestDecideExcludesDeclaredSubmoduleByDefault(t *testing.T) {
	isolateHome(t)
	root := t.TempDir()
	makeSubmodule(t, root, "third_party/lib", "lib")
	writeFile(t, filepath.Join(root, "third_party", "lib", "lib.go"), "package lib\n")

	resolver := NewResolver(nil, nil)
	ctx := context.Background()

	assertDecision(t, resolver.Decide(ctx, "cb", root, "third_party/lib/lib.go", statInfo(t, filepath.Join(root, "third_party", "lib", "lib.go"))), false, ReasonSubmodule)
	if !resolver.Ignored(ctx, "cb", root, "third_party/lib/lib.go", false) {
		t.Fatal("Ignored returned false for a file inside a default-excluded submodule")
	}
}

func TestDecideAllowsSubmoduleByName(t *testing.T) {
	isolateHome(t)
	root := t.TempDir()
	makeSubmodule(t, root, "third_party/lib", "lib")
	writeFile(t, filepath.Join(root, "third_party", "lib", ".gitignore"), "ignored.go\n")
	writeFile(t, filepath.Join(root, "third_party", "lib", "lib.go"), "package lib\n")
	writeFile(t, filepath.Join(root, "third_party", "lib", "ignored.go"), "package lib\n")

	resolver := NewResolver(nil, func(codebaseID string) []string {
		if codebaseID != "cb" {
			return nil
		}
		return []string{"lib"}
	})
	ctx := context.Background()

	assertDecision(t, resolver.Decide(ctx, "cb", root, "third_party/lib/lib.go", statInfo(t, filepath.Join(root, "third_party", "lib", "lib.go"))), true, Keep)
	assertDecision(t, resolver.Decide(ctx, "cb", root, "third_party/lib/ignored.go", statInfo(t, filepath.Join(root, "third_party", "lib", "ignored.go"))), false, ReasonIgnored)
}

func TestDecideAllowsSubmoduleByPath(t *testing.T) {
	isolateHome(t)
	root := t.TempDir()
	makeSubmodule(t, root, "third_party/lib", "lib")
	writeFile(t, filepath.Join(root, "third_party", "lib", "lib.go"), "package lib\n")

	resolver := NewResolver(nil, func(string) []string { return []string{"third_party/lib"} })
	ctx := context.Background()

	assertDecision(t, resolver.Decide(ctx, "cb", root, "third_party/lib/lib.go", statInfo(t, filepath.Join(root, "third_party", "lib", "lib.go"))), true, Keep)
}

func TestNormalizeSubmoduleTokenRejectsRootEscapesAndAbsolutePaths(t *testing.T) {
	cases := []string{
		"a/..",
		"..",
		"../outside",
		"/tmp/lib",
	}
	for _, value := range cases {
		if got := normalizeSubmoduleToken(value); got != "" {
			t.Fatalf("normalizeSubmoduleToken(%q) = %q, want empty", value, got)
		}
	}
}

func TestParseGitmodulesIgnoresPathOutsideSubmoduleSection(t *testing.T) {
	root := t.TempDir()
	gitmodulesPath := filepath.Join(root, ".gitmodules")
	writeFile(t, gitmodulesPath, "[core]\n\tpath = not-a-submodule\n[submodule \"lib\"]\n\tpath = third_party/lib\n")

	decls := parseGitmodules(context.Background(), gitmodulesPath, "")
	if len(decls) != 1 {
		t.Fatalf("parseGitmodules returned %d decls, want 1: %+v", len(decls), decls)
	}
	if decls[0].name != "lib" || decls[0].path != "third_party/lib" {
		t.Fatalf("parseGitmodules decl = %+v, want lib at third_party/lib", decls[0])
	}
}

func TestDecideRootSubmoduleStillIndexes(t *testing.T) {
	isolateHome(t)
	parent := t.TempDir()
	root := filepath.Join(parent, "modules", "lib")
	moduleGitDir := filepath.Join(parent, ".git", "modules", "lib")
	writeFile(t, filepath.Join(moduleGitDir, "HEAD"), "ref: refs/heads/main\n")
	writeFile(t, filepath.Join(root, ".git"), "gitdir: "+moduleGitDir+"\n")
	writeFile(t, filepath.Join(root, "lib.go"), "package lib\n")

	resolver := NewResolver(nil, nil)
	ctx := context.Background()

	assertDecision(t, resolver.Decide(ctx, "cb", root, "lib.go", statInfo(t, filepath.Join(root, "lib.go"))), true, Keep)
}

func TestIgnoreSourcesIncludesGitmodulesAndPrunesDefaultSubmodule(t *testing.T) {
	isolateHome(t)
	root := t.TempDir()
	makeSubmodule(t, root, "third_party/lib", "lib")
	writeFile(t, filepath.Join(root, "third_party", "lib", ".gitignore"), "*.tmp\n")

	resolver := NewResolver(nil, nil)
	sources := resolver.IgnoreSources(context.Background(), "cb", root)

	if !slices.Contains(sources, filepath.Join(root, ".gitmodules")) {
		t.Fatalf("IgnoreSources = %v, want root .gitmodules", sources)
	}
	if slices.Contains(sources, filepath.Join(root, "third_party", "lib", ".gitignore")) {
		t.Fatalf("IgnoreSources = %v, want default-excluded submodule .gitignore pruned", sources)
	}
}

// TestIgnoreSourcesIncludesAllowlistedSubmoduleGitignore proves IgnoreSources
// lists the .gitignore inside a submodule the codebase includes via the
// allowlist, so the periodic CheckSources backstop watches it. Without this the
// resolver cache could stay stale after a .gitignore edit inside an included
// submodule when the watcher is off or misses the event.
func TestIgnoreSourcesIncludesAllowlistedSubmoduleGitignore(t *testing.T) {
	isolateHome(t)
	root := t.TempDir()
	makeSubmodule(t, root, "third_party/lib", "lib")
	writeFile(t, filepath.Join(root, "third_party", "lib", ".gitignore"), "*.tmp\n")

	resolver := NewResolver(nil, func(string) []string { return []string{"third_party/lib"} })
	sources := resolver.IgnoreSources(context.Background(), "cb", root)

	if !slices.Contains(sources, filepath.Join(root, "third_party", "lib", ".gitignore")) {
		t.Fatalf("IgnoreSources = %v, want allowlisted submodule .gitignore included", sources)
	}
}

// TestIgnoreSourcesReadsGitmodulesOnlyAtRoots proves IgnoreSources consults
// .gitmodules only at the codebase root (and allowlisted submodule roots), not
// in every walked directory. Git itself reads .gitmodules only at a repo root,
// and listing a candidate per directory did a wasted read per dir and doubled
// the watched-source set, so a stray .gitmodules in an ordinary subdirectory is
// neither parsed nor listed.
func TestIgnoreSourcesReadsGitmodulesOnlyAtRoots(t *testing.T) {
	isolateHome(t)
	root := t.TempDir()
	writeFile(t, filepath.Join(root, ".gitmodules"), "[submodule \"lib\"]\n\tpath = third_party/lib\n")
	writeFile(t, filepath.Join(root, "pkg", "code.go"), "package pkg\n")
	writeFile(t, filepath.Join(root, "pkg", ".gitmodules"), "[submodule \"stray\"]\n\tpath = nested\n")

	resolver := NewResolver(nil, nil)
	sources := resolver.IgnoreSources(context.Background(), "cb", root)

	if !slices.Contains(sources, filepath.Join(root, ".gitmodules")) {
		t.Fatalf("IgnoreSources = %v, want root .gitmodules listed", sources)
	}
	if slices.Contains(sources, filepath.Join(root, "pkg", ".gitmodules")) {
		t.Fatalf("IgnoreSources = %v, want stray subdirectory .gitmodules not parsed or listed", sources)
	}
}

package gitworktree

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// fixture builds a main repo plus the given linked worktrees on disk, returning
// the resolved main root and common dir. Each linked worktree maps a name to a
// HEAD line so a test can choose attached or detached state.
type linkedSpec struct {
	name string
	dir  string
	head string
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

func resolved(t *testing.T, path string) string {
	t.Helper()
	out, err := filepath.EvalSymlinks(path)
	if err != nil {
		t.Fatalf("eval symlinks %s: %v", path, err)
	}
	return out
}

// buildRepo lays out a main worktree with a .git directory and zero or more
// linked worktrees registered under .git/worktrees/<name>.
func buildRepo(t *testing.T, mainHead string, linked []linkedSpec) (mainRoot string, commonDir string) {
	t.Helper()
	base := t.TempDir()
	mainRoot = filepath.Join(base, "repo")
	gitDir := filepath.Join(mainRoot, ".git")
	writeFile(t, filepath.Join(gitDir, "HEAD"), mainHead)

	for _, spec := range linked {
		perWorktree := filepath.Join(gitDir, "worktrees", spec.name)
		// commondir points back at the main .git relative to the per-worktree dir.
		writeFile(t, filepath.Join(perWorktree, "commondir"), "../..\n")
		gitFilePath := filepath.Join(spec.dir, ".git")
		writeFile(t, filepath.Join(perWorktree, "gitdir"), gitFilePath+"\n")
		writeFile(t, filepath.Join(perWorktree, "HEAD"), spec.head)
		// the linked worktree's .git file points at the per-worktree dir.
		writeFile(t, gitFilePath, "gitdir: "+perWorktree+"\n")
	}
	return mainRoot, resolved(t, gitDir)
}

func TestResolveMainWorktree(t *testing.T) {
	mainRoot, commonDir := buildRepo(t, "ref: refs/heads/main\n", nil)

	info, ok := Resolve(mainRoot)
	if !ok {
		t.Fatalf("Resolve(%s) returned not-a-worktree", mainRoot)
	}
	if info.WorktreeRoot != resolved(t, mainRoot) {
		t.Errorf("WorktreeRoot = %q, want %q", info.WorktreeRoot, resolved(t, mainRoot))
	}
	if info.CommonDir != commonDir {
		t.Errorf("CommonDir = %q, want %q", info.CommonDir, commonDir)
	}
	if info.Linked {
		t.Errorf("Linked = true, want false for main worktree")
	}
	if info.Branch != "main" {
		t.Errorf("Branch = %q, want %q", info.Branch, "main")
	}
	if info.Detached {
		t.Errorf("Detached = true, want false")
	}
}

func TestResolveLinkedWorktreeFromSubdir(t *testing.T) {
	base := t.TempDir()
	linkedDir := filepath.Join(base, "feature")
	mainRoot, commonDir := buildRepo(t, "ref: refs/heads/main\n", []linkedSpec{
		{name: "feature", dir: linkedDir, head: "ref: refs/heads/feature\n"},
	})
	_ = mainRoot
	// a nested subdirectory inside the linked worktree must resolve up to it.
	sub := filepath.Join(linkedDir, "src", "pkg")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatalf("mkdir sub: %v", err)
	}

	info, ok := Resolve(sub)
	if !ok {
		t.Fatalf("Resolve(%s) returned not-a-worktree", sub)
	}
	if info.WorktreeRoot != resolved(t, linkedDir) {
		t.Errorf("WorktreeRoot = %q, want %q", info.WorktreeRoot, resolved(t, linkedDir))
	}
	if info.CommonDir != commonDir {
		t.Errorf("CommonDir = %q, want %q", info.CommonDir, commonDir)
	}
	if !info.Linked {
		t.Errorf("Linked = false, want true for linked worktree")
	}
	if info.Branch != "feature" {
		t.Errorf("Branch = %q, want %q", info.Branch, "feature")
	}
}

func TestResolveDetachedHead(t *testing.T) {
	base := t.TempDir()
	linkedDir := filepath.Join(base, "detached")
	sha := "0123456789abcdef0123456789abcdef01234567"
	buildRepo(t, "ref: refs/heads/main\n", []linkedSpec{
		{name: "detached", dir: linkedDir, head: sha + "\n"},
	})

	info, ok := Resolve(linkedDir)
	if !ok {
		t.Fatalf("Resolve(%s) returned not-a-worktree", linkedDir)
	}
	if !info.Detached {
		t.Errorf("Detached = false, want true")
	}
	if info.Branch != "" {
		t.Errorf("Branch = %q, want empty for detached HEAD", info.Branch)
	}
	if info.Head != sha {
		t.Errorf("Head = %q, want %q", info.Head, sha)
	}
}

func TestResolveNonGitDirectory(t *testing.T) {
	plain := t.TempDir()
	if _, ok := Resolve(plain); ok {
		t.Errorf("Resolve(%s) reported a worktree for a non-git directory", plain)
	}
}

func TestCommonDirAtMainAndLinkedMatch(t *testing.T) {
	base := t.TempDir()
	linkedDir := filepath.Join(base, "feature")
	mainRoot, commonDir := buildRepo(t, "ref: refs/heads/main\n", []linkedSpec{
		{name: "feature", dir: linkedDir, head: "ref: refs/heads/feature\n"},
	})

	mainCommon, ok := CommonDirAt(mainRoot)
	if !ok {
		t.Fatalf("CommonDirAt(main) not ok")
	}
	linkedCommon, ok := CommonDirAt(linkedDir)
	if !ok {
		t.Fatalf("CommonDirAt(linked) not ok")
	}
	if mainCommon != commonDir {
		t.Errorf("main CommonDirAt = %q, want %q", mainCommon, commonDir)
	}
	if linkedCommon != commonDir {
		t.Errorf("linked CommonDirAt = %q, want %q", linkedCommon, commonDir)
	}
	if mainCommon != linkedCommon {
		t.Errorf("main and linked common dirs differ: %q vs %q", mainCommon, linkedCommon)
	}
}

func TestCommonDirAtNoGitEntry(t *testing.T) {
	plain := t.TempDir()
	if _, ok := CommonDirAt(plain); ok {
		t.Errorf("CommonDirAt(%s) reported a common dir for a non-git directory", plain)
	}
}

func TestCommonDirAtSubmoduleDiffersFromSuperproject(t *testing.T) {
	base := t.TempDir()
	superRoot := filepath.Join(base, "super")
	writeFile(t, filepath.Join(superRoot, ".git", "HEAD"), "ref: refs/heads/main\n")
	superCommon := resolved(t, filepath.Join(superRoot, ".git"))

	// a submodule's .git is a file pointing into the superproject's modules dir,
	// whose common dir is that modules path, distinct from the superproject's.
	subModuleDir := filepath.Join(superRoot, "vendor", "lib")
	moduleGitDir := filepath.Join(superRoot, ".git", "modules", "lib")
	writeFile(t, filepath.Join(moduleGitDir, "HEAD"), "ref: refs/heads/main\n")
	writeFile(t, filepath.Join(subModuleDir, ".git"), "gitdir: "+moduleGitDir+"\n")

	subCommon, ok := CommonDirAt(subModuleDir)
	if !ok {
		t.Fatalf("CommonDirAt(submodule) not ok")
	}
	if subCommon == superCommon {
		t.Errorf("submodule common dir %q must differ from superproject %q", subCommon, superCommon)
	}
}

func TestSiblingWorktreeRootsIncludesMainAndLinked(t *testing.T) {
	base := t.TempDir()
	featureDir := filepath.Join(base, "feature")
	hotfixDir := filepath.Join(base, "hotfix")
	mainRoot, commonDir := buildRepo(t, "ref: refs/heads/main\n", []linkedSpec{
		{name: "feature", dir: featureDir, head: "ref: refs/heads/feature\n"},
		{name: "hotfix", dir: hotfixDir, head: "ref: refs/heads/hotfix\n"},
	})

	got := SiblingWorktreeRoots(commonDir)
	want := []string{resolved(t, mainRoot), resolved(t, featureDir), resolved(t, hotfixDir)}
	slices.Sort(want)
	if !slices.Equal(got, want) {
		t.Errorf("SiblingWorktreeRoots = %v, want %v", got, want)
	}
}

func TestSiblingWorktreeRootsOmitsMissingDir(t *testing.T) {
	base := t.TempDir()
	featureDir := filepath.Join(base, "feature")
	goneDir := filepath.Join(base, "gone")
	mainRoot, commonDir := buildRepo(t, "ref: refs/heads/main\n", []linkedSpec{
		{name: "feature", dir: featureDir, head: "ref: refs/heads/feature\n"},
		{name: "gone", dir: goneDir, head: "ref: refs/heads/gone\n"},
	})
	// simulate a worktree moved away without `git worktree repair`: the pointer
	// remains under .git/worktrees but the working directory is gone.
	if err := os.RemoveAll(goneDir); err != nil {
		t.Fatalf("remove gone dir: %v", err)
	}

	got := SiblingWorktreeRoots(commonDir)
	want := []string{resolved(t, mainRoot), resolved(t, featureDir)}
	slices.Sort(want)
	if !slices.Equal(got, want) {
		t.Errorf("SiblingWorktreeRoots = %v, want %v", got, want)
	}
}

func TestWorktreeOfRepo(t *testing.T) {
	base := t.TempDir()
	linkedDir := filepath.Join(base, "feature")
	mainRoot, commonDir := buildRepo(t, "ref: refs/heads/main\n", []linkedSpec{
		{name: "feature", dir: linkedDir, head: "ref: refs/heads/feature\n"},
	})

	if !WorktreeOfRepo(linkedDir, commonDir) {
		t.Errorf("WorktreeOfRepo(linked, commonDir) = false, want true")
	}
	if !WorktreeOfRepo(mainRoot, commonDir) {
		t.Errorf("WorktreeOfRepo(main, commonDir) = false, want true")
	}
	if WorktreeOfRepo(base, commonDir) {
		t.Errorf("WorktreeOfRepo(non-git dir, commonDir) = true, want false")
	}
	if WorktreeOfRepo(linkedDir, "") {
		t.Errorf("WorktreeOfRepo(linked, empty commonDir) = true, want false")
	}

	// A submodule's .git resolves to a different common dir, so it must not match
	// the superproject's common dir.
	superRoot := filepath.Join(base, "super")
	writeFile(t, filepath.Join(superRoot, ".git", "HEAD"), "ref: refs/heads/main\n")
	superCommon := resolved(t, filepath.Join(superRoot, ".git"))
	subModuleDir := filepath.Join(superRoot, "vendor", "lib")
	moduleGitDir := filepath.Join(superRoot, ".git", "modules", "lib")
	writeFile(t, filepath.Join(moduleGitDir, "HEAD"), "ref: refs/heads/main\n")
	writeFile(t, filepath.Join(subModuleDir, ".git"), "gitdir: "+moduleGitDir+"\n")
	if WorktreeOfRepo(subModuleDir, superCommon) {
		t.Errorf("WorktreeOfRepo(submodule, superproject commonDir) = true, want false")
	}
}

func TestPathInsideNestedWorktree(t *testing.T) {
	// Build the fixture manually so the main root and the nested worktree share
	// one base (buildRepo allocates its own TempDir, which would put a passed-in
	// nested dir under a different root and defeat the prefix walk). The watcher
	// scenario is a linked worktree nested under the repo's own .claude/worktrees.
	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	gitDir := filepath.Join(mainRoot, ".git")
	writeFile(t, filepath.Join(gitDir, "HEAD"), "ref: refs/heads/main\n")
	commonDir := resolved(t, gitDir)

	nestedWt := filepath.Join(mainRoot, ".claude", "worktrees", "feature")
	perWt := filepath.Join(gitDir, "worktrees", "feature")
	writeFile(t, filepath.Join(perWt, "commondir"), "../..\n")
	writeFile(t, filepath.Join(perWt, "gitdir"), filepath.Join(nestedWt, ".git")+"\n")
	writeFile(t, filepath.Join(perWt, "HEAD"), "ref: refs/heads/feature\n")
	writeFile(t, filepath.Join(nestedWt, ".git"), "gitdir: "+perWt+"\n")

	// A submodule nested inside the main worktree (different common dir).
	subModuleDir := filepath.Join(mainRoot, "vendor", "lib")
	moduleGitDir := filepath.Join(gitDir, "modules", "lib")
	writeFile(t, filepath.Join(moduleGitDir, "HEAD"), "ref: refs/heads/main\n")
	writeFile(t, filepath.Join(subModuleDir, ".git"), "gitdir: "+moduleGitDir+"\n")

	cases := []struct {
		name string
		path string
		want bool
	}{
		{"file inside nested worktree", filepath.Join(nestedWt, "internal", "x.go"), true},
		{"file nested at depth", filepath.Join(nestedWt, "a", "b", "c", "file.go"), true},
		{"worktree .git pointer", filepath.Join(nestedWt, ".git"), true},
		{"worktree root itself", nestedWt, true},
		{"repo root itself", mainRoot, false},
		{"parent file", filepath.Join(mainRoot, "main.go"), false},
		{"plain nested dir file", filepath.Join(mainRoot, "internal", "x.go"), false},
		{"submodule file", filepath.Join(subModuleDir, "lib.go"), false},
		{"path outside root", filepath.Join(base, "elsewhere", "x.go"), false},
	}
	for _, testCase := range cases {
		got := PathInsideNestedWorktree(mainRoot, commonDir, testCase.path)
		if got != testCase.want {
			t.Errorf("PathInsideNestedWorktree(%q) = %v, want %v", testCase.name, got, testCase.want)
		}
	}

	if PathInsideNestedWorktree(mainRoot, "", filepath.Join(nestedWt, "x.go")) {
		t.Errorf("PathInsideNestedWorktree with empty commonDir = true, want false")
	}
}

func TestPathInsideSubmodule(t *testing.T) {
	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	gitDir := filepath.Join(mainRoot, ".git")
	writeFile(t, filepath.Join(gitDir, "HEAD"), "ref: refs/heads/main\n")
	commonDir := resolved(t, gitDir)

	subModuleDir := filepath.Join(mainRoot, "vendor", "lib")
	moduleGitDir := filepath.Join(gitDir, "modules", "vendor", "lib")
	writeFile(t, filepath.Join(moduleGitDir, "HEAD"), "ref: refs/heads/main\n")
	writeFile(t, filepath.Join(subModuleDir, ".git"), "gitdir: "+moduleGitDir+"\n")

	relRoot, ok := PathInsideSubmodule(mainRoot, commonDir, filepath.Join(subModuleDir, "internal", "lib.go"))
	if !ok {
		t.Fatal("PathInsideSubmodule returned ok=false for a file inside a submodule")
	}
	if relRoot != "vendor/lib" {
		t.Fatalf("relRoot = %q, want vendor/lib", relRoot)
	}
	if _, ok := PathInsideSubmodule(mainRoot, commonDir, mainRoot); ok {
		t.Fatal("PathInsideSubmodule reported the root itself as a submodule")
	}
	if _, ok := PathInsideSubmodule(mainRoot, commonDir, filepath.Join(mainRoot, "main.go")); ok {
		t.Fatal("PathInsideSubmodule reported a parent repo file as a submodule")
	}
	if _, ok := PathInsideSubmodule(mainRoot, "", filepath.Join(subModuleDir, "internal", "lib.go")); ok {
		t.Fatal("PathInsideSubmodule with empty commonDir = ok, want false")
	}
}

func TestWorktreeTrackedTrueThenFalseAfterRemoval(t *testing.T) {
	base := t.TempDir()
	wtDir := filepath.Join(base, "feat")
	_, commonDir := buildRepo(t, "ref: refs/heads/main\n", []linkedSpec{
		{name: "feat", dir: wtDir, head: "ref: refs/heads/feat\n"},
	})

	if !WorktreeTracked(commonDir, wtDir) {
		t.Fatalf("WorktreeTracked = false for a live linked worktree, want true")
	}

	// Simulate `git worktree remove`: git deletes the per-worktree admin entry.
	if err := os.RemoveAll(filepath.Join(commonDir, "worktrees", "feat")); err != nil {
		t.Fatalf("remove admin entry: %v", err)
	}
	if WorktreeTracked(commonDir, wtDir) {
		t.Fatalf("WorktreeTracked = true after the admin entry was removed, want false")
	}
	if WorktreeTracked("", wtDir) {
		t.Fatalf("WorktreeTracked = true for an empty common dir, want false")
	}
}

package daemon

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

func writeWorktreeFile(t *testing.T, path string, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir for %s: %v", path, err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func evalSym(t *testing.T, path string) string {
	t.Helper()
	out, err := filepath.EvalSymlinks(path)
	if err != nil {
		t.Fatalf("eval symlinks %s: %v", path, err)
	}
	return out
}

// makeMainRepo writes a main worktree (a .git directory) with one Go file.
func makeMainRepo(t *testing.T, root string) {
	t.Helper()
	writeWorktreeFile(t, filepath.Join(root, ".git", "HEAD"), "ref: refs/heads/main\n")
	writeWorktreeFile(t, filepath.Join(root, "main.go"), "package main\n")
}

// makeLinkedWorktree registers a linked worktree of the repo rooted at mainRoot,
// rooting the worktree at worktreeDir on the given branch with one Go file.
func makeLinkedWorktree(t *testing.T, mainRoot string, name string, worktreeDir string, branch string) {
	t.Helper()
	perWorktree := filepath.Join(mainRoot, ".git", "worktrees", name)
	writeWorktreeFile(t, filepath.Join(perWorktree, "commondir"), "../..\n")
	writeWorktreeFile(t, filepath.Join(perWorktree, "gitdir"), filepath.Join(worktreeDir, ".git")+"\n")
	writeWorktreeFile(t, filepath.Join(perWorktree, "HEAD"), "ref: refs/heads/"+branch+"\n")
	writeWorktreeFile(t, filepath.Join(worktreeDir, ".git"), "gitdir: "+perWorktree+"\n")
	writeWorktreeFile(t, filepath.Join(worktreeDir, "feature.go"), "package feature\n")
}

func waitForCodebaseSettled(t *testing.T, manager *Manager, path string) model.Codebase {
	t.Helper()
	var settled model.Codebase
	waitForCondition(t, func() bool {
		codebase, _, ok, _, err := manager.GetIndex(context.Background(), path)
		if err != nil || !ok || codebase.LastSuccessfulRun == nil {
			return false
		}
		settled = codebase
		return true
	})
	return settled
}

// TestGetIndexResolvesNestedWorktreeToOwnCodebase proves a worktree nested under
// an indexed parent repo resolves to its own codebase, never the covering
// parent, and that the daemon auto-creates that codebase on first use.
func TestGetIndexResolvesNestedWorktreeToOwnCodebase(t *testing.T) {
	manager, _, _ := newTestManager(t)
	manager.runner = fakeRunner{}

	mainRoot := filepath.Join(t.TempDir(), "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(mainRoot, ".claude", "worktrees", "foo")
	makeLinkedWorktree(t, mainRoot, "foo", worktreeDir, "feature")

	if _, _, _, _, err := manager.StartIndex(context.Background(), mainRoot, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget); err != nil {
		t.Fatalf("StartIndex(main) returned error: %v", err)
	}
	mainCodebase := waitForCodebaseSettled(t, manager, mainRoot)

	codebase, _, found, _, err := manager.GetIndex(context.Background(), worktreeDir)
	if err != nil {
		t.Fatalf("GetIndex(worktree) returned error: %v", err)
	}
	if !found {
		t.Fatal("GetIndex(worktree) did not resolve to a codebase")
	}
	if codebase.ID == mainCodebase.ID {
		t.Fatalf("worktree resolved to the parent codebase %q; it must be its own", mainCodebase.ID)
	}
	if codebase.CanonicalPath != evalSym(t, worktreeDir) {
		t.Fatalf("worktree codebase CanonicalPath = %q, want %q", codebase.CanonicalPath, evalSym(t, worktreeDir))
	}

	// let the auto-created build settle before cleanup removes the temp dirs.
	waitForCodebaseSettled(t, manager, worktreeDir)
}

// TestGetIndexResolvesExternalWorktreeToOwnCodebase proves an external sibling
// worktree (outside the parent tree) resolves to its own codebase once a sibling
// of the same repo is indexed.
func TestGetIndexResolvesExternalWorktreeToOwnCodebase(t *testing.T) {
	manager, _, _ := newTestManager(t)
	manager.runner = fakeRunner{}

	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(base, "feature")
	makeLinkedWorktree(t, mainRoot, "feature", worktreeDir, "feature")

	if _, _, _, _, err := manager.StartIndex(context.Background(), mainRoot, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget); err != nil {
		t.Fatalf("StartIndex(main) returned error: %v", err)
	}
	mainCodebase := waitForCodebaseSettled(t, manager, mainRoot)

	codebase, _, found, _, err := manager.GetIndex(context.Background(), worktreeDir)
	if err != nil {
		t.Fatalf("GetIndex(external worktree) returned error: %v", err)
	}
	if !found {
		t.Fatal("GetIndex(external worktree) did not resolve to a codebase")
	}
	if codebase.ID == mainCodebase.ID {
		t.Fatalf("external worktree resolved to the parent codebase; it must be its own")
	}
	if codebase.CanonicalPath != evalSym(t, worktreeDir) {
		t.Fatalf("external worktree CanonicalPath = %q, want %q", codebase.CanonicalPath, evalSym(t, worktreeDir))
	}

	waitForCodebaseSettled(t, manager, worktreeDir)
}

// TestStartIndexParentDoesNotAbsorbNestedWorktree proves a nested worktree that
// is already its own codebase survives a (re)index of the parent: the parent
// must not absorb a sibling worktree the way it absorbs ordinary child indexes.
func TestStartIndexParentDoesNotAbsorbNestedWorktree(t *testing.T) {
	manager, _, _ := newTestManager(t)
	manager.runner = fakeRunner{}
	// A path-distinct collection name plus a missing-collection probe forces the
	// parent index through the bootstrap+absorb path that would otherwise fold a
	// child index in, and gives the worktree codebase a non-empty collection so
	// it is a genuine absorb candidate.
	manager.semantic = &fakeSemantic{
		collectionName:       func(path string) string { return "cc_" + filepath.Base(path) },
		hasCollectionForPath: func(context.Context, string) (bool, error) { return false, nil },
	}

	mainRoot := filepath.Join(t.TempDir(), "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(mainRoot, ".claude", "worktrees", "foo")
	makeLinkedWorktree(t, mainRoot, "foo", worktreeDir, "feature")

	// index the worktree first so it is its own codebase, then index the parent.
	if _, _, _, _, err := manager.StartIndex(context.Background(), worktreeDir, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget); err != nil {
		t.Fatalf("StartIndex(worktree) returned error: %v", err)
	}
	worktreeCodebase := waitForCodebaseSettled(t, manager, worktreeDir)

	if _, _, _, _, err := manager.StartIndex(context.Background(), mainRoot, testClientInfo(), defaultIndexConfig(), true, emptyAdmissionBudget); err != nil {
		t.Fatalf("StartIndex(main, force) returned error: %v", err)
	}
	waitForCodebaseSettled(t, manager, mainRoot)

	resolved, _, found, _, err := manager.GetIndex(context.Background(), worktreeDir)
	if err != nil {
		t.Fatalf("GetIndex(worktree) after parent index returned error: %v", err)
	}
	if !found || resolved.ID != worktreeCodebase.ID {
		t.Fatalf("worktree codebase was absorbed by the parent: got found=%v id=%q want id=%q", found, resolved.ID, worktreeCodebase.ID)
	}
	if resolved.CanonicalPath != evalSym(t, worktreeDir) {
		t.Fatalf("after parent index, worktree CanonicalPath = %q, want %q", resolved.CanonicalPath, evalSym(t, worktreeDir))
	}
}

// TestWorktreeBuildReusesSiblingCollection proves a worktree's from-scratch
// build loads reuse vectors from an already-indexed sibling worktree of the same
// repo, so only the branch diff is embedded rather than the whole tree.
func TestWorktreeBuildReusesSiblingCollection(t *testing.T) {
	manager, _, _ := newTestManager(t)
	manager.runner = fakeRunner{}
	fake := &fakeSemantic{
		collectionName:       func(path string) string { return "cc_" + filepath.Base(path) },
		hasCollectionForPath: func(context.Context, string) (bool, error) { return false, nil },
	}
	manager.semantic = fake

	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(base, "feature")
	makeLinkedWorktree(t, mainRoot, "feature", worktreeDir, "feature")

	if _, _, _, _, err := manager.StartIndex(context.Background(), mainRoot, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget); err != nil {
		t.Fatalf("StartIndex(main) returned error: %v", err)
	}
	waitForCodebaseSettled(t, manager, mainRoot)

	if _, _, _, _, err := manager.StartIndex(context.Background(), worktreeDir, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget); err != nil {
		t.Fatalf("StartIndex(worktree) returned error: %v", err)
	}
	waitForCodebaseSettled(t, manager, worktreeDir)

	requestedSiblingCollection := func() bool {
		fake.mu.Lock()
		defer fake.mu.Unlock()
		for _, call := range fake.reuseCollections {
			for _, name := range call {
				if name == "cc_repo" {
					return true
				}
			}
		}
		return false
	}
	if !requestedSiblingCollection() {
		t.Fatalf("worktree build did not request reuse vectors from the sibling collection cc_repo; reuse calls = %v", fake.reuseCollections)
	}
}

// registerSiblingCodebase records a main-worktree codebase for the repo rooted at
// mainRoot with the given liveness fields, so the reuse gate can be unit-tested
// without driving a full concurrent build.
func registerSiblingCodebase(t *testing.T, manager *Manager, mainRoot string, mutate func(*model.Codebase)) {
	t.Helper()
	codebase := newCodebaseRecord(evalSym(t, mainRoot))
	codebase.CollectionName = "cc_repo"
	codebase.EffectiveConfig = defaultIndexConfig()
	mutate(&codebase)
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
}

// TestWorktreeReuseAcceptsBusyIndexedSibling proves the race fix: a sibling that
// was indexed once but is currently mid-sync (ActiveJobID set, Status indexing)
// is still an eligible reuse source.
func TestWorktreeReuseAcceptsBusyIndexedSibling(t *testing.T) {
	manager, _, _ := newTestManager(t)
	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(base, "feature")
	makeLinkedWorktree(t, mainRoot, "feature", worktreeDir, "feature")

	registerSiblingCodebase(t, manager, mainRoot, func(c *model.Codebase) {
		c.Status = model.CodebaseStatusIndexing // busy
		c.ActiveJobID = "job-sync-inflight"     // busy
		c.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 1, Status: "completed"}
	})

	got := manager.worktreeSiblingReuseCollections(evalSym(t, worktreeDir), defaultIndexConfig())
	if len(got) != 1 || got[0] != "cc_repo" {
		t.Fatalf("busy-but-indexed sibling not reused: got %v, want [cc_repo]", got)
	}
}

// TestWorktreeReuseAcceptsAdoptedSibling proves an adopted sibling (Status
// Indexed, no run record) is eligible, matching the auto-create trigger.
func TestWorktreeReuseAcceptsAdoptedSibling(t *testing.T) {
	manager, _, _ := newTestManager(t)
	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(base, "feature")
	makeLinkedWorktree(t, mainRoot, "feature", worktreeDir, "feature")

	registerSiblingCodebase(t, manager, mainRoot, func(c *model.Codebase) {
		c.Status = model.CodebaseStatusIndexed
		c.LastSuccessfulRun = nil
	})

	got := manager.worktreeSiblingReuseCollections(evalSym(t, worktreeDir), defaultIndexConfig())
	if len(got) != 1 || got[0] != "cc_repo" {
		t.Fatalf("adopted sibling not reused: got %v, want [cc_repo]", got)
	}
}

// TestWorktreeReuseSkipsNeverIndexedSibling proves a sibling that never produced
// a usable collection (not indexed, no run) is not an eligible source.
func TestWorktreeReuseSkipsNeverIndexedSibling(t *testing.T) {
	manager, _, _ := newTestManager(t)
	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(base, "feature")
	makeLinkedWorktree(t, mainRoot, "feature", worktreeDir, "feature")

	registerSiblingCodebase(t, manager, mainRoot, func(c *model.Codebase) {
		c.Status = model.CodebaseStatusIndexing
		c.LastSuccessfulRun = nil
	})

	got := manager.worktreeSiblingReuseCollections(evalSym(t, worktreeDir), defaultIndexConfig())
	if len(got) != 0 {
		t.Fatalf("never-indexed sibling should not be reused: got %v", got)
	}
}

// TestWorktreeReuseSkipsModelMismatch proves a sibling indexed with a different
// embedding model is not an eligible source.
func TestWorktreeReuseSkipsModelMismatch(t *testing.T) {
	manager, _, _ := newTestManager(t)
	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(base, "feature")
	makeLinkedWorktree(t, mainRoot, "feature", worktreeDir, "feature")

	mismatched := defaultIndexConfig()
	mismatched.EmbeddingModel = "some-other-model"
	registerSiblingCodebase(t, manager, mainRoot, func(c *model.Codebase) {
		c.Status = model.CodebaseStatusIndexed
		c.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 1, Status: "completed"}
		c.EffectiveConfig = mismatched
	})

	got := manager.worktreeSiblingReuseCollections(evalSym(t, worktreeDir), defaultIndexConfig())
	if len(got) != 0 {
		t.Fatalf("model-mismatched sibling should not be reused: got %v", got)
	}
}

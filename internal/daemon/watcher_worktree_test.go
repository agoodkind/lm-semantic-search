package daemon

import (
	"context"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/discovery"
	"goodkind.io/lm-semantic-search/internal/model"
)

// newRecordingWatcher builds a watcher whose queue records the relative paths
// enqueued per codebase id, returning the watcher and a snapshot accessor. The
// drain debounce is short so a dispatched event surfaces quickly.
func newRecordingWatcher(t *testing.T, manager *Manager) (*Watcher, func(codebaseID string) []string) {
	t.Helper()
	var mu sync.Mutex
	observed := map[string][]string{}
	queue := NewEventQueue(5*time.Millisecond, func(codebaseID string, relativePaths []string) {
		mu.Lock()
		observed[codebaseID] = append(observed[codebaseID], relativePaths...)
		mu.Unlock()
	})
	watcher := NewWatcher(manager, queue)
	snapshot := func(codebaseID string) []string {
		mu.Lock()
		defer mu.Unlock()
		out := make([]string, len(observed[codebaseID]))
		copy(out, observed[codebaseID])
		return out
	}
	return watcher, snapshot
}

// waitForRelativePath polls the recorded enqueues for codebaseID until want
// appears or the deadline passes, returning the final recorded set. It is the
// sentinel that lets a negative assertion be reliable: once a known-good path
// for the same codebase has drained, any leaked path would have drained too.
func waitForRelativePath(t *testing.T, snapshot func(string) []string, codebaseID string, want string) []string {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		paths := snapshot(codebaseID)
		if slicesContains(paths, want) {
			return paths
		}
		time.Sleep(10 * time.Millisecond)
	}
	return snapshot(codebaseID)
}

func slicesContains(paths []string, want string) bool {
	for _, path := range paths {
		if path == want {
			return true
		}
	}
	return false
}

func parentCodebase(root string) model.Codebase {
	return model.Codebase{
		ID:                  "cb_parent",
		CanonicalPath:       root,
		ResolvedIgnoreRules: discovery.IgnoreRules{Nodes: nil},
	}
}

// TestDispatchSkipsUnregisteredSameRepoWorktree is the core regression: a
// freshly-created nested worktree that has NOT been queried (so it is not a
// registered codebase) must not leak its files into the parent index. The
// boundary reads on-disk .git topology, so it holds without registration.
func TestDispatchSkipsUnregisteredSameRepoWorktree(t *testing.T) {
	t.Parallel()
	manager, _, _ := newTestManager(t)
	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(mainRoot, ".claude", "worktrees", "wt")
	makeLinkedWorktree(t, mainRoot, "wt", worktreeDir, "wt")

	watcher, snapshot := newRecordingWatcher(t, manager)
	// Register ONLY the parent; the worktree stays unregistered (the bug's core).
	watcher.AddCodebase(context.Background(), parentCodebase(mainRoot))

	// Event for a real worktree file, then a sentinel event for a real parent file.
	watcher.dispatch(context.Background(), stubNotifyEvent{path: filepath.Join(worktreeDir, "feature.go")})
	watcher.dispatch(context.Background(), stubNotifyEvent{path: filepath.Join(mainRoot, "main.go")})

	paths := waitForRelativePath(t, snapshot, "cb_parent", "main.go")
	if !slicesContains(paths, "main.go") {
		t.Fatalf("sentinel main.go was not enqueued for the parent; got %v", paths)
	}
	for _, path := range paths {
		if strings.Contains(path, "worktrees") {
			t.Fatalf("parent enqueued worktree path %q; the boundary leaked", path)
		}
	}
}

// TestDispatchRegisteredWorktreeRoutesToWorktree pins the existing-good case:
// when the worktree IS its own registered codebase, its events route to the
// worktree, never the parent.
func TestDispatchRegisteredWorktreeRoutesToWorktree(t *testing.T) {
	t.Parallel()
	manager, _, _ := newTestManager(t)
	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(mainRoot, ".claude", "worktrees", "wt")
	makeLinkedWorktree(t, mainRoot, "wt", worktreeDir, "wt")

	watcher, snapshot := newRecordingWatcher(t, manager)
	watcher.AddCodebase(context.Background(), parentCodebase(mainRoot))
	watcher.AddCodebase(context.Background(), model.Codebase{
		ID:                  "cb_worktree",
		CanonicalPath:       worktreeDir,
		ResolvedIgnoreRules: discovery.IgnoreRules{Nodes: nil},
	})

	watcher.dispatch(context.Background(), stubNotifyEvent{path: filepath.Join(worktreeDir, "feature.go")})
	watcher.dispatch(context.Background(), stubNotifyEvent{path: filepath.Join(mainRoot, "main.go")})

	worktreePaths := waitForRelativePath(t, snapshot, "cb_worktree", "feature.go")
	if !slicesContains(worktreePaths, "feature.go") {
		t.Fatalf("worktree codebase did not receive feature.go; got %v", worktreePaths)
	}
	parentPaths := waitForRelativePath(t, snapshot, "cb_parent", "main.go")
	for _, path := range parentPaths {
		if strings.Contains(path, "worktrees") {
			t.Fatalf("parent enqueued worktree path %q; it should route to the worktree", path)
		}
	}
}

// TestDispatchSubmoduleStillEnqueuesForParent guards the preserved invariant: a
// submodule resolves to a different common dir, so it is NOT a boundary and its
// files still index into the parent.
func TestDispatchSubmoduleStillEnqueuesForParent(t *testing.T) {
	t.Parallel()
	manager, _, _ := newTestManager(t)
	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)

	subModuleDir := filepath.Join(mainRoot, "vendor", "lib")
	moduleGitDir := filepath.Join(mainRoot, ".git", "modules", "lib")
	writeWorktreeFile(t, filepath.Join(moduleGitDir, "HEAD"), "ref: refs/heads/main\n")
	writeWorktreeFile(t, filepath.Join(subModuleDir, ".git"), "gitdir: "+moduleGitDir+"\n")
	writeWorktreeFile(t, filepath.Join(subModuleDir, "lib.go"), "package lib\n")

	watcher, snapshot := newRecordingWatcher(t, manager)
	watcher.AddCodebase(context.Background(), parentCodebase(mainRoot))

	watcher.dispatch(context.Background(), stubNotifyEvent{path: filepath.Join(subModuleDir, "lib.go")})

	want := filepath.ToSlash(filepath.Join("vendor", "lib", "lib.go"))
	paths := waitForRelativePath(t, snapshot, "cb_parent", want)
	if !slicesContains(paths, want) {
		t.Fatalf("parent did not index submodule file %q; got %v", want, paths)
	}
}

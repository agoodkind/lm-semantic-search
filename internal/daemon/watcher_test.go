package daemon

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/rjeczalik/notify"
	"goodkind.io/lm-semantic-search/internal/model"
)

// stubNotifyEvent satisfies notify.EventInfo so the watcher's dispatch
// routine can be exercised without registering with the platform's native
// watch backend.
type stubNotifyEvent struct {
	path string
}

func (event stubNotifyEvent) Event() notify.Event { return notify.Write }
func (event stubNotifyEvent) Path() string        { return event.path }
func (event stubNotifyEvent) Sys() any            { return nil }

// TestWatcherAddCodebaseIsIdempotent confirms AddCodebase tolerates a
// second call for the same codebase id without duplicating the in-memory
// registration.
func TestWatcherAddCodebaseIsIdempotent(t *testing.T) {
	t.Parallel()
	manager, _, _ := newTestManager(t)
	queue := NewEventQueue(time.Millisecond, func(_ string, _ []string) {})
	watcher := NewWatcher(manager, queue)

	codebase := model.Codebase{
		ID:            "cb_test_idempotent",
		CanonicalPath: t.TempDir(),
	}
	watcher.AddCodebase(context.Background(), codebase)
	watcher.AddCodebase(context.Background(), codebase)

	if got := watcher.activeRootCount(); got != 1 {
		t.Fatalf("activeRootCount=%d, want 1 after idempotent add", got)
	}
}

// TestWatcherDispatchHonorsResolverRules proves dispatch routes its ignore
// decision through the indexability resolver: a file under a directory the
// codebase's .gitignore excludes is dropped, while a tracked source file is
// enqueued.
func TestWatcherDispatchHonorsResolverRules(t *testing.T) {
	manager, _, _ := newTestManager(t)

	var observedMu sync.Mutex
	observed := map[string][]string{}
	queue := NewEventQueue(time.Millisecond, func(codebaseID string, relativePaths []string) {
		observedMu.Lock()
		observed[codebaseID] = append(observed[codebaseID], relativePaths...)
		observedMu.Unlock()
	})
	watcher := NewWatcher(manager, queue)

	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, ".gitignore"), []byte("secret/\n"), 0o644); err != nil {
		t.Fatalf("write .gitignore: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(root, "secret"), 0o755); err != nil {
		t.Fatalf("mkdir secret: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "secret", "x.txt"), []byte("x"), 0o644); err != nil {
		t.Fatalf("write secret file: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(root, "keep"), 0o755); err != nil {
		t.Fatalf("mkdir keep: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "keep", "y.go"), []byte("package keep\n"), 0o644); err != nil {
		t.Fatalf("write keep file: %v", err)
	}

	codebase := model.Codebase{
		ID:            "cb_resolver_rules",
		CanonicalPath: root,
	}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	watcher.AddCodebase(context.Background(), codebase)

	watcher.dispatch(context.Background(), stubNotifyEvent{path: filepath.Join(root, "secret", "x.txt")})
	watcher.dispatch(context.Background(), stubNotifyEvent{path: filepath.Join(root, "keep", "y.go")})

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		observedMu.Lock()
		keepObserved := len(observed[codebase.ID]) > 0
		observedMu.Unlock()
		if keepObserved {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	observedMu.Lock()
	defer observedMu.Unlock()
	paths := observed[codebase.ID]
	foundKeep := false
	for _, enqueued := range paths {
		if enqueued == "secret/x.txt" {
			t.Fatalf("dispatch enqueued an ignored path: %v", paths)
		}
		if enqueued == "keep/y.go" {
			foundKeep = true
		}
	}
	if !foundKeep {
		t.Fatalf("dispatch did not enqueue keep/y.go: %v", paths)
	}
}

// TestWatcherDispatchFiltersDeletedGitPath confirms a delete/rename event, where
// os.Lstat fails because the file is gone, still routes through the resolver: a
// deleted .git path is dropped (no git lock-file churn on removal) while a
// deleted real source file still enqueues so converge can remove it.
func TestWatcherDispatchFiltersDeletedGitPath(t *testing.T) {
	manager, _, _ := newTestManager(t)

	var observedMu sync.Mutex
	observed := map[string][]string{}
	queue := NewEventQueue(time.Millisecond, func(codebaseID string, relativePaths []string) {
		observedMu.Lock()
		observed[codebaseID] = append(observed[codebaseID], relativePaths...)
		observedMu.Unlock()
	})
	watcher := NewWatcher(manager, queue)

	root := t.TempDir()
	codebase := model.Codebase{
		ID:            "cb_delete_git",
		CanonicalPath: root,
	}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	watcher.AddCodebase(context.Background(), codebase)

	// Neither path exists on disk, so os.Lstat fails as it would for a delete.
	watcher.dispatch(context.Background(), stubNotifyEvent{path: filepath.Join(root, ".git", "index")})
	watcher.dispatch(context.Background(), stubNotifyEvent{path: filepath.Join(root, "pkg", "gone.go")})

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		observedMu.Lock()
		seen := len(observed[codebase.ID]) > 0
		observedMu.Unlock()
		if seen {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	observedMu.Lock()
	defer observedMu.Unlock()
	paths := observed[codebase.ID]
	foundGone := false
	for _, enqueued := range paths {
		if enqueued == ".git/index" {
			t.Fatalf("dispatch enqueued a deleted .git path: %v", paths)
		}
		if enqueued == "pkg/gone.go" {
			foundGone = true
		}
	}
	if !foundGone {
		t.Fatalf("dispatch dropped a deleted real file that should enqueue for removal: %v", paths)
	}
}

// TestWatcherRemoveCodebaseDropsRoot confirms RemoveCodebase clears the
// dispatch entry so events for that path stop enqueuing into the coalescer.
func TestWatcherRemoveCodebaseDropsRoot(t *testing.T) {
	t.Parallel()
	manager, _, _ := newTestManager(t)
	queue := NewEventQueue(time.Millisecond, func(_ string, _ []string) {})
	watcher := NewWatcher(manager, queue)

	codebase := model.Codebase{
		ID:            "cb_test_remove",
		CanonicalPath: t.TempDir(),
	}
	watcher.AddCodebase(context.Background(), codebase)
	watcher.RemoveCodebase(context.Background(), codebase.ID)
	if got := watcher.activeRootCount(); got != 0 {
		t.Fatalf("activeRootCount=%d, want 0 after remove", got)
	}
	watcher.RemoveCodebase(context.Background(), codebase.ID)
}

// TestWatcherDispatchBroadcastsToAllCovering confirms that an event under
// a path covered by two overlapping codebases enqueues onto both ids, not
// just the longest-prefix match.
func TestWatcherDispatchBroadcastsToAllCovering(t *testing.T) {
	t.Parallel()
	manager, _, _ := newTestManager(t)

	var observedMu sync.Mutex
	observed := map[string][]string{}
	queue := NewEventQueue(5*time.Millisecond, func(codebaseID string, relativePaths []string) {
		observedMu.Lock()
		observed[codebaseID] = append(observed[codebaseID], relativePaths...)
		observedMu.Unlock()
	})
	watcher := NewWatcher(manager, queue)

	rootDir := t.TempDir()
	nestedDir := filepath.Join(rootDir, "child")
	if err := os.MkdirAll(nestedDir, 0o755); err != nil {
		t.Fatalf("mkdir nested: %v", err)
	}

	outer := model.Codebase{
		ID:            "cb_outer",
		CanonicalPath: rootDir,
	}
	inner := model.Codebase{
		ID:            "cb_inner",
		CanonicalPath: nestedDir,
	}
	watcher.AddCodebase(context.Background(), outer)
	watcher.AddCodebase(context.Background(), inner)

	leaf := filepath.Join(nestedDir, "file.go")
	if err := os.WriteFile(leaf, []byte{}, 0o644); err != nil {
		t.Fatalf("write leaf: %v", err)
	}
	watcher.dispatch(context.Background(), stubNotifyEvent{path: leaf})

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		observedMu.Lock()
		bothObserved := len(observed["cb_outer"]) > 0 && len(observed["cb_inner"]) > 0
		observedMu.Unlock()
		if bothObserved {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	observedMu.Lock()
	defer observedMu.Unlock()
	t.Fatalf("dispatch did not broadcast to both codebases: %v", observed)
}

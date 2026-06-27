package daemon

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/rjeczalik/notify"
	"goodkind.io/lm-semantic-search/internal/discovery"
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
		ID:                  "cb_test_idempotent",
		CanonicalPath:       t.TempDir(),
		ResolvedIgnoreRules: discovery.IgnoreRules{Nodes: nil},
	}
	watcher.AddCodebase(context.Background(), codebase)
	watcher.AddCodebase(context.Background(), codebase)

	if got := watcher.activeRootCount(); got != 1 {
		t.Fatalf("activeRootCount=%d, want 1 after idempotent add", got)
	}
}

// TestWatcherDispatchReadsRecordRules proves dispatch filters against the
// codebase record's current ignore rules (the single source of truth) rather
// than any copy held by the watcher, and that updating the record's rules
// changes dispatch behavior with no watcher re-add. With empty record rules an
// event passes through; after the record resolves rules that ignore the path,
// the same watcher drops it.
func TestWatcherDispatchReadsRecordRules(t *testing.T) {
	t.Parallel()
	manager, _, _ := newTestManager(t)

	var observedMu sync.Mutex
	observed := map[string]struct{}{}
	queue := NewEventQueue(5*time.Millisecond, func(_ string, relativePaths []string) {
		observedMu.Lock()
		for _, relativePath := range relativePaths {
			observed[relativePath] = struct{}{}
		}
		observedMu.Unlock()
	})
	watcher := NewWatcher(manager, queue)

	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, ".gitignore"), []byte("ignored/\n"), 0o644); err != nil {
		t.Fatalf("write .gitignore: %v", err)
	}
	for _, name := range []string{"ignored/a.txt", "ignored/b.txt", "keep/c.go"} {
		full := filepath.Join(root, name)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", name, err)
		}
		if err := os.WriteFile(full, []byte{}, 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}

	codebase := model.Codebase{
		ID:                  "cb_test_record_rules",
		CanonicalPath:       root,
		ResolvedIgnoreRules: discovery.IgnoreRules{Nodes: nil},
	}
	manager.codebases[codebase.ID] = codebase
	watcher.AddCodebase(context.Background(), codebase)

	waitObserved := func(relativePath string) bool {
		deadline := time.Now().Add(2 * time.Second)
		for time.Now().Before(deadline) {
			observedMu.Lock()
			_, ok := observed[relativePath]
			observedMu.Unlock()
			if ok {
				return true
			}
			time.Sleep(10 * time.Millisecond)
		}
		return false
	}

	// Empty record rules: an ignored-by-gitignore path still passes through.
	watcher.dispatch(stubNotifyEvent{path: filepath.Join(root, "ignored/a.txt")})
	if !waitObserved("ignored/a.txt") {
		t.Fatal("with empty record rules, dispatch should enqueue ignored/a.txt")
	}

	// Resolve the real rules and fold them into the record. No watcher re-add.
	rules, err := discovery.EffectiveIgnorePatterns(context.Background(), root, nil)
	if err != nil {
		t.Fatalf("resolve rules: %v", err)
	}
	manager.cacheResolvedRules(codebase.ID, rules)

	// Now dispatch a fresh ignored path and a kept path. keep/c.go must enqueue;
	// ignored/b.txt must not, proving dispatch reads the updated record.
	watcher.dispatch(stubNotifyEvent{path: filepath.Join(root, "ignored/b.txt")})
	watcher.dispatch(stubNotifyEvent{path: filepath.Join(root, "keep/c.go")})
	if !waitObserved("keep/c.go") {
		t.Fatal("dispatch should enqueue keep/c.go after rules resolve")
	}
	observedMu.Lock()
	_, leaked := observed["ignored/b.txt"]
	observedMu.Unlock()
	if leaked {
		t.Fatal("dispatch enqueued ignored/b.txt after the record resolved rules that exclude it")
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
		ID:                  "cb_test_remove",
		CanonicalPath:       t.TempDir(),
		ResolvedIgnoreRules: discovery.IgnoreRules{Nodes: nil},
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
		ID:                  "cb_outer",
		CanonicalPath:       rootDir,
		ResolvedIgnoreRules: discovery.IgnoreRules{Nodes: nil},
	}
	inner := model.Codebase{
		ID:                  "cb_inner",
		CanonicalPath:       nestedDir,
		ResolvedIgnoreRules: discovery.IgnoreRules{Nodes: nil},
	}
	watcher.AddCodebase(context.Background(), outer)
	watcher.AddCodebase(context.Background(), inner)

	leaf := filepath.Join(nestedDir, "file.go")
	if err := os.WriteFile(leaf, []byte{}, 0o644); err != nil {
		t.Fatalf("write leaf: %v", err)
	}
	watcher.dispatch(stubNotifyEvent{path: leaf})

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

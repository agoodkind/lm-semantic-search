package daemon

import (
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/rjeczalik/notify"
	"goodkind.io/lm-semantic-search/internal/discovery"
	"goodkind.io/lm-semantic-search/internal/gitworktree"
	"goodkind.io/lm-semantic-search/internal/model"
)

const watcherEventBuffer = 4096

type watchRoot struct {
	codebaseID string
	root       string
	// commonDir is the git common dir when root is a worktree root, else "".
	// Two roots that share a non-empty commonDir are worktrees of the same
	// repository; dispatch uses that to keep a parent root from re-indexing a
	// nested sibling worktree's files, mirroring the discovery walk boundary.
	commonDir string
	rules     discovery.IgnoreRules
}

// Watcher converts filesystem events under tracked codebases into per-path
// converge tasks. Each codebase tree is watched recursively with the
// native platform backend (FSEvents on macOS, inotify on Linux), so one
// watch covers a whole tree without a descriptor per file. Changed paths
// are enqueued into the coalescing queue, and ignored paths are dropped
// using the same rule the full scan applies so the watcher and a scan
// agree on what belongs.
//
// Roots can be added and removed at runtime via AddCodebase and
// RemoveCodebase. The shared event channel stays registered with the
// underlying notify backend for the lifetime of the daemon; the
// rjeczalik/notify API exposes only one Stop primitive that tears down
// every watch on a channel, so hot-remove is implemented by dropping
// events whose canonical path no longer maps to a tracked root rather
// than asking notify to unregister a single subtree.
type Watcher struct {
	manager *Manager
	queue   *EventQueue
	events  chan notify.EventInfo
	mu      sync.Mutex
	roots   map[string]watchRoot
}

// NewWatcher constructs a Watcher that enqueues into queue.
func NewWatcher(manager *Manager, queue *EventQueue) *Watcher {
	return &Watcher{
		manager: manager,
		queue:   queue,
		events:  make(chan notify.EventInfo, watcherEventBuffer),
		mu:      sync.Mutex{},
		roots:   map[string]watchRoot{},
	}
}

// Run seeds the roots map from the manager's current registry then
// dispatches events until ctx is cancelled. Codebases added or removed
// after Run starts are picked up by AddCodebase and RemoveCodebase.
func (watcher *Watcher) Run(ctx context.Context) {
	for _, codebase := range watcher.manager.ListIndexes(ctx) {
		watcher.AddCodebase(ctx, codebase)
	}
	slog.InfoContext(ctx, "watcher.started", "component", "daemon", "subcomponent", "watcher", "codebases", watcher.activeRootCount())
	defer notify.Stop(watcher.events)

	for {
		select {
		case <-ctx.Done():
			return
		case event := <-watcher.events:
			watcher.dispatch(event)
		}
	}
}

// AddCodebase registers a recursive watch for the supplied codebase. Safe
// to call before or after Run starts. Idempotent per codebase id.
func (watcher *Watcher) AddCodebase(ctx context.Context, codebase model.Codebase) {
	if codebase.Kind == model.CodebaseKindDocument {
		return
	}

	// The record's rules are the walk-resolved cache the jobs maintain.
	// Empty rules mean no capture has run yet; dispatch passes a few extra
	// events until the first capture persists rules, and converge drops them.
	rules := codebase.ResolvedIgnoreRules

	watcher.mu.Lock()
	if _, found := watcher.roots[codebase.ID]; found {
		watcher.mu.Unlock()
		return
	}
	commonDir, _ := gitworktree.CommonDirAt(codebase.CanonicalPath)
	watcher.roots[codebase.ID] = watchRoot{codebaseID: codebase.ID, root: codebase.CanonicalPath, commonDir: commonDir, rules: rules}
	watcher.mu.Unlock()

	recursivePath := filepath.Join(codebase.CanonicalPath, "...")
	if err := notify.Watch(recursivePath, watcher.events, notify.Create, notify.Remove, notify.Write, notify.Rename); err != nil {
		slog.ErrorContext(ctx, "watcher.register_failed", "component", "daemon", "subcomponent", "watcher", "root", codebase.CanonicalPath, "err", err)
		return
	}
	slog.InfoContext(ctx, "watcher.codebase_added", "component", "daemon", "subcomponent", "watcher", "codebase_id", codebase.ID, "root", codebase.CanonicalPath)
}

// RemoveCodebase drops a codebase from the dispatch table so events for
// its path are no longer enqueued. The underlying notify watch stays
// registered until daemon shutdown because rjeczalik/notify exposes only
// a Stop primitive that tears down every watch on the channel.
func (watcher *Watcher) RemoveCodebase(ctx context.Context, codebaseID string) {
	watcher.mu.Lock()
	defer watcher.mu.Unlock()
	if _, found := watcher.roots[codebaseID]; !found {
		return
	}
	delete(watcher.roots, codebaseID)
	slog.InfoContext(ctx, "watcher.codebase_removed", "component", "daemon", "subcomponent", "watcher", "codebase_id", codebaseID)
}

func (watcher *Watcher) activeRootCount() int {
	watcher.mu.Lock()
	defer watcher.mu.Unlock()
	return len(watcher.roots)
}

func (watcher *Watcher) snapshotRoots() []watchRoot {
	watcher.mu.Lock()
	defer watcher.mu.Unlock()
	roots := make([]watchRoot, 0, len(watcher.roots))
	for _, root := range watcher.roots {
		roots = append(roots, root)
	}
	sort.Slice(roots, func(first int, second int) bool {
		return len(roots[first].root) > len(roots[second].root)
	})
	return roots
}

func (watcher *Watcher) dispatch(event notify.EventInfo) {
	path := event.Path()
	roots := watcher.snapshotRoots()
	covers := covering(roots, path)
	if len(covers) == 0 {
		return
	}
	if info, statErr := os.Lstat(path); statErr == nil && info.IsDir() {
		// A directory event; the recursive watch already covers its files,
		// and the contained files raise their own events.
		return
	}
	for _, root := range covers {
		if gitworktree.PathInsideNestedWorktree(root.root, root.commonDir, path) {
			// A nested same-repo worktree owns this path, so the parent root must
			// not also index it. This reads the on-disk .git topology, so the
			// boundary holds whether or not the worktree is registered as its own
			// codebase yet, matching the discovery walk boundary.
			continue
		}
		relativePath, err := filepath.Rel(root.root, path)
		if err != nil {
			continue
		}
		relativePath = filepath.ToSlash(relativePath)
		if relativePath == "." || relativePath == "" {
			continue
		}
		if excluded, _, _ := discovery.PathIgnored(relativePath, root.rules); excluded {
			continue
		}
		watcher.queue.Enqueue(root.codebaseID, relativePath)
	}
}

func covering(roots []watchRoot, path string) []watchRoot {
	matches := make([]watchRoot, 0, len(roots))
	for _, root := range roots {
		if path == root.root || strings.HasPrefix(path, root.root+string(os.PathSeparator)) {
			matches = append(matches, root)
		}
	}
	return matches
}

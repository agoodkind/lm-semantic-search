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
	"goodkind.io/lm-semantic-search/internal/gitworktree"
	"goodkind.io/lm-semantic-search/internal/model"
)

const watcherEventBuffer = 4096

type watchRoot struct {
	codebaseID string
	root       string
	// commonDir is root's git common dir: <root>/.git for a main worktree (set
	// by AddCodebase's CommonDirAt fallback) or the shared dir for a linked
	// worktree, and "" only when no git dir resolves. dispatch passes it to the
	// resolver's IsIgnoreSourcePath so an edit to <commonDir>/info/exclude
	// invalidates the codebase's rules. That per-event path fires only when
	// <commonDir>/info/exclude sits inside the watched root, which holds for a
	// main worktree. For a linked worktree commonDir points at the main repo's
	// shared .git outside this root, so those edits raise no watcher event and
	// the observer's periodic CheckSources backstop catches them instead.
	// Nested-worktree scope is no longer decided here; the resolver owns that
	// through Ignored.
	commonDir string
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
			watcher.dispatch(ctx, event)
		}
	}
}

// AddCodebase registers a recursive watch for the supplied codebase. Safe
// to call before or after Run starts. Idempotent per codebase id.
func (watcher *Watcher) AddCodebase(ctx context.Context, codebase model.Codebase) {
	if codebase.Kind == model.CodebaseKindDocument {
		return
	}

	watcher.mu.Lock()
	if _, found := watcher.roots[codebase.ID]; found {
		watcher.mu.Unlock()
		return
	}
	// WorktreeCommonDir is persisted only for linked worktrees (per
	// model.Codebase), so it is the source of truth for a linked worktree's
	// shared git common dir and the watcher reads it rather than recomputing. It
	// is empty by design for a main worktree and for a Go-adopted linked worktree
	// (the adopt path does not populate it), so fall back to CommonDirAt when the
	// record value is empty. commonDir feeds IsIgnoreSourcePath's repo
	// info/exclude check and dispatch's nested-worktree scoping. CommonDirAt can
	// still fail for a non-git or malformed root, leaving commonDir empty; both
	// callers degrade safely, treating empty as no info/exclude and no nested
	// scoping.
	commonDir := codebase.WorktreeCommonDir
	if commonDir == "" {
		commonDir, _ = gitworktree.CommonDirAt(codebase.CanonicalPath)
	}
	watcher.roots[codebase.ID] = watchRoot{codebaseID: codebase.ID, root: codebase.CanonicalPath, commonDir: commonDir}
	watcher.mu.Unlock()

	recursivePath := filepath.Join(codebase.CanonicalPath, "...")
	if err := notify.Watch(recursivePath, watcher.events, notify.Create, notify.Remove, notify.Write, notify.Rename); err != nil {
		// Roll the roots entry back so a failed registration does not leave the
		// codebase recorded with no OS watch installed, which would make a later
		// AddCodebase a no-op and block any retry.
		watcher.mu.Lock()
		delete(watcher.roots, codebase.ID)
		watcher.mu.Unlock()
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

func (watcher *Watcher) dispatch(ctx context.Context, event notify.EventInfo) {
	path := event.Path()
	roots := watcher.snapshotRoots()
	covers := covering(roots, path)
	if len(covers) == 0 {
		return
	}
	info, statErr := os.Lstat(path)
	if statErr == nil && info.IsDir() {
		// A directory event; the recursive watch already covers its files,
		// and the contained files raise their own events.
		return
	}
	for _, root := range covers {
		// A changed ignore source re-resolves this covering codebase's rules. The
		// check is a cheap per-event predicate over the loop's covering root, so
		// the hot path takes no manager lock and walks no tree. It runs
		// independent of whether the path is also enqueued below.
		if watcher.manager.indexability.IsIgnoreSourcePath(path, root.commonDir) {
			watcher.manager.observer.Invalidate(root.codebaseID)
		}
		relativePath, err := filepath.Rel(root.root, path)
		if err != nil {
			continue
		}
		relativePath = filepath.ToSlash(relativePath)
		if relativePath == "." || relativePath == "" {
			continue
		}
		// Decide scope and ignore without file info, so delete and rename events
		// (where os.Lstat fails) are filtered too. Otherwise .git lock-file churn
		// would enqueue on removal. The size gate is left to converge and the
		// indexer; the watcher only needs the ignore and scope verdict. Directory
		// events already returned above, so isDir is false here; pathIgnored's
		// ancestor walk still excludes paths under an ignored directory.
		if watcher.manager.indexability.Ignored(ctx, root.codebaseID, root.root, relativePath, false) {
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

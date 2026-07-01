package daemon

import (
	"context"
	"os"
	"sync"
	"time"
)

// ignoreSourceResolver is the slice of the resolver the observer drives: the one
// definition of which on-disk files are ignore sources, and the cache
// invalidation the observer alone now triggers.
type ignoreSourceResolver interface {
	IgnoreSources(ctx context.Context, codebaseID string, root string) []string
	InvalidateRules(codebaseID string)
}

// sourceStamp is the freshness signal the observer records for one ignore source:
// whether it existed at the last check and its modtime then. A change in either
// field means the source's contribution to a codebase's ignore rules may have
// changed, so the resolver entry must rebuild.
type sourceStamp struct {
	exists  bool
	modTime time.Time
}

// ignoreObserver is the single owner of resolver-cache invalidation. Every other
// daemon component routes its "a codebase's ignore rules may have changed" signal
// here instead of calling the resolver's invalidate directly, so invalidation has
// exactly one home. It maps two triggers to invalidate: a caller-supplied signal
// that a codebase's ignore rules may have changed (Invalidate), which the watcher
// raises for a per-event ignore-source path and the config-commit, sync,
// adoption, worktree-discovery, and conversation-registration paths raise after
// an effective-config mutation, and the periodic backstop that stats each
// codebase's ignore sources and notices an edit the watcher missed or that
// happened while the watcher was disabled (CheckSources).
//
// Locking: the observer's mutex guards only lastSeen, the per-codebase freshness
// record CheckSources reads and writes. The observer never holds that mutex while
// calling the resolver, so it cannot deadlock against an in-flight Decide.
// Invalidate only invalidates and takes no lock, so a caller holding manager.mu
// may call it safely, matching the prior direct InvalidateRules calls it
// replaces.
type ignoreObserver struct {
	resolver ignoreSourceResolver

	mu       sync.Mutex
	lastSeen map[string]map[string]sourceStamp
}

// newIgnoreObserver constructs the single ignore-source observer over the
// resolver it drives.
func newIgnoreObserver(resolver ignoreSourceResolver) *ignoreObserver {
	return &ignoreObserver{
		resolver: resolver,
		mu:       sync.Mutex{},
		lastSeen: map[string]map[string]sourceStamp{},
	}
}

// Invalidate drops the codebase's resolver entry so the next decision rebuilds
// the matcher. It is the observer's sole invalidate entry: the watcher raises it
// for a raw filesystem event on one of the codebase's ignore sources, and the
// config-commit, sync, adoption, worktree-discovery, and conversation-
// registration paths raise it after an effective-config mutation. It only
// invalidates and takes no lock, so a caller holding manager.mu may call it
// safely.
func (observer *ignoreObserver) Invalidate(codebaseID string) {
	observer.resolver.InvalidateRules(codebaseID)
}

// CheckSources is the periodic backstop that keeps a codebase's ignore rules
// fresh independent of the file watcher. It stats every path in the codebase's
// IgnoreSources set, compares each against the observer's last-seen freshness
// record, and invalidates the resolver entry when any source changed modtime,
// appeared, or disappeared. It then records the new freshness baseline. Running it
// every sweep catches edits to sources the watcher does not cover (a global
// core.excludesFile, ~/.config/git/ignore, ~/.context/.contextignore, or
// .git/info/exclude) and any source edit made while CLAUDE_CONTEXT_FILE_WATCHER is
// disabled, which is the watcher-off staleness fix.
func (observer *ignoreObserver) CheckSources(ctx context.Context, codebaseID string, root string) {
	sources := observer.resolver.IgnoreSources(ctx, codebaseID, root)
	current := make(map[string]sourceStamp, len(sources))
	for _, source := range sources {
		current[source] = stampForSource(source)
	}

	observer.mu.Lock()
	previous, hadPrevious := observer.lastSeen[codebaseID]
	observer.lastSeen[codebaseID] = current
	observer.mu.Unlock()

	if hadPrevious && sameStamps(previous, current) {
		return
	}
	// Invalidate on the first observation as well as on any later change. The
	// first sweep cannot tell whether a matcher was built before the baseline
	// existed, so an ignore-source edit between that build and this sweep would
	// otherwise be missed and the cached matcher would stay stale. InvalidateRules
	// is a no-op when nothing is cached, so the first sweep costs at most one
	// rebuild and guarantees the matcher matches the baseline just recorded.
	observer.resolver.InvalidateRules(codebaseID)
}

// stampForSource stats one ignore-source path into a freshness signal. A missing
// path records as absent, so its later appearance is detected as a change.
func stampForSource(path string) sourceStamp {
	info, err := os.Stat(path)
	if err != nil {
		return sourceStamp{exists: false, modTime: time.Time{}}
	}
	return sourceStamp{exists: true, modTime: info.ModTime()}
}

// sameStamps reports whether two freshness records cover the same source paths
// with the same existence and modtime. A differing path set means a source
// appeared or disappeared, which is itself a change.
func sameStamps(previous map[string]sourceStamp, current map[string]sourceStamp) bool {
	if len(previous) != len(current) {
		return false
	}
	for path, previousStamp := range previous {
		currentStamp, found := current[path]
		if !found {
			return false
		}
		if previousStamp.exists != currentStamp.exists {
			return false
		}
		if !previousStamp.modTime.Equal(currentStamp.modTime) {
			return false
		}
	}
	return true
}

package daemon

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/indexability"
)

// fakeIgnoreResolver records every invalidation and returns a controllable
// ignore-source set, so a test asserts exactly which codebases the observer
// invalidates without coupling to the resolver's private cache.
type fakeIgnoreResolver struct {
	mu          sync.Mutex
	sources     []string
	invalidated []string
}

func (resolver *fakeIgnoreResolver) IgnoreSources(_ context.Context, _ string) []string {
	resolver.mu.Lock()
	defer resolver.mu.Unlock()
	return slices.Clone(resolver.sources)
}

func (resolver *fakeIgnoreResolver) InvalidateRules(codebaseID string) {
	resolver.mu.Lock()
	defer resolver.mu.Unlock()
	resolver.invalidated = append(resolver.invalidated, codebaseID)
}

func (resolver *fakeIgnoreResolver) invalidations() []string {
	resolver.mu.Lock()
	defer resolver.mu.Unlock()
	return slices.Clone(resolver.invalidated)
}

// TestCheckSourcesInvalidatesOnSourceModtimeChange proves the periodic backstop
// invalidates a codebase on its first observation and again after one of its
// ignore sources changes modtime, while an unchanged re-check adds nothing. The
// first-observation invalidation refreshes a matcher that may have been built
// before the baseline existed. The source is a non-indexed file, so this is the
// watcher-off staleness fix.
func TestCheckSourcesInvalidatesOnSourceModtimeChange(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	excludePath := filepath.Join(root, "info-exclude")
	if err := os.WriteFile(excludePath, []byte("build/\n"), 0o644); err != nil {
		t.Fatalf("write source: %v", err)
	}

	resolver := &fakeIgnoreResolver{sources: []string{excludePath}}
	observer := newIgnoreObserver(resolver)

	observer.CheckSources(context.Background(), "cb", root)
	if got := resolver.invalidations(); !slices.Equal(got, []string{"cb"}) {
		t.Fatalf("first CheckSources invalidations = %v, want [cb] to refresh a possibly-stale matcher", got)
	}

	observer.CheckSources(context.Background(), "cb", root)
	if got := resolver.invalidations(); !slices.Equal(got, []string{"cb"}) {
		t.Fatalf("unchanged CheckSources invalidations = %v, want still [cb]", got)
	}

	bumped := time.Now().Add(2 * time.Second)
	if err := os.Chtimes(excludePath, bumped, bumped); err != nil {
		t.Fatalf("bump modtime: %v", err)
	}

	observer.CheckSources(context.Background(), "cb", root)
	if got := resolver.invalidations(); !slices.Equal(got, []string{"cb", "cb"}) {
		t.Fatalf("changed CheckSources invalidations = %v, want [cb cb]", got)
	}
}

// TestCheckSourcesInvalidatesWhenSourceAppears proves CheckSources notices a new
// ignore source that the resolver begins reporting, since an appeared source can
// change the ignore rules even with no modtime edit to an existing file.
func TestCheckSourcesInvalidatesWhenSourceAppears(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	first := filepath.Join(root, "a")
	if err := os.WriteFile(first, []byte("x\n"), 0o644); err != nil {
		t.Fatalf("write source: %v", err)
	}

	resolver := &fakeIgnoreResolver{sources: []string{first}}
	observer := newIgnoreObserver(resolver)
	observer.CheckSources(context.Background(), "cb", root)
	if got := resolver.invalidations(); !slices.Equal(got, []string{"cb"}) {
		t.Fatalf("first CheckSources invalidations = %v, want [cb] baseline refresh", got)
	}

	second := filepath.Join(root, "b")
	if err := os.WriteFile(second, []byte("y\n"), 0o644); err != nil {
		t.Fatalf("write second source: %v", err)
	}
	resolver.mu.Lock()
	resolver.sources = []string{first, second}
	resolver.mu.Unlock()

	observer.CheckSources(context.Background(), "cb", root)
	if got := resolver.invalidations(); !slices.Equal(got, []string{"cb", "cb"}) {
		t.Fatalf("appeared-source invalidations = %v, want [cb cb]", got)
	}
}

// TestInvalidateInvalidatesGivenCodebase proves the observer's sole invalidate
// entry routes straight to the resolver for exactly the codebase named, the
// signal the watcher and config-commit paths raise.
func TestInvalidateInvalidatesGivenCodebase(t *testing.T) {
	t.Parallel()
	resolver := &fakeIgnoreResolver{}
	observer := newIgnoreObserver(resolver)

	observer.Invalidate("cb")
	if got := resolver.invalidations(); !slices.Equal(got, []string{"cb"}) {
		t.Fatalf("Invalidate invalidations = %v, want [cb]", got)
	}
}

// TestIsIgnoreSourcePath proves the cheap per-event predicate matches a nested
// .gitignore and the repo info/exclude while rejecting an ordinary source file
// and an out-of-tree path, the membership decision the watcher hot path makes
// without a tree walk.
func TestIsIgnoreSourcePath(t *testing.T) {
	t.Parallel()
	resolver := indexability.NewResolver(nil)
	root := "/repo"
	commonDir := filepath.Join(root, ".git")

	nestedGitignore := filepath.Join(root, "pkg", "sub", ".gitignore")
	if !resolver.IsIgnoreSourcePath(nestedGitignore, commonDir) {
		t.Fatalf("nested .gitignore %q should be an ignore source", nestedGitignore)
	}

	infoExclude := filepath.Join(root, ".git", "info", "exclude")
	if !resolver.IsIgnoreSourcePath(infoExclude, commonDir) {
		t.Fatalf("info/exclude %q should be an ignore source", infoExclude)
	}

	ordinary := filepath.Join(root, "main.go")
	if resolver.IsIgnoreSourcePath(ordinary, commonDir) {
		t.Fatalf("ordinary source %q should not be an ignore source", ordinary)
	}

	outside := filepath.Join("/elsewhere", "main.go")
	if resolver.IsIgnoreSourcePath(outside, commonDir) {
		t.Fatalf("out-of-tree path %q should not be an ignore source", outside)
	}
}

package daemon

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
)

// refreshDependencyHealth turns an active backend probe into the health record:
// a clean probe keeps the banner off, a probe failure degrades with the matching
// mode, and the debounce skips a repeat probe until the interval elapses.
func TestRefreshDependencyHealthProbe(t *testing.T) {
	manager, _, _ := newTestManager(t)

	manager.semantic = &fakeSemantic{}
	manager.refreshDependencyHealth(context.Background())
	if manager.DependencyHealth().Degraded() {
		t.Fatal("a clean probe must not degrade the health record")
	}

	resetProbeClock(manager)
	manager.semantic = &fakeSemantic{probeErr: adapterr.NewMilvusUnavailable(nil)}
	manager.refreshDependencyHealth(context.Background())
	if health := manager.DependencyHealth(); !health.Degraded() || health.Mode != dependencyStoreUnavailable {
		t.Fatalf("store probe failure: mode=%q degraded=%v, want %q degraded", health.Mode, health.Degraded(), dependencyStoreUnavailable)
	}

	// Within the debounce window the probe is skipped, so a now-healthy backend
	// does not yet clear the degraded mode.
	manager.semantic = &fakeSemantic{}
	manager.refreshDependencyHealth(context.Background())
	if !manager.DependencyHealth().Degraded() {
		t.Fatal("debounce must skip the probe and keep the prior degraded mode")
	}

	// Past the window the probe runs again and clears the mode.
	resetProbeClock(manager)
	manager.refreshDependencyHealth(context.Background())
	if manager.DependencyHealth().Degraded() {
		t.Fatal("a clean probe past the debounce window must clear the degraded mode")
	}
}

// A clean store probe clears a store outage but must not clear an embedder
// outage: ProbeHealth exercises only the store, while embedder health is observed
// from real embed outcomes. Clearing it on a store probe was a real bug.
func TestStoreProbeDoesNotClearEmbedderOutage(t *testing.T) {
	manager, _, _ := newTestManager(t)

	manager.mu.Lock()
	manager.noteDependencyFailureLocked(adapterr.NewEmbedderUnreachable(nil))
	manager.mu.Unlock()
	if got := manager.DependencyHealth().Mode; got != dependencyEmbedderUnreachable {
		t.Fatalf("setup: mode = %q, want %q", got, dependencyEmbedderUnreachable)
	}

	resetProbeClock(manager)
	manager.semantic = &fakeSemantic{}
	manager.refreshDependencyHealth(context.Background())
	if got := manager.DependencyHealth().Mode; got != dependencyEmbedderUnreachable {
		t.Fatalf("a clean store probe must not clear an embedder outage: mode = %q", got)
	}
}

// GetIndex reports searchable only when the path is indexed and the active
// backend probe succeeds. A store or embedder outage flips searchable to false
// even though the on-disk classification stays KIND_IN_SCOPE_INDEXED.
func TestGetIndexSearchableReflectsBackendHealth(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	server := NewGRPCServer(manager, nil)

	cases := []struct {
		name           string
		semantic       *fakeSemantic
		wantSearchable bool
	}{
		{"healthy backend", &fakeSemantic{}, true},
		{"embedder down", &fakeSemantic{probeErr: adapterr.NewEmbedderUnreachable(nil)}, false},
		{"store down", &fakeSemantic{unavailable: true}, false},
		{"collection not loaded", &fakeSemantic{collectionState: func(context.Context, string) (bool, bool, error) { return true, false, nil }}, false},
		{"collection load state unanswerable", &fakeSemantic{collectionState: func(context.Context, string) (bool, bool, error) {
			return false, false, adapterr.NewMilvusUnavailable(nil)
		}}, false},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			resetProbeClock(manager)
			manager.semantic = testCase.semantic
			resp, getErr := server.GetIndex(context.Background(), &pb.GetIndexRequest{Path: repoPath})
			if getErr != nil {
				t.Fatalf("GetIndex returned error: %v", getErr)
			}
			if resp.GetClassification().GetKind() != pb.PathClassification_KIND_IN_SCOPE_INDEXED {
				t.Fatalf("classification = %v, want KIND_IN_SCOPE_INDEXED", resp.GetClassification().GetKind())
			}
			if got := resp.GetSearchable(); got != testCase.wantSearchable {
				t.Fatalf("searchable = %v, want %v", got, testCase.wantSearchable)
			}
		})
	}
}

// A per-path collection that is not loaded into query nodes, while the store
// itself is reachable, reads not-searchable and loading, NOT the global
// store-unavailable banner. The per-path readiness drives searchable and the
// display; the global dependency banner stays off because no global probe failed.
// A separate genuine store outage still raises the banner (covered by the
// "store down" case in TestGetIndexSearchableReflectsBackendHealth).
func TestGetIndexCollectionNotLoadedReadsLoading(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 1, Status: "completed", CompletedAt: time.Now()}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	resetProbeClock(manager)
	// Store reachable (ProbeHealth nil) but this collection exists and is not loaded.
	manager.semantic = &fakeSemantic{collectionState: func(context.Context, string) (bool, bool, error) {
		return true, false, nil
	}}

	server := NewGRPCServer(manager, nil)
	resp, getErr := server.GetIndex(context.Background(), &pb.GetIndexRequest{Path: repoPath})
	if getErr != nil {
		t.Fatalf("GetIndex returned error: %v", getErr)
	}
	if resp.GetSearchable() {
		t.Fatal("searchable = true for a not-loaded collection, want false")
	}
	if got := resp.GetCodebase().GetDisplayStatus(); got != "loading" {
		t.Fatalf("display status = %q, want loading (not the global waiting banner)", got)
	}
	if dh := resp.GetDependencyHealth(); dh != nil && dh.GetMode() != "" {
		t.Fatalf("a per-path not-loaded collection must not set a global dependency mode, got %q", dh.GetMode())
	}
	if got := resp.GetCollectionReadiness(); got != "loading" {
		t.Fatalf("collection_readiness = %q, want loading", got)
	}
}

func resetProbeClock(manager *Manager) {
	manager.mu.Lock()
	manager.lastDepProbeAt = time.Time{}
	manager.mu.Unlock()
}

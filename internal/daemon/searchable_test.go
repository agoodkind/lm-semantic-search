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
	manager.semantic = &fakeSemantic{probeErr: adapterr.NewEmbedderUnreachable(nil)}
	manager.refreshDependencyHealth(context.Background())
	if health := manager.DependencyHealth(); !health.Degraded() || health.Mode != dependencyEmbedderUnreachable {
		t.Fatalf("embedder probe failure: mode=%q degraded=%v, want %q degraded", health.Mode, health.Degraded(), dependencyEmbedderUnreachable)
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

func resetProbeClock(manager *Manager) {
	manager.mu.Lock()
	manager.lastDepProbeAt = time.Time{}
	manager.mu.Unlock()
}

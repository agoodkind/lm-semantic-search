package daemon

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/model"
)

// newParentWithChildRepo builds a parent directory with one root file and a
// nested child directory with its own file, returning the canonical parent path
// and the canonical child path. It models the codex case: an indexed child
// directory inside a not-yet-indexed parent.
func newParentWithChildRepo(t *testing.T) (parentCanonical string, childCanonical string) {
	t.Helper()
	parentDir := filepath.Join(t.TempDir(), "parent")
	childDir := filepath.Join(parentDir, "child")
	if err := os.MkdirAll(childDir, 0o755); err != nil {
		t.Fatalf("MkdirAll returned error: %v", err)
	}
	if err := os.WriteFile(filepath.Join(parentDir, "root.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	if err := os.WriteFile(filepath.Join(childDir, "leaf.go"), []byte("package child\n"), 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	canonical, err := filepath.EvalSymlinks(parentDir)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	return canonical, filepath.Join(canonical, "child")
}

// registerIndexedChild seeds a settled, indexed child codebase whose embedding
// config matches cfg so the merge-down path treats its collection as reusable.
func registerIndexedChild(manager *Manager, id string, childCanonical string, cfg model.IndexConfig, collection string) {
	manager.mu.Lock()
	manager.codebases[id] = model.Codebase{
		ID:              id,
		CanonicalPath:   childCanonical,
		Status:          model.CodebaseStatusIndexed,
		CollectionName:  collection,
		EffectiveConfig: cfg,
		LastSuccessfulRun: &model.IndexRunSummary{
			IndexedFiles: 1,
			TotalChunks:  3,
			Status:       "completed",
			CompletedAt:  clock.Now(),
		},
	}
	manager.mu.Unlock()
}

// TestMergeDownReusesAndAbsorbsIndexedChild proves that a from-scratch parent
// build loads reuse vectors from an already-indexed child's collection, then
// absorbs the child: the child leaves the registry, but its shared collection
// is never dropped.
func TestMergeDownReusesAndAbsorbsIndexedChild(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 2)
	fake := &fakeSemantic{}
	manager.semantic = fake
	var mu sync.Mutex
	embedded := make([]string, 0)
	manager.runner = recordingRunner(&mu, &embedded)

	parentCanonical, childCanonical := newParentWithChildRepo(t)
	cfg := defaultIndexConfig()
	cfg.IgnoreDigest = "sha256:merge-down"
	const childID = "cb-merge-child"
	const childCollection = "hybrid_code_chunks_child"
	registerIndexedChild(manager, childID, childCanonical, cfg, childCollection)

	_, job := seedBootstrapCodebase(t, manager, parentCanonical, cfg)
	manager.runBootstrap(context.Background(), job, newCodeItemSource(manager.runner, manager.indexability, job.CodebaseID, job.CanonicalPath, job.Config))

	fake.mu.Lock()
	reuseCalls := fake.reuseCollections
	dropped := append([]string(nil), fake.dropped...)
	fake.mu.Unlock()

	reusedChildCollection := false
	for _, set := range reuseCalls {
		for _, name := range set {
			if name == childCollection {
				reusedChildCollection = true
			}
		}
	}
	if !reusedChildCollection {
		t.Fatalf("merge-down did not load reuse vectors from the child collection; reuse calls = %v", reuseCalls)
	}
	if len(dropped) != 0 {
		t.Fatalf("absorb dropped collection(s) %v; the shared child collection must never be dropped", dropped)
	}

	manager.mu.Lock()
	_, childStillTracked := manager.codebases[childID]
	manager.mu.Unlock()
	if childStillTracked {
		t.Fatal("child codebase was not absorbed; it is still in the registry")
	}

	codebase, _, found, _, err := manager.GetIndex(context.Background(), parentCanonical)
	if err != nil || !found {
		t.Fatalf("parent GetIndex after merge-down: found=%v err=%v", found, err)
	}
	if codebase.Status != model.CodebaseStatusIndexed {
		t.Fatalf("parent status = %q, want indexed", codebase.Status)
	}
}

// TestMergeUpRedirectsNestedIndexToParent proves that indexing a nested path
// already covered by an indexed parent does not build a second index: it
// resolves to the parent and returns the parent's sync job.
func TestMergeUpRedirectsNestedIndexToParent(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 2)
	manager.semantic = &fakeSemantic{}
	var mu sync.Mutex
	embedded := make([]string, 0)
	manager.runner = recordingRunner(&mu, &embedded)

	parentCanonical, childCanonical := newParentWithChildRepo(t)
	cfg := manager.enrichIndexConfig(defaultIndexConfig())
	cfg.IgnoreDigest = digestIndexConfig(cfg)
	manager.mu.Lock()
	manager.codebases["cb-merge-parent"] = model.Codebase{
		ID:                "cb-merge-parent",
		CanonicalPath:     parentCanonical,
		Status:            model.CodebaseStatusIndexed,
		CollectionName:    "hybrid_code_chunks_parent",
		EffectiveConfig:   cfg,
		LastSuccessfulRun: &model.IndexRunSummary{IndexedFiles: 2, TotalChunks: 5, Status: "completed", CompletedAt: clock.Now()},
	}
	manager.mu.Unlock()

	job, codebase, deduped, _, err := manager.StartIndex(context.Background(), childCanonical, testClientInfo(), defaultIndexConfig(), false)
	if err != nil {
		t.Fatalf("StartIndex returned error: %v", err)
	}
	if deduped {
		t.Fatal("merge-up redirect should not report deduplicated")
	}
	if codebase.ID != "cb-merge-parent" || codebase.CanonicalPath != parentCanonical {
		t.Fatalf("merge-up resolved to %q (%s), want the parent cb-merge-parent (%s)", codebase.ID, codebase.CanonicalPath, parentCanonical)
	}
	if job.ID == "" {
		t.Fatal("merge-up should return the parent's sync job")
	}

	manager.mu.Lock()
	_, childTracked := manager.findCodebaseByExactRoot(childCanonical)
	manager.mu.Unlock()
	if childTracked {
		t.Fatal("merge-up created a redundant child codebase instead of resolving to the parent")
	}

	// Let the redirected sync job settle so its goroutine stops touching the
	// repo before the temp dirs are removed.
	waitForCondition(t, func() bool {
		manager.mu.Lock()
		settled, ok := manager.jobs[job.ID]
		manager.mu.Unlock()
		return ok && (settled.State == model.JobStateCompleted || settled.State == model.JobStateFailed || settled.State == model.JobStateCancelled)
	})
}

func TestChunkUnderPrefix(t *testing.T) {
	cases := []struct {
		relativePath string
		prefix       string
		want         bool
	}{
		{"child/leaf.go", "child", true},
		{"child", "child", true},
		{"childish/leaf.go", "child", false},
		{"other/leaf.go", "child", false},
		{"child/deep/leaf.go", "child", true},
	}
	for _, testCase := range cases {
		if got := chunkUnderPrefix(testCase.relativePath, testCase.prefix); got != testCase.want {
			t.Fatalf("chunkUnderPrefix(%q, %q) = %v, want %v", testCase.relativePath, testCase.prefix, got, testCase.want)
		}
	}
}

func TestSubtreePrefixResolvesNestedPath(t *testing.T) {
	parentCanonical, childCanonical := newParentWithChildRepo(t)
	if got := subtreePrefix(childCanonical, parentCanonical); got != "child" {
		t.Fatalf("subtreePrefix(child, parent) = %q, want \"child\"", got)
	}
	if got := subtreePrefix(parentCanonical, parentCanonical); got != "" {
		t.Fatalf("subtreePrefix(parent, parent) = %q, want \"\"", got)
	}
}

func TestRenderIndexedDescendantsHintNamesSubfolderAndCommand(t *testing.T) {
	descendants := []model.Codebase{
		{
			CanonicalPath:     "/repo/codex/source/codex-rs",
			LastSuccessfulRun: &model.IndexRunSummary{IndexedFiles: 4292},
		},
	}
	out := descendantsHint("/repo/codex/source", descendants)
	for _, want := range []string{"4292", "codex-rs", "index_codebase /repo/codex/source"} {
		if !strings.Contains(out, want) {
			t.Fatalf("descendants hint missing %q in:\n%s", want, out)
		}
	}
}

func TestRenderCoveringResolutionNamesLargerIndex(t *testing.T) {
	parentCanonical, childCanonical := newParentWithChildRepo(t)
	codebase := &model.Codebase{CanonicalPath: parentCanonical}
	out := coveringResolutionLine(childCanonical, true, codebase)
	for _, want := range []string{"Resolved to larger index", parentCanonical, "child/"} {
		if !strings.Contains(out, want) {
			t.Fatalf("covering resolution missing %q in:\n%s", want, out)
		}
	}
	if got := coveringResolutionLine(parentCanonical, true, codebase); got != "" {
		t.Fatalf("covering resolution for the root itself = %q, want empty", got)
	}
}

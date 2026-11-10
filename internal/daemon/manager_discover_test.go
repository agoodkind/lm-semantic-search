package daemon

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/view"
)

// indexedSiblingScene builds a main repo with one already-indexed sibling
// codebase registered directly (no live build), plus an untracked linked
// worktree the daemon can discover on a read. It pins deferredBuildDelay to an
// hour so the deferred build timer never fires mid-test, and returns the
// manager and the canonical worktree root the discovery read should resolve to.
func indexedSiblingScene(t *testing.T) (*Manager, string) {
	t.Helper()

	manager, _, _ := newTestManager(t)
	manager.runner = fakeRunner{}
	manager.deferredBuildDelay = time.Hour
	manager.semantic = &fakeSemantic{
		hasCollectionForPath: func(context.Context, string) (bool, error) { return false, nil },
	}

	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(base, "feature")
	makeLinkedWorktree(t, mainRoot, "feature", worktreeDir, "feature")

	// Register the main worktree as an already-indexed sibling so the discovery
	// trigger's "has an indexed sibling" condition holds without driving a build.
	registerSiblingCodebase(t, manager, mainRoot, func(c *model.Codebase) {
		c.Status = model.CodebaseStatusIndexed
		c.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 1, Status: "completed"}
	})

	return manager, evalSym(t, worktreeDir)
}

// jobsForCodebase counts the jobs the manager tracks for one codebase id.
func jobsForCodebase(manager *Manager, codebaseID string) int {
	return len(manager.ListJobs(codebaseID))
}

// TestGetIndexDiscoversWorktreeWithoutJob proves a status read of an untracked
// worktree of an indexed sibling registers it as a discovered codebase with no
// active job and starts no job synchronously, because the build is deferred.
func TestGetIndexDiscoversWorktreeWithoutJob(t *testing.T) {
	manager, worktreeRoot := indexedSiblingScene(t)

	codebase, activeJob, found, _, err := manager.GetIndex(context.Background(), worktreeRoot)
	if err != nil {
		t.Fatalf("GetIndex(worktree) returned error: %v", err)
	}
	if !found {
		t.Fatal("GetIndex(worktree) did not resolve to a codebase")
	}
	if codebase.Status != model.CodebaseStatusDiscovered {
		t.Fatalf("discovered codebase Status = %q, want %q", codebase.Status, model.CodebaseStatusDiscovered)
	}
	if !codebase.PolicyPendingInitialization {
		t.Fatal("discovered codebase PolicyPendingInitialization = false, want true before its first build")
	}
	if codebase.CanonicalPath != worktreeRoot {
		t.Fatalf("discovered codebase CanonicalPath = %q, want %q", codebase.CanonicalPath, worktreeRoot)
	}
	if codebase.ActiveJobID != "" {
		t.Fatalf("discovered codebase ActiveJobID = %q, want empty", codebase.ActiveJobID)
	}
	if activeJob != nil {
		t.Fatalf("GetIndex returned active job %s, want nil for a discovered codebase", activeJob.ID)
	}
	if got := jobsForCodebase(manager, codebase.ID); got != 0 {
		t.Fatalf("discovered codebase has %d jobs, want 0 (the read must not start an embed)", got)
	}
}

func TestGetIndexDiscoversWorktreeWithExistingCollectionInitialized(t *testing.T) {
	manager, worktreeRoot := indexedSiblingScene(t)
	manager.semantic = &fakeSemantic{
		hasCollectionForPath: func(context.Context, string) (bool, error) { return true, nil },
	}

	codebase, _, found, _, err := manager.GetIndex(context.Background(), worktreeRoot)
	if err != nil || !found {
		t.Fatalf("GetIndex(worktree) returned err=%v found=%v", err, found)
	}
	if codebase.PolicyPendingInitialization {
		t.Fatal("discovered worktree with an existing collection PolicyPendingInitialization = true, want false")
	}
}

func TestGetIndexDiscoversWorktreeWithoutCollectionEvidencePendingInitialization(t *testing.T) {
	manager, worktreeRoot := indexedSiblingScene(t)
	manager.semantic = nil

	codebase, _, found, _, err := manager.GetIndex(context.Background(), worktreeRoot)
	if err != nil || !found {
		t.Fatalf("GetIndex(worktree) returned err=%v found=%v", err, found)
	}
	if !codebase.PolicyPendingInitialization {
		t.Fatal("worktree without collection evidence PolicyPendingInitialization = false, want true")
	}
}

// TestSearchCodeDiscoversWorktreeReturnsForecastNote proves a search of the same
// untracked worktree returns no results, a non-empty discovery state note, and
// starts no job. It must not fall back to the indexed sibling's results.
func TestSearchCodeDiscoversWorktreeReturnsForecastNote(t *testing.T) {
	manager, worktreeRoot := indexedSiblingScene(t)

	outcome, err := manager.SearchCode(context.Background(), worktreeRoot, "anything", 5, nil)
	if err != nil {
		t.Fatalf("SearchCode(worktree) returned error: %v", err)
	}
	if len(outcome.Results) != 0 {
		t.Fatalf("SearchCode returned %d results, want 0 for a just-discovered worktree", len(outcome.Results))
	}
	if outcome.StateNote == "" {
		t.Fatal("SearchCode returned an empty StateNote, want a discovery note")
	}
	if outcome.Codebase.Status != model.CodebaseStatusDiscovered {
		t.Fatalf("SearchCode outcome codebase Status = %q, want %q", outcome.Codebase.Status, model.CodebaseStatusDiscovered)
	}
	if got := jobsForCodebase(manager, outcome.Codebase.ID); got != 0 {
		t.Fatalf("discovered codebase has %d jobs after search, want 0", got)
	}
}

// TestStartDeferredBuildStartsOneBootstrap proves driving the deferred build
// directly starts exactly one job for the discovered worktree and flips it into
// the indexing state. The build path goes through StartIndex, which dedups.
func TestStartDeferredBuildStartsOneBootstrap(t *testing.T) {
	manager, worktreeRoot := indexedSiblingScene(t)

	// Hold the per-file embed open so the codebase stays in the indexing state
	// long enough to observe, then release it for a clean teardown.
	release := make(chan struct{})
	manager.runner = fakeRunner{
		indexOne: func(ctx context.Context, root string, relativePath string, indexConfig model.IndexConfig) (indexer.OneFileResult, error) {
			<-release
			content := "package feature\n"
			return indexer.OneFileResult{
				Chunks: []model.StoredChunk{{
					Content:       content,
					RelativePath:  relativePath,
					StartLine:     1,
					EndLine:       1,
					Language:      "go",
					FileExtension: ".go",
				}},
				FileHash: hashText(content),
				Skipped:  false,
				Removed:  false,
			}, nil
		},
	}

	// First discover the worktree so a discovered codebase record exists.
	discovered, _, found, _, err := manager.GetIndex(context.Background(), worktreeRoot)
	if err != nil || !found {
		t.Fatalf("GetIndex(worktree) returned err=%v found=%v", err, found)
	}
	if got := jobsForCodebase(manager, discovered.ID); got != 0 {
		t.Fatalf("pre-build: discovered codebase has %d jobs, want 0", got)
	}

	manager.startDeferredBuild(context.Background(), worktreeRoot)

	waitForCodebaseStatus(t, manager, worktreeRoot, model.CodebaseStatusIndexing)
	codebase, _, found, _, err := manager.GetIndex(context.Background(), worktreeRoot)
	if err != nil || !found {
		t.Fatalf("GetIndex(worktree) after build start returned err=%v found=%v", err, found)
	}
	if !codebase.PolicyPendingInitialization {
		t.Fatal("building worktree PolicyPendingInitialization = false, want true until an explicit index initializes policy")
	}
	if got := jobsForCodebase(manager, discovered.ID); got != 1 {
		t.Fatalf("deferred build started %d jobs, want exactly 1", got)
	}

	close(release)
	waitForCodebaseStatus(t, manager, worktreeRoot, model.CodebaseStatusIndexed)
}

// TestAdoptUnregisteredCodebaseMarksIndexedAndDefersRefresh proves adoption marks
// the codebase indexed and starts no sync job synchronously: the refresh sync is
// deferred on the same timer the discovered-worktree build uses, pinned to an
// hour here so it cannot fire during the assertions.
func TestAdoptUnregisteredCodebaseMarksIndexedAndDefersRefresh(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 2)
	manager.deferredBuildDelay = time.Hour
	manager.semantic = &fakeSemantic{
		collectionName:       func(path string) string { return "cc_" + filepath.Base(path) },
		hasCollectionForPath: func(context.Context, string) (bool, error) { return true, nil },
	}

	canonical := newCapTestRepo(t)

	codebase, adopted := manager.adoptUnregisteredCodebase(context.Background(), canonical)
	if !adopted {
		t.Fatal("adoptUnregisteredCodebase returned adopted=false")
	}
	if codebase.Status != model.CodebaseStatusIndexed {
		t.Fatalf("adopted codebase Status = %q, want %q", codebase.Status, model.CodebaseStatusIndexed)
	}
	if got := jobsForCodebase(manager, codebase.ID); got != 0 {
		t.Fatalf("adoption started %d jobs synchronously, want 0 (the refresh sync is deferred)", got)
	}
}

func writeWorktreeFile(t *testing.T, path string, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir for %s: %v", path, err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func evalSym(t *testing.T, path string) string {
	t.Helper()
	out, err := filepath.EvalSymlinks(path)
	if err != nil {
		t.Fatalf("eval symlinks %s: %v", path, err)
	}
	return out
}

// makeMainRepo writes a main worktree (a .git directory) with one Go file.
func makeMainRepo(t *testing.T, root string) {
	t.Helper()
	writeWorktreeFile(t, filepath.Join(root, ".git", "HEAD"), "ref: refs/heads/main\n")
	writeWorktreeFile(t, filepath.Join(root, "main.go"), "package main\n")
}

// makeLinkedWorktree registers a linked worktree of the repo rooted at mainRoot,
// rooting the worktree at worktreeDir on the given branch with one Go file.
func makeLinkedWorktree(t *testing.T, mainRoot string, name string, worktreeDir string, branch string) {
	t.Helper()
	perWorktree := filepath.Join(mainRoot, ".git", "worktrees", name)
	writeWorktreeFile(t, filepath.Join(perWorktree, "commondir"), "../..\n")
	writeWorktreeFile(t, filepath.Join(perWorktree, "gitdir"), filepath.Join(worktreeDir, ".git")+"\n")
	writeWorktreeFile(t, filepath.Join(perWorktree, "HEAD"), "ref: refs/heads/"+branch+"\n")
	writeWorktreeFile(t, filepath.Join(worktreeDir, ".git"), "gitdir: "+perWorktree+"\n")
	writeWorktreeFile(t, filepath.Join(worktreeDir, "feature.go"), "package feature\n")
}

func waitForCodebaseSettled(t *testing.T, manager *Manager, path string) model.Codebase {
	t.Helper()
	var settled model.Codebase
	waitForCondition(t, func() bool {
		codebase, _, ok, _, err := manager.GetIndex(context.Background(), path)
		if err != nil || !ok || codebase.LastSuccessfulRun == nil {
			return false
		}
		settled = codebase
		return true
	})
	return settled
}

// TestGetIndexResolvesNestedWorktreeToOwnCodebase proves a worktree nested under
// an indexed parent repo resolves to its own codebase, never the covering
// parent, and that the daemon auto-creates that codebase on first use.
func TestGetIndexResolvesNestedWorktreeToOwnCodebase(t *testing.T) {
	manager, _, _ := newTestManager(t)
	manager.runner = fakeRunner{}

	mainRoot := filepath.Join(t.TempDir(), "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(mainRoot, ".claude", "worktrees", "foo")
	makeLinkedWorktree(t, mainRoot, "foo", worktreeDir, "feature")

	if _, _, _, _, err := manager.StartIndex(context.Background(), mainRoot, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget); err != nil {
		t.Fatalf("StartIndex(main) returned error: %v", err)
	}
	mainCodebase := waitForCodebaseSettled(t, manager, mainRoot)

	codebase, _, found, _, err := manager.GetIndex(context.Background(), worktreeDir)
	if err != nil {
		t.Fatalf("GetIndex(worktree) returned error: %v", err)
	}
	if !found {
		t.Fatal("GetIndex(worktree) did not resolve to a codebase")
	}
	if codebase.ID == mainCodebase.ID {
		t.Fatalf("worktree resolved to the parent codebase %q; it must be its own", mainCodebase.ID)
	}
	if codebase.CanonicalPath != evalSym(t, worktreeDir) {
		t.Fatalf("worktree codebase CanonicalPath = %q, want %q", codebase.CanonicalPath, evalSym(t, worktreeDir))
	}

	// let the auto-created build settle before cleanup removes the temp dirs.
	waitForCodebaseSettled(t, manager, worktreeDir)
}

// TestGetIndexResolvesExternalWorktreeToOwnCodebase proves an external sibling
// worktree (outside the parent tree) resolves to its own codebase once a sibling
// of the same repo is indexed.
func TestGetIndexResolvesExternalWorktreeToOwnCodebase(t *testing.T) {
	manager, _, _ := newTestManager(t)
	manager.runner = fakeRunner{}

	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(base, "feature")
	makeLinkedWorktree(t, mainRoot, "feature", worktreeDir, "feature")

	if _, _, _, _, err := manager.StartIndex(context.Background(), mainRoot, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget); err != nil {
		t.Fatalf("StartIndex(main) returned error: %v", err)
	}
	mainCodebase := waitForCodebaseSettled(t, manager, mainRoot)

	codebase, _, found, _, err := manager.GetIndex(context.Background(), worktreeDir)
	if err != nil {
		t.Fatalf("GetIndex(external worktree) returned error: %v", err)
	}
	if !found {
		t.Fatal("GetIndex(external worktree) did not resolve to a codebase")
	}
	if codebase.ID == mainCodebase.ID {
		t.Fatalf("external worktree resolved to the parent codebase; it must be its own")
	}
	if codebase.CanonicalPath != evalSym(t, worktreeDir) {
		t.Fatalf("external worktree CanonicalPath = %q, want %q", codebase.CanonicalPath, evalSym(t, worktreeDir))
	}

	waitForCodebaseSettled(t, manager, worktreeDir)
}

// TestStartIndexParentDoesNotAbsorbNestedWorktree proves a nested worktree that
// is already its own codebase survives a (re)index of the parent: the parent
// must not absorb a sibling worktree the way it absorbs ordinary child indexes.
func TestStartIndexParentDoesNotAbsorbNestedWorktree(t *testing.T) {
	manager, _, _ := newTestManager(t)
	manager.runner = fakeRunner{}
	manager.semantic = &fakeSemantic{
		collectionName:       func(path string) string { return "cc_" + filepath.Base(path) },
		hasCollectionForPath: func(context.Context, string) (bool, error) { return false, nil },
	}

	mainRoot := filepath.Join(t.TempDir(), "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(mainRoot, ".claude", "worktrees", "foo")
	makeLinkedWorktree(t, mainRoot, "foo", worktreeDir, "feature")

	if _, _, _, _, err := manager.StartIndex(context.Background(), worktreeDir, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget); err != nil {
		t.Fatalf("StartIndex(worktree) returned error: %v", err)
	}
	worktreeCodebase := waitForCodebaseSettled(t, manager, worktreeDir)

	if _, _, _, _, err := manager.StartIndex(context.Background(), mainRoot, testClientInfo(), defaultIndexConfig(), true, emptyAdmissionBudget); err != nil {
		t.Fatalf("StartIndex(main, force) returned error: %v", err)
	}
	waitForCodebaseSettled(t, manager, mainRoot)

	resolved, _, found, _, err := manager.GetIndex(context.Background(), worktreeDir)
	if err != nil {
		t.Fatalf("GetIndex(worktree) after parent index returned error: %v", err)
	}
	if !found || resolved.ID != worktreeCodebase.ID {
		t.Fatalf("worktree codebase was absorbed by the parent: got found=%v id=%q want id=%q", found, resolved.ID, worktreeCodebase.ID)
	}
	if resolved.CanonicalPath != evalSym(t, worktreeDir) {
		t.Fatalf("after parent index, worktree CanonicalPath = %q, want %q", resolved.CanonicalPath, evalSym(t, worktreeDir))
	}
}

func startWorktreeReuseBuild(t *testing.T, declareSource bool) *fakeSemantic {
	t.Helper()

	manager, _, _ := newTestManager(t)
	manager.runner = fakeRunner{}
	fake := &fakeSemantic{
		collectionName:       func(path string) string { return "cc_" + filepath.Base(path) },
		hasCollectionForPath: func(context.Context, string) (bool, error) { return false, nil },
	}
	manager.semantic = fake

	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(base, "feature")
	makeLinkedWorktree(t, mainRoot, "feature", worktreeDir, "feature")

	if _, _, _, _, err := manager.StartIndex(context.Background(), mainRoot, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget); err != nil {
		t.Fatalf("StartIndex(main) returned error: %v", err)
	}
	waitForCodebaseSettled(t, manager, mainRoot)
	fake.mu.Lock()
	fake.reuseContentCollections = nil
	fake.mu.Unlock()

	source := model.WorktreeSourceRequest{SourcePath: "", StartingCommit: ""}
	if declareSource {
		source.SourcePath = mainRoot
	}
	noPolicy := model.SchedulingPolicyPatch{Priority: nil, Quiet: nil, IdleAfterSeconds: nil}
	if _, _, _, _, err := manager.StartIndexWithSource(context.Background(), worktreeDir, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget, noPolicy, source); err != nil {
		t.Fatalf("StartIndexWithSource(worktree) returned error: %v", err)
	}
	waitForCodebaseSettled(t, manager, worktreeDir)
	return fake
}

func requireWorktreeReuseReads(t *testing.T, fake *fakeSemantic, wantCollections []string) {
	t.Helper()

	readCollections := fake.reuseContentCollectionNames()
	if len(readCollections) == 0 {
		t.Fatal("The build recorded no content lookup.")
	}
	if !slices.Equal(readCollections, wantCollections) {
		t.Fatalf("The content lookups read collections %v instead of the expected collections %v.", readCollections, wantCollections)
	}
	for _, call := range fake.reuseCollectionsSnapshot() {
		if slices.Contains(call, "cc_repo") {
			t.Fatalf("The build must not read the whole collection cc_repo in recorded call %v.", call)
		}
	}
}

func TestWorktreeBuildReadsTargetAndDeclaredSource(t *testing.T) {
	fake := startWorktreeReuseBuild(t, true)
	requireWorktreeReuseReads(t, fake, []string{"cc_feature", "cc_repo"})
}

func TestWorktreeBuildWithoutDeclaredSourceReadsOnlyTarget(t *testing.T) {
	fake := startWorktreeReuseBuild(t, false)
	requireWorktreeReuseReads(t, fake, []string{"cc_feature"})
}

func registerSiblingCodebase(t *testing.T, manager *Manager, mainRoot string, mutate func(*model.Codebase)) {
	t.Helper()
	codebase := newCodebaseRecord(evalSym(t, mainRoot))
	codebase.CollectionName = "cc_repo"
	codebase.EffectiveConfig = defaultIndexConfig()
	mutate(&codebase)
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
}

func TestWorktreeStatusReportsDeclaredSourceSelection(t *testing.T) {
	completedRun := &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 1, Status: "completed"}
	cases := []struct {
		name           string
		declareSource  bool
		registerSource func(*model.Codebase)
		wantReason     view.WorktreeReuseReason
	}{
		{
			name:          "busy indexed source",
			declareSource: true,
			registerSource: func(source *model.Codebase) {
				source.Status = model.CodebaseStatusIndexing
				source.ActiveJobID = "job-sync-inflight"
				source.LastSuccessfulRun = completedRun
			},
			wantReason: view.WorktreeReuseAvailable,
		},
		{
			name:          "adopted source",
			declareSource: true,
			registerSource: func(source *model.Codebase) {
				source.Status = model.CodebaseStatusIndexed
				source.LastSuccessfulRun = nil
			},
			wantReason: view.WorktreeReuseAvailable,
		},
		{
			name:          "never indexed source",
			declareSource: true,
			registerSource: func(source *model.Codebase) {
				source.Status = model.CodebaseStatusIndexing
				source.LastSuccessfulRun = nil
			},
			wantReason: view.WorktreeReuseSourceNotIndexed,
		},
		{
			name:          "source with another embedding model",
			declareSource: true,
			registerSource: func(source *model.Codebase) {
				source.Status = model.CodebaseStatusIndexed
				source.LastSuccessfulRun = completedRun
				source.EffectiveConfig.EmbeddingModel = "some-other-model"
			},
			wantReason: view.WorktreeReuseSourceEmbeddingConfigurationDiffers,
		},
		{
			name:           "unregistered source",
			declareSource:  true,
			registerSource: nil,
			wantReason:     view.WorktreeReuseSourceNotRegistered,
		},
		{
			name:          "indexed sibling without a declared source",
			declareSource: false,
			registerSource: func(source *model.Codebase) {
				source.Status = model.CodebaseStatusIndexed
				source.LastSuccessfulRun = completedRun
			},
			wantReason: view.WorktreeReuseAvailable,
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			manager, _, _ := newTestManager(t)
			manager.runner = fakeRunner{}
			base := t.TempDir()
			mainRoot := filepath.Join(base, "repo")
			makeMainRepo(t, mainRoot)
			worktreeDir := filepath.Join(base, "feature")
			makeLinkedWorktree(t, mainRoot, "feature", worktreeDir, "feature")

			source := model.WorktreeSourceRequest{SourcePath: "", StartingCommit: ""}
			if testCase.declareSource {
				source.SourcePath = mainRoot
			}
			noPolicy := model.SchedulingPolicyPatch{Priority: nil, Quiet: nil, IdleAfterSeconds: nil}
			if _, _, _, _, err := manager.StartIndexWithSource(context.Background(), worktreeDir, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget, noPolicy, source); err != nil {
				t.Fatalf("StartIndexWithSource(worktree) returned error: %v", err)
			}
			target := waitForCodebaseSettled(t, manager, worktreeDir)
			if testCase.registerSource != nil {
				registerSiblingCodebase(t, manager, mainRoot, func(sourceCodebase *model.Codebase) {
					sourceCodebase.EffectiveConfig = target.EffectiveConfig
					testCase.registerSource(sourceCodebase)
				})
			}

			response, err := NewGRPCServer(manager, nil).GetIndex(context.Background(), &pb.GetIndexRequest{
				Path:   target.CanonicalPath,
				Client: &pb.ClientInfo{Name: "worktree-reuse-test"},
			})
			if err != nil {
				t.Fatalf("GetIndex returned error: %v", err)
			}
			gotReason := view.WorktreeReuseReason(response.GetCodebase().GetWorktreeReuseUnavailableReason())
			if gotReason != testCase.wantReason {
				t.Fatalf("GetIndex returned reuse reason %q instead of the expected reason %q.", gotReason, testCase.wantReason)
			}
			reasonField := "worktree_reuse_unavailable_reason=" + string(testCase.wantReason)
			hasReasonField := strings.Contains(response.GetDisplayText(), "worktree_reuse_unavailable_reason=")
			if testCase.wantReason == view.WorktreeReuseAvailable && hasReasonField {
				t.Fatalf("The status text %s must omit the reuse reason field.", response.GetDisplayText())
			}
			if testCase.wantReason != view.WorktreeReuseAvailable && !strings.Contains(response.GetDisplayText(), reasonField) {
				t.Fatalf("The field %q does not appear in the status text %s.", reasonField, response.GetDisplayText())
			}
		})
	}
}

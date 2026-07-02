package daemon

import (
	"context"
	"testing"

	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func TestDecideStartIndexMode(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name          string
		codebaseFound bool
		status        model.CodebaseStatus
		configMatches bool
		force         bool
		presence      collectionPresence
		want          startIndexMode
	}{
		{name: "indexed matching present already indexed", codebaseFound: true, status: model.CodebaseStatusIndexed, configMatches: true, force: false, presence: collectionPresencePresent, want: startIndexModeAlreadyIndexed},
		{name: "indexed matching unknown still already indexed", codebaseFound: true, status: model.CodebaseStatusIndexed, configMatches: true, force: false, presence: collectionPresenceUnknown, want: startIndexModeAlreadyIndexed},
		{name: "indexed matching missing bootstraps", codebaseFound: true, status: model.CodebaseStatusIndexed, configMatches: true, force: false, presence: collectionPresenceMissing, want: startIndexModeBootstrap},
		{name: "force on indexed present stays incremental", codebaseFound: true, status: model.CodebaseStatusIndexed, configMatches: true, force: true, presence: collectionPresencePresent, want: startIndexModeIncremental},
		{name: "force on indexed unknown stays incremental", codebaseFound: true, status: model.CodebaseStatusIndexed, configMatches: true, force: true, presence: collectionPresenceUnknown, want: startIndexModeIncremental},
		{name: "stale present stays incremental", codebaseFound: true, status: model.CodebaseStatusStale, configMatches: true, force: false, presence: collectionPresencePresent, want: startIndexModeIncremental},
		{name: "stale unknown stays incremental", codebaseFound: true, status: model.CodebaseStatusStale, configMatches: true, force: false, presence: collectionPresenceUnknown, want: startIndexModeIncremental},
		{name: "stale missing bootstraps", codebaseFound: true, status: model.CodebaseStatusStale, configMatches: true, force: false, presence: collectionPresenceMissing, want: startIndexModeBootstrap},
		{name: "not indexed missing bootstraps", codebaseFound: true, status: model.CodebaseStatusNotIndexed, configMatches: false, force: false, presence: collectionPresenceMissing, want: startIndexModeBootstrap},
		{name: "untracked present increments existing collection", codebaseFound: false, status: model.CodebaseStatusNotIndexed, configMatches: false, force: false, presence: collectionPresencePresent, want: startIndexModeIncremental},
		{name: "untracked unknown bootstraps", codebaseFound: false, status: model.CodebaseStatusNotIndexed, configMatches: false, force: false, presence: collectionPresenceUnknown, want: startIndexModeBootstrap},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			got := decideStartIndexMode(testCase.codebaseFound, testCase.status, testCase.configMatches, testCase.force, testCase.presence)
			if got != testCase.want {
				t.Fatalf("decideStartIndexMode() = %v, want %v", got, testCase.want)
			}
		})
	}
}

func emptyDiffEvidence(presence collectionPresence, rows int32, rowsKnown bool) collectionEvidence {
	return collectionEvidence{
		presence:   presence,
		rows:       rows,
		rowsKnown:  rowsKnown,
		collection: "",
		nameSource: "",
	}
}

func TestDecideEmptyDiffMode(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name          string
		evidence      collectionEvidence
		seedFileCount int
		want          emptyDiffMode
	}{
		{
			name:          "present populated collection completes noop",
			evidence:      emptyDiffEvidence(collectionPresencePresent, 7, true),
			seedFileCount: 4,
			want:          emptyDiffModeCompleteNoop,
		},
		{
			name:          "present emptied collection with seed bootstraps",
			evidence:      emptyDiffEvidence(collectionPresencePresent, 0, true),
			seedFileCount: 4,
			want:          emptyDiffModeFallbackBootstrap,
		},
		{
			name:          "present collection with unknown rows completes noop",
			evidence:      emptyDiffEvidence(collectionPresencePresent, 0, false),
			seedFileCount: 4,
			want:          emptyDiffModeCompleteNoop,
		},
		{
			name:          "missing collection bootstraps",
			evidence:      emptyDiffEvidence(collectionPresenceMissing, 0, false),
			seedFileCount: 0,
			want:          emptyDiffModeFallbackBootstrap,
		},
		{
			name:          "unknown collection completes noop",
			evidence:      emptyDiffEvidence(collectionPresenceUnknown, 0, false),
			seedFileCount: 4,
			want:          emptyDiffModeCompleteNoop,
		},
		{
			name:          "present empty collection with empty seed completes noop",
			evidence:      emptyDiffEvidence(collectionPresencePresent, 0, true),
			seedFileCount: 0,
			want:          emptyDiffModeCompleteNoop,
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			got := decideEmptyDiffMode(testCase.evidence, testCase.seedFileCount)
			if got != testCase.want {
				t.Fatalf("decideEmptyDiffMode() = %v, want %v", got, testCase.want)
			}
		})
	}
}

func TestPlanSyncDiffEmptyCollectionWithSeedFallsBack(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	manager, _, repoPath := newTestManager(t)
	config := manager.enrichIndexConfig(defaultIndexConfig())
	config.IgnoreDigest = digestIndexConfig(config)
	codebaseID := "cb-empty-collection"
	storedCollection := "stored_collection_empty"
	manager.semantic = &fakeSemantic{
		inspectCollection: func(_ context.Context, collectionName string) (semantic.CollectionFacts, error) {
			if collectionName != storedCollection {
				t.Fatalf("InspectCollection collection = %q, want %q", collectionName, storedCollection)
			}
			return semantic.CollectionFacts{Exists: true, Rows: 0, RowsKnown: true}, nil
		},
	}
	manager.mu.Lock()
	manager.codebases[codebaseID] = model.Codebase{
		ID:              codebaseID,
		CanonicalPath:   repoPath,
		Status:          model.CodebaseStatusIndexed,
		EffectiveConfig: config,
		CollectionName:  storedCollection,
	}
	manager.mu.Unlock()

	seed, err := merkle.Capture(ctx, manager.indexability, codebaseID, repoPath, config)
	if err != nil {
		t.Fatalf("Capture returned error: %v", err)
	}
	seed.ConfigDigest = config.IgnoreDigest
	if err := merkle.WriteSnapshot(manager.merklePath(codebaseID), seed); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	job := model.Job{
		ID:            "job-empty-collection",
		CodebaseID:    codebaseID,
		CanonicalPath: repoPath,
		Config:        config,
	}
	source := newCodeItemSource(manager.runner, manager.indexability, codebaseID, repoPath, config)
	plan := manager.planSyncDiff(ctx, job, codebaseID, source)
	if !plan.fallback {
		t.Fatalf("plan.fallback = false, want true")
	}
	if plan.handled {
		t.Fatalf("plan.handled = true, want false")
	}
}

func TestShouldQueueMissingCollectionRepair(t *testing.T) {
	t.Parallel()

	if !shouldQueueMissingCollectionRepair(model.Codebase{Status: model.CodebaseStatusIndexed}, false, collectionPresenceMissing) {
		t.Fatal("indexed + missing should queue repair")
	}
	if !shouldQueueMissingCollectionRepair(model.Codebase{Status: model.CodebaseStatusStale}, false, collectionPresenceMissing) {
		t.Fatal("stale + missing should queue repair")
	}
	if shouldQueueMissingCollectionRepair(model.Codebase{Status: model.CodebaseStatusIndexed}, false, collectionPresenceUnknown) {
		t.Fatal("unknown must not queue repair")
	}
	if shouldQueueMissingCollectionRepair(model.Codebase{Status: model.CodebaseStatusIndexed}, true, collectionPresenceMissing) {
		t.Fatal("active job must suppress repair queueing")
	}
}

func TestShouldDeferWatcherConvergeForFirstBuild(t *testing.T) {
	t.Parallel()

	firstBuild := model.Codebase{
		Kind:              model.CodebaseKindCode,
		Status:            model.CodebaseStatusIndexing,
		LastSuccessfulRun: nil,
	}
	if !shouldDeferWatcherConvergeForFirstBuild(firstBuild) {
		t.Fatal("code first build in pre-indexed status should defer watcher converge")
	}

	discovered := firstBuild
	discovered.Status = model.CodebaseStatusDiscovered
	if !shouldDeferWatcherConvergeForFirstBuild(discovered) {
		t.Fatal("discovered worktree has no collection yet and should defer watcher converge")
	}

	document := firstBuild
	document.Kind = model.CodebaseKindDocument
	if shouldDeferWatcherConvergeForFirstBuild(document) {
		t.Fatal("document codebase should not defer watcher converge")
	}

	indexed := firstBuild
	indexed.Status = model.CodebaseStatusIndexed
	if shouldDeferWatcherConvergeForFirstBuild(indexed) {
		t.Fatal("indexed codebase should not defer watcher converge")
	}

	rebuilt := firstBuild
	rebuilt.LastSuccessfulRun = &model.IndexRunSummary{}
	if shouldDeferWatcherConvergeForFirstBuild(rebuilt) {
		t.Fatal("codebase with prior successful run should not defer watcher converge")
	}
}

func TestShouldSkipForActiveFirstBuildStagingRequiresActiveJob(t *testing.T) {
	t.Parallel()

	firstBuild := model.Codebase{
		Kind:              model.CodebaseKindCode,
		Status:            model.CodebaseStatusIndexing,
		LastSuccessfulRun: nil,
	}
	if !shouldSkipForActiveFirstBuildStaging(firstBuild, true) {
		t.Fatal("active code first build should skip work that assumes a live collection")
	}
	if shouldSkipForActiveFirstBuildStaging(firstBuild, false) {
		t.Fatal("interrupted first build without active job should not skip live-collection work")
	}
}

func TestDecideSearchCollectionMode(t *testing.T) {
	t.Parallel()

	codebase := model.Codebase{Status: model.CodebaseStatusStale}
	if got := decideSearchCollectionMode(codebase, nil, collectionPresenceMissing); got != searchCollectionModeAutomaticRepair {
		t.Fatalf("stale + missing search mode = %v, want %v", got, searchCollectionModeAutomaticRepair)
	}

	codebase = model.Codebase{Status: model.CodebaseStatusIndexed}
	activeJob := &model.Job{}
	if got := decideSearchCollectionMode(codebase, activeJob, collectionPresenceMissing); got != searchCollectionModeAutomaticRepair {
		t.Fatalf("active job + missing search mode = %v, want %v", got, searchCollectionModeAutomaticRepair)
	}

	if got := decideSearchCollectionMode(codebase, nil, collectionPresenceMissing); got != searchCollectionModeMissing {
		t.Fatalf("indexed + missing search mode = %v, want %v", got, searchCollectionModeMissing)
	}

	if got := decideSearchCollectionMode(codebase, nil, collectionPresenceUnknown); got != searchCollectionModeProceed {
		t.Fatalf("unknown search mode = %v, want %v", got, searchCollectionModeProceed)
	}
}

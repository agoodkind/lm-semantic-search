package daemon

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
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

func TestDecideEmptyDiffMode(t *testing.T) {
	t.Parallel()

	if got := decideEmptyDiffMode(collectionPresencePresent); got != emptyDiffModeCompleteNoop {
		t.Fatalf("present empty diff = %v, want %v", got, emptyDiffModeCompleteNoop)
	}
	if got := decideEmptyDiffMode(collectionPresenceUnknown); got != emptyDiffModeCompleteNoop {
		t.Fatalf("unknown empty diff = %v, want %v", got, emptyDiffModeCompleteNoop)
	}
	if got := decideEmptyDiffMode(collectionPresenceMissing); got != emptyDiffModeFallbackBootstrap {
		t.Fatalf("missing empty diff = %v, want %v", got, emptyDiffModeFallbackBootstrap)
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

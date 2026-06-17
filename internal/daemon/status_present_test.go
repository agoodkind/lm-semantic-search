package daemon

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/model"
)

func TestBuildJobSuccessorsLinksEachTerminalJobToTheNext(t *testing.T) {
	t.Parallel()
	t0 := renderTestTime
	mk := func(id, cb string, state model.JobState, completedOffset time.Duration) model.Job {
		completed := t0.Add(completedOffset)
		return model.Job{ID: id, CodebaseID: cb, State: state, StartedAt: t0, CompletedAt: &completed}
	}
	running := model.Job{ID: "live", CodebaseID: "A", State: model.JobStateRunning, StartedAt: t0}
	jobs := []model.Job{
		mk("a1", "A", model.JobStateFailed, 1*time.Minute),
		mk("a2", "A", model.JobStateFailed, 2*time.Minute),
		mk("a3", "A", model.JobStateCompleted, 3*time.Minute),
		running, // active jobs are not part of the terminal chain
		mk("b1", "B", model.JobStateFailed, 1*time.Minute),
	}
	got := buildJobSuccessors(jobs)
	if got["a1"] != "a2" {
		t.Fatalf("a1 successor = %q, want a2", got["a1"])
	}
	if got["a2"] != "a3" {
		t.Fatalf("a2 successor = %q, want a3", got["a2"])
	}
	if _, ok := got["a3"]; ok {
		t.Fatalf("a3 is the latest terminal job and must have no successor, got %q", got["a3"])
	}
	if _, ok := got["live"]; ok {
		t.Fatalf("an active job must not appear in the successor chain")
	}
	if _, ok := got["b1"]; ok {
		t.Fatalf("b1 is the only job for its codebase and must have no successor")
	}
}

func TestComputeDisplayStatusNeverNotIndexed(t *testing.T) {
	t.Parallel()
	indexedRun := &model.IndexRunSummary{IndexedFiles: 5}
	embeddingJob := &model.Job{Operation: "index", State: model.JobStateRunning, Progress: model.Progress{FilesTotal: 10}}
	queuedJob := &model.Job{Operation: "index", State: model.JobStateQueued, Progress: model.Progress{}}
	backgroundSyncJob := &model.Job{Operation: "sync", State: model.JobStateRunning, Progress: model.Progress{FilesInCodebase: 10, FilesModified: 1}}

	cases := []struct {
		name     string
		codebase model.Codebase
		job      *model.Job
		want     displayStatus
	}{
		{"embedding job", model.Codebase{Status: model.CodebaseStatusIndexing}, embeddingJob, displayIndexing},
		{"queued job, scope unknown", model.Codebase{Status: model.CodebaseStatusIndexing}, queuedJob, displayPreparing},
		{"background sync over indexed", model.Codebase{Status: model.CodebaseStatusIndexed, LastSuccessfulRun: indexedRun}, backgroundSyncJob, displayIndexed},
		{"no job, quarantined", model.Codebase{Status: model.CodebaseStatusQuarantined}, nil, displayQuarantined},
		{"no job, indexed", model.Codebase{Status: model.CodebaseStatusIndexed}, nil, displayIndexed},
		{"no job, stale", model.Codebase{Status: model.CodebaseStatusStale}, nil, displayStale},
		{"no job, failed", model.Codebase{Status: model.CodebaseStatusFailed}, nil, displayFailed},
		{"no job, missing", model.Codebase{Status: model.CodebaseStatusMissing}, nil, displayMissing},
		{"interrupted: indexing, no job", model.Codebase{Status: model.CodebaseStatusIndexing}, nil, displayPreparing},
		{"interrupted: not_indexed, no job", model.Codebase{Status: model.CodebaseStatusNotIndexed}, nil, displayPreparing},
	}
	for _, testCase := range cases {
		got := computeDisplayStatus(testCase.codebase, testCase.job, false)
		if got != testCase.want {
			t.Errorf("%s: computeDisplayStatus = %q, want %q", testCase.name, got, testCase.want)
		}
		if string(got) == string(model.CodebaseStatusNotIndexed) {
			t.Errorf("%s: a tracked codebase must never present as not_indexed", testCase.name)
		}
	}
}

// TestComputeDisplayStatusWaitingFold proves that during a pipeline outage any
// codebase that cannot be searched right now folds to "waiting": an interrupted
// build, a not-indexed build, an already-indexed codebase, and a background sync
// over an indexed codebase all read "waiting" because a query embed would fail.
// A codebase with a live scoped job keeps reading "indexing" (it is embedding
// right now), and a local terminal state (stale, missing) is never rewritten by
// pipeline health.
func TestComputeDisplayStatusWaitingFold(t *testing.T) {
	t.Parallel()
	indexedRun := &model.IndexRunSummary{IndexedFiles: 5}
	embeddingJob := &model.Job{Operation: "index", State: model.JobStateRunning, Progress: model.Progress{FilesTotal: 10}}
	backgroundSyncJob := &model.Job{Operation: "sync", State: model.JobStateRunning, Progress: model.Progress{FilesInCodebase: 10, FilesModified: 1}}

	cases := []struct {
		name     string
		codebase model.Codebase
		job      *model.Job
		want     displayStatus
	}{
		{"interrupted first index folds to waiting", model.Codebase{Status: model.CodebaseStatusIndexing}, nil, displayWaiting},
		{"not_indexed folds to waiting", model.Codebase{Status: model.CodebaseStatusNotIndexed}, nil, displayWaiting},
		{"live scoped job stays indexing", model.Codebase{Status: model.CodebaseStatusIndexing}, embeddingJob, displayIndexing},
		{"already indexed folds to waiting", model.Codebase{Status: model.CodebaseStatusIndexed}, nil, displayWaiting},
		{"background sync over indexed folds to waiting", model.Codebase{Status: model.CodebaseStatusIndexed, LastSuccessfulRun: indexedRun}, backgroundSyncJob, displayWaiting},
		{"quarantined stays quarantined", model.Codebase{Status: model.CodebaseStatusQuarantined}, nil, displayQuarantined},
		{"stale stays stale", model.Codebase{Status: model.CodebaseStatusStale}, nil, displayStale},
		{"missing stays missing", model.Codebase{Status: model.CodebaseStatusMissing}, nil, displayMissing},
	}
	for _, testCase := range cases {
		got := computeDisplayStatus(testCase.codebase, testCase.job, true)
		if got != testCase.want {
			t.Errorf("%s: computeDisplayStatus(degraded) = %q, want %q", testCase.name, got, testCase.want)
		}
	}
}

func TestShouldResumeInterruptedBuild(t *testing.T) {
	t.Parallel()
	cases := []struct {
		status       model.CodebaseStatus
		hasActiveJob bool
		want         bool
	}{
		{model.CodebaseStatusIndexing, false, true},
		{model.CodebaseStatusNotIndexed, false, true},
		{model.CodebaseStatusIndexing, true, false},
		{model.CodebaseStatusQuarantined, false, false},
		{model.CodebaseStatusIndexed, false, false},
		{model.CodebaseStatusStale, false, false},
		{model.CodebaseStatusFailed, false, false},
	}
	for _, testCase := range cases {
		got := shouldResumeInterruptedBuild(model.Codebase{Status: testCase.status}, testCase.hasActiveJob)
		if got != testCase.want {
			t.Errorf("shouldResumeInterruptedBuild(%s, active=%v) = %v, want %v", testCase.status, testCase.hasActiveJob, got, testCase.want)
		}
	}
}

// TestPlanRepairsResumesInterruptedBuild proves the background pass re-queues a
// build left at "indexing" with no live job, so an interrupted build auto-retries
// rather than sitting stuck.
func TestPlanRepairsResumesInterruptedBuild(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexing
	codebase.ActiveJobID = ""
	codebase.CollectionName = "interrupted_collection"
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	manager.semantic = &fakeSemantic{
		collectionName:  func(string) string { return "interrupted_collection" },
		listCollections: func(context.Context) ([]string, error) { return []string{}, nil },
	}

	plans, _, err := manager.planMissingCollectionRepairs(context.Background())
	if err != nil {
		t.Fatalf("planMissingCollectionRepairs returned error: %v", err)
	}
	found := false
	for _, plan := range plans {
		if plan.codebaseID == codebase.ID {
			found = true
		}
	}
	if !found {
		t.Fatalf("interrupted build was not queued for retry; plans=%+v", plans)
	}
}

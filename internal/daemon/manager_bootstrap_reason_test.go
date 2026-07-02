package daemon

import (
	"context"
	"testing"

	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func TestBootstrapReasonVocabulary(t *testing.T) {
	t.Parallel()

	reasons := []bootstrapReason{
		bootstrapReasonFirstIndex,
		bootstrapReasonForcedReindex,
		bootstrapReasonStagingResume,
		bootstrapReasonEmptyDiffCollectionMissing,
		bootstrapReasonEmptyDiffCollectionEmpty,
		bootstrapReasonDeltaCollectionMissing,
		bootstrapReasonDeltaCodebaseMissing,
	}
	want := []string{
		"first_index",
		"forced_reindex",
		"staging_resume",
		"empty_diff_collection_missing",
		"empty_diff_collection_empty",
		"delta_collection_missing",
		"delta_codebase_missing",
	}

	for index, reason := range reasons {
		if string(reason) != want[index] {
			t.Fatalf("reason[%d] = %q, want %q", index, reason, want[index])
		}
	}
}

func TestPlanSyncDiffEmptyDiffMissingCollectionStampsBootstrapReason(t *testing.T) {
	t.Parallel()

	manager, job, plan := planEmptyDiffWithCollectionFacts(t, semantic.CollectionFacts{
		Exists:    false,
		Rows:      0,
		RowsKnown: false,
	})
	if !plan.fallback {
		t.Fatal("plan.fallback = false, want true")
	}
	got, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("GetJob(%s) not found", job.ID)
	}
	if got.Progress.BootstrapReason != string(bootstrapReasonEmptyDiffCollectionMissing) {
		t.Fatalf(
			"BootstrapReason = %q, want %q",
			got.Progress.BootstrapReason,
			bootstrapReasonEmptyDiffCollectionMissing,
		)
	}
}

func TestPlanSyncDiffEmptyRowsCollectionStampsBootstrapReason(t *testing.T) {
	t.Parallel()

	manager, job, plan := planEmptyDiffWithCollectionFacts(t, semantic.CollectionFacts{
		Exists:    true,
		Rows:      0,
		RowsKnown: true,
	})
	if !plan.fallback {
		t.Fatal("plan.fallback = false, want true")
	}
	got, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("GetJob(%s) not found", job.ID)
	}
	if got.Progress.BootstrapReason != string(bootstrapReasonEmptyDiffCollectionEmpty) {
		t.Fatalf(
			"BootstrapReason = %q, want %q",
			got.Progress.BootstrapReason,
			bootstrapReasonEmptyDiffCollectionEmpty,
		)
	}
}

func TestClassifyReindexErrCollectionMissingStampsBootstrapReason(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	job := model.Job{ID: "job-delta-missing", CodebaseID: "cb-delta-missing"}
	manager.mu.Lock()
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	outcome := manager.classifyReindexErr(context.Background(), job, semantic.ErrCollectionMissing, "per-file reindex")

	if !outcome.fallback {
		t.Fatal("outcome.fallback = false, want true")
	}
	got, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("GetJob(%s) not found", job.ID)
	}
	if got.Progress.BootstrapReason != string(bootstrapReasonDeltaCollectionMissing) {
		t.Fatalf(
			"BootstrapReason = %q, want %q",
			got.Progress.BootstrapReason,
			bootstrapReasonDeltaCollectionMissing,
		)
	}
}

func TestRunDeltaSyncMissingCodebaseStampsBootstrapReason(t *testing.T) {
	t.Parallel()

	manager, _, _ := newTestManager(t)
	job := model.Job{ID: "job-delta-codebase-missing", CodebaseID: "cb-delta-codebase-missing"}
	manager.mu.Lock()
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	handled := manager.runDeltaSync(context.Background(), job, nil)

	if handled {
		t.Fatal("runDeltaSync returned true, want false")
	}
	got, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("GetJob(%s) not found", job.ID)
	}
	const wantReason = string(bootstrapReasonDeltaCodebaseMissing)
	if got.Progress.BootstrapReason != wantReason {
		t.Fatalf("BootstrapReason = %q, want %q", got.Progress.BootstrapReason, wantReason)
	}
}

func TestStartIndexFreshBootstrapStampsFirstIndexReason(t *testing.T) {
	t.Parallel()

	manager, _, repoPath := newTestManager(t)
	manager.semantic = &fakeSemantic{unavailable: true}

	job, _, _, _, err := manager.StartIndex(context.Background(), repoPath, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget)
	if err != nil {
		t.Fatalf("StartIndex returned error: %v", err)
	}
	waitForCodebaseStatus(t, manager, repoPath, model.CodebaseStatusIndexed)

	got, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("GetJob(%s) not found", job.ID)
	}
	if got.Progress.BootstrapReason != string(bootstrapReasonFirstIndex) {
		t.Fatalf("BootstrapReason = %q, want %q", got.Progress.BootstrapReason, bootstrapReasonFirstIndex)
	}
}

func TestStartIndexForcedBootstrapStampsForcedReindexReason(t *testing.T) {
	t.Parallel()

	const collectionName = "forced_bootstrap_collection"

	manager, _, repoPath := newTestManager(t)
	manager.semantic = &fakeSemantic{
		collectionName: func(string) string { return collectionName },
		inspectCollection: func(_ context.Context, gotCollection string) (semantic.CollectionFacts, error) {
			if gotCollection != collectionName {
				t.Fatalf("InspectCollection collection = %q, want %q", gotCollection, collectionName)
			}
			return semantic.CollectionFacts{Exists: false, Rows: 0, RowsKnown: false}, nil
		},
	}
	config := manager.enrichIndexConfig(defaultIndexConfig())
	config.IgnoreDigest = digestIndexConfig(config)
	manager.mu.Lock()
	manager.codebases["cb-forced-bootstrap"] = model.Codebase{
		ID:                "cb-forced-bootstrap",
		Kind:              model.CodebaseKindCode,
		CanonicalPath:     repoPath,
		Status:            model.CodebaseStatusIndexed,
		LastSuccessfulRun: &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 1},
		EffectiveConfig:   config,
		CollectionName:    collectionName,
	}
	manager.mu.Unlock()

	job, _, _, _, err := manager.StartIndex(context.Background(), repoPath, testClientInfo(), defaultIndexConfig(), true, emptyAdmissionBudget)
	if err != nil {
		t.Fatalf("StartIndex(force=true) returned error: %v", err)
	}
	waitForCodebaseStatus(t, manager, repoPath, model.CodebaseStatusIndexed)

	got, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("GetJob(%s) not found", job.ID)
	}
	if got.Progress.BootstrapReason != string(bootstrapReasonForcedReindex) {
		t.Fatalf("BootstrapReason = %q, want %q", got.Progress.BootstrapReason, bootstrapReasonForcedReindex)
	}
}

func planEmptyDiffWithCollectionFacts(t *testing.T, facts semantic.CollectionFacts) (*Manager, model.Job, deltaPlan) {
	t.Helper()

	const codebaseID = "cb-empty-diff-bootstrap"
	const collectionName = "empty_diff_bootstrap_collection"

	ctx := context.Background()
	manager, _, repoPath := newTestManager(t)
	config := manager.enrichIndexConfig(defaultIndexConfig())
	config.IgnoreDigest = digestIndexConfig(config)
	manager.semantic = &fakeSemantic{
		inspectCollection: func(_ context.Context, gotCollection string) (semantic.CollectionFacts, error) {
			if gotCollection != collectionName {
				t.Fatalf("InspectCollection collection = %q, want %q", gotCollection, collectionName)
			}
			return facts, nil
		},
	}
	manager.mu.Lock()
	manager.codebases[codebaseID] = model.Codebase{
		ID:              codebaseID,
		Kind:            model.CodebaseKindCode,
		CanonicalPath:   repoPath,
		Status:          model.CodebaseStatusIndexed,
		EffectiveConfig: config,
		CollectionName:  collectionName,
	}
	job := model.Job{
		ID:            "job-empty-diff-bootstrap",
		CodebaseID:    codebaseID,
		CanonicalPath: repoPath,
		Config:        config,
	}
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	seed, err := merkle.Capture(ctx, manager.indexability, codebaseID, repoPath, config)
	if err != nil {
		t.Fatalf("Capture returned error: %v", err)
	}
	seed.ConfigDigest = config.IgnoreDigest
	if err := merkle.WriteSnapshot(manager.merklePath(codebaseID), seed); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	source := newCodeItemSource(manager.runner, manager.indexability, codebaseID, repoPath, config)
	return manager, job, manager.planSyncDiff(ctx, job, codebaseID, source)
}

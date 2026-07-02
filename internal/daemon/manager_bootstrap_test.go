package daemon

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// recordingRunner returns a fakeRunner whose IndexOne appends each embedded
// relative path to embedded and returns one chunk per file, so a test can
// assert exactly which files a from-scratch build re-embedded.
func recordingRunner(mu *sync.Mutex, embedded *[]string) fakeRunner {
	return fakeRunner{
		index:      nil,
		indexFiles: nil,
		indexOne: func(_ context.Context, _ string, relativePath string, _ model.IndexConfig) (indexer.OneFileResult, error) {
			mu.Lock()
			*embedded = append(*embedded, relativePath)
			mu.Unlock()
			content := "package main\n// " + relativePath + "\n"
			return indexer.OneFileResult{
				Chunks:   []model.StoredChunk{{Content: content, RelativePath: relativePath, StartLine: 1, EndLine: 1, Language: "go", FileExtension: ".go"}},
				FileHash: hashText(content),
				Skipped:  false,
				Removed:  false,
			}, nil
		},
	}
}

// newMultiFileRepo writes one valid Go file per name under a fresh repo
// directory and returns its canonicalized path.
func newMultiFileRepo(t *testing.T, names ...string) string {
	t.Helper()

	repoPath := filepath.Join(t.TempDir(), "repo")
	if err := os.MkdirAll(repoPath, 0o755); err != nil {
		t.Fatalf("MkdirAll returned error: %v", err)
	}
	for _, name := range names {
		if err := os.WriteFile(filepath.Join(repoPath, name), []byte("package main\n// "+name+"\n"), 0o644); err != nil {
			t.Fatalf("WriteFile returned error: %v", err)
		}
	}
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	return canonical
}

// seedBootstrapCodebase registers a from-scratch build target and returns the
// job runBootstrap consumes. The codebase starts in the indexing state so the
// run mirrors a resume after an interrupted first index.
func seedBootstrapCodebase(t *testing.T, manager *Manager, canonical string, cfg model.IndexConfig) (string, model.Job) {
	t.Helper()

	codebaseID := "cb-bootstrap-" + filepath.Base(filepath.Dir(canonical))
	manager.mu.Lock()
	manager.codebases[codebaseID] = model.Codebase{
		ID:              codebaseID,
		CanonicalPath:   canonical,
		Status:          model.CodebaseStatusIndexing,
		EffectiveConfig: cfg,
	}
	manager.mu.Unlock()

	job := newQueuedJob(codebaseID, canonical, canonical, testClientInfo(), string(jobOperationIndex), false, cfg, emptyAdmissionBudget, clock.Now())
	manager.mu.Lock()
	manager.jobs[job.ID] = job
	manager.mu.Unlock()
	return codebaseID, job
}

func setBootstrapCollectionName(t *testing.T, manager *Manager, codebaseID string, collectionName string) {
	t.Helper()

	manager.mu.Lock()
	defer manager.mu.Unlock()
	codebase := manager.codebases[codebaseID]
	codebase.CollectionName = collectionName
	manager.codebases[codebaseID] = codebase
}

func emitReuseProgress(progress func(semantic.Progress), chunks []model.StoredChunk, reuse map[string][]float32) {
	if progress == nil {
		return
	}
	chunkCount := safeInt32(len(chunks))
	var reused int32
	var embedded int32
	for _, chunk := range chunks {
		if _, present := reuse[hashText(chunk.Content)]; present {
			reused++
		} else {
			embedded++
		}
	}
	progress(semantic.Progress{ChunksProcessed: chunkCount, ChunksReused: reused, ChunksEmbedded: embedded})
}

func TestRunBootstrapReusesLiveCollectionVectors(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 2)
	liveCollection := "cc_bootstrap_live"
	fake := &fakeSemantic{
		inspectCollection: func(_ context.Context, collectionName string) (semantic.CollectionFacts, error) {
			if collectionName != liveCollection {
				return semantic.CollectionFacts{Exists: false, Rows: 0, RowsKnown: false}, nil
			}
			return semantic.CollectionFacts{Exists: true, Rows: 2, RowsKnown: true}, nil
		},
		hasStaging: func(context.Context, string) (bool, error) { return true, nil },
		loadReuseForPath: func(_ context.Context, collectionName string, relativePath string) (map[string][]float32, error) {
			if collectionName != liveCollection {
				return nil, nil
			}
			content := "package main\n// " + relativePath + "\n"
			return map[string][]float32{hashText(content): {1, 2, 3}}, nil
		},
		stageReindexWithReuse: func(_ context.Context, _ string, chunks []model.StoredChunk, _ []string, progress func(semantic.Progress), reuse map[string][]float32) error {
			emitReuseProgress(progress, chunks, reuse)
			return nil
		},
	}
	manager.semantic = fake

	var mu sync.Mutex
	embedded := make([]string, 0)
	manager.runner = recordingRunner(&mu, &embedded)

	canonical := newMultiFileRepo(t, "a.go", "b.go")
	cfg := defaultIndexConfig()
	cfg.IgnoreDigest = "sha256:bootstrap-live-reuse"
	codebaseID, job := seedBootstrapCodebase(t, manager, canonical, cfg)
	setBootstrapCollectionName(t, manager, codebaseID, liveCollection)
	source := newCodeItemSource(manager.runner, manager.indexability, job.CodebaseID, job.CanonicalPath, job.Config).withCollectionName(liveCollection)

	manager.runBootstrap(context.Background(), job, source)

	completed, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("job %s was not found", job.ID)
	}
	if completed.State != model.JobStateCompleted {
		t.Fatalf("job state = %q, want completed", completed.State)
	}
	if completed.Progress.ChunksEmbedded != 0 {
		t.Fatalf("ChunksEmbedded = %d, want 0 when live vectors cover every bootstrap chunk", completed.Progress.ChunksEmbedded)
	}
	if completed.Progress.ChunksReused <= 0 {
		t.Fatalf("ChunksReused = %d, want > 0 from live collection vectors", completed.Progress.ChunksReused)
	}

	pathCalls := fake.reusePathCallsSnapshot()
	if len(pathCalls) != 2 {
		t.Fatalf("reuse path loads = %d, want 2 live collection reads: %+v", len(pathCalls), pathCalls)
	}
	seenPaths := map[string]bool{}
	for _, call := range pathCalls {
		if call.Collection != liveCollection {
			t.Fatalf("reuse load collection = %q, want live collection %q", call.Collection, liveCollection)
		}
		seenPaths[call.Path] = true
	}
	if !seenPaths["a.go"] || !seenPaths["b.go"] {
		t.Fatalf("reuse load paths = %v, want a.go and b.go", seenPaths)
	}
	if got := fake.stageCallsSnapshot(); len(got) != 2 {
		t.Fatalf("StageReindex calls = %d, want 2 staging writes: %+v", len(got), got)
	}
	if promoted := fake.promotedSnapshot(); !slices.Equal(promoted, []string{canonical}) {
		t.Fatalf("PromoteStaging calls = %v, want [%s]", promoted, canonical)
	}
	if _, err := merkle.ReadSnapshot(manager.merklePath(codebaseID)); err != nil {
		t.Fatalf("ReadSnapshot returned error after promotion: %v", err)
	}
}

func TestRunBootstrapMissingLiveCollectionEmbedsEverything(t *testing.T) {
	t.Run("code", func(t *testing.T) {
		manager, _ := newTestManagerWithCap(t, 2)
		liveCollection := "cc_bootstrap_missing"
		fake := &fakeSemantic{
			inspectCollection: func(_ context.Context, collectionName string) (semantic.CollectionFacts, error) {
				if collectionName != liveCollection {
					t.Fatalf("InspectCollection(%q), want %q", collectionName, liveCollection)
				}
				return semantic.CollectionFacts{Exists: false, Rows: 0, RowsKnown: false}, nil
			},
			hasStaging: func(context.Context, string) (bool, error) { return true, nil },
			stageReindexWithReuse: func(_ context.Context, _ string, chunks []model.StoredChunk, _ []string, progress func(semantic.Progress), reuse map[string][]float32) error {
				emitReuseProgress(progress, chunks, reuse)
				return nil
			},
		}
		manager.semantic = fake

		var mu sync.Mutex
		embedded := make([]string, 0)
		manager.runner = recordingRunner(&mu, &embedded)

		canonical := newMultiFileRepo(t, "a.go", "b.go", "c.go")
		cfg := defaultIndexConfig()
		cfg.IgnoreDigest = "sha256:bootstrap-missing-live"
		codebaseID, job := seedBootstrapCodebase(t, manager, canonical, cfg)
		setBootstrapCollectionName(t, manager, codebaseID, liveCollection)
		source := newCodeItemSource(manager.runner, manager.indexability, job.CodebaseID, job.CanonicalPath, job.Config).withCollectionName(liveCollection)

		manager.runBootstrap(context.Background(), job, source)

		completed, found := manager.GetJob(job.ID)
		if !found {
			t.Fatalf("job %s was not found", job.ID)
		}
		if completed.State != model.JobStateCompleted {
			t.Fatalf("job state = %q, want completed", completed.State)
		}
		if calls := fake.reusePathCallsSnapshot(); len(calls) != 0 {
			t.Fatalf("reuse path loads = %d, want 0 when live collection is missing: %+v", len(calls), calls)
		}
		if completed.Progress.ChunksEmbedded != 3 {
			t.Fatalf("ChunksEmbedded = %d, want 3 when every file embeds into staging", completed.Progress.ChunksEmbedded)
		}
		if completed.Progress.ChunksReused != 0 {
			t.Fatalf("ChunksReused = %d, want 0 with no live reuse", completed.Progress.ChunksReused)
		}
		if got := fake.stageCallsSnapshot(); len(got) != 3 {
			t.Fatalf("StageReindex calls = %d, want 3 staging writes: %+v", len(got), got)
		}
		if promoted := fake.promotedSnapshot(); !slices.Equal(promoted, []string{canonical}) {
			t.Fatalf("PromoteStaging calls = %v, want [%s]", promoted, canonical)
		}
	})

	t.Run("conversation", func(t *testing.T) {
		manager, _, _ := newTestManager(t)
		liveCollection := "conv_chunks_first_ingest"
		fake := &fakeSemantic{
			conversationName: func(string) string { return liveCollection },
			inspectCollection: func(_ context.Context, collectionName string) (semantic.CollectionFacts, error) {
				if collectionName != liveCollection {
					t.Fatalf("InspectCollection(%q), want %q", collectionName, liveCollection)
				}
				return semantic.CollectionFacts{Exists: false, Rows: 0, RowsKnown: false}, nil
			},
			hasStaging: func(context.Context, string) (bool, error) { return true, nil },
			stageReindexWithReuse: func(_ context.Context, _ string, chunks []model.StoredChunk, _ []string, progress func(semantic.Progress), reuse map[string][]float32) error {
				emitReuseProgress(progress, chunks, reuse)
				return nil
			},
		}
		manager.semantic = fake

		codebase, err := manager.RegisterConversationCollection(context.Background(), "first-ingest")
		if err != nil {
			t.Fatalf("RegisterConversationCollection returned error: %v", err)
		}
		manifest := map[string]string{"conv-a": "fp-a", "conv-b": "fp-b"}
		documents := []model.ConversationDocument{
			{ConversationID: "conv-a", MessageIndex: 0, Role: "user", TimestampUnix: 1712346100, Text: "alpha"},
			{ConversationID: "conv-b", MessageIndex: 0, Role: "assistant", TimestampUnix: 1712346101, Text: "beta"},
		}
		job := stageConversationJob(t, manager, codebase, conversationJobPayload{
			Kind:           conversationJobKindUpsert,
			CollectionName: liveCollection,
			Manifest:       manifest,
			Documents:      documents,
		})
		source := newConversationItemSource(liveCollection, manifest, documents, nil)

		manager.runBootstrap(context.Background(), job, source)

		completed, found := manager.GetJob(job.ID)
		if !found {
			t.Fatalf("job %s was not found", job.ID)
		}
		if completed.State != model.JobStateCompleted {
			t.Fatalf("job state = %q, want completed", completed.State)
		}
		if calls := fake.reusePrefixCallsSnapshot(); len(calls) != 0 {
			t.Fatalf("reuse prefix loads = %d, want 0 for first conversation ingest: %+v", len(calls), calls)
		}
		if completed.Progress.ChunksEmbedded != 2 {
			t.Fatalf("ChunksEmbedded = %d, want 2 first-ingest conversation chunks", completed.Progress.ChunksEmbedded)
		}
		if completed.Progress.ChunksReused != 0 {
			t.Fatalf("ChunksReused = %d, want 0 for first conversation ingest", completed.Progress.ChunksReused)
		}
	})
}

func TestRunBootstrapForcedSkipsLiveItemReuse(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 2)
	liveCollection := "cc_bootstrap_forced"
	fake := &fakeSemantic{
		inspectCollection: func(_ context.Context, collectionName string) (semantic.CollectionFacts, error) {
			if collectionName != liveCollection {
				return semantic.CollectionFacts{Exists: false, Rows: 0, RowsKnown: false}, nil
			}
			return semantic.CollectionFacts{Exists: true, Rows: 1, RowsKnown: true}, nil
		},
		hasStaging: func(context.Context, string) (bool, error) { return true, nil },
		stageReindexWithReuse: func(_ context.Context, _ string, chunks []model.StoredChunk, _ []string, progress func(semantic.Progress), reuse map[string][]float32) error {
			emitReuseProgress(progress, chunks, reuse)
			return nil
		},
	}
	manager.semantic = fake

	var mu sync.Mutex
	embedded := make([]string, 0)
	manager.runner = recordingRunner(&mu, &embedded)

	canonical := newMultiFileRepo(t, "main.go")
	cfg := defaultIndexConfig()
	cfg.IgnoreDigest = "sha256:bootstrap-forced-skip-reuse"
	codebaseID, job := seedBootstrapCodebase(t, manager, canonical, cfg)
	job.Forced = true
	manager.mu.Lock()
	manager.jobs[job.ID] = job
	manager.mu.Unlock()
	setBootstrapCollectionName(t, manager, codebaseID, liveCollection)
	source := newCodeItemSource(manager.runner, manager.indexability, job.CodebaseID, job.CanonicalPath, job.Config).withCollectionName(liveCollection)

	manager.runBootstrap(context.Background(), job, source)

	completed, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("job %s was not found", job.ID)
	}
	if completed.State != model.JobStateCompleted {
		t.Fatalf("job state = %q, want completed", completed.State)
	}
	if calls := fake.reusePathCallsSnapshot(); len(calls) != 0 {
		t.Fatalf("reuse path loads = %d, want 0 for forced bootstrap: %+v", len(calls), calls)
	}
	if completed.Progress.ChunksEmbedded != 1 {
		t.Fatalf("ChunksEmbedded = %d, want 1 forced embed", completed.Progress.ChunksEmbedded)
	}
	if completed.Progress.ChunksReused != 0 {
		t.Fatalf("ChunksReused = %d, want 0 for forced bootstrap", completed.Progress.ChunksReused)
	}
}

func TestRunBootstrapResumesSkippingEmbeddedFiles(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 2)
	var mu sync.Mutex
	embedded := make([]string, 0)
	manager.runner = recordingRunner(&mu, &embedded)

	canonical := newMultiFileRepo(t, "a.go", "b.go", "c.go")
	cfg := defaultIndexConfig()
	cfg.IgnoreDigest = "sha256:bootstrap-resume"

	captured, err := merkle.Capture(context.Background(), manager.indexability, "cb", canonical, cfg)
	if err != nil {
		t.Fatalf("Capture returned error: %v", err)
	}

	codebaseID, job := seedBootstrapCodebase(t, manager, canonical, cfg)

	// A checkpoint recording only a.go as embedded; the resume must skip it
	// and embed only b.go and c.go.
	checkpoint := merkle.Snapshot{
		ConfigDigest: cfg.IgnoreDigest,
		Files:        map[string]string{"a.go": captured.Files["a.go"]},
		Inodes:       nil,
	}
	if err := merkle.WriteSnapshot(manager.stagingMerklePath(codebaseID), checkpoint); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	manager.runBootstrap(context.Background(), job, newCodeItemSource(manager.runner, manager.indexability, job.CodebaseID, job.CanonicalPath, job.Config))

	mu.Lock()
	slices.Sort(embedded)
	got := slices.Clone(embedded)
	mu.Unlock()
	if want := []string{"b.go", "c.go"}; !slices.Equal(got, want) {
		t.Fatalf("embedded files = %v, want %v (a.go must be skipped via checkpoint)", got, want)
	}

	finalSnapshot, err := merkle.ReadSnapshot(manager.merklePath(codebaseID))
	if err != nil {
		t.Fatalf("ReadSnapshot returned error: %v", err)
	}
	if len(finalSnapshot.Files) != 3 {
		t.Fatalf("final snapshot files = %d, want 3 (a.go, b.go, c.go)", len(finalSnapshot.Files))
	}
	waitForCodebaseStatus(t, manager, canonical, model.CodebaseStatusIndexed)
}

func TestRunBootstrapEmbedsEveryFileWithoutCheckpoint(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 2)
	var mu sync.Mutex
	embedded := make([]string, 0)
	manager.runner = recordingRunner(&mu, &embedded)

	canonical := newMultiFileRepo(t, "a.go", "b.go", "c.go")
	cfg := defaultIndexConfig()
	cfg.IgnoreDigest = "sha256:bootstrap-fresh"

	_, job := seedBootstrapCodebase(t, manager, canonical, cfg)

	manager.runBootstrap(context.Background(), job, newCodeItemSource(manager.runner, manager.indexability, job.CodebaseID, job.CanonicalPath, job.Config))

	mu.Lock()
	slices.Sort(embedded)
	got := slices.Clone(embedded)
	mu.Unlock()
	if want := []string{"a.go", "b.go", "c.go"}; !slices.Equal(got, want) {
		t.Fatalf("embedded files = %v, want %v (a full build embeds every file)", got, want)
	}
	waitForCodebaseStatus(t, manager, canonical, model.CodebaseStatusIndexed)
}

func TestRunBootstrapPromotesCollectionBeforeCommittingMerkle(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 2)
	fake := &fakeSemantic{
		hasStaging: func(context.Context, string) (bool, error) { return true, nil },
	}
	manager.semantic = fake

	var mu sync.Mutex
	embedded := make([]string, 0)
	manager.runner = recordingRunner(&mu, &embedded)

	canonical := newMultiFileRepo(t, "main.go")
	cfg := defaultIndexConfig()
	cfg.IgnoreDigest = "sha256:bootstrap-promote-order"
	codebaseID, job := seedBootstrapCodebase(t, manager, canonical, cfg)

	liveMerklePath := manager.merklePath(codebaseID)
	if err := os.MkdirAll(liveMerklePath, 0o755); err != nil {
		t.Fatalf("MkdirAll(%s) returned error: %v", liveMerklePath, err)
	}

	manager.runBootstrap(context.Background(), job, newCodeItemSource(manager.runner, manager.indexability, job.CodebaseID, job.CanonicalPath, job.Config))

	completed, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("job %s was not found", job.ID)
	}
	if completed.State != model.JobStateFailed {
		t.Fatalf("job state = %q, want failed", completed.State)
	}
	if len(fake.stageCallsSnapshot()) == 0 {
		t.Fatal("StageReindex was not called before the Merkle rename failure")
	}
	if promoted := fake.promotedSnapshot(); !slices.Equal(promoted, []string{canonical}) {
		t.Fatalf("PromoteStaging calls = %v, want [%s] before Merkle rename", promoted, canonical)
	}
	if dropped := fake.droppedStagingSnapshot(); len(dropped) == 0 {
		t.Fatal("DropStaging was not called after Merkle rename failure")
	}
	if _, err := os.Stat(manager.stagingMerklePath(codebaseID)); !os.IsNotExist(err) {
		t.Fatalf("staging Merkle path still exists after failed promotion: %v", err)
	}
}

func TestRunBootstrapSemanticPromotionFailureLeavesLiveMerkleUnchanged(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 2)
	fake := &fakeSemantic{
		hasStaging: func(context.Context, string) (bool, error) { return true, nil },
	}
	manager.semantic = &failingPromoteSemantic{
		fakeSemantic: fake,
		err:          errors.New("semantic promotion failed"),
	}

	var mu sync.Mutex
	embedded := make([]string, 0)
	manager.runner = recordingRunner(&mu, &embedded)

	canonical := newMultiFileRepo(t, "main.go")
	cfg := defaultIndexConfig()
	cfg.IgnoreDigest = "sha256:bootstrap-semantic-promote-failure"
	codebaseID, job := seedBootstrapCodebase(t, manager, canonical, cfg)

	liveSnapshot := merkle.Snapshot{
		ConfigDigest: cfg.IgnoreDigest,
		Files:        map[string]string{"main.go": hashText("old live content")},
		Inodes:       nil,
	}
	if err := merkle.WriteSnapshot(manager.merklePath(codebaseID), liveSnapshot); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	manager.runBootstrap(context.Background(), job, newCodeItemSource(manager.runner, manager.indexability, job.CodebaseID, job.CanonicalPath, job.Config))

	completed, found := manager.GetJob(job.ID)
	if !found {
		t.Fatalf("job %s was not found", job.ID)
	}
	if completed.State != model.JobStateFailed {
		t.Fatalf("job state = %q, want failed", completed.State)
	}
	after, err := merkle.ReadSnapshot(manager.merklePath(codebaseID))
	if err != nil {
		t.Fatalf("ReadSnapshot returned error: %v", err)
	}
	if !merkle.Equal(after, liveSnapshot) {
		t.Fatalf("live snapshot changed after semantic promotion failure: got %+v want %+v", after, liveSnapshot)
	}
	if dropped := fake.droppedStagingSnapshot(); len(dropped) == 0 || dropped[len(dropped)-1] != canonical {
		t.Fatalf("DropStaging calls = %v, want final drop for %s", dropped, canonical)
	}
	if _, err := os.Stat(manager.stagingMerklePath(codebaseID)); !os.IsNotExist(err) {
		t.Fatalf("staging Merkle path still exists after semantic promotion failure: %v", err)
	}
}

type failingPromoteSemantic struct {
	*fakeSemantic
	err error
}

func (semantic *failingPromoteSemantic) PromoteStaging(_ context.Context, path string) error {
	semantic.fakeSemantic.mu.Lock()
	semantic.fakeSemantic.promoted = append(semantic.fakeSemantic.promoted, path)
	semantic.fakeSemantic.mu.Unlock()
	return semantic.err
}

func TestResumeOrphanedJobsParksNoCheckpointInterruptedForRetry(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 2)
	manager.config.ResumeIndexingOnBoot = true

	canonical := newMultiFileRepo(t, "a.go")
	codebaseID := "cb-no-checkpoint"
	manager.mu.Lock()
	manager.codebases[codebaseID] = model.Codebase{
		ID:              codebaseID,
		CanonicalPath:   canonical,
		Status:          model.CodebaseStatusIndexing,
		ActiveJobID:     "stale-orphan-job",
		EffectiveConfig: defaultIndexConfig(),
	}
	manager.mu.Unlock()

	manager.ResumeOrphanedJobs(context.Background())

	if jobs := manager.ListJobs(""); len(jobs) != 0 {
		t.Fatalf("no-checkpoint resume launched %d jobs, want 0 (the background pass re-queues it)", len(jobs))
	}

	manager.mu.Lock()
	codebase := manager.codebases[codebaseID]
	manager.mu.Unlock()
	// An interrupted build with no checkpoint is parked re-queueable, not failed:
	// it stays indexing with the active job cleared so the background pass starts
	// a fresh build. Only clearing the index stops the retry.
	if codebase.Status != model.CodebaseStatusIndexing {
		t.Fatalf("status = %q, want Indexing so the background pass re-queues it", codebase.Status)
	}
	if codebase.ActiveJobID != "" {
		t.Fatalf("ActiveJobID = %q, want cleared so the pass sees no live job", codebase.ActiveJobID)
	}
	if codebase.LastFailedRun != nil {
		t.Fatalf("LastFailedRun = %+v, want nil; an interrupted build is not a failure", codebase.LastFailedRun)
	}
}

func TestResumeOrphanedJobsResumesFromStagingCheckpoint(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 2)
	manager.config.ResumeIndexingOnBoot = true
	manager.semantic = &fakeSemantic{
		inspectCollection: func(context.Context, string) (semantic.CollectionFacts, error) {
			return semantic.CollectionFacts{Exists: false, Rows: 0, RowsKnown: false}, nil
		},
		hasStaging: func(context.Context, string) (bool, error) { return true, nil },
	}
	var mu sync.Mutex
	embedded := make([]string, 0)
	manager.runner = recordingRunner(&mu, &embedded)

	canonical := newMultiFileRepo(t, "a.go", "b.go", "c.go")
	cfg := manager.enrichIndexConfig(defaultIndexConfig())
	cfg.IgnoreDigest = digestIndexConfig(cfg)
	captured, err := merkle.Capture(context.Background(), manager.indexability, "cb-staging-resume", canonical, cfg)
	if err != nil {
		t.Fatalf("Capture returned error: %v", err)
	}

	codebaseID := "cb-staging-resume"
	manager.mu.Lock()
	manager.codebases[codebaseID] = model.Codebase{
		ID:              codebaseID,
		CanonicalPath:   canonical,
		Status:          model.CodebaseStatusIndexing,
		ActiveJobID:     "stale-orphan-job",
		EffectiveConfig: cfg,
	}
	manager.mu.Unlock()
	checkpoint := merkle.Snapshot{
		ConfigDigest: cfg.IgnoreDigest,
		Files:        map[string]string{"a.go": captured.Files["a.go"]},
		Inodes:       nil,
	}
	if err := merkle.WriteSnapshot(manager.stagingMerklePath(codebaseID), checkpoint); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}
	staleLive := merkle.Snapshot{
		ConfigDigest: cfg.IgnoreDigest,
		Files:        map[string]string{"old.go": hashText("stale live")},
		Inodes:       nil,
	}
	if err := merkle.WriteSnapshot(manager.merklePath(codebaseID), staleLive); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}
	if manager.resumableCheckpointKind(codebaseID, cfg.IgnoreDigest) != resumeCheckpointStaging {
		t.Fatal("resumableCheckpointKind did not report a staging checkpoint")
	}

	manager.ResumeOrphanedJobs(context.Background())
	waitForCodebaseStatus(t, manager, canonical, model.CodebaseStatusIndexed)

	mu.Lock()
	slices.Sort(embedded)
	got := slices.Clone(embedded)
	mu.Unlock()
	if want := []string{"b.go", "c.go"}; !slices.Equal(got, want) {
		t.Fatalf("embedded files = %v, want %v (a.go must be skipped via staging checkpoint)", got, want)
	}
}

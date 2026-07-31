package daemon

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/store"
)

// newTestManagerWithCap builds a manager whose index-slot cap is set to
// maxConcurrent, so cap-sensitive tests can drive distinct-path jobs against
// a known concurrency limit. It returns the manager and its state root so the
// caller can mint additional repos under the same config.
func newTestManagerWithCap(t *testing.T, maxConcurrent int) (*Manager, config.Config) {
	t.Helper()

	stateRoot := t.TempDir()
	cfg := config.Config{
		StateRoot:              stateRoot,
		SocketPath:             filepath.Join(stateRoot, "sockets", "lm-semantic-search-daemon.sock"),
		RegistryPath:           filepath.Join(stateRoot, "registry.json"),
		JobsPath:               filepath.Join(stateRoot, "jobs.jsonl"),
		EventsPath:             filepath.Join(stateRoot, "events.jsonl"),
		LogsDir:                filepath.Join(stateRoot, "logs"),
		LogPath:                filepath.Join(stateRoot, "logs", "lm-semantic-search-daemon.log"),
		MerkleDir:              filepath.Join(stateRoot, "merkle"),
		LocksDir:               filepath.Join(stateRoot, "locks"),
		SocketsDir:             filepath.Join(stateRoot, "sockets"),
		ChunksDir:              filepath.Join(stateRoot, "chunks"),
		GraphDir:               filepath.Join(stateRoot, "graph"),
		ContextRoot:            filepath.Join(stateRoot, "context"),
		EmbeddingProvider:      "OpenAI",
		EmbeddingModel:         "nvidia/NV-EmbedCode-7b-v1",
		HybridMode:             true,
		SyncIntervalMS:         300000,
		MaxConcurrentIndexJobs: maxConcurrent,
		ResumeIndexingOnBoot:   true,
	}
	for _, path := range []string{cfg.StateRoot, cfg.LogsDir, cfg.MerkleDir, cfg.LocksDir, cfg.SocketsDir, cfg.ChunksDir, cfg.GraphDir, cfg.ContextRoot} {
		if err := store.EnsureDir(path); err != nil {
			t.Fatalf("EnsureDir returned error: %v", err)
		}
	}
	if err := store.WriteRegistry(cfg.RegistryPath, model.RegistryFile{}); err != nil {
		t.Fatalf("WriteRegistry returned error: %v", err)
	}

	manager, err := NewManager(context.Background(), cfg)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}
	t.Cleanup(manager.CloseGraphEngines)
	return manager, cfg
}

// newCapTestRepo mints a fresh repo directory containing a single Go file so
// each cap-test job converges on a distinct codebase path.
func newCapTestRepo(t *testing.T) string {
	t.Helper()

	repoPath := filepath.Join(t.TempDir(), "repo")
	if err := store.EnsureDir(repoPath); err != nil {
		t.Fatalf("EnsureDir returned error: %v", err)
	}
	if err := os.WriteFile(filepath.Join(repoPath, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	return repoPath
}

// blockingRunner returns a fakeRunner whose IndexOne blocks on release after
// signalling entry on entered, so a test can observe how many jobs run at once
// and then let them finish. The from-scratch build drives the per-file path,
// so the block lands in IndexOne rather than the no-longer-used bulk Index.
func blockingRunner(entered chan<- struct{}, release <-chan struct{}, inFlight *atomic.Int32, maxInFlight *atomic.Int32) fakeRunner {
	return fakeRunner{
		index:      nil,
		indexFiles: nil,
		indexOne: func(ctx context.Context, root string, relativePath string, indexConfig model.IndexConfig) (indexer.OneFileResult, error) {
			current := inFlight.Add(1)
			for {
				observed := maxInFlight.Load()
				if current <= observed || maxInFlight.CompareAndSwap(observed, current) {
					break
				}
			}
			entered <- struct{}{}
			select {
			case <-release:
			case <-ctx.Done():
				inFlight.Add(-1)
				return indexer.OneFileResult{Chunks: nil, FileHash: "", Skipped: false, Removed: false}, ctx.Err()
			}
			inFlight.Add(-1)
			content := "package main\n"
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
}

func TestStartIndexCapsConcurrentRunningJobs(t *testing.T) {
	const cap = 2
	const totalJobs = 5

	manager, _ := newTestManagerWithCap(t, cap)
	entered := make(chan struct{}, totalJobs)
	release := make(chan struct{})
	inFlight := atomic.Int32{}
	maxInFlight := atomic.Int32{}
	manager.runner = blockingRunner(entered, release, &inFlight, &maxInFlight)

	repos := make([]string, totalJobs)
	for i := range repos {
		repos[i] = newCapTestRepo(t)
		if _, _, _, _, err := manager.StartIndex(context.Background(), repos[i], testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget); err != nil {
			t.Fatalf("StartIndex %d returned error: %v", i, err)
		}
	}

	// Exactly cap jobs may enter the runner before any slot frees.
	for i := 0; i < cap; i++ {
		<-entered
	}
	waitForCondition(t, func() bool {
		return inFlight.Load() == int32(cap)
	})
	if got := maxInFlight.Load(); got > int32(cap) {
		t.Fatalf("max in-flight jobs = %d, want <= %d", got, cap)
	}

	close(release)
	for i := cap; i < totalJobs; i++ {
		<-entered
	}
	for _, repo := range repos {
		waitForCodebaseStatus(t, manager, repo, model.CodebaseStatusIndexed)
	}
	if got := maxInFlight.Load(); got > int32(cap) {
		t.Fatalf("max in-flight jobs = %d over full run, want <= %d", got, cap)
	}
}

func TestStartIndexQueuedBehindCapReportsQueuedThenRunning(t *testing.T) {
	const cap = 1

	manager, _ := newTestManagerWithCap(t, cap)
	entered := make(chan struct{}, 2)
	release := make(chan struct{})
	inFlight := atomic.Int32{}
	maxInFlight := atomic.Int32{}
	manager.runner = blockingRunner(entered, release, &inFlight, &maxInFlight)

	firstRepo := newCapTestRepo(t)
	secondRepo := newCapTestRepo(t)

	if _, _, _, _, err := manager.StartIndex(context.Background(), firstRepo, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget); err != nil {
		t.Fatalf("first StartIndex returned error: %v", err)
	}
	<-entered

	secondJob, _, _, _, err := manager.StartIndex(context.Background(), secondRepo, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget)
	if err != nil {
		t.Fatalf("second StartIndex returned error: %v", err)
	}

	queued, found := manager.GetJob(secondJob.ID)
	if !found {
		t.Fatal("second job not found")
	}
	if queued.State != model.JobStateQueued {
		t.Fatalf("second job state = %q, want %q while the cap is full", queued.State, model.JobStateQueued)
	}

	close(release)
	<-entered
	waitForCondition(t, func() bool {
		running, ok := manager.GetJob(secondJob.ID)
		return ok && (running.State == model.JobStateRunning || running.State == model.JobStateCompleted)
	})
	// Both jobs must settle before the test returns, so neither runner goroutine
	// is still writing into the state root when t.Cleanup removes it.
	waitForCodebaseStatus(t, manager, firstRepo, model.CodebaseStatusIndexed)
	waitForCodebaseStatus(t, manager, secondRepo, model.CodebaseStatusIndexed)
}

// A reuse collection can spend the full bounded load window in Milvus. Once
// that read has run past the release grace, it gives up both scarce holds, so
// another job can start even when the configured job cap is one.
func TestStuckReuseLoadDoesNotHoldIndexSlotOrSyncLock(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 1)
	manager.jobCapacityTimings.ReleaseGrace = 10 * time.Millisecond
	parentRepo, childRepo := newParentWithChildRepo(t)
	secondRepo := newCapTestRepo(t)

	loadEntered := make(chan struct{}, 1)
	releaseLoad := make(chan struct{})
	t.Cleanup(func() {
		select {
		case <-releaseLoad:
		default:
			close(releaseLoad)
		}
	})
	manager.semantic = &fakeSemantic{
		loadReuse: func(
			ctx context.Context,
			_ []string,
		) (map[string][]float32, error) {
			select {
			case loadEntered <- struct{}{}:
			default:
			}
			select {
			case <-releaseLoad:
				return map[string][]float32{}, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		},
	}

	secondStarted := make(chan struct{}, 1)
	manager.runner = fakeRunner{
		indexOne: func(
			_ context.Context,
			_ string,
			relativePath string,
			_ model.IndexConfig,
		) (indexer.OneFileResult, error) {
			if relativePath == "main.go" {
				select {
				case secondStarted <- struct{}{}:
				default:
				}
			}
			content := "package main\n"
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
			}, nil
		},
	}

	indexConfig := defaultIndexConfig()
	_, firstJob := seedBootstrapCodebase(t, manager, parentRepo, indexConfig)
	registerIndexedChild(
		manager,
		"cb-stuck-reuse-child",
		childRepo,
		indexConfig,
		"hybrid_code_chunks_stuck",
	)
	manager.runJobAsync(context.Background(), firstJob.ID)

	select {
	case <-loadEntered:
	case <-time.After(5 * time.Second):
		t.Fatal("parent job never entered the stuck reuse load")
	}
	waitForCondition(t, func() bool {
		manager.syncLock.mu.Lock()
		lockRefcount := manager.syncLock.refcount
		manager.syncLock.mu.Unlock()
		return len(manager.indexSlots) == 0 && lockRefcount == 0
	})

	if _, _, _, _, err := manager.StartIndex(
		context.Background(),
		secondRepo,
		testClientInfo(),
		indexConfig,
		false,
		emptyAdmissionBudget,
	); err != nil {
		t.Fatalf("second StartIndex returned error: %v", err)
	}
	select {
	case <-secondStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("second job did not start while the first reuse load was stuck")
	}

	close(releaseLoad)
	waitForCondition(t, func() bool {
		completed, found := manager.GetJob(firstJob.ID)
		return found && completed.State == model.JobStateCompleted
	})
	waitForCodebaseStatus(t, manager, secondRepo, model.CodebaseStatusIndexed)
}

// A job whose stalled reuse read finally returns must reach a terminal state
// even when another job has taken the slot it gave up. The read here stalls
// well past the release grace, so the holds really are surrendered; the resume
// then has its own deadline, and the caller observes a failed terminal job
// instead of one left running behind the holder.
func TestReuseLoadResumeDeadlineTerminatesJobWhileReplacementHoldsSlot(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 1)
	manager.jobCapacityTimings.ReleaseGrace = 10 * time.Millisecond
	manager.jobCapacityTimings.Reacquire = 100 * time.Millisecond
	parentRepo, childRepo := newParentWithChildRepo(t)
	replacementRepo := newCapTestRepo(t)

	loadEntered := make(chan struct{}, 1)
	releaseLoad := make(chan struct{})
	replacementStarted := make(chan struct{}, 1)
	releaseReplacement := make(chan struct{})
	t.Cleanup(func() {
		select {
		case <-releaseLoad:
		default:
			close(releaseLoad)
		}
		select {
		case <-releaseReplacement:
		default:
			close(releaseReplacement)
		}
	})

	manager.semantic = &fakeSemantic{
		loadReuse: func(
			ctx context.Context,
			_ []string,
		) (map[string][]float32, error) {
			select {
			case loadEntered <- struct{}{}:
			default:
			}
			select {
			case <-releaseLoad:
				return map[string][]float32{}, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		},
	}
	manager.runner = fakeRunner{
		indexOne: func(
			ctx context.Context,
			_ string,
			relativePath string,
			_ model.IndexConfig,
		) (indexer.OneFileResult, error) {
			if relativePath == "main.go" {
				select {
				case replacementStarted <- struct{}{}:
				default:
				}
				select {
				case <-releaseReplacement:
				case <-ctx.Done():
					return indexer.OneFileResult{}, ctx.Err()
				}
			}
			content := "package main\n"
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
			}, nil
		},
	}

	indexConfig := defaultIndexConfig()
	_, firstJob := seedBootstrapCodebase(t, manager, parentRepo, indexConfig)
	registerIndexedChild(
		manager,
		"cb-resume-deadline-child",
		childRepo,
		indexConfig,
		"hybrid_code_chunks_resume_deadline",
	)
	manager.runJobAsync(context.Background(), firstJob.ID)

	select {
	case <-loadEntered:
	case <-time.After(5 * time.Second):
		t.Fatal("parent job never entered the reuse load")
	}
	if _, _, _, _, err := manager.StartIndex(
		context.Background(),
		replacementRepo,
		testClientInfo(),
		indexConfig,
		false,
		emptyAdmissionBudget,
	); err != nil {
		t.Fatalf("replacement StartIndex returned error: %v", err)
	}
	select {
	case <-replacementStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("replacement job did not take the released slot")
	}

	close(releaseLoad)
	terminalDeadline := time.NewTimer(2 * time.Second)
	defer terminalDeadline.Stop()
	terminalPoll := time.NewTicker(10 * time.Millisecond)
	defer terminalPoll.Stop()
	var terminal model.Job
	for terminal.State != model.JobStateFailed {
		select {
		case <-terminalDeadline.C:
			t.Fatalf("parent job remained non-terminal after its resume deadline: state=%q", terminal.State)
		case <-terminalPoll.C:
			observed, found := manager.GetJob(firstJob.ID)
			if !found {
				t.Fatalf("parent job %s disappeared", firstJob.ID)
			}
			terminal = observed
		}
	}
	if terminal.CompletedAt == nil || terminal.Error == nil {
		t.Fatalf("failed parent job lacks terminal error metadata: %+v", terminal)
	}

	close(releaseReplacement)
	waitForCodebaseStatus(t, manager, replacementRepo, model.CodebaseStatusIndexed)
}

// An ordinary sync reads stored vectors once per changed file, and every one of
// those reads answers straight away. Such a job keeps its indexing slot for its
// whole run even while another job sits queued behind the cap waiting for one,
// so the queued job only reaches its own first file once every one of the
// sync's per-file reads is behind it.
func TestHealthyPerFileReuseReadsKeepTheIndexSlot(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 1)
	// The production release grace stands, so a read that answers immediately
	// must never reach it. The resume bound is shortened so that a job which did
	// give up its holds fails fast rather than stretching the test.
	manager.jobCapacityTimings.Reacquire = 100 * time.Millisecond

	const syncFileCount = 6
	syncRepo := newMultiFileRepo(t, "a.go", "b.go", "c.go", "d.go", "e.go", "f.go")
	queuedRepo := newCapTestRepo(t)

	semanticFake := &fakeSemantic{}
	manager.semantic = semanticFake

	var openSyncGate sync.Once
	syncEntered := make(chan struct{})
	syncGate := make(chan struct{})
	queuedGate := make(chan struct{})
	syncReadsWhenQueuedRan := make(chan int, 1)
	t.Cleanup(func() {
		closeOnce(syncGate)
		closeOnce(queuedGate)
	})

	indexConfig := defaultIndexConfig()
	_, syncJob := seedBootstrapCodebase(t, manager, syncRepo, indexConfig)

	manager.runner = fakeRunner{
		indexOne: func(
			ctx context.Context,
			root string,
			relativePath string,
			_ model.IndexConfig,
		) (indexer.OneFileResult, error) {
			if root == syncRepo {
				openSyncGate.Do(func() {
					close(syncEntered)
					<-syncGate
				})
			} else {
				select {
				case syncReadsWhenQueuedRan <- len(semanticFake.reusePathCallsSnapshot()):
				default:
				}
				select {
				case <-queuedGate:
				case <-ctx.Done():
					return indexer.OneFileResult{}, ctx.Err()
				}
			}
			content := "package main\n// " + relativePath + "\n"
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
			}, nil
		},
	}

	manager.runJobAsync(context.Background(), syncJob.ID)
	select {
	case <-syncEntered:
	case <-time.After(5 * time.Second):
		t.Fatal("sync job never reached its first file")
	}

	queuedJob, _, _, _, err := manager.StartIndex(
		context.Background(),
		queuedRepo,
		testClientInfo(),
		indexConfig,
		false,
		emptyAdmissionBudget,
	)
	if err != nil {
		t.Fatalf("queued StartIndex returned error: %v", err)
	}
	// Wait until the queued job's runner goroutine exists, so it is contending
	// for the one slot while the sync performs its per-file reads.
	waitForCondition(t, func() bool {
		manager.mu.Lock()
		defer manager.mu.Unlock()
		return manager.done[queuedJob.ID] != nil
	})
	close(syncGate)

	waitForCondition(t, func() bool {
		observed, found := manager.GetJob(syncJob.ID)
		return found && (observed.State == model.JobStateCompleted ||
			observed.State == model.JobStateFailed ||
			observed.State == model.JobStateCancelled)
	})
	settled, _ := manager.GetJob(syncJob.ID)
	if settled.State != model.JobStateCompleted {
		t.Fatalf("sync job state = %q, want %q; error = %v", settled.State, model.JobStateCompleted, settled.Error)
	}
	if reads := len(semanticFake.reusePathCallsSnapshot()); reads != syncFileCount {
		t.Fatalf("per-file reuse reads = %d, want %d so the run really exercised them", reads, syncFileCount)
	}

	select {
	case observed := <-syncReadsWhenQueuedRan:
		if observed != syncFileCount {
			t.Fatalf(
				"queued job took the slot after only %d of the sync's %d per-file reads, want it to wait for all of them",
				observed,
				syncFileCount,
			)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("queued job never started after the sync released its slot")
	}

	close(queuedGate)
	waitForCodebaseStatus(t, manager, queuedRepo, model.CodebaseStatusIndexed)
}

// closeOnce closes a gate channel that a test may already have closed, so a
// cleanup can unblock a runner without knowing how far the test got.
func closeOnce(gate chan struct{}) {
	select {
	case <-gate:
	default:
		close(gate)
	}
}

func TestCancelQueuedJobBehindCapReachesCancelled(t *testing.T) {
	const cap = 1

	manager, _ := newTestManagerWithCap(t, cap)
	entered := make(chan struct{}, 1)
	release := make(chan struct{})
	inFlight := atomic.Int32{}
	maxInFlight := atomic.Int32{}
	manager.runner = blockingRunner(entered, release, &inFlight, &maxInFlight)

	firstRepo := newCapTestRepo(t)
	secondRepo := newCapTestRepo(t)

	if _, _, _, _, err := manager.StartIndex(context.Background(), firstRepo, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget); err != nil {
		t.Fatalf("first StartIndex returned error: %v", err)
	}
	<-entered

	secondJob, _, _, _, err := manager.StartIndex(context.Background(), secondRepo, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget)
	if err != nil {
		t.Fatalf("second StartIndex returned error: %v", err)
	}

	manager.mu.Lock()
	jobDone := manager.done[secondJob.ID]
	manager.mu.Unlock()
	if jobDone == nil {
		t.Fatal("queued job has no done channel")
	}

	if _, err := manager.CancelJob(context.Background(), secondJob.ID); err != nil {
		t.Fatalf("CancelJob returned error: %v", err)
	}

	select {
	case <-jobDone:
	case <-time.After(5 * time.Second):
		t.Fatal("queued job done channel did not close after cancel")
	}

	cancelled, found := manager.GetJob(secondJob.ID)
	if !found {
		t.Fatal("cancelled job not found")
	}
	if cancelled.State != model.JobStateCancelled {
		t.Fatalf("cancelled job state = %q, want %q", cancelled.State, model.JobStateCancelled)
	}

	// A force-reindex against the same queued path must not hang now that the
	// queued job is cancelled and its slot was never held.
	forceDone := make(chan struct{})
	go func() {
		defer close(forceDone)
		_, _, _, _, _ = manager.StartIndex(context.Background(), secondRepo, testClientInfo(), defaultIndexConfig(), true, emptyAdmissionBudget)
	}()
	select {
	case <-forceDone:
	case <-time.After(5 * time.Second):
		t.Fatal("force-reindex of a queued path hung")
	}

	// Release the blocked first job and wait for both repos to settle so the
	// runner goroutines stop writing before t.Cleanup removes the temp dirs.
	close(release)
	waitForCodebaseStatus(t, manager, firstRepo, model.CodebaseStatusIndexed)
	waitForCodebaseStatus(t, manager, secondRepo, model.CodebaseStatusIndexed)
}

func TestResumeOrphanedJobsHonorsResumeOnBoot(t *testing.T) {
	cases := []struct {
		name          string
		resumeOnBoot  bool
		hasCheckpoint bool
		wantResumeJob bool
	}{
		{name: "resume enabled with checkpoint launches", resumeOnBoot: true, hasCheckpoint: true, wantResumeJob: true},
		{name: "resume enabled without checkpoint skips", resumeOnBoot: true, hasCheckpoint: false, wantResumeJob: false},
		{name: "resume disabled skips", resumeOnBoot: false, hasCheckpoint: true, wantResumeJob: false},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			manager, _ := newTestManagerWithCap(t, 2)
			manager.config.ResumeIndexingOnBoot = testCase.resumeOnBoot
			release := make(chan struct{})
			manager.runner = fakeRunner{
				index:      nil,
				indexFiles: nil,
				indexOne: func(ctx context.Context, root string, relativePath string, indexConfig model.IndexConfig) (indexer.OneFileResult, error) {
					<-release
					content := "package main\n"
					return indexer.OneFileResult{
						Chunks:   []model.StoredChunk{{Content: content, RelativePath: relativePath, StartLine: 1, EndLine: 1, Language: "go", FileExtension: ".go"}},
						FileHash: hashText(content),
						Skipped:  false,
						Removed:  false,
					}, nil
				},
			}

			repoPath := newCapTestRepo(t)
			canonical, err := filepath.EvalSymlinks(repoPath)
			if err != nil {
				t.Fatalf("EvalSymlinks returned error: %v", err)
			}
			codebaseID := "cb-resume-test"
			resumeConfig := defaultIndexConfig()
			resumeConfig.IgnoreDigest = "sha256:resume-checkpoint"
			manager.mu.Lock()
			manager.codebases[codebaseID] = model.Codebase{
				ID:              codebaseID,
				CanonicalPath:   canonical,
				Status:          model.CodebaseStatusIndexing,
				EffectiveConfig: resumeConfig,
			}
			manager.mu.Unlock()

			if testCase.hasCheckpoint {
				snapshot := merkle.Snapshot{
					ConfigDigest: resumeConfig.IgnoreDigest,
					Files:        map[string]string{"main.go": hashText("package main\n")},
				}
				if err := merkle.WriteSnapshot(manager.merklePath(codebaseID), snapshot); err != nil {
					t.Fatalf("WriteSnapshot returned error: %v", err)
				}
			}

			manager.ResumeOrphanedJobs(context.Background())

			if testCase.wantResumeJob {
				waitForCondition(t, func() bool {
					return len(manager.ListJobs("")) >= 1
				})
			} else {
				if jobs := manager.ListJobs(""); len(jobs) != 0 {
					t.Fatalf("resume disabled launched %d jobs, want 0", len(jobs))
				}
			}

			manager.mu.Lock()
			_, stillTracked := manager.codebases[codebaseID]
			manager.mu.Unlock()
			if !stillTracked {
				t.Fatal("codebase no longer tracked after ResumeOrphanedJobs")
			}

			// Drain any launched job before returning so the runner goroutine
			// stops writing before t.Cleanup removes the temp dirs.
			close(release)
			if testCase.wantResumeJob {
				waitForCodebaseStatus(t, manager, canonical, model.CodebaseStatusIndexed)
			}
		})
	}
}

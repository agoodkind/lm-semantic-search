package daemon

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/localvec"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// indexedTestCodebase registers one indexed codebase on the manager and returns
// it, so a self-check has a real target to query.
func indexedTestCodebase(t *testing.T, manager *Manager, repoPath string) model.Codebase {
	t.Helper()

	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.LastSuccessfulRun = &model.IndexRunSummary{
		IndexedFiles: 1,
		TotalChunks:  1,
		Status:       "completed",
		CompletedAt:  clock.Now(),
	}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	return codebase
}

// healthModeNow reads the raw health record. DependencyHealth() applies a
// reconnect shortcut that clears a store-unavailable mode when the backend
// reports available, which would mask exactly what these tests assert, so they
// read the record the self-check wrote.
func healthModeNow(manager *Manager) dependencyMode {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	return manager.health.Mode
}

func snapshotCollectionFiles(t *testing.T, root string) map[string][]byte {
	t.Helper()

	snapshot := make(map[string][]byte)
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if !entry.Type().IsRegular() {
			return nil
		}
		relativePath, relativeErr := filepath.Rel(root, path)
		if relativeErr != nil {
			return relativeErr
		}
		content, readErr := os.ReadFile(path)
		if readErr != nil {
			return readErr
		}
		snapshot[relativePath] = content
		return nil
	})
	if err != nil {
		t.Fatalf("snapshot collection files: %v", err)
	}
	return snapshot
}

// The local backend may repair a torn metadata tail during an ordinary lazy
// load, so the boot check must skip that backend rather than mutate the
// collection while claiming to probe it read-only.
func TestBootSelfCheckLeavesLocalCollectionFilesUnchanged(t *testing.T) {
	manager, cfg, repoPath := newTestManager(t)
	codebase := indexedTestCodebase(t, manager, repoPath)

	localStore, err := localvec.New(
		context.Background(),
		config.Config{StateRoot: cfg.StateRoot},
	)
	if err != nil {
		t.Fatalf("create local vector store: %v", err)
	}
	manager.config.IndexBackend = config.IndexBackendLocal
	manager.semantic = localStore

	collectionPath := filepath.Join(
		cfg.StateRoot,
		"localvec",
		localStore.CollectionName(codebase.CanonicalPath),
	)
	if err := os.MkdirAll(collectionPath, 0o700); err != nil {
		t.Fatalf("create local collection directory: %v", err)
	}
	validRow := `{"id":"chunk_a","relativePath":"a.go","content":"a","contentVectorKey":"ka","vector":[1,0]}` + "\n"
	tornRow := `{"id":"chunk_b","relativePath":"b.go","conte`
	if err := os.WriteFile(
		filepath.Join(collectionPath, "metadata.jsonl"),
		[]byte(validRow+tornRow),
		0o600,
	); err != nil {
		t.Fatalf("write torn local metadata: %v", err)
	}
	if err := os.WriteFile(
		filepath.Join(collectionPath, "index.usearch"),
		[]byte("invalid index is sufficient because the probe must not open it"),
		0o600,
	); err != nil {
		t.Fatalf("write local index fixture: %v", err)
	}

	before := snapshotCollectionFiles(t, collectionPath)
	outcome, err := manager.runBootSelfCheck(context.Background())
	after := snapshotCollectionFiles(t, collectionPath)

	if !reflect.DeepEqual(after, before) {
		t.Fatalf("local collection files changed during boot check:\nbefore=%q\nafter=%q", before, after)
	}
	if err != nil {
		t.Fatalf("runBootSelfCheck returned error for skipped local backend: %v", err)
	}
	if outcome != bootSelfCheckSkipped {
		t.Fatalf("outcome = %q, want %q for local backend", outcome, bootSelfCheckSkipped)
	}
}

// The ONNX inference boundary cannot be interrupted safely. The boot check must
// not enter it, so later daemon searches retain the same runtime.
func TestBootSelfCheckSkipsUninterruptibleInferenceAndLeavesSearchUsable(
	t *testing.T,
) {
	manager, _, repoPath := newTestManager(t)
	indexedTestCodebase(t, manager, repoPath)
	manager.config.IndexBackend = config.IndexBackendMilvus
	manager.config.EmbeddingProvider = config.EmbeddingProviderONNX

	var runtimeMutex sync.Mutex
	var bootCalls atomic.Int32
	bootEntered := make(chan struct{})
	releaseBoot := make(chan struct{})
	t.Cleanup(func() {
		close(releaseBoot)
	})
	manager.semantic = &fakeSemantic{
		search: func(
			_ context.Context,
			_ string,
			query string,
			_ int32,
			_ []string,
			_ string,
		) ([]model.StoredChunk, error) {
			runtimeMutex.Lock()
			defer runtimeMutex.Unlock()
			if query == bootSelfCheckQuery {
				bootCalls.Add(1)
				close(bootEntered)
				<-releaseBoot
				return nil, errors.New("released blocked boot inference")
			}
			return []model.StoredChunk{{
				Content:      "later search",
				RelativePath: "main.go",
			}}, nil
		},
	}

	checkDone := make(chan bootSelfCheckOutcome, 1)
	go func() {
		outcome, _ := manager.runBootSelfCheck(context.Background())
		checkDone <- outcome
	}()

	select {
	case outcome := <-checkDone:
		if outcome != bootSelfCheckSkipped {
			t.Fatalf("boot outcome = %q, want %q", outcome, bootSelfCheckSkipped)
		}
	case <-bootEntered:
	case <-time.After(time.Second):
		t.Fatal("boot check neither skipped nor entered the dependency")
	}

	searchDone := make(chan error, 1)
	go func() {
		outcome, searchErr := manager.SearchCode(
			context.Background(),
			repoPath,
			"later query",
			1,
			nil,
		)
		if searchErr == nil && len(outcome.Results) != 1 {
			searchErr = errors.New("later search returned no result")
		}
		searchDone <- searchErr
	}()

	select {
	case searchErr := <-searchDone:
		if searchErr != nil {
			t.Fatalf("later daemon search failed: %v", searchErr)
		}
	case <-time.After(250 * time.Millisecond):
		t.Fatal("later daemon search blocked behind boot inference")
	}
	if got := bootCalls.Load(); got != 0 {
		t.Fatalf("boot inference calls = %d, want 0", got)
	}
}

// A boot that can still answer a real query reports one passing line and clears
// the store banner the boot dial left behind.
func TestBootSelfCheckPassesAndClearsBootBanner(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	codebase := indexedTestCodebase(t, manager, repoPath)

	queried := make(chan string, 1)
	manager.semantic = &fakeSemantic{
		search: func(_ context.Context, codebasePath string, _ string, limit int32, _ []string, _ string) ([]model.StoredChunk, error) {
			queried <- codebasePath
			if limit != bootSelfCheckLimit {
				t.Errorf("self-check search limit = %d, want %d", limit, bootSelfCheckLimit)
			}
			return []model.StoredChunk{{Content: "package main", RelativePath: "main.go"}}, nil
		},
	}
	manager.mu.Lock()
	manager.health = dependencyHealth{Mode: dependencyStoreUnavailable, Since: clock.Now(), LastHealthyAt: time.Time{}}
	manager.mu.Unlock()

	outcome, err := manager.runBootSelfCheck(context.Background())
	if err != nil {
		t.Fatalf("runBootSelfCheck returned error: %v", err)
	}
	if outcome != bootSelfCheckPassed {
		t.Fatalf("outcome = %q, want %q", outcome, bootSelfCheckPassed)
	}
	select {
	case got := <-queried:
		if got != codebase.CanonicalPath {
			t.Fatalf("self-check queried %q, want the indexed codebase %q", got, codebase.CanonicalPath)
		}
	default:
		t.Fatal("self-check ran no query; it must exercise the real read path")
	}
	if got := healthModeNow(manager); got != dependencyHealthy {
		t.Fatalf("health mode = %q after a passing self-check, want %q", got, dependencyHealthy)
	}
}

// A restore that left the collection unusable must show up as a failed check
// that degrades the shared readiness record, so no surface reports ready while
// the data cannot answer.
func TestBootSelfCheckFailureDegradesReadiness(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	indexedTestCodebase(t, manager, repoPath)

	manager.semantic = &fakeSemantic{
		search: func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error) {
			return nil, semantic.ErrUnavailable
		},
	}

	outcome, err := manager.runBootSelfCheck(context.Background())
	if outcome != bootSelfCheckFailed {
		t.Fatalf("outcome = %q, want %q", outcome, bootSelfCheckFailed)
	}
	if !errors.Is(err, semantic.ErrUnavailable) {
		t.Fatalf("runBootSelfCheck error = %v, want it to carry the store outage", err)
	}
	if got := healthModeNow(manager); got != dependencyStoreUnavailable {
		t.Fatalf("health mode = %q after a failed self-check, want %q", got, dependencyStoreUnavailable)
	}
}

// A collection that is gone is the bad-restore case the check exists to catch,
// and it must be reported rather than swallowed.
func TestBootSelfCheckReportsMissingCollection(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	indexedTestCodebase(t, manager, repoPath)

	manager.semantic = &fakeSemantic{
		search: func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error) {
			return nil, semantic.ErrCollectionMissing
		},
	}

	outcome, err := manager.runBootSelfCheck(context.Background())
	if outcome != bootSelfCheckFailed {
		t.Fatalf("outcome = %q, want %q", outcome, bootSelfCheckFailed)
	}
	if !errors.Is(err, semantic.ErrCollectionMissing) {
		t.Fatalf("runBootSelfCheck error = %v, want the missing-collection classification", err)
	}
}

// A daemon with nothing indexed has no data to prove usable, so the check skips
// without touching the health record or issuing a query.
func TestBootSelfCheckSkipsWithNothingIndexed(t *testing.T) {
	manager, _, _ := newTestManager(t)
	manager.semantic = &fakeSemantic{
		search: func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error) {
			t.Error("self-check queried the store with no indexed codebase")
			return nil, nil
		},
	}

	outcome, err := manager.runBootSelfCheck(context.Background())
	if err != nil {
		t.Fatalf("runBootSelfCheck returned error: %v", err)
	}
	if outcome != bootSelfCheckSkipped {
		t.Fatalf("outcome = %q, want %q", outcome, bootSelfCheckSkipped)
	}
	if got := healthModeNow(manager); got != dependencyHealthy {
		t.Fatalf("health mode = %q after a skipped self-check, want it untouched (%q)", got, dependencyHealthy)
	}
}

// The check never gates startup: StartBootSelfCheck returns while a hung store
// still has the probe query parked, so the daemon goes on to serve.
func TestStartBootSelfCheckDoesNotBlockStartup(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	indexedTestCodebase(t, manager, repoPath)

	entered := make(chan struct{})
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	manager.bootSelfCheckDelay = time.Millisecond
	manager.semantic = &fakeSemantic{
		search: func(ctx context.Context, _ string, _ string, _ int32, _ []string, _ string) ([]model.StoredChunk, error) {
			close(entered)
			select {
			case <-release:
			case <-ctx.Done():
			}
			return nil, ctx.Err()
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	returnedAt := make(chan time.Duration, 1)
	startedAt := clock.Now()
	manager.StartBootSelfCheck(ctx)
	returnedAt <- clock.Now().Sub(startedAt)

	if elapsed := <-returnedAt; elapsed > time.Second {
		t.Fatalf("StartBootSelfCheck took %s to return; startup must not wait on the check", elapsed)
	}

	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("the background self-check never ran")
	}
	// The hung probe is still parked here, which is the point: startup already
	// returned and the daemon would be serving.
}

// A failed remote check keeps retrying on its bounded schedule. A later pass
// clears readiness without waiting for a job, file change, or user query.
func TestBootSelfCheckRetriesFailureUntilReadinessRecovers(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	indexedTestCodebase(t, manager, repoPath)
	manager.bootSelfCheckDelay = time.Millisecond

	var attempts atomic.Int32
	manager.semantic = &fakeSemantic{
		search: func(
			context.Context,
			string,
			string,
			int32,
			[]string,
			string,
		) ([]model.StoredChunk, error) {
			if attempts.Add(1) == 1 {
				return nil, adapterr.NewEmbedderUnreachable(
					errors.New("embedding endpoint is unavailable"),
				)
			}
			return []model.StoredChunk{{Content: "healthy"}}, nil
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	manager.StartBootSelfCheck(ctx)

	deadline := time.NewTimer(time.Second)
	defer deadline.Stop()
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	for {
		if attempts.Load() >= 2 && healthModeNow(manager) == dependencyHealthy {
			return
		}
		select {
		case <-deadline.C:
			t.Fatalf(
				"boot attempts = %d and health mode = %q, want a retry and healthy readiness",
				attempts.Load(),
				healthModeNow(manager),
			)
		case <-ticker.C:
		}
	}
}

// A boot query that started before a later dependency failure is stale evidence.
// Its success must leave the newer failure visible.
func TestBootSelfCheckSuccessDoesNotClearNewerFailure(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	indexedTestCodebase(t, manager, repoPath)

	entered := make(chan struct{})
	release := make(chan struct{})
	manager.semantic = &fakeSemantic{
		search: func(
			context.Context,
			string,
			string,
			int32,
			[]string,
			string,
		) ([]model.StoredChunk, error) {
			close(entered)
			<-release
			return []model.StoredChunk{{Content: "stale success"}}, nil
		},
	}

	done := make(chan error, 1)
	go func() {
		_, err := manager.runBootSelfCheck(context.Background())
		done <- err
	}()
	<-entered
	manager.noteDependencyFailure(
		adapterr.NewEmbedderBusy(errors.New("embedding endpoint remained at capacity")),
	)
	close(release)

	if err := <-done; err != nil {
		t.Fatalf("runBootSelfCheck returned error: %v", err)
	}
	if got := healthModeNow(manager); got != dependencyEmbedderBusy {
		t.Fatalf(
			"health mode = %q after stale boot success, want newer %q failure preserved",
			got,
			dependencyEmbedderBusy,
		)
	}
}

// The target is the most recently completed index, so the probe hits the data
// most likely to matter and two boots on the same registry pick the same one.
func TestBootSelfCheckTargetPicksMostRecentIndex(t *testing.T) {
	manager, _, _ := newTestManager(t)

	older := newCodebaseRecord("/tmp/older-repo")
	older.Status = model.CodebaseStatusIndexed
	older.LastSuccessfulRun = &model.IndexRunSummary{CompletedAt: clock.Now().Add(-2 * time.Hour)}
	newer := newCodebaseRecord("/tmp/newer-repo")
	newer.Status = model.CodebaseStatusIndexed
	newer.LastSuccessfulRun = &model.IndexRunSummary{CompletedAt: clock.Now()}
	failed := newCodebaseRecord("/tmp/failed-repo")
	failed.Status = model.CodebaseStatusFailed
	failed.LastSuccessfulRun = &model.IndexRunSummary{CompletedAt: clock.Now().Add(time.Hour)}

	manager.mu.Lock()
	for _, codebase := range []model.Codebase{older, newer, failed} {
		manager.codebases[codebase.ID] = codebase
	}
	manager.mu.Unlock()

	target, found := manager.bootSelfCheckTarget()
	if !found {
		t.Fatal("bootSelfCheckTarget found no indexed codebase")
	}
	if target.ID != newer.ID {
		t.Fatalf("target = %q (%s), want the most recently indexed %q", target.ID, target.CanonicalPath, newer.ID)
	}
}

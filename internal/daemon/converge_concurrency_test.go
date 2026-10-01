package daemon

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/metrics"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/tshash"
)

// fakeSemantic is a semanticIndex double for converge tests. reindex and
// copyChunks are the only behaviors a converge exercises; the rest return inert
// values so the manager treats the backend as available and empty.
type fakeSemantic struct {
	unavailable bool
	probeErr    error
	// backendName and embeddingProviderName are what this double reports about
	// itself. They default to the local backend and the embedded model so a test
	// that does not care about identity still describes a coherent backend, and a
	// test that does care states the value it wants outright.
	//
	// reportProviderVerbatim turns off the default, so a test can describe a
	// backend that built no embedder at all rather than one that left the field
	// unset.
	backendName            model.VectorBackend
	embeddingProviderName  model.EmbeddingProvider
	reportProviderVerbatim bool
	// probe, when set, replaces the canned probe outcome so a test can observe
	// the context the probe runs under or hold the probe open. probeCount records
	// every ProbeHealth call, which is how a test measures probe volume rather
	// than assuming the debounce holds.
	probe                 func(ctx context.Context) error
	probeCount            atomic.Int64
	ensureMmap            func(context.Context)
	backfillCollections   func(context.Context)
	reindex               func(ctx context.Context, codebasePath string, chunks []model.StoredChunk, removed []string) error
	reindexWithReuse      func(ctx context.Context, codebasePath string, chunks []model.StoredChunk, removed []string, progress func(semantic.Progress), reuse map[string][]float32) error
	stageReindexWithReuse func(ctx context.Context, codebasePath string, chunks []model.StoredChunk, removed []string, progress func(semantic.Progress), reuse map[string][]float32) error
	copyChunks            func(ctx context.Context, codebasePath string, src string, dst string) (int, error)
	deleteItemRows        func(ctx context.Context, collectionName string, removal semantic.Removal) error
	collectionName        func(codebasePath string) string
	documentName          func(collectionID string) string
	inspectCollection     func(context.Context, string) (semantic.CollectionFacts, error)
	describeScalars       func(context.Context, string) ([]model.ScalarColumn, bool, error)
	listCollections       func(context.Context) ([]string, error)
	hasCollectionForPath  func(context.Context, string) (bool, error)
	collectionState       func(context.Context, string) (bool, bool, error)
	observeCollection     func(context.Context, string) (semantic.CollectionObservation, error)
	hasStaging            func(context.Context, string) (bool, error)
	search                func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error)
	collectionSearch      func(context.Context, string, string, int32) ([]model.StoredChunk, error)
	prepareCollection     func(context.Context, string) error
	acquireCollection     func(context.Context, string) (semantic.CollectionLease, error)
	pinStaging            func(context.Context, string) (semantic.CollectionPin, error)
	count                 func(context.Context, string) (int32, error)
	// loadReuse, when set, supplies the reuse map a merge-down build receives and
	// records which collections were asked for. dropped records every Drop call
	// so a test can prove an absorb never drops the absorbed child collection.
	loadReuse              func(ctx context.Context, collectionNames []string) (map[string][]float32, error)
	reuseCollections       [][]string
	loadReuseForPrefix     func(ctx context.Context, collectionName string, relativePathPrefix string) (map[string][]float32, error)
	reusePrefixCalls       []reusePrefixCall
	loadReuseForPath       func(ctx context.Context, collectionName string, relativePath string) (map[string][]float32, error)
	reusePathCalls         []reusePathCall
	loadReuseForContents   func(ctx context.Context, collectionName string, chunks []model.StoredChunk) (map[string][]float32, error)
	collectionSearchScopes [][]string
	dropped                []string
	droppedStaging         []string
	reindexCalls           []reindexCall
	stageCalls             []reindexCall
	promoted               []string
	promoteStaging         func(context.Context, string) error
	reindexEmit            func(progress func(semantic.Progress))
	// maintenanceGate records the last SetMaintenance value the manager passed,
	// so a test can prove the operator's mode reached the backend.
	maintenanceGate atomic.Bool
	mu              sync.Mutex
}

func (f *fakeSemantic) SetMaintenance(enabled bool) {
	f.maintenanceGate.Store(enabled)
}

// RecordCollectionDeclaration accepts the manager's declaration record. The
// fake has no schema migrations for the record to steer.
func (f *fakeSemantic) RecordCollectionDeclaration(string, model.CollectionDeclaration) {}

// LoadCollectionItemBatch reports no stored rows for a generic collection.
func (f *fakeSemantic) LoadCollectionItemBatch(context.Context, string, string, []string) (semantic.CollectionItemBatchState, error) {
	return semantic.CollectionItemBatchState{Rows: map[string]semantic.CollectionItemRows{}, Reuse: map[string][]float32{}}, nil
}

type reindexCall struct {
	CodebasePath string
	Chunks       int
	Removed      []string
	Removal      semantic.Removal
	ColumnSet    semantic.StoreColumnSet
}

func (f *fakeSemantic) BackendName() model.VectorBackend {
	if f.backendName != "" {
		return f.backendName
	}
	return config.IndexBackendLocal
}

func (f *fakeSemantic) EmbeddingProviderName() model.EmbeddingProvider {
	if f.reportProviderVerbatim || f.embeddingProviderName != "" {
		return f.embeddingProviderName
	}
	return config.EmbeddingProviderONNX
}

func (f *fakeSemantic) Available() bool { return !f.unavailable }
func (f *fakeSemantic) ProbeHealth(ctx context.Context) error {
	f.probeCount.Add(1)
	if f.probe != nil {
		return f.probe(ctx)
	}
	if f.unavailable {
		return semantic.ErrUnavailable
	}
	return f.probeErr
}

func (f *fakeSemantic) CollectionName(codebasePath string) string {
	if f.collectionName != nil {
		return f.collectionName(codebasePath)
	}
	return "code_chunks_test"
}

func (f *fakeSemantic) DocumentCollectionName(collectionID string) string {
	if f.documentName != nil {
		return f.documentName(collectionID)
	}
	return "conv_chunks_" + tshash.PathPrefix(collectionID)
}

func (f *fakeSemantic) DescribeScalarColumns(ctx context.Context, collectionName string) ([]model.ScalarColumn, bool, error) {
	if f.describeScalars != nil {
		return f.describeScalars(ctx, collectionName)
	}
	return nil, false, nil
}

func (f *fakeSemantic) HasStaging(ctx context.Context, codebasePath string) (bool, error) {
	if f.hasStaging != nil {
		return f.hasStaging(ctx, codebasePath)
	}
	return false, nil
}

func (f *fakeSemantic) Search(ctx context.Context, codebasePath string, query string, limit int32, extensionFilter []string, relativePathPrefix string) ([]model.StoredChunk, error) {
	if f.search != nil {
		return f.search(ctx, codebasePath, query, limit, extensionFilter, relativePathPrefix)
	}
	return nil, nil
}

func (f *fakeSemantic) AcquireCollection(
	ctx context.Context,
	collectionName string,
) (semantic.CollectionLease, error) {
	if f.acquireCollection != nil {
		return f.acquireCollection(ctx, collectionName)
	}
	return fakeCollectionLease{}, nil
}

func (f *fakeSemantic) PrepareCollection(ctx context.Context, collectionName string) error {
	if f.prepareCollection != nil {
		return f.prepareCollection(ctx, collectionName)
	}
	return nil
}

func (f *fakeSemantic) PinStaging(
	ctx context.Context,
	codebasePath string,
) (semantic.CollectionPin, error) {
	if f.pinStaging != nil {
		return f.pinStaging(ctx, codebasePath)
	}
	return fakeCollectionLease{}, nil
}

type fakeCollectionLease struct {
	release func()
}

func (lease fakeCollectionLease) Release() {
	if lease.release != nil {
		lease.release()
	}
}

func (lease fakeCollectionLease) ReleaseContext(context.Context) {
	lease.Release()
}

func (f *fakeSemantic) SearchCollection(ctx context.Context, search semantic.CollectionSearch) ([]semantic.CollectionHit, error) {
	f.mu.Lock()
	f.collectionSearchScopes = append(f.collectionSearchScopes, itemIDScope(search.Filter, search.Declaration.ItemIDColumn))
	f.mu.Unlock()
	if f.collectionSearch == nil {
		return nil, nil
	}
	chunks, err := f.collectionSearch(ctx, search.CollectionName, search.Query, search.Limit)
	if err != nil {
		return nil, err
	}
	hits := make([]semantic.CollectionHit, 0, len(chunks))
	for _, chunk := range chunks {
		hits = append(hits, semantic.CollectionHit{Chunk: chunk, Scalars: nil})
	}
	return hits, nil
}

func itemIDScope(filter *semantic.CollectionFilter, itemIDColumn string) []string {
	if filter == nil || filter.Kind != semantic.CollectionFilterAll {
		return nil
	}
	for _, child := range filter.Children {
		if child.Kind != semantic.CollectionFilterIn || child.Column != itemIDColumn {
			continue
		}
		scope := make([]string, 0, len(child.Values))
		for _, value := range child.Values {
			scope = append(scope, value.String)
		}
		return scope
	}
	return nil
}

func (f *fakeSemantic) Count(ctx context.Context, codebasePath string) (int32, error) {
	if f.count != nil {
		return f.count(ctx, codebasePath)
	}
	return 0, nil
}

func (f *fakeSemantic) ListCollections(ctx context.Context) ([]string, error) {
	if f.listCollections != nil {
		return f.listCollections(ctx)
	}
	return []string{"code_chunks_test"}, nil
}

func (f *fakeSemantic) InspectCollection(ctx context.Context, collectionName string) (semantic.CollectionFacts, error) {
	if f.inspectCollection != nil {
		return f.inspectCollection(ctx, collectionName)
	}
	if f.hasCollectionForPath != nil {
		// This fallback preserves older repair-test fixtures, but this path
		// passes a collection name. Fixtures keyed by codebase path must set
		// inspectCollection explicitly so they do not compare unlike values.
		exists, err := f.hasCollectionForPath(ctx, collectionName)
		if err != nil {
			return semantic.CollectionFacts{}, err
		}
		if !exists {
			return semantic.CollectionFacts{Exists: false, Rows: 0, RowsKnown: false}, nil
		}
	}
	return semantic.CollectionFacts{Exists: true, Rows: 1, RowsKnown: true}, nil
}

func (f *fakeSemantic) HasCollectionForPath(ctx context.Context, codebasePath string) (bool, error) {
	if f.hasCollectionForPath != nil {
		return f.hasCollectionForPath(ctx, codebasePath)
	}
	return true, nil
}

func (f *fakeSemantic) CollectionState(ctx context.Context, codebasePath string) (bool, bool, error) {
	if f.collectionState != nil {
		return f.collectionState(ctx, codebasePath)
	}
	return true, true, nil
}

func (f *fakeSemantic) ObserveCollection(
	ctx context.Context,
	codebasePath string,
) (semantic.CollectionObservation, error) {
	if f.observeCollection != nil {
		return f.observeCollection(ctx, codebasePath)
	}
	exists, loaded, err := f.CollectionState(ctx, codebasePath)
	if err != nil {
		return semantic.CollectionObservation{}, err
	}
	state := semantic.CollectionStateReady
	if !exists {
		state = semantic.CollectionStateAbsent
	} else if !loaded {
		state = semantic.CollectionStateLoading
	}
	observation := semantic.CollectionObservation{State: state}
	if state == semantic.CollectionStateReady {
		rows, countErr := f.Count(ctx, codebasePath)
		if countErr == nil {
			observation.Rows = rows
			observation.RowsKnown = true
		}
	}
	return observation, nil
}

func (f *fakeSemantic) LoadReuseVectors(ctx context.Context, collectionNames []string) (map[string][]float32, error) {
	f.mu.Lock()
	f.reuseCollections = append(f.reuseCollections, collectionNames)
	f.mu.Unlock()
	if f.loadReuse != nil {
		return f.loadReuse(ctx, collectionNames)
	}
	return map[string][]float32{}, nil
}

// reusePrefixCall records one prefix-scoped reuse load: the collection asked
// for and the relativePath prefix that scoped the read.
type reusePrefixCall struct {
	Collection string
	Prefix     string
}

// reusePathCall records one exact-path reuse load.
type reusePathCall struct {
	Collection string
	Path       string
}

// messageStateCall records one per-message state load.

func (f *fakeSemantic) LoadReuseVectorsForPrefix(ctx context.Context, collectionName string, relativePathPrefix string) (map[string][]float32, error) {
	f.mu.Lock()
	f.reusePrefixCalls = append(f.reusePrefixCalls, reusePrefixCall{Collection: collectionName, Prefix: relativePathPrefix})
	f.mu.Unlock()
	if f.loadReuseForPrefix != nil {
		return f.loadReuseForPrefix(ctx, collectionName, relativePathPrefix)
	}
	return map[string][]float32{}, nil
}

func (f *fakeSemantic) LoadReuseVectorsForPath(ctx context.Context, collectionName string, relativePath string) (map[string][]float32, error) {
	f.mu.Lock()
	f.reusePathCalls = append(f.reusePathCalls, reusePathCall{Collection: collectionName, Path: relativePath})
	f.mu.Unlock()
	if f.loadReuseForPath != nil {
		return f.loadReuseForPath(ctx, collectionName, relativePath)
	}
	return map[string][]float32{}, nil
}

func (f *fakeSemantic) LoadReuseVectorsForContents(
	ctx context.Context,
	collectionName string,
	chunks []model.StoredChunk,
) (map[string][]float32, error) {
	if f.loadReuseForContents != nil {
		return f.loadReuseForContents(ctx, collectionName, chunks)
	}
	return map[string][]float32{}, nil
}

// reusePrefixCallsSnapshot returns a copy of the recorded prefix reuse loads.
func (f *fakeSemantic) reusePrefixCallsSnapshot() []reusePrefixCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]reusePrefixCall(nil), f.reusePrefixCalls...)
}

func (f *fakeSemantic) reusePathCallsSnapshot() []reusePathCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]reusePathCall(nil), f.reusePathCalls...)
}

func (f *fakeSemantic) Reindex(ctx context.Context, codebasePath string, chunks []model.StoredChunk, removal semantic.Removal, progress func(semantic.Progress), reuse map[string][]float32, columnSet semantic.StoreColumnSet) error {
	recordedRemoval := copyRemoval(removal)
	f.mu.Lock()
	f.reindexCalls = append(f.reindexCalls, reindexCall{CodebasePath: codebasePath, Chunks: len(chunks), Removed: removalPaths(recordedRemoval), Removal: recordedRemoval, ColumnSet: columnSet})
	f.mu.Unlock()
	if f.reindexWithReuse != nil {
		return f.reindexWithReuse(ctx, codebasePath, chunks, removalPaths(removal), progress, reuse)
	}
	if f.reindexEmit != nil && progress != nil {
		f.reindexEmit(progress)
	}
	if f.reindex != nil {
		return f.reindex(ctx, codebasePath, chunks, removalPaths(removal))
	}
	return nil
}

func (f *fakeSemantic) StageReindex(ctx context.Context, codebasePath string, chunks []model.StoredChunk, removal semantic.Removal, progress func(semantic.Progress), reuse map[string][]float32, columnSet semantic.StoreColumnSet) error {
	recordedRemoval := copyRemoval(removal)
	f.mu.Lock()
	f.stageCalls = append(f.stageCalls, reindexCall{CodebasePath: codebasePath, Chunks: len(chunks), Removed: removalPaths(recordedRemoval), Removal: recordedRemoval, ColumnSet: columnSet})
	f.mu.Unlock()
	if f.stageReindexWithReuse != nil {
		return f.stageReindexWithReuse(ctx, codebasePath, chunks, removalPaths(removal), progress, reuse)
	}
	if f.reindexEmit != nil && progress != nil {
		f.reindexEmit(progress)
	}
	return nil
}

func (f *fakeSemantic) PromoteStaging(ctx context.Context, codebasePath string) error {
	f.mu.Lock()
	f.promoted = append(f.promoted, codebasePath)
	f.mu.Unlock()
	if f.promoteStaging != nil {
		return f.promoteStaging(ctx, codebasePath)
	}
	return nil
}

// removalPaths flattens a removal into the legacy path list the converge tests
// assert on: exact paths first, then prefixes.
func removalPaths(removal semantic.Removal) []string {
	combined := make([]string, 0, len(removal.Paths)+len(removal.Prefixes))
	combined = append(combined, removal.Paths...)
	combined = append(combined, removal.Prefixes...)
	return combined
}

func copyRemoval(removal semantic.Removal) semantic.Removal {
	return semantic.Removal{
		Paths:    append([]string(nil), removal.Paths...),
		Prefixes: append([]string(nil), removal.Prefixes...),
	}
}

func (f *fakeSemantic) DeleteItemRows(ctx context.Context, collectionName string, removal semantic.Removal) error {
	if f.deleteItemRows != nil {
		return f.deleteItemRows(ctx, collectionName, removal)
	}
	return nil
}

// BackfillCollectionScalars reports no rows that need a backfill.
func (f *fakeSemantic) BackfillCollectionScalars(context.Context, string, semantic.ScalarBackfill) (int, int, error) {
	return 0, 0, nil
}

func (f *fakeSemantic) CopyChunks(ctx context.Context, codebasePath string, src string, dst string) (int, error) {
	if f.copyChunks != nil {
		return f.copyChunks(ctx, codebasePath, src, dst)
	}
	return 0, nil
}

func (f *fakeSemantic) PruneToCurrent(context.Context, string, []string) error { return nil }

func (f *fakeSemantic) EnsureMmapEnabledAllCollections(ctx context.Context) {
	if f.ensureMmap != nil {
		f.ensureMmap(ctx)
	}
}

func (f *fakeSemantic) Drop(_ context.Context, codebasePath string) error {
	f.mu.Lock()
	f.dropped = append(f.dropped, codebasePath)
	f.mu.Unlock()
	return nil
}

func (f *fakeSemantic) DropStaging(_ context.Context, codebasePath string) error {
	f.mu.Lock()
	f.droppedStaging = append(f.droppedStaging, codebasePath)
	f.mu.Unlock()
	return nil
}

// TestConvergeViaWatcherRunsCodebasesConcurrentlyUpToCap proves that several
// codebases converge at once up to the index-slot cap while another waits, that
// the shared lock is held while any converge runs, and that another holder can
// acquire it once all converges finish.
func TestConvergeViaWatcherRunsCodebasesConcurrentlyUpToCap(t *testing.T) {
	const cap = 2
	const codebases = 3

	manager, cfg := newTestManagerWithCap(t, cap)
	entered := make(chan struct{}, codebases)
	release := make(chan struct{})
	inFlight := atomic.Int32{}
	maxInFlight := atomic.Int32{}
	manager.semantic = &fakeSemantic{
		reindex: func(_ context.Context, _ string, _ []model.StoredChunk, _ []string) error {
			current := inFlight.Add(1)
			for {
				observed := maxInFlight.Load()
				if current <= observed || maxInFlight.CompareAndSwap(observed, current) {
					break
				}
			}
			entered <- struct{}{}
			<-release
			inFlight.Add(-1)
			return nil
		},
	}

	syncer := NewBackgroundSync(cfg, manager)
	syncer.queue = NewEventQueue(time.Hour, func(string, []string) {})

	ids := make([]string, 0, codebases)
	for i := range codebases {
		canonical := newCapTestRepo(t)
		id := fmt.Sprintf("cb-converge-%d", i)
		manager.mu.Lock()
		manager.codebases[id] = model.Codebase{
			ID:              id,
			CanonicalPath:   canonical,
			Status:          model.CodebaseStatusIndexed,
			EffectiveConfig: defaultIndexConfig(),
		}
		manager.mu.Unlock()
		ids = append(ids, id)
	}

	var wg sync.WaitGroup
	for _, id := range ids {
		wg.Add(1)
		go func(codebaseID string) {
			defer wg.Done()
			syncer.convergeViaWatcher(context.Background(), codebaseID, []string{"main.go"})
		}(id)
	}

	// Exactly cap converges embed before any slot frees.
	for range cap {
		<-entered
	}
	waitForCondition(t, func() bool { return inFlight.Load() == int32(cap) })
	if got := maxInFlight.Load(); got > int32(cap) {
		t.Fatalf("max concurrent converges = %d, want <= %d", got, cap)
	}

	// The kernel lock file is never unlinked, so its presence proves nothing. An
	// independent descriptor conflicts with the converge's own hold, which is what
	// proves the lock is still held while the embeds run.
	lockPath := filepath.Join(cfg.ContextRoot, "mcp-sync.flock")
	if !probeLockHeld(t, lockPath) {
		t.Fatal("sync lock should be held while converges run")
	}
	close(release)
	for i := cap; i < codebases; i++ {
		<-entered
	}
	wg.Wait()

	if got := maxInFlight.Load(); got > int32(cap) {
		t.Fatalf("max concurrent converges over the run = %d, want <= %d", got, cap)
	}

	if probeLockHeld(t, lockPath) {
		t.Fatal("sync lock should be released after converges finish")
	}
}

// TestWatcherUsesSchedulerWhenExternalLockBusy proves a converge releases its
// scheduler slot while an external process holds the sync lock, then resumes
// the same queued job after the lock becomes available.
func TestWatcherUsesSchedulerWhenExternalLockBusy(t *testing.T) {
	manager, cfg := newTestManagerWithCap(t, 2)
	var reindexCalls atomic.Int32
	manager.semantic = &fakeSemantic{
		reindex: func(_ context.Context, _ string, _ []model.StoredChunk, _ []string) error {
			reindexCalls.Add(1)
			return nil
		},
	}

	syncer := NewBackgroundSync(cfg, manager)
	syncer.queue = NewEventQueue(time.Hour, func(string, []string) {})

	canonical := newCapTestRepo(t)
	codebaseID := "cb-external-lock"
	manager.mu.Lock()
	manager.codebases[codebaseID] = model.Codebase{
		ID:              codebaseID,
		CanonicalPath:   canonical,
		Status:          model.CodebaseStatusIndexed,
		EffectiveConfig: defaultIndexConfig(),
	}
	manager.mu.Unlock()

	lockPath := filepath.Join(cfg.ContextRoot, "mcp-sync.flock")
	if err := os.MkdirAll(cfg.ContextRoot, 0o755); err != nil {
		t.Fatalf("MkdirAll returned error: %v", err)
	}
	holder, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		t.Fatalf("OpenFile returned error: %v", err)
	}
	t.Cleanup(func() {
		_ = holder.Close()
	})
	if err := syscall.Flock(int(holder.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		t.Fatalf("Flock returned error: %v", err)
	}

	convergeDone := make(chan struct{})
	go func() {
		defer close(convergeDone)
		syncer.convergeViaWatcher(context.Background(), codebaseID, []string{"main.go"})
	}()

	if got := reindexCalls.Load(); got != 0 {
		t.Fatalf("converge embedded %d time(s) while the lock was held externally, want 0", got)
	}
	waitForCondition(t, func() bool {
		snapshot := manager.jobScheduler.Snapshot()
		slots, _ := manager.IndexSlots()
		return snapshot.Paused[model.JobPriorityNormal] == 1 && slots == 0
	})
	if err := syscall.Flock(int(holder.Fd()), syscall.LOCK_UN); err != nil {
		t.Fatalf("unlock external holder: %v", err)
	}
	select {
	case <-convergeDone:
	case <-time.After(5 * time.Second):
		t.Fatal("watcher converge did not resume after the external lock released")
	}
	if got := reindexCalls.Load(); got != 1 {
		t.Fatalf("converge embedded %d time(s), want 1 after lock release", got)
	}
}

// TestConvergeCopyChunksFiresOnRename proves a renamed file converges through
// the CopyChunks fast path rather than a re-embed, incrementing
// converge_copy_chunks_total.
func TestConvergeCopyChunksFiresOnRename(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 2)
	var copyCalls atomic.Int32
	manager.semantic = &fakeSemantic{
		copyChunks: func(_ context.Context, _ string, _ string, _ string) (int, error) {
			copyCalls.Add(1)
			return 5, nil
		},
		reindex: func(_ context.Context, _ string, _ []model.StoredChunk, _ []string) error {
			return nil
		},
	}

	canonical := newCapTestRepo(t)
	if err := os.WriteFile(filepath.Join(canonical, "src.go"), []byte("package main\nfunc Moved() {}\n"), 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}

	cfg := defaultIndexConfig()
	cfg.IgnoreDigest = "sha256:rename-test"
	codebaseID := "cb-rename"
	manager.mu.Lock()
	manager.codebases[codebaseID] = model.Codebase{
		ID:              codebaseID,
		CanonicalPath:   canonical,
		Status:          model.CodebaseStatusIndexed,
		EffectiveConfig: cfg,
	}
	manager.mu.Unlock()

	// Seed a checkpoint recording src.go with its real content hash and inode,
	// so the renamed file is recognized as a move of src.go.
	captured, err := merkle.Capture(context.Background(), manager.indexability, codebaseID, canonical, cfg)
	if err != nil {
		t.Fatalf("Capture returned error: %v", err)
	}
	identity, err := statInode(filepath.Join(canonical, "src.go"))
	if err != nil {
		t.Fatalf("statInode returned error: %v", err)
	}
	checkpoint := merkle.Snapshot{
		ConfigDigest: cfg.IgnoreDigest,
		Files:        map[string]string{"src.go": captured.Files["src.go"]},
		Inodes:       map[string]merkle.InodeRef{"src.go": {Device: identity.device, Inode: identity.inode}},
	}
	if err := merkle.WriteSnapshot(manager.merklePath(codebaseID), checkpoint); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	// Rename on the same filesystem preserves the inode, which is what the fast
	// path keys on.
	if err := os.Rename(filepath.Join(canonical, "src.go"), filepath.Join(canonical, "dst.go")); err != nil {
		t.Fatalf("Rename returned error: %v", err)
	}

	before := metrics.Read().ConvergeCopyChunksTotal
	if _, err := manager.ConvergePaths(context.Background(), codebaseID, []string{"src.go", "dst.go"}, nil); err != nil {
		t.Fatalf("ConvergePaths returned error: %v", err)
	}

	if got := copyCalls.Load(); got != 1 {
		t.Fatalf("CopyChunks called %d time(s), want 1 (the rename fast path)", got)
	}
	if delta := metrics.Read().ConvergeCopyChunksTotal - before; delta != 1 {
		t.Fatalf("converge_copy_chunks_total moved by %d, want 1", delta)
	}

	snapshot, err := merkle.ReadSnapshot(manager.merklePath(codebaseID))
	if err != nil {
		t.Fatalf("ReadSnapshot returned error: %v", err)
	}
	if _, present := snapshot.Files["dst.go"]; !present {
		t.Fatalf("snapshot missing renamed destination dst.go; have %v", snapshot.Files)
	}
}

func waitForCompletedJobCount(t *testing.T, manager *Manager, want int) {
	t.Helper()
	waitForCondition(t, func() bool {
		jobs := manager.ListJobs("")
		if len(jobs) != want {
			return false
		}
		for _, job := range jobs {
			if job.State != model.JobStateCompleted {
				return false
			}
		}
		return true
	})
}

// TestCodeIndexCoalescesNonMatchingConfigAndDrains proves the code admission
// path: a non-matching-config request while a code job is active coalesces
// (returns the active job id, no conflict error), a matching-config request
// still dedups, and the pending config drains into a fresh sync after terminal.
func TestCodeIndexCoalescesNonMatchingConfigAndDrains(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	entered := make(chan struct{}, 1)
	release := make(chan struct{})
	var indexCalls atomic.Int32
	manager.runner = fakeRunner{
		index:      nil,
		indexFiles: nil,
		indexOne: func(_ context.Context, _ string, relativePath string, _ model.IndexConfig) (indexer.OneFileResult, error) {
			if indexCalls.Add(1) == 1 {
				entered <- struct{}{}
				<-release
			}
			content := "package main\n"
			return indexer.OneFileResult{
				Chunks:   []model.StoredChunk{{Content: content, RelativePath: relativePath, StartLine: 1, EndLine: 1, Language: "go", FileExtension: ".go"}},
				FileHash: hashText(content),
				Skipped:  false,
				Removed:  false,
			}, nil
		},
	}

	firstJob, _, deduplicated, _, err := manager.StartIndex(context.Background(), repoPath, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget)
	if err != nil {
		t.Fatalf("first StartIndex returned error: %v", err)
	}
	if deduplicated {
		t.Fatal("first request should not be a dedup hit")
	}
	<-entered // first job blocked inside the indexer, ActiveJobID set

	// A matching-config request still dedups onto the active job.
	dedupJob, _, dedupHit, _, err := manager.StartIndex(context.Background(), repoPath, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget)
	if err != nil {
		t.Fatalf("matching-config StartIndex returned error: %v", err)
	}
	if !dedupHit || dedupJob.ID != firstJob.ID {
		t.Fatalf("matching-config request returned (dedup=%v, job=%s), want dedup onto %s", dedupHit, dedupJob.ID, firstJob.ID)
	}

	// A non-matching-config request coalesces instead of refusing.
	conflictConfig := defaultIndexConfig()
	conflictConfig.SplitterType = "langchain"
	coalescedJob, _, coalescedHit, _, err := manager.StartIndex(context.Background(), repoPath, testClientInfo(), conflictConfig, false, emptyAdmissionBudget)
	if err != nil {
		t.Fatalf("non-matching-config StartIndex returned error, want coalesced success: %v", err)
	}
	if !coalescedHit || coalescedJob.ID != firstJob.ID {
		t.Fatalf("non-matching-config request returned (coalesced=%v, job=%s), want the active job %s", coalescedHit, coalescedJob.ID, firstJob.ID)
	}

	codebase, _, found, _, err := manager.GetIndex(context.Background(), repoPath)
	if err != nil || !found {
		t.Fatalf("GetIndex returned err=%v found=%v", err, found)
	}
	manager.mu.Lock()
	pendingCode, ok := manager.pendingCodeJobs[codebase.ID]
	manager.mu.Unlock()
	if !ok {
		t.Fatal("pending code slot empty after non-matching coalesce")
	}
	if pendingCode.indexConfig.SplitterType != "langchain" {
		t.Fatalf("pending code config SplitterType = %q, want langchain", pendingCode.indexConfig.SplitterType)
	}

	close(release)
	// The drained sync runs as a second job carrying the coalesced langchain config.
	waitForCompletedJobCount(t, manager, 2)
	drainedFound := false
	for _, job := range manager.ListJobs("") {
		if job.ID != firstJob.ID {
			if job.Config.SplitterType != "langchain" {
				t.Fatalf("drained job config SplitterType = %q, want langchain", job.Config.SplitterType)
			}
			drainedFound = true
		}
	}
	if !drainedFound {
		t.Fatal("no drained successor job found after terminal")
	}
}

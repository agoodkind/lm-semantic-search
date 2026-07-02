package daemon

import (
	"context"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

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
	unavailable           bool
	probeErr              error
	reindex               func(ctx context.Context, codebasePath string, chunks []model.StoredChunk, removed []string) error
	reindexWithReuse      func(ctx context.Context, codebasePath string, chunks []model.StoredChunk, removed []string, progress func(semantic.Progress), reuse map[string][]float32) error
	stageReindexWithReuse func(ctx context.Context, codebasePath string, chunks []model.StoredChunk, removed []string, progress func(semantic.Progress), reuse map[string][]float32) error
	copyChunks            func(ctx context.Context, codebasePath string, src string, dst string) (int, error)
	deleteConversation    func(ctx context.Context, collectionName string, conversationID string) error
	backfillConversations func(ctx context.Context, collectionName string, enrichment semantic.ConversationEnrichment, dryRun bool) (int, int, error)
	collectionName        func(codebasePath string) string
	conversationName      func(collectionID string) string
	inspectCollection     func(context.Context, string) (semantic.CollectionFacts, error)
	listCollections       func(context.Context) ([]string, error)
	hasCollectionForPath  func(context.Context, string) (bool, error)
	collectionState       func(context.Context, string) (bool, bool, error)
	hasStaging            func(context.Context, string) (bool, error)
	search                func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error)
	conversationSearch    func(context.Context, string, string, int32) ([]model.StoredChunk, error)
	count                 func(context.Context, string) (int32, error)
	// loadReuse, when set, supplies the reuse map a merge-down build receives and
	// records which collections were asked for. dropped records every Drop call
	// so a test can prove an absorb never drops the absorbed child collection.
	loadReuse        func(ctx context.Context, collectionNames []string) (map[string][]float32, error)
	reuseCollections [][]string
	// loadReuseForPrefix, when set, supplies the per-conversation reuse map a
	// conversation ingest loads by conv/<id>/ prefix; reusePrefixCalls records
	// every such load and reindexReuse records the reuse map each conversation's
	// Reindex actually received.
	loadReuseForPrefix func(ctx context.Context, collectionName string, relativePathPrefix string) (map[string][]float32, error)
	reusePrefixCalls   []reusePrefixCall
	loadReuseForPath   func(ctx context.Context, collectionName string, relativePath string) (map[string][]float32, error)
	reusePathCalls     []reusePathCall
	loadMessageState   func(ctx context.Context, collectionName string, conversationPrefix string) (map[int32]semantic.StoredMessageState, map[string][]float32, error)
	messageStateCalls  []messageStateCall
	reindexReuse       map[string]map[string][]float32
	// conversationSearchScopes records the conversation-id scope each
	// conversation search received, so tests can prove native scoping.
	conversationSearchScopes [][]string
	dropped                  []string
	droppedStaging           []string
	reindexCalls             []reindexCall
	stageCalls               []reindexCall
	promoted                 []string
	// reindexEmit, when set, is invoked with the live progress callback during
	// Reindex and StageReindex so a test can drive reuse-vs-embed progress
	// reporting, including a conversation ingest's batch progress.
	reindexEmit func(progress func(semantic.Progress))
	mu          sync.Mutex
}

type reindexCall struct {
	CodebasePath string
	Chunks       int
	Removed      []string
	Removal      semantic.Removal
}

func (f *fakeSemantic) Available() bool { return !f.unavailable }
func (f *fakeSemantic) ProbeHealth(context.Context) error {
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

func (f *fakeSemantic) ConversationCollectionName(collectionID string) string {
	if f.conversationName != nil {
		return f.conversationName(collectionID)
	}
	return "conv_chunks_" + tshash.PathPrefix(collectionID)
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

func (f *fakeSemantic) SearchConversationCollectionCapped(ctx context.Context, collectionName string, query string, limit int32, _ int32, _ float64, filter semantic.ConversationFilter) ([]model.StoredChunk, error) {
	f.mu.Lock()
	f.conversationSearchScopes = append(f.conversationSearchScopes, append([]string(nil), filter.ConversationIDs...))
	f.mu.Unlock()
	if f.conversationSearch != nil {
		return f.conversationSearch(ctx, collectionName, query, limit)
	}
	return nil, nil
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
type messageStateCall struct {
	Collection string
	Prefix     string
}

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

func (f *fakeSemantic) LoadConversationMessageState(ctx context.Context, collectionName string, conversationPrefix string) (map[int32]semantic.StoredMessageState, map[string][]float32, error) {
	f.mu.Lock()
	f.messageStateCalls = append(f.messageStateCalls, messageStateCall{Collection: collectionName, Prefix: conversationPrefix})
	f.mu.Unlock()
	if f.loadMessageState != nil {
		return f.loadMessageState(ctx, collectionName, conversationPrefix)
	}
	return map[int32]semantic.StoredMessageState{}, map[string][]float32{}, nil
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

func (f *fakeSemantic) messageStateCallsSnapshot() []messageStateCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]messageStateCall(nil), f.messageStateCalls...)
}

// reindexReuseSnapshot returns, per conversation id, a copy of the reuse map
// the last Reindex for that conversation received.
func (f *fakeSemantic) reindexReuseSnapshot() map[string]map[string][]float32 {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make(map[string]map[string][]float32, len(f.reindexReuse))
	for conversationID, reuse := range f.reindexReuse {
		copied := make(map[string][]float32, len(reuse))
		maps.Copy(copied, reuse)
		out[conversationID] = copied
	}
	return out
}

// recordReindexReuse stores the reuse map a Reindex call carried, keyed by the
// conversation id of its first chunk, so conversation tests can assert which
// reuse map reached which conversation's reindex.
func (f *fakeSemantic) recordReindexReuse(chunks []model.StoredChunk, reuse map[string][]float32) {
	if len(chunks) == 0 || chunks[0].ConversationID == "" {
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.reindexReuse == nil {
		f.reindexReuse = make(map[string]map[string][]float32)
	}
	copied := make(map[string][]float32, len(reuse))
	maps.Copy(copied, reuse)
	f.reindexReuse[chunks[0].ConversationID] = copied
}

func (f *fakeSemantic) Reindex(ctx context.Context, codebasePath string, chunks []model.StoredChunk, removal semantic.Removal, progress func(semantic.Progress), reuse map[string][]float32) error {
	recordedRemoval := copyRemoval(removal)
	f.mu.Lock()
	f.reindexCalls = append(f.reindexCalls, reindexCall{CodebasePath: codebasePath, Chunks: len(chunks), Removed: removalPaths(recordedRemoval), Removal: recordedRemoval})
	f.mu.Unlock()
	f.recordReindexReuse(chunks, reuse)
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

func (f *fakeSemantic) StageReindex(ctx context.Context, codebasePath string, chunks []model.StoredChunk, removal semantic.Removal, progress func(semantic.Progress), reuse map[string][]float32) error {
	recordedRemoval := copyRemoval(removal)
	f.mu.Lock()
	f.stageCalls = append(f.stageCalls, reindexCall{CodebasePath: codebasePath, Chunks: len(chunks), Removed: removalPaths(recordedRemoval), Removal: recordedRemoval})
	f.mu.Unlock()
	f.recordReindexReuse(chunks, reuse)
	if f.stageReindexWithReuse != nil {
		return f.stageReindexWithReuse(ctx, codebasePath, chunks, removalPaths(removal), progress, reuse)
	}
	if f.reindexEmit != nil && progress != nil {
		f.reindexEmit(progress)
	}
	return nil
}

func (f *fakeSemantic) PromoteStaging(_ context.Context, codebasePath string) error {
	f.mu.Lock()
	f.promoted = append(f.promoted, codebasePath)
	f.mu.Unlock()
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

func (f *fakeSemantic) DeleteConversation(ctx context.Context, collectionName string, conversationID string) error {
	if f.deleteConversation != nil {
		return f.deleteConversation(ctx, collectionName, conversationID)
	}
	return nil
}

func (f *fakeSemantic) BackfillConversationEnrichment(ctx context.Context, collectionName string, enrichment semantic.ConversationEnrichment, dryRun bool) (int, int, error) {
	if f.backfillConversations != nil {
		return f.backfillConversations(ctx, collectionName, enrichment, dryRun)
	}
	return 0, 0, nil
}

func (f *fakeSemantic) CopyChunks(ctx context.Context, codebasePath string, src string, dst string) (int, error) {
	if f.copyChunks != nil {
		return f.copyChunks(ctx, codebasePath, src, dst)
	}
	return 0, nil
}

func (f *fakeSemantic) PruneToCurrent(context.Context, string, []string) error { return nil }

func (f *fakeSemantic) EnsureMmapEnabledAllCollections(context.Context) {}

func (f *fakeSemantic) BackfillConversationCollectionsOnce(context.Context) {}

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
// the shared lock exists while any converge runs, and that it is gone once all
// converges finish.
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

	lockPath := filepath.Join(cfg.ContextRoot, "mcp-sync.lock")
	if _, err := os.Stat(lockPath); err != nil {
		t.Fatalf("sync lock should exist while converges run: %v", err)
	}

	close(release)
	for i := cap; i < codebases; i++ {
		<-entered
	}
	wg.Wait()

	if got := maxInFlight.Load(); got > int32(cap) {
		t.Fatalf("max concurrent converges over the run = %d, want <= %d", got, cap)
	}
	waitForCondition(t, func() bool {
		_, err := os.Stat(lockPath)
		return os.IsNotExist(err)
	})
}

// TestConvergeViaWatcherDefersToExternalLock proves a converge yields when the
// shared lock is held externally (a fresh directory with no daemon owner
// marker, as the upstream TS adapter leaves it) and requeues its paths instead
// of embedding.
func TestConvergeViaWatcherDefersToExternalLock(t *testing.T) {
	manager, cfg := newTestManagerWithCap(t, 2)
	var reindexCalls atomic.Int32
	manager.semantic = &fakeSemantic{
		reindex: func(_ context.Context, _ string, _ []model.StoredChunk, _ []string) error {
			reindexCalls.Add(1)
			return nil
		},
	}

	var requeued atomic.Int32
	syncer := NewBackgroundSync(cfg, manager)
	// A short debounce lets the requeued path drain promptly; the drain just
	// records the requeue rather than re-running the converge.
	syncer.queue = NewEventQueue(20*time.Millisecond, func(string, []string) { requeued.Add(1) })

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

	// An external holder: a fresh lock directory with no owner marker.
	lockPath := filepath.Join(cfg.ContextRoot, "mcp-sync.lock")
	if err := os.MkdirAll(cfg.ContextRoot, 0o755); err != nil {
		t.Fatalf("MkdirAll returned error: %v", err)
	}
	if err := os.Mkdir(lockPath, 0o755); err != nil {
		t.Fatalf("Mkdir returned error: %v", err)
	}

	syncer.convergeViaWatcher(context.Background(), codebaseID, []string{"main.go"})

	if got := reindexCalls.Load(); got != 0 {
		t.Fatalf("converge embedded %d time(s) while the lock was held externally, want 0", got)
	}
	waitForCondition(t, func() bool { return requeued.Load() >= 1 })
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
	if err := manager.ConvergePaths(context.Background(), codebaseID, []string{"src.go", "dst.go"}); err != nil {
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

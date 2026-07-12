package daemon

import (
	"context"
	"slices"
	"sync"
	"testing"

	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// indexOneSpySource wraps a conversationItemSource and counts indexOne calls so a
// test can prove chunk regeneration runs only for the pruned work set. It embeds
// the real source by value, sharing its single-flight batch pointer, so the cheap
// classifier and the per-item loop behave exactly as in production. A pointer
// receiver on indexOne and a pointer itemSource keep the call log shared across
// the value copies the delta routine makes of deltaState.
type indexOneSpySource struct {
	conversationItemSource
	mu    sync.Mutex
	calls []string
}

func (source *indexOneSpySource) indexOne(ctx context.Context, itemID string) (indexer.OneFileResult, error) {
	source.mu.Lock()
	source.calls = append(source.calls, itemID)
	source.mu.Unlock()
	return source.conversationItemSource.indexOne(ctx, itemID)
}

func (source *indexOneSpySource) indexOneCalls() []string {
	source.mu.Lock()
	defer source.mu.Unlock()
	sorted := append([]string(nil), source.calls...)
	slices.Sort(sorted)
	return sorted
}

// TestForcedWorkSetPrunesNoOpsBeforeDenominatorAndRegen proves the core
// chokepoint contract: of M delivered conversations, all but K are already fully
// present, so the up-front classifier prunes the M-K no-ops before the diff union
// fixes the denominator and before the per-item loop. The changed set and the
// progress FilesTotal equal K, only the K conversations reach indexOne, and the
// single-flight batch is read exactly once for the whole run.
//
// A regeneration counter locks the stronger no-regen invariant directly: it wraps
// conversationDocumentsToStoredChunks, the single derived-chunk generation entry
// point, so the count catches a regression where forcedWorkSet itself calls it,
// which the indexOne spy alone would miss. The count must be 0 after the up-front
// classification (the classifier regenerates nothing) and exactly K after the
// per-item loop (regeneration happens only inside the loop for the pruned work
// set).
func TestForcedWorkSetPrunesNoOpsBeforeDenominatorAndRegen(t *testing.T) {
	manager, _, _ := newTestManager(t)

	// Wrap the single chunk-generation entry point with a counter, so the test
	// observes exactly how many times chunks are regenerated and at which phase.
	// The test is non-parallel, so no parallel test runs concurrently while the
	// package-level var is swapped, and the defer restores it.
	var regenMu sync.Mutex
	regenCalls := 0
	realGenerate := conversationDocumentsToStoredChunks
	conversationDocumentsToStoredChunks = func(ctx context.Context, documents []model.ConversationDocument) ([]model.StoredChunk, error) {
		regenMu.Lock()
		regenCalls++
		regenMu.Unlock()
		return realGenerate(ctx, documents)
	}
	defer func() { conversationDocumentsToStoredChunks = realGenerate }()
	regenCount := func() int {
		regenMu.Lock()
		defer regenMu.Unlock()
		return regenCalls
	}

	const collectionName = "conv_chunks_prune"
	codebaseID := "cb-prune"

	// M = 4 delivered conversations, each an assistant message carrying thinking,
	// so each expects a convthink/<id>/0 derived row. Only "codex:d" is missing
	// its derived row, so K = 1.
	docs := []model.ConversationDocument{
		{ConversationID: "claude:a", MessageIndex: 0, Role: "assistant", Text: "a", Thinking: "ra"},
		{ConversationID: "claude:b", MessageIndex: 0, Role: "assistant", Text: "b", Thinking: "rb"},
		{ConversationID: "plain:c", MessageIndex: 0, Role: "user", Text: "c"},
		{ConversationID: "codex:d", MessageIndex: 0, Role: "assistant", Text: "d", Thinking: "rd"},
	}
	manifest := map[string]string{"claude:a": "fp1", "claude:b": "fp2", "plain:c": "fp3", "codex:d": "fp4"}

	present := func(conversationID string) semantic.ConversationStoredRows {
		return semantic.ConversationStoredRows{
			Messages:     map[int32]semantic.StoredMessageState{},
			DerivedPaths: map[string]string{"convthink/" + conversationID + "/0": "h"},
		}
	}
	fake := &fakeSemantic{
		collectionName: func(string) string { return collectionName },
		loadDerivedBatch: func(_ context.Context, _ string, _ []string) (semantic.ConversationBatchState, error) {
			return semantic.ConversationBatchState{
				Rows: map[string]semantic.ConversationStoredRows{
					"claude:a": present("claude:a"),
					"claude:b": present("claude:b"),
					// plain:c expects no derived rows (no tools, no thinking).
					"plain:c": {Messages: map[int32]semantic.StoredMessageState{}, DerivedPaths: map[string]string{}},
					// codex:d is missing its expected thinking row -> real work.
					"codex:d": {Messages: map[int32]semantic.StoredMessageState{}, DerivedPaths: map[string]string{}},
				},
				Reuse: map[string][]float32{},
			}, nil
		},
		inspectCollection: func(_ context.Context, _ string) (semantic.CollectionFacts, error) {
			return semantic.CollectionFacts{Exists: true, Rows: 8, RowsKnown: true}, nil
		},
	}
	manager.semantic = fake

	config := model.IndexConfig{IgnoreDigest: "prune-digest"}
	manager.mu.Lock()
	manager.codebases[codebaseID] = model.Codebase{
		ID:              codebaseID,
		CanonicalPath:   collectionName,
		Status:          model.CodebaseStatusIndexed,
		EffectiveConfig: config,
		CollectionName:  collectionName,
	}
	manager.mu.Unlock()

	// Seed the checkpoint with the same fingerprints, so the merkle diff is empty
	// and only the up-front classifier contributes the changed set.
	seed := merkle.Snapshot{ConfigDigest: config.IgnoreDigest, Files: map[string]string{}, Inodes: nil}
	for conversationID, fingerprint := range manifest {
		seed.Files[conversationID] = fingerprint
	}
	if err := merkle.WriteSnapshot(manager.merklePath(codebaseID), seed); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	job := model.Job{
		ID:            "job-prune",
		CodebaseID:    codebaseID,
		CanonicalPath: collectionName,
		Config:        config,
		State:         model.JobStateRunning,
	}
	manager.mu.Lock()
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	base := newConversationItemSource(collectionName, manifest, docs, fake, absenceRetain, true, false)
	spy := &indexOneSpySource{conversationItemSource: base}

	plan := manager.planSyncDiff(context.Background(), job, codebaseID, spy)
	if plan.handled || plan.fallback {
		t.Fatalf("plan handled=%v fallback=%v, want a normal non-empty plan", plan.handled, plan.fallback)
	}
	// The up-front classifier (forcedWorkSet) ran inside planSyncDiff. It must have
	// regenerated zero chunks: presence classification reads store keys only.
	if got := regenCount(); got != 0 {
		t.Fatalf("chunk regenerations after up-front classification = %d, want 0 (the classifier regenerates nothing)", got)
	}
	changed := append(append([]string(nil), plan.diff.Added...), plan.diff.Modified...)
	slices.Sort(changed)
	if !slices.Equal(changed, []string{"codex:d"}) {
		t.Fatalf("changed set = %v, want [codex:d] (K=1, the only conversation with missing work)", changed)
	}
	if !slices.Equal(plan.forced, []string{"codex:d"}) {
		t.Fatalf("plan.forced = %v, want [codex:d]", plan.forced)
	}

	state := deltaState{
		plan:         plan,
		snapshotPath: manager.merklePath(codebaseID),
		working:      map[string]string{},
		source:       spy,
		semantic:     true,
		chunkCounts:  &chunkCounters{},
		forced:       forcedItemsSet(plan.forced),
	}
	for conversationID, fingerprint := range seed.Files {
		state.working[conversationID] = fingerprint
	}

	_, outcome := manager.applyDeltaChanges(context.Background(), job, state)
	if outcome.fallback || outcome.handled {
		t.Fatalf("applyDeltaChanges outcome = %+v, want normal completion", outcome)
	}

	if calls := spy.indexOneCalls(); !slices.Equal(calls, []string{"codex:d"}) {
		t.Fatalf("indexOne calls = %v, want only [codex:d] (no regeneration for the 3 pruned no-ops)", calls)
	}
	// Regeneration ran exactly K=1 time, inside the per-item loop for the single
	// work item. The 3 pruned no-ops regenerated nothing, so the total is K, not M.
	if got := regenCount(); got != 1 {
		t.Fatalf("chunk regenerations after the per-item loop = %d, want 1 (only the K=1 work item regenerates, inside the loop)", got)
	}
	if batches := fake.derivedBatchCallsSnapshot(); len(batches) != 1 {
		t.Fatalf("derived batch reads = %d, want exactly one single-flight read", len(batches))
	}

	manager.mu.Lock()
	filesTotal := manager.jobs[job.ID].Progress.FilesTotal
	manager.mu.Unlock()
	if filesTotal != 1 {
		t.Fatalf("FilesTotal = %d, want 1 (denominator equals real pending work, not the 4 delivered)", filesTotal)
	}

	// Second run: everything is now present, so the classifier prunes all of them
	// and the pruned work set is empty -> the changed set is empty, an instant
	// no-op that regenerates nothing.
	fake.loadDerivedBatch = func(_ context.Context, _ string, _ []string) (semantic.ConversationBatchState, error) {
		return semantic.ConversationBatchState{
			Rows: map[string]semantic.ConversationStoredRows{
				"claude:a": present("claude:a"),
				"claude:b": present("claude:b"),
				"plain:c":  {Messages: map[int32]semantic.StoredMessageState{}, DerivedPaths: map[string]string{}},
				"codex:d":  present("codex:d"),
			},
			Reuse: map[string][]float32{},
		}, nil
	}
	secondBase := newConversationItemSource(collectionName, manifest, docs, fake, absenceRetain, true, false)
	captured, err := secondBase.capture(context.Background())
	if err != nil {
		t.Fatalf("second capture returned error: %v", err)
	}
	forcedSecond, err := secondBase.forcedWorkSet(context.Background())
	if err != nil {
		t.Fatalf("second forcedWorkSet returned error: %v", err)
	}
	secondDiff := unionForcedItems(merkle.DiffSnapshots(seed, captured), forcedSecond, captured)
	if !secondDiff.Empty() {
		t.Fatalf("second run changed set = %+v, want empty (instant no-op)", secondDiff)
	}
}

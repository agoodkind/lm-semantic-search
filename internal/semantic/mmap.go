package semantic

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

// mmap storage moves a field or index off the query node's heap onto
// memory-mapped, disk-backed files. We enable it for the dense vector field and
// its index on every collection alike, which is where almost all of the
// per-collection resident memory sits, while leaving the scalar fields, scalar
// indexes, and the sparse BM25 index resident so native filtering and keyword
// lookup stay fast. The rule is uniform across conversation and code
// collections; nothing here branches on collection content.
const (
	mmapEnabledKey   = "mmap.enabled"
	mmapEnabledValue = "true"
	// releasePoll* bound the wait for a release to take effect. Milvus
	// ReleaseCollection returns as soon as the request is accepted, but the unload
	// is asynchronous, and AlterCollectionFieldProperty is rejected while the
	// collection still reports loaded, so the migration polls GetLoadState until
	// the collection is NotLoad before it alters.
	releasePollInterval = 250 * time.Millisecond
	releasePollTimeout  = 90 * time.Second
)

// mmapOutcome is the per-collection result of an mmap-enable attempt, used to
// build the end-of-sweep summary. The zero value is the unknown outcome so an
// error path never reports as a real result.
type mmapOutcome int

const (
	mmapOutcomeUnknown mmapOutcome = iota
	mmapOutcomeMigrated
	mmapOutcomeAlready
	mmapOutcomeSkipped
)

// releaseCollection unloads collectionName from the query node's memory.
// Enabling mmap on an already-loaded collection requires releasing it first,
// because the in-memory layout of a loaded collection is fixed until it is
// released and loaded again.
func (service *Service) releaseCollection(ctx context.Context, collectionName string) error {
	if err := service.milvus.ReleaseCollection(ctx, milvusclient.NewReleaseCollectionOption(collectionName)); err != nil {
		return wrapStoreError(ctx, err, "release Milvus collection "+collectionName)
	}
	return nil
}

// collectionLoaded reports whether collectionName is loaded, read from the
// authoritative GetLoadState RPC. DescribeCollection's Loaded field is not
// reliably populated, so the alter/release decision must not key off it.
func (service *Service) collectionLoaded(ctx context.Context, collectionName string) (bool, error) {
	state, err := service.milvus.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(collectionName))
	if err != nil {
		return false, wrapStoreError(ctx, err, "get load state for "+collectionName)
	}
	return state.State == entity.LoadStateLoaded, nil
}

// awaitCollectionReleased polls GetLoadState until collectionName reports
// NotLoad, because ReleaseCollection returns before the unload completes and the
// subsequent property alter is rejected while the collection still reports
// loaded.
func (service *Service) awaitCollectionReleased(ctx context.Context, collectionName string) error {
	pollCtx, cancel := context.WithTimeout(ctx, releasePollTimeout)
	defer cancel()
	for {
		state, err := service.milvus.GetLoadState(pollCtx, milvusclient.NewGetLoadStateOption(collectionName))
		if err != nil {
			return wrapStoreError(ctx, err, "poll release state for "+collectionName)
		}
		if state.State == entity.LoadStateNotLoad {
			return nil
		}
		select {
		case <-pollCtx.Done():
			slog.ErrorContext(ctx, "await collection release timed out", "collection", collectionName, "last_state", int(state.State), "err", pollCtx.Err())
			return fmt.Errorf("await release of %s: %w", collectionName, pollCtx.Err())
		case <-time.After(releasePollInterval):
		}
	}
}

// ensureMmapEnabledOnCollection enables mmap on collectionName's dense vector
// field and dense index in place, preserving the existing vectors with no
// re-embedding. It is idempotent: a collection whose dense index already reports
// mmap is left untouched (a describe-only no-op, no release/reload). A collection
// with no dense index is skipped, never treated as an error, so an odd or
// half-built collection does not break the sweep; a genuine list/describe RPC
// fault is returned so the caller can surface it rather than swallow it as a
// skip. The describe gate sits ahead of the release, so a steady-state run never
// unloads a collection it has already migrated, and the release before the alter
// is gated on the collection's load state, so a collection that came up unloaded
// skips straight to the alter and a single load.
func (service *Service) ensureMmapEnabledOnCollection(ctx context.Context, collectionName string) (mmapOutcome, error) {
	hasCollection, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		return mmapOutcomeUnknown, wrapStoreError(ctx, err, "check collection "+collectionName+" for mmap")
	}
	if !hasCollection {
		return mmapOutcomeSkipped, nil
	}

	indexNames, err := service.milvus.ListIndexes(ctx, milvusclient.NewListIndexOption(collectionName).WithFieldName(denseVectorFieldName))
	if err != nil {
		return mmapOutcomeUnknown, wrapStoreError(ctx, err, "list dense indexes for "+collectionName)
	}
	if len(indexNames) == 0 {
		slog.DebugContext(ctx, "semantic.mmap_skip_no_dense_index", "collection", collectionName)
		return mmapOutcomeSkipped, nil
	}
	indexName := indexNames[0]

	indexDesc, err := service.milvus.DescribeIndex(ctx, milvusclient.NewDescribeIndexOption(collectionName, indexName))
	if err != nil {
		return mmapOutcomeUnknown, wrapStoreError(ctx, err, "describe index "+indexName+" on "+collectionName)
	}
	alreadyMmapped := indexDesc.Params()[mmapEnabledKey] == mmapEnabledValue

	loaded, err := service.collectionLoaded(ctx, collectionName)
	if err != nil {
		return mmapOutcomeUnknown, err
	}

	if alreadyMmapped {
		// Already migrated. Normally the collection is loaded (Milvus restores
		// load state on restart), so this is a no-op. Only load it if it somehow
		// came up released.
		if loaded {
			return mmapOutcomeAlready, nil
		}
		if err := service.loadCollection(ctx, collectionName); err != nil {
			return mmapOutcomeUnknown, err
		}
		return mmapOutcomeAlready, nil
	}

	// Altering mmap requires a released collection, and ReleaseCollection returns
	// before the unload completes, so release and then wait until the collection
	// actually reports NotLoad before the alter, otherwise the alter is rejected
	// with "collection already loaded".
	if loaded {
		if err := service.releaseCollection(ctx, collectionName); err != nil {
			return mmapOutcomeUnknown, err
		}
		if err := service.awaitCollectionReleased(ctx, collectionName); err != nil {
			return mmapOutcomeUnknown, err
		}
	}
	if err := service.milvus.AlterCollectionFieldProperty(ctx, milvusclient.NewAlterCollectionFieldPropertiesOption(collectionName, denseVectorFieldName).WithProperty(mmapEnabledKey, mmapEnabledValue)); err != nil {
		return mmapOutcomeUnknown, wrapStoreError(ctx, err, "enable mmap on dense field of "+collectionName)
	}
	if err := service.milvus.AlterIndexProperties(ctx, milvusclient.NewAlterIndexPropertiesOption(collectionName, indexName).WithProperty(mmapEnabledKey, mmapEnabledValue)); err != nil {
		return mmapOutcomeUnknown, wrapStoreError(ctx, err, "enable mmap on dense index "+indexName+" of "+collectionName)
	}
	if err := service.loadCollection(ctx, collectionName); err != nil {
		return mmapOutcomeUnknown, err
	}
	slog.InfoContext(ctx, "semantic.mmap_enabled", "collection", collectionName, "index", indexName, "was_loaded", loaded)
	return mmapOutcomeMigrated, nil
}

// ensureMmapEnabledOnce runs the dense mmap migration at most once per collection
// per process. The per-process guard means exactly "confirmed mmap-migrated":
// it is recorded only when mmap is verified enabled (a fresh migration or an
// already-mmapped collection), never on an error and never on a no-index skip.
// So a transient fault retries on the next sweep, and a collection that was still
// mid-staging (no dense index yet) stays eligible until its index exists.
func (service *Service) ensureMmapEnabledOnce(ctx context.Context, collectionName string) (mmapOutcome, error) {
	if _, done := service.ensuredMmapEnabled.Load(collectionName); done {
		return mmapOutcomeAlready, nil
	}
	outcome, err := service.ensureMmapEnabledOnCollection(ctx, collectionName)
	if err != nil {
		return mmapOutcomeUnknown, err
	}
	if outcome == mmapOutcomeMigrated || outcome == mmapOutcomeAlready {
		service.ensuredMmapEnabled.Store(collectionName, struct{}{})
	}
	return outcome, nil
}

// EnsureMmapEnabledAllCollections sweeps every collection and enables dense mmap
// on each, uniformly and content-agnostically. The daemon's periodic sync loop
// drives it on the startup tick and every interval tick, which gives the
// migration convergence (a failed collection retries next tick), self-heal on a
// degraded boot (the first tick no-ops while Milvus is down, a later tick catches
// it coming up), and coverage of collections created after the first sweep, all
// without owning any scheduling of its own. After the first fully successful
// sweep, later ticks are near-free: every migrated collection is a guard hit with
// no RPC. It is fault-tolerant per collection: one failure is counted and the
// sweep continues, and a non-zero failed count is surfaced in the end-of-sweep
// summary so a broken sweep cannot look healthy.
func (service *Service) EnsureMmapEnabledAllCollections(ctx context.Context) {
	if !service.Available() {
		return
	}
	collections, err := service.ListCollections(ctx)
	if err != nil {
		slog.ErrorContext(ctx, "semantic.mmap_sweep_list_failed", "err", err)
		return
	}

	migrated := 0
	already := 0
	skipped := 0
	failed := 0
	for _, collectionName := range collections {
		// ensureMmapEnabledOnce logs the specific failing operation at its source;
		// here we only count it so one collection's failure never blocks the rest
		// and the end-of-sweep summary still surfaces a non-zero failed count.
		outcome, err := service.ensureMmapEnabledOnce(ctx, collectionName)
		if err != nil {
			failed++
			continue
		}
		switch outcome {
		case mmapOutcomeMigrated:
			migrated++
		case mmapOutcomeAlready:
			already++
		case mmapOutcomeSkipped:
			skipped++
		case mmapOutcomeUnknown:
		}
	}

	attrs := []slog.Attr{
		slog.Int("total", len(collections)),
		slog.Int("migrated", migrated),
		slog.Int("already_mmapped", already),
		slog.Int("skipped_no_index", skipped),
		slog.Int("failed", failed),
	}
	// Keep the steady state quiet: a tick that changed nothing logs at debug, a
	// tick that migrated something logs at info, and any failure logs at warn so
	// a persistently-failing collection stays visible across the retry cadence.
	level := slog.LevelDebug
	message := "semantic.mmap_sweep_complete"
	switch {
	case failed > 0:
		level = slog.LevelWarn
		message = "semantic.mmap_sweep_complete_with_failures"
	case migrated > 0:
		level = slog.LevelInfo
	}
	slog.LogAttrs(ctx, level, message, attrs...)
}

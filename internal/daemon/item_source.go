package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"sort"

	"goodkind.io/lm-semantic-search/internal/indexability"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// forcedItemsSet collects the precomputed forced item ids into a set for O(1)
// lookup in applyDeltaChanges. The classification runs once up front in
// planSyncDiff (forcedWorkSet), so this takes the resulting slice rather than
// re-asking the source. It returns nil when nothing is forced (the normal sync),
// so the hash-equality skip stays in force for every item.
func forcedItemsSet(forced []string) map[string]struct{} {
	if len(forced) == 0 {
		return nil
	}
	set := make(map[string]struct{}, len(forced))
	for _, itemID := range forced {
		set[itemID] = struct{}{}
	}
	return set
}

// unionForcedItems folds a source's forced item ids into the diff's Modified
// set, so the delta routine re-examines them even when their captured
// fingerprint matches the stored checkpoint. Only ids present in the current
// capture are added, and ids already classified as Added or Modified are left
// as-is, so forcing is idempotent and never invents an item the source did not
// deliver. indexOne then re-runs its normal content diff, reusing unchanged
// chunks and re-stamping the delivered fingerprint, so a forced item leaves the
// committed checkpoint unchanged when its content is unchanged. A code source
// forces nothing, so this is a no-op for filesystem syncs.
func unionForcedItems(diff merkle.Diff, forced []string, captured merkle.Snapshot) merkle.Diff {
	if len(forced) == 0 {
		return diff
	}
	already := make(map[string]struct{}, len(diff.Added)+len(diff.Modified))
	for _, itemID := range diff.Added {
		already[itemID] = struct{}{}
	}
	for _, itemID := range diff.Modified {
		already[itemID] = struct{}{}
	}
	added := false
	for _, itemID := range forced {
		if _, present := captured.Files[itemID]; !present {
			continue
		}
		if _, seen := already[itemID]; seen {
			continue
		}
		already[itemID] = struct{}{}
		diff.Modified = append(diff.Modified, itemID)
		added = true
	}
	if added {
		sort.Strings(diff.Modified)
	}
	return diff
}

// itemSource is the one part of the indexing routine that differs by kind. The
// shared delta and bootstrap routine asks a source to list the current items
// with a content fingerprint each, to produce one item's chunks on request, to
// name the store rows that drop when an item changes or leaves, and to name the
// progress unit. A code source walks the filesystem and reads files; a
// conversation source reads the manifest and documents the daemon was handed.
type itemSource interface {
	// capture lists the current items as itemID -> content fingerprint.
	capture(ctx context.Context) (merkle.Snapshot, error)
	// forcedWorkSet names the delivered item ids that still have real missing
	// work, classified up front this run from store presence alone. The delta
	// routine unions these into the changed set BEFORE the per-item loop, so a
	// unit whose expected rows are all present is pruned here and never reaches
	// indexOne, which is what removes the per-item no-op cost. The classification
	// must stay cheap: it reads store presence and compares expected prefixes, and
	// must not regenerate chunks. A code source forces nothing (its merkle diff
	// already runs up front); a conversation backfill returns the delivered ids
	// whose expected derived rows are not all present. On an unrecoverable
	// classification failure a source fails safe by returning every delivered id
	// so the run never under-embeds.
	forcedWorkSet(ctx context.Context) ([]string, error)
	// columnSet names the store column family this source's rows carry, so the
	// store write is told the row shape instead of inferring it from the
	// collection name.
	columnSet() semantic.StoreColumnSet
	// indexOne produces the stored chunks and fingerprint for one item.
	indexOne(ctx context.Context, itemID string) (indexer.OneFileResult, error)
	// removalFor maps item ids to the store removal that drops their prior rows.
	removalFor(itemIDs []string) semantic.Removal
	// absencePolicy reports what the delta routine does with an item the store
	// holds that the current capture omits. A code source deletes the missing
	// item under the large-delete quarantine guard. A conversation source
	// retains it, because a transcript missing from a push is almost always a
	// transient disappearance rather than an intended deletion.
	absencePolicy() absencePolicy
	// reuseSource names where one item's already-embedded vectors live: the
	// collection and the relativePath scope that limits the read. Scope none
	// means the item has no per-item reuse source and every chunk embeds. A
	// conversation returns its live collection and conv/<id>/ prefix, while a
	// code file returns its live collection and exact relativePath so like-prefix
	// neighbors never seed the file's reuse map.
	reuseSource(itemID string) itemReuseSource
	// unit is the human progress noun, "file" or "document".
	unit() string
	// producesGraph reports whether a completed run schedules the code-graph
	// build task. A code source builds a call and reference graph from its files,
	// so the spine stamps a graph task; a conversation source has no such graph,
	// so the spine skips it. This is the capability the delta and bootstrap
	// routines consult instead of switching on codebase.Kind.
	producesGraph() bool
	// tracksByteTotals reports whether a delta reconstructs the whole-codebase
	// byte total from the persisted chunk cache. A code source does, so a
	// one-file edit still reports the whole tree's bytes rather than only the
	// delta's; a conversation source does not, and the spine carries the prior
	// total forward instead. This is the capability normalizeDeltaTotalBytes
	// consults instead of switching on codebase.Kind.
	tracksByteTotals() bool
}

// absencePolicy is what runDeltaSync does with an item the store holds that the
// current capture omits. absenceRetain is the zero value and the safe default: it
// keeps the item and its rows so a transient mass disappearance cannot wipe the
// index. absenceDeleteGuarded removes the item; the large-delete quarantine gates
// that removal for code collections only (shouldQuarantineLargeRemoval is
// code-kind gated), so a conversation upsert that opts into deletion has no such
// guard.
type absencePolicy int

const (
	absenceRetain absencePolicy = iota
	absenceDeleteGuarded
)

type itemReuseScope string

const (
	itemReuseScopeNone itemReuseScope = ""
	itemReuseScopePath itemReuseScope = "path"
)

type itemReuseSource struct {
	CollectionName string
	RelativePath   string
	Scope          itemReuseScope
}

// codeItemSource lists and reads a filesystem codebase. It is the byte-for-byte
// behavior the daemon ran before the routine became source-driven: capture is a
// merkle walk and indexOne is one file read and split.
type codeItemSource struct {
	runner         indexingRunner
	resolver       *indexability.Resolver
	codebaseID     string
	canonicalPath  string
	collectionName string
	config         model.IndexConfig
}

func newCodeItemSource(runner indexingRunner, resolver *indexability.Resolver, codebaseID string, canonicalPath string, config model.IndexConfig) codeItemSource {
	return codeItemSource{runner: runner, resolver: resolver, codebaseID: codebaseID, canonicalPath: canonicalPath, collectionName: "", config: config}
}

func (source codeItemSource) withCollectionName(collectionName string) codeItemSource {
	source.collectionName = collectionName
	return source
}

// forcedWorkSet is always empty for a code source: a filesystem sync never
// re-examines a file whose content hash is unchanged, and its merkle diff
// already classifies missing work up front in planSyncDiff.
func (source codeItemSource) forcedWorkSet(_ context.Context) ([]string, error) {
	return nil, nil
}

// columnSet is the base column family: a code file's rows carry no conversation
// scalar columns.
func (source codeItemSource) columnSet() semantic.StoreColumnSet {
	return semantic.CodeColumns()
}

func (source codeItemSource) capture(ctx context.Context) (merkle.Snapshot, error) {
	snapshot, err := merkle.Capture(ctx, source.resolver, source.codebaseID, source.canonicalPath, source.config)
	if err != nil {
		slog.ErrorContext(ctx, "capture code snapshot failed", "path", source.canonicalPath, "err", err)
		return merkle.Snapshot{}, fmt.Errorf("capture code snapshot for %s: %w", source.canonicalPath, err)
	}
	return snapshot, nil
}

func (source codeItemSource) indexOne(ctx context.Context, relativePath string) (indexer.OneFileResult, error) {
	type indexOneOutcome struct {
		result indexer.OneFileResult
		err    error
	}
	done := make(chan indexOneOutcome, 1)
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				var empty indexer.OneFileResult
				done <- indexOneOutcome{result: empty, err: fmt.Errorf("index code file panic: %v", recovered)}
			}
		}()
		result, err := source.runner.IndexOne(ctx, source.resolver, source.codebaseID, source.canonicalPath, relativePath, source.config)
		done <- indexOneOutcome{result: result, err: err}
	}()

	select {
	case <-ctx.Done():
		return indexer.OneFileResult{}, fmt.Errorf("index code file %s cancelled: %w", relativePath, ctx.Err())
	case outcome := <-done:
		if outcome.err == nil {
			return outcome.result, nil
		}
		slog.ErrorContext(ctx, "index code file failed", "path", relativePath, "err", outcome.err)
		return indexer.OneFileResult{}, fmt.Errorf("index code file %s: %w", relativePath, outcome.err)
	}
}

func (source codeItemSource) removalFor(itemIDs []string) semantic.Removal {
	return semantic.RemovePaths(itemIDs)
}

// absencePolicy deletes a file the walk no longer finds, under the large-delete
// quarantine guard, because an absent file is a real filesystem deletion.
func (source codeItemSource) absencePolicy() absencePolicy {
	return absenceDeleteGuarded
}

// reuseSource points one code file's reuse read at exactly its existing live
// rows. Loaded before the delete, those vectors let unchanged chunks inside a
// modified file skip the embedder without reading like-prefix neighbor paths.
func (source codeItemSource) reuseSource(relativePath string) itemReuseSource {
	if source.collectionName == "" || relativePath == "" {
		return itemReuseSource{
			CollectionName: "",
			RelativePath:   "",
			Scope:          itemReuseScopeNone,
		}
	}
	return itemReuseSource{CollectionName: source.collectionName, RelativePath: relativePath, Scope: itemReuseScopePath}
}

func (source codeItemSource) unit() string {
	return "file"
}

// producesGraph is true: a code source builds the call and reference graph from
// its files, so a completed run schedules the graph task.
func (source codeItemSource) producesGraph() bool {
	return true
}

// tracksByteTotals is true: a code delta rebuilds the whole-codebase byte total
// from the persisted chunk cache, so a one-file edit still reports the tree's
// total rather than only the changed files' bytes.
func (source codeItemSource) tracksByteTotals() bool {
	return true
}

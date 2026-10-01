package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"strings"

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

type itemSource interface {
	// capture lists the current items as itemID -> content fingerprint.
	capture(ctx context.Context) (merkle.Snapshot, error)
	forcedWorkSet(ctx context.Context) ([]string, error)
	// columnSet names the store column family this source's rows carry, so the
	// store write is told the row shape instead of inferring it from the
	// collection name.
	columnSet() semantic.StoreColumnSet
	// indexOne produces the stored chunks and fingerprint for one item.
	indexOne(ctx context.Context, itemID string) (indexer.OneFileResult, error)
	// removalFor maps item ids to the store removal that drops their prior rows.
	removalFor(itemIDs []string) semantic.Removal
	absencePolicy() absencePolicy
	reuseSource(itemID string) itemReuseSource
	// unit is the human progress noun, "file" or "document".
	unit() string
	producesGraph() bool
	tracksByteTotals() bool
}

type absencePolicy int

const (
	absenceRetain absencePolicy = iota
	absenceDeleteGuarded
)

type itemReuseScope string

const (
	itemReuseScopeNone   itemReuseScope = ""
	itemReuseScopePrefix itemReuseScope = "prefix"
	itemReuseScopePath   itemReuseScope = "path"
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

func derivedPrefixPresent(storedDerivedPaths map[string]struct{}, prefix string, exact string) bool {
	for relativePath := range storedDerivedPaths {
		if exact != "" && relativePath == exact {
			return true
		}
		if prefix != "" && strings.HasPrefix(relativePath, prefix) {
			return true
		}
	}
	return false
}

package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"maps"
	"sort"
	"strings"
	"sync"

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
	itemReuseScopeNone   itemReuseScope = ""
	itemReuseScopePrefix itemReuseScope = "prefix"
	itemReuseScopePath   itemReuseScope = "path"
)

type itemReuseSource struct {
	CollectionName string
	RelativePath   string
	Scope          itemReuseScope
}

type conversationRowReader interface {
	// LoadConversationDerivedBatch reads the stored rows for a batch of
	// conversations in one Milvus query per id batch. The examination path resolves
	// every delivered conversation from this single read instead of one
	// per-conversation state load.
	LoadConversationDerivedBatch(ctx context.Context, collectionName string, conversationIDs []string) (semantic.ConversationBatchState, error)
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
	return semantic.StoreColumnSetCode
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

// conversationItemSource lists and reads conversation documents the daemon was
// handed over the wire. capture returns the manifest clyde sent (every
// conversation id with its content fingerprint), and indexOne returns the row
// delta for the documents clyde delivered for one conversation. A conversation's
// messages span many rows under one conv/<id>/ prefix, so whole-conversation
// fallback removal is a prefix delete rather than the code file's exact path.
type conversationItemSource struct {
	// collectionName is the live conversation collection, the reuse source for
	// a changed conversation's already-embedded vectors.
	collectionName string
	manifest       map[string]string
	documents      map[string][]model.ConversationDocument
	rowReader      conversationRowReader
	splitterID     string
	// absence is the caller-declared policy for a conversation the manifest
	// omits. clyde sends the mode on the upsert header; the stream handler maps
	// the wire enum to this internal value in conversationAbsencePolicyFromProto,
	// so the delta core never sees the proto type. The constructor only stores the
	// already-mapped value.
	absence absencePolicy
	// backfill, when set by an operator-run backfill, drives forcedWorkSet: it
	// forces each delivered conversation whose expected derived rows are not all
	// present in the store into the changed set, judged from live store presence,
	// and prunes conversations whose derived rows are all present.
	backfill bool
	// force, when set, rebuilds every delivered conversation regardless of
	// presence: forcedWorkSet returns all delivered ids, indexOne regenerates the
	// whole conversation, and reuseSource returns no reuse, so present rows
	// re-embed. When both flags are set, force wins.
	force bool
	// batch caches the one batched read of the live collection for every delivered
	// conversation. indexOne loads it once on first use, so a run of many
	// conversations costs one Milvus query per id batch instead of one read per
	// conversation. It is a pointer so the single load survives the value copies the
	// delta routine makes of the source.
	batch *conversationDerivedBatch
}

// conversationDerivedBatch is the single-flight cache of the batched stored-row
// read shared across every indexOne call in one run.
type conversationDerivedBatch struct {
	once  sync.Once
	state semantic.ConversationBatchState
	err   error
}

func newConversationItemSource(collectionName string, manifest map[string]string, documents []model.ConversationDocument, rowReader conversationRowReader, absence absencePolicy, backfill bool, force bool) conversationItemSource {
	byID := make(map[string][]model.ConversationDocument, len(manifest))
	for _, document := range documents {
		conversationID := document.ConversationID
		byID[conversationID] = append(byID[conversationID], document)
	}
	return conversationItemSource{collectionName: collectionName, manifest: manifest, documents: byID, rowReader: rowReader, splitterID: "", absence: absence, backfill: backfill, force: force, batch: &conversationDerivedBatch{once: sync.Once{}, state: semantic.ConversationBatchState{Rows: nil, Reuse: nil}, err: nil}}
}

// loadDerivedBatch reads the stored rows for every delivered conversation once,
// caching the result so each indexOne consults it instead of issuing its own
// per-conversation query. The batched reuse map spans conversations, so an absent
// target row can reuse a vector embedded for identical content in another
// delivered conversation while still inserting the missing row.
func (source conversationItemSource) loadDerivedBatch(ctx context.Context) (semantic.ConversationBatchState, error) {
	if source.batch == nil {
		return semantic.ConversationBatchState{Rows: map[string]semantic.ConversationStoredRows{}, Reuse: map[string][]float32{}}, nil
	}
	source.batch.once.Do(func() {
		conversationIDs := make([]string, 0, len(source.documents))
		for conversationID := range source.documents {
			conversationIDs = append(conversationIDs, conversationID)
		}
		source.batch.state, source.batch.err = source.rowReader.LoadConversationDerivedBatch(ctx, source.collectionName, conversationIDs)
	})
	return source.batch.state, source.batch.err
}

// forcedWorkSet names the delivered ids the run must re-examine, chosen by the
// two orthogonal flags:
//
//   - force: return ALL delivered ids with no prune. Force rebuilds every
//     delivered conversation regardless of presence, so nothing is classified
//     away here; indexOne regenerates each in full and reuseSource returns no
//     reuse, so present rows re-embed.
//   - backfill (force off): classify CHEAPLY, up front, from store presence
//     alone. It loads the single-flight derived batch once (the same read every
//     indexOne reuses), then for each delivered conversation compares the derived
//     prefixes its messages EXPECT against the derived-path keys the store already
//     holds. A conversation whose expected derived prefixes are all present is a
//     no-op and is pruned; only conversations with at least one missing prefix are
//     returned, so the per-item loop regenerates and embeds only real work.
//   - neither: nil (the normal delta sync forces nothing).
//
// Correctness bound for backfill: it is a PRESENCE check, not a staleness check.
// A backfill treats a conversation with all expected derived prefixes present as
// a no-op even when those rows were derived by an older pipeline, so a plain
// backfill fills MISSING rows and never rebuilds present-but-stale ones.
// Rebuilding present rows is the force path's job: it re-embeds present units
// unpruned rather than relying on this classifier.
//
// The backfill path never calls diffConversationMessages or
// conversationDocumentsToStoredChunks: those regenerate chunks and would
// reintroduce the per-item cost this prune removes. On a batch-load error it logs
// and fails safe by returning every delivered id, so a store read failure never
// causes an under-embed.
func (source conversationItemSource) forcedWorkSet(ctx context.Context) ([]string, error) {
	if source.force {
		// Force rebuilds everything delivered, so no prune: every delivered id is
		// forced into the changed set and re-embedded with reuse disabled.
		return source.allDeliveredIDs(), nil
	}
	if !source.backfill {
		return nil, nil
	}
	if source.rowReader == nil || source.collectionName == "" {
		// No live collection to read presence from, so indexOne would rebuild each
		// conversation in full anyway. Force the whole delivery rather than skip
		// work we cannot verify is already present.
		return source.allDeliveredIDs(), nil
	}
	batch, err := source.loadDerivedBatch(ctx)
	if err != nil {
		slog.WarnContext(ctx, "load conversation derived batch for forced work set failed; forcing all delivered conversations", "collection", source.collectionName, "err", err)
		return source.allDeliveredIDs(), nil
	}
	forced := make([]string, 0, len(source.documents))
	for conversationID, documents := range source.documents {
		stored := batch.Rows[conversationID]
		if conversationNeedsDerivedWork(conversationID, documents, stored.DerivedPaths) {
			forced = append(forced, conversationID)
		}
	}
	sort.Strings(forced)
	return forced, nil
}

// allDeliveredIDs returns every delivered conversation id, sorted. It backs the
// fail-safe path: when store presence cannot be read, forcing the whole delivery
// keeps the run from silently skipping conversations that may need work.
func (source conversationItemSource) allDeliveredIDs() []string {
	ids := make([]string, 0, len(source.documents))
	for conversationID := range source.documents {
		ids = append(ids, conversationID)
	}
	sort.Strings(ids)
	return ids
}

// columnSet is the conversation column family: a conversation's rows carry the
// conversation scalar columns.
func (source conversationItemSource) columnSet() semantic.StoreColumnSet {
	return semantic.StoreColumnSetConversation
}

// conversationNeedsDerivedWork reports whether a delivered conversation still has
// derived (tool or thinking) rows missing from the store, judged CHEAPLY from
// message metadata and stored derived-path presence alone. A tool-carrying
// message expects at least one stored row under convtool/<id>/<msgIdx>/, and a
// thinking-carrying message expects a stored row at convthink/<id>/<msgIdx> or
// under its multipart prefix. It never regenerates chunks, so a fully present
// conversation is classified as a no-op without paying the per-item embed cost.
// A message carrying no tools and no thinking expects no derived rows, so a
// conversation of only such messages needs no work.
//
// It judges PRESENCE only: a message whose expected derived prefix is present
// needs no work, regardless of the pipeline version that produced the stored row.
// Detecting stale or older-version content is the force path's responsibility,
// not this backfill classifier's.
func conversationNeedsDerivedWork(conversationID string, documents []model.ConversationDocument, storedDerivedPaths map[string]string) bool {
	for _, document := range documents {
		if len(document.Tools) > 0 {
			toolPrefix := conversationToolMessagePath(conversationID, document.MessageIndex) + "/"
			if !derivedPrefixPresent(storedDerivedPaths, toolPrefix, "") {
				return true
			}
		}
		if document.Thinking != "" {
			thinkingPath := conversationThinkingPath(conversationID, document.MessageIndex)
			if !derivedPrefixPresent(storedDerivedPaths, thinkingPath+"/", thinkingPath) {
				return true
			}
		}
	}
	return false
}

// derivedPrefixPresent reports whether any stored derived-path key matches the
// exact path or begins with the slash-terminated prefix. The trailing slash on
// prefix is load-bearing: a bare prefix would like-match a sibling index
// (message 1 catching message 12), the same boundary conversationDerivedPathsForMessage
// enforces.
func derivedPrefixPresent(storedDerivedPaths map[string]string, prefix string, exact string) bool {
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

func (source conversationItemSource) capture(_ context.Context) (merkle.Snapshot, error) {
	files := make(map[string]string, len(source.manifest))
	maps.Copy(files, source.manifest)
	return merkle.Snapshot{ConfigDigest: "", Files: files, Inodes: nil}, nil
}

// indexOne diffs the delivered messages against the LIVE collection's rows.
// A bootstrap writes into a staging collection, so this is safe only because
// every route into a conversation bootstrap guarantees the live collection is
// missing or empty (decideEmptyDiffMode requires definitive evidence, the
// delta fallback fires on ErrCollectionMissing, and a first ingest has no
// collection), which degrades the diff to full delivery. A bootstrap over a
// populated live collection would drop unchanged rows at promote; do not
// create such a route.
func (source conversationItemSource) indexOne(ctx context.Context, conversationID string) (indexer.OneFileResult, error) {
	documents, delivered := source.documents[conversationID]
	if !delivered || len(documents) == 0 {
		// clyde asked for this conversation's documents but delivered none, so it
		// cannot be embedded this pass. Skip it as pending without advancing the
		// checkpoint, so the next manifest sync still classifies it changed and clyde
		// resends. Pending is transient, distinct from an unreadable real error.
		return indexer.OneFileResult{
			Chunks:          nil,
			FileHash:        "",
			Skipped:         true,
			SkipReason:      indexer.SkipPending,
			Removed:         false,
			RemovalOverride: false,
			RemovalPaths:    nil,
			RemovalPrefixes: nil,
			ReuseVectors:    nil,
		}, nil
	}
	if source.force {
		// Force rebuilds the whole conversation: regenerate every chunk and drop the
		// prior rows (removalOverride), and carry no reuse map so reuseSource returns
		// no reuse and embedChunkBatch re-embeds every present chunk.
		return source.fullConversationResult(ctx, conversationID, documents, true)
	}
	if source.rowReader == nil || source.collectionName == "" {
		return source.fullConversationResult(ctx, conversationID, documents, false)
	}

	batch, err := source.loadDerivedBatch(ctx)
	if err != nil {
		slog.WarnContext(ctx, "load conversation derived batch failed; falling back to full conversation reindex", "conversation_id", conversationID, "collection", source.collectionName, "err", err)
		return source.fullConversationResult(ctx, conversationID, documents, true)
	}
	stored := batch.Rows[conversationID]
	if stored.Messages == nil {
		stored.Messages = map[int32]semantic.StoredMessageState{}
	}
	if stored.DerivedPaths == nil {
		stored.DerivedPaths = map[string]string{}
	}
	// The batch-wide reuse map spans every delivered conversation, so a changed
	// message reuses a vector embedded for identical content anywhere in the batch
	// while its missing target row is still inserted. A non-nil map keeps the
	// message-delta path from falling back to a per-conversation reuse load.
	batchReuse := batch.Reuse
	if batchReuse == nil {
		batchReuse = map[string][]float32{}
	}

	delta, diffErr := diffConversationMessages(ctx, conversationID, documents, stored)
	if diffErr != nil {
		return indexer.OneFileResult{
			Chunks:          nil,
			FileHash:        "",
			Skipped:         false,
			SkipReason:      indexer.SkipNone,
			Removed:         false,
			RemovalOverride: false,
			RemovalPaths:    nil,
			RemovalPrefixes: nil,
			ReuseVectors:    nil,
		}, diffErr
	}
	chunks, err := conversationDocumentsToStoredChunks(ctx, delta.documents)
	if err != nil {
		return indexer.OneFileResult{
			Chunks:          nil,
			FileHash:        "",
			Skipped:         false,
			SkipReason:      indexer.SkipNone,
			Removed:         false,
			RemovalOverride: false,
			RemovalPaths:    nil,
			RemovalPrefixes: nil,
			ReuseVectors:    nil,
		}, err
	}
	return indexer.OneFileResult{
		Chunks:          chunks,
		FileHash:        source.manifest[conversationID],
		Skipped:         false,
		SkipReason:      indexer.SkipNone,
		Removed:         false,
		RemovalOverride: true,
		RemovalPaths:    delta.removalPaths,
		RemovalPrefixes: delta.removalPrefixes,
		ReuseVectors:    batchReuse,
	}, nil
}

func (source conversationItemSource) fullConversationResult(ctx context.Context, conversationID string, documents []model.ConversationDocument, removalOverride bool) (indexer.OneFileResult, error) {
	chunks, err := conversationDocumentsToStoredChunks(ctx, documents)
	if err != nil {
		return indexer.OneFileResult{
			Chunks:          nil,
			FileHash:        "",
			Skipped:         false,
			SkipReason:      indexer.SkipNone,
			Removed:         false,
			RemovalOverride: false,
			RemovalPaths:    nil,
			RemovalPrefixes: nil,
			ReuseVectors:    nil,
		}, err
	}
	result := indexer.OneFileResult{
		Chunks:          chunks,
		FileHash:        source.manifest[conversationID],
		Skipped:         false,
		SkipReason:      indexer.SkipNone,
		Removed:         false,
		RemovalOverride: false,
		RemovalPaths:    nil,
		RemovalPrefixes: nil,
		ReuseVectors:    nil,
	}
	if removalOverride {
		result.RemovalOverride = true
		result.RemovalPrefixes = conversationFullRemovalPrefixes(conversationID)
	}
	return result, nil
}

func (source conversationItemSource) removalFor(itemIDs []string) semantic.Removal {
	prefixes := make([]string, 0, len(itemIDs)*3)
	for _, conversationID := range itemIDs {
		prefixes = append(prefixes, conversationFullRemovalPrefixes(conversationID)...)
	}
	return semantic.RemovePrefixes(prefixes)
}

// absencePolicy returns the caller-declared policy for a conversation the
// manifest omits. conversationAbsencePolicyFromProto maps an unset or RETAIN wire
// mode to absenceRetain, so a transient disappearance keeps the rows and a later
// restoring push is a no-op. Only an AUTHORITATIVE upsert or the explicit
// single-conversation delete removes a conversation.
func (source conversationItemSource) absencePolicy() absencePolicy {
	return source.absence
}

// reuseSource stays prefix scoped for loader-error fallback. The normal
// message-delta path carries reuse in OneFileResult and skips this per-item
// load, but a full fallback still needs the existing conv/<id>/ rows. Under
// force the reuse lever is disabled: it returns the no-reuse scope so a present
// chunk cannot be served from a stored vector and every chunk re-embeds, which
// is what makes force rebuild present-but-stale rows.
func (source conversationItemSource) reuseSource(conversationID string) itemReuseSource {
	if source.force || source.collectionName == "" {
		return itemReuseSource{
			CollectionName: "",
			RelativePath:   "",
			Scope:          itemReuseScopeNone,
		}
	}
	return itemReuseSource{CollectionName: source.collectionName, RelativePath: conversationRelativePathPrefix(conversationID), Scope: itemReuseScopePrefix}
}

func (source conversationItemSource) unit() string {
	return "document"
}

// producesGraph is false: a conversation collection has no code graph, so the
// spine skips the graph task for it.
func (source conversationItemSource) producesGraph() bool {
	return false
}

// tracksByteTotals is false: a conversation ingest does not reconstruct a
// whole-collection byte total from a chunk cache, so the spine carries the prior
// total forward instead of running the code chunk-cache normalization.
func (source conversationItemSource) tracksByteTotals() bool {
	return false
}

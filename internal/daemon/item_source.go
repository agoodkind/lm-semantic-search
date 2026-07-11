package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"maps"
	"sort"
	"sync"

	"goodkind.io/lm-semantic-search/internal/indexability"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// forcedItemsSet collects a source's forced item ids into a set for O(1) lookup
// in applyDeltaChanges. It returns nil when the source forces nothing (the
// normal sync), so the hash-equality skip stays in force for every item.
func forcedItemsSet(source itemSource) map[string]struct{} {
	forced := source.forcedItems()
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
	// forcedItems names item ids that must be re-examined this run even when
	// their captured fingerprint matches the stored checkpoint. The delta routine
	// unions these into the changed set, so indexOne runs for them and its normal
	// content diff reuses unchanged chunks while embedding only new ones. A code
	// source never forces (returns nil); a conversation source returns its
	// delivered ids when an operator-run backfill asks to re-examine them.
	forcedItems() []string
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

// forcedItems is always nil for a code source: a filesystem sync never
// re-examines a file whose content hash is unchanged.
func (source codeItemSource) forcedItems() []string {
	return nil
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
	// derivedVersions records the tool and thinking chunking version that last
	// completed for each conversation. It is conversation-specific state and is
	// persisted beside the Merkle checkpoint.
	derivedVersions map[string]string
	// absence is the caller-declared policy for a conversation the manifest
	// omits. clyde sends the mode on the upsert header; the stream handler maps
	// the wire enum to this internal value in conversationAbsencePolicyFromProto,
	// so the delta core never sees the proto type. The constructor only stores the
	// already-mapped value.
	absence absencePolicy
	// reexamine, when set by an operator-run backfill, forces each delivered
	// conversation whose derived marker is absent or stale into the changed set.
	reexamine bool
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

func newConversationItemSource(collectionName string, manifest map[string]string, documents []model.ConversationDocument, rowReader conversationRowReader, absence absencePolicy, reexamine bool) conversationItemSource {
	byID := make(map[string][]model.ConversationDocument, len(manifest))
	for _, document := range documents {
		conversationID := document.ConversationID
		byID[conversationID] = append(byID[conversationID], document)
	}
	return conversationItemSource{collectionName: collectionName, manifest: manifest, documents: byID, rowReader: rowReader, splitterID: "", derivedVersions: map[string]string{}, absence: absence, reexamine: reexamine, batch: &conversationDerivedBatch{once: sync.Once{}, state: semantic.ConversationBatchState{Rows: nil, Reuse: nil}, err: nil}}
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

// forcedItems returns delivered conversation ids whose derived marker is absent
// or stale when an operator-run backfill asks to reexamine them. It returns nil
// before consulting markers for a normal sync, so that path stays unchanged.
func (source conversationItemSource) forcedItems() []string {
	if !source.reexamine {
		return nil
	}
	forced := make([]string, 0, len(source.documents))
	for conversationID := range source.documents {
		if source.derivedVersions[conversationID] == derivedPipelineVersion {
			continue
		}
		forced = append(forced, conversationID)
	}
	return forced
}

func (source conversationItemSource) checkpointDerivedMarker(snapshotPath string, conversationID string) error {
	if _, delivered := source.documents[conversationID]; !delivered {
		return nil
	}
	nextVersions := maps.Clone(source.derivedVersions)
	nextVersions[conversationID] = derivedPipelineVersion
	markerPath := conversationDerivedMarkerPath(snapshotPath)
	if err := writeConversationDerivedMarkers(markerPath, nextVersions); err != nil {
		return err
	}
	source.derivedVersions[conversationID] = derivedPipelineVersion
	return nil
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
// load, but a full fallback still needs the existing conv/<id>/ rows.
func (source conversationItemSource) reuseSource(conversationID string) itemReuseSource {
	if source.collectionName == "" {
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

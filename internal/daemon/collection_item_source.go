package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"maps"
	"sort"
	"sync"

	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// collectionRow is one validated client row of a document collection. RowKey
// is stored as the row's relativePath. Scalars includes the item id column,
// which validation sets from ItemID. ContinuationPrefix starts every stored
// part after the first of a split row.
type collectionRow struct {
	RowKey             string
	ItemID             string
	Text               string
	Scalars            map[string]model.ScalarValue
	ContinuationPrefix string
}

type collectionRowFamily struct {
	Key    string
	Chunks []model.StoredChunk
}

type collectionItemDelivery struct {
	rows map[string][]collectionRow

	chunkByteBudget int
}

func newCollectionItemDelivery(rows []collectionRow, chunkByteBudget int) collectionItemDelivery {
	delivery := collectionItemDelivery{rows: make(map[string][]collectionRow), chunkByteBudget: resolveCollectionChunkBudget([]int{chunkByteBudget})}
	for _, row := range rows {
		delivery.rows[row.ItemID] = append(delivery.rows[row.ItemID], row)
	}
	return delivery
}

func (delivery collectionItemDelivery) itemIDs() []string {
	ids := make([]string, 0, len(delivery.rows))
	for itemID := range delivery.rows {
		ids = append(ids, itemID)
	}
	sort.Strings(ids)
	return ids
}

func (delivery collectionItemDelivery) delivered(itemID string) bool {
	return len(delivery.rows[itemID]) > 0
}

func (delivery collectionItemDelivery) backfillFamilies(itemID string) []string {
	families := make([]string, 0, len(delivery.rows[itemID]))
	for _, row := range delivery.rows[itemID] {
		if collectionTextIsStorable(row.Text) {
			families = append(families, row.RowKey)
		}
	}
	return families
}

func (delivery collectionItemDelivery) rowFamilies(ctx context.Context, itemID string) ([]collectionRowFamily, error) {

	rows := delivery.rows[itemID]
	families := make([]collectionRowFamily, 0, len(rows))
	for _, row := range rows {
		families = append(families, collectionRowFamily{Key: row.RowKey, Chunks: delivery.rowChunks(row)})
	}
	return families, nil
}

func (delivery collectionItemDelivery) rowChunks(row collectionRow) []model.StoredChunk {
	return appendContinuedStorableField(
		nil,
		row.Text,
		delivery.chunkByteBudget,
		row.ContinuationPrefix,
		func(piece string, partIndex int, multipart bool) model.StoredChunk {
			relativePath := row.RowKey
			if multipart {
				relativePath = fmt.Sprintf("%s/%d", row.RowKey, partIndex)
			}
			return newCollectionRowChunk(row, relativePath, piece)
		},
	)
}

func newCollectionRowChunk(row collectionRow, relativePath string, content string) model.StoredChunk {
	return model.StoredChunk{Content: content, RelativePath: relativePath, SplitPartRecorded: true, Scalars: maps.Clone(row.Scalars)}
}

// collectionStoredItems is one batched read of the live collection for every
// delivered item. usablePaths maps an item id to the relativePath values of
// its stored rows with usable content. reuse maps a content hash to a stored
// dense vector.
type collectionStoredItems struct {
	usablePaths map[string]map[string]struct{}
	reuse       map[string][]float32
}

// collectionStoredReader reads the stored rows of a batch of items.
type collectionStoredReader interface {
	loadItems(ctx context.Context, collectionName string, itemIDs []string) (collectionStoredItems, error)
}

// collectionItemLoader is the store read of a generic document collection.
type collectionItemLoader interface {
	LoadCollectionItemBatch(ctx context.Context, collectionName string, itemColumn string, itemIDs []string) (semantic.CollectionItemBatchState, error)
}

// declaredStoredReader reads a generic document collection by its declared
// item id column.
type declaredStoredReader struct {
	loader     collectionItemLoader
	itemColumn string
}

func (reader declaredStoredReader) loadItems(ctx context.Context, collectionName string, itemIDs []string) (collectionStoredItems, error) {
	batch, err := reader.loader.LoadCollectionItemBatch(ctx, collectionName, reader.itemColumn, itemIDs)
	if err != nil {
		slog.ErrorContext(ctx, "load stored collection item rows failed", "collection", collectionName, "item_column", reader.itemColumn, "items", len(itemIDs), "err", err)
		return collectionStoredItems{usablePaths: nil, reuse: nil}, fmt.Errorf("load collection item rows: %w", err)
	}
	usable := make(map[string]map[string]struct{}, len(batch.Rows))
	for itemID, stored := range batch.Rows {
		usable[itemID] = stored.UsablePaths
	}
	return collectionStoredItems{usablePaths: usable, reuse: batch.Reuse}, nil
}

type collectionItemSelector struct {
	itemColumn string
}

func newCollectionItemSelector(declaration model.CollectionDeclaration) collectionItemSelector {
	return collectionItemSelector{itemColumn: declaration.ItemIDColumn}
}

func (selector collectionItemSelector) removal(itemIDs []string) semantic.Removal {
	return semantic.RemoveItems(selector.itemColumn, itemIDs, nil)
}

// collectionItemSource is the item source of every document collection ingest.
// capture returns the delivered manifest. indexOne emits the delivered row
// families absent from the live collection, and force replaces the item's
// rows. The shared delta and bootstrap routine runs it. Document ingest then
// uses the same checkpoint, coalescing, vector reuse, and absence handling as
// code ingest.
type collectionItemSource struct {
	// collectionName is the live collection, the source of stored presence and
	// reuse vectors.
	collectionName string
	manifest       map[string]string
	delivery       collectionItemDelivery
	stored         collectionStoredReader
	selector       collectionItemSelector
	columns        semantic.StoreColumnSet
	// absence is the caller-declared policy for an item the manifest omits.
	absence absencePolicy
	// backfill forces delivered items with an absent backfill family (see
	// backfillFamilies) into the changed set and prunes items with every such
	// family present.
	backfill bool
	// force replaces every delivered item's rows with reuse disabled. When both
	// flags are set, force wins.
	force bool
	// batch caches the one batched stored-row read shared by every indexOne in
	// a run. The pointer keeps the single read across the value copies the
	// delta routine makes of the source.
	batch *collectionStoredBatch
}

type collectionStoredBatch struct {
	once  sync.Once
	items collectionStoredItems
	err   error
}

// collectionItemSourceConfig lists the inputs of one document ingest source.
type collectionItemSourceConfig struct {
	collectionName string
	manifest       map[string]string
	delivery       collectionItemDelivery
	stored         collectionStoredReader
	selector       collectionItemSelector
	columns        semantic.StoreColumnSet
	absence        absencePolicy
	backfill       bool
	force          bool
}

func newCollectionItemSource(config collectionItemSourceConfig) collectionItemSource {
	return collectionItemSource{
		collectionName: config.collectionName,
		manifest:       config.manifest,
		delivery:       config.delivery,
		stored:         config.stored,
		selector:       config.selector,
		columns:        config.columns,
		absence:        config.absence,
		backfill:       config.backfill,
		force:          config.force,
		batch:          &collectionStoredBatch{once: sync.Once{}, items: collectionStoredItems{usablePaths: nil, reuse: nil}, err: nil},
	}
}

// documentDelivery is the content and flags of one document collection job.
type documentDelivery struct {
	manifest map[string]string

	rows            []collectionRow
	absence         absencePolicy
	backfill        bool
	force           bool
	chunkByteBudget int
}

func newDocumentItemSource(collectionName string, declaration model.CollectionDeclaration, stored collectionStoredReader, delivery documentDelivery) collectionItemSource {
	return newCollectionItemSource(collectionItemSourceConfig{
		collectionName: collectionName,
		manifest:       delivery.manifest,
		delivery:       newCollectionItemDelivery(delivery.rows, delivery.chunkByteBudget),
		stored:         stored,
		selector:       newCollectionItemSelector(declaration),
		columns:        semantic.ColumnsForDeclaration(declaration),
		absence:        delivery.absence,
		backfill:       delivery.backfill,
		force:          delivery.force,
	})
}

func (source collectionItemSource) loadStored(ctx context.Context) (collectionStoredItems, error) {
	if source.batch == nil {
		return collectionStoredItems{usablePaths: map[string]map[string]struct{}{}, reuse: map[string][]float32{}}, nil
	}
	source.batch.once.Do(func() {
		source.batch.items, source.batch.err = source.stored.loadItems(ctx, source.collectionName, source.delivery.itemIDs())
	})
	return source.batch.items, source.batch.err
}

func (source collectionItemSource) capture(_ context.Context) (merkle.Snapshot, error) {
	files := make(map[string]string, len(source.manifest))
	maps.Copy(files, source.manifest)
	return merkle.Snapshot{ConfigDigest: "", Files: files, Inodes: nil}, nil
}

// forcedWorkSet returns every delivered item under force. Under backfill it
// returns the delivered items with at least one backfill family absent from
// the store, judged from one stored-row read without generating chunks. A
// failed read forces every delivered item. The run then never under-embeds.
// Without either flag it forces nothing.
func (source collectionItemSource) forcedWorkSet(ctx context.Context) ([]string, error) {
	if source.force {
		return source.delivery.itemIDs(), nil
	}
	if !source.backfill {
		return nil, nil
	}
	if source.stored == nil || source.collectionName == "" {
		return source.delivery.itemIDs(), nil
	}
	stored, err := source.loadStored(ctx)
	if err != nil {
		slog.WarnContext(ctx, "load stored item rows for forced work set failed; forcing all delivered items", "collection", source.collectionName, "err", err)
		return source.delivery.itemIDs(), nil
	}
	forced := make([]string, 0)
	for _, itemID := range source.delivery.itemIDs() {
		usable := stored.usablePaths[itemID]
		for _, family := range source.delivery.backfillFamilies(itemID) {
			if !derivedPrefixPresent(usable, family+"/", family) {
				forced = append(forced, itemID)
				break
			}
		}
	}
	return forced, nil
}

func (source collectionItemSource) columnSet() semantic.StoreColumnSet {
	return source.columns
}

// indexOne emits one delivered item's chunks. An undelivered item is pending,
// and its checkpoint entry stays behind. Force emits every chunk and leaves the
// removal to removalFor, which drops the item's rows. Without a stored-row
// reader the item also embeds in full after that removal. Otherwise indexOne
// emits only the families absent from the live collection and removes nothing.
//
// A bootstrap writes into a staging collection. Every route into a document
// bootstrap has a missing or empty live collection. The family filter then
// sees no stored rows and emits the full delivery.
func (source collectionItemSource) indexOne(ctx context.Context, itemID string) (indexer.OneFileResult, error) {
	result := indexer.OneFileResult{
		Chunks:          nil,
		FileHash:        "",
		Skipped:         false,
		SkipReason:      indexer.SkipNone,
		Removed:         false,
		RemovalOverride: false,
		RemovalPaths:    nil,
		RemovalPrefixes: nil,
		ReuseVectors:    nil,
	}
	if !source.delivery.delivered(itemID) {
		result.Skipped = true
		result.SkipReason = indexer.SkipPending
		return result, nil
	}
	families, err := source.delivery.rowFamilies(ctx, itemID)
	if err != nil {
		return result, err
	}
	result.FileHash = source.manifest[itemID]
	if source.force || source.stored == nil || source.collectionName == "" {
		result.Chunks = familyChunks(families, nil)
		return result, nil
	}
	stored, err := source.loadStored(ctx)
	if err != nil {
		slog.WarnContext(ctx, "load stored item rows failed", "item_id", itemID, "collection", source.collectionName, "err", err)
		return indexer.OneFileResult{}, fmt.Errorf("load stored rows for item %q: %w", itemID, err)
	}
	reuse := stored.reuse
	if reuse == nil {
		reuse = map[string][]float32{}
	}
	usable := stored.usablePaths[itemID]
	if usable == nil {
		usable = map[string]struct{}{}
	}
	result.Chunks = familyChunks(families, usable)
	result.RemovalOverride = true
	result.ReuseVectors = reuse
	return result, nil
}

// familyChunks returns the chunks of every family absent from usablePaths. A
// family is present when a usable stored path equals its key or starts with
// the key and a slash. A nil usablePaths returns every chunk.
func familyChunks(families []collectionRowFamily, usablePaths map[string]struct{}) []model.StoredChunk {
	chunks := make([]model.StoredChunk, 0)
	for _, family := range families {
		if usablePaths != nil && derivedPrefixPresent(usablePaths, family.Key+"/", family.Key) {
			continue
		}
		chunks = append(chunks, family.Chunks...)
	}
	return chunks
}

func (source collectionItemSource) removalFor(itemIDs []string) semantic.Removal {
	return source.selector.removal(itemIDs)
}

// absencePolicy returns the caller-declared policy for an item the manifest
// omits. Only an authoritative upsert removes an omitted item.
func (source collectionItemSource) absencePolicy() absencePolicy {
	return source.absence
}

func (source collectionItemSource) reuseSource(_ string) itemReuseSource {
	return itemReuseSource{Scope: itemReuseScopeNone}
}

func (source collectionItemSource) unit() string {
	return "document"
}

// producesGraph is false: a document collection has no code graph.
func (source collectionItemSource) producesGraph() bool {
	return false
}

// tracksByteTotals is false: a document ingest keeps the prior byte total.
func (source collectionItemSource) tracksByteTotals() bool {
	return false
}

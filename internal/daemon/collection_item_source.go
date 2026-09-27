package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"maps"
	"sort"
	"strings"
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

// collectionRowFamily is the stored rows one delivered row produces. Key is
// the family path: a client row key, or the message, tool call, or thinking
// path of a conversation document. Chunks are the family's physical rows. A
// family longer than the split budget stores parts under Key/0, Key/1, and so
// on. Presence is judged per family. A family stored under a different part
// layout still counts as present.
type collectionRowFamily struct {
	Key    string
	Chunks []model.StoredChunk
}

// collectionItemDelivery is the per-item content one upsert delivered. Either
// conversation documents or client rows deliver an item, never both.
type collectionItemDelivery struct {
	documents map[string][]model.ConversationDocument
	rows      map[string][]collectionRow
	// projectConversation stores client rows of a collection with the
	// conversation declaration in the conversation fields of each chunk. Both
	// ingest RPCs then write identical conversation rows.
	projectConversation bool
	chunkByteBudget     int
}

func newCollectionItemDelivery(documents []model.ConversationDocument, rows []collectionRow, projectConversation bool, chunkByteBudget int) collectionItemDelivery {
	delivery := collectionItemDelivery{
		documents:           make(map[string][]model.ConversationDocument),
		rows:                make(map[string][]collectionRow),
		projectConversation: projectConversation,
		chunkByteBudget:     resolveConversationChunkBudget([]int{chunkByteBudget}),
	}
	for _, document := range documents {
		delivery.documents[document.ConversationID] = append(delivery.documents[document.ConversationID], document)
	}
	for _, row := range rows {
		delivery.rows[row.ItemID] = append(delivery.rows[row.ItemID], row)
	}
	return delivery
}

// itemIDs returns every delivered item id, sorted.
func (delivery collectionItemDelivery) itemIDs() []string {
	ids := make([]string, 0, len(delivery.documents)+len(delivery.rows))
	for itemID := range delivery.documents {
		ids = append(ids, itemID)
	}
	for itemID := range delivery.rows {
		if _, documented := delivery.documents[itemID]; documented {
			continue
		}
		ids = append(ids, itemID)
	}
	sort.Strings(ids)
	return ids
}

func (delivery collectionItemDelivery) delivered(itemID string) bool {
	if documents := delivery.documents[itemID]; len(documents) > 0 {
		return true
	}
	return len(delivery.rows[itemID]) > 0
}

// backfillFamilies returns the family keys a backfill checks for one delivered
// item without generating chunks. A conversation item checks its tool call and
// thinking families only. Conversation documents list them from document
// metadata. In a collection with the conversation declaration, client rows
// list every convtool/ and convthink/ row with storable text. Every other
// client item checks every row with storable text.
func (delivery collectionItemDelivery) backfillFamilies(itemID string) []string {
	if documents, found := delivery.documents[itemID]; found {
		families := make([]string, 0)
		for _, document := range documents {
			for toolIndex := range document.Tools {
				families = append(families, conversationToolCallPath(itemID, document.MessageIndex, toolIndex))
			}
			if document.Thinking != "" {
				families = append(families, conversationThinkingPath(itemID, document.MessageIndex))
			}
		}
		return families
	}
	families := make([]string, 0, len(delivery.rows[itemID]))
	for _, row := range delivery.rows[itemID] {
		if !conversationTextIsStorable(row.Text) {
			continue
		}
		if delivery.projectConversation && strings.HasPrefix(row.RowKey, conversationRelativePathPrefix(itemID)) {
			continue
		}
		families = append(families, row.RowKey)
	}
	return families
}

// rowFamilies generates the stored chunks of one delivered item grouped by
// family, in delivery order. Conversation documents generate chunks through
// conversationDocumentsToStoredChunks. Client rows generate chunks through
// rowChunks.
func (delivery collectionItemDelivery) rowFamilies(ctx context.Context, itemID string) ([]collectionRowFamily, error) {
	if documents, found := delivery.documents[itemID]; found {
		chunks, err := conversationDocumentsToStoredChunks(ctx, documents, delivery.chunkByteBudget)
		if err != nil {
			return nil, err
		}
		return groupConversationChunkFamilies(itemID, chunks), nil
	}
	rows := delivery.rows[itemID]
	families := make([]collectionRowFamily, 0, len(rows))
	for _, row := range rows {
		families = append(families, collectionRowFamily{Key: row.RowKey, Chunks: delivery.rowChunks(row)})
	}
	return families, nil
}

// rowChunks splits one client row's text at the chunk byte budget with the
// row's continuation prefix, through the appendContinuedStorableField split
// that conversation tool call rows also use.
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
			return newCollectionRowChunk(row, relativePath, piece, delivery.projectConversation)
		},
	)
}

// groupConversationChunkFamilies groups one conversation's chunks by message
// text, tool call, and thinking family, keeping chunk order.
func groupConversationChunkFamilies(conversationID string, chunks []model.StoredChunk) []collectionRowFamily {
	families := make([]collectionRowFamily, 0)
	index := make(map[string]int)
	for _, chunk := range chunks {
		key := conversationChunkFamilyKey(conversationID, chunk.RelativePath)
		position, found := index[key]
		if !found {
			position = len(families)
			index[key] = position
			families = append(families, collectionRowFamily{Key: key, Chunks: nil})
		}
		families[position].Chunks = append(families[position].Chunks, chunk)
	}
	return families
}

// conversationChunkFamilyKey strips the part suffix from a conversation chunk
// path. A message text family is conv/<id>/<message>, a tool call family is
// convtool/<id>/<message>/<tool>, and a thinking family is
// convthink/<id>/<message>.
func conversationChunkFamilyKey(conversationID string, relativePath string) string {
	families := []struct {
		prefix   string
		segments int
	}{
		{prefix: conversationRelativePathPrefix(conversationID), segments: 1},
		{prefix: conversationToolRelativePathPrefix(conversationID), segments: 2},
		{prefix: conversationThinkingRelativePathPrefix(conversationID), segments: 1},
	}
	for _, family := range families {
		remainder, found := strings.CutPrefix(relativePath, family.prefix)
		if !found {
			continue
		}
		parts := strings.Split(remainder, "/")
		if len(parts) <= family.segments {
			return relativePath
		}
		return family.prefix + strings.Join(parts[:family.segments], "/")
	}
	return relativePath
}

// newCollectionRowChunk builds one stored chunk of a client row. A collection
// with the conversation declaration stores the declared values in the
// conversation fields. Every other collection stores them in Scalars.
func newCollectionRowChunk(row collectionRow, relativePath string, content string, projectConversation bool) model.StoredChunk {
	chunk := model.StoredChunk{
		Content:              content,
		RelativePath:         relativePath,
		StartLine:            0,
		EndLine:              0,
		Language:             "",
		FileExtension:        "",
		ConversationID:       "",
		ParentConversationID: "",
		MessageIndex:         0,
		Role:                 "",
		TimestampUnix:        0,
		WorkspaceRoot:        "",
		Archived:             false,
		SplitPart:            0,
		SplitPartRecorded:    true,
		LoadRules:            "",
		Scalars:              nil,
		Score:                0,
	}
	if !projectConversation {
		chunk.Scalars = maps.Clone(row.Scalars)
		return chunk
	}
	chunk.ConversationID = row.ItemID
	chunk.ParentConversationID = row.Scalars[semantic.ConversationParentColumn].String
	chunk.Role = row.Scalars[semantic.ConversationRoleColumn].String
	chunk.WorkspaceRoot = row.Scalars[semantic.ConversationWorkspaceRootColumn].String
	chunk.Archived = row.Scalars[semantic.ConversationArchivedColumn].Bool
	chunk.TimestampUnix = row.Scalars[semantic.ConversationTimestampColumn].Int64
	chunk.MessageIndex = safeInt32(int(row.Scalars[semantic.ConversationMessageIndexColumn].Int64))
	chunk.LoadRules = row.Scalars[semantic.ConversationLoadRulesColumn].String
	return chunk
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

// conversationStoredReader reads a conversation collection through the
// conversation batch read, which also selects legacy rows without a
// conversationId by path prefix. A message with storable stored text reports
// the message text family path. Presence then matches by message index exactly
// as the conversation delta did.
type conversationStoredReader struct {
	rowReader conversationRowReader
}

func (reader conversationStoredReader) loadItems(ctx context.Context, collectionName string, itemIDs []string) (collectionStoredItems, error) {
	batch, err := reader.rowReader.LoadConversationDerivedBatch(ctx, collectionName, itemIDs)
	if err != nil {
		slog.ErrorContext(ctx, "load stored conversation rows failed", "collection", collectionName, "items", len(itemIDs), "err", err)
		return collectionStoredItems{usablePaths: nil, reuse: nil}, fmt.Errorf("load conversation rows: %w", err)
	}
	usable := make(map[string]map[string]struct{}, len(batch.Rows))
	for conversationID, stored := range batch.Rows {
		paths := maps.Clone(usableConversationDerivedPaths(stored))
		if paths == nil {
			paths = map[string]struct{}{}
		}
		for messageIndex, message := range stored.Messages {
			if conversationStorableText(message.Text) != "" {
				paths[conversationRelativePath(conversationID, messageIndex, 0, false)] = struct{}{}
			}
		}
		usable[conversationID] = paths
	}
	return collectionStoredItems{usablePaths: usable, reuse: batch.Reuse}, nil
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

// collectionItemSelector selects an item's stored rows by the declared item id
// column. legacyConversation adds the conv/, convtool/, and convthink/ path
// prefixes, which select conversation rows written before the conversationId
// column existed.
type collectionItemSelector struct {
	itemColumn         string
	legacyConversation bool
}

// newCollectionItemSelector returns the stored-row selector of a saved
// declaration. Only the conversation declaration adds the legacy conversation
// path prefixes.
func newCollectionItemSelector(declaration model.CollectionDeclaration) collectionItemSelector {
	return collectionItemSelector{
		itemColumn:         declaration.ItemIDColumn,
		legacyConversation: semantic.IsConversationDeclaration(declaration),
	}
}

func (selector collectionItemSelector) removal(itemIDs []string) semantic.Removal {
	var prefixes []string
	if selector.legacyConversation {
		prefixes = make([]string, 0, len(itemIDs)*3)
		for _, itemID := range itemIDs {
			prefixes = append(prefixes, conversationFullRemovalPrefixes(itemID)...)
		}
	}
	return semantic.RemoveItems(selector.itemColumn, itemIDs, prefixes)
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
	manifest        map[string]string
	documents       []model.ConversationDocument
	rows            []collectionRow
	absence         absencePolicy
	backfill        bool
	force           bool
	chunkByteBudget int
}

// newDocumentItemSource builds the ingest source of one document collection
// job from the collection's saved declaration. The conversation declaration
// stores client rows in the conversation fields and selects legacy
// conversation rows by path prefix. stored reads the live collection and may
// be nil, which embeds every delivered item in full.
func newDocumentItemSource(collectionName string, declaration model.CollectionDeclaration, stored collectionStoredReader, delivery documentDelivery) collectionItemSource {
	conversation := semantic.IsConversationDeclaration(declaration)
	return newCollectionItemSource(collectionItemSourceConfig{
		collectionName: collectionName,
		manifest:       delivery.manifest,
		delivery:       newCollectionItemDelivery(delivery.documents, delivery.rows, conversation, delivery.chunkByteBudget),
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

// reuseSource returns the per-item reuse read of the full-delivery path. The
// family filter returns reuse vectors in OneFileResult and skips this read.
// Force disables reuse, and every chunk embeds again. A conversation item
// reads its conv/<id>/ rows. A generic item has no per-item reuse read beyond
// the batched stored-row read.
func (source collectionItemSource) reuseSource(itemID string) itemReuseSource {
	if source.force || source.collectionName == "" || !source.selector.legacyConversation {
		return itemReuseSource{CollectionName: "", RelativePath: "", Scope: itemReuseScopeNone}
	}
	return itemReuseSource{CollectionName: source.collectionName, RelativePath: conversationRelativePathPrefix(itemID), Scope: itemReuseScopePrefix}
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

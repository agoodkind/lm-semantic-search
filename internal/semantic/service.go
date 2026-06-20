// Package semantic implements embedding and Milvus-backed code indexing.
package semantic

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"unicode"
	"unicode/utf8"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/embedding"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/spans"
	"goodkind.io/lm-semantic-search/internal/tshash"
)

// Milvus field names match the upstream TS schema at
// packages/core/src/vectordb/milvus-vectordb.ts so the Go daemon reads and
// writes the same collections the TS adapter does. The names are camelCase
// because that is what the TS adapter wrote.
const (
	maxCollectionNameLength = 255
	stagingCollectionSuffix = "_stg"
	denseVectorFieldName    = "vector"
	sparseVectorFieldName   = "sparse_vector"
	contentFieldName        = "content"
	relativePathFieldName   = "relativePath"
	startLineFieldName      = "startLine"
	endLineFieldName        = "endLine"
	fileExtensionFieldName  = "fileExtension"
	metadataFieldName       = "metadata"
	idFieldName             = "id"
	countOutputField        = "count(*)"
)

// Progress reports semantic indexing progress after chunk extraction.
//
// ChunksReused counts chunks served a vector from the reuse map (no embedder
// call), and ChunksEmbedded counts chunks whose vector came from the embedder
// this run. Their sum is the chunks written so far, so a surface can show
// total = reused + embedded and make the reuse-vs-redo split visible.
type Progress struct {
	Phase                     string
	OverallPercent            float64
	EmbeddingBatchesTotal     int32
	EmbeddingBatchesCompleted int32
	CollectionRowsWritten     int32
	ChunksProcessed           int32
	ChunksReused              int32
	ChunksEmbedded            int32
}

// Service owns the embedding provider and Milvus client for semantic search.
type Service struct {
	cfg             config.Config
	embedder        embedding.Provider
	milvus          *milvusclient.Client
	available       atomic.Bool
	reconnectCancel context.CancelFunc
	reconnectDone   chan struct{}
	closeOnce       sync.Once
	// ensuredConvColumns maps a conversation collection name to its
	// *conversationScalarMigration, gating the one-time scalar-column migration to
	// once per collection per process. See ensureConversationScalarColumnsOnce.
	ensuredConvColumns sync.Map
	// ensuredMmapEnabled records the collections this process has confirmed
	// mmap-migrated, so the daemon's periodic mmap sweep skips them with no RPC.
	// See ensureMmapEnabledOnce.
	ensuredMmapEnabled sync.Map
	// ensuredBackfill records the conversation collections this process has
	// scalar-column backfilled, so the daemon's periodic backfill sweep runs the
	// metadata-only backfill at most once per collection per process.
	ensuredBackfill sync.Map
}

// NewService constructs the semantic search runtime.
func NewService(ctx context.Context, cfg config.Config) (*Service, error) {
	if strings.TrimSpace(cfg.MilvusAddress) == "" {
		return &Service{
			cfg:                cfg,
			embedder:           nil,
			milvus:             nil,
			available:          atomic.Bool{},
			reconnectCancel:    nil,
			reconnectDone:      nil,
			closeOnce:          sync.Once{},
			ensuredConvColumns: sync.Map{},
			ensuredMmapEnabled: sync.Map{},
			ensuredBackfill:    sync.Map{},
		}, nil
	}

	embedder, err := embedding.NewProvider(cfg)
	if err != nil {
		slog.ErrorContext(ctx, "create embedding provider failed", "provider", cfg.EmbeddingProvider, "err", err)
		return nil, fmt.Errorf("create embedding provider: %w", err)
	}

	service := &Service{
		cfg:                cfg,
		embedder:           embedder,
		milvus:             nil,
		available:          atomic.Bool{},
		reconnectCancel:    nil,
		reconnectDone:      nil,
		closeOnce:          sync.Once{},
		ensuredConvColumns: sync.Map{},
		ensuredMmapEnabled: sync.Map{},
		ensuredBackfill:    sync.Map{},
	}

	client, err := service.dialMilvus(ctx)
	if err != nil {
		slog.WarnContext(ctx, "connect to Milvus failed; starting degraded semantic service", "address", cfg.MilvusAddress, "err", err)
		service.startReconnector(context.WithoutCancel(ctx))
		return service, nil
	}

	service.publishClient(client)
	return service, nil
}

// Close shuts down external resources held by the semantic service.
func (service *Service) Close(ctx context.Context) error {
	if service == nil {
		return nil
	}

	var closeErr error
	service.closeOnce.Do(func() {
		if service.reconnectCancel != nil {
			service.reconnectCancel()
		}
		if service.reconnectDone != nil {
			select {
			case <-service.reconnectDone:
			case <-ctx.Done():
				closeErr = fmt.Errorf("wait for Milvus reconnect shutdown: %w", ctx.Err())
				return
			}
		}
		if !service.Available() || service.milvus == nil {
			return
		}
		if err := service.milvus.Close(ctx); err != nil {
			slog.ErrorContext(ctx, "close Milvus client failed", "err", err)
			closeErr = fmt.Errorf("close Milvus client: %w", err)
			return
		}
		service.available.Store(false)
	})
	return closeErr
}

// Available reports whether semantic indexing is configured.
func (service *Service) Available() bool {
	return service != nil && service.available.Load()
}

// Degraded reports that Milvus is configured but not yet connected.
func (service *Service) Degraded() bool {
	return service != nil && strings.TrimSpace(service.cfg.MilvusAddress) != "" && !service.Available()
}

// conversationPathPrefix marks a virtual conversation collection's canonical
// path. A path with this prefix is not a filesystem directory; its collection
// name derives from the trailing collection id rather than a path hash, so the
// shared embed, staging, and count functions address the conversation
// collection when handed the conversation codebase's canonical path.
const conversationPathPrefix = "chat:///"

// conversationCollectionIDFromPath returns the conversation collection id
// encoded in a canonical path and whether the path is a conversation path.
func conversationCollectionIDFromPath(codebasePath string) (string, bool) {
	if !strings.HasPrefix(codebasePath, conversationPathPrefix) {
		return "", false
	}
	return strings.TrimPrefix(codebasePath, conversationPathPrefix), true
}

// CollectionName matches the TypeScript collection naming contract at
// packages/core/src/context.ts:275 so the Go daemon reads and writes the
// same Milvus collections as the upstream TS adapter. A conversation canonical
// path resolves to the conversation collection so every shared embed, staging,
// and count function addresses the right collection from the codebase path
// alone.
func (service *Service) CollectionName(codebasePath string) string {
	if collectionID, isConversation := conversationCollectionIDFromPath(codebasePath); isConversation {
		return service.ConversationCollectionName(collectionID)
	}

	prefix := "code_chunks"
	if service.cfg.HybridMode {
		prefix = "hybrid_code_chunks"
	}

	normalizedPath, err := filepath.Abs(codebasePath)
	if err != nil {
		normalizedPath = codebasePath
	}
	pathHash := tshash.PathPrefix(normalizedPath)

	override := strings.TrimSpace(service.cfg.CollectionNameOverride)
	if override == "" {
		return prefix + "_" + pathHash
	}

	hashSuffix := "_" + pathHash
	maxReadableLength := maxCollectionNameLength - len(prefix) - 1 - len(hashSuffix)
	sanitized := sanitizeCollectionSuffix(override)
	if len(sanitized) > maxReadableLength {
		sanitized = sanitized[:maxReadableLength]
	}
	if sanitized == "" {
		sanitized = "custom"
	}
	return prefix + "_" + sanitized + hashSuffix
}

// ConversationCollectionName returns the Milvus collection name for a virtual
// conversation document collection.
func (service *Service) ConversationCollectionName(collectionID string) string {
	_ = service
	return "conv_chunks_" + tshash.PathPrefix(strings.TrimSpace(collectionID))
}

func (service *Service) renameCollection(ctx context.Context, oldName string, newName string) error {
	if err := service.milvus.RenameCollection(ctx, milvusclient.NewRenameCollectionOption(oldName, newName)); err != nil {
		slog.ErrorContext(ctx, "rename Milvus collection failed", "from", oldName, "to", newName, "err", err)
		return fmt.Errorf("rename Milvus collection %s to %s: %w", oldName, newName, err)
	}
	return nil
}

// Reindex applies a per-item delta against an existing live collection.
//
// removal deletes the item's prior rows (a code file by exact relativePath, a
// conversation by relativePath prefix). The chunk batch is then embedded and
// inserted through the same batched flow the staging build uses. Reindex
// returns ErrCollectionMissing when the live collection no longer exists, so
// callers can fall back to a full staging build.
func (service *Service) Reindex(ctx context.Context, codebasePath string, addedOrModifiedChunks []model.StoredChunk, removal Removal, progress func(Progress), reuse map[string][]float32) (err error) {
	ctx, done := spans.Open(ctx, "semantic.reindex")
	defer done(&err)

	if !service.Available() {
		return nil
	}

	collectionName := service.CollectionName(codebasePath)
	hasCollection, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		slog.ErrorContext(ctx, "check Milvus collection for reindex failed", "collection", collectionName, "err", err)
		return fmt.Errorf("check Milvus collection %s: %w", collectionName, err)
	}
	if !hasCollection {
		return ErrCollectionMissing
	}

	if !removal.Empty() {
		if err := service.deleteByRemoval(ctx, collectionName, removal); err != nil {
			return err
		}
	}

	if len(addedOrModifiedChunks) == 0 {
		return nil
	}
	addedOrModifiedChunks = service.guardrailExpand(ctx, codebasePath, addedOrModifiedChunks, "reindex")
	return service.insertChunksBatched(ctx, collectionName, addedOrModifiedChunks, true, "Reindexing changed files...", progress, reuse)
}

// PruneToCurrent removes rows whose relativePath is outside the provided
// set of current files. Use it after a streaming reindex to drop chunks
// left over from files that no longer exist on disk.
func (service *Service) PruneToCurrent(ctx context.Context, codebasePath string, currentRelativePaths []string) error {
	if !service.Available() {
		return nil
	}
	if len(currentRelativePaths) == 0 {
		return nil
	}
	collectionName := service.CollectionName(codebasePath)
	hasCollection, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		slog.ErrorContext(ctx, "check Milvus collection for prune failed", "collection", collectionName, "err", err)
		return fmt.Errorf("check Milvus collection %s: %w", collectionName, err)
	}
	if !hasCollection {
		return ErrCollectionMissing
	}

	quoted := make([]string, 0, len(currentRelativePaths))
	for _, path := range currentRelativePaths {
		quoted = append(quoted, `"`+escapeMilvusString(path)+`"`)
	}
	expression := fmt.Sprintf(`%s not in [%s]`, relativePathFieldName, strings.Join(quoted, ","))

	if _, err := service.milvus.Delete(ctx, milvusclient.NewDeleteOption(collectionName).WithExpr(expression)); err != nil {
		slog.ErrorContext(ctx, "prune orphans failed", "collection", collectionName, "current_count", len(currentRelativePaths), "err", err)
		return fmt.Errorf("prune orphans from %s: %w", collectionName, err)
	}
	return nil
}

// deleteByRelativePaths removes existing chunks for the given relative paths.
// Paths are escaped to be safe inside the Milvus filter expression.
func (service *Service) deleteByRelativePaths(ctx context.Context, collectionName string, relativePaths []string) error {
	if len(relativePaths) == 0 {
		return nil
	}

	quoted := make([]string, 0, len(relativePaths))
	for _, path := range relativePaths {
		quoted = append(quoted, `"`+escapeMilvusString(path)+`"`)
	}
	expression := fmt.Sprintf(`%s in [%s]`, relativePathFieldName, strings.Join(quoted, ","))

	if _, err := service.milvus.Delete(ctx, milvusclient.NewDeleteOption(collectionName).WithExpr(expression)); err != nil {
		slog.ErrorContext(ctx, "delete by relative path failed", "collection", collectionName, "count", len(relativePaths), "err", err)
		return fmt.Errorf("delete from %s by relative path: %w", collectionName, err)
	}
	return nil
}

// escapeMilvusString matches the TS escape rule at context.ts:412 and quotes
// any backslash or double quote in a filter string.
func escapeMilvusString(value string) string {
	value = strings.ReplaceAll(value, `\`, `\\`)
	value = strings.ReplaceAll(value, `"`, `\"`)
	return value
}

// Search executes semantic or hybrid search against the configured collection.
//
// relativePathPrefix scopes the search to one subtree of the collection: when
// it is a non-empty relative directory, only rows whose relativePath equals it
// or descends from it are returned. The covering-codebase resolution uses this
// so a query aimed at a nested directory of a larger index returns only that
// directory's chunks.
func (service *Service) Search(ctx context.Context, codebasePath string, query string, limit int32, extensionFilter []string, relativePathPrefix string) ([]model.StoredChunk, error) {
	if !service.Available() {
		return nil, ErrUnavailable
	}

	collectionName := service.CollectionName(codebasePath)
	return service.searchCollection(ctx, collectionName, query, limit, buildSearchFilter(extensionFilter, []string{relativePathPrefix}))
}

// queryTextForEmbedding applies the configured query instruction prefix to
// the dense query embed. The sparse (BM25) leg keeps the raw query text, and
// stored document vectors are never prefixed, so the index stays valid.
func (service *Service) queryTextForEmbedding(query string) string {
	prefix := service.cfg.QueryInstructionPrefix
	if prefix == "" {
		return query
	}
	return prefix + query
}

func (service *Service) searchCollection(ctx context.Context, collectionName string, query string, limit int32, filterExpr string) ([]model.StoredChunk, error) {
	hasCollection, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		slog.ErrorContext(ctx, "check Milvus collection failed", "collection", collectionName, "err", err)
		return nil, fmt.Errorf("check Milvus collection %s: %w", collectionName, err)
	}
	if !hasCollection {
		return nil, ErrCollectionMissing
	}

	queryVector, err := service.embedder.Embed(ctx, service.queryTextForEmbedding(query))
	if err != nil {
		slog.ErrorContext(ctx, "embed query failed", "err", err)
		return nil, fmt.Errorf("embed query: %w", err)
	}

	return service.searchCollectionWithVector(ctx, collectionName, queryVector, query, int(limit), 0, filterExpr)
}

// searchCollectionWithVector runs one search at the given offset using a
// precomputed dense query vector, so a paged caller embeds the query exactly
// once and reuses the vector across pages. rawQuery feeds the BM25 sparse leg,
// which is lexical and never embeds. The caller confirms the collection exists;
// offset zero is an ordinary first-page search.
func (service *Service) searchCollectionWithVector(ctx context.Context, collectionName string, queryVector []float32, rawQuery string, limit int, offset int, filterExpr string) ([]model.StoredChunk, error) {
	searchLimit := limit
	if searchLimit <= 0 {
		searchLimit = 10
	}

	outputFields := []string{
		contentFieldName,
		relativePathFieldName,
		startLineFieldName,
		endLineFieldName,
		fileExtensionFieldName,
		metadataFieldName,
	}
	if isConversationCollection(collectionName) {
		// Conversation collections carry workspaceRoot as a native scalar column.
		// Request it so a workspace_roots post-filter on the daemon side sees the
		// real value rather than the empty default; code collections have no such
		// column, so they keep the base output set.
		outputFields = append(outputFields, workspaceRootFieldName)
	}

	if service.cfg.HybridMode {
		denseRequest := milvusclient.NewAnnRequest(denseVectorFieldName, maxInt(searchLimit, 10), entity.FloatVector(queryVector))
		sparseRequest := milvusclient.NewAnnRequest(sparseVectorFieldName, maxInt(searchLimit, 10), entity.Text(rawQuery))
		if filterExpr != "" {
			denseRequest = denseRequest.WithFilter(filterExpr)
			sparseRequest = sparseRequest.WithFilter(filterExpr)
		}
		hybridOption := milvusclient.NewHybridSearchOption(
			collectionName,
			searchLimit,
			denseRequest,
			sparseRequest,
		).WithReranker(milvusclient.NewRRFReranker()).WithOutputFields(outputFields...)
		if offset > 0 {
			hybridOption = hybridOption.WithOffset(offset)
		}
		resultSets, err := service.milvus.HybridSearch(ctx, hybridOption)
		if err != nil {
			return nil, searchErr(ctx, "hybrid search", collectionName, err)
		}
		return resultSetsToChunks(resultSets)
	}

	searchOption := milvusclient.NewSearchOption(
		collectionName,
		searchLimit,
		[]entity.Vector{entity.FloatVector(queryVector)},
	).WithANNSField(denseVectorFieldName).WithOutputFields(outputFields...)
	if filterExpr != "" {
		searchOption = searchOption.WithFilter(filterExpr)
	}
	if offset > 0 {
		searchOption = searchOption.WithOffset(offset)
	}

	resultSets, err := service.milvus.Search(ctx, searchOption)
	if err != nil {
		return nil, searchErr(ctx, "dense search", collectionName, err)
	}
	return resultSetsToChunks(resultSets)
}

// searchErr logs a Milvus search failure and maps it to a typed store sentinel
// when one applies, otherwise wraps it with the operation and collection for
// context.
func searchErr(ctx context.Context, operation string, collectionName string, err error) error {
	slog.ErrorContext(ctx, operation+" failed", "collection", collectionName, "err", err)
	if sentinel := storeSearchSentinel(err); sentinel != nil {
		return sentinel
	}
	return fmt.Errorf("%s collection %s: %w", operation, collectionName, err)
}

// Drop removes one semantic index collection.
func (service *Service) Drop(ctx context.Context, codebasePath string) error {
	if !service.Available() {
		return nil
	}
	return service.dropIfExists(ctx, service.CollectionName(codebasePath))
}

// Count returns the current number of chunk rows in one semantic collection.
// It asks Milvus to count the collection directly with a count(*) query under
// Strong consistency, so the result includes rows a just-finished run wrote
// and excludes deleted rows. The store is the single source of this number;
// the daemon keeps no separate running tally that could drift from it.
func (service *Service) Count(ctx context.Context, codebasePath string) (int32, error) {
	if !service.Available() {
		return 0, ErrUnavailable
	}

	collectionName := service.CollectionName(codebasePath)
	resultSet, err := service.milvus.Query(ctx, milvusclient.NewQueryOption(collectionName).
		WithOutputFields(countOutputField).
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		slog.ErrorContext(ctx, "count collection rows failed", "collection", collectionName, "err", err)
		return 0, fmt.Errorf("count collection %s: %w", collectionName, err)
	}

	countColumn := resultSet.GetColumn(countOutputField)
	if countColumn == nil {
		slog.ErrorContext(ctx, "count query missing count column", "collection", collectionName, "err", errors.New("missing count(*) column"))
		return 0, errors.New("milvus count query missing count(*) column")
	}
	total, err := countColumn.GetAsInt64(0)
	if err != nil {
		slog.ErrorContext(ctx, "read count column failed", "collection", collectionName, "err", err)
		return 0, fmt.Errorf("read count(*) column for %s: %w", collectionName, err)
	}
	return safeInt32FromInt64(total), nil
}

// ListCollections returns the current semantic collection names from Milvus.
func (service *Service) ListCollections(ctx context.Context) ([]string, error) {
	if !service.Available() {
		return nil, ErrUnavailable
	}
	collections, err := service.milvus.ListCollections(ctx, milvusclient.NewListCollectionOption())
	if err != nil {
		slog.ErrorContext(ctx, "list Milvus collections failed", "err", err)
		return nil, fmt.Errorf("list Milvus collections: %w", err)
	}
	return collections, nil
}

// HasCollectionForPath reports whether Milvus has the collection for the
// given codebase path.
func (service *Service) HasCollectionForPath(ctx context.Context, codebasePath string) (bool, error) {
	if !service.Available() {
		return false, ErrUnavailable
	}
	collectionName := service.CollectionName(codebasePath)
	hasCollection, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		slog.ErrorContext(ctx, "check Milvus collection presence failed", "collection", collectionName, "err", err)
		if storeUnavailable(err) {
			return false, adapterr.NewMilvusUnavailable(fmt.Errorf("check Milvus collection %s: %w", collectionName, err))
		}
		return false, fmt.Errorf("check Milvus collection %s: %w", collectionName, err)
	}
	return hasCollection, nil
}

func (service *Service) dropIfExists(ctx context.Context, collectionName string) error {
	hasCollection, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		slog.ErrorContext(ctx, "check Milvus collection before drop failed", "collection", collectionName, "err", err)
		return fmt.Errorf("check Milvus collection %s: %w", collectionName, err)
	}
	if !hasCollection {
		return nil
	}
	if err := service.milvus.DropCollection(ctx, milvusclient.NewDropCollectionOption(collectionName)); err != nil {
		slog.ErrorContext(ctx, "drop Milvus collection failed", "collection", collectionName, "err", err)
		return fmt.Errorf("drop Milvus collection %s: %w", collectionName, err)
	}
	return nil
}

func (service *Service) insertBatch(ctx context.Context, collectionName string, chunks []model.StoredChunk, vectors [][]float32) (err error) {
	ctx, done := spans.Open(ctx, "semantic.insertBatch")
	defer done(&err)

	insertOption := milvusclient.NewColumnBasedInsertOption(collectionName)
	conversationCollection := isConversationCollection(collectionName)
	if conversationCollection {
		// A pre-existing conv_chunks_* collection created before the scalar columns
		// existed has no conversationId/provider/etc. fields, so an insert that
		// populates them fails with unknown fields. Run the once-guarded migration
		// first so the columns exist before this batch references them.
		if err := service.ensureConversationScalarColumnsOnce(ctx, collectionName); err != nil {
			return err
		}
	}

	ids := make([]string, 0, len(chunks))
	contents := make([]string, 0, len(chunks))
	relativePaths := make([]string, 0, len(chunks))
	startLines := make([]int64, 0, len(chunks))
	endLines := make([]int64, 0, len(chunks))
	fileExtensions := make([]string, 0, len(chunks))
	metadataValues := make([]string, 0, len(chunks))
	scalars := newConversationScalarColumns(conversationCollection, len(chunks))

	sanitizedCount := 0
	for index, chunk := range chunks {
		content, contentChanged := sanitizeUTF8(chunk.Content)
		relativePath, pathChanged := sanitizeUTF8(chunk.RelativePath)
		fileExtension, extChanged := sanitizeUTF8(chunk.FileExtension)
		metadataValue, metaChanged := sanitizeUTF8(encodeMetadata(chunk))
		if contentChanged || pathChanged || extChanged || metaChanged {
			sanitizedCount++
			slog.WarnContext(ctx, "semantic.sanitized_invalid_utf8", "relative_path", chunk.RelativePath, "start_line", chunk.StartLine, "end_line", chunk.EndLine, "content_changed", contentChanged, "path_changed", pathChanged, "extension_changed", extChanged, "metadata_changed", metaChanged)
		}
		ids = append(ids, generateID(chunk, index))
		contents = append(contents, content)
		relativePaths = append(relativePaths, relativePath)
		startLines = append(startLines, int64(chunk.StartLine))
		endLines = append(endLines, int64(chunk.EndLine))
		fileExtensions = append(fileExtensions, fileExtension)
		metadataValues = append(metadataValues, metadataValue)
		scalars.append(chunk)
	}
	if sanitizedCount > 0 {
		slog.WarnContext(ctx, "semantic.insertBatch sanitized chunks before Milvus marshal", "collection", collectionName, "sanitized", sanitizedCount, "batch_size", len(chunks))
	}

	insertOption = insertOption.
		WithVarcharColumn(idFieldName, ids).
		WithVarcharColumn(contentFieldName, contents).
		WithVarcharColumn(relativePathFieldName, relativePaths).
		WithInt64Column(startLineFieldName, startLines).
		WithInt64Column(endLineFieldName, endLines).
		WithVarcharColumn(fileExtensionFieldName, fileExtensions).
		WithVarcharColumn(metadataFieldName, metadataValues).
		WithFloatVectorColumn(denseVectorFieldName, len(vectors[0]), vectors)
	if conversationCollection {
		insertOption = insertOption.
			WithVarcharColumn(conversationIDFieldName, scalars.conversationIDs).
			WithVarcharColumn(parentConversationIDFieldName, scalars.parentConversationIDs).
			WithVarcharColumn(roleFieldName, scalars.roles).
			WithVarcharColumn(providerFieldName, scalars.providers).
			WithVarcharColumn(workspaceRootFieldName, scalars.workspaceRoots).
			WithBoolColumn(archivedFieldName, scalars.archiveds).
			WithInt64Column(timestampUnixFieldName, scalars.timestamps).
			WithInt64Column(messageIndexFieldName, scalars.messageIndexes)
	}

	if _, err := service.milvus.Insert(ctx, insertOption); err != nil {
		slog.ErrorContext(ctx, "insert Milvus batch failed", "collection", collectionName, "err", err)
		return fmt.Errorf("insert Milvus batch into %s: %w", collectionName, err)
	}
	return nil
}

func resultSetsToChunks(resultSets []milvusclient.ResultSet) ([]model.StoredChunk, error) {
	if len(resultSets) == 0 {
		return []model.StoredChunk{}, nil
	}

	resultSet := resultSets[0]
	if resultSet.ResultCount == 0 {
		return []model.StoredChunk{}, nil
	}
	contentColumn := resultSet.GetColumn(contentFieldName)
	relativePathColumn := resultSet.GetColumn(relativePathFieldName)
	startLineColumn := resultSet.GetColumn(startLineFieldName)
	endLineColumn := resultSet.GetColumn(endLineFieldName)
	fileExtensionColumn := resultSet.GetColumn(fileExtensionFieldName)
	metadataColumn := resultSet.GetColumn(metadataFieldName)
	// workspaceRoot is only present on conversation-collection result sets, where
	// the search requests the native scalar column. It is nil for code
	// collections and on rows that never carried a workspace root, so reads stay
	// optional and default to empty.
	workspaceRootColumn := resultSet.GetColumn(workspaceRootFieldName)
	if contentColumn == nil || relativePathColumn == nil || startLineColumn == nil || endLineColumn == nil || fileExtensionColumn == nil {
		return nil, ErrSearchResultIncomplete
	}

	chunks := make([]model.StoredChunk, 0, resultSet.ResultCount)
	for index := range resultSet.ResultCount {
		contentValue, err := contentColumn.GetAsString(index)
		if err != nil {
			slog.Error("read content column failed", "index", index, "err", err)
			return nil, fmt.Errorf("read content column at %d: %w", index, err)
		}
		relativePathValue, err := relativePathColumn.GetAsString(index)
		if err != nil {
			slog.Error("read relative path column failed", "index", index, "err", err)
			return nil, fmt.Errorf("read relative path column at %d: %w", index, err)
		}
		startLineValue, err := startLineColumn.GetAsInt64(index)
		if err != nil {
			slog.Error("read start line column failed", "index", index, "err", err)
			return nil, fmt.Errorf("read start line column at %d: %w", index, err)
		}
		endLineValue, err := endLineColumn.GetAsInt64(index)
		if err != nil {
			slog.Error("read end line column failed", "index", index, "err", err)
			return nil, fmt.Errorf("read end line column at %d: %w", index, err)
		}
		fileExtensionValue, err := fileExtensionColumn.GetAsString(index)
		if err != nil {
			slog.Error("read file extension column failed", "index", index, "err", err)
			return nil, fmt.Errorf("read file extension column at %d: %w", index, err)
		}
		metadataValue := emptyChunkMetadata()
		if metadataColumn != nil {
			rawMetadata, metadataErr := metadataColumn.GetAsString(index)
			if metadataErr == nil {
				metadataValue = decodeMetadata(rawMetadata)
			}
		}

		workspaceRootValue := ""
		if workspaceRootColumn != nil {
			if rootValue, rootErr := workspaceRootColumn.GetAsString(index); rootErr == nil {
				workspaceRootValue = rootValue
			}
		}

		score := 0.0
		if index < len(resultSet.Scores) {
			score = float64(resultSet.Scores[index])
		}
		chunks = append(chunks, model.StoredChunk{
			Content:              contentValue,
			RelativePath:         relativePathValue,
			StartLine:            safeInt32FromInt64(startLineValue),
			EndLine:              safeInt32FromInt64(endLineValue),
			Language:             metadataValue.Language,
			FileExtension:        fileExtensionValue,
			ConversationID:       metadataValue.ConversationID,
			ParentConversationID: metadataValue.ParentConversationID,
			MessageIndex:         metadataValue.messageIndex(),
			Role:                 metadataValue.Role,
			TimestampUnix:        metadataValue.timestampUnix(),
			WorkspaceRoot:        workspaceRootValue,
			Archived:             false,
			Score:                score,
		})
	}
	return chunks, nil
}

// sanitizeUTF8 returns a copy of value with invalid UTF-8 byte sequences
// replaced by the Unicode replacement character. Milvus rejects VarChar
// payloads with invalid UTF-8 at the gRPC marshal boundary, so any chunk
// content that survives the file-level skip but still slices through a
// multi-byte codepoint (for example from a tree-sitter byte-offset
// boundary) gets repaired here. The second return value reports whether
// the input needed repair so callers can log the event.
func sanitizeUTF8(value string) (string, bool) {
	if utf8.ValidString(value) {
		return value, false
	}
	return strings.ToValidUTF8(value, "�"), true
}

// milvusVarcharMaxBytes mirrors the schema's WithMaxLength(65535) for VarChar
// fields. Chunks longer than this fail the Milvus insert with "length of
// varchar field content exceeds max length". The splitter is supposed to
// keep every chunk under chunk_size (default 2500), so an oversize chunk
// at insert time signals a splitter regression. The expansion in
// expandOversizeChunks turns the splitter regression into multiple rows
// rather than a dropped insert, so no content is lost.
const milvusVarcharMaxBytes = 65000

// guardrailExpand wraps expandOversizeChunks with logging. Each oversize
// chunk hitting this path signals an upstream splitter regression that
// emitted content longer than chunk_size. The log carries the codebase
// path, the operation that requested the embed, and the relative path of
// the offending file so the regression can be diagnosed without losing
// the data.
func (service *Service) guardrailExpand(ctx context.Context, codebasePath string, chunks []model.StoredChunk, operation string) []model.StoredChunk {
	ctx, done := spans.Open(ctx, "semantic.guardrailExpand")
	defer done(nil)

	expanded, changed := expandOversizeChunks(chunks)
	if !changed {
		return chunks
	}
	offenders := make([]string, 0)
	for _, chunk := range chunks {
		if len(chunk.Content) > milvusVarcharMaxBytes {
			offenders = append(offenders, fmt.Sprintf("%s:%d-%d (%d bytes)", chunk.RelativePath, chunk.StartLine, chunk.EndLine, len(chunk.Content)))
		}
	}
	slog.WarnContext(ctx, "semantic.expanded_oversize_chunks", "codebase_path", codebasePath, "operation", operation, "expanded_from", len(chunks), "expanded_to", len(expanded), "max_bytes", milvusVarcharMaxBytes, "offenders", offenders)
	return expanded
}

// expandOversizeChunks returns a list where any chunk over
// milvusVarcharMaxBytes has been split into multiple chunks aligned to
// codepoint boundaries. The boolean reports whether any expansion
// happened so the caller can log the upstream regression.
func expandOversizeChunks(chunks []model.StoredChunk) ([]model.StoredChunk, bool) {
	expanded := false
	for _, chunk := range chunks {
		if len(chunk.Content) > milvusVarcharMaxBytes {
			expanded = true
			break
		}
	}
	if !expanded {
		return chunks, false
	}
	out := make([]model.StoredChunk, 0, len(chunks))
	for _, chunk := range chunks {
		if len(chunk.Content) <= milvusVarcharMaxBytes {
			out = append(out, chunk)
			continue
		}
		for _, piece := range splitForVarchar(chunk.Content) {
			child := chunk
			child.Content = piece
			out = append(out, child)
		}
	}
	return out, true
}

// splitForVarchar cuts value into sub-strings of at most
// milvusVarcharMaxBytes bytes, each ending on a UTF-8 codepoint boundary.
func splitForVarchar(value string) []string {
	out := make([]string, 0, (len(value)+milvusVarcharMaxBytes-1)/milvusVarcharMaxBytes)
	start := 0
	for start < len(value) {
		end := start + milvusVarcharMaxBytes
		if end >= len(value) {
			out = append(out, value[start:])
			break
		}
		for end > start && !utf8.RuneStart(value[end]) {
			end--
		}
		if end == start {
			_, size := utf8.DecodeRuneInString(value[start:])
			end = start + size
		}
		out = append(out, value[start:end])
		start = end
	}
	return out
}

// generateID matches the TS chunk-ID format at packages/core/src/context.ts:1067.
func generateID(chunk model.StoredChunk, _ int) string {
	hashInput := fmt.Sprintf("%s:%d:%d:%s", chunk.RelativePath, chunk.StartLine, chunk.EndLine, chunk.Content)
	sum := sha256.Sum256([]byte(hashInput))
	return "chunk_" + hex.EncodeToString(sum[:])[:16]
}

func sanitizeCollectionSuffix(value string) string {
	var builder strings.Builder
	for _, runeValue := range value {
		switch {
		case runeValue >= 'A' && runeValue <= 'Z':
			builder.WriteRune(runeValue)
		case runeValue >= 'a' && runeValue <= 'z':
			builder.WriteRune(runeValue)
		case runeValue >= '0' && runeValue <= '9':
			builder.WriteRune(runeValue)
		default:
			builder.WriteByte('_')
		}
	}
	return builder.String()
}

func maxInt(values ...int) int {
	currentMax := 0
	for _, value := range values {
		if value > currentMax {
			currentMax = value
		}
	}
	return currentMax
}

// ValidateExtensionFilter returns the normalized extension list or an error if any entry is invalid.
func ValidateExtensionFilter(extensionFilter []string) ([]string, error) {
	cleanedExtensions := normalizeExtensionFilter(extensionFilter)
	invalidExtensions := make([]string, 0)
	for _, extension := range cleanedExtensions {
		if !isValidExtension(extension) {
			invalidExtensions = append(invalidExtensions, extension)
		}
	}
	if len(invalidExtensions) > 0 {
		err := fmt.Errorf("invalid file extensions in extensionFilter: %v. Use proper extensions like '.ts', '.py'", invalidExtensions)
		slog.Error("validate extension filter failed", "err", err)
		return nil, err
	}
	return cleanedExtensions, nil
}

// DeduplicateChunks removes overlapping results from the same file.
func DeduplicateChunks(chunks []model.StoredChunk) []model.StoredChunk {
	keptChunks := make([]model.StoredChunk, 0, len(chunks))

	for _, chunk := range chunks {
		hasOverlap := false
		for _, existingChunk := range keptChunks {
			if existingChunk.RelativePath != chunk.RelativePath {
				continue
			}
			overlapStart := maxInt32(existingChunk.StartLine, chunk.StartLine)
			overlapEnd := minInt32(existingChunk.EndLine, chunk.EndLine)
			if overlapStart > overlapEnd {
				continue
			}
			overlapSize := overlapEnd - overlapStart + 1
			chunkSize := chunk.EndLine - chunk.StartLine + 1
			if chunkSize > 0 && float64(overlapSize)/float64(chunkSize) > 0.5 {
				hasOverlap = true
				break
			}
		}
		if !hasOverlap {
			keptChunks = append(keptChunks, chunk)
		}
	}

	return keptChunks
}

// normalizeExtensionFilter trims whitespace and prepends a leading dot when
// missing so the filter matches the dot-prefixed values that [filepath.Ext]
// writes into the file_extension column.
func normalizeExtensionFilter(extensionFilter []string) []string {
	cleanedExtensions := make([]string, 0, len(extensionFilter))
	for _, extension := range extensionFilter {
		trimmedExtension := strings.TrimSpace(extension)
		if trimmedExtension == "" {
			continue
		}
		if !strings.HasPrefix(trimmedExtension, ".") {
			trimmedExtension = "." + trimmedExtension
		}
		cleanedExtensions = append(cleanedExtensions, trimmedExtension)
	}
	return cleanedExtensions
}

func isValidExtension(extension string) bool {
	if !strings.HasPrefix(extension, ".") || len(extension) <= 1 {
		return false
	}
	for _, runeValue := range extension {
		if unicode.IsSpace(runeValue) {
			return false
		}
	}
	return true
}

func maxInt32(left int32, right int32) int32 {
	if left > right {
		return left
	}
	return right
}

func minInt32(left int32, right int32) int32 {
	if left < right {
		return left
	}
	return right
}

func safeInt32FromInt(value int) int32 {
	if value > math.MaxInt32 {
		return math.MaxInt32
	}
	if value < math.MinInt32 {
		return math.MinInt32
	}
	return int32(value)
}

func safeInt32FromInt64(value int64) int32 {
	if value > math.MaxInt32 {
		return math.MaxInt32
	}
	if value < math.MinInt32 {
		return math.MinInt32
	}
	return int32(value)
}

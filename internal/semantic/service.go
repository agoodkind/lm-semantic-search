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

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
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
	splitPartFieldName      = "splitPart"
	contentHashFieldName    = "contentHash"
	embeddingModelFieldName = "embeddingModel"
	countOutputField        = "count(*)"
)

// Progress reports semantic indexing progress after chunk extraction.
//
// ChunksReused counts chunks served a vector from the reuse map, ChunksEmbedded
// counts chunks whose vector came from the embedder, and ChunksDropped counts
// refused inputs that could not be split safely.
type Progress struct {
	Phase                     string
	OverallPercent            float64
	EmbeddingBatchesTotal     int32
	EmbeddingBatchesCompleted int32
	CollectionRowsWritten     int32
	ChunksProcessed           int32
	ChunksReused              int32
	ChunksEmbedded            int32
	ChunksDropped             int32
}

// CollectionFacts reports the live store facts for one collection name.
type CollectionFacts struct {
	Exists    bool
	Rows      int32
	RowsKnown bool
}

type insertRowsFunc func(
	context.Context,
	milvusclient.InsertOption,
) (milvusclient.InsertResult, error)

// BackendName names this service for any surface reporting which backend is in
// use, so the answer comes from the service that exists rather than the setting
// that asked for one.
func (service *Service) BackendName() model.VectorBackend {
	return config.IndexBackendMilvus
}

// EmbeddingProviderName is what this service's embedder calls itself. A service
// constructed with no embedder reports none rather than naming one, because an
// absent embedder is a fact a caller needs and a guessed name would hide it.
func (service *Service) EmbeddingProviderName() model.EmbeddingProvider {
	if service.embedder == nil {
		return model.EmbeddingProviderNone
	}
	return service.embedder.ProviderName()
}

// Service owns the embedding provider and Milvus client for semantic search.
type Service struct {
	cfg                     config.Config
	embedder                embedding.Provider
	milvus                  *milvusclient.Client
	insertRows              insertRowsFunc
	available               atomic.Bool
	reconnectCancel         context.CancelFunc
	reconnectDone           chan struct{}
	closeOnce               sync.Once
	reuseCatalogReady       sync.Map
	reuseCatalogMutex       sync.Mutex
	reuseCatalogAppendMutex sync.Mutex
	// collectionLoads collapses concurrent initial load, wait, and recovery work
	// for the same collection name into one shared flight.
	collectionLoads collectionLoadCoordinator
	residency       *collectionResidencyController
	// ensuredConvColumns maps a conversation collection name to its
	// *conversationScalarMigration, gating the one-time scalar-column migration to
	// once per collection per process. See ensureConversationScalarColumnsOnce.
	ensuredConvColumns sync.Map
	// ensuredSplitPartColumns gates the nullable splitPart schema migration once
	// per collection per process.
	ensuredSplitPartColumns sync.Map
	// ensuredReuseIdentityColumns gates the nullable content hash and embedding
	// model columns plus the content-hash index once per collection per process.
	ensuredReuseIdentityColumns sync.Map
	// mmapPolicyVersions records the mmap policy version fully verified for each
	// collection. Lifecycle and metadata changes invalidate the stored version.
	mmapPolicyVersions   map[string]int
	mmapPolicyMutex      sync.Mutex
	mmapPolicyGeneration map[string]uint64
	mmapPolicyFailures   map[string]mmapPolicyFailure
	// ensuredBackfill records the conversation collections this process has
	// scalar-column backfilled, so the daemon's periodic backfill sweep runs the
	// metadata-only backfill at most once per collection per process.
	ensuredBackfill sync.Map
}

// NewService constructs the semantic search runtime.
func NewService(ctx context.Context, cfg config.Config) (*Service, error) {
	if strings.TrimSpace(cfg.MilvusAddress) == "" {
		service := &Service{
			cfg:                     cfg,
			embedder:                nil,
			milvus:                  nil,
			insertRows:              nil,
			available:               atomic.Bool{},
			reconnectCancel:         nil,
			reconnectDone:           nil,
			closeOnce:               sync.Once{},
			reuseCatalogReady:       sync.Map{},
			reuseCatalogMutex:       sync.Mutex{},
			reuseCatalogAppendMutex: sync.Mutex{},
			collectionLoads: collectionLoadCoordinator{
				mutex:   sync.Mutex{},
				flights: nil,
			},
			residency:                   nil,
			ensuredConvColumns:          sync.Map{},
			ensuredSplitPartColumns:     sync.Map{},
			ensuredReuseIdentityColumns: sync.Map{},
			mmapPolicyVersions:          make(map[string]int),
			mmapPolicyMutex:             sync.Mutex{},
			mmapPolicyGeneration:        make(map[string]uint64),
			mmapPolicyFailures:          make(map[string]mmapPolicyFailure),
			ensuredBackfill:             sync.Map{},
		}
		service.initializeResidencyController()
		return service, nil
	}

	embedder, err := embedding.NewProvider(ctx, cfg)
	if err != nil {
		slog.ErrorContext(ctx, "create embedding provider failed", "provider", cfg.EmbeddingProvider, "err", err)
		return nil, fmt.Errorf("create embedding provider: %w", err)
	}

	service := &Service{
		cfg:                     cfg,
		embedder:                embedder,
		milvus:                  nil,
		insertRows:              nil,
		available:               atomic.Bool{},
		reconnectCancel:         nil,
		reconnectDone:           nil,
		closeOnce:               sync.Once{},
		reuseCatalogReady:       sync.Map{},
		reuseCatalogMutex:       sync.Mutex{},
		reuseCatalogAppendMutex: sync.Mutex{},
		collectionLoads: collectionLoadCoordinator{
			mutex:   sync.Mutex{},
			flights: nil,
		},
		residency:                   nil,
		ensuredConvColumns:          sync.Map{},
		ensuredSplitPartColumns:     sync.Map{},
		ensuredReuseIdentityColumns: sync.Map{},
		mmapPolicyVersions:          make(map[string]int),
		mmapPolicyMutex:             sync.Mutex{},
		mmapPolicyGeneration:        make(map[string]uint64),
		mmapPolicyFailures:          make(map[string]mmapPolicyFailure),
		ensuredBackfill:             sync.Map{},
	}
	service.initializeResidencyController()

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
			}
		}
		if service.residency != nil {
			if err := service.residency.Close(ctx); err != nil {
				slog.ErrorContext(ctx, "close collection residency controller failed", "err", err)
				closeErr = errors.Join(
					closeErr,
					fmt.Errorf("close collection residency controller: %w", err),
				)
			}
		}
		if !service.Available() || service.milvus == nil {
			return
		}
		if err := service.milvus.Close(ctx); err != nil {
			slog.ErrorContext(ctx, "close Milvus client failed", "err", err)
			closeErr = errors.Join(closeErr, fmt.Errorf("close Milvus client: %w", err))
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
		return wrapStoreError(ctx, err, "rename Milvus collection "+oldName+" to "+newName)
	}
	service.invalidateCollectionCaches(oldName)
	service.invalidateCollectionCaches(newName)
	return nil
}

func (service *Service) invalidateCollectionCaches(collectionName string) {
	service.ensuredConvColumns.Delete(collectionName)
	service.ensuredSplitPartColumns.Delete(collectionName)
	service.ensuredReuseIdentityColumns.Delete(collectionName)
	service.invalidateMmapPolicy(collectionName)
	service.ensuredBackfill.Delete(collectionName)
}

func (service *Service) invalidateMmapPolicy(collectionName string) {
	service.mmapPolicyMutex.Lock()
	defer service.mmapPolicyMutex.Unlock()
	if service.mmapPolicyGeneration == nil {
		service.mmapPolicyGeneration = make(map[string]uint64)
	}
	service.mmapPolicyGeneration[collectionName]++
	delete(service.mmapPolicyVersions, collectionName)
	delete(service.mmapPolicyFailures, collectionName)
}

// hasCollection centralizes collection-presence probes and invalidates every
// per-name lifecycle cache when Milvus confirms the collection is absent. A
// shared collection can be dropped and recreated by another process, so no
// cached schema or index migration remains valid after a confirmed absence.
func (service *Service) hasCollection(
	ctx context.Context,
	collectionName string,
	operation string,
) (bool, error) {
	hasCollection, err := service.milvus.HasCollection(
		ctx,
		milvusclient.NewHasCollectionOption(collectionName),
	)
	if err != nil {
		return false, wrapStoreError(ctx, err, operation)
	}
	if !hasCollection {
		service.invalidateCollectionCaches(collectionName)
	}
	return hasCollection, nil
}

// Reindex applies a per-item delta against an existing live collection.
//
// removal deletes the item's prior rows (a code file by exact relativePath, a
// conversation by relativePath prefix). The chunk batch is then embedded and
// inserted through the same batched flow the staging build uses. Reindex
// returns ErrCollectionMissing when the live collection no longer exists, so
// callers can fall back to a full staging build.
func (service *Service) Reindex(ctx context.Context, codebasePath string, addedOrModifiedChunks []model.StoredChunk, removal Removal, progress func(Progress), reuse map[string][]float32, columnSet StoreColumnSet) (err error) {
	ctx, done := spans.Open(ctx, "semantic.reindex")
	defer done(&err)

	if !service.Available() {
		return nil
	}

	collectionName := service.CollectionName(codebasePath)
	hasCollection, err := service.hasCollection(ctx, collectionName, "check Milvus collection "+collectionName)
	if err != nil {
		return err
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
	return service.insertChunksBatched(ctx, collectionName, addedOrModifiedChunks, true, "Reindexing changed files...", progress, reuse, columnSet)
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
	hasCollection, err := service.hasCollection(ctx, collectionName, "check Milvus collection "+collectionName)
	if err != nil {
		return err
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
		return wrapStoreError(ctx, err, "prune orphans from "+collectionName)
	}
	return nil
}

// deleteByRelativePaths removes existing chunks for the given relative paths.
// Paths are escaped to be safe inside the Milvus filter expression.
func (service *Service) deleteByRelativePaths(
	ctx context.Context,
	collectionName string,
	relativePaths []string,
) (int64, error) {
	if len(relativePaths) == 0 {
		return 0, nil
	}

	quoted := make([]string, 0, len(relativePaths))
	for _, path := range relativePaths {
		quoted = append(quoted, `"`+escapeMilvusString(path)+`"`)
	}
	expression := fmt.Sprintf(`%s in [%s]`, relativePathFieldName, strings.Join(quoted, ","))

	result, err := service.milvus.Delete(
		ctx,
		milvusclient.NewDeleteOption(collectionName).WithExpr(expression),
	)
	if err != nil {
		return 0, wrapStoreError(
			ctx,
			err,
			"delete from "+collectionName+" by relative path",
		)
	}
	return result.DeleteCount, nil
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
	hasCollection, err := service.hasCollection(ctx, collectionName, "check Milvus collection "+collectionName)
	if err != nil {
		return nil, err
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
	if err := service.ensureSplitPartColumnOnce(ctx, collectionName); err != nil {
		return nil, err
	}
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
		splitPartFieldName,
	}
	if isConversationCollection(collectionName) {
		// Conversation collections carry workspaceRoot as a native scalar column.
		// Request it so a workspace_roots post-filter on the daemon side sees the
		// real value rather than the empty default; code collections have no such
		// column, so they keep the base output set. loadRules rides along so a
		// search hit can report which loading rules produced its message index.
		outputFields = append(outputFields, workspaceRootFieldName, loadRulesFieldName)
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
	return service.collectionRowCount(ctx, collectionName)
}

func (service *Service) collectionRowCount(ctx context.Context, collectionName string) (int32, error) {
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

// InspectCollection reports whether one collection exists and, when possible,
// how many rows it currently holds.
func (service *Service) InspectCollection(ctx context.Context, collectionName string) (CollectionFacts, error) {
	if !service.Available() {
		return CollectionFacts{}, ErrUnavailable
	}

	hasCollection, err := service.hasCollection(ctx, collectionName, "check Milvus collection "+collectionName)
	if err != nil {
		return CollectionFacts{}, err
	}
	if !hasCollection {
		return CollectionFacts{Exists: false, Rows: 0, RowsKnown: false}, nil
	}

	rows, err := service.collectionRowCount(ctx, collectionName)
	if err != nil {
		return collectionExistsWithUnknownRows()
	}
	return CollectionFacts{Exists: true, Rows: rows, RowsKnown: true}, nil
}

// collectionExistsWithUnknownRows keeps Exists true when only the row count
// failed, because a count error must never demote a Present collection: a
// false Missing verdict is what routes callers into a full rebuild.
func collectionExistsWithUnknownRows() (CollectionFacts, error) {
	return CollectionFacts{Exists: true, Rows: 0, RowsKnown: false}, nil
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
	hasCollection, err := service.hasCollection(ctx, collectionName, "check Milvus collection "+collectionName)
	if err != nil {
		return false, err
	}
	return hasCollection, nil
}

func (service *Service) dropIfExists(ctx context.Context, collectionName string) error {
	hasCollection, err := service.hasCollection(ctx, collectionName, "check Milvus collection "+collectionName)
	if err != nil {
		return err
	}
	if !hasCollection {
		return nil
	}
	if err := service.milvus.DropCollection(ctx, milvusclient.NewDropCollectionOption(collectionName)); err != nil {
		return wrapStoreError(ctx, err, "drop Milvus collection "+collectionName)
	}
	service.invalidateCollectionCaches(collectionName)
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
	splitPartColumn := resultSet.GetColumn(splitPartFieldName)
	// workspaceRoot is only present on conversation-collection result sets, where
	// the search requests the native scalar column. It is nil for code
	// collections and on rows that never carried a workspace root, so reads stay
	// optional and default to empty. loadRules follows the same contract.
	workspaceRootColumn := resultSet.GetColumn(workspaceRootFieldName)
	loadRulesColumn := resultSet.GetColumn(loadRulesFieldName)
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

		workspaceRootValue := backfillString(workspaceRootColumn, index)
		loadRulesValue := backfillString(loadRulesColumn, index)
		splitPartValue, splitPartRecorded, splitPartErr := splitPartAt(
			splitPartColumn,
			index,
		)
		if splitPartErr != nil {
			return nil, splitPartErr
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
			SplitPart:            splitPartValue,
			SplitPartRecorded:    splitPartRecorded,
			LoadRules:            loadRulesValue,
			Score:                score,
		})
	}
	return chunks, nil
}

// generateID matches the TS chunk-ID format at packages/core/src/context.ts:1067
// for an unsplit chunk. A split child (SplitPart > 0) folds its split position
// into the hash so identical pieces of repeated oversized content get distinct
// primary keys; an unsplit chunk keeps the original identity so the normal
// single-chunk case is not re-embedded.
func generateID(chunk model.StoredChunk, _ int) string {
	hashInput := fmt.Sprintf("%s:%d:%d:%s", chunk.RelativePath, chunk.StartLine, chunk.EndLine, chunk.Content)
	if chunk.SplitPart > 0 {
		hashInput = fmt.Sprintf("%s:%d:%d:%d:%s", chunk.RelativePath, chunk.StartLine, chunk.EndLine, chunk.SplitPart, chunk.Content)
	}
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

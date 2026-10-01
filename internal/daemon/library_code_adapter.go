package daemon

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/library"
)

const (
	// libraryCodeDirectory is the StateRoot subdirectory of the codebase catalog.
	libraryCodeDirectory = "library/codebase"
	// libraryCodePoolID identifies the codebase vector pool in the catalog.
	libraryCodePoolID = "codebase"
	// libraryCodeCollection is the Milvus collection of the codebase vector pool.
	libraryCodeCollection = "lms_library_codebase"
	// libraryCodeEmbeddingRevision is the model revision the catalog records. A
	// change to how code text is embedded needs a new revision.
	libraryCodeEmbeddingRevision = "1"
	// libraryCodeNormalization records that stored vectors are not normalized.
	libraryCodeNormalization = "none"
	// libraryCodeMaxBatchBytes bounds the text bytes of one staged batch.
	libraryCodeMaxBatchBytes = 8 << 20
	// libraryCodeContentHashLength is the hex length of the content hash in a
	// row key.
	libraryCodeContentHashLength = 16
	// libraryCodeProgressPhase is the job phase while code owners are published.
	libraryCodeProgressPhase = "Reindexing changed files..."
	// libraryCodeNamespacePrefix starts every codebase namespace ID.
	libraryCodeNamespacePrefix = "code_"
	// libraryCodeNamespaceHashLength is the hex length of the path hash in a
	// codebase namespace ID.
	libraryCodeNamespaceHashLength = 16
	// defaultMilvusDatabaseName is the database a Milvus client without a
	// database name uses.
	defaultMilvusDatabaseName = "default"
)

// Scalar columns of a codebase namespace.
const (
	codeColumnRelativePath  = "relative_path"
	codeColumnLanguage      = "language"
	codeColumnFileExtension = "file_extension"
	codeColumnStartLine     = "start_line"
	codeColumnEndLine       = "end_line"
	codeColumnSplitPart     = "split_part"
	codeColumnPart          = "part"

	codeRelativePathMaxBytes = 4096
	codeLanguageMaxBytes     = 64
	codeExtensionMaxBytes    = 64
)

type libraryCodeIndex struct {
	semanticIndex
	store         *library.Library
	vectorClient  *milvusclient.Client
	tokenizer     library.Tokenizer
	maxTokens     int
	maxBytes      int
	maxBatchRows  int
	maxBatchBytes int64
	// codeNamespaces maps each collection name that the wrapped index returned
	// for a filesystem codebase to the codebase namespace ID.
	codeNamespaces sync.Map
	// registered stores the namespaces this process registered.
	registered sync.Map
}

// newLibraryCodeIndex opens the codebase catalog and vector pool for cfg and
// wraps inner. The standard profile uses a Milvus pool and the
// OpenAI-compatible embedder; the offline profile uses a library/embedded pool,
// the ONNX embedder, and its exact tokenizer.
func newLibraryCodeIndex(ctx context.Context, cfg config.Config, inner semanticIndex) (*libraryCodeIndex, error) {
	if cfg.EmbeddingDimension <= 0 {
		err := errors.New("the library codebase store requires EMBEDDING_DIMENSION")
		slog.ErrorContext(ctx, "open library codebase store failed", "err", err)
		return nil, err
	}
	backends, err := newLibraryCodeBackends(ctx, cfg)
	if err != nil {
		return nil, err
	}
	batchRows := max(cfg.EmbeddingBatchSize, 1)
	directory := filepath.Join(cfg.StateRoot, libraryCodeDirectory)
	opened, err := library.Open(ctx, library.Config{
		Store: library.StoreDescriptor{
			CatalogPath:       filepath.Join(directory, "catalog.sqlite"),
			LockPath:          filepath.Join(directory, "catalog.lock"),
			PoolID:            libraryCodePoolID,
			EmbeddingModel:    cfg.EmbeddingModel,
			EmbeddingRevision: libraryCodeEmbeddingRevision,
			Dimension:         int(cfg.EmbeddingDimension),
			Normalization:     libraryCodeNormalization,
		},
		Vectors:                backends.vectors,
		Embedder:               backends.embedder,
		MaxBatchRows:           batchRows,
		MaxBatchBytes:          libraryCodeMaxBatchBytes,
		QueryInstructionPrefix: cfg.QueryInstructionPrefix,
		SearchMode:             librarySearchMode(cfg),
	})
	if err != nil {
		slog.ErrorContext(ctx, "open library codebase catalog failed", "path", directory, "err", err)
		return nil, errors.Join(fmt.Errorf("open library codebase catalog: %w", err), closeLibraryVectorClient(ctx, backends.vectorClient))
	}
	return &libraryCodeIndex{
		semanticIndex:  inner,
		store:          opened,
		vectorClient:   backends.vectorClient,
		tokenizer:      backends.tokenizer,
		maxTokens:      backends.maxTokens,
		maxBytes:       backends.maxBytes,
		maxBatchRows:   batchRows,
		maxBatchBytes:  libraryCodeMaxBatchBytes,
		codeNamespaces: sync.Map{},
		registered:     sync.Map{},
	}, nil
}

// librarySearchMode maps the daemon hybrid setting to the library search mode.
func librarySearchMode(cfg config.Config) library.SearchMode {
	if cfg.HybridMode {
		return library.Hybrid
	}
	return library.Dense
}

func milvusDatabaseName(cfg config.Config) string {
	if cfg.MilvusDatabase == "" {
		return defaultMilvusDatabaseName
	}
	return cfg.MilvusDatabase
}

// libraryCodeMaxTokens returns the model input limit, tightened by a smaller
// configured EmbeddingMaxTokens.
func libraryCodeMaxTokens(cfg config.Config) int {
	limit := config.ActiveEmbedTokenLimit(cfg)
	if cfg.EmbeddingMaxTokens > 0 && cfg.EmbeddingMaxTokens < limit {
		return cfg.EmbeddingMaxTokens
	}
	return limit
}

// wrapDelegated logs and wraps an error from the wrapped index. It returns nil
// for a nil error.
func wrapDelegated(ctx context.Context, operation string, err error) error {
	if err == nil {
		return nil
	}
	slog.WarnContext(ctx, "wrapped semantic index call failed", "operation", operation, "err", err)
	return fmt.Errorf("%s: %w", operation, err)
}

// closeLibraryVectorClient closes client and returns nil for a nil client,
// which the offline profile has.
func closeLibraryVectorClient(ctx context.Context, client *milvusclient.Client) error {
	if client == nil {
		return nil
	}
	if err := client.Close(ctx); err != nil {
		slog.ErrorContext(ctx, "close library codebase vector client failed", "err", err)
		return fmt.Errorf("close library codebase vector client: %w", err)
	}
	return nil
}

// codebaseSpec declares one codebase namespace.
func codebaseSpec(namespace string) library.NamespaceSpec {
	return library.NamespaceSpec{
		ID:     namespace,
		Policy: library.ReplaceAllowed,
		Scalars: []library.ScalarColumn{
			{Name: codeColumnRelativePath, Type: library.String, Nullable: false, Mutable: false, MaxLength: codeRelativePathMaxBytes},
			{Name: codeColumnLanguage, Type: library.String, Nullable: false, Mutable: false, MaxLength: codeLanguageMaxBytes},
			{Name: codeColumnFileExtension, Type: library.String, Nullable: false, Mutable: false, MaxLength: codeExtensionMaxBytes},
			{Name: codeColumnStartLine, Type: library.Int64, Nullable: false, Mutable: false, MaxLength: 0},
			{Name: codeColumnEndLine, Type: library.Int64, Nullable: false, Mutable: false, MaxLength: 0},
			{Name: codeColumnSplitPart, Type: library.Int64, Nullable: false, Mutable: false, MaxLength: 0},
			{Name: codeColumnPart, Type: library.Int64, Nullable: false, Mutable: false, MaxLength: 0},
		},
	}
}

// CollectionName returns the wrapped index's name. For a filesystem codebase it
// stores the codebase namespace under that name.
func (index *libraryCodeIndex) CollectionName(codebasePath string) string {
	if semantic.IsDocumentPath(codebasePath) {
		return index.semanticIndex.CollectionName(codebasePath)
	}
	name, _ := index.codebaseNames(codebasePath)
	return name
}

// codebaseNamespaceID returns "code_" and the first 16 hex characters of the
// SHA-256 of the canonical codebase path.
func codebaseNamespaceID(codebasePath string) string {
	sum := sha256.Sum256([]byte(codebasePath))
	return libraryCodeNamespacePrefix + hex.EncodeToString(sum[:])[:libraryCodeNamespaceHashLength]
}

// codebaseNamespace returns the namespace of codebasePath and stores it under
// the wrapped index's collection name.
func (index *libraryCodeIndex) codebaseNamespace(codebasePath string) string {
	_, namespace := index.codebaseNames(codebasePath)
	return namespace
}

func (index *libraryCodeIndex) codebaseNames(codebasePath string) (string, string) {
	name := index.semanticIndex.CollectionName(codebasePath)
	namespace := codebaseNamespaceID(codebasePath)
	index.codeNamespaces.Store(name, namespace)
	return name, namespace
}

// collectionNamespace returns the codebase namespace stored for a collection
// name, and false for any other name.
func (index *libraryCodeIndex) collectionNamespace(collectionName string) (string, bool) {
	stored, found := index.codeNamespaces.Load(collectionName)
	if !found {
		return "", false
	}
	namespace, isString := stored.(string)
	return namespace, isString
}

func (index *libraryCodeIndex) isCodeNamespace(collectionName string) bool {
	_, found := index.collectionNamespace(collectionName)
	return found
}

// namespace registers and returns the codebase namespace of codebasePath.
func (index *libraryCodeIndex) namespace(ctx context.Context, codebasePath string) (string, error) {
	name := index.codebaseNamespace(codebasePath)
	if _, done := index.registered.Load(name); done {
		return name, nil
	}
	if err := index.store.RegisterNamespace(ctx, codebaseSpec(name)); err != nil {
		slog.ErrorContext(ctx, "register codebase namespace failed", "namespace", name, "err", err)
		return "", fmt.Errorf("register codebase namespace %s: %w", name, err)
	}
	index.registered.Store(name, struct{}{})
	return name, nil
}

// namespaceStats returns the statistics of a codebase namespace and whether it
// is registered.
func (index *libraryCodeIndex) namespaceStats(ctx context.Context, namespace string) (library.NamespaceStats, bool, error) {
	stats, err := index.store.NamespaceStats(ctx, namespace)
	if errors.Is(err, library.ErrInvalidRequest) {
		return library.NamespaceStats{Owners: 0, Occurrences: 0}, false, nil
	}
	if err != nil {
		slog.ErrorContext(ctx, "read codebase namespace statistics failed", "namespace", namespace, "err", err)
		return library.NamespaceStats{Owners: 0, Occurrences: 0}, false, fmt.Errorf("read codebase namespace %s statistics: %w", namespace, err)
	}
	return stats, true, nil
}

func (index *libraryCodeIndex) Reindex(
	ctx context.Context,
	codebasePath string,
	chunks []model.StoredChunk,
	removal semantic.Removal,
	progress func(semantic.Progress),
	reuse map[string][]float32,
	columnSet semantic.StoreColumnSet,
) error {
	if semantic.IsDocumentPath(codebasePath) {
		return wrapDelegated(ctx, "reindex", index.semanticIndex.Reindex(ctx, codebasePath, chunks, removal, progress, reuse, columnSet))
	}
	return index.replaceOwners(ctx, codebasePath, chunks, removal, progress)
}

// StageReindex publishes a codebase path in the same way as Reindex. Every
// owner generation commits atomically, and a first build writes no staging
// collection.
func (index *libraryCodeIndex) StageReindex(
	ctx context.Context,
	codebasePath string,
	chunks []model.StoredChunk,
	removal semantic.Removal,
	progress func(semantic.Progress),
	reuse map[string][]float32,
	columnSet semantic.StoreColumnSet,
) error {
	if semantic.IsDocumentPath(codebasePath) {
		return wrapDelegated(ctx, "stage reindex", index.semanticIndex.StageReindex(ctx, codebasePath, chunks, removal, progress, reuse, columnSet))
	}
	return index.replaceOwners(ctx, codebasePath, chunks, removal, progress)
}

// replaceOwners commits one generation for every owner in removal and chunks.
// An owner with no chunks gets an empty generation, which removes its
// published rows.
func (index *libraryCodeIndex) replaceOwners(
	ctx context.Context,
	codebasePath string,
	chunks []model.StoredChunk,
	removal semantic.Removal,
	progress func(semantic.Progress),
) error {
	if len(removal.ItemIDs) > 0 {
		err := fmt.Errorf("codebase removal by item column %q is not supported", removal.ItemColumn)
		slog.ErrorContext(ctx, "codebase reindex rejected", "path", codebasePath, "err", err)
		return err
	}
	namespace, err := index.namespace(ctx, codebasePath)
	if err != nil {
		return err
	}
	owners := make(map[string][]model.StoredChunk, len(removal.Paths)+1)
	for _, path := range removal.Paths {
		owners[path] = nil
	}
	if len(removal.Prefixes) > 0 {
		if err := index.addPrefixOwners(ctx, namespace, removal.Prefixes, owners); err != nil {
			return err
		}
	}
	for _, chunk := range chunks {
		owners[chunk.RelativePath] = append(owners[chunk.RelativePath], chunk)
	}
	ownerIDs := make([]string, 0, len(owners))
	for ownerID := range owners {
		ownerIDs = append(ownerIDs, ownerID)
	}
	slices.Sort(ownerIDs)
	published := 0
	for _, ownerID := range ownerIDs {
		parts, err := index.replaceOwner(ctx, namespace, ownerID, owners[ownerID])
		if err != nil {
			return err
		}
		published += parts
		if progress != nil {
			progress(semantic.Progress{
				Phase:                     libraryCodeProgressPhase,
				OverallPercent:            0,
				EmbeddingBatchesTotal:     0,
				EmbeddingBatchesCompleted: 0,
				CollectionRowsWritten:     safeInt32(published),
				ChunksProcessed:           safeInt32(published),
				ChunksReused:              0,
				ChunksEmbedded:            0,
				ChunksDropped:             0,
			})
		}
	}
	return nil
}

func (index *libraryCodeIndex) addPrefixOwners(ctx context.Context, namespace string, prefixes []string, owners map[string][]model.StoredChunk) error {
	published, err := index.store.ListOwners(ctx, namespace)
	if err != nil {
		slog.ErrorContext(ctx, "list codebase owners failed", "namespace", namespace, "err", err)
		return fmt.Errorf("list codebase %s owners: %w", namespace, err)
	}
	for _, ownerID := range published {
		for _, prefix := range prefixes {
			if strings.HasPrefix(ownerID, prefix) {
				owners[ownerID] = nil
				break
			}
		}
	}
	return nil
}

// replaceOwner stages every part of one file in bounded batches and commits
// the complete generation once. The generation order is one above the
// committed order, and the token is the manifest hash. A restarted write of
// the same content at the same order stages the same rows again, and Stage
// accepts identical staged rows.
func (index *libraryCodeIndex) replaceOwner(ctx context.Context, namespace string, ownerID string, chunks []model.StoredChunk) (int, error) {
	state, err := index.store.GetOwnerState(ctx, namespace, ownerID)
	if err != nil {
		slog.ErrorContext(ctx, "read codebase owner state failed", "owner", ownerID, "err", err)
		return 0, fmt.Errorf("read codebase owner %s state: %w", ownerID, err)
	}
	if len(chunks) == 0 && state.GenerationOrder == 0 {
		return 0, nil
	}
	rows, err := index.codeOccurrences(ctx, chunks)
	if err != nil {
		return 0, err
	}
	seal, err := library.SealRows(rows)
	if err != nil {
		slog.ErrorContext(ctx, "seal codebase owner failed", "owner", ownerID, "err", err)
		return 0, fmt.Errorf("seal codebase owner %s: %w", ownerID, err)
	}
	key := library.GenerationKey{
		Namespace:        namespace,
		OwnerID:          ownerID,
		GenerationOrder:  state.GenerationOrder + 1,
		IdempotencyToken: seal.ManifestHash,
	}
	for _, batch := range index.stageBatches(rows) {
		if err := index.store.Stage(ctx, library.StageBatch{Key: key, Mode: library.Replace, Rows: batch}); err != nil {
			slog.ErrorContext(ctx, "stage codebase owner failed", "owner", ownerID, "rows", len(batch), "err", err)
			return 0, fmt.Errorf("stage codebase owner %s: %w", ownerID, err)
		}
	}
	if _, err := index.store.CommitGeneration(ctx, key, seal); err != nil {
		slog.ErrorContext(ctx, "commit codebase owner failed", "owner", ownerID, "err", err)
		return 0, fmt.Errorf("commit codebase owner %s: %w", ownerID, err)
	}
	return len(rows), nil
}

// stageBatches splits rows into batches of at most maxBatchRows rows and
// maxBatchBytes text bytes. It returns one empty batch for no rows, because a
// commit requires a staged generation.
func (index *libraryCodeIndex) stageBatches(rows []library.Occurrence) [][]library.Occurrence {
	if len(rows) == 0 {
		return [][]library.Occurrence{nil}
	}
	batches := make([][]library.Occurrence, 0, len(rows)/index.maxBatchRows+1)
	var current []library.Occurrence
	var currentBytes int64
	for _, row := range rows {
		rowBytes := int64(len(row.SourceText) + len(row.SearchText) + len(row.EmbeddingInput))
		full := len(current) == index.maxBatchRows || currentBytes+rowBytes > index.maxBatchBytes
		if len(current) > 0 && full {
			batches = append(batches, current)
			current = nil
			currentBytes = 0
		}
		current = append(current, row)
		currentBytes += rowBytes
	}
	return append(batches, current)
}

// codeOccurrences prepares every chunk at the model limit and returns one
// occurrence per part. It replaces NUL bytes with spaces in lexical and
// embedding inputs, preserves source bytes, and skips whitespace-only chunks.
func (index *libraryCodeIndex) codeOccurrences(ctx context.Context, chunks []model.StoredChunk) ([]library.Occurrence, error) {
	rows := make([]library.Occurrence, 0, len(chunks))
	for _, chunk := range chunks {
		content := strings.ReplaceAll(chunk.Content, "\x00", " ")
		if strings.TrimSpace(content) == "" {
			continue
		}
		parts, err := library.PrepareText(ctx, library.PrepareRequest{
			Text:           content,
			DocumentPrefix: "",
			MaxTokens:      index.maxTokens,
			MaxBytes:       index.maxBytes,
			Tokenizer:      index.tokenizer,
		})
		if err != nil {
			slog.ErrorContext(ctx, "prepare code chunk failed", "path", chunk.RelativePath, "start_line", chunk.StartLine, "err", err)
			return nil, fmt.Errorf("prepare %s:%d: %w", chunk.RelativePath, chunk.StartLine, err)
		}
		contentHash := sha256.Sum256([]byte(chunk.Content))
		hashText := hex.EncodeToString(contentHash[:])[:libraryCodeContentHashLength]
		for _, part := range parts {
			rows = append(rows, codeOccurrence(chunk, content, hashText, part))
		}
	}
	return rows, nil
}

func codeOccurrence(chunk model.StoredChunk, content string, hashText string, part library.PreparedPart) library.Occurrence {
	partNumber, _ := strconv.Atoi(part.Suffix)
	text := content[part.ByteStart:part.ByteEnd]
	return library.Occurrence{
		RowKey:         fmt.Sprintf("%d:%d:%d:%s:%s", chunk.StartLine, chunk.EndLine, chunk.SplitPart, hashText, part.Suffix),
		SortKey:        fmt.Sprintf("%010d:%010d:%010d:%010d", chunk.StartLine, chunk.EndLine, chunk.SplitPart, partNumber),
		SourceText:     chunk.Content[part.ByteStart:part.ByteEnd],
		SearchText:     text,
		EmbeddingInput: part.EmbeddingInput,
		Scalars: map[string]library.ScalarValue{
			codeColumnRelativePath:  libraryStringScalar(chunk.RelativePath),
			codeColumnLanguage:      libraryStringScalar(chunk.Language),
			codeColumnFileExtension: libraryStringScalar(chunk.FileExtension),
			codeColumnStartLine:     libraryInt64Scalar(int64(chunk.StartLine)),
			codeColumnEndLine:       libraryInt64Scalar(int64(chunk.EndLine)),
			codeColumnSplitPart:     libraryInt64Scalar(int64(chunk.SplitPart)),
			codeColumnPart:          libraryInt64Scalar(int64(partNumber)),
		},
	}
}

func libraryStringScalar(value string) library.ScalarValue {
	return library.ScalarValue{Type: library.String, Null: false, String: value, Bool: false, Int64: 0}
}

func libraryInt64Scalar(value int64) library.ScalarValue {
	return library.ScalarValue{Type: library.Int64, Null: false, String: "", Bool: false, Int64: value}
}

// Drop removes every published owner of a codebase path with an empty
// generation. Canonical vectors stay in the pool.
func (index *libraryCodeIndex) Drop(ctx context.Context, codebasePath string) error {
	if semantic.IsDocumentPath(codebasePath) {
		return wrapDelegated(ctx, "drop", index.semanticIndex.Drop(ctx, codebasePath))
	}
	return index.removeOwners(ctx, codebasePath, nil)
}

// PruneToCurrent removes every published owner of a codebase path outside
// currentRelativePaths.
func (index *libraryCodeIndex) PruneToCurrent(ctx context.Context, codebasePath string, currentRelativePaths []string) error {
	if semantic.IsDocumentPath(codebasePath) {
		return wrapDelegated(ctx, "prune to current", index.semanticIndex.PruneToCurrent(ctx, codebasePath, currentRelativePaths))
	}
	return index.removeOwners(ctx, codebasePath, currentRelativePaths)
}

// removeOwners commits an empty generation for every published owner of a
// codebase path that keep does not list.
func (index *libraryCodeIndex) removeOwners(ctx context.Context, codebasePath string, keep []string) error {
	namespace := index.codebaseNamespace(codebasePath)
	_, registered, err := index.namespaceStats(ctx, namespace)
	if err != nil || !registered {
		return err
	}
	owners, err := index.store.ListOwners(ctx, namespace)
	if err != nil {
		slog.ErrorContext(ctx, "list codebase owners failed", "namespace", namespace, "err", err)
		return fmt.Errorf("list codebase %s owners: %w", namespace, err)
	}
	kept := make(map[string]struct{}, len(keep))
	for _, path := range keep {
		kept[path] = struct{}{}
	}
	for _, ownerID := range owners {
		if _, found := kept[ownerID]; found {
			continue
		}
		if _, err := index.replaceOwner(ctx, namespace, ownerID, nil); err != nil {
			return err
		}
	}
	return nil
}

// Count returns the published occurrence count of a codebase path.
func (index *libraryCodeIndex) Count(ctx context.Context, codebasePath string) (int32, error) {
	if semantic.IsDocumentPath(codebasePath) {
		value, err := index.semanticIndex.Count(ctx, codebasePath)
		return value, wrapDelegated(ctx, "count", err)
	}
	stats, _, err := index.namespaceStats(ctx, index.codebaseNamespace(codebasePath))
	if err != nil {
		return 0, err
	}
	return safeInt32(int(stats.Occurrences)), nil
}

// HasCollectionForPath reports whether a codebase path has a registered
// namespace.
func (index *libraryCodeIndex) HasCollectionForPath(ctx context.Context, codebasePath string) (bool, error) {
	if semantic.IsDocumentPath(codebasePath) {
		value, err := index.semanticIndex.HasCollectionForPath(ctx, codebasePath)
		return value, wrapDelegated(ctx, "has collection for path", err)
	}
	_, registered, err := index.namespaceStats(ctx, index.codebaseNamespace(codebasePath))
	return registered, err
}

// InspectCollection reports the namespace facts of a codebase collection name
// and passes any other name to the wrapped index.
func (index *libraryCodeIndex) InspectCollection(ctx context.Context, collectionName string) (semantic.CollectionFacts, error) {
	namespace, isCode := index.collectionNamespace(collectionName)
	if !isCode {
		value, err := index.semanticIndex.InspectCollection(ctx, collectionName)
		return value, wrapDelegated(ctx, "inspect collection", err)
	}
	stats, registered, err := index.namespaceStats(ctx, namespace)
	if err != nil {
		return semantic.CollectionFacts{Exists: false, Rows: 0, RowsKnown: false}, err
	}
	return semantic.CollectionFacts{Exists: registered, Rows: safeInt32(int(stats.Occurrences)), RowsKnown: registered}, nil
}

// ObserveCollection reports a registered codebase namespace as ready with its
// occurrence count.
func (index *libraryCodeIndex) ObserveCollection(ctx context.Context, codebasePath string) (semantic.CollectionObservation, error) {
	if semantic.IsDocumentPath(codebasePath) {
		value, err := index.semanticIndex.ObserveCollection(ctx, codebasePath)
		return value, wrapDelegated(ctx, "observe collection", err)
	}
	stats, registered, err := index.namespaceStats(ctx, index.codebaseNamespace(codebasePath))
	if err != nil {
		return semantic.CollectionObservation{State: semantic.CollectionStateUnknown, Rows: 0, RowsKnown: false}, err
	}
	if !registered {
		return semantic.CollectionObservation{State: semantic.CollectionStateAbsent, Rows: 0, RowsKnown: false}, nil
	}
	return semantic.CollectionObservation{State: semantic.CollectionStateReady, Rows: safeInt32(int(stats.Occurrences)), RowsKnown: true}, nil
}

// CollectionState reports a registered codebase namespace as existing and
// loaded.
func (index *libraryCodeIndex) CollectionState(ctx context.Context, codebasePath string) (bool, bool, error) {
	if semantic.IsDocumentPath(codebasePath) {
		exists, loaded, err := index.semanticIndex.CollectionState(ctx, codebasePath)
		return exists, loaded, wrapDelegated(ctx, "collection state", err)
	}
	_, registered, err := index.namespaceStats(ctx, index.codebaseNamespace(codebasePath))
	return registered, registered, err
}

// HasStaging reports no staging collection for a codebase path.
func (index *libraryCodeIndex) HasStaging(ctx context.Context, codebasePath string) (bool, error) {
	if semantic.IsDocumentPath(codebasePath) {
		value, err := index.semanticIndex.HasStaging(ctx, codebasePath)
		return value, wrapDelegated(ctx, "has staging", err)
	}
	return false, nil
}

// PinStaging returns a no-op pin for a codebase path.
func (index *libraryCodeIndex) PinStaging(ctx context.Context, codebasePath string) (semantic.CollectionPin, error) {
	if semantic.IsDocumentPath(codebasePath) {
		value, err := index.semanticIndex.PinStaging(ctx, codebasePath)
		return value, wrapDelegated(ctx, "pin staging", err)
	}
	return semantic.NoopCollectionPin{}, nil
}

// PromoteStaging returns nil for a codebase path. StageReindex already
// committed every owner generation.
func (index *libraryCodeIndex) PromoteStaging(ctx context.Context, codebasePath string) error {
	if semantic.IsDocumentPath(codebasePath) {
		return wrapDelegated(ctx, "promote staging", index.semanticIndex.PromoteStaging(ctx, codebasePath))
	}
	return nil
}

// DropStaging returns nil for a codebase path, which has no staging
// collection.
func (index *libraryCodeIndex) DropStaging(ctx context.Context, codebasePath string) error {
	if semantic.IsDocumentPath(codebasePath) {
		return wrapDelegated(ctx, "drop staging", index.semanticIndex.DropStaging(ctx, codebasePath))
	}
	return nil
}

// CopyChunks copies no rows for a codebase path. The caller then indexes the
// destination file, and the library reuses the stored vector of every
// identical embedding input.
func (index *libraryCodeIndex) CopyChunks(ctx context.Context, codebasePath string, srcRelativePath string, dstRelativePath string) (int, error) {
	if semantic.IsDocumentPath(codebasePath) {
		value, err := index.semanticIndex.CopyChunks(ctx, codebasePath, srcRelativePath, dstRelativePath)
		return value, wrapDelegated(ctx, "copy chunks", err)
	}
	return 0, nil
}

// LoadReuseVectors returns no vectors for codebase collection names. The
// library reuses stored vectors by embedding input identity.
func (index *libraryCodeIndex) LoadReuseVectors(ctx context.Context, collectionNames []string) (map[string][]float32, error) {
	others := make([]string, 0, len(collectionNames))
	for _, name := range collectionNames {
		if !index.isCodeNamespace(name) {
			others = append(others, name)
		}
	}
	if len(others) == 0 {
		return map[string][]float32{}, nil
	}
	value, err := index.semanticIndex.LoadReuseVectors(ctx, others)
	return value, wrapDelegated(ctx, "load reuse vectors", err)
}

// LoadReuseVectorsForPrefix returns no vectors for a codebase collection name.
func (index *libraryCodeIndex) LoadReuseVectorsForPrefix(ctx context.Context, collectionName string, relativePathPrefix string) (map[string][]float32, error) {
	if index.isCodeNamespace(collectionName) {
		return map[string][]float32{}, nil
	}
	value, err := index.semanticIndex.LoadReuseVectorsForPrefix(ctx, collectionName, relativePathPrefix)
	return value, wrapDelegated(ctx, "load reuse vectors for prefix", err)
}

// LoadReuseVectorsForPath returns no vectors for a codebase collection name.
func (index *libraryCodeIndex) LoadReuseVectorsForPath(ctx context.Context, collectionName string, relativePath string) (map[string][]float32, error) {
	if index.isCodeNamespace(collectionName) {
		return map[string][]float32{}, nil
	}
	value, err := index.semanticIndex.LoadReuseVectorsForPath(ctx, collectionName, relativePath)
	return value, wrapDelegated(ctx, "load reuse vectors for path", err)
}

// LoadReuseVectorsForContents returns no vectors for a codebase collection
// name.
func (index *libraryCodeIndex) LoadReuseVectorsForContents(ctx context.Context, collectionName string, chunks []model.StoredChunk) (map[string][]float32, error) {
	if index.isCodeNamespace(collectionName) {
		return map[string][]float32{}, nil
	}
	value, err := index.semanticIndex.LoadReuseVectorsForContents(ctx, collectionName, chunks)
	return value, wrapDelegated(ctx, "load reuse vectors for contents", err)
}

// Close closes the catalog, the vector client, and the wrapped index.
func (index *libraryCodeIndex) Close(ctx context.Context) error {
	var closeErr error
	if err := index.store.Close(); err != nil {
		slog.ErrorContext(ctx, "close library codebase catalog failed", "err", err)
		closeErr = errors.Join(closeErr, fmt.Errorf("close library codebase catalog: %w", err))
	}
	closeErr = errors.Join(closeErr, closeLibraryVectorClient(ctx, index.vectorClient))
	if closer, ok := index.semanticIndex.(semanticCloser); ok {
		if err := closer.Close(ctx); err != nil {
			slog.ErrorContext(ctx, "close wrapped semantic index failed", "err", err)
			closeErr = errors.Join(closeErr, fmt.Errorf("close wrapped semantic index: %w", err))
		}
	}
	return closeErr
}

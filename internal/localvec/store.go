// Package localvec provides the embedded vector store for the offline profile.
package localvec

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/embedding"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/tshash"
)

const (
	collectionHashLength      = 16
	stagingCollectionSuffix   = ".staging"
	backupCollectionSuffix    = ".backup"
	conversationPathPrefix    = "chat:///"
	localCollectionNamePrefix = "local_code_chunks_"
)

var errEmbeddingProviderUnconfigured = errors.New(
	"local vector store embedding provider is not configured",
)

// Store is the embedded vector store used by the offline profile.
type Store struct {
	cfg         config.Config
	root        string
	embedder    embedding.Provider
	mutex       sync.RWMutex
	collections map[string]*collection
	available   bool
}

// New constructs an embedded vector store.
func New(ctx context.Context, cfg config.Config) (*Store, error) {
	var provider embedding.Provider
	if strings.TrimSpace(cfg.EmbeddingProvider) != "" {
		configuredProvider, err := embedding.NewProvider(cfg)
		if err != nil {
			slog.ErrorContext(
				ctx,
				"create local vector embedding provider failed",
				"provider",
				cfg.EmbeddingProvider,
				"err",
				err,
			)
			return nil, fmt.Errorf("create local vector embedding provider: %w", err)
		}
		provider = configuredProvider
	}
	return newStoreWithProvider(cfg, provider)
}

func newStoreWithProvider(
	cfg config.Config,
	provider embedding.Provider,
) (*Store, error) {
	root := filepath.Join(cfg.StateRoot, "localvec")
	if err := os.MkdirAll(root, 0o700); err != nil {
		slog.Error("create local vector store directory failed", "path", root, "err", err)
		return nil, fmt.Errorf("create local vector store directory %s: %w", root, err)
	}
	store := &Store{
		cfg:         cfg,
		root:        root,
		embedder:    provider,
		mutex:       sync.RWMutex{},
		collections: make(map[string]*collection),
		available:   true,
	}
	if err := store.recoverCollectionBackups(); err != nil {
		return nil, err
	}
	return store, nil
}

// Available reports whether the embedded vector store can serve requests.
func (store *Store) Available() bool {
	return store != nil && store.available
}

// CollectionName returns the local collection name for a codebase path.
func (store *Store) CollectionName(codebasePath string) string {
	if collectionID, found := strings.CutPrefix(codebasePath, conversationPathPrefix); found {
		return store.ConversationCollectionName(collectionID)
	}
	resolvedPath := codebasePath
	absolutePath, err := filepath.Abs(codebasePath)
	if err == nil {
		resolvedPath = absolutePath
	}
	evaluatedPath, err := filepath.EvalSymlinks(resolvedPath)
	if err == nil {
		resolvedPath = evaluatedPath
	}
	sum := sha256.Sum256([]byte(resolvedPath))
	pathHash := hex.EncodeToString(sum[:])[:collectionHashLength]
	return localCollectionNamePrefix + pathHash
}

// ConversationCollectionName returns the collection name for a conversation collection.
func (store *Store) ConversationCollectionName(collectionID string) string {
	return "conv_chunks_" + tshash.PathPrefix(strings.TrimSpace(collectionID))
}

// Count returns the number of stored chunks for a codebase.
func (store *Store) Count(_ context.Context, codebasePath string) (int32, error) {
	stored, err := store.collectionForName(store.CollectionName(codebasePath), false)
	if err != nil {
		return 0, err
	}
	count, exists, err := stored.rowCount()
	if err != nil {
		return 0, err
	}
	if !exists {
		return 0, semantic.ErrCollectionMissing
	}
	return count, nil
}

// ListCollections returns the stored collection names.
func (store *Store) ListCollections(ctx context.Context) ([]string, error) {
	entries, err := os.ReadDir(store.root)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"list local vector collections failed",
			"path",
			store.root,
			"err",
			err,
		)
		return nil, fmt.Errorf("list local vector collections in %s: %w", store.root, err)
	}
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		name := entry.Name()
		if strings.HasPrefix(name, ".") ||
			strings.HasSuffix(name, stagingCollectionSuffix) ||
			strings.HasSuffix(name, backupCollectionSuffix) {
			continue
		}
		if validateCollectionName(name) != nil {
			continue
		}
		metadataPath := filepath.Join(store.root, name, metadataFileName)
		if _, statErr := os.Stat(metadataPath); statErr != nil {
			continue
		}
		names = append(names, name)
	}
	sort.Strings(names)
	return names, nil
}

func (store *Store) recoverCollectionBackups() error {
	entries, err := os.ReadDir(store.root)
	if err != nil {
		slog.Error(
			"inspect local vector collection backups failed",
			"path",
			store.root,
			"err",
			err,
		)
		return fmt.Errorf("inspect local vector collection backups in %s: %w", store.root, err)
	}
	for _, entry := range entries {
		if !entry.IsDir() || !strings.HasSuffix(entry.Name(), backupCollectionSuffix) {
			continue
		}
		pathName := strings.TrimSuffix(entry.Name(), backupCollectionSuffix)
		collectionName := strings.TrimSuffix(pathName, stagingCollectionSuffix)
		if validateCollectionName(collectionName) != nil {
			continue
		}
		if err := recoverCollectionDirectory(filepath.Join(store.root, pathName)); err != nil {
			return err
		}
	}
	return nil
}

// InspectCollection returns facts about a stored collection.
func (store *Store) InspectCollection(
	_ context.Context,
	collectionName string,
) (semantic.CollectionFacts, error) {
	stored, err := store.collectionForName(collectionName, false)
	if err != nil {
		return semantic.CollectionFacts{}, err
	}
	count, exists, err := stored.rowCount()
	if err != nil {
		return semantic.CollectionFacts{}, err
	}
	if !exists {
		return semantic.CollectionFacts{Exists: false, Rows: 0, RowsKnown: false}, nil
	}
	return semantic.CollectionFacts{Exists: true, Rows: count, RowsKnown: true}, nil
}

// HasCollectionForPath reports whether a codebase has a stored collection.
func (store *Store) HasCollectionForPath(
	ctx context.Context,
	codebasePath string,
) (bool, error) {
	facts, err := store.InspectCollection(ctx, store.CollectionName(codebasePath))
	if err != nil {
		return false, err
	}
	return facts.Exists, nil
}

// HasStaging reports whether a codebase has a staging collection.
func (store *Store) HasStaging(_ context.Context, codebasePath string) (bool, error) {
	stored, err := store.collectionForName(store.CollectionName(codebasePath), true)
	if err != nil {
		return false, err
	}
	_, exists, err := stored.rowCount()
	return exists, err
}

// ProbeHealth checks whether the embedded vector store is reachable.
func (store *Store) ProbeHealth(context.Context) error {
	return nil
}

// CollectionState reports whether a codebase collection exists and is loaded.
func (store *Store) CollectionState(
	ctx context.Context,
	codebasePath string,
) (exists bool, loaded bool, err error) {
	exists, err = store.HasCollectionForPath(ctx, codebasePath)
	return exists, true, err
}

// Drop removes a live collection.
func (store *Store) Drop(_ context.Context, codebasePath string) error {
	return store.dropCollection(store.CollectionName(codebasePath), false)
}

// DropStaging removes a staging collection.
func (store *Store) DropStaging(_ context.Context, codebasePath string) error {
	return store.dropCollection(store.CollectionName(codebasePath), true)
}

// EnsureMmapEnabledAllCollections applies store maintenance to all collections.
func (store *Store) EnsureMmapEnabledAllCollections(context.Context) {}

// BackfillConversationCollectionsOnce applies conversation collection maintenance.
func (store *Store) BackfillConversationCollectionsOnce(context.Context) {}

func (store *Store) collectionForName(
	collectionName string,
	staging bool,
) (*collection, error) {
	if err := validateCollectionName(collectionName); err != nil {
		return nil, err
	}
	key := collectionName
	pathName := collectionName
	if staging {
		key += "\x00staging"
		pathName += stagingCollectionSuffix
	}
	store.mutex.RLock()
	stored := store.collections[key]
	store.mutex.RUnlock()
	if stored != nil {
		return stored, nil
	}

	store.mutex.Lock()
	defer store.mutex.Unlock()
	if stored = store.collections[key]; stored != nil {
		return stored, nil
	}
	collectionPath := filepath.Join(store.root, pathName)
	if err := recoverCollectionDirectory(collectionPath); err != nil {
		return nil, err
	}
	stored = newCollection(collectionName, collectionPath)
	store.collections[key] = stored
	return stored, nil
}

func (store *Store) dropCollection(collectionName string, staging bool) error {
	stored, err := store.collectionForName(collectionName, staging)
	if err != nil {
		return err
	}
	if err := stored.drop(); err != nil {
		return err
	}
	key := collectionName
	if staging {
		key += "\x00staging"
	}
	store.mutex.Lock()
	if store.collections[key] == stored {
		delete(store.collections, key)
	}
	store.mutex.Unlock()
	return nil
}

func (store *Store) embeddingProvider() (embedding.Provider, error) {
	if store.embedder == nil {
		return nil, errEmbeddingProviderUnconfigured
	}
	return store.embedder, nil
}

func validateCollectionName(collectionName string) error {
	if collectionName == "" {
		return errors.New("local vector collection name is required")
	}
	for _, character := range collectionName {
		if character >= 'a' && character <= 'z' {
			continue
		}
		if character >= 'A' && character <= 'Z' {
			continue
		}
		if character >= '0' && character <= '9' {
			continue
		}
		if character == '_' || character == '-' {
			continue
		}
		return fmt.Errorf("invalid local vector collection name %q", collectionName)
	}
	return nil
}

func operationContextError(ctx context.Context, operation string) error {
	err := ctx.Err()
	if err == nil {
		return nil
	}
	slog.WarnContext(
		ctx,
		"local vector operation canceled",
		"operation",
		operation,
		"err",
		err,
	)
	return fmt.Errorf("%s: %w", operation, err)
}

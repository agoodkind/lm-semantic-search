// Package localvec provides the embedded vector store for the offline profile.
package localvec

import (
	"context"
	"errors"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

var errNotImplemented = errors.New("local vector store is not implemented")

// Store is the embedded vector store used by the offline profile.
type Store struct{}

// New constructs an embedded vector store.
func New(ctx context.Context, cfg config.Config) (*Store, error) {
	return &Store{}, nil
}

// Available reports whether the embedded vector store can serve requests.
func (store *Store) Available() bool {
	return false
}

// CollectionName returns the collection name for a codebase path.
func (store *Store) CollectionName(codebasePath string) string {
	return ""
}

// ConversationCollectionName returns the collection name for a conversation collection.
func (store *Store) ConversationCollectionName(collectionID string) string {
	return ""
}

// Search searches a codebase collection.
func (store *Store) Search(ctx context.Context, codebasePath string, query string, limit int32, extensionFilter []string, relativePathPrefix string) ([]model.StoredChunk, error) {
	return nil, errNotImplemented
}

// SearchConversationCollectionCapped searches a conversation collection with per-conversation limits.
func (store *Store) SearchConversationCollectionCapped(ctx context.Context, collectionName string, query string, limit int32, perConversationLimit int32, minScore float64, filter semantic.ConversationFilter) ([]model.StoredChunk, error) {
	return nil, errNotImplemented
}

// Count returns the number of stored chunks for a codebase.
func (store *Store) Count(ctx context.Context, codebasePath string) (int32, error) {
	return 0, errNotImplemented
}

// ListCollections returns the stored collection names.
func (store *Store) ListCollections(ctx context.Context) ([]string, error) {
	return nil, errNotImplemented
}

// InspectCollection returns facts about a stored collection.
func (store *Store) InspectCollection(ctx context.Context, collectionName string) (semantic.CollectionFacts, error) {
	return semantic.CollectionFacts{}, errNotImplemented
}

// HasCollectionForPath reports whether a codebase has a stored collection.
func (store *Store) HasCollectionForPath(ctx context.Context, codebasePath string) (bool, error) {
	return false, errNotImplemented
}

// HasStaging reports whether a codebase has a staging collection.
func (store *Store) HasStaging(ctx context.Context, codebasePath string) (bool, error) {
	return false, errNotImplemented
}

// ProbeHealth checks whether the embedded vector store is reachable. The store
// is a local on-disk file, always reachable, so this never reports an outage.
func (store *Store) ProbeHealth(ctx context.Context) error {
	return nil
}

// CollectionState reports whether a codebase collection exists and is loaded.
// The stub holds no collections yet, so it reports absent-and-not-loaded with no
// error, which the daemon renders as not-yet-built rather than an outage. The
// real store (a later task) reports the on-disk collection's true state.
func (store *Store) CollectionState(ctx context.Context, codebasePath string) (exists bool, loaded bool, err error) {
	return false, false, nil
}

// LoadReuseVectors loads reusable vectors from collections.
func (store *Store) LoadReuseVectors(ctx context.Context, collectionNames []string) (map[string][]float32, error) {
	return nil, errNotImplemented
}

// LoadReuseVectorsForPrefix loads reusable vectors matching a relative path prefix.
func (store *Store) LoadReuseVectorsForPrefix(ctx context.Context, collectionName string, relativePathPrefix string) (map[string][]float32, error) {
	return nil, errNotImplemented
}

// LoadReuseVectorsForPath loads reusable vectors matching a relative path.
func (store *Store) LoadReuseVectorsForPath(ctx context.Context, collectionName string, relativePath string) (map[string][]float32, error) {
	return nil, errNotImplemented
}

// LoadConversationDerivedBatch loads stored state for a batch of conversations.
func (store *Store) LoadConversationDerivedBatch(ctx context.Context, collectionName string, conversationIDs []string) (semantic.ConversationBatchState, error) {
	return semantic.ConversationBatchState{}, errNotImplemented
}

// Reindex applies changed chunks and removals to a live collection.
func (store *Store) Reindex(ctx context.Context, codebasePath string, addedOrModifiedChunks []model.StoredChunk, removal semantic.Removal, progress func(semantic.Progress), reuse map[string][]float32, columnSet semantic.StoreColumnSet) error {
	return errNotImplemented
}

// StageReindex applies changed chunks and removals to a staging collection.
func (store *Store) StageReindex(ctx context.Context, codebasePath string, chunks []model.StoredChunk, removal semantic.Removal, progress func(semantic.Progress), reuse map[string][]float32, columnSet semantic.StoreColumnSet) error {
	return errNotImplemented
}

// PromoteStaging promotes a staging collection to its live name.
func (store *Store) PromoteStaging(ctx context.Context, codebasePath string) error {
	return errNotImplemented
}

// DeleteConversation removes one conversation from a collection.
func (store *Store) DeleteConversation(ctx context.Context, collectionName string, conversationID string) error {
	return errNotImplemented
}

// BackfillConversationEnrichment updates stored conversation enrichment.
func (store *Store) BackfillConversationEnrichment(ctx context.Context, collectionName string, enrichment semantic.ConversationEnrichment, dryRun bool) (int, int, error) {
	return 0, 0, errNotImplemented
}

// CopyChunks copies stored chunks between relative paths.
func (store *Store) CopyChunks(ctx context.Context, codebasePath string, srcRelativePath string, dstRelativePath string) (int, error) {
	return 0, errNotImplemented
}

// PruneToCurrent removes chunks whose paths are no longer current.
func (store *Store) PruneToCurrent(ctx context.Context, codebasePath string, currentRelativePaths []string) error {
	return errNotImplemented
}

// Drop removes a live collection.
func (store *Store) Drop(ctx context.Context, codebasePath string) error {
	return errNotImplemented
}

// DropStaging removes a staging collection.
func (store *Store) DropStaging(ctx context.Context, codebasePath string) error {
	return errNotImplemented
}

// EnsureMmapEnabledAllCollections applies store maintenance to all collections.
func (store *Store) EnsureMmapEnabledAllCollections(ctx context.Context) {}

// BackfillConversationCollectionsOnce applies conversation collection maintenance.
func (store *Store) BackfillConversationCollectionsOnce(ctx context.Context) {}

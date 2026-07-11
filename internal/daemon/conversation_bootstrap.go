package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"sort"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// StampFullyEmbeddedConversations is the one-time bootstrap that lets the
// derived-content marker skip engage on an already-embedded corpus. It reads the
// live collection in batches, and for every delivered conversation whose expected
// derived rows are all already present at the current pipeline version it stamps
// the marker = derivedPipelineVersion. It never embeds: a partially embedded
// conversation is left unstamped so the next sync examines and completes it, and
// a conversation the operator omits keeps whatever marker it already had. It
// returns the number of conversations examined and the number newly stamped.
func (manager *Manager) StampFullyEmbeddedConversations(ctx context.Context, collectionID string, documents []model.ConversationDocument) (examined int, stamped int, err error) {
	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		return 0, 0, err
	}
	return manager.stampFullyEmbeddedConversations(ctx, codebase.CollectionName, manager.merklePath(codebase.ID), documents)
}

// stampFullyEmbeddedConversations is the shared bootstrap core. It reads the live
// collection in batches and stamps the derived marker for every delivered
// conversation whose expected rows are all already present, writing the merged
// markers beside the snapshot at snapshotPath. Both the operator CLI-facing
// method and the automatic reexamine trigger call it, so they stamp identically.
func (manager *Manager) stampFullyEmbeddedConversations(ctx context.Context, collectionName string, snapshotPath string, documents []model.ConversationDocument) (examined int, stamped int, err error) {
	if manager.semantic == nil || !manager.semantic.Available() {
		return 0, 0, semantic.ErrUnavailable
	}

	documentsByID := groupConversationDocuments(documents)
	if len(documentsByID) == 0 {
		return 0, 0, nil
	}

	conversationIDs := sortedConversationIDs(documentsByID)
	batch, err := manager.semantic.LoadConversationDerivedBatch(ctx, collectionName, conversationIDs)
	if err != nil {
		slog.ErrorContext(ctx, "load conversation derived batch for bootstrap failed", "collection", collectionName, "err", err)
		return 0, 0, fmt.Errorf("load conversation derived batch for %s: %w", collectionName, err)
	}

	markerPath := conversationDerivedMarkerPath(snapshotPath)
	versions := loadConversationDerivedMarkers(markerPath)
	changed := false
	for _, conversationID := range conversationIDs {
		examined++
		fullyEmbedded, checkErr := conversationFullyEmbedded(ctx, conversationID, documentsByID[conversationID], batch.Rows[conversationID])
		if checkErr != nil {
			// A later conversation's diff errored, but earlier conversations in this
			// run may already be stamped in versions. Persist them before returning so
			// the reported stamped count matches what is durably on disk, rather than
			// claiming stamps that were never written.
			if changed {
				if writeErr := writeConversationDerivedMarkers(markerPath, versions); writeErr != nil {
					slog.ErrorContext(ctx, "persist partial bootstrap markers before returning error failed", "collection", collectionName, "err", writeErr)
					return examined, 0, checkErr
				}
			}
			return examined, stamped, checkErr
		}
		if !fullyEmbedded {
			continue
		}
		if versions[conversationID] == derivedPipelineVersion {
			continue
		}
		versions[conversationID] = derivedPipelineVersion
		changed = true
		stamped++
	}
	if changed {
		if writeErr := writeConversationDerivedMarkers(markerPath, versions); writeErr != nil {
			return examined, stamped, writeErr
		}
	}
	slog.InfoContext(ctx, "conversation.bootstrap_stamped", "collection", collectionName, "examined", examined, "stamped", stamped)
	return examined, stamped, nil
}

// conversationReexamineNeedsBootstrap reports whether an upsert should run the
// one-time stamping pass before planning: only a reexamine backfill whose marker
// store is still empty, which is the migration case. A fresh install also has an
// empty store, but its batched read finds nothing embedded, so the pass stamps
// nothing; a store with any marker has already run, so the pass never repeats
// once it has stamped at least one conversation.
func conversationReexamineNeedsBootstrap(reexamine bool, snapshotPath string) bool {
	if !reexamine {
		return false
	}
	return len(loadConversationDerivedMarkers(conversationDerivedMarkerPath(snapshotPath))) == 0
}

// conversationFullyEmbedded reports whether every expected derived and base row
// for one conversation is already present in the store at the current pipeline
// version. It reuses the examination diff, so "fully embedded" means exactly the
// same no-op the delta path would produce: nothing to embed and nothing to
// remove. A conversation with any stored row missing, mismatched, or stale is not
// fully embedded, so bootstrap leaves it unstamped for a normal examination.
func conversationFullyEmbedded(ctx context.Context, conversationID string, documents []model.ConversationDocument, stored semantic.ConversationStoredRows) (bool, error) {
	if len(documents) == 0 {
		return false, nil
	}
	if stored.Messages == nil {
		stored.Messages = map[int32]semantic.StoredMessageState{}
	}
	if stored.DerivedPaths == nil {
		stored.DerivedPaths = map[string]string{}
	}
	diff, err := diffConversationMessages(ctx, conversationID, documents, stored)
	if err != nil {
		return false, err
	}
	noEmbed := len(diff.documents) == 0
	noRemoval := len(diff.removalPaths) == 0 && len(diff.removalPrefixes) == 0
	return noEmbed && noRemoval, nil
}

func groupConversationDocuments(documents []model.ConversationDocument) map[string][]model.ConversationDocument {
	byID := make(map[string][]model.ConversationDocument)
	for _, document := range documents {
		conversationID := document.ConversationID
		if conversationID == "" {
			continue
		}
		byID[conversationID] = append(byID[conversationID], document)
	}
	return byID
}

func sortedConversationIDs(documentsByID map[string][]model.ConversationDocument) []string {
	conversationIDs := make([]string, 0, len(documentsByID))
	for conversationID := range documentsByID {
		conversationIDs = append(conversationIDs, conversationID)
	}
	sort.Strings(conversationIDs)
	return conversationIDs
}

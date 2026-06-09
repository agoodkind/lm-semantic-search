package semantic

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/spans"
)

// UpsertConversationChunks replaces the stored chunks for every conversation id
// present in chunks, then embeds and inserts the provided pre-chunked messages.
func (service *Service) UpsertConversationChunks(ctx context.Context, collectionName string, chunks []model.StoredChunk, progress func(Progress)) (err error) {
	ctx, done := spans.Open(ctx, "semantic.upsertConversationChunks")
	defer done(&err)

	if !service.Available() {
		return nil
	}
	trimmedCollectionName := strings.TrimSpace(collectionName)
	if trimmedCollectionName == "" {
		return errors.New("conversation collection name is required")
	}

	hasCollection, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(trimmedCollectionName))
	if err != nil {
		slog.ErrorContext(ctx, "check conversation collection failed", "collection", trimmedCollectionName, "err", err)
		return fmt.Errorf("check conversation collection %s: %w", trimmedCollectionName, err)
	}
	if hasCollection {
		// The prefix delete below runs an expression-filtered Delete, which Milvus
		// only serves on a loaded collection. The first ingest creates and loads
		// the collection through insertChunksBatched, but a later ingest reaches
		// this branch without ever loading it, so load it here before deleting.
		if err := service.loadCollection(ctx, trimmedCollectionName); err != nil {
			return err
		}
		for _, conversationID := range conversationIDsFromChunks(chunks) {
			prefix := conversationRelativePathPrefix(conversationID)
			if err := service.deleteByRelativePathPrefix(ctx, trimmedCollectionName, prefix); err != nil {
				return err
			}
		}
	}
	if len(chunks) == 0 {
		return nil
	}
	return service.insertChunksBatched(ctx, trimmedCollectionName, chunks, hasCollection, "Generating conversation embeddings...", progress, nil)
}

// DeleteConversation removes every chunk stored for one conversation id.
func (service *Service) DeleteConversation(ctx context.Context, collectionName string, conversationID string) (err error) {
	ctx, done := spans.Open(ctx, "semantic.deleteConversation")
	defer done(&err)

	if !service.Available() {
		return nil
	}
	trimmedCollectionName := strings.TrimSpace(collectionName)
	if trimmedCollectionName == "" {
		return errors.New("conversation collection name is required")
	}
	trimmedConversationID := strings.TrimSpace(conversationID)
	if trimmedConversationID == "" {
		return errors.New("conversation id is required")
	}

	hasCollection, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(trimmedCollectionName))
	if err != nil {
		slog.ErrorContext(ctx, "check conversation collection before delete failed", "collection", trimmedCollectionName, "err", err)
		return fmt.Errorf("check conversation collection %s: %w", trimmedCollectionName, err)
	}
	if !hasCollection {
		return nil
	}
	// Load before the expression-filtered delete for the same reason as the
	// upsert path: a daemon that did not create this collection never loaded it.
	if err := service.loadCollection(ctx, trimmedCollectionName); err != nil {
		return err
	}
	return service.deleteByRelativePathPrefix(ctx, trimmedCollectionName, conversationRelativePathPrefix(trimmedConversationID))
}

func (service *Service) deleteByRelativePathPrefix(ctx context.Context, collectionName string, prefix string) error {
	if prefix == "" {
		return nil
	}
	expression := fmt.Sprintf(`%s like "%s%%"`, relativePathFieldName, escapeMilvusString(prefix))
	if _, err := service.milvus.Delete(ctx, milvusclient.NewDeleteOption(collectionName).WithExpr(expression)); err != nil {
		slog.ErrorContext(ctx, "delete by relative path prefix failed", "collection", collectionName, "prefix", prefix, "err", err)
		return fmt.Errorf("delete from %s by relative path prefix: %w", collectionName, err)
	}
	return nil
}

func conversationIDsFromChunks(chunks []model.StoredChunk) []string {
	seen := make(map[string]struct{})
	conversationIDs := make([]string, 0)
	for _, chunk := range chunks {
		conversationID := strings.TrimSpace(chunk.ConversationID)
		if conversationID == "" {
			continue
		}
		if _, found := seen[conversationID]; found {
			continue
		}
		seen[conversationID] = struct{}{}
		conversationIDs = append(conversationIDs, conversationID)
	}
	return conversationIDs
}

func conversationRelativePathPrefix(conversationID string) string {
	return "conv/" + conversationID + "/"
}

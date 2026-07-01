package semantic

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/spans"
)

// DeleteConversation removes every chunk stored for one conversation id. The
// manifest-driven sync drops a removed conversation on its own, so this serves
// only an explicit single-conversation delete request.
func (service *Service) DeleteConversation(ctx context.Context, collectionName string, conversationID string) (err error) {
	ctx, done := spans.Open(ctx, "semantic.deleteConversation")
	defer done(&err)

	if !service.Available() {
		return ErrUnavailable
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
	if err := service.loadCollection(ctx, trimmedCollectionName); err != nil {
		return err
	}
	return service.deleteByRelativePathPrefix(ctx, trimmedCollectionName, conversationRelativePathPrefix(trimmedConversationID))
}

// conversationRelativePathPrefix is the relativePath prefix every message row of
// one conversation shares, so a prefix delete drops the whole conversation.
func conversationRelativePathPrefix(conversationID string) string {
	return "conv/" + conversationID + "/"
}

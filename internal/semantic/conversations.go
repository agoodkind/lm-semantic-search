package semantic

import (
	"context"
	"errors"
	"strings"

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

	hasCollection, err := service.hasCollection(
		ctx,
		trimmedCollectionName,
		"check conversation collection "+trimmedCollectionName,
	)
	if err != nil {
		return err
	}
	if !hasCollection {
		return nil
	}
	if err := service.loadCollection(ctx, trimmedCollectionName); err != nil {
		return err
	}
	for _, prefix := range conversationRelativePathPrefixes(trimmedConversationID) {
		if _, err := service.deleteByRelativePathPrefix(
			ctx,
			trimmedCollectionName,
			prefix,
		); err != nil {
			return err
		}
	}
	return nil
}

// conversationRelativePathPrefix is the relativePath prefix every message row of
// one conversation shares, so a prefix delete drops the whole conversation.
func conversationRelativePathPrefix(conversationID string) string {
	return "conv/" + conversationID + "/"
}

func conversationRelativePathPrefixes(conversationID string) []string {
	return []string{
		conversationRelativePathPrefix(conversationID),
		"convtool/" + conversationID + "/",
		"convthink/" + conversationID + "/",
	}
}

package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/model"
)

const conversationCanonicalPathPrefix = "chat:///"

// RegisterConversationCollection records a virtual document collection that is
// addressed by logical collection id rather than a filesystem directory.
func (manager *Manager) RegisterConversationCollection(ctx context.Context, collectionID string) (model.Codebase, error) {
	trimmedCollectionID := strings.TrimSpace(collectionID)
	if trimmedCollectionID == "" {
		return model.Codebase{}, errors.New("collection id is required")
	}
	canonicalPath := conversationCanonicalPath(trimmedCollectionID)

	manager.mu.Lock()
	defer manager.mu.Unlock()

	for _, codebase := range manager.codebases {
		if codebase.Kind != model.CodebaseKindDocument {
			continue
		}
		if codebase.CanonicalPath == canonicalPath {
			return codebase, nil
		}
	}

	collectionName := ""
	if manager.semantic != nil {
		collectionName = manager.semantic.ConversationCollectionName(trimmedCollectionID)
	}
	if collectionName == "" {
		return model.Codebase{}, errors.New("conversation collection name is unavailable")
	}

	codebase := newCodebaseRecord(canonicalPath)
	codebase.Kind = model.CodebaseKindDocument
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = manager.enrichIndexConfig(emptyAutoIndexConfig())
	codebase.EffectiveConfig.IgnoreDigest = digestIndexConfig(codebase.EffectiveConfig)
	codebase.CollectionName = collectionName
	codebase.UpdatedAt = clock.Now()
	manager.codebases[codebase.ID] = codebase
	if err := manager.saveLocked(); err != nil {
		slog.ErrorContext(ctx, "persist conversation collection registration failed", "collection_id", trimmedCollectionID, "err", err)
		return model.Codebase{}, fmt.Errorf("persist conversation collection %s: %w", trimmedCollectionID, err)
	}
	return codebase, nil
}

func conversationCanonicalPath(collectionID string) string {
	return conversationCanonicalPathPrefix + collectionID
}

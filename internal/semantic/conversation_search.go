package semantic

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/model"
)

// SearchConversationCollectionCapped returns a semantic conversation search
// already reduced to at most limit rows, with at most perConversationLimit rows
// per conversation and a minScore floor. It fills the limit deterministically:
// when the per-conversation cap drops rows it pages the ranked search by offset,
// reusing one query embedding across pages, until limit survivors are collected,
// the score frontier falls below minScore, the result set is exhausted, or the
// Milvus 16384 search window is reached. Every scope dimension is enforced
// natively in Milvus by the filter expression, so the per-page reduction applies
// only the cap and the score floor.
func (service *Service) SearchConversationCollectionCapped(ctx context.Context, collectionName string, query string, limit int32, perConversationLimit int32, minScore float64, filter ConversationFilter) ([]model.StoredChunk, error) {
	if !service.Available() {
		return nil, ErrUnavailable
	}
	trimmedCollectionName := strings.TrimSpace(collectionName)
	if trimmedCollectionName == "" {
		return nil, errors.New("conversation collection name is required")
	}
	if err := service.ensureConversationScalarColumnsOnce(ctx, trimmedCollectionName); err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 10
	}

	// A conversation-id scope larger than one membership clause is the rare legacy
	// path: it keeps the existing batched merge with no paged fill and reduces in
	// process. The common path (no scope or a small scope) is one paged search.
	if len(batchConversationIDs(filter.ConversationIDs, conversationFilterIDBatchSize)) > 1 {
		chunks, err := service.searchConversationBatched(ctx, trimmedCollectionName, query, limit, filter)
		if err != nil {
			return nil, err
		}
		return capConversationChunks(chunks, perConversationLimit, minScore, limit), nil
	}

	hasCollection, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(trimmedCollectionName))
	if err != nil {
		slog.ErrorContext(ctx, "check Milvus collection failed", "collection", trimmedCollectionName, "err", err)
		return nil, fmt.Errorf("check Milvus collection %s: %w", trimmedCollectionName, err)
	}
	if !hasCollection {
		return nil, ErrCollectionMissing
	}
	queryVector, err := service.embedder.Embed(ctx, service.queryTextForEmbedding(query))
	if err != nil {
		slog.ErrorContext(ctx, "embed query failed", "err", err)
		return nil, fmt.Errorf("embed query: %w", err)
	}

	return service.fillCappedConversationSearch(ctx, trimmedCollectionName, queryVector, query, filter.buildExpr(), limit, perConversationLimit, minScore)
}

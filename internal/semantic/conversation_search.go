package semantic

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/model"
	"google.golang.org/grpc/peer"
)

// SearchConversationCollectionCapped returns a semantic conversation search
// already reduced to at most limit rows, with at most perConversationLimit rows
// per conversation and a minScore floor. It embeds the query once and runs one
// ranking search at conversationRankingDepth that returns only each row's
// primary key, relativePath, conversationId, and score. The filter expression
// restricts that ranking natively, and a conversation id scope of any size
// binds as one template parameter. The search sorts the ranking by descending
// score, then relativePath, then primary key, and walks it once to apply the
// score floor, the per-conversation cap, and the limit. It then reads content
// and output columns for the selected rows only and returns them in the walked
// order. One query and filter therefore always return the same rows in the
// same order, and a smaller limit returns a prefix of a larger one.
func (service *Service) SearchConversationCollectionCapped(ctx context.Context, collectionName string, query string, limit int32, perConversationLimit int32, minScore float64, filter ConversationFilter) ([]model.StoredChunk, error) {
	peerInfo, _ := peer.FromContext(ctx)
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

	hasCollection, err := service.hasCollection(
		ctx,
		trimmedCollectionName,
		"check Milvus collection "+trimmedCollectionName,
	)
	if err != nil {
		return nil, err
	}
	if !hasCollection {
		return nil, ErrCollectionMissing
	}
	if err := service.ensureSplitPartColumnOnce(ctx, trimmedCollectionName); err != nil {
		return nil, err
	}
	queryVector, err := service.embedder.Embed(ctx, service.queryTextForEmbedding(query))
	if err != nil {
		slog.ErrorContext(ctx, "embed query failed", "peer", peerInfo.String(), "err", err)
		return nil, fmt.Errorf("embed query: %w", err)
	}

	candidates, err := service.rankConversationCandidates(ctx, trimmedCollectionName, queryVector, query, filter)
	if err != nil {
		return nil, err
	}
	sortRankedCandidates(candidates)
	selected := selectRankedCandidates(candidates, perConversationLimit, minScore, limit)
	return service.loadRankedChunks(ctx, trimmedCollectionName, selected)
}

// rankConversationCandidates runs the one ranking search of a conversation
// search. A hybrid collection runs both legs at conversationRankingDepth and
// fuses them with the RRF reranker into at most conversationRankingDepth rows.
// A dense collection runs one search at the same depth. Both request only
// relativePath and conversationId, and Milvus returns the primary key and score
// with every row.
func (service *Service) rankConversationCandidates(ctx context.Context, collectionName string, queryVector []float32, rawQuery string, filter ConversationFilter) ([]rankedCandidate, error) {
	expression := filter.buildExpr()
	outputFields := []string{relativePathFieldName, conversationIDFieldName}
	if service.cfg.HybridMode {
		denseRequest := milvusclient.NewAnnRequest(denseVectorFieldName, conversationRankingDepth, entity.FloatVector(queryVector))
		sparseRequest := milvusclient.NewAnnRequest(sparseVectorFieldName, conversationRankingDepth, entity.Text(rawQuery))
		if expression != "" {
			denseRequest = denseRequest.WithFilter(expression)
			sparseRequest = sparseRequest.WithFilter(expression)
		}
		if filter.HasConversationScope() {
			denseRequest = denseRequest.WithTemplateParam(conversationIDsTemplateParam, filter.ConversationIDs)
			sparseRequest = sparseRequest.WithTemplateParam(conversationIDsTemplateParam, filter.ConversationIDs)
		}
		hybridOption := milvusclient.NewHybridSearchOption(
			collectionName,
			conversationRankingDepth,
			denseRequest,
			sparseRequest,
		).WithReranker(milvusclient.NewRRFReranker()).WithOutputFields(outputFields...)
		resultSets, err := service.milvus.HybridSearch(ctx, hybridOption)
		if err != nil {
			return nil, searchErr(ctx, "hybrid ranking search", collectionName, err)
		}
		return rankedCandidatesFromResultSets(ctx, collectionName, resultSets)
	}

	searchOption := milvusclient.NewSearchOption(
		collectionName,
		conversationRankingDepth,
		[]entity.Vector{entity.FloatVector(queryVector)},
	).WithANNSField(denseVectorFieldName).WithOutputFields(outputFields...)
	if expression != "" {
		searchOption = searchOption.WithFilter(expression)
	}
	if filter.HasConversationScope() {
		searchOption = searchOption.WithTemplateParam(conversationIDsTemplateParam, filter.ConversationIDs)
	}
	resultSets, err := service.milvus.Search(ctx, searchOption)
	if err != nil {
		return nil, searchErr(ctx, "dense ranking search", collectionName, err)
	}
	return rankedCandidatesFromResultSets(ctx, collectionName, resultSets)
}

// rankedCandidatesFromResultSets decodes the ranking rows. A null
// conversationId decodes as an unknown conversation.
func rankedCandidatesFromResultSets(ctx context.Context, collectionName string, resultSets []milvusclient.ResultSet) ([]rankedCandidate, error) {
	if len(resultSets) == 0 || resultSets[0].ResultCount == 0 {
		return []rankedCandidate{}, nil
	}
	resultSet := resultSets[0]
	relativePathColumn := resultSet.GetColumn(relativePathFieldName)
	conversationIDColumn := resultSet.GetColumn(conversationIDFieldName)
	if resultSet.IDs == nil || relativePathColumn == nil || conversationIDColumn == nil {
		return nil, ErrSearchResultIncomplete
	}
	candidates := make([]rankedCandidate, 0, resultSet.ResultCount)
	for index := range resultSet.ResultCount {
		primaryKey, err := resultSet.IDs.GetAsString(index)
		if err != nil {
			return nil, rankingReadError(ctx, collectionName, "primary key", index, err)
		}
		relativePath, err := relativePathColumn.GetAsString(index)
		if err != nil {
			return nil, rankingReadError(ctx, collectionName, relativePathFieldName, index, err)
		}
		conversationID, known, err := readOptionalStringAt(conversationIDColumn, index)
		if err != nil {
			return nil, rankingReadError(ctx, collectionName, conversationIDFieldName, index, err)
		}
		score := 0.0
		if index < len(resultSet.Scores) {
			score = float64(resultSet.Scores[index])
		}
		candidates = append(candidates, rankedCandidate{
			PrimaryKey:          primaryKey,
			RelativePath:        relativePath,
			ConversationID:      conversationID,
			ConversationIDKnown: known,
			Score:               score,
		})
	}
	return candidates, nil
}

func rankingReadError(ctx context.Context, collectionName string, field string, index int, err error) error {
	slog.ErrorContext(ctx, "read conversation ranking row failed", "collection", collectionName, "field", field, "index", index, "err", err)
	return fmt.Errorf("read ranking %s at %d from %s: %w", field, index, collectionName, err)
}

// loadRankedChunks reads the content and output columns of the selected rows
// by primary key and returns them in the selected order with their ranking
// scores. A selected row that no longer exists is skipped.
func (service *Service) loadRankedChunks(ctx context.Context, collectionName string, selected []rankedCandidate) ([]model.StoredChunk, error) {
	peerInfo, _ := peer.FromContext(ctx)
	if len(selected) == 0 {
		return []model.StoredChunk{}, nil
	}
	primaryKeys := make([]string, 0, len(selected))
	for _, candidate := range selected {
		primaryKeys = append(primaryKeys, candidate.PrimaryKey)
	}
	queryOption := milvusclient.NewQueryOption(collectionName).
		WithIDs(column.NewColumnVarChar(idFieldName, primaryKeys)).
		WithOutputFields(
			idFieldName,
			contentFieldName,
			relativePathFieldName,
			startLineFieldName,
			endLineFieldName,
			fileExtensionFieldName,
			metadataFieldName,
			splitPartFieldName,
			workspaceRootFieldName,
			loadRulesFieldName,
		)
	resultSet, err := service.milvus.Query(ctx, queryOption)
	if err != nil {
		return nil, searchErr(ctx, "load ranked rows", collectionName, err)
	}
	chunks, err := resultSetsToChunks([]milvusclient.ResultSet{resultSet})
	if err != nil {
		return nil, err
	}
	idColumn := resultSet.GetColumn(idFieldName)
	if idColumn == nil && len(chunks) > 0 {
		return nil, ErrSearchResultIncomplete
	}
	byPrimaryKey := make(map[string]model.StoredChunk, len(chunks))
	for index, chunk := range chunks {
		primaryKey, idErr := idColumn.GetAsString(index)
		if idErr != nil {
			return nil, rankingReadError(ctx, collectionName, idFieldName, index, idErr)
		}
		byPrimaryKey[primaryKey] = chunk
	}
	ordered := make([]model.StoredChunk, 0, len(selected))
	for _, candidate := range selected {
		chunk, found := byPrimaryKey[candidate.PrimaryKey]
		if !found {
			slog.WarnContext(ctx, "ranked conversation row disappeared before its content load", "collection", collectionName, "relative_path", candidate.RelativePath, "peer", peerInfo.String())
			continue
		}
		chunk.Score = candidate.Score
		ordered = append(ordered, chunk)
	}
	return ordered, nil
}

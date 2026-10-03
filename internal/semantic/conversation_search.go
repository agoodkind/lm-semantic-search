package semantic

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"
	"google.golang.org/grpc/peer"
)

// SearchConversationCollection runs the typed collection search over a
// conversation collection. It first runs the conversation scalar migration.
// When the search caps hits per conversation, it then resolves the
// conversation of each row written before the conversationId column existed.
func (service *Service) SearchConversationCollection(ctx context.Context, search CollectionSearch) ([]CollectionHit, error) {
	peerInfo, _ := peer.FromContext(ctx)
	if !service.Available() {
		return nil, ErrUnavailable
	}
	if err := service.ensureConversationScalarColumnsOnce(ctx, search.CollectionName); err != nil {
		return nil, err
	}
	request, err := service.prepareCollectionSearch(ctx, search)
	if err != nil {
		return nil, err
	}
	store := service.collectionStore()
	candidates, err := store.Rank(ctx, request)
	if err != nil {
		slog.ErrorContext(ctx, "rank conversation collection failed", "collection", request.Collection, "peer", peerInfo.String(), "err", err)
		return nil, fmt.Errorf("rank %s: %w", request.Collection, err)
	}
	groupColumn, grouped := milvusstore.GroupColumnFor(request)
	perGroupLimit := int32(0)
	if grouped {
		perGroupLimit = request.PerGroupLimit
		if groupColumn.Name == search.Declaration.ItemIDColumn {
			candidates, err = service.resolveLegacyConversationGroups(ctx, request.Collection, groupColumn.Name, candidates)
			if err != nil {
				return nil, err
			}
		}
	}
	selected := milvusstore.SelectCandidates(candidates, perGroupLimit, request.MinScore, request.Limit)
	hits, err := store.Load(ctx, request.Collection, selected, search.Declaration.Scalars)
	if err != nil {
		slog.ErrorContext(ctx, "load conversation search rows failed", "collection", request.Collection, "peer", peerInfo.String(), "err", err)
		return nil, fmt.Errorf("load %s: %w", request.Collection, err)
	}
	return collectionHits(hits, search.Declaration), nil
}

// resolveLegacyConversationGroups sets the group of each candidate with a null
// conversationId column. It queries those rows by primary key and reads
// conversation_id from each row's metadata JSON. Rows written before the
// conversationId column existed store their identity only there. A row without
// a metadata conversation_id groups under the empty conversation id. The
// returned candidates omit a legacy row deleted after the ranking search.
func (service *Service) resolveLegacyConversationGroups(ctx context.Context, collectionName string, groupColumnName string, candidates []milvusstore.Candidate) ([]milvusstore.Candidate, error) {
	legacyKeys := make([]string, 0)
	for _, candidate := range candidates {
		if candidate.Group.State != collection.ScalarCellValue {
			legacyKeys = append(legacyKeys, candidate.PrimaryKey)
		}
	}
	if len(legacyKeys) == 0 {
		return candidates, nil
	}
	resultSet, err := service.milvus.Query(ctx, milvusclient.NewQueryOption(collectionName).
		WithIDs(column.NewColumnVarChar(idFieldName, legacyKeys)).
		WithOutputFields(idFieldName, metadataFieldName))
	if err != nil {
		return nil, searchErr(ctx, "load legacy conversation identity", collectionName, err)
	}
	idColumn := resultSet.GetColumn(idFieldName)
	metadataColumn := resultSet.GetColumn(metadataFieldName)
	if resultSet.ResultCount > 0 && (idColumn == nil || metadataColumn == nil) {
		return nil, collection.ErrSearchResultIncomplete
	}
	legacyIDs := make(map[string]string, resultSet.ResultCount)
	for index := range resultSet.ResultCount {
		primaryKey, idErr := idColumn.GetAsString(index)
		if idErr != nil {
			return nil, legacyReadError(ctx, collectionName, idFieldName, index, idErr)
		}
		metadata, metadataErr := metadataColumn.GetAsString(index)
		if metadataErr != nil {
			return nil, legacyReadError(ctx, collectionName, metadataFieldName, index, metadataErr)
		}
		legacyIDs[primaryKey] = decodeMetadata(metadata).ConversationID
	}
	return applyLegacyConversationGroups(candidates, groupColumnName, legacyIDs), nil
}

func legacyReadError(ctx context.Context, collectionName string, field string, index int, err error) error {
	slog.ErrorContext(ctx, "read collection ranking row failed", "collection", collectionName, "field", field, "index", index, "err", err)
	return fmt.Errorf("read ranking %s at %d from %s: %w", field, index, collectionName, err)
}

// applyLegacyConversationGroups sets each null-group candidate's group from
// legacyIDs and drops a null-group candidate that legacyIDs does not contain.
func applyLegacyConversationGroups(candidates []milvusstore.Candidate, groupColumnName string, legacyIDs map[string]string) []milvusstore.Candidate {
	resolved := make([]milvusstore.Candidate, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate.Group.State != collection.ScalarCellValue {
			conversationID, found := legacyIDs[candidate.PrimaryKey]
			if !found {
				continue
			}
			candidate.Group = collection.ValueCell(groupColumnName, collection.StringScalar(conversationID))
		}
		resolved = append(resolved, candidate)
	}
	return resolved
}

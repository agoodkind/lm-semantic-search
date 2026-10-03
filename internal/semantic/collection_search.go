package semantic

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
	"goodkind.io/lm-semantic-search/internal/model"
	"google.golang.org/grpc/peer"
)

// defaultCollectionSearchLimit is the hit count a search returns when the
// caller sets no positive limit.
const defaultCollectionSearchLimit = 10

// CollectionHit is one ranked search hit. Chunk is the stored row decoded
// through the legacy metadata JSON, and Chunk.RelativePath is the logical row
// key. Scalars lists every declared scalar column in declaration order.
type CollectionHit struct {
	Chunk   model.StoredChunk
	Scalars []collection.ScalarCell
}

// Scalar returns the hit's cell for column. It reports false when column is
// not declared.
func (hit CollectionHit) Scalar(column string) (collection.ScalarCell, bool) {
	for _, cell := range hit.Scalars {
		if cell.Column == column {
			return cell, true
		}
	}
	return collection.AbsentCell(column), false
}

// CollectionSearch is one validated typed search of a registered collection.
// Filter is nil to match every row. PerGroupLimit caps the hits that share one
// GroupBy value, and zero means uncapped. Declaration is the collection's saved
// declaration.
type CollectionSearch struct {
	CollectionName string
	Query          string
	Limit          int32
	MinScore       float64
	Filter         *collection.Filter
	GroupBy        string
	PerGroupLimit  int32
	Declaration    collection.Declaration
}

// groupColumnFor returns the declared column the per-group cap reads. It
// reports false when the search is uncapped.
func groupColumnFor(search CollectionSearch) (collection.ScalarColumn, bool) {
	if search.GroupBy == "" || search.PerGroupLimit <= 0 {
		return collection.ScalarColumn{Name: "", Type: "", Nullable: false, MaxLength: 0}, false
	}
	for _, declared := range search.Declaration.Scalars {
		if declared.Name == search.GroupBy {
			return declared, true
		}
	}
	return collection.ScalarColumn{Name: search.GroupBy, Type: collection.ScalarTypeString, Nullable: true, MaxLength: 0}, true
}

// SearchCollection runs a typed search and returns at most Limit hits, at most
// PerGroupLimit per GroupBy value, none scoring below MinScore. On an unchanged
// collection, repeating the same query with the same filter returns the same
// rows in the same order, and a smaller limit returns a prefix of a larger
// one. The filter restricts one fixed-depth ranking natively, and every
// membership set binds as one template parameter.
func (service *Service) SearchCollection(ctx context.Context, search CollectionSearch) ([]CollectionHit, error) {
	peerInfo, _ := peer.FromContext(ctx)
	if !service.Available() {
		return nil, ErrUnavailable
	}
	collectionName := strings.TrimSpace(search.CollectionName)
	if collectionName == "" {
		return nil, errors.New("collection name is required")
	}
	if IsConversationDeclaration(search.Declaration) {
		if err := service.ensureConversationScalarColumnsOnce(ctx, collectionName); err != nil {
			return nil, err
		}
	}
	limit := search.Limit
	if limit <= 0 {
		limit = defaultCollectionSearchLimit
	}
	compiled, err := collection.Compile(search.Filter)
	if err != nil {
		slog.ErrorContext(ctx, "compile collection filter failed", "collection", collectionName, "err", err)
		return nil, fmt.Errorf("compile filter for %s: %w", collectionName, err)
	}
	hasCollection, err := service.hasCollection(ctx, collectionName, "check Milvus collection "+collectionName)
	if err != nil {
		return nil, err
	}
	if !hasCollection {
		return nil, ErrCollectionMissing
	}
	if err := service.ensureSplitPartColumnOnce(ctx, collectionName); err != nil {
		return nil, err
	}
	queryVector, err := service.embedder.Embed(ctx, service.queryTextForEmbedding(search.Query))
	if err != nil {
		slog.ErrorContext(ctx, "embed query failed", "peer", peerInfo.String(), "err", err)
		return nil, fmt.Errorf("embed query: %w", err)
	}

	groupColumn, grouped := groupColumnFor(search)
	candidates, err := service.rankCollectionCandidates(ctx, collectionName, queryVector, search.Query, compiled, groupColumn, grouped)
	if err != nil {
		return nil, err
	}
	if grouped && IsConversationDeclaration(search.Declaration) && groupColumn.Name == search.Declaration.ItemIDColumn {
		candidates, err = service.resolveLegacyConversationGroups(ctx, collectionName, groupColumn.Name, candidates)
		if err != nil {
			return nil, err
		}
	}
	sortRankedCandidates(candidates)
	perGroupLimit := int32(0)
	if grouped {
		perGroupLimit = search.PerGroupLimit
	}
	selected := selectRankedCandidates(candidates, perGroupLimit, search.MinScore, limit)
	return service.loadRankedHits(ctx, collectionName, selected, search.Declaration.Scalars)
}

// rankedCandidate is one row of a collection search's fused ranking. It
// stores only the row identity, the group column cell, and the score.
type rankedCandidate struct {
	PrimaryKey   string
	RelativePath string
	Group        collection.ScalarCell
	Score        float64
}

// sortRankedCandidates orders candidates by descending score, then ascending
// relativePath, then ascending primary key. The order is total.
func sortRankedCandidates(candidates []rankedCandidate) {
	sort.Slice(candidates, func(first int, second int) bool {
		left := candidates[first]
		right := candidates[second]
		if left.Score != right.Score {
			return left.Score > right.Score
		}
		if left.RelativePath != right.RelativePath {
			return left.RelativePath < right.RelativePath
		}
		return left.PrimaryKey < right.PrimaryKey
	})
}

// selectRankedCandidates walks sorted candidates once. It drops a candidate
// scoring below minScore, keeps at most perGroupLimit candidates per group
// key, and stops at limit. A zero perGroupLimit is uncapped, and a zero
// minScore is no floor. A smaller limit returns a prefix of a larger limit's
// result, which search paging relies on.
func selectRankedCandidates(candidates []rankedCandidate, perGroupLimit int32, minScore float64, limit int32) []rankedCandidate {
	kept := make([]rankedCandidate, 0, min(len(candidates), int(max(limit, 0))))
	perGroup := make(map[string]int32)
	for _, candidate := range candidates {
		if limit > 0 && len(kept) >= int(limit) {
			break
		}
		if minScore > 0 && candidate.Score < minScore {
			continue
		}
		if perGroupLimit > 0 {
			key := candidate.Group.GroupKey()
			if perGroup[key] >= perGroupLimit {
				continue
			}
			perGroup[key]++
		}
		kept = append(kept, candidate)
	}
	return kept
}

// rankCollectionCandidates runs the one ranking search of a collection search.
// A hybrid collection runs both legs at collection.RankingDepth and fuses them
// with the RRF reranker into at most collection.RankingDepth rows. A dense
// collection runs one search at the same depth.
func (service *Service) rankCollectionCandidates(ctx context.Context, collectionName string, queryVector []float32, rawQuery string, compiled collection.CompiledFilter, groupColumn collection.ScalarColumn, grouped bool) ([]rankedCandidate, error) {
	outputFields := []string{relativePathFieldName}
	if grouped {
		outputFields = append(outputFields, groupColumn.Name)
	}
	if service.cfg.HybridMode {
		denseRequest := milvusclient.NewAnnRequest(denseVectorFieldName, collection.RankingDepth, entity.FloatVector(queryVector))
		sparseRequest := milvusclient.NewAnnRequest(sparseVectorFieldName, collection.RankingDepth, entity.Text(rawQuery))
		if compiled.Expression != "" {
			denseRequest = denseRequest.WithFilter(compiled.Expression)
			sparseRequest = sparseRequest.WithFilter(compiled.Expression)
		}
		for _, param := range compiled.Params {
			denseRequest = bindAnnTemplateParam(denseRequest, param)
			sparseRequest = bindAnnTemplateParam(sparseRequest, param)
		}
		hybridOption := milvusclient.NewHybridSearchOption(
			collectionName,
			collection.RankingDepth,
			denseRequest,
			sparseRequest,
		).WithReranker(milvusclient.NewRRFReranker()).WithOutputFields(outputFields...)
		resultSets, err := service.milvus.HybridSearch(ctx, hybridOption)
		if err != nil {
			return nil, searchErr(ctx, "hybrid ranking search", collectionName, err)
		}
		return rankedCandidatesFromResultSets(ctx, collectionName, resultSets, groupColumn, grouped)
	}

	searchOption := milvusclient.NewSearchOption(
		collectionName,
		collection.RankingDepth,
		[]entity.Vector{entity.FloatVector(queryVector)},
	).WithANNSField(denseVectorFieldName).WithOutputFields(outputFields...)
	if compiled.Expression != "" {
		searchOption = searchOption.WithFilter(compiled.Expression)
	}
	for _, param := range compiled.Params {
		switch param.Type {
		case collection.ScalarTypeBool:
			searchOption = searchOption.WithTemplateParam(param.Name, param.Bools)
		case collection.ScalarTypeInt64:
			searchOption = searchOption.WithTemplateParam(param.Name, param.Int64s)
		case collection.ScalarTypeString:
			searchOption = searchOption.WithTemplateParam(param.Name, param.Strings)
		default:
			searchOption = searchOption.WithTemplateParam(param.Name, param.Strings)
		}
	}
	resultSets, err := service.milvus.Search(ctx, searchOption)
	if err != nil {
		return nil, searchErr(ctx, "dense ranking search", collectionName, err)
	}
	return rankedCandidatesFromResultSets(ctx, collectionName, resultSets, groupColumn, grouped)
}

func bindAnnTemplateParam(request *milvusclient.AnnRequest, param collection.TemplateParam) *milvusclient.AnnRequest {
	switch param.Type {
	case collection.ScalarTypeBool:
		return request.WithTemplateParam(param.Name, param.Bools)
	case collection.ScalarTypeInt64:
		return request.WithTemplateParam(param.Name, param.Int64s)
	case collection.ScalarTypeString:
		return request.WithTemplateParam(param.Name, param.Strings)
	default:
		return request.WithTemplateParam(param.Name, param.Strings)
	}
}

// rankedCandidatesFromResultSets decodes the ranking rows and each row's group
// column cell. A result without a score for every row returns
// ErrSearchResultIncomplete.
func rankedCandidatesFromResultSets(ctx context.Context, collectionName string, resultSets []milvusclient.ResultSet, groupColumn collection.ScalarColumn, grouped bool) ([]rankedCandidate, error) {
	if len(resultSets) == 0 || resultSets[0].ResultCount == 0 {
		return []rankedCandidate{}, nil
	}
	resultSet := resultSets[0]
	relativePathColumn := resultSet.GetColumn(relativePathFieldName)
	if resultSet.IDs == nil || relativePathColumn == nil || len(resultSet.Scores) < resultSet.ResultCount {
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
		group := collection.AbsentCell(groupColumn.Name)
		if grouped {
			group, err = scalarCellAt(resultSet.GetColumn(groupColumn.Name), groupColumn, index)
			if err != nil {
				return nil, err
			}
		}
		candidates = append(candidates, rankedCandidate{PrimaryKey: primaryKey, RelativePath: relativePath, Group: group, Score: float64(resultSet.Scores[index])})
	}
	return candidates, nil
}

func rankingReadError(ctx context.Context, collectionName string, field string, index int, err error) error {
	slog.ErrorContext(ctx, "read collection ranking row failed", "collection", collectionName, "field", field, "index", index, "err", err)
	return fmt.Errorf("read ranking %s at %d from %s: %w", field, index, collectionName, err)
}

// resolveLegacyConversationGroups sets the group of each candidate with a null
// conversationId column. It queries those rows by primary key and reads
// conversation_id from each row's metadata JSON. Rows written before the
// conversationId column existed store their identity only there. A row without
// a metadata conversation_id groups under the empty conversation id. The
// returned candidates omit a legacy row deleted after the ranking search.
func (service *Service) resolveLegacyConversationGroups(ctx context.Context, collectionName string, groupColumnName string, candidates []rankedCandidate) ([]rankedCandidate, error) {
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
		return nil, ErrSearchResultIncomplete
	}
	legacyIDs := make(map[string]string, resultSet.ResultCount)
	for index := range resultSet.ResultCount {
		primaryKey, idErr := idColumn.GetAsString(index)
		if idErr != nil {
			return nil, rankingReadError(ctx, collectionName, idFieldName, index, idErr)
		}
		metadata, metadataErr := metadataColumn.GetAsString(index)
		if metadataErr != nil {
			return nil, rankingReadError(ctx, collectionName, metadataFieldName, index, metadataErr)
		}
		legacyIDs[primaryKey] = decodeMetadata(metadata).ConversationID
	}
	return applyLegacyConversationGroups(candidates, groupColumnName, legacyIDs), nil
}

// applyLegacyConversationGroups sets each null-group candidate's group from
// legacyIDs and drops a null-group candidate that legacyIDs does not contain.
func applyLegacyConversationGroups(candidates []rankedCandidate, groupColumnName string, legacyIDs map[string]string) []rankedCandidate {
	resolved := make([]rankedCandidate, 0, len(candidates))
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

// loadRankedHits queries content, output columns, and declared scalar columns
// for the selected rows by primary key and returns them in selection order. A
// row deleted after the ranking search is skipped.
func (service *Service) loadRankedHits(ctx context.Context, collectionName string, selected []rankedCandidate, scalarColumns []collection.ScalarColumn) ([]CollectionHit, error) {
	peerInfo, _ := peer.FromContext(ctx)
	if len(selected) == 0 {
		return []CollectionHit{}, nil
	}
	primaryKeys := make([]string, 0, len(selected))
	for _, candidate := range selected {
		primaryKeys = append(primaryKeys, candidate.PrimaryKey)
	}
	outputFields := []string{
		idFieldName,
		contentFieldName,
		relativePathFieldName,
		startLineFieldName,
		endLineFieldName,
		fileExtensionFieldName,
		metadataFieldName,
		splitPartFieldName,
	}
	for _, declared := range scalarColumns {
		outputFields = append(outputFields, declared.Name)
	}
	resultSet, err := service.milvus.Query(ctx, milvusclient.NewQueryOption(collectionName).
		WithIDs(column.NewColumnVarChar(idFieldName, primaryKeys)).
		WithOutputFields(outputFields...))
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
	byPrimaryKey := make(map[string]CollectionHit, len(chunks))
	for index, chunk := range chunks {
		primaryKey, idErr := idColumn.GetAsString(index)
		if idErr != nil {
			return nil, rankingReadError(ctx, collectionName, idFieldName, index, idErr)
		}
		cells, cellErr := scalarCellsAt(resultSet, scalarColumns, index)
		if cellErr != nil {
			return nil, cellErr
		}
		byPrimaryKey[primaryKey] = CollectionHit{Chunk: chunk, Scalars: cells}
	}
	ordered := make([]CollectionHit, 0, len(selected))
	for _, candidate := range selected {
		hit, found := byPrimaryKey[candidate.PrimaryKey]
		if !found {
			slog.WarnContext(ctx, "ranked row disappeared before its content load", "collection", collectionName, "relative_path", candidate.RelativePath, "peer", peerInfo.String())
			continue
		}
		hit.Chunk.Score = candidate.Score
		ordered = append(ordered, hit)
	}
	return ordered, nil
}

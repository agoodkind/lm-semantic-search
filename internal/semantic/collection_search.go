package semantic

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/model"
	"google.golang.org/grpc/peer"
)

// defaultCollectionSearchLimit is the hit count a search returns when the
// caller sets no positive limit.
const defaultCollectionSearchLimit = 10

// CollectionRankingDepth is the largest candidate count of one collection
// ranking search, the Milvus single-search ceiling. RankingDepth caps the
// depth of every ranking search at this value.
const CollectionRankingDepth = 16384

// rankingConsistency is the Milvus consistency level of every read in a
// collection search: the eligible count, the ranking search, the legacy group
// query, and the content load. A Strong read observes every write Milvus
// acknowledged before the read started.
const rankingConsistency = entity.ClStrong

// nullGroupKey is the group key of every hit with a null or absent group
// column value. Those hits share one group.
const nullGroupKey = "null"

// ScalarCellState is the closed set of states of one declared scalar column on
// a stored row.
type ScalarCellState string

const (
	// ScalarCellAbsent means the stored row has no such column.
	ScalarCellAbsent ScalarCellState = "absent"
	// ScalarCellNull means the stored row has the column and its value is null.
	ScalarCellNull ScalarCellState = "null"
	// ScalarCellValue means the stored row has a concrete value in Value.
	ScalarCellValue ScalarCellState = "value"
)

// ScalarCell is one declared scalar column value on a search hit.
type ScalarCell struct {
	Column string
	State  ScalarCellState
	Value  ScalarValue
}

// AbsentCell returns the cell of a column the stored row does not have.
func AbsentCell(column string) ScalarCell {
	return ScalarCell{Column: column, State: ScalarCellAbsent, Value: ScalarValue{Type: "", String: "", Bool: false, Int64: 0}}
}

// NullCell returns the cell of a column with a null value.
func NullCell(column string) ScalarCell {
	return ScalarCell{Column: column, State: ScalarCellNull, Value: ScalarValue{Type: "", String: "", Bool: false, Int64: 0}}
}

// ValueCell returns the cell of a column with a concrete value.
func ValueCell(column string, value ScalarValue) ScalarCell {
	return ScalarCell{Column: column, State: ScalarCellValue, Value: value}
}

// GroupKey returns the key that groups hits by this cell's value for a
// per-group cap. Every null or absent cell returns one shared key. Values of
// different types never share a key.
func (cell ScalarCell) GroupKey() string {
	if cell.State != ScalarCellValue {
		return nullGroupKey
	}
	switch cell.Value.Type {
	case model.ScalarTypeBool:
		return "bool:" + strconv.FormatBool(cell.Value.Bool)
	case model.ScalarTypeInt64:
		return "int64:" + strconv.FormatInt(cell.Value.Int64, 10)
	case model.ScalarTypeString:
		return "string:" + cell.Value.String
	default:
		return "string:" + cell.Value.String
	}
}

// CollectionHit is one ranked search hit. Chunk is the stored row decoded
// through the legacy metadata JSON, and Chunk.RelativePath is the logical row
// key. Scalars lists every declared scalar column in declaration order.
type CollectionHit struct {
	Chunk   model.StoredChunk
	Scalars []ScalarCell
}

// Scalar returns the hit's cell for column. It reports false when column is
// not declared.
func (hit CollectionHit) Scalar(column string) (ScalarCell, bool) {
	for _, cell := range hit.Scalars {
		if cell.Column == column {
			return cell, true
		}
	}
	return AbsentCell(column), false
}

// CollectionSearch is one validated typed search of a registered collection.
// Filter is nil to match every row. PerGroupLimit caps the hits that share one
// GroupBy value, and zero means uncapped. Declaration is the collection's saved
// declaration. CallerState is an opaque value the caller reads before the
// search. It is part of the ranking cache key, and the result returns the
// CallerState stored with the ranking that served the hits.
type CollectionSearch struct {
	CollectionName string
	Query          string
	Limit          int32
	MinScore       float64
	Filter         *CollectionFilter
	GroupBy        string
	PerGroupLimit  int32
	Declaration    model.CollectionDeclaration
	CallerState    string
}

// CollectionSearchResult is the outcome of one collection search. Hits is the
// selected page. RankingTruncated is true when more rows matched the filter
// than CollectionRankingDepth, and the ranking then covers only the first
// CollectionRankingDepth rows. CallerState is the CollectionSearch.CallerState
// stored with the ranking that served Hits.
type CollectionSearchResult struct {
	Hits             []CollectionHit
	RankingTruncated bool
	CallerState      string
}

// groupColumnFor returns the declared column the per-group cap reads. It
// reports false when the search is uncapped.
func groupColumnFor(search CollectionSearch) (model.ScalarColumn, bool) {
	if search.GroupBy == "" || search.PerGroupLimit <= 0 {
		return model.ScalarColumn{Name: "", Type: "", Nullable: false, MaxLength: 0}, false
	}
	for _, declared := range search.Declaration.Scalars {
		if declared.Name == search.GroupBy {
			return declared, true
		}
	}
	return model.ScalarColumn{Name: search.GroupBy, Type: model.ScalarTypeString, Nullable: true, MaxLength: 0}, true
}

// SearchCollection runs a typed search and returns at most Limit hits, at most
// PerGroupLimit per GroupBy value, none scoring below MinScore.
//
// Every request counts the rows that match the filter. A count of zero returns
// no hits without a ranking search. The first request for a ranking key embeds
// the query and runs one ranking search at RankingDepth of the count. The
// sorted candidates are cached under the key. A later request with the same
// key and the same count reads the cached candidates, and every limit then
// selects a prefix of one ranking. A committed write through this service, a
// different count, a recreated collection, a restart, eviction, and
// RankingCacheTTL each make the next request rank again.
func (service *Service) SearchCollection(ctx context.Context, search CollectionSearch) (CollectionSearchResult, error) {
	emptyResult := CollectionSearchResult{Hits: nil, RankingTruncated: false, CallerState: ""}
	peerInfo, _ := peer.FromContext(ctx)
	if !service.Available() {
		return emptyResult, ErrUnavailable
	}
	collectionName := strings.TrimSpace(search.CollectionName)
	if collectionName == "" {
		return emptyResult, errors.New("collection name is required")
	}
	if IsConversationDeclaration(search.Declaration) {
		if err := service.ensureConversationScalarColumnsOnce(ctx, collectionName); err != nil {
			return emptyResult, err
		}
	}
	limit := search.Limit
	if limit <= 0 {
		limit = defaultCollectionSearchLimit
	}
	compiled, err := compileCollectionFilterExpr(search.Filter)
	if err != nil {
		slog.ErrorContext(ctx, "compile collection filter failed", "collection", collectionName, "peer", peerInfo.String(), "err", err)
		return emptyResult, fmt.Errorf("compile filter for %s: %w", collectionName, err)
	}
	hasCollection, err := service.hasCollection(ctx, collectionName, "check Milvus collection "+collectionName)
	if err != nil {
		return emptyResult, err
	}
	if !hasCollection {
		return emptyResult, ErrCollectionMissing
	}
	if err := service.ensureSplitPartColumnOnce(ctx, collectionName); err != nil {
		return emptyResult, err
	}
	collectionID, err := service.collectionIdentity(ctx, collectionName)
	if err != nil {
		return emptyResult, err
	}
	writeGeneration := service.rankings.writeGeneration(collectionName)
	eligible, err := service.countEligibleRows(ctx, collectionName, compiled)
	if err != nil {
		return emptyResult, err
	}
	if eligible == 0 {
		return CollectionSearchResult{Hits: []CollectionHit{}, RankingTruncated: false, CallerState: search.CallerState}, nil
	}

	groupColumn, grouped := groupColumnFor(search)
	perGroupLimit := int32(0)
	groupColumnName := ""
	if grouped {
		perGroupLimit = search.PerGroupLimit
		groupColumnName = groupColumn.Name
	}
	digest := rankingKey{
		CollectionName:  collectionName,
		CollectionID:    collectionID,
		WriteGeneration: writeGeneration,
		Query:           search.Query,
		Hybrid:          service.cfg.HybridMode,
		Filter:          compiled,
		MinScore:        search.MinScore,
		GroupColumn:     groupColumnName,
		PerGroupLimit:   perGroupLimit,
		CallerState:     search.CallerState,
	}.digest()
	ranking, err := service.rankings.rank(digest, eligible, func() (collectionRanking, error) {
		return service.computeRanking(ctx, collectionName, search, compiled, eligible)
	})
	if err != nil {
		return emptyResult, err
	}
	selected := selectRankedCandidates(ranking.Candidates, perGroupLimit, search.MinScore, limit)
	hits, err := service.loadRankedHits(ctx, collectionName, selected, search.Declaration.Scalars)
	if err != nil {
		return emptyResult, err
	}
	return CollectionSearchResult{Hits: hits, RankingTruncated: ranking.Truncated, CallerState: ranking.CallerState}, nil
}

// computeRanking embeds the query and runs one ranking search over eligible
// matching rows at RankingDepth(eligible). It resolves legacy conversation
// groups and sorts the candidates.
func (service *Service) computeRanking(ctx context.Context, collectionName string, search CollectionSearch, compiled compiledFilter, eligible int64) (collectionRanking, error) {
	var failed collectionRanking
	peerInfo, _ := peer.FromContext(ctx)
	queryVector, err := service.embedder.Embed(ctx, service.queryTextForEmbedding(search.Query))
	if err != nil {
		slog.ErrorContext(ctx, "embed query failed", "peer", peerInfo.String(), "err", err)
		return failed, fmt.Errorf("embed query: %w", err)
	}
	groupColumn, grouped := groupColumnFor(search)
	depth := RankingDepth(eligible)
	candidates, err := service.rankCollectionCandidates(ctx, collectionName, queryVector, search.Query, compiled, groupColumn, grouped, depth)
	if err != nil {
		return failed, err
	}
	if grouped && IsConversationDeclaration(search.Declaration) && groupColumn.Name == search.Declaration.ItemIDColumn {
		candidates, err = service.resolveLegacyConversationGroups(ctx, collectionName, groupColumn.Name, candidates)
		if err != nil {
			return failed, err
		}
	}
	if len(candidates) < depth {
		slog.InfoContext(ctx, "collection ranking returned fewer rows than its depth",
			"collection", collectionName, "eligible", eligible, "depth", depth, "ranked", len(candidates), "peer", peerInfo.String())
	}
	sortRankedCandidates(candidates)
	return collectionRanking{
		Candidates:  candidates,
		Eligible:    eligible,
		Truncated:   RankingTruncated(eligible),
		CallerState: search.CallerState,
	}, nil
}

// collectionIdentity returns the Milvus collection ID of collectionName. A
// dropped and recreated collection has a new ID.
func (service *Service) collectionIdentity(ctx context.Context, collectionName string) (int64, error) {
	collection, err := service.milvus.DescribeCollection(ctx, milvusclient.NewDescribeCollectionOption(collectionName))
	if err != nil {
		return 0, searchErr(ctx, "describe collection for ranking identity", collectionName, err)
	}
	if collection == nil {
		return 0, fmt.Errorf("describe collection %s returned no collection", collectionName)
	}
	return collection.ID, nil
}

// countEligibleRows counts the rows that match compiled with a count(*) query
// that binds the same expression and template parameters as the ranking
// search.
func (service *Service) countEligibleRows(ctx context.Context, collectionName string, compiled compiledFilter) (int64, error) {
	option := milvusclient.NewQueryOption(collectionName).
		WithOutputFields(countOutputField).
		WithConsistencyLevel(rankingConsistency)
	if compiled.Expression != "" {
		option = option.WithFilter(compiled.Expression)
	}
	for _, param := range compiled.Params {
		switch param.Type {
		case model.ScalarTypeBool:
			option = option.WithTemplateParam(param.Name, param.Bools)
		case model.ScalarTypeInt64:
			option = option.WithTemplateParam(param.Name, param.Int64s)
		case model.ScalarTypeString:
			option = option.WithTemplateParam(param.Name, param.Strings)
		default:
			option = option.WithTemplateParam(param.Name, param.Strings)
		}
	}
	resultSet, err := service.milvus.Query(ctx, option)
	if err != nil {
		return 0, searchErr(ctx, "count eligible rows", collectionName, err)
	}
	countColumn := resultSet.GetColumn(countOutputField)
	if countColumn == nil {
		return 0, ErrSearchResultIncomplete
	}
	eligible, err := countColumn.GetAsInt64(0)
	if err != nil {
		return 0, rankingReadError(ctx, collectionName, countOutputField, 0, err)
	}
	return eligible, nil
}

// rankedCandidate is one row of a collection search's fused ranking. It
// stores only the row identity, the group column cell, and the score.
type rankedCandidate struct {
	PrimaryKey   string
	RelativePath string
	Group        ScalarCell
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
// A hybrid collection runs both legs at depth and fuses them with the RRF
// reranker into at most depth rows. A dense collection runs one search at
// depth.
func (service *Service) rankCollectionCandidates(ctx context.Context, collectionName string, queryVector []float32, rawQuery string, compiled compiledFilter, groupColumn model.ScalarColumn, grouped bool, depth int) ([]rankedCandidate, error) {
	outputFields := []string{relativePathFieldName}
	if grouped {
		outputFields = append(outputFields, groupColumn.Name)
	}
	if service.cfg.HybridMode {
		denseRequest := milvusclient.NewAnnRequest(denseVectorFieldName, depth, entity.FloatVector(queryVector))
		sparseRequest := milvusclient.NewAnnRequest(sparseVectorFieldName, depth, entity.Text(rawQuery))
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
			depth,
			denseRequest,
			sparseRequest,
		).WithReranker(milvusclient.NewRRFReranker()).WithOutputFields(outputFields...).WithConsistencyLevel(rankingConsistency)
		resultSets, err := service.milvus.HybridSearch(ctx, hybridOption)
		if err != nil {
			return nil, searchErr(ctx, "hybrid ranking search", collectionName, err)
		}
		return rankedCandidatesFromResultSets(ctx, collectionName, resultSets, groupColumn, grouped)
	}

	searchOption := milvusclient.NewSearchOption(
		collectionName,
		depth,
		[]entity.Vector{entity.FloatVector(queryVector)},
	).WithANNSField(denseVectorFieldName).WithOutputFields(outputFields...).WithConsistencyLevel(rankingConsistency)
	if compiled.Expression != "" {
		searchOption = searchOption.WithFilter(compiled.Expression)
	}
	for _, param := range compiled.Params {
		switch param.Type {
		case model.ScalarTypeBool:
			searchOption = searchOption.WithTemplateParam(param.Name, param.Bools)
		case model.ScalarTypeInt64:
			searchOption = searchOption.WithTemplateParam(param.Name, param.Int64s)
		case model.ScalarTypeString:
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

func bindAnnTemplateParam(request *milvusclient.AnnRequest, param filterTemplateParam) *milvusclient.AnnRequest {
	switch param.Type {
	case model.ScalarTypeBool:
		return request.WithTemplateParam(param.Name, param.Bools)
	case model.ScalarTypeInt64:
		return request.WithTemplateParam(param.Name, param.Int64s)
	case model.ScalarTypeString:
		return request.WithTemplateParam(param.Name, param.Strings)
	default:
		return request.WithTemplateParam(param.Name, param.Strings)
	}
}

// rankedCandidatesFromResultSets decodes the ranking rows and each row's group
// column cell. A result without a score for every row returns
// ErrSearchResultIncomplete.
func rankedCandidatesFromResultSets(ctx context.Context, collectionName string, resultSets []milvusclient.ResultSet, groupColumn model.ScalarColumn, grouped bool) ([]rankedCandidate, error) {
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
		group := AbsentCell(groupColumn.Name)
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
		if candidate.Group.State != ScalarCellValue {
			legacyKeys = append(legacyKeys, candidate.PrimaryKey)
		}
	}
	if len(legacyKeys) == 0 {
		return candidates, nil
	}
	resultSet, err := service.milvus.Query(ctx, milvusclient.NewQueryOption(collectionName).
		WithIDs(primaryKeyColumn(legacyKeys)).
		WithOutputFields(idFieldName, metadataFieldName).
		WithConsistencyLevel(rankingConsistency))
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

// primaryKeyColumn returns an id column over a copy of primaryKeys. The Milvus
// client WithIDs option quotes each value of the column's backing slice in
// place, and the copy keeps that write off the caller's slice.
func primaryKeyColumn(primaryKeys []string) column.Column {
	return column.NewColumnVarChar(idFieldName, slices.Clone(primaryKeys))
}

// applyLegacyConversationGroups sets each null-group candidate's group from
// legacyIDs and drops a null-group candidate that legacyIDs does not contain.
func applyLegacyConversationGroups(candidates []rankedCandidate, groupColumnName string, legacyIDs map[string]string) []rankedCandidate {
	resolved := make([]rankedCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate.Group.State != ScalarCellValue {
			conversationID, found := legacyIDs[candidate.PrimaryKey]
			if !found {
				continue
			}
			candidate.Group = ValueCell(groupColumnName, StringScalar(conversationID))
		}
		resolved = append(resolved, candidate)
	}
	return resolved
}

// loadRankedHits queries content, output columns, and declared scalar columns
// for the selected rows by primary key and returns them in selection order. A
// row deleted after the ranking search is skipped.
func (service *Service) loadRankedHits(ctx context.Context, collectionName string, selected []rankedCandidate, scalarColumns []model.ScalarColumn) ([]CollectionHit, error) {
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
		WithIDs(primaryKeyColumn(primaryKeys)).
		WithOutputFields(outputFields...).
		WithConsistencyLevel(rankingConsistency))
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

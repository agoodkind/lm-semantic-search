package milvus

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
	"google.golang.org/grpc/peer"
)

// defaultSearchLimit is the hit count a search returns when the request sets no
// positive limit.
const defaultSearchLimit = 10

// minimumLegDepth is the smallest topK of one hybrid leg of an expression
// search.
const minimumLegDepth = 10

// Candidate is one row of a collection search's fused ranking. It stores only
// the row identity, the group column cell, and the score.
type Candidate struct {
	PrimaryKey   string
	RelativePath string
	Group        collection.ScalarCell
	Score        float64
}

// GroupColumnFor returns the declared column the per-group cap reads. It
// reports false when the search is uncapped.
func GroupColumnFor(request collection.SearchRequest) (collection.ScalarColumn, bool) {
	if request.GroupBy == "" || request.PerGroupLimit <= 0 {
		return collection.ScalarColumn{Name: "", Type: "", Nullable: false, MaxLength: 0}, false
	}
	for _, declared := range request.Declaration.Scalars {
		if declared.Name == request.GroupBy {
			return declared, true
		}
	}
	return collection.ScalarColumn{Name: request.GroupBy, Type: collection.ScalarTypeString, Nullable: true, MaxLength: 0}, true
}

// Search runs a typed search and returns at most Limit hits, at most
// PerGroupLimit per GroupBy value, none scoring below MinScore. On an unchanged
// collection, repeating the same query with the same filter returns the same
// rows in the same order, and a smaller limit returns a prefix of a larger
// one. The filter restricts one fixed-depth ranking natively, and every
// membership set binds as one template parameter.
func (store *Store) Search(ctx context.Context, request collection.SearchRequest) ([]collection.Hit, error) {
	candidates, err := store.Rank(ctx, request)
	if err != nil {
		return nil, err
	}
	_, grouped := GroupColumnFor(request)
	perGroupLimit := int32(0)
	if grouped {
		perGroupLimit = request.PerGroupLimit
	}
	limit := request.Limit
	if limit <= 0 {
		limit = defaultSearchLimit
	}
	selected := SelectCandidates(candidates, perGroupLimit, request.MinScore, limit)
	return store.Load(ctx, request.Collection, selected, request.Declaration.Scalars)
}

// Rank runs the one ranking search of a collection search and returns every
// ranked candidate, up to [collection.RankingDepth], in no guaranteed order. It
// returns [collection.ErrCollectionMissing] when the collection is absent.
func (store *Store) Rank(ctx context.Context, request collection.SearchRequest) ([]Candidate, error) {
	collectionName := strings.TrimSpace(request.Collection)
	if collectionName == "" {
		return nil, errors.New("collection name is required")
	}
	compiled, err := collection.Compile(request.Filter)
	if err != nil {
		slog.ErrorContext(ctx, "compile collection filter failed", "collection", collectionName, "err", err)
		return nil, fmt.Errorf("compile filter for %s: %w", collectionName, err)
	}
	hasCollection, err := store.client.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		return nil, WrapError(ctx, err, "check Milvus collection "+collectionName)
	}
	if !hasCollection {
		return nil, collection.ErrCollectionMissing
	}
	groupColumn, grouped := GroupColumnFor(request)
	limit := request.Limit
	if limit <= 0 {
		limit = defaultSearchLimit
	}
	depth := rankingDepthFor(limit)
	return store.rankCandidates(ctx, collectionName, request.Vector, request.Query, compiled, groupColumn, grouped, depth)
}

// rankingDepthFor returns the rows each ranking leg requests for a search
// limit: min(max(limit*4, 64), collection.RankingDepth).
func rankingDepthFor(limit int32) int {
	return min(max(int(limit)*4, minimumRankingDepth), collection.RankingDepth)
}

// minimumRankingDepth is the fewest rows a ranking leg requests.
const minimumRankingDepth = 64

// SelectCandidates orders candidates by descending score, then ascending
// relativePath, then ascending primary key. It then walks them once to drop a
// candidate scoring below minScore, keep at most perGroupLimit per group, and
// stop at limit.
func SelectCandidates(candidates []Candidate, perGroupLimit int32, minScore float64, limit int32) []Candidate {
	sortRankedCandidates(candidates)
	return selectRankedCandidates(candidates, perGroupLimit, minScore, limit)
}

// sortRankedCandidates orders candidates by descending score, then ascending
// relativePath, then ascending primary key. The order is total.
func sortRankedCandidates(candidates []Candidate) {
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
func selectRankedCandidates(candidates []Candidate, perGroupLimit int32, minScore float64, limit int32) []Candidate {
	kept := make([]Candidate, 0, min(len(candidates), int(max(limit, 0))))
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

// rankCandidates runs the one ranking search of a collection search.
// A hybrid collection runs both legs at collection.RankingDepth and fuses them
// with the RRF reranker into at most collection.RankingDepth rows. A dense
// collection runs one search at the same depth.
func (store *Store) rankCandidates(ctx context.Context, collectionName string, queryVector []float32, rawQuery string, compiled collection.CompiledFilter, groupColumn collection.ScalarColumn, grouped bool, depth int) ([]Candidate, error) {
	outputFields := []string{RelativePathField}
	if grouped {
		outputFields = append(outputFields, groupColumn.Name)
	}
	if store.options.Hybrid {
		denseRequest := milvusclient.NewAnnRequest(DenseVectorField, depth, entity.FloatVector(queryVector))
		sparseRequest := milvusclient.NewAnnRequest(SparseVectorField, depth, entity.Text(rawQuery))
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
		).WithReranker(milvusclient.NewRRFReranker()).WithOutputFields(outputFields...)
		resultSets, err := store.client.HybridSearch(ctx, hybridOption)
		if err != nil {
			return nil, SearchError(ctx, "hybrid ranking search", collectionName, err)
		}
		return rankedCandidatesFromResultSets(ctx, collectionName, resultSets, groupColumn, grouped)
	}

	searchOption := milvusclient.NewSearchOption(
		collectionName,
		depth,
		[]entity.Vector{entity.FloatVector(queryVector)},
	).WithANNSField(DenseVectorField).WithOutputFields(outputFields...)
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
	resultSets, err := store.client.Search(ctx, searchOption)
	if err != nil {
		return nil, SearchError(ctx, "dense ranking search", collectionName, err)
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
// collection.ErrSearchResultIncomplete.
func rankedCandidatesFromResultSets(ctx context.Context, collectionName string, resultSets []milvusclient.ResultSet, groupColumn collection.ScalarColumn, grouped bool) ([]Candidate, error) {
	if len(resultSets) == 0 || resultSets[0].ResultCount == 0 {
		return []Candidate{}, nil
	}
	resultSet := resultSets[0]
	relativePathColumn := resultSet.GetColumn(RelativePathField)
	if resultSet.IDs == nil || relativePathColumn == nil || len(resultSet.Scores) < resultSet.ResultCount {
		return nil, collection.ErrSearchResultIncomplete
	}
	candidates := make([]Candidate, 0, resultSet.ResultCount)
	for index := range resultSet.ResultCount {
		primaryKey, err := resultSet.IDs.GetAsString(index)
		if err != nil {
			return nil, rankingReadError(ctx, collectionName, "primary key", index, err)
		}
		relativePath, err := relativePathColumn.GetAsString(index)
		if err != nil {
			return nil, rankingReadError(ctx, collectionName, RelativePathField, index, err)
		}
		group := collection.AbsentCell(groupColumn.Name)
		if grouped {
			group, err = ScalarCellAt(resultSet.GetColumn(groupColumn.Name), groupColumn, index)
			if err != nil {
				return nil, err
			}
		}
		candidates = append(candidates, Candidate{PrimaryKey: primaryKey, RelativePath: relativePath, Group: group, Score: float64(resultSet.Scores[index])})
	}
	return candidates, nil
}

func rankingReadError(ctx context.Context, collectionName string, field string, index int, err error) error {
	slog.ErrorContext(ctx, "read collection ranking row failed", "collection", collectionName, "field", field, "index", index, "err", err)
	return fmt.Errorf("read ranking %s at %d from %s: %w", field, index, collectionName, err)
}

// Load queries content, output columns, and declared scalar columns for the
// selected rows by primary key and returns them in selection order. A row
// deleted after the ranking search is skipped.
func (store *Store) Load(ctx context.Context, collectionName string, selected []Candidate, scalarColumns []collection.ScalarColumn) ([]collection.Hit, error) {
	peerInfo, _ := peer.FromContext(ctx)
	if len(selected) == 0 {
		return []collection.Hit{}, nil
	}
	primaryKeys := make([]string, 0, len(selected))
	for _, candidate := range selected {
		primaryKeys = append(primaryKeys, candidate.PrimaryKey)
	}
	outputFields := []string{
		IDField,
		ContentField,
		RelativePathField,
		StartLineField,
		EndLineField,
		FileExtensionField,
		MetadataField,
		SplitPartField,
	}
	for _, declared := range scalarColumns {
		outputFields = append(outputFields, declared.Name)
	}
	resultSet, err := store.client.Query(ctx, milvusclient.NewQueryOption(collectionName).
		WithIDs(column.NewColumnVarChar(IDField, primaryKeys)).
		WithOutputFields(outputFields...))
	if err != nil {
		return nil, SearchError(ctx, "load ranked rows", collectionName, err)
	}
	hits, err := HitsFromResultSet(resultSet, scalarColumns)
	if err != nil {
		return nil, err
	}
	if resultSet.GetColumn(IDField) == nil && len(hits) > 0 {
		return nil, collection.ErrSearchResultIncomplete
	}
	byPrimaryKey := make(map[string]collection.Hit, len(hits))
	for _, hit := range hits {
		byPrimaryKey[hit.ID] = hit
	}
	ordered := make([]collection.Hit, 0, len(selected))
	for _, candidate := range selected {
		hit, found := byPrimaryKey[candidate.PrimaryKey]
		if !found {
			slog.WarnContext(ctx, "ranked row disappeared before its content load", "collection", collectionName, "relative_path", candidate.RelativePath, "peer", peerInfo.String())
			continue
		}
		hit.Score = candidate.Score
		ordered = append(ordered, hit)
	}
	return ordered, nil
}

// ExpressionSearch is a top-K search with a raw Milvus filter expression.
// Query feeds the BM25 sparse leg, which is lexical and never embeds. A
// non-positive Limit searches for defaultSearchLimit hits.
type ExpressionSearch struct {
	Collection string
	Vector     []float32
	Query      string
	Limit      int
	Expression string
}

// SearchExpression runs one top-K search limited to Limit and returns its rows
// without a second load. A hybrid store fuses a dense ANN leg and a BM25 leg,
// each at least ten deep, with the RRF reranker. A dense store runs one ANN
// search. The caller confirms the collection exists.
func (store *Store) SearchExpression(ctx context.Context, search ExpressionSearch) ([]collection.Hit, error) {
	searchLimit := search.Limit
	if searchLimit <= 0 {
		searchLimit = defaultSearchLimit
	}
	outputFields := []string{
		ContentField,
		RelativePathField,
		StartLineField,
		EndLineField,
		FileExtensionField,
		MetadataField,
		SplitPartField,
	}
	if store.options.Hybrid {
		legDepth := max(searchLimit, minimumLegDepth)
		denseRequest := milvusclient.NewAnnRequest(DenseVectorField, legDepth, entity.FloatVector(search.Vector))
		sparseRequest := milvusclient.NewAnnRequest(SparseVectorField, legDepth, entity.Text(search.Query))
		if search.Expression != "" {
			denseRequest = denseRequest.WithFilter(search.Expression)
			sparseRequest = sparseRequest.WithFilter(search.Expression)
		}
		hybridOption := milvusclient.NewHybridSearchOption(
			search.Collection,
			searchLimit,
			denseRequest,
			sparseRequest,
		).WithReranker(milvusclient.NewRRFReranker()).WithOutputFields(outputFields...)
		resultSets, err := store.client.HybridSearch(ctx, hybridOption)
		if err != nil {
			return nil, SearchError(ctx, "hybrid search", search.Collection, err)
		}
		return firstResultSetHits(resultSets)
	}

	searchOption := milvusclient.NewSearchOption(
		search.Collection,
		searchLimit,
		[]entity.Vector{entity.FloatVector(search.Vector)},
	).WithANNSField(DenseVectorField).WithOutputFields(outputFields...)
	if search.Expression != "" {
		searchOption = searchOption.WithFilter(search.Expression)
	}
	resultSets, err := store.client.Search(ctx, searchOption)
	if err != nil {
		return nil, SearchError(ctx, "dense search", search.Collection, err)
	}
	return firstResultSetHits(resultSets)
}

// firstResultSetHits decodes the rows of the first result set. A search returns
// one result set for its single query vector.
func firstResultSetHits(resultSets []milvusclient.ResultSet) ([]collection.Hit, error) {
	if len(resultSets) == 0 {
		return []collection.Hit{}, nil
	}
	return HitsFromResultSet(resultSets[0], nil)
}

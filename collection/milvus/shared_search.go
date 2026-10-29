package milvus

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"

	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
)

const (
	// rrfRankConstant is the k of the Milvus RRF reranker default:
	// score = sum of 1 / (k + rank), with rank starting at 1.
	rrfRankConstant = 60
	// denseFetchGrowth multiplies the vector search depth when the expanded
	// occurrences do not fill the leg.
	denseFetchGrowth = 4
	// fallbackKeyBatch bounds the vector keys one exact dense search lists.
	fallbackKeyBatch = 8192
	// matchQueryLimit is the default maximum number of rows one Milvus query returns.
	matchQueryLimit = 16_384
)

// legHit is one occurrence on a ranking leg.
type legHit struct {
	id           string
	relativePath string
	group        collection.ScalarCell
}

// Rank runs the dense and BM25 legs of a shared search and fuses them with the
// same RRF formula the Milvus reranker uses. The BM25 leg searches the
// occurrence collection with the exact filter. The dense leg searches the
// vector collection with the superset filter and expands each vector to the
// occurrences that match the exact filter, in vector score order.
func (store *SharedStore) Rank(ctx context.Context, request collection.SearchRequest) ([]Candidate, error) {
	occurrences := strings.TrimSpace(request.Collection)
	if occurrences == "" {
		return nil, errors.New("collection name is required")
	}
	compiled, err := collection.Compile(request.Filter)
	if err != nil {
		slog.ErrorContext(ctx, "compile collection filter failed", "collection", occurrences, "err", err)
		return nil, fmt.Errorf("compile filter for %s: %w", occurrences, err)
	}
	inline, err := collection.CompileInline(request.Filter)
	if err != nil {
		slog.ErrorContext(ctx, "compile inline collection filter failed", "collection", occurrences, "err", err)
		return nil, fmt.Errorf("compile filter for %s: %w", occurrences, err)
	}
	superset, err := SupersetExpression(request.Filter, request.Declaration)
	if err != nil {
		slog.ErrorContext(ctx, "compile superset filter failed", "collection", occurrences, "err", err)
		return nil, fmt.Errorf("compile superset filter for %s: %w", occurrences, err)
	}
	hasCollection, err := store.inner.client.HasCollection(ctx, milvusclient.NewHasCollectionOption(occurrences))
	if err != nil {
		return nil, WrapError(ctx, err, "check Milvus collection "+occurrences)
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
	// The BM25 and dense legs run at the same time. The first leg error cancels
	// the other leg, and Rank returns that error.
	legsCtx, cancelLegs := context.WithCancel(ctx)
	defer cancelLegs()
	var (
		legs    sync.WaitGroup
		failure sync.Once
		legErr  error
		sparse  []legHit
		dense   []legHit
	)
	fail := func(err error) {
		failure.Do(func() {
			legErr = err
			cancelLegs()
		})
	}
	legs.Go(func() {
		hits, err := store.sparseLeg(legsCtx, occurrences, request.Query, compiled, groupColumn, grouped, depth)
		if err != nil {
			fail(err)
			return
		}
		sparse = hits
	})
	legs.Go(func() {
		hits, err := store.denseLeg(legsCtx, occurrences, request.Vector, inline, superset, groupColumn, grouped, depth)
		if err != nil {
			fail(err)
			return
		}
		dense = hits
	})
	legs.Wait()
	if legErr != nil {
		return nil, legErr
	}
	return fuseLegs(dense, sparse, depth), nil
}

// sparseLeg returns the depth best BM25 occurrences in score order and then
// primary key order. Milvus keeps rows with equal scores inside one segment in
// index order. The leg reads past depth until the rows that tie with the row at
// depth are all present, and then cuts at depth.
func (store *SharedStore) sparseLeg(ctx context.Context, occurrences string, query string, compiled collection.CompiledFilter, groupColumn collection.ScalarColumn, grouped bool, depth int) ([]legHit, error) {
	fetch := min(depth*2, collection.RankingDepth)
	for {
		candidates, err := store.sparseCandidates(ctx, occurrences, query, compiled, groupColumn, grouped, fetch)
		if err != nil {
			return nil, err
		}
		sort.SliceStable(candidates, func(left int, right int) bool {
			if candidates[left].Score != candidates[right].Score {
				return candidates[left].Score > candidates[right].Score
			}
			return candidates[left].PrimaryKey < candidates[right].PrimaryKey
		})
		complete := len(candidates) < fetch || fetch >= collection.RankingDepth ||
			len(candidates) <= depth || candidates[depth-1].Score != candidates[len(candidates)-1].Score
		if complete {
			if len(candidates) > depth {
				candidates = candidates[:depth]
			}
			hits := make([]legHit, 0, len(candidates))
			for _, candidate := range candidates {
				hits = append(hits, legHit{id: candidate.PrimaryKey, relativePath: candidate.RelativePath, group: candidate.Group})
			}
			return hits, nil
		}
		fetch = min(fetch*denseFetchGrowth, collection.RankingDepth)
	}
}

func (store *SharedStore) sparseCandidates(ctx context.Context, occurrences string, query string, compiled collection.CompiledFilter, groupColumn collection.ScalarColumn, grouped bool, limit int) ([]Candidate, error) {
	outputFields := []string{RelativePathField}
	if grouped {
		outputFields = append(outputFields, groupColumn.Name)
	}
	option := milvusclient.NewSearchOption(occurrences, limit, []entity.Vector{entity.Text(query)}).
		WithANNSField(SparseVectorField).
		WithOutputFields(outputFields...)
	if compiled.Expression != "" {
		option = option.WithFilter(compiled.Expression)
	}
	for _, param := range compiled.Params {
		switch param.Type {
		case collection.ScalarTypeBool:
			option = option.WithTemplateParam(param.Name, param.Bools)
		case collection.ScalarTypeInt64:
			option = option.WithTemplateParam(param.Name, param.Int64s)
		case collection.ScalarTypeString:
			option = option.WithTemplateParam(param.Name, param.Strings)
		default:
			option = option.WithTemplateParam(param.Name, param.Strings)
		}
	}
	resultSets, err := store.inner.client.Search(ctx, option)
	if err != nil {
		return nil, SearchError(ctx, "sparse ranking search", occurrences, err)
	}
	return rankedCandidatesFromResultSets(ctx, occurrences, resultSets, groupColumn, grouped)
}

// scoredKey is one vector collection hit.
type scoredKey struct {
	key   string
	score float32
}

func (store *SharedStore) denseLeg(ctx context.Context, occurrences string, vector []float32, inline string, superset string, groupColumn collection.ScalarColumn, grouped bool, depth int) ([]legHit, error) {
	vectors := VectorCollectionName(occurrences)
	fetch := depth
	for {
		keys, err := store.searchVectors(ctx, vectors, vector, superset, fetch)
		if err != nil {
			return nil, err
		}
		hits, err := store.expandKeys(ctx, occurrences, keys, inline, groupColumn, grouped, depth)
		if err != nil {
			return nil, err
		}
		exhausted := len(keys) < fetch
		if len(hits) >= depth || exhausted {
			return hits, nil
		}
		if fetch >= collection.RankingDepth {
			return store.exactDenseLeg(ctx, occurrences, vector, inline, groupColumn, grouped, depth)
		}
		fetch = min(fetch*denseFetchGrowth, collection.RankingDepth)
	}
}

// exactDenseLeg runs when the superset filter keeps more vectors than one
// search can return. It lists the vector keys of every occurrence that
// matches the exact filter and searches only those vectors.
func (store *SharedStore) exactDenseLeg(ctx context.Context, occurrences string, vector []float32, inline string, groupColumn collection.ScalarColumn, grouped bool, depth int) ([]legHit, error) {
	expression := inline
	if expression == "" {
		expression = IDField + ` != ""`
	}
	candidateKeys, err := store.iterateVectorKeys(ctx, occurrences, expression)
	if err != nil {
		return nil, err
	}
	vectors := VectorCollectionName(occurrences)
	merged := make([]scoredKey, 0)
	for start := 0; start < len(candidateKeys); start += fallbackKeyBatch {
		batch := candidateKeys[start:min(start+fallbackKeyBatch, len(candidateKeys))]
		keys, err := store.searchVectors(ctx, vectors, vector, vectorKeysExpression(batch), min(depth, len(batch)))
		if err != nil {
			return nil, err
		}
		merged = append(merged, keys...)
	}
	sort.SliceStable(merged, func(left int, right int) bool {
		if merged[left].score != merged[right].score {
			return merged[left].score > merged[right].score
		}
		return merged[left].key < merged[right].key
	})
	return store.expandKeys(ctx, occurrences, merged, inline, groupColumn, grouped, depth)
}

func (store *SharedStore) searchVectors(ctx context.Context, vectors string, vector []float32, expression string, limit int) ([]scoredKey, error) {
	option := milvusclient.NewSearchOption(vectors, limit, []entity.Vector{entity.FloatVector(vector)}).
		WithANNSField(DenseVectorField).
		WithOutputFields(VectorKeyField)
	if len(store.inner.options.DenseSearchParams) > 0 {
		option = option.WithAnnParam(denseAnnParam(store.inner.options.DenseSearchParams))
	}
	if expression != "" {
		option = option.WithFilter(expression)
	}
	resultSets, err := store.inner.client.Search(ctx, option)
	if err != nil {
		return nil, SearchError(ctx, "dense vector search", vectors, err)
	}
	if len(resultSets) == 0 || resultSets[0].ResultCount == 0 {
		return nil, nil
	}
	resultSet := resultSets[0]
	if resultSet.IDs == nil || len(resultSet.Scores) < resultSet.ResultCount {
		return nil, collection.ErrSearchResultIncomplete
	}
	keys := make([]scoredKey, 0, resultSet.ResultCount)
	for index := range resultSet.ResultCount {
		key, err := resultSet.IDs.GetAsString(index)
		if err != nil {
			return nil, rankingReadError(ctx, vectors, VectorKeyField, index, err)
		}
		keys = append(keys, scoredKey{key: key, score: resultSet.Scores[index]})
	}
	sort.SliceStable(keys, func(left int, right int) bool {
		if keys[left].score != keys[right].score {
			return keys[left].score > keys[right].score
		}
		return keys[left].key < keys[right].key
	})
	return keys, nil
}

// expandKeys returns up to depth occurrences of keys that match the exact
// filter. It walks keys in the given order and lists the occurrences of one
// vector by primary key.
func (store *SharedStore) expandKeys(ctx context.Context, occurrences string, keys []scoredKey, inline string, groupColumn collection.ScalarColumn, grouped bool, depth int) ([]legHit, error) {
	hits := make([]legHit, 0, depth)
	outputFields := []string{IDField, RelativePathField, VectorKeyField}
	if grouped {
		outputFields = append(outputFields, groupColumn.Name)
	}
	for start := 0; start < len(keys) && len(hits) < depth; start += vectorKeyBatchSize {
		batch := keys[start:min(start+vectorKeyBatchSize, len(keys))]
		names := make([]string, 0, len(batch))
		for _, key := range batch {
			names = append(names, key.key)
		}
		expression := vectorKeysExpression(names)
		if inline != "" {
			expression = "(" + expression + ") and (" + inline + ")"
		}
		byKey := map[string][]legHit{}
		err := store.readMatches(ctx, occurrences, expression, outputFields, func(resultSet milvusclient.ResultSet) error {
			return collectLegHits(ctx, occurrences, resultSet, groupColumn, grouped, byKey)
		})
		if err != nil {
			return nil, err
		}
		for _, key := range batch {
			hits = appendOrderedMembers(hits, byKey[key.key], depth)
		}
	}
	return hits, nil
}

// readMatches calls page with every row in collectionName that matches expression.
// It first uses one Strong query, which costs less than a Strong query iterator.
// If the query returns matchQueryLimit rows, the result may be incomplete.
// readMatches discards that result and uses the Strong query iterator for all matches.
func (store *SharedStore) readMatches(ctx context.Context, collectionName string, expression string, outputFields []string, page func(milvusclient.ResultSet) error) error {
	resultSet, err := store.inner.client.Query(ctx, milvusclient.NewQueryOption(collectionName).
		WithFilter(expression).
		WithOutputFields(outputFields...).
		WithLimit(matchQueryLimit).
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		return SearchError(ctx, "query", collectionName, err)
	}
	if resultSet.ResultCount >= matchQueryLimit {
		return store.iterate(ctx, collectionName, expression, outputFields, page)
	}
	if resultSet.ResultCount == 0 {
		return nil
	}
	return page(resultSet)
}

// collectLegHits adds the occurrence rows of one result page to byKey under
// each row's vector key.
func collectLegHits(ctx context.Context, occurrences string, resultSet milvusclient.ResultSet, groupColumn collection.ScalarColumn, grouped bool, byKey map[string][]legHit) error {
	idColumn := resultSet.GetColumn(IDField)
	pathColumn := resultSet.GetColumn(RelativePathField)
	keyColumn := resultSet.GetColumn(VectorKeyField)
	if idColumn == nil || pathColumn == nil || keyColumn == nil {
		return collection.ErrSearchResultIncomplete
	}
	for rowIndex := range resultSet.ResultCount {
		id, err := idColumn.GetAsString(rowIndex)
		if err != nil {
			return rankingReadError(ctx, occurrences, IDField, rowIndex, err)
		}
		relativePath, err := pathColumn.GetAsString(rowIndex)
		if err != nil {
			return rankingReadError(ctx, occurrences, RelativePathField, rowIndex, err)
		}
		key, err := keyColumn.GetAsString(rowIndex)
		if err != nil {
			return rankingReadError(ctx, occurrences, VectorKeyField, rowIndex, err)
		}
		group := collection.AbsentCell(groupColumn.Name)
		if grouped {
			group, err = ScalarCellAt(resultSet.GetColumn(groupColumn.Name), groupColumn, rowIndex)
			if err != nil {
				return err
			}
		}
		byKey[key] = append(byKey[key], legHit{id: id, relativePath: relativePath, group: group})
	}
	return nil
}

// appendOrderedMembers sorts the occurrences of one vector by primary key, the
// order Milvus gives rows with equal scores, and appends them to hits until
// hits has depth rows.
func appendOrderedMembers(hits []legHit, members []legHit, depth int) []legHit {
	sort.Slice(members, func(left int, right int) bool {
		return members[left].id < members[right].id
	})
	for _, member := range members {
		if len(hits) >= depth {
			break
		}
		hits = append(hits, member)
	}
	return hits
}

// fuseLegs scores each occurrence with the sum over legs of 1 / (60 + rank)
// and keeps the depth best. The score is rounded to float32, the precision
// the Milvus reranker returns.
func fuseLegs(dense []legHit, sparse []legHit, depth int) []Candidate {
	scores := map[string]float64{}
	rows := map[string]legHit{}
	for _, leg := range [][]legHit{dense, sparse} {
		for position, hit := range leg {
			scores[hit.id] += 1.0 / float64(rrfRankConstant+position+1)
			if _, seen := rows[hit.id]; !seen {
				rows[hit.id] = hit
			}
		}
	}
	candidates := make([]Candidate, 0, len(scores))
	for id, score := range scores {
		row := rows[id]
		candidates = append(candidates, Candidate{PrimaryKey: id, RelativePath: row.relativePath, Group: row.group, Score: float64(float32(score))})
	}
	sortRankedCandidates(candidates)
	if len(candidates) > depth {
		candidates = candidates[:depth]
	}
	return candidates
}

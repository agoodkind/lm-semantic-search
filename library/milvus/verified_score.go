package milvus

import (
	"context"
	"fmt"
	"log/slog"
	"math"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/internal/vectorcodec"
	"goodkind.io/lm-semantic-search/library/observation"
)

// ScoreExactVerified returns native FLAT COSINE scores and verifies each
// identity and canonical float32 checksum from one strongly consistent search.
func (store *Store) ScoreExactVerified(ctx context.Context, query []float32, identities []library.VectorIdentity) (_ []library.VectorScore, err error) {
	if store.config.QueryMode == QueryModeLargeTopK {
		reader, err := store.beginExactSnapshot(ctx)
		if err != nil {
			return nil, err
		}
		return reader.ScoreExactVerified(ctx, query, identities)
	}
	ctx, span := observation.Start(ctx, store.config.Observer, observation.VerifiedExactScoring)
	counts := observation.VectorData{Requested: len(identities)}
	defer func() { span.End(ctx, err, observation.Data{Vector: counts}) }()
	defer func() {
		slog.InfoContext(ctx, "verified exact vector search completed",
			"operation", string(observation.VerifiedExactScoring),
			"vectors", counts.Requested, "verified_vectors", counts.Verified,
			"client_search_duration_ns", counts.ClientSearchDuration.Nanoseconds(),
			"local_verification_duration_ns", counts.LocalVerificationDuration.Nanoseconds(),
		)
		if err != nil {
			slog.ErrorContext(ctx, "verified exact vector search failed", "vectors", len(identities), "err", err)
		}
	}()
	bound, err := store.binding(ctx)
	if err != nil {
		return nil, err
	}
	ids := make([]string, len(identities))
	for index, identity := range identities {
		ids[index] = identity.ID
	}
	if err := validateScoreRequest(ctx, query, ids, bound.Dimension, store.MaxExactScoreIDs()); err != nil {
		return nil, err
	}
	if len(ids) == 0 {
		return []library.VectorScore{}, nil
	}
	started := clock.Now()
	request, err := milvusclient.NewSearchOption(store.config.Collection, len(ids), []entity.Vector{entity.FloatVector(query)}).
		WithANNSField(fieldVector).
		WithFilter(idsFilterExpression).
		WithTemplateParam(idsTemplateName, ids).
		WithOutputFields(fieldIdentityDigest, fieldChecksum, fieldVector).
		WithConsistencyLevel(entity.ClStrong).Request()
	if err != nil {
		return nil, fmt.Errorf("verified exact vector search request: %w", err)
	}
	service := store.client.GetService()
	if service == nil {
		return nil, fmt.Errorf("verified exact vector search: %w", merr.WrapErrServiceNotReady("SDK", 0, "not connected"))
	}
	response, err := service.Search(ctx, request)
	counts.ClientSearchDuration = clock.Now().Sub(started)
	if err := merr.CheckRPCCall(response, err); err != nil {
		return nil, fmt.Errorf("verified exact vector search over %d vectors: %w", len(ids), err)
	}
	started = clock.Now()
	results, failure := verifiedNativeResults(response, bound.Dimension)
	var scores []library.VectorScore
	if failure == nil {
		scores, failure = verifiedSearchScores(results, identities, bound.Dimension)
	}
	counts.LocalVerificationDuration = clock.Now().Sub(started)
	if failure != nil {
		if failure.category == nil {
			return nil, fmt.Errorf("%s: %w", failure.detail, failure.cause)
		}
		if failure.cause != nil {
			return nil, fmt.Errorf("%w: %s: %w", failure.category, failure.detail, failure.cause)
		}
		return nil, fmt.Errorf("%w: %s", failure.category, failure.detail)
	}
	counts.Verified = len(identities)
	return scores, nil
}

type verifiedSearchFailure struct {
	category error
	detail   string
	cause    error
}

func verifiedSearchScores(results []milvusclient.ResultSet, identities []library.VectorIdentity, dimension int) ([]library.VectorScore, *verifiedSearchFailure) {
	if len(results) == 0 {
		return nil, &verifiedSearchFailure{category: library.ErrVectorMissing, detail: "verified exact search returned no result"}
	}
	if len(results) != 1 {
		return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: fmt.Sprintf("verified exact search returned %d query results", len(results))}
	}
	result := results[0]
	if result.Err != nil {
		return nil, &verifiedSearchFailure{detail: "verified exact search result", cause: result.Err}
	}
	if result.ResultCount == 0 {
		return nil, &verifiedSearchFailure{category: library.ErrVectorMissing, detail: "verified exact search returned no vectors"}
	}
	if result.ResultCount > len(identities) {
		return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "verified exact search returned too many vectors"}
	}
	fields, err := verifiedSearchFields(result)
	if err != nil {
		return nil, err
	}
	expected := make(map[string]library.VectorIdentity, len(identities))
	for _, identity := range identities {
		expected[identity.ID] = identity
	}
	scores := make(map[string]float64, result.ResultCount)
	for row := range result.ResultCount {
		id := fields.ids.Data()[row]
		identity, found := expected[id]
		if !found {
			return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "verified exact search returned unexpected ID " + id}
		}
		if _, duplicate := scores[id]; duplicate {
			return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "verified exact search returned ID " + id + " twice"}
		}
		if err := verifySearchVector(fields, row, identity, dimension); err != nil {
			return nil, err
		}
		score := float64(result.Scores[row])
		if math.IsNaN(score) || math.IsInf(score, 0) {
			return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "verified exact score for " + id + " is not finite"}
		}
		scores[id] = score
	}
	ordered := make([]library.VectorScore, len(identities))
	for index, identity := range identities {
		score, found := scores[identity.ID]
		if !found {
			return nil, &verifiedSearchFailure{category: library.ErrVectorMissing, detail: "verified exact search returned no score for " + identity.ID}
		}
		ordered[index] = library.VectorScore{ID: identity.ID, Score: score}
	}
	return ordered, nil
}

type verifiedFields struct {
	ids, digests, checksums *column.ColumnVarChar
	vectors                 *column.ColumnFloatVector
}

func verifiedSearchFields(result milvusclient.ResultSet) (verifiedFields, *verifiedSearchFailure) {
	ids, idsOK := result.IDs.(*column.ColumnVarChar)
	digests, digestsOK := result.GetColumn(fieldIdentityDigest).(*column.ColumnVarChar)
	checksums, checksumsOK := result.GetColumn(fieldChecksum).(*column.ColumnVarChar)
	vectors, vectorsOK := result.GetColumn(fieldVector).(*column.ColumnFloatVector)
	if !idsOK || !digestsOK || !checksumsOK || !vectorsOK || ids == nil || digests == nil || checksums == nil || vectors == nil {
		return verifiedFields{}, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "verified exact search returned invalid field types"}
	}
	count := result.ResultCount
	if count < 0 || ids.Len() != count || digests.Len() != count || checksums.Len() != count || vectors.Len() != count || len(result.Scores) != count {
		return verifiedFields{}, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "verified exact search returned inconsistent field lengths"}
	}
	return verifiedFields{ids: ids, digests: digests, checksums: checksums, vectors: vectors}, nil
}

func verifySearchVector(fields verifiedFields, row int, identity library.VectorIdentity, dimension int) *verifiedSearchFailure {
	values := fields.vectors.Data()[row]
	if err := vectorcodec.Validate(values, dimension); err != nil {
		return &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "vector " + identity.ID, cause: err}
	}
	if fields.digests.Data()[row] != identity.IdentityDigest || fields.checksums.Data()[row] != identity.Checksum || vectorcodec.Checksum(values) != identity.Checksum {
		return &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "vector " + identity.ID + " does not match its identity or checksum"}
	}
	return nil
}

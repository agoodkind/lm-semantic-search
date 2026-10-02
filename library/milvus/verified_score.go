package milvus

import (
	"context"
	"fmt"
	"log/slog"
	"math"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/internal/vectorcodec"
	"goodkind.io/lm-semantic-search/library/observation"
)

// ScoreExactVerified returns native FLAT COSINE scores and verifies each
// identity and canonical float32 checksum from one strongly consistent search.
func (store *Store) ScoreExactVerified(ctx context.Context, query []float32, identities []library.VectorIdentity) (_ []library.VectorScore, err error) {
	ctx, span := observation.Start(ctx, store.config.Observer, observation.VerifiedExactScoring)
	counts := observation.VectorData{Requested: len(identities)}
	defer func() { span.End(ctx, err, observation.Data{Vector: counts}) }()
	defer func() {
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
	if err := validateScoreRequest(ctx, query, ids, bound.Dimension); err != nil {
		return nil, err
	}
	if len(ids) == 0 {
		return []library.VectorScore{}, nil
	}
	results, err := store.client.Search(ctx,
		milvusclient.NewSearchOption(store.config.Collection, len(ids), []entity.Vector{entity.FloatVector(query)}).
			WithANNSField(fieldVector).
			WithFilter(idsFilterExpression).
			WithTemplateParam(idsTemplateName, ids).
			WithOutputFields(fieldIdentityDigest, fieldChecksum, fieldVector).
			WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		return nil, fmt.Errorf("verified exact vector search over %d vectors: %w", len(ids), err)
	}
	scores, err := verifiedSearchScores(ctx, results, identities, bound.Dimension)
	if err == nil {
		counts.Verified = len(identities)
	}
	return scores, err
}

func verifiedSearchScores(ctx context.Context, results []milvusclient.ResultSet, identities []library.VectorIdentity, dimension int) (_ []library.VectorScore, err error) {
	defer func() {
		if err != nil {
			slog.ErrorContext(ctx, "decode verified exact search failed", "err", err)
		}
	}()
	if len(results) == 0 {
		return nil, fmt.Errorf("%w: verified exact search returned no result", library.ErrVectorMissing)
	}
	if len(results) != 1 {
		return nil, fmt.Errorf("%w: verified exact search returned %d query results", library.ErrVectorCorrupt, len(results))
	}
	result := results[0]
	if result.Err != nil {
		return nil, fmt.Errorf("verified exact search result: %w", result.Err)
	}
	if result.ResultCount == 0 {
		return nil, fmt.Errorf("%w: verified exact search returned no vectors", library.ErrVectorMissing)
	}
	if result.ResultCount > len(identities) {
		return nil, fmt.Errorf("%w: verified exact search returned too many vectors", library.ErrVectorCorrupt)
	}
	fields, err := verifiedSearchFields(ctx, result)
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
			return nil, fmt.Errorf("%w: verified exact search returned unexpected ID %s", library.ErrVectorCorrupt, id)
		}
		if _, duplicate := scores[id]; duplicate {
			return nil, fmt.Errorf("%w: verified exact search returned ID %s twice", library.ErrVectorCorrupt, id)
		}
		if err := verifySearchVector(ctx, fields, row, identity, dimension); err != nil {
			return nil, err
		}
		score := float64(result.Scores[row])
		if math.IsNaN(score) || math.IsInf(score, 0) {
			return nil, fmt.Errorf("%w: verified exact score for %s is not finite", library.ErrVectorCorrupt, id)
		}
		scores[id] = score
	}
	ordered := make([]library.VectorScore, len(identities))
	for index, identity := range identities {
		score, found := scores[identity.ID]
		if !found {
			return nil, fmt.Errorf("%w: verified exact search returned no score for %s", library.ErrVectorMissing, identity.ID)
		}
		ordered[index] = library.VectorScore{ID: identity.ID, Score: score}
	}
	return ordered, nil
}

type verifiedFields struct {
	ids, digests, checksums *column.ColumnVarChar
	vectors                 *column.ColumnFloatVector
}

func verifiedSearchFields(ctx context.Context, result milvusclient.ResultSet) (_ verifiedFields, err error) {
	defer func() {
		if err != nil {
			slog.ErrorContext(ctx, "decode verified exact fields failed", "err", err)
		}
	}()
	ids, idsOK := result.IDs.(*column.ColumnVarChar)
	digests, digestsOK := result.GetColumn(fieldIdentityDigest).(*column.ColumnVarChar)
	checksums, checksumsOK := result.GetColumn(fieldChecksum).(*column.ColumnVarChar)
	vectors, vectorsOK := result.GetColumn(fieldVector).(*column.ColumnFloatVector)
	if !idsOK || !digestsOK || !checksumsOK || !vectorsOK || ids == nil || digests == nil || checksums == nil || vectors == nil {
		return verifiedFields{}, fmt.Errorf("%w: verified exact search returned invalid field types", library.ErrVectorCorrupt)
	}
	count := result.ResultCount
	if count < 0 || ids.Len() != count || digests.Len() != count || checksums.Len() != count || vectors.Len() != count || len(result.Scores) != count {
		return verifiedFields{}, fmt.Errorf("%w: verified exact search returned inconsistent field lengths", library.ErrVectorCorrupt)
	}
	return verifiedFields{ids: ids, digests: digests, checksums: checksums, vectors: vectors}, nil
}

func verifySearchVector(ctx context.Context, fields verifiedFields, row int, identity library.VectorIdentity, dimension int) (err error) {
	defer func() {
		if err != nil {
			slog.ErrorContext(ctx, "verify exact search vector failed", "err", err)
		}
	}()
	values := fields.vectors.Data()[row]
	if err := vectorcodec.Validate(values, dimension); err != nil {
		return fmt.Errorf("%w: vector %s: %w", library.ErrVectorCorrupt, identity.ID, err)
	}
	if fields.digests.Data()[row] != identity.IdentityDigest || fields.checksums.Data()[row] != identity.Checksum || vectorcodec.Checksum(values) != identity.Checksum {
		return fmt.Errorf("%w: vector %s does not match its identity or checksum", library.ErrVectorCorrupt, identity.ID)
	}
	return nil
}

package milvus

import (
	"context"
	"fmt"
	"log/slog"
	"math"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/observation"
)

type snapshotReader struct {
	store     *Store
	timestamp uint64
	dimension int
}

// BeginExactScoring fixes the backend timestamp for every score and byte request
// in large_topk mode. Normal mode retains its existing strong operations.
func (store *Store) BeginExactScoring(ctx context.Context) (library.ExactScoreReader, error) {
	if store.config.QueryMode != QueryModeLargeTopK {
		return store, nil
	}
	return store.beginExactSnapshot(ctx)
}

func (store *Store) beginExactSnapshot(ctx context.Context) (_ *snapshotReader, err error) {
	defer func() {
		if err != nil {
			slog.ErrorContext(ctx, "begin native scoring snapshot failed", "err", err)
		}
	}()
	bound, err := store.binding(ctx)
	if err != nil {
		return nil, err
	}
	service := store.client.GetService()
	if service == nil {
		return nil, fmt.Errorf("begin native snapshot: %w", merr.WrapErrServiceNotReady("SDK", 0, "not connected"))
	}
	response, err := service.Query(ctx, &milvuspb.QueryRequest{
		CollectionName:   store.config.Collection,
		Expr:             fieldVectorID + " != \"\"",
		OutputFields:     []string{fieldVectorID},
		ConsistencyLevel: commonpb.ConsistencyLevel_Strong,
		QueryParams:      []*commonpb.KeyValuePair{{Key: "iterator", Value: "true"}, {Key: "limit", Value: "1"}},
	})
	if err := merr.CheckRPCCall(response, err); err != nil {
		return nil, fmt.Errorf("begin native snapshot: %w", err)
	}
	if response.GetSessionTs() == 0 {
		return nil, fmt.Errorf("%w: native snapshot timestamp is absent", library.ErrVectorCorrupt)
	}
	return &snapshotReader{store: store, timestamp: response.GetSessionTs(), dimension: bound.Dimension}, nil
}

func (reader *snapshotReader) ScoreExact(ctx context.Context, query []float32, ids []string) (_ []library.VectorScore, err error) {
	defer func() {
		if err != nil {
			slog.ErrorContext(ctx, "snapshot exact scoring failed", "vectors", len(ids), "err", err)
		}
	}()
	if err := validateScoreRequest(ctx, query, ids, reader.dimension, reader.store.MaxExactScoreIDs()); err != nil {
		return nil, err
	}
	if len(ids) == 0 {
		return []library.VectorScore{}, nil
	}
	request, err := milvusclient.NewSearchOption(reader.store.config.Collection, len(ids), []entity.Vector{entity.FloatVector(query)}).
		WithANNSField(fieldVector).WithFilter(idsFilterExpression).WithTemplateParam(idsTemplateName, ids).
		WithConsistencyLevel(entity.ClStrong).WithSearchParam("iterator", "true").Request()
	if err != nil {
		return nil, fmt.Errorf("snapshot exact scoring request: %w", err)
	}
	request.GuaranteeTimestamp = reader.timestamp
	service := reader.store.client.GetService()
	if service == nil {
		return nil, fmt.Errorf("snapshot exact scoring: %w", merr.WrapErrServiceNotReady("SDK", 0, "not connected"))
	}
	response, err := service.Search(ctx, request)
	if err := merr.CheckRPCCall(response, err); err != nil {
		return nil, fmt.Errorf("snapshot exact scoring: %w", err)
	}
	scores, failure := snapshotScores(response, ids)
	if failure != nil {
		return nil, fmt.Errorf("%w: %s", failure.category, failure.detail)
	}
	return scores, nil
}

func snapshotScores(response *milvuspb.SearchResults, expected []string) ([]library.VectorScore, *verifiedSearchFailure) {
	data := response.GetResults()
	if data == nil || data.GetNumQueries() != 1 || len(data.GetTopks()) != 1 || data.GetTopks()[0] < 0 || data.GetTopks()[0] > int64(len(expected)) {
		return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "native score response has invalid counts"}
	}
	count := int(data.GetTopks()[0])
	ids, ok := data.GetIds().GetIdField().(*schemapb.IDs_StrId)
	if !ok || ids == nil || ids.StrId == nil || len(ids.StrId.GetData()) != count || len(data.GetScores()) != count {
		return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "native score response has invalid IDs or scores"}
	}
	byID := make(map[string]float64, count)
	requested := make(map[string]bool, len(expected))
	for _, id := range expected {
		requested[id] = true
	}
	for index, id := range ids.StrId.GetData() {
		if !requested[id] {
			return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "native score response contains unexpected ID " + id}
		}
		value := float64(data.GetScores()[index])
		if _, duplicate := byID[id]; duplicate || math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "native score response has duplicate IDs or nonfinite scores"}
		}
		byID[id] = value
	}
	scores := make([]library.VectorScore, len(expected))
	for index, id := range expected {
		value, found := byID[id]
		if !found {
			return nil, &verifiedSearchFailure{category: library.ErrVectorMissing, detail: "native score response is missing " + id}
		}
		scores[index] = library.VectorScore{ID: id, Score: value}
	}
	if count != len(expected) {
		return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "native score response contains unexpected IDs"}
	}
	return scores, nil
}

func (reader *snapshotReader) VerifyStrong(ctx context.Context, identities []library.VectorIdentity) error {
	_, err := reader.verifyStrong(ctx, identities)
	return err
}

func (reader *snapshotReader) verifyStrong(ctx context.Context, identities []library.VectorIdentity) (counts observation.VectorData, err error) {
	ctx, span := observation.Start(ctx, reader.store.config.Observer, observation.StrongVerification)
	counts = observation.VectorData{Requested: len(identities)}
	defer func() {
		span.End(ctx, err, observation.Data{Vector: counts})
		if err != nil {
			slog.ErrorContext(ctx, "snapshot vector verification failed", "err", err)
		}
	}()
	step := reader.store.verificationBatchRows(reader.dimension)
	for begin := 0; begin < len(identities); begin += step {
		batch := identities[begin:min(begin+step, len(identities))]
		failure := reader.verifyBatch(ctx, batch, &counts)
		if failure != nil {
			if failure.cause != nil {
				if failure.category != nil {
					return counts, fmt.Errorf("%w: %s: %w", failure.category, failure.detail, failure.cause)
				}
				return counts, fmt.Errorf("%s: %w", failure.detail, failure.cause)
			}
			return counts, fmt.Errorf("%w: %s", failure.category, failure.detail)
		}
		counts.Verified += len(batch)
	}
	return counts, nil
}

func (store *Store) verificationBatchRows(dimension int) int {
	return min(store.config.MaxVerifyBatchRows, max(1, maxVerifyVectorBytes/(dimension*4)))
}

func (reader *snapshotReader) verifyBatch(ctx context.Context, identities []library.VectorIdentity, counts *observation.VectorData) *verifiedSearchFailure {
	ids := make([]string, len(identities))
	for index, identity := range identities {
		ids[index] = identity.ID
	}
	request, err := milvusclient.NewQueryOption(reader.store.config.Collection).
		WithFilter(idsFilterExpression).WithTemplateParam(idsTemplateName, ids).
		WithOutputFields(fieldVectorID, fieldIdentityDigest, fieldChecksum, fieldVector).
		WithConsistencyLevel(entity.ClStrong).WithLimit(len(ids)).Request()
	if err != nil {
		return &verifiedSearchFailure{detail: "snapshot vector request", cause: err}
	}
	request.GuaranteeTimestamp = reader.timestamp
	request.QueryParams = append(request.QueryParams, &commonpb.KeyValuePair{Key: "iterator", Value: "true"})
	service := reader.store.client.GetService()
	if service == nil {
		return &verifiedSearchFailure{detail: "snapshot vector verification", cause: merr.WrapErrServiceNotReady("SDK", 0, "not connected")}
	}
	started := clock.Now()
	response, err := service.Query(ctx, request)
	counts.ClientQueryDuration += clock.Now().Sub(started)
	if err := merr.CheckRPCCall(response, err); err != nil {
		return &verifiedSearchFailure{detail: "snapshot vector verification", cause: err}
	}
	started = clock.Now()
	failure := verifySnapshotFields(response, identities, reader.dimension)
	counts.LocalVerificationDuration += clock.Now().Sub(started)
	return failure
}

func verifySnapshotFields(response *milvuspb.QueryResults, identities []library.VectorIdentity, dimension int) *verifiedSearchFailure {
	var ids *column.ColumnVarChar
	seen := make(map[string]bool, 4)
	result := milvusclient.ResultSet{}
	for _, field := range response.GetFieldsData() {
		name := field.GetFieldName()
		if name != fieldVectorID && name != fieldIdentityDigest && name != fieldChecksum && name != fieldVector {
			continue
		}
		if seen[name] {
			return &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "snapshot vector response contains duplicate fields"}
		}
		seen[name] = true
		if name == fieldVectorID {
			count := int64(len(field.GetScalars().GetStringData().GetData()))
			decoded, failure := verifiedNativeStringField(field, count)
			if failure != nil {
				return failure
			}
			ids = decoded
		}
	}
	if ids == nil {
		return &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "snapshot vector response has no valid ID field"}
	}
	count := ids.Len()
	if count > len(identities) {
		return &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "snapshot vector response has too many IDs"}
	}
	if count == 0 {
		return &verifiedSearchFailure{category: library.ErrVectorMissing, detail: "snapshot vector response has no vectors"}
	}
	result.IDs, result.ResultCount = ids, count
	result.Scores = make([]float32, count)
	for _, field := range response.GetFieldsData() {
		name := field.GetFieldName()
		if name != fieldIdentityDigest && name != fieldChecksum && name != fieldVector {
			continue
		}
		decoded, failure := verifiedNativeField(field, int64(count), dimension)
		if failure != nil {
			return failure
		}
		result.Fields = append(result.Fields, decoded)
	}
	_, failure := verifiedSearchScores([]milvusclient.ResultSet{result}, identities, dimension)
	return failure
}

func verifiedNativeStringField(field *schemapb.FieldData, count int64) (*column.ColumnVarChar, *verifiedSearchFailure) {
	values, ok := field.GetScalars().GetData().(*schemapb.ScalarField_StringData)
	if field.GetType() != schemapb.DataType_VarChar || !ok || values == nil || values.StringData == nil || int64(len(values.StringData.GetData())) != count {
		return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "snapshot vector response has invalid ID types"}
	}
	valid := field.GetValidData()
	if len(valid) != 0 && int64(len(valid)) != count {
		return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "snapshot vector response has invalid ID validity lengths"}
	}
	for _, present := range valid {
		if !present {
			return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "snapshot vector response has null IDs"}
		}
	}
	return column.NewColumnVarChar(field.GetFieldName(), values.StringData.GetData()), nil
}

func (reader *snapshotReader) ScoreExactVerified(ctx context.Context, query []float32, identities []library.VectorIdentity) (_ []library.VectorScore, err error) {
	ctx, span := observation.Start(ctx, reader.store.config.Observer, observation.VerifiedExactScoring)
	counts := observation.VectorData{Requested: len(identities)}
	defer func() {
		span.End(ctx, err, observation.Data{Vector: counts})
		slog.InfoContext(ctx, "snapshot verified scoring completed", "vectors", counts.Requested, "verified_vectors", counts.Verified,
			"client_search_duration_ns", counts.ClientSearchDuration.Nanoseconds(), "client_query_duration_ns", counts.ClientQueryDuration.Nanoseconds(),
			"local_verification_duration_ns", counts.LocalVerificationDuration.Nanoseconds())
	}()
	ids := make([]string, len(identities))
	for index, identity := range identities {
		ids[index] = identity.ID
	}
	started := clock.Now()
	scores, err := reader.ScoreExact(ctx, query, ids)
	counts.ClientSearchDuration = clock.Now().Sub(started)
	if err != nil {
		return nil, err
	}
	verified, err := reader.verifyStrong(ctx, identities)
	counts.ClientQueryDuration = verified.ClientQueryDuration
	counts.LocalVerificationDuration = verified.LocalVerificationDuration
	counts.Verified = verified.Verified
	if err != nil {
		return nil, err
	}
	return scores, nil
}

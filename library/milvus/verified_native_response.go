package milvus

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
	"goodkind.io/lm-semantic-search/library"
)

func (store *Store) searchVerifiedNative(ctx context.Context, option milvusclient.SearchOption) (_ *milvuspb.SearchResults, err error) {
	defer func() {
		if err != nil {
			slog.ErrorContext(ctx, "verified native response failed", "err", err)
		}
	}()
	request, err := option.Request()
	if err != nil {
		return nil, fmt.Errorf("%w", err)
	}
	service := store.client.GetService()
	if service == nil {
		return nil, fmt.Errorf("%w", merr.WrapErrServiceNotReady("SDK", 0, "not connected"))
	}
	response, err := service.Search(ctx, request)
	if err := merr.CheckRPCCall(response, err); err != nil {
		return response, fmt.Errorf("%w", err)
	}
	return response, nil
}

func verifiedNativeResults(response *milvuspb.SearchResults, dimension int) ([]milvusclient.ResultSet, *verifiedSearchFailure) {
	data := response.GetResults()
	if data == nil || data.GetNumQueries() != 1 || len(data.GetTopks()) != 1 {
		return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "verified exact search returned invalid query counts"}
	}
	count := data.GetTopks()[0]
	if count < 0 || count > maxSearchLimit || int64(len(data.GetScores())) != count {
		return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "verified exact search returned inconsistent score lengths"}
	}
	if count == 0 {
		return nil, &verifiedSearchFailure{category: library.ErrVectorMissing, detail: "verified exact search returned no vectors"}
	}
	ids, ok := data.GetIds().GetIdField().(*schemapb.IDs_StrId)
	if !ok || ids == nil || ids.StrId == nil || int64(len(ids.StrId.GetData())) != count {
		return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "verified exact search returned invalid ID fields"}
	}
	result := milvusclient.ResultSet{
		ResultCount: int(count),
		IDs:         column.NewColumnVarChar(fieldVectorID, ids.StrId.GetData()),
		Scores:      data.GetScores(),
	}
	seen := make(map[string]bool, 3)
	for _, field := range data.GetFieldsData() {
		name := field.GetFieldName()
		if name != fieldIdentityDigest && name != fieldChecksum && name != fieldVector {
			continue
		}
		if seen[name] {
			return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "verified exact search returned duplicate field " + name}
		}
		seen[name] = true
		decoded, failure := verifiedNativeField(field, count, dimension)
		if failure != nil {
			return nil, failure
		}
		result.Fields = append(result.Fields, decoded)
	}
	return []milvusclient.ResultSet{result}, nil
}

func verifiedNativeField(field *schemapb.FieldData, count int64, dimension int) (column.Column, *verifiedSearchFailure) {
	name := field.GetFieldName()
	validData := field.GetValidData()
	if len(validData) != 0 && int64(len(validData)) != count {
		return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "verified exact search returned inconsistent validity lengths"}
	}
	for _, valid := range validData {
		if !valid {
			return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "verified exact search returned a null field " + name}
		}
	}
	switch name {
	case fieldIdentityDigest, fieldChecksum:
		if field.GetType() != schemapb.DataType_VarChar {
			return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "verified exact search returned invalid scalar field " + name}
		}
		values, valid := field.GetScalars().GetData().(*schemapb.ScalarField_StringData)
		if !valid || values == nil || values.StringData == nil || int64(len(values.StringData.GetData())) != count {
			return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "verified exact search returned inconsistent scalar lengths"}
		}
		return column.NewColumnVarChar(name, values.StringData.GetData()), nil
	case fieldVector:
		vectors := field.GetVectors()
		values, valid := vectors.GetData().(*schemapb.VectorField_FloatVector)
		if field.GetType() != schemapb.DataType_FloatVector || vectors.GetDim() != int64(dimension) || !valid || values == nil || values.FloatVector == nil || int64(len(values.FloatVector.GetData())) != count*int64(dimension) {
			return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "verified exact search returned invalid vector shape"}
		}
		flat := values.FloatVector.GetData()
		rows := make([][]float32, count)
		for index := range rows {
			begin := index * dimension
			end := begin + dimension
			rows[index] = flat[begin:end:end]
		}
		return column.NewColumnFloatVector(name, dimension, rows), nil
	}
	return nil, &verifiedSearchFailure{category: library.ErrVectorCorrupt, detail: "verified exact search returned an unexpected field " + name}
}

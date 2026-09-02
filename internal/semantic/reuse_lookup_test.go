package semantic

import (
	"context"
	"fmt"
	"net"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/config"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const reuseLookupTestDimension = 4096

type recordedReuseRequest struct {
	Filter       string
	OutputFields []string
	IDs          []string
}

type reuseLookupRow struct {
	id             string
	content        string
	contentHash    *string
	embeddingModel *string
}

type reuseLookupServer struct {
	milvuspb.UnimplementedMilvusServiceServer

	mutex         sync.Mutex
	rows          []reuseLookupRow
	requests      []recordedReuseRequest
	consistencies []commonpb.ConsistencyLevel
}

func (server *reuseLookupServer) Connect(
	context.Context,
	*milvuspb.ConnectRequest,
) (*milvuspb.ConnectResponse, error) {
	return &milvuspb.ConnectResponse{
		Status:     reuseLookupSuccessStatus(),
		Identifier: 1,
	}, nil
}

func (server *reuseLookupServer) DescribeCollection(
	_ context.Context,
	request *milvuspb.DescribeCollectionRequest,
) (*milvuspb.DescribeCollectionResponse, error) {
	return &milvuspb.DescribeCollectionResponse{
		Status:         reuseLookupSuccessStatus(),
		CollectionName: request.GetCollectionName(),
		Schema: &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
			{Name: idFieldName, DataType: schemapb.DataType_VarChar, IsPrimaryKey: true},
			{Name: contentFieldName, DataType: schemapb.DataType_VarChar},
			{Name: contentHashFieldName, DataType: schemapb.DataType_VarChar, Nullable: true},
			{Name: embeddingModelFieldName, DataType: schemapb.DataType_VarChar, Nullable: true},
			{
				Name:     denseVectorFieldName,
				DataType: schemapb.DataType_FloatVector,
				TypeParams: []*commonpb.KeyValuePair{{
					Key:   "dim",
					Value: strconv.Itoa(reuseLookupTestDimension),
				}},
			},
		}},
	}, nil
}

func (server *reuseLookupServer) Query(
	_ context.Context,
	request *milvuspb.QueryRequest,
) (*milvuspb.QueryResults, error) {
	ids := reuseLookupRequestIDs(request.GetExpr())
	server.mutex.Lock()
	server.requests = append(server.requests, recordedReuseRequest{
		Filter:       request.GetExpr(),
		OutputFields: slices.Clone(request.GetOutputFields()),
		IDs:          ids,
	})
	server.consistencies = append(server.consistencies, request.GetConsistencyLevel())
	rows := server.matchingRows(request.GetExpr(), ids)
	server.mutex.Unlock()

	if slices.Contains(request.GetOutputFields(), denseVectorFieldName) && len(ids) == 0 {
		return nil, status.Error(codes.InvalidArgument, "scalar candidate discovery requested vector")
	}
	if slices.Contains(request.GetOutputFields(), denseVectorFieldName) && len(ids) > 1 {
		return nil, status.Error(codes.InvalidArgument, "vector read requested multiple candidate IDs")
	}
	return reuseLookupQueryResults(request.GetOutputFields(), rows), nil
}

func (server *reuseLookupServer) matchingRows(filter string, ids []string) []reuseLookupRow {
	rows := make([]reuseLookupRow, 0, len(server.rows))
	for _, row := range server.rows {
		if slices.Contains(ids, row.id) {
			rows = append(rows, row)
		}
	}
	if len(ids) > 0 {
		return rows
	}
	if strings.Contains(filter, contentHashFieldName+" is null") {
		contents := reuseLookupFilterValues(filter, contentFieldName)
		for _, row := range server.rows {
			if row.contentHash == nil && slices.Contains(contents, row.content) {
				rows = append(rows, row)
			}
		}
		return rows
	}
	hashes := reuseLookupFilterValues(filter, contentHashFieldName)
	for _, row := range server.rows {
		if row.contentHash != nil && slices.Contains(hashes, *row.contentHash) {
			rows = append(rows, row)
		}
	}
	return rows
}

func (server *reuseLookupServer) snapshot() ([]recordedReuseRequest, []commonpb.ConsistencyLevel) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	return slices.Clone(server.requests), slices.Clone(server.consistencies)
}

func reuseLookupRequestIDs(filter string) []string {
	return reuseLookupFilterValues(filter, idFieldName)
}

func reuseLookupFilterValues(filter string, fieldName string) []string {
	start := strings.Index(filter, fieldName+" in [")
	if start < 0 {
		return nil
	}
	start += len(fieldName + " in [")
	end := strings.Index(filter[start:], "]")
	if end < 0 {
		return nil
	}
	var ids []string
	for _, quotedID := range strings.Split(filter[start:start+end], ",") {
		id, err := strconv.Unquote(strings.TrimSpace(quotedID))
		if err == nil {
			ids = append(ids, id)
		}
	}
	return ids
}

func reuseLookupQueryResults(outputFields []string, rows []reuseLookupRow) *milvuspb.QueryResults {
	fields := make([]*schemapb.FieldData, 0, len(outputFields))
	for _, outputField := range outputFields {
		switch outputField {
		case idFieldName:
			ids := make([]string, 0, len(rows))
			for _, row := range rows {
				ids = append(ids, row.id)
			}
			fields = append(fields, column.NewColumnVarChar(idFieldName, ids).FieldData())
		case contentFieldName:
			contents := make([]string, 0, len(rows))
			for _, row := range rows {
				contents = append(contents, row.content)
			}
			fields = append(fields, column.NewColumnVarChar(contentFieldName, contents).FieldData())
		case contentHashFieldName:
			hashes := make([]string, 0, len(rows))
			valid := make([]bool, 0, len(rows))
			for _, row := range rows {
				if row.contentHash == nil {
					valid = append(valid, false)
					continue
				}
				hashes = append(hashes, *row.contentHash)
				valid = append(valid, true)
			}
			field := column.NewColumnVarChar(contentHashFieldName, hashes).FieldData()
			field.ValidData = valid
			fields = append(fields, field)
		case embeddingModelFieldName:
			models := make([]string, 0, len(rows))
			valid := make([]bool, 0, len(rows))
			for _, row := range rows {
				if row.embeddingModel == nil {
					valid = append(valid, false)
					continue
				}
				models = append(models, *row.embeddingModel)
				valid = append(valid, true)
			}
			field := column.NewColumnVarChar(embeddingModelFieldName, models).FieldData()
			field.ValidData = valid
			fields = append(fields, field)
		case denseVectorFieldName:
			vectors := make([][]float32, 0, len(rows))
			for range rows {
				vectors = append(vectors, make([]float32, reuseLookupTestDimension))
			}
			fields = append(fields, column.NewColumnFloatVector(denseVectorFieldName, reuseLookupTestDimension, vectors).FieldData())
		}
	}
	return &milvuspb.QueryResults{
		Status:       reuseLookupSuccessStatus(),
		FieldsData:   fields,
		OutputFields: outputFields,
	}
}

func reuseLookupSuccessStatus() *commonpb.Status {
	return &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}
}

func newReuseLookupTestService(t *testing.T, rows []reuseLookupRow) (*Service, *reuseLookupServer) {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen for fake Milvus: %v", err)
	}
	server := &reuseLookupServer{rows: rows}
	grpcServer := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(grpcServer, server)
	go func() {
		_ = grpcServer.Serve(listener)
	}()
	t.Cleanup(grpcServer.Stop)

	client, err := milvusclient.New(context.Background(), &milvusclient.ClientConfig{
		Address: listener.Addr().String(),
	})
	if err != nil {
		t.Fatalf("construct Milvus client: %v", err)
	}
	t.Cleanup(func() {
		if closeErr := client.Close(context.Background()); closeErr != nil {
			t.Errorf("close Milvus client: %v", closeErr)
		}
	})
	return &Service{
		cfg: config.Config{
			EmbeddingDimension: reuseLookupTestDimension,
			EmbeddingModel:     "current-model",
		},
		milvus: client,
	}, server
}

func loadReuseLookupRows(t *testing.T, service *Service, contents []string) map[string][]float32 {
	t.Helper()
	contentsByHash := make(map[string]string, len(contents))
	for _, content := range contents {
		contentsByHash[contentHash(content)] = content
	}
	reuse := make(map[string][]float32)
	if err := service.loadCollectionReuseFromSource(
		context.Background(),
		"reuse_source",
		contentsByHash,
		reuse,
	); err != nil {
		t.Fatalf("load collection reuse: %v", err)
	}
	return reuse
}

func TestReuseScalarCandidateDiscoveryNeverRequestsVectors(t *testing.T) {
	rows := make([]reuseLookupRow, 0, 16384)
	for index := range 16384 {
		rows = append(rows, reuseLookupRow{
			id:      fmt.Sprintf("legacy-%05d", index),
			content: "duplicate legacy content",
		})
	}
	service, server := newReuseLookupTestService(t, rows)

	_ = loadReuseLookupRows(t, service, []string{"duplicate legacy content"})

	requests, _ := server.snapshot()
	for _, request := range requests {
		if len(request.IDs) == 0 && slices.Contains(request.OutputFields, denseVectorFieldName) {
			t.Fatalf("scalar candidate request selected vector: %+v", request)
		}
	}
}

func TestReuseModernCandidateDiscoverySelectsOneCompatibleID(t *testing.T) {
	for _, testCase := range []struct {
		name           string
		embeddingModel *string
		wantReuse      bool
	}{
		{name: "null", wantReuse: true},
		{name: "empty", embeddingModel: ptr(""), wantReuse: true},
		{name: "equal", embeddingModel: ptr("current-model"), wantReuse: true},
		{name: "unequal", embeddingModel: ptr("different-model"), wantReuse: false},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			content := "modern " + testCase.name
			service, server := newReuseLookupTestService(t, []reuseLookupRow{{
				id:             testCase.name,
				content:        content,
				contentHash:    ptr(contentHash(content)),
				embeddingModel: testCase.embeddingModel,
			}})

			reuse := loadReuseLookupRows(t, service, []string{content})
			if testCase.wantReuse && len(reuse) != 1 {
				t.Fatalf("reuse vector count = %d, want 1", len(reuse))
			}
			if !testCase.wantReuse && len(reuse) != 0 {
				t.Fatalf("reuse vector count = %d, want 0", len(reuse))
			}
			requests, _ := server.snapshot()
			ids := reuseLookupGetIDs(requests)
			if testCase.wantReuse && !slices.Equal(ids, []string{testCase.name}) {
				t.Fatalf("vector Get ids = %v, want %q", ids, testCase.name)
			}
			if !testCase.wantReuse && len(ids) != 0 {
				t.Fatalf("vector Get ids = %v, want none", ids)
			}
		})
	}
}

func TestReuseLegacyCandidateDiscoveryQueriesOneContentAtATime(t *testing.T) {
	service, server := newReuseLookupTestService(t, []reuseLookupRow{
		{id: "legacy-first", content: "legacy first"},
		{id: "legacy-second", content: "legacy second"},
	})

	_ = loadReuseLookupRows(t, service, []string{"legacy first", "legacy second"})

	requests, _ := server.snapshot()
	legacyFilters := make([]string, 0)
	for _, request := range requests {
		if strings.Contains(request.Filter, contentHashFieldName+" is null") {
			legacyFilters = append(legacyFilters, request.Filter)
		}
	}
	if len(legacyFilters) != 2 {
		t.Fatalf("legacy candidate queries = %d, want 2", len(legacyFilters))
	}
	for _, filter := range legacyFilters {
		if strings.Count(filter, "\"") != 2 {
			t.Fatalf("legacy candidate filter = %q, want one content", filter)
		}
	}
}

func TestReuseCandidateDiscoveryRejectsKnownUnequalModel(t *testing.T) {
	unequalModel := "different-model"
	service, server := newReuseLookupTestService(t, []reuseLookupRow{{
		id:             "unequal",
		content:        "known unequal content",
		contentHash:    ptr(contentHash("known unequal content")),
		embeddingModel: &unequalModel,
	}})

	reuse := loadReuseLookupRows(t, service, []string{"known unequal content"})

	if len(reuse) != 0 {
		t.Fatalf("reuse vector count = %d, want 0", len(reuse))
	}
	requests, _ := server.snapshot()
	if ids := reuseLookupGetIDs(requests); len(ids) != 0 {
		t.Fatalf("vector Get ids = %v, included known unequal model", ids)
	}
}

func reuseLookupGetIDs(requests []recordedReuseRequest) []string {
	var ids []string
	for _, request := range requests {
		if len(request.IDs) > 0 && slices.Contains(request.OutputFields, denseVectorFieldName) {
			ids = append(ids, request.IDs...)
		}
	}
	return ids
}

func ptr(value string) *string {
	return &value
}

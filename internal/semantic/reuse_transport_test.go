package semantic

import (
	"context"
	"fmt"
	"net"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/embedding"
	"goodkind.io/lm-semantic-search/internal/model"
	"google.golang.org/grpc"
)

const (
	transportReuseDimension    = 4096
	transportReuseMaxSendBytes = 64 * 1024
)

type transportReuseRow struct {
	id      string
	content string
	vector  []float32
}

type transportReuseRequest struct {
	ids          []string
	outputFields []string
}

type transportReuseServer struct {
	milvuspb.UnimplementedMilvusServiceServer

	mutex         sync.Mutex
	rows          []transportReuseRow
	requests      []transportReuseRequest
	mutationCalls atomic.Int32
}

type transportReuseEmbedder struct {
	calls atomic.Int32
}

func (embedder *transportReuseEmbedder) Embed(
	context.Context,
	string,
) ([]float32, error) {
	embedder.calls.Add(1)
	return make([]float32, transportReuseDimension), nil
}

func (embedder *transportReuseEmbedder) EmbedBatch(
	context.Context,
	[]string,
) (embedding.BatchResult, error) {
	embedder.calls.Add(1)
	return embedding.BatchResult{}, nil
}

func (*transportReuseEmbedder) ProviderName() model.EmbeddingProvider {
	return "transport-reuse-test"
}

func (*transportReuseEmbedder) Health(context.Context) error {
	return nil
}

func (server *transportReuseServer) Connect(
	context.Context,
	*milvuspb.ConnectRequest,
) (*milvuspb.ConnectResponse, error) {
	return &milvuspb.ConnectResponse{
		Status:     transportReuseSuccessStatus(),
		Identifier: 1,
	}, nil
}

func (server *transportReuseServer) HasCollection(
	_ context.Context,
	request *milvuspb.HasCollectionRequest,
) (*milvuspb.BoolResponse, error) {
	return &milvuspb.BoolResponse{
		Status: transportReuseSuccessStatus(),
		Value:  request.GetCollectionName() == "reuse_source",
	}, nil
}

func (server *transportReuseServer) DescribeCollection(
	_ context.Context,
	request *milvuspb.DescribeCollectionRequest,
) (*milvuspb.DescribeCollectionResponse, error) {
	if request.GetCollectionName() != "reuse_source" {
		return &milvuspb.DescribeCollectionResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_CollectionNotExists,
				Reason:    "collection not found",
			},
		}, nil
	}
	return &milvuspb.DescribeCollectionResponse{
		Status:         transportReuseSuccessStatus(),
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
					Value: strconv.Itoa(transportReuseDimension),
				}},
			},
		}},
	}, nil
}

func (server *transportReuseServer) Query(
	_ context.Context,
	request *milvuspb.QueryRequest,
) (*milvuspb.QueryResults, error) {
	ids := transportReuseIDs(request.GetExpr(), server.rows)
	server.mutex.Lock()
	server.requests = append(server.requests, transportReuseRequest{
		ids:          slices.Clone(ids),
		outputFields: slices.Clone(request.GetOutputFields()),
	})
	server.mutex.Unlock()

	rows := make([]transportReuseRow, 0, len(server.rows))
	switch {
	case len(ids) > 0:
		for _, row := range server.rows {
			if slices.Contains(ids, row.id) {
				rows = append(rows, row)
			}
		}
	case strings.Contains(request.GetExpr(), contentHashFieldName+" is null"):
		rows = append(rows, server.rows...)
	}
	return transportReuseQueryResults(request.GetOutputFields(), rows), nil
}

func (server *transportReuseServer) snapshot() []transportReuseRequest {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	requests := make([]transportReuseRequest, len(server.requests))
	for index, request := range server.requests {
		requests[index] = transportReuseRequest{
			ids:          slices.Clone(request.ids),
			outputFields: slices.Clone(request.outputFields),
		}
	}
	return requests
}

func transportReuseIDs(filter string, rows []transportReuseRow) []string {
	ids := make([]string, 0)
	for _, row := range rows {
		if strings.Contains(filter, `"`+row.id+`"`) {
			ids = append(ids, row.id)
		}
	}
	return ids
}

func transportReuseQueryResults(
	outputFields []string,
	rows []transportReuseRow,
) *milvuspb.QueryResults {
	fields := make([]*schemapb.FieldData, 0, len(outputFields))
	for _, outputField := range outputFields {
		switch outputField {
		case idFieldName:
			values := make([]string, 0, len(rows))
			for _, row := range rows {
				values = append(values, row.id)
			}
			fields = append(fields, column.NewColumnVarChar(idFieldName, values).FieldData())
		case contentFieldName:
			values := make([]string, 0, len(rows))
			for _, row := range rows {
				values = append(values, row.content)
			}
			fields = append(fields, column.NewColumnVarChar(contentFieldName, values).FieldData())
		case contentHashFieldName:
			values := make([]string, len(rows))
			field := column.NewColumnVarChar(contentHashFieldName, values).FieldData()
			field.ValidData = make([]bool, len(rows))
			fields = append(fields, field)
		case embeddingModelFieldName:
			values := make([]string, len(rows))
			field := column.NewColumnVarChar(embeddingModelFieldName, values).FieldData()
			field.ValidData = make([]bool, len(rows))
			fields = append(fields, field)
		case denseVectorFieldName:
			vectors := make([][]float32, 0, len(rows))
			for _, row := range rows {
				vectors = append(vectors, row.vector)
			}
			fields = append(fields, column.NewColumnFloatVector(
				denseVectorFieldName,
				transportReuseDimension,
				vectors,
			).FieldData())
		}
	}
	return &milvuspb.QueryResults{
		Status:       transportReuseSuccessStatus(),
		FieldsData:   fields,
		OutputFields: slices.Clone(outputFields),
	}
}

func transportReuseSuccessStatus() *commonpb.Status {
	return &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}
}

func transportReuseMutationMethod(fullMethod string) bool {
	method := fullMethod[strings.LastIndex(fullMethod, "/")+1:]
	if strings.HasPrefix(method, "Alter") || strings.HasPrefix(method, "Create") ||
		strings.HasPrefix(method, "Drop") {
		return true
	}
	switch method {
	case "Delete", "Flush", "Import", "Insert", "LoadCollection", "ReleaseCollection",
		"RenameCollection", "TruncateCollection", "Upsert":
		return true
	default:
		return false
	}
}

func newTransportReuseTestService(
	t *testing.T,
	rows []transportReuseRow,
) (*Service, *transportReuseServer, *transportReuseEmbedder) {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen for bounded transport Milvus: %v", err)
	}
	server := &transportReuseServer{rows: rows}
	grpcServer := grpc.NewServer(
		grpc.MaxSendMsgSize(transportReuseMaxSendBytes),
		grpc.UnaryInterceptor(func(
			ctx context.Context,
			request interface{},
			info *grpc.UnaryServerInfo,
			handler grpc.UnaryHandler,
		) (interface{}, error) {
			if transportReuseMutationMethod(info.FullMethod) {
				server.mutationCalls.Add(1)
			}
			return handler(ctx, request)
		}),
	)
	milvuspb.RegisterMilvusServiceServer(grpcServer, server)
	go func() {
		_ = grpcServer.Serve(listener)
	}()
	t.Cleanup(grpcServer.Stop)

	client, err := milvusclient.New(context.Background(), &milvusclient.ClientConfig{
		Address: listener.Addr().String(),
	})
	if err != nil {
		t.Fatalf("construct bounded transport Milvus client: %v", err)
	}
	t.Cleanup(func() {
		if closeErr := client.Close(context.Background()); closeErr != nil {
			t.Errorf("close bounded transport Milvus client: %v", closeErr)
		}
	})

	embedder := &transportReuseEmbedder{}
	service := &Service{
		cfg: config.Config{
			EmbeddingDimension: transportReuseDimension,
			EmbeddingModel:     "transport-reuse-model",
		},
		milvus:   client,
		embedder: embedder,
	}
	service.available.Store(true)
	reuseIdentity := &reuseIdentityMigration{}
	reuseIdentity.once.Do(func() {})
	service.ensuredReuseIdentityColumns.Store("reuse_source", reuseIdentity)
	service.residency = newCollectionResidencyController(residencyControllerConfig{
		waitTimeout: time.Second,
		idleTimeout: time.Minute,
		loadCeiling: time.Second,
		load: func(context.Context, string) error {
			return nil
		},
		unload: func(context.Context, string) error {
			return nil
		},
	})
	t.Cleanup(func() {
		if closeErr := service.residency.Close(context.Background()); closeErr != nil {
			t.Errorf("close bounded transport residency controller: %v", closeErr)
		}
	})
	return service, server, embedder
}

func TestLoadReuseVectorsForContentsBoundsCandidateTransportResponse(t *testing.T) {
	content := "bounded transport duplicate legacy content"
	expectedVector := make([]float32, transportReuseDimension)
	expectedVector[0] = 1
	expectedVector[len(expectedVector)-1] = 2
	rows := make([]transportReuseRow, 0, 8)
	for index := range 8 {
		rows = append(rows, transportReuseRow{
			id:      fmt.Sprintf("legacy-%02d", index),
			content: content,
			vector:  expectedVector,
		})
	}
	service, server, embedder := newTransportReuseTestService(t, rows)

	reuse, err := service.LoadReuseVectorsForContents(
		context.Background(),
		"reuse_source",
		[]model.StoredChunk{{Content: content}},
	)
	if err != nil {
		t.Fatalf("load reuse through bounded transport: %v", err)
	}
	actualVector, found := reuse[ContentVectorKey(content)]
	if !found || !slices.Equal(actualVector, expectedVector) {
		t.Fatalf("reuse vector found=%t dimension=%d, want exact selected vector", found, len(actualVector))
	}

	requests := server.snapshot()
	selectedVectorRequests := 0
	for _, request := range requests {
		requestsVector := slices.Contains(request.outputFields, denseVectorFieldName)
		if len(request.ids) == 0 && requestsVector {
			t.Fatalf("candidate discovery requested vectors: %+v", request)
		}
		if len(request.ids) > 0 && requestsVector {
			selectedVectorRequests++
			if !slices.Equal(request.ids, []string{"legacy-00"}) {
				t.Fatalf("selected vector IDs = %v, want [legacy-00]", request.ids)
			}
		}
	}
	if selectedVectorRequests != 1 {
		t.Fatalf("selected vector requests = %d, want 1", selectedVectorRequests)
	}
	if embedder.calls.Load() != 0 {
		t.Fatalf("embedding calls = %d, want 0", embedder.calls.Load())
	}
	if server.mutationCalls.Load() != 0 {
		t.Fatalf("mutation calls = %d, want 0", server.mutationCalls.Load())
	}
}

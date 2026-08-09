package semantic_test

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/client/v2/entity"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/grpc"
)

type splitPartLifecycleServer struct {
	milvuspb.UnimplementedMilvusServiceServer

	mu               sync.Mutex
	collectionExists bool
	splitPartExists  bool
	splitPartAdds    int
}

func (server *splitPartLifecycleServer) setCollection(exists bool, hasSplitPart bool) {
	server.mu.Lock()
	defer server.mu.Unlock()
	server.collectionExists = exists
	server.splitPartExists = hasSplitPart
}

func (server *splitPartLifecycleServer) addedSplitPart() bool {
	server.mu.Lock()
	defer server.mu.Unlock()
	return server.splitPartAdds == 1 && server.splitPartExists
}

func (server *splitPartLifecycleServer) Connect(
	context.Context,
	*milvuspb.ConnectRequest,
) (*milvuspb.ConnectResponse, error) {
	return &milvuspb.ConnectResponse{
		Status:     successStatus(),
		Identifier: 1,
	}, nil
}

func (server *splitPartLifecycleServer) ShowCollections(
	context.Context,
	*milvuspb.ShowCollectionsRequest,
) (*milvuspb.ShowCollectionsResponse, error) {
	return &milvuspb.ShowCollectionsResponse{Status: successStatus()}, nil
}

func (server *splitPartLifecycleServer) DescribeCollection(
	_ context.Context,
	request *milvuspb.DescribeCollectionRequest,
) (*milvuspb.DescribeCollectionResponse, error) {
	server.mu.Lock()
	defer server.mu.Unlock()
	if !server.collectionExists {
		return &milvuspb.DescribeCollectionResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_CollectionNotExists,
				Reason:    "collection not found",
			},
		}, nil
	}
	return &milvuspb.DescribeCollectionResponse{
		Status:         successStatus(),
		CollectionName: request.GetCollectionName(),
		Schema:         splitPartTestSchema(request.GetCollectionName(), server.splitPartExists),
	}, nil
}

func (server *splitPartLifecycleServer) AddCollectionField(
	_ context.Context,
	_ *milvuspb.AddCollectionFieldRequest,
) (*commonpb.Status, error) {
	server.mu.Lock()
	defer server.mu.Unlock()
	server.splitPartExists = true
	server.splitPartAdds++
	return successStatus(), nil
}

func (server *splitPartLifecycleServer) Search(
	_ context.Context,
	_ *milvuspb.SearchRequest,
) (*milvuspb.SearchResults, error) {
	server.mu.Lock()
	defer server.mu.Unlock()
	if !server.splitPartExists {
		return &milvuspb.SearchResults{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
				Reason:    "field splitPart does not exist",
			},
		}, nil
	}
	return &milvuspb.SearchResults{
		Status: successStatus(),
		Results: &schemapb.SearchResultData{
			NumQueries: 1,
			TopK:       10,
			Topks:      []int64{0},
			Ids: &schemapb.IDs{
				IdField: &schemapb.IDs_StrId{
					StrId: &schemapb.StringArray{Data: []string{}},
				},
			},
		},
	}, nil
}

func TestSearchRemigratesCollectionAfterInspectObservesAbsence(t *testing.T) {
	t.Parallel()

	embeddingServer := httptest.NewServer(http.HandlerFunc(
		func(writer http.ResponseWriter, _ *http.Request) {
			writer.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(writer).Encode(map[string][]map[string][]float64{
				"data": {{"embedding": {0}}},
			}); err != nil {
				t.Errorf("encode embedding response: %v", err)
			}
		},
	))
	t.Cleanup(embeddingServer.Close)

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen for fake Milvus: %v", err)
	}
	fakeMilvus := &splitPartLifecycleServer{
		collectionExists: true,
		splitPartExists:  true,
	}
	grpcServer := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(grpcServer, fakeMilvus)
	t.Cleanup(grpcServer.Stop)
	go func() {
		if serveErr := grpcServer.Serve(listener); serveErr != nil {
			t.Errorf("serve fake Milvus: %v", serveErr)
		}
	}()

	codebasePath := t.TempDir()
	service, err := semantic.NewService(context.Background(), config.Config{
		EmbeddingProvider:  "OpenAI",
		EmbeddingModel:     "test-model",
		EmbeddingDimension: 1,
		OpenAIAPIKey:       "test-key",
		OpenAIBaseURL:      embeddingServer.URL,
		MilvusAddress:      listener.Addr().String(),
	})
	if err != nil {
		t.Fatalf("NewService returned error: %v", err)
	}
	t.Cleanup(func() {
		if closeErr := service.Close(context.Background()); closeErr != nil {
			t.Errorf("Close returned error: %v", closeErr)
		}
	})

	if _, err := service.Search(context.Background(), codebasePath, "first", 1, nil, ""); err != nil {
		t.Fatalf("initial Search returned error: %v", err)
	}

	collectionName := service.CollectionName(codebasePath)
	fakeMilvus.setCollection(false, false)
	facts, err := service.InspectCollection(context.Background(), collectionName)
	if err != nil {
		t.Fatalf("InspectCollection returned error: %v", err)
	}
	if facts.Exists {
		t.Fatal("InspectCollection reported a dropped collection as present")
	}

	fakeMilvus.setCollection(true, false)
	if _, err := service.Search(context.Background(), codebasePath, "second", 1, nil, ""); err != nil {
		t.Fatalf("Search after external recreation returned error: %v", err)
	}
	if !fakeMilvus.addedSplitPart() {
		t.Fatal("Search did not restore splitPart on the recreated collection")
	}
}

func successStatus() *commonpb.Status {
	return &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}
}

func splitPartTestSchema(collectionName string, withSplitPart bool) *schemapb.CollectionSchema {
	schema := entity.NewSchema().
		WithName(collectionName).
		WithField(entity.NewField().
			WithName("id").
			WithDataType(entity.FieldTypeVarChar).
			WithMaxLength(512).
			WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName("content").WithDataType(entity.FieldTypeVarChar).WithMaxLength(65535)).
		WithField(entity.NewField().WithName("relativePath").WithDataType(entity.FieldTypeVarChar).WithMaxLength(1024)).
		WithField(entity.NewField().WithName("startLine").WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName("endLine").WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName("fileExtension").WithDataType(entity.FieldTypeVarChar).WithMaxLength(32)).
		WithField(entity.NewField().WithName("metadata").WithDataType(entity.FieldTypeVarChar).WithMaxLength(65535))
	if withSplitPart {
		schema = schema.WithField(entity.NewField().
			WithName("splitPart").
			WithDataType(entity.FieldTypeInt64).
			WithNullable(true))
	}
	return schema.
		WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithDim(1)).
		ProtoMessage()
}

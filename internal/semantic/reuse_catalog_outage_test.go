package semantic_test

import (
	"context"
	"errors"
	"net"
	"testing"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type reuseCatalogOutageServer struct {
	milvuspb.UnimplementedMilvusServiceServer
}

func (reuseCatalogOutageServer) Connect(
	context.Context,
	*milvuspb.ConnectRequest,
) (*milvuspb.ConnectResponse, error) {
	return &milvuspb.ConnectResponse{
		Status:     &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
		Identifier: 1,
	}, nil
}

func (reuseCatalogOutageServer) ShowCollections(
	context.Context,
	*milvuspb.ShowCollectionsRequest,
) (*milvuspb.ShowCollectionsResponse, error) {
	return &milvuspb.ShowCollectionsResponse{
		Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
	}, nil
}

func (reuseCatalogOutageServer) DescribeCollection(
	_ context.Context,
	request *milvuspb.DescribeCollectionRequest,
) (*milvuspb.DescribeCollectionResponse, error) {
	return &milvuspb.DescribeCollectionResponse{
		Status:         &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
		CollectionName: request.GetCollectionName(),
		Schema: &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
			{
				Name:         "catalogKey",
				DataType:     schemapb.DataType_VarChar,
				IsPrimaryKey: true,
			},
		}},
	}, nil
}

func (reuseCatalogOutageServer) Query(
	context.Context,
	*milvuspb.QueryRequest,
) (*milvuspb.QueryResults, error) {
	return nil, status.Error(codes.DeadlineExceeded, "acceptance Milvus outage")
}

func TestLoadReuseCatalogRowKeysClassifiesMilvusTransportOutage(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen for fake Milvus: %v", err)
	}
	grpcServer := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(grpcServer, reuseCatalogOutageServer{})
	t.Cleanup(grpcServer.Stop)
	go func() {
		if serveErr := grpcServer.Serve(listener); serveErr != nil {
			t.Errorf("serve fake Milvus: %v", serveErr)
		}
	}()

	service, err := semantic.NewService(context.Background(), config.Config{
		EmbeddingProvider:  "OpenAI",
		EmbeddingModel:     "test-model",
		EmbeddingDimension: 16,
		OpenAIAPIKey:       "test-key",
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

	err = semantic.LoadReuseCatalogRowKeysForTest(context.Background(), service, []string{"row-key"}, 16)
	var adapterErr *adapterr.AdapterError
	if !errors.As(err, &adapterErr) {
		t.Fatalf("loadReuseCatalogRowKeys error type = %T, want AdapterError: %v", err, err)
	}
	if adapterErr.Class != adapterr.ClassMilvusUnavailable {
		t.Fatalf("loadReuseCatalogRowKeys class = %q, want %q", adapterErr.Class, adapterr.ClassMilvusUnavailable)
	}
	if got := status.Code(err); got != codes.DeadlineExceeded {
		t.Fatalf("loadReuseCatalogRowKeys cause code = %s, want %s", got, codes.DeadlineExceeded)
	}
}

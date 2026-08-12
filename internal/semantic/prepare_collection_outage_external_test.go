package semantic_test

import (
	"context"
	"errors"
	"net"
	"testing"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type prepareCollectionOutageServer struct {
	milvuspb.UnimplementedMilvusServiceServer
}

func (prepareCollectionOutageServer) Connect(
	context.Context,
	*milvuspb.ConnectRequest,
) (*milvuspb.ConnectResponse, error) {
	return &milvuspb.ConnectResponse{
		Status:     &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
		Identifier: 1,
	}, nil
}

func (prepareCollectionOutageServer) ShowCollections(
	context.Context,
	*milvuspb.ShowCollectionsRequest,
) (*milvuspb.ShowCollectionsResponse, error) {
	return &milvuspb.ShowCollectionsResponse{
		Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
	}, nil
}

func (prepareCollectionOutageServer) DescribeCollection(
	context.Context,
	*milvuspb.DescribeCollectionRequest,
) (*milvuspb.DescribeCollectionResponse, error) {
	return nil, status.Error(codes.DeadlineExceeded, "acceptance Milvus outage")
}

func TestPrepareCollectionClassifiesMilvusTransportOutage(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen for fake Milvus: %v", err)
	}
	grpcServer := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(grpcServer, prepareCollectionOutageServer{})
	t.Cleanup(grpcServer.Stop)
	go func() {
		if serveErr := grpcServer.Serve(listener); serveErr != nil {
			t.Errorf("serve fake Milvus: %v", serveErr)
		}
	}()

	service, err := semantic.NewService(context.Background(), config.Config{
		EmbeddingProvider: "OpenAI",
		EmbeddingModel:    "test-model",
		OpenAIAPIKey:      "test-key",
		MilvusAddress:     listener.Addr().String(),
	})
	if err != nil {
		t.Fatalf("NewService returned error: %v", err)
	}
	t.Cleanup(func() {
		if closeErr := service.Close(context.Background()); closeErr != nil {
			t.Errorf("Close returned error: %v", closeErr)
		}
	})

	err = service.PrepareCollection(context.Background(), "hybrid_code_chunks_acceptance")
	var adapterErr *adapterr.AdapterError
	if !errors.As(err, &adapterErr) {
		t.Fatalf("PrepareCollection error type = %T, want AdapterError: %v", err, err)
	}
	if adapterErr.Class != adapterr.ClassMilvusUnavailable {
		t.Fatalf("PrepareCollection class = %q, want %q", adapterErr.Class, adapterr.ClassMilvusUnavailable)
	}
}

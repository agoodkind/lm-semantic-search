package semantic_test

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"os"
	"testing"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/grpc"
)

type reconnectMilvusServer struct {
	milvuspb.UnimplementedMilvusServiceServer
}

func TestMain(testMain *testing.M) {
	listener, err := (&net.ListenConfig{}).Listen(
		context.Background(),
		"tcp",
		"127.0.0.1:0",
	)
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "start promotion test Milvus server: %v\n", err)
		os.Exit(1)
	}
	grpcServer := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(grpcServer, semantic.PromotionRecoveryServerForTest())
	semantic.SetPromotionRecoveryServerAddressForTest(listener.Addr().String())
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.Error(
					"promotion test Milvus server panicked",
					"error",
					fmt.Errorf("%v", recovered),
				)
			}
		}()
		_ = grpcServer.Serve(listener)
	}()
	exitCode := testMain.Run()
	grpcServer.Stop()
	os.Exit(exitCode)
}

func (reconnectMilvusServer) Connect(
	context.Context,
	*milvuspb.ConnectRequest,
) (*milvuspb.ConnectResponse, error) {
	return &milvuspb.ConnectResponse{
		Status:     &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
		Identifier: 1,
	}, nil
}

func (reconnectMilvusServer) ShowCollections(
	context.Context,
	*milvuspb.ShowCollectionsRequest,
) (*milvuspb.ShowCollectionsResponse, error) {
	return &milvuspb.ShowCollectionsResponse{
		Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
	}, nil
}

// This test lives in the external semantic_test package on purpose: it needs a
// fake Milvus built from google.golang.org/grpc, and importing grpc inside the
// production package would make the grpc-handler lint heuristic treat every
// *Service method as a gRPC handler. Startup recovery requires the fake to list
// collections before the reconnector can publish it.
func TestReconnectMakesServiceAvailableAgainstFakeMilvus(t *testing.T) {
	restoreTimeout := semantic.SetBootDialTimeoutForTest(20 * time.Millisecond)
	restoreSleep := semantic.SetReconnectSleepForTest(func(ctx context.Context, _ time.Duration) bool {
		timer := time.NewTimer(time.Millisecond)
		defer timer.Stop()
		select {
		case <-ctx.Done():
			return false
		case <-timer.C:
			return true
		}
	})
	restoreJitter := semantic.SetReconnectJitterForTest(func(time.Duration) time.Duration {
		return time.Millisecond
	})
	t.Cleanup(func() {
		restoreTimeout()
		restoreSleep()
		restoreJitter()
	})

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen returned error: %v", err)
	}
	address := listener.Addr().String()
	if err := listener.Close(); err != nil {
		t.Fatalf("Close listener returned error: %v", err)
	}

	cfg := config.Config{
		EmbeddingProvider: "OpenAI",
		EmbeddingModel:    "text-embedding-3-small",
		OpenAIAPIKey:      "test-key",
		MilvusAddress:     address,
		HybridMode:        true,
	}
	service, err := semantic.NewService(context.Background(), cfg)
	if err != nil {
		t.Fatalf("NewService returned error: %v", err)
	}
	t.Cleanup(func() {
		if closeErr := service.Close(context.Background()); closeErr != nil {
			t.Fatalf("Close returned error: %v", closeErr)
		}
	})
	if service.Available() {
		t.Fatal("Available() = true before fake Milvus starts")
	}

	serverListener, err := net.Listen("tcp", address)
	if err != nil {
		t.Fatalf("Listen returned error: %v", err)
	}
	grpcServer := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(grpcServer, reconnectMilvusServer{})
	t.Cleanup(func() {
		grpcServer.Stop()
	})
	go func() {
		_ = grpcServer.Serve(serverListener)
	}()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if service.Available() && !service.Degraded() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("service did not become available after the fake Milvus started")
}

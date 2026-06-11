package semantic_test

import (
	"context"
	"net"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/grpc"
)

// This test lives in the external semantic_test package on purpose: it needs a
// fake Milvus built from google.golang.org/grpc, and importing grpc inside the
// production package would make the grpc-handler lint heuristic treat every
// *Service method as a gRPC handler. The SDK accepts the empty server's
// Unimplemented Connect response as a successful connection, which is all the
// reconnector needs to flip Available.
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

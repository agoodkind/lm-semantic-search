package semantic_test

import (
	"context"
	"net"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

const (
	defaultTestDatabase  = "default"
	sandboxTestDatabase  = "live_sandbox"
	operatorCollection   = "operator_collection"
	temporaryCollection  = "temporary_collection"
	databaseTestDeadline = 5 * time.Second
)

type databaseIsolationCall struct {
	database       string
	collectionName string
}

type databaseIsolationServer struct {
	milvuspb.UnimplementedMilvusServiceServer

	mutex       sync.Mutex
	collections map[string][]string
	showCalls   []string
	loadCalls   []databaseIsolationCall
}

func (server *databaseIsolationServer) ShowCollections(
	ctx context.Context,
	_ *milvuspb.ShowCollectionsRequest,
) (*milvuspb.ShowCollectionsResponse, error) {
	databaseName := databaseNameFromContext(ctx)
	server.mutex.Lock()
	server.showCalls = append(server.showCalls, databaseName)
	collectionNames := slices.Clone(server.collections[databaseName])
	server.mutex.Unlock()
	return &milvuspb.ShowCollectionsResponse{
		Status:          databaseIsolationSuccessStatus(),
		CollectionNames: collectionNames,
	}, nil
}

func (server *databaseIsolationServer) GetLoadState(
	ctx context.Context,
	request *milvuspb.GetLoadStateRequest,
) (*milvuspb.GetLoadStateResponse, error) {
	server.mutex.Lock()
	server.loadCalls = append(server.loadCalls, databaseIsolationCall{
		database:       databaseNameFromContext(ctx),
		collectionName: request.GetCollectionName(),
	})
	server.mutex.Unlock()
	return &milvuspb.GetLoadStateResponse{
		Status: databaseIsolationSuccessStatus(),
		State:  commonpb.LoadState_LoadStateLoaded,
	}, nil
}

func databaseNameFromContext(ctx context.Context) string {
	values, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return ""
	}
	databaseNames := values.Get("dbname")
	if len(databaseNames) == 0 {
		return ""
	}
	return databaseNames[0]
}

func (server *databaseIsolationServer) snapshot() ([]string, []databaseIsolationCall) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	return slices.Clone(server.showCalls), slices.Clone(server.loadCalls)
}

func databaseIsolationSuccessStatus() *commonpb.Status {
	return &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}
}

func TestServiceReconcilesOnlyConfiguredMilvusDatabase(t *testing.T) {
	server := &databaseIsolationServer{collections: map[string][]string{
		defaultTestDatabase: {operatorCollection},
		sandboxTestDatabase: {temporaryCollection},
	}}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen for fake Milvus: %v", err)
	}
	grpcServer := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(grpcServer, server)
	go func() {
		_ = grpcServer.Serve(listener)
	}()
	t.Cleanup(grpcServer.Stop)

	service, err := semantic.NewService(context.Background(), config.Config{
		EmbeddingProvider: "OpenAI",
		EmbeddingModel:    "test-model",
		OpenAIAPIKey:      "test-key",
		OpenAIBaseURL:     "http://127.0.0.1:1/v1",
		MilvusAddress:     listener.Addr().String(),
		MilvusDatabase:    sandboxTestDatabase,
		HybridMode:        true,
	})
	if err != nil {
		t.Fatalf("construct semantic service: %v", err)
	}
	t.Cleanup(func() {
		if closeErr := service.Close(context.Background()); closeErr != nil {
			t.Errorf("close semantic service: %v", closeErr)
		}
	})

	deadline := time.Now().Add(databaseTestDeadline)
	for {
		showCalls, loadCalls := server.snapshot()
		if len(showCalls) >= 2 && len(loadCalls) >= 1 {
			for _, databaseName := range showCalls {
				if databaseName != sandboxTestDatabase {
					t.Fatalf("ShowCollections database = %q, want %q", databaseName, sandboxTestDatabase)
				}
			}
			for _, call := range loadCalls {
				if call.database != sandboxTestDatabase || call.collectionName != temporaryCollection {
					t.Fatalf("GetLoadState call = %+v, want sandbox temporary collection", call)
				}
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("reconciliation calls = show %v load %v", showCalls, loadCalls)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

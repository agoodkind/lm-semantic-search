//go:build live

package live

import (
	"context"
	"net"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

const cleanupTestDatabase = "live_sandbox"

type harnessCleanupServer struct {
	milvuspb.UnimplementedMilvusServiceServer

	mutex                             sync.Mutex
	databaseExists                    bool
	transientSandboxInventoryFailures int
	collections                       map[string][]string
	collectionProperties              map[string]string
}

func (server *harnessCleanupServer) Connect(
	context.Context,
	*milvuspb.ConnectRequest,
) (*milvuspb.ConnectResponse, error) {
	return &milvuspb.ConnectResponse{Status: harnessCleanupSuccess(), Identifier: 1}, nil
}

func (server *harnessCleanupServer) ShowCollections(
	ctx context.Context,
	request *milvuspb.ShowCollectionsRequest,
) (*milvuspb.ShowCollectionsResponse, error) {
	databaseName := request.GetDbName()
	if databaseName == "" {
		databaseName = harnessCleanupDatabase(ctx)
	}
	server.mutex.Lock()
	defer server.mutex.Unlock()
	if databaseName == cleanupTestDatabase &&
		server.transientSandboxInventoryFailures > 0 {
		server.transientSandboxInventoryFailures--
		return nil, status.Error(codes.Unavailable, "transient inventory failure")
	}
	return &milvuspb.ShowCollectionsResponse{
		Status:          harnessCleanupSuccess(),
		CollectionNames: slices.Clone(server.collections[databaseName]),
	}, nil
}

func (server *harnessCleanupServer) DropCollection(
	ctx context.Context,
	request *milvuspb.DropCollectionRequest,
) (*commonpb.Status, error) {
	databaseName := request.GetDbName()
	if databaseName == "" {
		databaseName = harnessCleanupDatabase(ctx)
	}
	server.mutex.Lock()
	server.collections[databaseName] = slices.DeleteFunc(
		server.collections[databaseName],
		func(collectionName string) bool {
			return collectionName == request.GetCollectionName()
		},
	)
	server.mutex.Unlock()
	return harnessCleanupSuccess(), nil
}

func (server *harnessCleanupServer) DropDatabase(
	_ context.Context,
	request *milvuspb.DropDatabaseRequest,
) (*commonpb.Status, error) {
	server.mutex.Lock()
	if request.GetDbName() == cleanupTestDatabase {
		server.databaseExists = false
	}
	server.mutex.Unlock()
	return harnessCleanupSuccess(), nil
}

func (server *harnessCleanupServer) ListDatabases(
	context.Context,
	*milvuspb.ListDatabasesRequest,
) (*milvuspb.ListDatabasesResponse, error) {
	server.mutex.Lock()
	databaseNames := []string{defaultMilvusDatabase}
	if server.databaseExists {
		databaseNames = append(databaseNames, cleanupTestDatabase)
	}
	server.mutex.Unlock()
	return &milvuspb.ListDatabasesResponse{
		Status:  harnessCleanupSuccess(),
		DbNames: databaseNames,
	}, nil
}

func (server *harnessCleanupServer) GetLoadState(
	context.Context,
	*milvuspb.GetLoadStateRequest,
) (*milvuspb.GetLoadStateResponse, error) {
	return &milvuspb.GetLoadStateResponse{
		Status: harnessCleanupSuccess(),
		State:  commonpb.LoadState_LoadStateLoaded,
	}, nil
}

func (server *harnessCleanupServer) DescribeCollection(
	_ context.Context,
	request *milvuspb.DescribeCollectionRequest,
) (*milvuspb.DescribeCollectionResponse, error) {
	schema := entity.NewSchema().
		WithName(request.GetCollectionName()).
		WithField(entity.NewField().
			WithName("id").
			WithDataType(entity.FieldTypeVarChar).
			WithIsPrimaryKey(true))
	server.mutex.Lock()
	properties := make([]*commonpb.KeyValuePair, 0, len(server.collectionProperties))
	for key, value := range server.collectionProperties {
		properties = append(properties, &commonpb.KeyValuePair{Key: key, Value: value})
	}
	server.mutex.Unlock()
	return &milvuspb.DescribeCollectionResponse{
		Status:         harnessCleanupSuccess(),
		CollectionName: request.GetCollectionName(),
		Schema:         schema.ProtoMessage(),
		Properties:     properties,
	}, nil
}

func (server *harnessCleanupServer) DescribeIndex(
	context.Context,
	*milvuspb.DescribeIndexRequest,
) (*milvuspb.DescribeIndexResponse, error) {
	return &milvuspb.DescribeIndexResponse{Status: harnessCleanupSuccess()}, nil
}

func TestCleanupMilvusDropsDatabaseAfterTransientInventoryFailure(t *testing.T) {
	server := &harnessCleanupServer{
		databaseExists:                    true,
		transientSandboxInventoryFailures: 1,
		collections: map[string][]string{
			defaultMilvusDatabase: {},
			cleanupTestDatabase:   {"temporary_collection"},
		},
		collectionProperties: nil,
	}
	address := startHarnessCleanupServer(t, server)
	h := &harness{
		t:               t,
		operatorMilvus:  newHarnessCleanupClient(t, address, ""),
		milvus:          newHarnessCleanupClient(t, address, cleanupTestDatabase),
		databaseName:    cleanupTestDatabase,
		beforeDatabases: []string{defaultMilvusDatabase},
		operatorBefore:  milvusInventory{},
		sandboxBefore:   milvusInventory{},
		temporaryNames:  map[string]struct{}{},
		callRecorder:    &milvusCallRecorder{},
	}

	cleanupErrors := h.cleanupMilvus()
	if len(cleanupErrors) == 0 {
		t.Fatal("cleanup returned no transient inventory error")
	}
	server.mutex.Lock()
	databaseExists := server.databaseExists
	remainingCollections := slices.Clone(server.collections[cleanupTestDatabase])
	server.mutex.Unlock()
	if databaseExists {
		t.Fatal("temporary database still exists after transient inventory failure")
	}
	if len(remainingCollections) != 0 {
		t.Fatalf("temporary collections remain after inventory failure: %v", remainingCollections)
	}
}

func TestReadMilvusInventoryIncludesCollectionProperties(t *testing.T) {
	server := &harnessCleanupServer{
		databaseExists: true,
		collections: map[string][]string{
			defaultMilvusDatabase: {"operator_collection"},
		},
		collectionProperties: map[string]string{
			"mmap.enabled":           "true",
			"collection.ttl.seconds": "60",
		},
	}
	address := startHarnessCleanupServer(t, server)
	client := newHarnessCleanupClient(t, address, "")
	t.Cleanup(func() { closeMilvusClient(client) })

	inventory, err := readMilvusInventory(client)
	if err != nil {
		t.Fatalf("read inventory: %v", err)
	}
	properties := inventory["operator_collection"]
	if properties["collection:mmap.enabled"] != "true" {
		t.Fatalf("collection mmap property = %q, want true", properties["collection:mmap.enabled"])
	}
	if properties["collection:collection.ttl.seconds"] != "60" {
		t.Fatalf(
			"collection ttl property = %q, want 60",
			properties["collection:collection.ttl.seconds"],
		)
	}
}

func startHarnessCleanupServer(t *testing.T, server *harnessCleanupServer) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen for cleanup fake Milvus: %v", err)
	}
	grpcServer := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(grpcServer, server)
	go func() { _ = grpcServer.Serve(listener) }()
	t.Cleanup(grpcServer.Stop)
	return listener.Addr().String()
}

func newHarnessCleanupClient(
	t *testing.T,
	address string,
	databaseName string,
) *milvusclient.Client {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	client, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: address,
		DBName:  databaseName,
	})
	if err != nil {
		t.Fatalf("connect cleanup fake Milvus: %v", err)
	}
	return client
}

func harnessCleanupDatabase(ctx context.Context) string {
	values, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return defaultMilvusDatabase
	}
	databaseNames := values.Get("dbname")
	if len(databaseNames) == 0 || databaseNames[0] == "" {
		return defaultMilvusDatabase
	}
	return databaseNames[0]
}

func harnessCleanupSuccess() *commonpb.Status {
	return &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}
}

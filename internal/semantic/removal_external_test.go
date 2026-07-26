package semantic_test

import (
	"context"
	"log/slog"
	"net"
	"sync"
	"testing"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	removalCompletedMessage = "semantic.removal_completed"
	removedRowsMessage      = "semantic.rows_removed"
)

type removalTestEvent struct {
	kind  string
	attrs map[string]string
}

type removalTestEvents struct {
	mu     sync.Mutex
	events []removalTestEvent
}

func (events *removalTestEvents) append(event removalTestEvent) {
	events.mu.Lock()
	defer events.mu.Unlock()
	events.events = append(events.events, event)
}

func (events *removalTestEvents) snapshot() []removalTestEvent {
	events.mu.Lock()
	defer events.mu.Unlock()
	snapshot := make([]removalTestEvent, len(events.events))
	copy(snapshot, events.events)
	return snapshot
}

type removalLogHandler struct {
	events *removalTestEvents
}

func (handler *removalLogHandler) Enabled(_ context.Context, _ slog.Level) bool {
	return true
}

func (handler *removalLogHandler) Handle(_ context.Context, record slog.Record) error {
	if record.Message != removalCompletedMessage && record.Message != removedRowsMessage {
		return nil
	}

	attrs := make(map[string]string, record.NumAttrs())
	record.Attrs(func(attr slog.Attr) bool {
		attrs[attr.Key] = attr.Value.String()
		return true
	})
	handler.events.append(removalTestEvent{kind: record.Message, attrs: attrs})
	return nil
}

func (handler *removalLogHandler) WithAttrs(_ []slog.Attr) slog.Handler {
	return handler
}

func (handler *removalLogHandler) WithGroup(_ string) slog.Handler {
	return handler
}

type removalMilvusServer struct {
	milvuspb.UnimplementedMilvusServiceServer

	mu              sync.Mutex
	deleteCallCount int
	deleteCounts    []int64
	failDeleteCall  int
	events          *removalTestEvents
}

func (server *removalMilvusServer) DescribeCollection(
	_ context.Context,
	request *milvuspb.DescribeCollectionRequest,
) (*milvuspb.DescribeCollectionResponse, error) {
	return &milvuspb.DescribeCollectionResponse{
		Status:         removalSuccessStatus(),
		CollectionName: request.GetCollectionName(),
	}, nil
}

func (server *removalMilvusServer) LoadCollection(
	_ context.Context,
	_ *milvuspb.LoadCollectionRequest,
) (*commonpb.Status, error) {
	return removalSuccessStatus(), nil
}

// GetLoadState answers the bounded readiness wait, which reads the load-state
// enum rather than the client's unbounded await. The fake collection is
// immediately queryable, so it reports Loaded on the first probe.
func (server *removalMilvusServer) GetLoadState(
	_ context.Context,
	_ *milvuspb.GetLoadStateRequest,
) (*milvuspb.GetLoadStateResponse, error) {
	return &milvuspb.GetLoadStateResponse{
		Status: removalSuccessStatus(),
		State:  commonpb.LoadState_LoadStateLoaded,
	}, nil
}

func (server *removalMilvusServer) GetLoadingProgress(
	_ context.Context,
	_ *milvuspb.GetLoadingProgressRequest,
) (*milvuspb.GetLoadingProgressResponse, error) {
	return &milvuspb.GetLoadingProgressResponse{
		Status:   removalSuccessStatus(),
		Progress: 100,
	}, nil
}

func (server *removalMilvusServer) Delete(
	_ context.Context,
	_ *milvuspb.DeleteRequest,
) (*milvuspb.MutationResult, error) {
	server.mu.Lock()
	server.deleteCallCount++
	deleteCallCount := server.deleteCallCount
	server.mu.Unlock()

	server.events.append(removalTestEvent{kind: "delete", attrs: nil})
	if deleteCallCount == server.failDeleteCall {
		return nil, status.Error(codes.Internal, "configured delete failure")
	}
	deleteCount := server.deleteCounts[deleteCallCount-1]
	return &milvuspb.MutationResult{
		Status:    removalSuccessStatus(),
		DeleteCnt: deleteCount,
	}, nil
}

func (server *removalMilvusServer) deleteCalls() int {
	server.mu.Lock()
	defer server.mu.Unlock()
	return server.deleteCallCount
}

func removalSuccessStatus() *commonpb.Status {
	return &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}
}

func TestReindexLogsRemovedRowCountsAfterEveryDeleteSucceeds(t *testing.T) {
	events := &removalTestEvents{}
	service := newRemovalTestService(t, events, []int64{500, 0, 7}, 0)
	captureRemovalLogs(t, events)

	removal := semantic.Removal{
		Paths:    []string{"obsolete.go"},
		Prefixes: []string{"conv/one/", "conv/two/"},
	}
	err := service.Reindex(
		context.Background(),
		t.TempDir(),
		nil,
		removal,
		nil,
		nil,
		semantic.StoreColumnSetCode,
	)
	if err != nil {
		t.Fatalf("Reindex returned error: %v", err)
	}

	got := events.snapshot()
	if len(got) != 4 {
		t.Fatalf("event count = %d, want three deletes followed by one success log: %+v", len(got), got)
	}
	for index := 0; index < 3; index++ {
		if got[index].kind != "delete" {
			t.Fatalf("event %d = %q, want delete before the success log", index, got[index].kind)
		}
	}
	completed := got[3]
	if completed.kind != removalCompletedMessage {
		t.Fatalf("last event = %q, want %q after every delete", completed.kind, removalCompletedMessage)
	}
	if completed.attrs["path_rows_removed"] != "500" {
		t.Fatalf(
			"path_rows_removed = %q, want \"500\"",
			completed.attrs["path_rows_removed"],
		)
	}
	if completed.attrs["prefix_rows_removed"] != "7" {
		t.Fatalf(
			"prefix_rows_removed = %q, want \"7\"",
			completed.attrs["prefix_rows_removed"],
		)
	}
	if completed.attrs["rows_removed"] != "507" {
		t.Fatalf("rows_removed = %q, want \"507\"", completed.attrs["rows_removed"])
	}
	if _, found := completed.attrs["path_filters"]; found {
		t.Fatal("success log still exposes the requested path-filter count")
	}
	if _, found := completed.attrs["prefix_filters"]; found {
		t.Fatal("success log still exposes the requested prefix-filter count")
	}
}

func TestReindexDoesNotLogRemovalSuccessWhenSecondPrefixDeleteFails(t *testing.T) {
	events := &removalTestEvents{}
	server, service := newRemovalTestServerAndService(t, events, []int64{11, 13}, 2)
	captureRemovalLogs(t, events)

	err := service.Reindex(
		context.Background(),
		t.TempDir(),
		nil,
		semantic.RemovePrefixes([]string{"conv/one/", "conv/two/"}),
		nil,
		nil,
		semantic.StoreColumnSetCode,
	)
	if err == nil {
		t.Fatal("Reindex returned nil error for the configured second prefix-delete failure")
	}
	if server.deleteCalls() != 2 {
		t.Fatalf("delete call count = %d, want 2", server.deleteCalls())
	}
	for _, event := range events.snapshot() {
		if event.kind == removalCompletedMessage || event.kind == removedRowsMessage {
			t.Fatalf("removal success %q was logged before the second prefix delete failed", event.kind)
		}
	}
}

func captureRemovalLogs(t *testing.T, events *removalTestEvents) {
	t.Helper()
	previous := slog.Default()
	slog.SetDefault(slog.New(&removalLogHandler{events: events}))
	t.Cleanup(func() {
		slog.SetDefault(previous)
	})
}

func newRemovalTestService(
	t *testing.T,
	events *removalTestEvents,
	deleteCounts []int64,
	failDeleteCall int,
) *semantic.Service {
	t.Helper()
	_, service := newRemovalTestServerAndService(
		t,
		events,
		deleteCounts,
		failDeleteCall,
	)
	return service
}

func newRemovalTestServerAndService(
	t *testing.T,
	events *removalTestEvents,
	deleteCounts []int64,
	failDeleteCall int,
) (*removalMilvusServer, *semantic.Service) {
	t.Helper()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen returned error: %v", err)
	}
	server := &removalMilvusServer{
		failDeleteCall: failDeleteCall,
		deleteCounts:   deleteCounts,
		events:         events,
	}
	grpcServer := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(grpcServer, server)
	t.Cleanup(func() {
		grpcServer.Stop()
	})
	go func() {
		_ = grpcServer.Serve(listener)
	}()

	cfg := config.Config{
		EmbeddingProvider: "OpenAI",
		EmbeddingModel:    "test-model",
		OpenAIAPIKey:      "test-key",
		OpenAIBaseURL:     "http://127.0.0.1:1/v1",
		MilvusAddress:     listener.Addr().String(),
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
	return server, service
}

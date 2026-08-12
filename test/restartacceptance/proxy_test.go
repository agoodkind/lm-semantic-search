//go:build restartacceptance

package restartacceptance

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

func TestEmbeddingProxyForwardsAndCanFailRequests(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("X-Backend-Path", request.URL.Path)
		_, _ = io.WriteString(writer, "forwarded")
	}))
	t.Cleanup(backend.Close)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	proxy, err := newEmbeddingProxy(listener, backend.URL)
	if err != nil {
		t.Fatalf("new embedding proxy: %v", err)
	}
	t.Cleanup(func() { _ = proxy.Close() })
	go func() { _ = proxy.Serve() }()

	response, err := http.Get("http://" + listener.Addr().String() + "/v1/embeddings")
	if err != nil {
		t.Fatalf("forwarded request: %v", err)
	}
	body, _ := io.ReadAll(response.Body)
	_ = response.Body.Close()
	if string(body) != "forwarded" || response.Header.Get("X-Backend-Path") != "/v1/embeddings" {
		t.Fatalf("forwarded response status=%d headers=%v body=%q", response.StatusCode, response.Header, body)
	}

	proxy.SetFailure(http.StatusServiceUnavailable, "embedding unavailable")
	response, err = http.Get("http://" + listener.Addr().String() + "/v1/embeddings")
	if err != nil {
		t.Fatalf("failed request: %v", err)
	}
	body, _ = io.ReadAll(response.Body)
	_ = response.Body.Close()
	if response.StatusCode != http.StatusServiceUnavailable || !strings.Contains(string(body), "embedding unavailable") {
		t.Fatalf("controlled response status=%d body=%q", response.StatusCode, body)
	}
	proxy.ClearFailure()
}

func TestEmbeddingProxyGateCountsOnlyAcceptanceFixtureRequests(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(backend.Close)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	proxy, err := newEmbeddingProxy(listener, backend.URL)
	if err != nil {
		t.Fatalf("new embedding proxy: %v", err)
	}
	t.Cleanup(func() { _ = proxy.Close() })
	go func() { _ = proxy.Serve() }()
	client := &http.Client{Timeout: time.Second}
	post := func(input string) error {
		body := strings.NewReader(`{"input":["` + input + `"]}`)
		response, postErr := client.Post("http://"+listener.Addr().String()+"/v1/embeddings", "application/json", body)
		if postErr != nil {
			return postErr
		}
		return response.Body.Close()
	}

	proxy.GateAfter(1)
	if err := post("daemon boot check"); err != nil {
		t.Fatalf("untracked request: %v", err)
	}
	if err := post("restart_acceptance_id:01.go"); err != nil {
		t.Fatalf("first fixture request: %v", err)
	}
	gated := make(chan error, 1)
	go func() { gated <- post("restart_acceptance_id:02.go") }()
	select {
	case <-proxy.GateReached():
	case <-time.After(time.Second):
		t.Fatal("second fixture request did not reach the gate")
	}
	select {
	case err = <-gated:
		t.Fatalf("second fixture request passed the active gate: %v", err)
	default:
	}
	proxy.ClearGate()
	select {
	case err = <-gated:
		if err != nil {
			t.Fatalf("second fixture request after clearing gate: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("second fixture request remained gated after clear")
	}
}

func TestMilvusProxyForwardsNormalTrafficAndInterceptsConfiguredLoadOnly(t *testing.T) {
	backendListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("backend listen: %v", err)
	}
	backend := grpc.NewServer()
	backendService := &proxyTestMilvusServer{}
	milvuspb.RegisterMilvusServiceServer(backend, backendService)
	go func() { _ = backend.Serve(backendListener) }()
	t.Cleanup(backend.Stop)

	proxyListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("proxy listen: %v", err)
	}
	proxy, err := newMilvusProxy(proxyListener, backendListener.Addr().String())
	if err != nil {
		t.Fatalf("new Milvus proxy: %v", err)
	}
	t.Cleanup(func() { _ = proxy.Close() })
	go func() { _ = proxy.Serve() }()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	connection, err := grpc.DialContext(ctx, proxyListener.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock())
	if err != nil {
		t.Fatalf("dial proxy: %v", err)
	}
	t.Cleanup(func() { _ = connection.Close() })
	client := milvuspb.NewMilvusServiceClient(connection)

	response, err := client.GetLoadState(ctx, &milvuspb.GetLoadStateRequest{DbName: "default", CollectionName: "normal"})
	if err != nil {
		t.Fatalf("forward GetLoadState: %v", err)
	}
	if response.GetState() != commonpb.LoadState_LoadStateLoaded {
		t.Fatalf("normal state = %v", response.GetState())
	}
	proxy.SetLoadState("default", "blocked", commonpb.LoadState_LoadStateLoading)
	response, err = client.GetLoadState(ctx, &milvuspb.GetLoadStateRequest{DbName: "default", CollectionName: "blocked"})
	if err != nil {
		t.Fatalf("intercept GetLoadState: %v", err)
	}
	if response.GetState() != commonpb.LoadState_LoadStateLoading {
		t.Fatalf("blocked state = %v", response.GetState())
	}
	metadataContext := metadata.NewOutgoingContext(ctx, metadata.Pairs("dbname", "default"))
	response, err = client.GetLoadState(metadataContext, &milvuspb.GetLoadStateRequest{CollectionName: "blocked"})
	if err != nil || response.GetState() != commonpb.LoadState_LoadStateLoading {
		t.Fatalf("metadata database fault routing: response=%v error=%v", response, err)
	}
	response, err = client.GetLoadState(metadataContext, &milvuspb.GetLoadStateRequest{DbName: "other", CollectionName: "blocked"})
	if err != nil || response.GetState() != commonpb.LoadState_LoadStateLoaded {
		t.Fatalf("request database did not override metadata: response=%v error=%v", response, err)
	}
	response, err = client.GetLoadState(ctx, &milvuspb.GetLoadStateRequest{DbName: "default", CollectionName: "normal"})
	if err != nil || response.GetState() != commonpb.LoadState_LoadStateLoaded {
		t.Fatalf("unconfigured collection was intercepted: response=%v error=%v", response, err)
	}
	other, err := client.GetLoadState(ctx, &milvuspb.GetLoadStateRequest{DbName: "other", CollectionName: "blocked"})
	if err != nil || other.GetState() != commonpb.LoadState_LoadStateLoaded {
		t.Fatalf("same collection in other database was intercepted: response=%v error=%v", other, err)
	}
	proxy.SetLoadFailure("default", "blocked", codes.Unavailable, "load fault")
	_, err = client.LoadCollection(ctx, &milvuspb.LoadCollectionRequest{DbName: "default", CollectionName: "blocked"})
	if status.Code(err) != codes.Unavailable {
		t.Fatalf("load failure code = %v, error=%v", status.Code(err), err)
	}
	if got := proxy.CallCount("GetLoadState", "default", "blocked"); got != 2 {
		t.Fatalf("GetLoadState blocked count = %d, want 2", got)
	}
	if got := proxy.CallCount("LoadCollection", "default", "blocked"); got != 1 {
		t.Fatalf("LoadCollection blocked count = %d, want 1", got)
	}
}

func TestMilvusProxyRelaysServerAndBidirectionalStreams(t *testing.T) {
	backendListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("backend listen: %v", err)
	}
	backendService := &proxyTestMilvusServer{}
	backend := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(backend, backendService)
	go func() { _ = backend.Serve(backendListener) }()
	t.Cleanup(backend.Stop)
	proxyListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("proxy listen: %v", err)
	}
	proxy, err := newMilvusProxy(proxyListener, backendListener.Addr().String())
	if err != nil {
		t.Fatalf("new proxy: %v", err)
	}
	go func() { _ = proxy.Serve() }()
	t.Cleanup(func() { _ = proxy.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	connection, err := grpc.DialContext(ctx, proxyListener.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock())
	if err != nil {
		t.Fatalf("dial proxy: %v", err)
	}
	t.Cleanup(func() { _ = connection.Close() })
	client := milvuspb.NewMilvusServiceClient(connection)
	dump, err := client.DumpMessages(ctx, &milvuspb.DumpMessagesRequest{})
	if err != nil {
		t.Fatalf("start server stream: %v", err)
	}
	if headers, err := dump.Header(); err != nil || len(headers.Get("x-relayed")) == 0 {
		t.Fatalf("server stream headers=%v error=%v", headers, err)
	}
	for index := 0; index < 2; index++ {
		if _, err := dump.Recv(); err != nil {
			t.Fatalf("receive server message %d: %v", index, err)
		}
	}
	if _, err := dump.Recv(); !errors.Is(err, io.EOF) {
		t.Fatalf("server stream end: %v", err)
	}
	if len(dump.Trailer().Get("x-trailer")) == 0 {
		t.Fatal("server stream trailer was not relayed")
	}
	replicate, err := client.CreateReplicateStream(ctx)
	if err != nil {
		t.Fatalf("start bidi stream: %v", err)
	}
	for index := 0; index < 2; index++ {
		if err := replicate.Send(&milvuspb.ReplicateRequest{}); err != nil {
			t.Fatalf("send bidi message %d: %v", index, err)
		}
	}
	if err := replicate.CloseSend(); err != nil {
		t.Fatalf("half-close bidi stream: %v", err)
	}
	for index := 0; index < 2; index++ {
		if _, err := replicate.Recv(); err != nil {
			t.Fatalf("receive bidi message %d: %v", index, err)
		}
	}
	if _, err := replicate.Recv(); !errors.Is(err, io.EOF) {
		t.Fatalf("bidi stream end: %v", err)
	}
	if backendService.replicateCount.Load() != 2 {
		t.Fatalf("backend received %d bidi messages", backendService.replicateCount.Load())
	}
}

func TestMilvusProxyWaitsForBothRelayDirections(t *testing.T) {
	relayContext, cancelRelay := context.WithCancel(context.Background())
	t.Cleanup(cancelRelay)
	clientResult := make(chan error, 1)
	serverResult := make(chan error, 1)
	completed := make(chan error, 1)
	go func() {
		completed <- waitForRelay(context.Background(), cancelRelay, clientResult, serverResult)
	}()

	serverResult <- nil
	select {
	case <-relayContext.Done():
	case <-time.After(time.Second):
		t.Fatal("relay did not cancel after the server direction stopped")
	}
	select {
	case err := <-completed:
		t.Fatalf("relay returned before client direction stopped: %v", err)
	default:
	}
	clientResult <- nil
	select {
	case err := <-completed:
		if err != nil {
			t.Fatalf("relay result: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("relay did not return after both directions stopped")
	}
}

type proxyTestMilvusServer struct {
	milvuspb.UnimplementedMilvusServiceServer
	replicateCount atomic.Int32
}

func (*proxyTestMilvusServer) GetLoadState(
	context.Context,
	*milvuspb.GetLoadStateRequest,
) (*milvuspb.GetLoadStateResponse, error) {
	return &milvuspb.GetLoadStateResponse{
		Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
		State:  commonpb.LoadState_LoadStateLoaded,
	}, nil
}

func (server *proxyTestMilvusServer) DumpMessages(_ *milvuspb.DumpMessagesRequest, stream milvuspb.MilvusService_DumpMessagesServer) error {
	if err := stream.SendHeader(metadata.Pairs("x-relayed", "yes")); err != nil {
		return err
	}
	stream.SetTrailer(metadata.Pairs("x-trailer", "yes"))
	for index := 0; index < 2; index++ {
		if err := stream.Send(&milvuspb.DumpMessagesResponse{}); err != nil {
			return err
		}
	}
	return nil
}

func (server *proxyTestMilvusServer) CreateReplicateStream(stream milvuspb.MilvusService_CreateReplicateStreamServer) error {
	for {
		_, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		server.replicateCount.Add(1)
		if err := stream.Send(&milvuspb.ReplicateResponse{}); err != nil {
			return err
		}
	}
}

func (*proxyTestMilvusServer) LoadCollection(
	context.Context,
	*milvuspb.LoadCollectionRequest,
) (*commonpb.Status, error) {
	return &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil
}

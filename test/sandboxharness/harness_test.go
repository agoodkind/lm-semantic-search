//go:build restartacceptance

package sandboxharness

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func TestEmbeddingProviderReturnsDeterministicVectors(t *testing.T) {
	provider, err := StartEmbeddingProvider(EmbeddingProviderOptions{
		Model: "sandbox-model", Dimension: 16,
	})
	if err != nil {
		t.Fatalf("start embedding provider: %v", err)
	}
	t.Cleanup(func() { _ = provider.Close() })

	response, err := http.Post(
		provider.URL()+"/embeddings",
		"application/json",
		strings.NewReader("{\"model\":\"sandbox-model\",\"input\":[\"same\",\"same\"],\"dimensions\":16}"),
	)
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	defer func() { _ = response.Body.Close() }()
	var decoded struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		t.Fatalf("decode embedding response: %v", err)
	}
	if response.StatusCode != http.StatusOK || len(decoded.Data) != 2 {
		t.Fatalf("status = %d, rows = %d", response.StatusCode, len(decoded.Data))
	}
	if len(decoded.Data[0].Embedding) != 16 {
		t.Fatalf("vector width = %d, want 16", len(decoded.Data[0].Embedding))
	}
	if !slices.Equal(decoded.Data[0].Embedding, decoded.Data[1].Embedding) {
		t.Fatal("identical inputs returned different vectors")
	}
}

func TestEmbeddingProxyRecordsInputsAndControlsFailureAndGate(t *testing.T) {
	provider, err := StartEmbeddingProvider(EmbeddingProviderOptions{
		Model: "sandbox-model", Dimension: 16,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = provider.Close() })
	proxy, err := StartEmbeddingProxy(EmbeddingProxyOptions{
		BackendURL: provider.URL(), Start: true,
		IdentifyInput: func(input string) (string, bool) {
			return strings.CutPrefix(input, "tracked:")
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = proxy.Close() })
	post := func(input string) error {
		body := strings.NewReader("{\"model\":\"sandbox-model\",\"input\":[\"" + input + "\"],\"dimensions\":16}")
		response, postErr := http.Post(proxy.URL()+"/embeddings", "application/json", body)
		if postErr != nil {
			return postErr
		}
		return response.Body.Close()
	}
	proxy.GateAfter(1)
	if err := post("tracked:first"); err != nil {
		t.Fatal(err)
	}
	waiting := make(chan error, 1)
	go func() { waiting <- post("tracked:second") }()
	select {
	case <-proxy.GateReached():
	case <-time.After(time.Second):
		t.Fatal("second tracked request did not reach gate")
	}
	proxy.ClearGate()
	if err := <-waiting; err != nil {
		t.Fatal(err)
	}
	if got := proxy.Inputs(); !slices.Equal(got, []string{"first", "second"}) {
		t.Fatalf("inputs = %v", got)
	}
	proxy.SetFailure(http.StatusServiceUnavailable, "provider unavailable")
	response, err := http.Get(proxy.URL() + "/models")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = response.Body.Close() }()
	body, _ := io.ReadAll(response.Body)
	if response.StatusCode != http.StatusServiceUnavailable ||
		!strings.Contains(string(body), "provider unavailable") {
		t.Fatalf("controlled failure status = %d body = %q", response.StatusCode, body)
	}
}

func TestEmbeddingStoreProxyForwardsAndControlsCollectionLoad(t *testing.T) {
	backendListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	backend := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(backend, &sandboxMilvusServer{})
	go func() { _ = backend.Serve(backendListener) }()
	t.Cleanup(backend.Stop)

	proxy, err := StartEmbeddingStoreProxy(EmbeddingStoreProxyOptions{
		BackendAddress: backendListener.Addr().String(), Start: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = proxy.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	connection, err := grpc.DialContext(
		ctx,
		proxy.Address(),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = connection.Close() })
	client := milvuspb.NewMilvusServiceClient(connection)

	response, err := client.GetLoadState(ctx, &milvuspb.GetLoadStateRequest{
		DbName: "sandbox", CollectionName: "normal",
	})
	if err != nil || response.GetState() != commonpb.LoadState_LoadStateLoaded {
		t.Fatalf("forwarded response = %v error = %v", response, err)
	}
	proxy.SetLoadState("sandbox", "blocked", commonpb.LoadState_LoadStateLoading)
	response, err = client.GetLoadState(ctx, &milvuspb.GetLoadStateRequest{
		DbName: "sandbox", CollectionName: "blocked",
	})
	if err != nil || response.GetState() != commonpb.LoadState_LoadStateLoading {
		t.Fatalf("fault response = %v error = %v", response, err)
	}
	if got := proxy.CallCount("GetLoadState", "sandbox", "blocked"); got != 1 {
		t.Fatalf("load-state calls = %d, want 1", got)
	}
}

type sandboxMilvusServer struct {
	milvuspb.UnimplementedMilvusServiceServer
}

func (*sandboxMilvusServer) GetLoadState(
	context.Context,
	*milvuspb.GetLoadStateRequest,
) (*milvuspb.GetLoadStateResponse, error) {
	return &milvuspb.GetLoadStateResponse{
		Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
		State:  commonpb.LoadState_LoadStateLoaded,
	}, nil
}

func TestSandboxProxiesUseCallerProvidedListeners(t *testing.T) {
	provider, err := StartEmbeddingProvider(EmbeddingProviderOptions{
		Model: "sandbox-model", Dimension: 16,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = provider.Close() })
	embeddingListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	embeddingAddress := embeddingListener.Addr().String()
	embedding, err := StartEmbeddingProxy(EmbeddingProxyOptions{
		Listener: embeddingListener, BackendURL: provider.URL(), Start: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = embedding.Close() })
	if got := strings.TrimPrefix(embedding.URL(), "http://"); !strings.HasPrefix(got, embeddingAddress) {
		t.Fatalf("embedding proxy URL = %q, want listener %q", embedding.URL(), embeddingAddress)
	}

	backendListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	backend := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(backend, &sandboxMilvusServer{})
	go func() { _ = backend.Serve(backendListener) }()
	t.Cleanup(backend.Stop)
	storeListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	storeAddress := storeListener.Addr().String()
	store, err := StartEmbeddingStoreProxy(EmbeddingStoreProxyOptions{
		Listener: storeListener, BackendAddress: backendListener.Addr().String(), Start: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	if store.Address() != storeAddress {
		t.Fatalf("store proxy address = %q, want %q", store.Address(), storeAddress)
	}
}

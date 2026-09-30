//go:build restartacceptance || live

package sandboxharness

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"sync"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// EmbeddingStoreProxyOptions configures a local gRPC proxy and its real backend.
type EmbeddingStoreProxyOptions struct {
	Listener       net.Listener
	BackendAddress string
	Start          bool
}

// StoreCall records a collection operation observed by the proxy.
type StoreCall struct {
	Database   string `json:"database"`
	Collection string `json:"collection"`
	Method     string `json:"method"`
}

type storeFault struct {
	state       *commonpb.LoadState
	failureCode codes.Code
	failureText string
}

type storeCallKey struct {
	method     string
	database   string
	collection string
}

type storeTarget struct {
	database   string
	collection string
}

// EmbeddingStoreProxy forwards Milvus requests and supports explicit test barriers.
type EmbeddingStoreProxy struct {
	listener    net.Listener
	server      *grpc.Server
	backend     *grpc.ClientConn
	mutex       sync.RWMutex
	faults      map[storeTarget]storeFault
	counts      map[storeCallKey]int
	calls       []StoreCall
	unavailable *storeFault
	publication *storePublicationPause
}

// StartEmbeddingStoreProxy opens a proxy connection to the configured backend.
func StartEmbeddingStoreProxy(
	options EmbeddingStoreProxyOptions,
) (*EmbeddingStoreProxy, error) {
	if options.BackendAddress == "" {
		return nil, fmt.Errorf("embedding store backend address is empty")
	}
	connection, err := grpc.NewClient(
		options.BackendAddress,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.ForceCodec(rawCodec{})),
	)
	if err != nil {
		slog.Error("create embedding store backend connection", "err", err)
		return nil, fmt.Errorf("create embedding store backend connection: %w", err)
	}
	listener := options.Listener
	if listener == nil {
		listener, err = (&net.ListenConfig{}).Listen(context.Background(), "tcp", "127.0.0.1:0")
		if err != nil {
			_ = connection.Close()
			slog.Error("listen embedding store proxy", "err", err)
			return nil, fmt.Errorf("listen embedding store proxy: %w", err)
		}
	}
	proxy := &EmbeddingStoreProxy{
		listener: listener,
		backend:  connection,
		faults:   make(map[storeTarget]storeFault),
		counts:   make(map[storeCallKey]int),
	}
	proxy.server = grpc.NewServer(
		grpc.ForceServerCodec(rawCodec{}),
		grpc.UnknownServiceHandler(proxy.streamHandler),
	)
	if options.Start {
		go func() {
			defer func() {
				if recovered := recover(); recovered != nil {
					slog.Error("embedding store proxy panic", "err", fmt.Errorf("proxy panic: %v", recovered))
				}
			}()
			_ = proxy.Serve()
		}()
	}
	return proxy, nil
}

// Address returns the proxy listener address.
func (proxy *EmbeddingStoreProxy) Address() string {
	return proxy.listener.Addr().String()
}

// Serve accepts requests until the proxy stops.
func (proxy *EmbeddingStoreProxy) Serve() error {
	if err := proxy.server.Serve(proxy.listener); err != nil &&
		!errors.Is(err, grpc.ErrServerStopped) {
		slog.Error("serve embedding store proxy", "err", err)
		return fmt.Errorf("serve embedding store proxy: %w", err)
	}
	return nil
}

// Close stops the listener and closes the backend connection.
func (proxy *EmbeddingStoreProxy) Close() error {
	proxy.server.Stop()
	if err := proxy.backend.Close(); err != nil {
		slog.Warn("embedding store forwarding failed", "err", err)
		return fmt.Errorf("close embedding store backend: %w", err)
	}
	return nil
}

// SetLoadState overrides the load state for a selected collection.
func (proxy *EmbeddingStoreProxy) SetLoadState(
	database string,
	collection string,
	state commonpb.LoadState,
) {
	proxy.mutex.Lock()
	target := storeTarget{database: database, collection: collection}
	fault := proxy.faults[target]
	fault.state = &state
	proxy.faults[target] = fault
	proxy.mutex.Unlock()
}

// SetLoadFailure rejects load operations for a selected collection.
func (proxy *EmbeddingStoreProxy) SetLoadFailure(
	database string,
	collection string,
	code codes.Code,
	message string,
) {
	proxy.mutex.Lock()
	target := storeTarget{database: database, collection: collection}
	fault := proxy.faults[target]
	fault.failureCode = code
	fault.failureText = message
	proxy.faults[target] = fault
	proxy.mutex.Unlock()
}

// ClearLoadFault restores forwarding for a selected collection.
func (proxy *EmbeddingStoreProxy) ClearLoadFault(database string, collection string) {
	proxy.mutex.Lock()
	delete(proxy.faults, storeTarget{database: database, collection: collection})
	proxy.mutex.Unlock()
}

// SetUnavailable rejects every request with the configured status.
func (proxy *EmbeddingStoreProxy) SetUnavailable(code codes.Code, message string) {
	proxy.mutex.Lock()
	proxy.unavailable = &storeFault{failureCode: code, failureText: message}
	proxy.mutex.Unlock()
}

// ClearUnavailable restores ordinary request forwarding.
func (proxy *EmbeddingStoreProxy) ClearUnavailable() {
	proxy.mutex.Lock()
	proxy.unavailable = nil
	proxy.mutex.Unlock()
}

// IsUnavailable reports whether every request is configured to fail.
func (proxy *EmbeddingStoreProxy) IsUnavailable() bool {
	proxy.mutex.RLock()
	defer proxy.mutex.RUnlock()
	return proxy.unavailable != nil
}

// CallCount returns the observed count for a selected collection operation.
func (proxy *EmbeddingStoreProxy) CallCount(
	method string,
	database string,
	collection string,
) int {
	proxy.mutex.RLock()
	defer proxy.mutex.RUnlock()
	return proxy.counts[storeCallKey{
		method: method, database: database, collection: collection,
	}]
}

// Calls returns a copy of observed collection operations.
func (proxy *EmbeddingStoreProxy) Calls() []StoreCall {
	proxy.mutex.RLock()
	defer proxy.mutex.RUnlock()
	return append([]StoreCall(nil), proxy.calls...)
}

func (proxy *EmbeddingStoreProxy) forward(stream grpc.ServerStream) error {
	method, ok := grpc.MethodFromServerStream(stream)
	if !ok {
		return status.Error(codes.Internal, "embedding store proxy cannot identify method")
	}
	proxy.mutex.RLock()
	unavailable := proxy.unavailable
	proxy.mutex.RUnlock()
	if unavailable != nil {
		return status.Error(unavailable.failureCode, unavailable.failureText)
	}
	var request []byte
	if err := stream.RecvMsg(&request); err != nil {
		slog.Warn("embedding store forwarding failed", "err", err)
		return fmt.Errorf("receive frontend request: %w", err)
	}
	methodName := methodBase(method)
	incoming, _ := metadata.FromIncomingContext(stream.Context())
	metadataDatabase := ""
	if values := incoming.Get("dbname"); len(values) != 0 {
		metadataDatabase = values[0]
	}
	target := targetForLoadMethod(methodName, request, metadataDatabase)
	if target.collection != "" {
		intercepted, err := proxy.respondToLoadFault(methodName, target, stream)
		if err != nil || intercepted {
			return err
		}
	}
	if methodName == "Upsert" {
		var mutation milvuspb.UpsertRequest
		if err := proto.Unmarshal(request, &mutation); err != nil {
			slog.Warn("embedding store forwarding failed", "err", err)
			return fmt.Errorf("decode publication request: %w", err)
		}
		proxy.mutex.RLock()
		pause := proxy.publication
		proxy.mutex.RUnlock()
		if pause != nil && pause.matches(mutation.GetDbName(), mutation.GetCollectionName(), metadataDatabase) {
			return proxy.forwardPausedPublication(method, request, stream, pause)
		}
	}
	return proxy.relay(method, request, stream)
}

func (proxy *EmbeddingStoreProxy) respondToLoadFault(method string, target storeTarget, stream grpc.ServerStream) (bool, error) {
	proxy.mutex.Lock()
	proxy.counts[storeCallKey{method: method, database: target.database, collection: target.collection}]++
	proxy.calls = append(proxy.calls, StoreCall{Database: target.database, Collection: target.collection, Method: method})
	fault, configured := proxy.faults[target]
	proxy.mutex.Unlock()
	if !configured {
		return false, nil
	}
	response, intercepted, err := interceptLoadMethod(method, fault)
	if err != nil || !intercepted {
		return intercepted, err
	}
	if err := stream.SendMsg(response); err != nil {
		slog.Warn("embedding store forwarding failed", "err", err)
		return true, fmt.Errorf("send configured load response: %w", err)
	}
	return true, nil
}

type storeMethod string

const (
	loadCollectionMethod storeMethod = "LoadCollection"
	loadStateMethod      storeMethod = "GetLoadState"
	loadProgressMethod   storeMethod = "GetLoadingProgress"
)

func targetForLoadMethod(
	method string,
	body []byte,
	metadataDatabase string,
) storeTarget {
	var request proto.Message
	switch storeMethod(method) {
	case loadCollectionMethod:
		request = &milvuspb.LoadCollectionRequest{}
	case loadStateMethod:
		request = &milvuspb.GetLoadStateRequest{}
	case loadProgressMethod:
		request = &milvuspb.GetLoadingProgressRequest{}
	default:
		return storeTarget{}
	}
	if err := proto.Unmarshal(body, request); err != nil {
		return storeTarget{}
	}
	named, ok := request.(interface{ GetCollectionName() string })
	if !ok {
		return storeTarget{}
	}
	database := ""
	if databaseRequest, ok := request.(interface{ GetDbName() string }); ok {
		database = databaseRequest.GetDbName()
	}
	if database == "" {
		database = metadataDatabase
	}
	return storeTarget{database: database, collection: named.GetCollectionName()}
}

func (proxy *EmbeddingStoreProxy) relay(
	method string,
	firstRequest []byte,
	frontend grpc.ServerStream,
) error {
	incoming, _ := metadata.FromIncomingContext(frontend.Context())
	relayContext, cancel := context.WithCancel(
		metadata.NewOutgoingContext(frontend.Context(), incoming.Copy()),
	)
	defer cancel()
	backend, err := proxy.backend.NewStream(
		relayContext,
		&grpc.StreamDesc{ClientStreams: true, ServerStreams: true},
		method,
	)
	if err != nil {
		slog.Warn("embedding store forwarding failed", "err", err)
		return fmt.Errorf("open backend stream: %w", err)
	}
	if err := backend.SendMsg(firstRequest); err != nil {
		slog.Warn("embedding store forwarding failed", "err", err)
		return fmt.Errorf("send backend request: %w", err)
	}
	clientResult := make(chan error, 1)
	serverResult := make(chan error, 1)
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.Error("embedding store proxy panic", "err", fmt.Errorf("proxy panic: %v", recovered))
			}
		}()
		relayClientMessages(frontend, backend, clientResult)
	}()
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.Error("embedding store proxy panic", "err", fmt.Errorf("proxy panic: %v", recovered))
			}
		}()
		relayServerMessages(frontend, backend, serverResult)
	}()
	return WaitForRelay(frontend.Context(), cancel, clientResult, serverResult)
}

// WaitForRelay waits for both directions and returns the first stream failure.
func WaitForRelay(
	frontendContext context.Context,
	cancel context.CancelFunc,
	clientResult <-chan error,
	serverResult <-chan error,
) error {
	clientDone := false
	serverDone := false
	var firstError error
	frontendDone := frontendContext.Done()
	for !clientDone || !serverDone {
		select {
		case err := <-clientResult:
			clientDone = true
			if err != nil && firstError == nil {
				firstError = err
				cancel()
			}
		case err := <-serverResult:
			serverDone = true
			if err != nil && firstError == nil {
				firstError = err
			}
			cancel()
		case <-frontendDone:
			if firstError == nil {
				firstError = context.Cause(frontendContext)
			}
			cancel()
			frontendDone = nil
		}
	}
	if firstError != nil {
		slog.Warn("embedding store forwarding failed", "err", firstError)
		return fmt.Errorf("relay stream: %w", firstError)
	}
	return nil
}

func relayClientMessages(
	frontend grpc.ServerStream,
	backend grpc.ClientStream,
	result chan<- error,
) {
	for {
		var message []byte
		err := frontend.RecvMsg(&message)
		if errors.Is(err, io.EOF) {
			result <- backend.CloseSend()
			return
		}
		if err != nil {
			result <- err
			return
		}
		if err := backend.SendMsg(message); err != nil {
			result <- err
			return
		}
	}
}

func relayServerMessages(
	frontend grpc.ServerStream,
	backend grpc.ClientStream,
	result chan<- error,
) {
	headers, err := backend.Header()
	if err != nil {
		result <- err
		return
	}
	if len(headers) != 0 {
		if err := frontend.SendHeader(headers); err != nil {
			result <- err
			return
		}
	}
	for {
		var message []byte
		err := backend.RecvMsg(&message)
		if errors.Is(err, io.EOF) {
			frontend.SetTrailer(backend.Trailer())
			result <- nil
			return
		}
		if err != nil {
			frontend.SetTrailer(backend.Trailer())
			result <- err
			return
		}
		if err := frontend.SendMsg(message); err != nil {
			result <- err
			return
		}
	}
}

func interceptLoadMethod(
	method string,
	fault storeFault,
) ([]byte, bool, error) {
	if fault.failureCode != codes.OK {
		return nil, true, status.Error(fault.failureCode, fault.failureText)
	}
	if fault.state == nil {
		return nil, false, nil
	}
	success := &commonpb.Status{Code: 0}
	var response proto.Message
	switch storeMethod(method) {
	case loadCollectionMethod:
		return nil, false, nil
	case loadStateMethod:
		response = &milvuspb.GetLoadStateResponse{Status: success, State: *fault.state}
	case loadProgressMethod:
		progress := int64(0)
		if *fault.state == commonpb.LoadState_LoadStateLoaded {
			progress = 100
		}
		response = &milvuspb.GetLoadingProgressResponse{
			Status: success, Progress: progress,
		}
	default:
		return nil, false, nil
	}
	body, err := proto.Marshal(response)
	if err != nil {
		return nil, true, status.Errorf(
			codes.Internal,
			"encode embedding store proxy response: %v",
			err,
		)
	}
	return body, true, nil
}

func methodBase(method string) string {
	for index := len(method) - 1; index >= 0; index-- {
		if method[index] == '/' {
			return method[index+1:]
		}
	}
	return method
}

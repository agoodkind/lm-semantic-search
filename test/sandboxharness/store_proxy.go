//go:build restartacceptance

package sandboxharness

import (
	"context"
	"errors"
	"fmt"
	"io"
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

type EmbeddingStoreProxyOptions struct {
	Listener       net.Listener
	BackendAddress string
	Start          bool
}

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

type EmbeddingStoreProxy struct {
	listener    net.Listener
	server      *grpc.Server
	backend     *grpc.ClientConn
	mutex       sync.RWMutex
	faults      map[storeTarget]storeFault
	counts      map[storeCallKey]int
	calls       []StoreCall
	unavailable *storeFault
}

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
		return nil, fmt.Errorf("create embedding store backend connection: %w", err)
	}
	listener := options.Listener
	if listener == nil {
		listener, err = net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			_ = connection.Close()
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
		grpc.UnknownServiceHandler(proxy.forward),
	)
	if options.Start {
		go func() { _ = proxy.Serve() }()
	}
	return proxy, nil
}

func (proxy *EmbeddingStoreProxy) Address() string {
	return proxy.listener.Addr().String()
}

func (proxy *EmbeddingStoreProxy) Serve() error {
	if err := proxy.server.Serve(proxy.listener); err != nil &&
		!errors.Is(err, grpc.ErrServerStopped) {
		return fmt.Errorf("serve embedding store proxy: %w", err)
	}
	return nil
}

func (proxy *EmbeddingStoreProxy) Close() error {
	proxy.server.Stop()
	return proxy.backend.Close()
}

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

func (proxy *EmbeddingStoreProxy) ClearLoadFault(database string, collection string) {
	proxy.mutex.Lock()
	delete(proxy.faults, storeTarget{database: database, collection: collection})
	proxy.mutex.Unlock()
}

func (proxy *EmbeddingStoreProxy) SetUnavailable(code codes.Code, message string) {
	proxy.mutex.Lock()
	proxy.unavailable = &storeFault{failureCode: code, failureText: message}
	proxy.mutex.Unlock()
}

func (proxy *EmbeddingStoreProxy) ClearUnavailable() {
	proxy.mutex.Lock()
	proxy.unavailable = nil
	proxy.mutex.Unlock()
}

func (proxy *EmbeddingStoreProxy) IsUnavailable() bool {
	proxy.mutex.RLock()
	defer proxy.mutex.RUnlock()
	return proxy.unavailable != nil
}

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

func (proxy *EmbeddingStoreProxy) Calls() []StoreCall {
	proxy.mutex.RLock()
	defer proxy.mutex.RUnlock()
	return append([]StoreCall(nil), proxy.calls...)
}

func (proxy *EmbeddingStoreProxy) forward(_ interface{}, stream grpc.ServerStream) error {
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
		return err
	}
	methodName := methodBase(method)
	incoming, _ := metadata.FromIncomingContext(stream.Context())
	metadataDatabase := ""
	if values := incoming.Get("dbname"); len(values) != 0 {
		metadataDatabase = values[0]
	}
	target := targetForLoadMethod(methodName, request, metadataDatabase)
	if target.collection != "" {
		proxy.mutex.Lock()
		key := storeCallKey{
			method:     methodName,
			database:   target.database,
			collection: target.collection,
		}
		proxy.counts[key]++
		proxy.calls = append(proxy.calls, StoreCall{
			Database:   target.database,
			Collection: target.collection,
			Method:     methodName,
		})
		fault, configured := proxy.faults[target]
		proxy.mutex.Unlock()
		if configured {
			response, intercepted, err := interceptLoadMethod(methodName, fault)
			if err != nil {
				return err
			}
			if intercepted {
				return stream.SendMsg(response)
			}
		}
	}
	return proxy.relay(method, request, stream)
}

func targetForLoadMethod(
	method string,
	body []byte,
	metadataDatabase string,
) storeTarget {
	var request proto.Message
	switch method {
	case "LoadCollection":
		request = &milvuspb.LoadCollectionRequest{}
	case "GetLoadState":
		request = &milvuspb.GetLoadStateRequest{}
	case "GetLoadingProgress":
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
		return err
	}
	if err := backend.SendMsg(firstRequest); err != nil {
		return err
	}
	clientResult := make(chan error, 1)
	serverResult := make(chan error, 1)
	go relayClientMessages(frontend, backend, clientResult)
	go relayServerMessages(frontend, backend, serverResult)
	return WaitForRelay(frontend.Context(), cancel, clientResult, serverResult)
}

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
	return firstError
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
	success := &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}
	var response proto.Message
	switch method {
	case "GetLoadState":
		response = &milvuspb.GetLoadStateResponse{Status: success, State: *fault.state}
	case "GetLoadingProgress":
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

type rawCodec struct{}

func (rawCodec) Name() string {
	return "proto"
}

func (rawCodec) Marshal(value interface{}) ([]byte, error) {
	body, ok := value.([]byte)
	if !ok {
		return nil, fmt.Errorf("raw codec cannot marshal %T", value)
	}
	return body, nil
}

func (rawCodec) Unmarshal(body []byte, value interface{}) error {
	target, ok := value.(*[]byte)
	if !ok {
		return fmt.Errorf("raw codec cannot unmarshal into %T", value)
	}
	*target = append((*target)[:0], body...)
	return nil
}

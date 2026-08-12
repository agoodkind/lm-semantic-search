//go:build restartacceptance

package restartacceptance

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"regexp"
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

type embeddingFailure struct {
	statusCode int
	body       string
}

type embeddingProxy struct {
	listener               net.Listener
	server                 *http.Server
	mutex                  sync.RWMutex
	failure                *embeddingFailure
	gateAfter              int
	fixtureInputsForwarded int
	gate                   chan struct{}
	gateReached            chan struct{}
	inputs                 []string
}

var acceptanceInputPattern = regexp.MustCompile(`restart_acceptance_id:([0-9]+\.go)`)

func newEmbeddingProxy(listener net.Listener, backendURL string) (*embeddingProxy, error) {
	parsedURL, err := url.Parse(backendURL)
	if err != nil {
		return nil, fmt.Errorf("parse embedding backend URL: %w", err)
	}
	if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
		return nil, fmt.Errorf("embedding backend URL scheme %q is not HTTP", parsedURL.Scheme)
	}
	if parsedURL.Host == "" {
		return nil, fmt.Errorf("embedding backend URL has no host")
	}
	proxy := &embeddingProxy{listener: listener}
	reverseProxy := httputil.NewSingleHostReverseProxy(parsedURL)
	proxy.server = &http.Server{
		Handler: http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			body, readErr := io.ReadAll(request.Body)
			if readErr != nil {
				http.Error(writer, "read embedding request", http.StatusBadRequest)
				return
			}
			request.Body = io.NopCloser(bytes.NewReader(body))
			inputIDs := embeddingInputIDs(body)
			for {
				proxy.mutex.Lock()
				failure := proxy.failure
				if failure != nil {
					proxy.mutex.Unlock()
					http.Error(writer, failure.body, failure.statusCode)
					return
				}
				if len(inputIDs) > 0 && proxy.gateAfter > 0 && proxy.fixtureInputsForwarded >= proxy.gateAfter {
					gate := proxy.gate
					select {
					case proxy.gateReached <- struct{}{}:
					default:
					}
					proxy.mutex.Unlock()
					select {
					case <-gate:
						continue
					case <-request.Context().Done():
						proxy.ClearGate()
						return
					}
				}
				proxy.fixtureInputsForwarded += len(inputIDs)
				proxy.inputs = append(proxy.inputs, inputIDs...)
				proxy.mutex.Unlock()
				break
			}
			reverseProxy.ServeHTTP(writer, request)
		}),
	}
	return proxy, nil
}

func (proxy *embeddingProxy) Serve() error {
	err := proxy.server.Serve(proxy.listener)
	if err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("serve embedding proxy: %w", err)
	}
	return nil
}

func (proxy *embeddingProxy) Close() error {
	return proxy.server.Close()
}

func (proxy *embeddingProxy) SetFailure(statusCode int, body string) {
	proxy.mutex.Lock()
	proxy.failure = &embeddingFailure{statusCode: statusCode, body: body}
	proxy.clearGateLocked()
	proxy.mutex.Unlock()
}

func (proxy *embeddingProxy) ClearFailure() {
	proxy.mutex.Lock()
	proxy.failure = nil
	proxy.mutex.Unlock()
}

func (proxy *embeddingProxy) GateAfter(forwarded int) {
	proxy.mutex.Lock()
	proxy.clearGateLocked()
	proxy.gateAfter = forwarded
	proxy.gate = make(chan struct{})
	proxy.gateReached = make(chan struct{}, 1)
	proxy.mutex.Unlock()
}

func (proxy *embeddingProxy) GateReached() <-chan struct{} {
	proxy.mutex.RLock()
	defer proxy.mutex.RUnlock()
	return proxy.gateReached
}

func (proxy *embeddingProxy) ClearGate() {
	proxy.mutex.Lock()
	proxy.clearGateLocked()
	proxy.mutex.Unlock()
}

func (proxy *embeddingProxy) clearGateLocked() {
	if proxy.gate != nil {
		close(proxy.gate)
	}
	proxy.gate = nil
	proxy.gateReached = nil
	proxy.gateAfter = 0
}

func (proxy *embeddingProxy) Inputs() []string {
	proxy.mutex.RLock()
	defer proxy.mutex.RUnlock()
	return append([]string(nil), proxy.inputs...)
}

func embeddingInputIDs(body []byte) []string {
	var request struct {
		Input json.RawMessage `json:"input"`
	}
	if json.Unmarshal(body, &request) != nil {
		return nil
	}
	var values []string
	if json.Unmarshal(request.Input, &values) != nil {
		var value string
		if json.Unmarshal(request.Input, &value) == nil {
			values = []string{value}
		}
	}
	result := make([]string, 0, len(values))
	for _, value := range values {
		match := acceptanceInputPattern.FindStringSubmatch(value)
		if len(match) == 2 {
			result = append(result, match[1])
		}
	}
	return result
}

type loadFault struct {
	state       *commonpb.LoadState
	failureCode codes.Code
	failureText string
}

type loadCallKey struct {
	method     string
	database   string
	collection string
}

type loadTarget struct {
	database   string
	collection string
}

type milvusProxy struct {
	listener    net.Listener
	server      *grpc.Server
	backend     *grpc.ClientConn
	mutex       sync.RWMutex
	faults      map[loadTarget]loadFault
	counts      map[loadCallKey]int
	calls       []milvusProxyCall
	unavailable *loadFault
}

func newMilvusProxy(listener net.Listener, backendAddress string) (*milvusProxy, error) {
	connection, err := grpc.NewClient(
		backendAddress,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.ForceCodec(rawCodec{})),
	)
	if err != nil {
		return nil, fmt.Errorf("create Milvus backend connection: %w", err)
	}
	proxy := &milvusProxy{
		listener: listener,
		backend:  connection,
		faults:   make(map[loadTarget]loadFault),
		counts:   make(map[loadCallKey]int),
	}
	proxy.server = grpc.NewServer(
		grpc.ForceServerCodec(rawCodec{}),
		grpc.UnknownServiceHandler(proxy.forward),
	)
	return proxy, nil
}

func (proxy *milvusProxy) Serve() error {
	if err := proxy.server.Serve(proxy.listener); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
		return fmt.Errorf("serve Milvus proxy: %w", err)
	}
	return nil
}

func (proxy *milvusProxy) Close() error {
	proxy.server.Stop()
	return proxy.backend.Close()
}

func (proxy *milvusProxy) SetLoadState(database string, collection string, state commonpb.LoadState) {
	proxy.mutex.Lock()
	target := loadTarget{database: database, collection: collection}
	fault := proxy.faults[target]
	fault.state = &state
	proxy.faults[target] = fault
	proxy.mutex.Unlock()
}

func (proxy *milvusProxy) SetLoadFailure(database string, collection string, code codes.Code, message string) {
	proxy.mutex.Lock()
	target := loadTarget{database: database, collection: collection}
	fault := proxy.faults[target]
	fault.failureCode = code
	fault.failureText = message
	proxy.faults[target] = fault
	proxy.mutex.Unlock()
}

func (proxy *milvusProxy) ClearLoadFault(database string, collection string) {
	proxy.mutex.Lock()
	delete(proxy.faults, loadTarget{database: database, collection: collection})
	proxy.mutex.Unlock()
}

func (proxy *milvusProxy) SetUnavailable(code codes.Code, message string) {
	proxy.mutex.Lock()
	proxy.unavailable = &loadFault{failureCode: code, failureText: message}
	proxy.mutex.Unlock()
}

func (proxy *milvusProxy) ClearUnavailable() {
	proxy.mutex.Lock()
	proxy.unavailable = nil
	proxy.mutex.Unlock()
}

func (proxy *milvusProxy) CallCount(method string, database string, collection string) int {
	proxy.mutex.RLock()
	defer proxy.mutex.RUnlock()
	return proxy.counts[loadCallKey{method: method, database: database, collection: collection}]
}

func (proxy *milvusProxy) Calls() []milvusProxyCall {
	proxy.mutex.RLock()
	defer proxy.mutex.RUnlock()
	return append([]milvusProxyCall(nil), proxy.calls...)
}

func (proxy *milvusProxy) forward(_ interface{}, stream grpc.ServerStream) error {
	method, ok := grpc.MethodFromServerStream(stream)
	if !ok {
		return status.Error(codes.Internal, "Milvus proxy cannot identify method")
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
		proxy.counts[loadCallKey{method: methodName, database: target.database, collection: target.collection}]++
		proxy.calls = append(proxy.calls, milvusProxyCall{Database: target.database, Collection: target.collection, Method: methodName})
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

func targetForLoadMethod(method string, body []byte, metadataDatabase string) loadTarget {
	var request proto.Message
	switch method {
	case "LoadCollection":
		request = &milvuspb.LoadCollectionRequest{}
	case "GetLoadState":
		request = &milvuspb.GetLoadStateRequest{}
	case "GetLoadingProgress":
		request = &milvuspb.GetLoadingProgressRequest{}
	default:
		return loadTarget{}
	}
	if err := proto.Unmarshal(body, request); err != nil {
		return loadTarget{}
	}
	named, ok := request.(interface{ GetCollectionName() string })
	if !ok {
		return loadTarget{}
	}
	database := ""
	if databaseRequest, ok := request.(interface{ GetDbName() string }); ok {
		database = databaseRequest.GetDbName()
	}
	if database == "" {
		database = metadataDatabase
	}
	return loadTarget{database: database, collection: named.GetCollectionName()}
}

func (proxy *milvusProxy) relay(method string, firstRequest []byte, frontend grpc.ServerStream) error {
	incoming, _ := metadata.FromIncomingContext(frontend.Context())
	relayContext, cancel := context.WithCancel(metadata.NewOutgoingContext(frontend.Context(), incoming.Copy()))
	defer cancel()
	backend, err := proxy.backend.NewStream(relayContext, &grpc.StreamDesc{ClientStreams: true, ServerStreams: true}, method)
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
	return waitForRelay(frontend.Context(), cancel, clientResult, serverResult)
}

func waitForRelay(
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

func relayClientMessages(frontend grpc.ServerStream, backend grpc.ClientStream, result chan<- error) {
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

func relayServerMessages(frontend grpc.ServerStream, backend grpc.ClientStream, result chan<- error) {
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

func interceptLoadMethod(method string, fault loadFault) ([]byte, bool, error) {
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
		response = &milvuspb.GetLoadingProgressResponse{Status: success, Progress: progress}
	default:
		return nil, false, nil
	}
	body, err := proto.Marshal(response)
	if err != nil {
		return nil, true, status.Errorf(codes.Internal, "encode Milvus proxy response: %v", err)
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

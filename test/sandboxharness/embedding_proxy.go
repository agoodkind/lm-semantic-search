//go:build restartacceptance

package sandboxharness

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"sync"
)

type InputIdentifier func(string) (string, bool)

type EmbeddingProxyOptions struct {
	Listener      net.Listener
	BackendURL    string
	IdentifyInput InputIdentifier
	Start         bool
}

type embeddingFailure struct {
	statusCode int
	body       string
}

type EmbeddingProxy struct {
	listener         net.Listener
	server           *http.Server
	identifyInput    InputIdentifier
	mutex            sync.RWMutex
	failure          *embeddingFailure
	gateAfter        int
	inputsForwarded  int
	gate             chan struct{}
	gateReached      chan struct{}
	gateReachedCount int
	inputs           []string
}

func StartEmbeddingProxy(options EmbeddingProxyOptions) (*EmbeddingProxy, error) {
	parsedURL, err := url.Parse(options.BackendURL)
	if err != nil {
		return nil, fmt.Errorf("parse embedding backend URL: %w", err)
	}
	if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
		return nil, fmt.Errorf("embedding backend URL scheme %q is not HTTP", parsedURL.Scheme)
	}
	if parsedURL.Host == "" {
		return nil, fmt.Errorf("embedding backend URL has no host")
	}
	listener := options.Listener
	if listener == nil {
		listener, err = net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			return nil, fmt.Errorf("listen embedding proxy: %w", err)
		}
	}
	proxy := &EmbeddingProxy{listener: listener, identifyInput: options.IdentifyInput}
	reverseProxy := httputil.NewSingleHostReverseProxy(parsedURL)
	proxy.server = &http.Server{
		Handler: http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			proxy.forward(writer, request, reverseProxy)
		}),
	}
	if options.Start {
		go func() { _ = proxy.Serve() }()
	}
	return proxy, nil
}

func (proxy *EmbeddingProxy) URL() string {
	return "http://" + proxy.listener.Addr().String() + "/v1"
}

func (proxy *EmbeddingProxy) Serve() error {
	err := proxy.server.Serve(proxy.listener)
	if err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("serve embedding proxy: %w", err)
	}
	return nil
}

func (proxy *EmbeddingProxy) Close() error {
	return proxy.server.Close()
}

func (proxy *EmbeddingProxy) forward(
	writer http.ResponseWriter,
	request *http.Request,
	reverseProxy *httputil.ReverseProxy,
) {
	body, err := io.ReadAll(request.Body)
	if err != nil {
		http.Error(writer, "read embedding request", http.StatusBadRequest)
		return
	}
	request.Body = io.NopCloser(bytes.NewReader(body))
	inputs := proxy.identifyInputs(body)
	for {
		proxy.mutex.Lock()
		failure := proxy.failure
		if failure != nil {
			proxy.mutex.Unlock()
			http.Error(writer, failure.body, failure.statusCode)
			return
		}
		if len(inputs) > 0 && proxy.gateAfter > 0 &&
			proxy.inputsForwarded >= proxy.gateAfter {
			gate := proxy.gate
			proxy.gateReachedCount++
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
		proxy.inputsForwarded += len(inputs)
		proxy.inputs = append(proxy.inputs, inputs...)
		proxy.mutex.Unlock()
		break
	}
	reverseProxy.ServeHTTP(writer, request)
}

func (proxy *EmbeddingProxy) identifyInputs(body []byte) []string {
	if proxy.identifyInput == nil {
		return nil
	}
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
	identities := make([]string, 0, len(values))
	for _, value := range values {
		if identity, tracked := proxy.identifyInput(value); tracked {
			identities = append(identities, identity)
		}
	}
	return identities
}

func (proxy *EmbeddingProxy) SetFailure(statusCode int, body string) {
	proxy.mutex.Lock()
	proxy.failure = &embeddingFailure{statusCode: statusCode, body: body}
	proxy.clearGateLocked()
	proxy.mutex.Unlock()
}

func (proxy *EmbeddingProxy) ClearFailure() {
	proxy.mutex.Lock()
	proxy.failure = nil
	proxy.mutex.Unlock()
}

func (proxy *EmbeddingProxy) GateAfter(forwarded int) {
	proxy.mutex.Lock()
	proxy.clearGateLocked()
	proxy.gateAfter = forwarded
	proxy.gate = make(chan struct{})
	proxy.gateReached = make(chan struct{}, 1)
	proxy.mutex.Unlock()
}

func (proxy *EmbeddingProxy) GateReached() <-chan struct{} {
	proxy.mutex.RLock()
	defer proxy.mutex.RUnlock()
	return proxy.gateReached
}

func (proxy *EmbeddingProxy) GateReachedCount() int {
	proxy.mutex.RLock()
	defer proxy.mutex.RUnlock()
	return proxy.gateReachedCount
}

func (proxy *EmbeddingProxy) ClearGate() {
	proxy.mutex.Lock()
	proxy.clearGateLocked()
	proxy.mutex.Unlock()
}

func (proxy *EmbeddingProxy) clearGateLocked() {
	if proxy.gate != nil {
		close(proxy.gate)
	}
	proxy.gate = nil
	proxy.gateReached = nil
	proxy.gateAfter = 0
}

func (proxy *EmbeddingProxy) Inputs() []string {
	proxy.mutex.RLock()
	defer proxy.mutex.RUnlock()
	return append([]string(nil), proxy.inputs...)
}

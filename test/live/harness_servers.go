//go:build live

package live

import (
	"context"
	cryptorand "crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"google.golang.org/grpc"
)

// startInProcessServer serves the daemon gRPC service on a throwaway unix socket
// in a goroutine and returns a stop closure that GracefulStops the server and
// removes the socket. Readiness is a successful dial by the caller, so no log
// tailing is needed. It mirrors internal/daemon's own test helper.
func startInProcessServer(t *testing.T, ctx context.Context, manager *daemon.Manager, socketPath string) func() {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(socketPath), 0o755); err != nil {
		t.Fatalf("mkdir socket dir returned error: %v", err)
	}
	_ = os.Remove(socketPath)

	listener, err := (&net.ListenConfig{}).Listen(ctx, "unix", socketPath)
	if err != nil {
		t.Fatalf("listen on unix socket returned error: %v", err)
	}
	server := grpc.NewServer(
		grpc.MaxRecvMsgSize(grpcutil.MaxMessageBytes),
		grpc.MaxSendMsgSize(grpcutil.MaxMessageBytes),
	)
	pb.RegisterSemanticSearchDaemonServiceServer(server, daemon.NewGRPCServer(manager, nil))
	slog.Debug("start live daemon server", "socket", socketPath)
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.Error("live daemon server panic", "err", fmt.Errorf("test server panic: %v", recovered))
			}
		}()
		_ = server.Serve(listener)
	}()
	return func() {
		server.GracefulStop()
		_ = listener.Close()
		_ = os.Remove(socketPath)
	}
}

// newFakeEmbeddingServer starts a local OpenAI-compatible embedding endpoint. It
// answers the health probe (GET .../models) with a minimal models list and every
// embed request (POST .../embeddings) with one fixed-width vector per input,
// keyed by a content hash so identical content yields an identical vector and the
// engine's content-hash reuse path stays exercised.
// embedGate lets a test pace embedding requests. When installed, every embed
// request announces its batch size on arrived, then blocks until the test sends
// on release, so the test can read job progress between batches. The models
// (health) route is never gated.
type embedGate struct {
	arrived chan int
	release chan struct{}
}

func newFakeEmbeddingServer(t *testing.T, gate *embedGate) *httptest.Server {
	t.Helper()
	return newFakeEmbeddingServerWithDimension(t, gate, fakeEmbeddingDimension)
}

func newFakeEmbeddingServerWithDimension(
	t *testing.T,
	gate *embedGate,
	dimension int,
) *httptest.Server {
	t.Helper()
	return newFakeEmbeddingServerWithRecorder(t, gate, dimension, nil)
}

func newFakeEmbeddingServerWithRecorder(
	t *testing.T,
	gate *embedGate,
	dimension int,
	recorder *embeddingCallRecorder,
) *httptest.Server {
	t.Helper()
	handler := http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		switch {
		case strings.HasSuffix(request.URL.Path, "/models"):
			writeModelsList(writer)
		case strings.HasSuffix(request.URL.Path, "/embeddings"):
			writeEmbeddings(t, writer, request, gate, dimension, recorder)
		default:
			http.Error(writer, "unexpected path "+request.URL.Path, http.StatusNotFound)
		}
	})
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return server
}

type modelListResponse struct {
	Object string                   `json:"object"`
	Data   []embeddingModelResponse `json:"data"`
}

type embeddingModelResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int    `json:"created"`
	OwnedBy string `json:"owned_by"`
}

func writeModelsList(writer http.ResponseWriter) {
	writer.Header().Set("Content-Type", "application/json")
	response := modelListResponse{Object: "list", Data: []embeddingModelResponse{{ID: "text-embedding-3-small", Object: "model", Created: 0, OwnedBy: "live-harness"}}}
	if err := json.NewEncoder(writer).Encode(response); err != nil {
		slog.Error("encode test model list", "err", err)
	}
}

func writeEmbeddings(
	t *testing.T,
	writer http.ResponseWriter,
	request *http.Request,
	gate *embedGate,
	dimension int,
	recorder *embeddingCallRecorder,
) {
	t.Helper()
	inputs, err := ReadEmbeddingInputs(request)
	if err != nil {
		http.Error(writer, err.Error(), http.StatusBadRequest)
		return
	}
	if recorder != nil {
		recorder.record(inputs)
	}
	if gate != nil {
		gate.arrived <- len(inputs)
		<-gate.release
	}
	type row struct {
		Object    string    `json:"object"`
		Index     int       `json:"index"`
		Embedding []float64 `json:"embedding"`
	}
	rows := make([]row, 0, len(inputs))
	for index, text := range inputs {
		rows = append(rows, row{
			Object:    "embedding",
			Index:     index,
			Embedding: deterministicVector(text, dimension),
		})
	}
	writer.Header().Set("Content-Type", "application/json")
	response := struct {
		Object string         `json:"object"`
		Model  string         `json:"model"`
		Data   []row          `json:"data"`
		Usage  map[string]int `json:"usage"`
	}{Object: "list", Model: "text-embedding-3-small", Data: rows, Usage: map[string]int{"prompt_tokens": 1, "total_tokens": 1}}
	if err := json.NewEncoder(writer).Encode(response); err != nil {
		t.Logf("encode embedding response failed: %v", err)
	}
}

// ReadEmbeddingInputs reads the request's input field, accepting both the array
// form the batch embedder sends and a bare single string, so the fake is robust
// to either shape.
func ReadEmbeddingInputs(request *http.Request) ([]string, error) {
	var body struct {
		Input json.RawMessage `json:"input"`
	}
	if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
		slog.Warn("live test dependency failed", "err", err)
		return nil, fmt.Errorf("decode embedding request: %w", err)
	}
	var asArray []string
	if err := json.Unmarshal(body.Input, &asArray); err == nil {
		return asArray, nil
	}
	var asString string
	if err := json.Unmarshal(body.Input, &asString); err == nil {
		return []string{asString}, nil
	}
	slog.Warn("reject embedding input", "err", fmt.Errorf("embedding request input was neither an array nor a string"))
	return nil, fmt.Errorf("embedding request input was neither an array nor a string")
}

// deterministicVector maps content to a fixed-width unit vector derived from its
// SHA-256 digest, so identical content always yields an identical vector (reuse
// works) and distinct content yields a distinct one.
func deterministicVector(content string, dimension int) []float64 {
	digest := sha256.Sum256([]byte(content))
	vector := make([]float64, dimension)
	var norm float64
	for i := range dimension {
		value := (float64(digest[i%len(digest)]) - 128.0) / 128.0
		vector[i] = value
		norm += value * value
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		vector[0] = 1
		return vector
	}
	for i := range vector {
		vector[i] /= norm
	}
	return vector
}

// correlatedContext wraps ctx with the trace/span identity the daemon requires in
// strict mode, so every RPC and manager read carries a correlation.
func correlatedContext() context.Context {
	return grpcutil.WithCorrelation(context.Background())
}

// randomID returns a hex token unique per test, so each run's collection id (and
// therefore its derived Milvus collection name) is fresh and never collides with
// another run or with production.
func randomID() string {
	buffer := make([]byte, 16)
	if _, err := cryptorand.Read(buffer); err != nil {
		return strconv.FormatInt(clock.Now().UnixNano(), 16)
	}
	return hex.EncodeToString(buffer)
}

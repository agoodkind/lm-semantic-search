//go:build restartacceptance

package sandboxharness

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net"
	"net/http"
	"strings"
	"time"
)

type EmbeddingProviderOptions struct {
	Model     string
	Dimension int
}

type EmbeddingProvider struct {
	listener  net.Listener
	server    *http.Server
	done      chan error
	model     string
	dimension int
}

func StartEmbeddingProvider(options EmbeddingProviderOptions) (*EmbeddingProvider, error) {
	if strings.TrimSpace(options.Model) == "" {
		return nil, fmt.Errorf("embedding provider model is empty")
	}
	if options.Dimension <= 0 {
		return nil, fmt.Errorf("embedding provider dimension must be positive")
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("listen embedding provider: %w", err)
	}
	provider := &EmbeddingProvider{
		listener:  listener,
		done:      make(chan error, 1),
		model:     options.Model,
		dimension: options.Dimension,
	}
	provider.server = &http.Server{Handler: http.HandlerFunc(provider.handle)}
	go func() {
		serveErr := provider.server.Serve(listener)
		if errors.Is(serveErr, http.ErrServerClosed) {
			serveErr = nil
		}
		provider.done <- serveErr
	}()
	return provider, nil
}

func (provider *EmbeddingProvider) URL() string {
	return "http://" + provider.listener.Addr().String() + "/v1"
}

func (provider *EmbeddingProvider) Close() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	shutdownErr := provider.server.Shutdown(ctx)
	select {
	case serveErr := <-provider.done:
		return errors.Join(shutdownErr, serveErr)
	case <-ctx.Done():
		return errors.Join(shutdownErr, context.Cause(ctx))
	}
}

func (provider *EmbeddingProvider) handle(writer http.ResponseWriter, request *http.Request) {
	switch {
	case strings.HasSuffix(request.URL.Path, "/models"):
		writer.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(writer).Encode(struct {
			Object string                `json:"object"`
			Data   []embeddingModelEntry `json:"data"`
		}{
			Object: "list",
			Data: []embeddingModelEntry{{
				ID: provider.model, Object: "model", OwnedBy: "sandbox",
			}},
		})
	case strings.HasSuffix(request.URL.Path, "/embeddings"):
		provider.handleEmbeddings(writer, request)
	default:
		http.Error(writer, "unsupported embedding route", http.StatusNotFound)
	}
}

type embeddingModelEntry struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	OwnedBy string `json:"owned_by"`
}

type embeddingRow struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

func (provider *EmbeddingProvider) handleEmbeddings(
	writer http.ResponseWriter,
	request *http.Request,
) {
	var payload struct {
		Input      json.RawMessage `json:"input"`
		Model      string          `json:"model"`
		Dimensions int             `json:"dimensions"`
	}
	if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
		http.Error(writer, "decode embedding request", http.StatusBadRequest)
		return
	}
	inputs, err := decodeEmbeddingInputs(payload.Input)
	if err != nil {
		http.Error(writer, err.Error(), http.StatusBadRequest)
		return
	}
	dimension := payload.Dimensions
	if dimension == 0 {
		dimension = provider.dimension
	}
	if dimension != provider.dimension {
		http.Error(writer, "embedding dimension does not match provider", http.StatusBadRequest)
		return
	}
	rows := make([]embeddingRow, len(inputs))
	for index, input := range inputs {
		rows[index] = embeddingRow{
			Object:    "embedding",
			Index:     index,
			Embedding: deterministicVector(input, dimension),
		}
	}
	writer.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(writer).Encode(struct {
		Object string         `json:"object"`
		Data   []embeddingRow `json:"data"`
		Model  string         `json:"model"`
	}{
		Object: "list", Data: rows, Model: payload.Model,
	})
}

func decodeEmbeddingInputs(body json.RawMessage) ([]string, error) {
	var inputs []string
	if err := json.Unmarshal(body, &inputs); err == nil && len(inputs) > 0 {
		return inputs, nil
	}
	var input string
	if err := json.Unmarshal(body, &input); err == nil && input != "" {
		return []string{input}, nil
	}
	return nil, fmt.Errorf("embedding input is empty or invalid")
}

func deterministicVector(input string, dimension int) []float32 {
	digest := sha256.Sum256([]byte(input))
	vector := make([]float32, dimension)
	for index := range dimension {
		offset := (index * 4) % len(digest)
		word := binary.LittleEndian.Uint32(digest[offset : offset+4])
		vector[index] = float32(float64(word)/float64(math.MaxUint32)*2 - 1)
	}
	return vector
}

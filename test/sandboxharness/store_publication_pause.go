//go:build restartacceptance || live

package sandboxharness

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/proto"
)

// StorePublicationObservation records the paused request and real backend outcome.
type StorePublicationObservation struct {
	Database         string
	Collection       string
	Method           string
	Phase            StorePublicationPhase
	ResponseReceived bool
	BackendCode      int32
	BackendReason    string
}

// StorePublicationPhase selects a pause before or after backend publication.
type StorePublicationPhase int

const (
	// BeforeStorePublication pauses before forwarding the Upsert request.
	BeforeStorePublication StorePublicationPhase = iota
	// AfterStorePublication pauses after receiving the real backend response.
	AfterStorePublication
)

type storePublicationPause struct {
	database   string
	collection string
	phase      StorePublicationPhase
	observed   chan StorePublicationObservation
	release    chan struct{}
	once       sync.Once
}

// PausePublication suspends real Upsert forwarding at the selected boundary.
// Other methods and targets continue to use the configured backend.
func (proxy *EmbeddingStoreProxy) PausePublication(database string, collection string, phase StorePublicationPhase) (<-chan StorePublicationObservation, func()) {
	pause := &storePublicationPause{
		database: database, collection: collection, phase: phase,
		observed: make(chan StorePublicationObservation, 1), release: make(chan struct{}),
	}
	proxy.mutex.Lock()
	proxy.publication = pause
	proxy.mutex.Unlock()
	var once sync.Once
	return pause.observed, func() { once.Do(func() { close(pause.release) }) }
}

func (pause *storePublicationPause) matches(database string, collection string, metadataDatabase string) bool {
	if database == "" {
		database = metadataDatabase
	}
	return database == pause.database && collection == pause.collection
}

func (pause *storePublicationPause) wait(ctx context.Context, observation StorePublicationObservation) error {
	pause.once.Do(func() { pause.observed <- observation })
	select {
	case <-pause.release:
		return nil
	case <-ctx.Done():
		slog.Warn("live test dependency failed", "err", context.Cause(ctx))
		return fmt.Errorf("publication barrier cancelled: %w", context.Cause(ctx))
	}
}

func (proxy *EmbeddingStoreProxy) forwardPausedPublication(method string, request []byte, frontend grpc.ServerStream, pause *storePublicationPause) error {
	observation := StorePublicationObservation{Database: pause.database, Collection: pause.collection, Method: "Upsert", Phase: pause.phase}
	if pause.phase == BeforeStorePublication {
		if err := pause.wait(frontend.Context(), observation); err != nil {
			return err
		}
		return proxy.relay(method, request, frontend)
	}
	incoming, _ := metadata.FromIncomingContext(frontend.Context())
	ctx := metadata.NewOutgoingContext(frontend.Context(), incoming.Copy())
	var response []byte
	var headers metadata.MD
	var trailers metadata.MD
	err := proxy.backend.Invoke(ctx, method, request, &response, grpc.Header(&headers), grpc.Trailer(&trailers))
	frontend.SetTrailer(trailers)
	if err != nil {
		slog.Warn("live test dependency failed", "err", err)
		return fmt.Errorf("forward publication to backend: %w", err)
	}
	var outcome milvuspb.MutationResult
	if err := proto.Unmarshal(response, &outcome); err != nil {
		slog.Warn("live test dependency failed", "err", err)
		return fmt.Errorf("decode real publication outcome: %w", err)
	}
	observation.ResponseReceived = true
	observation.BackendCode = outcome.GetStatus().GetCode()
	observation.BackendReason = outcome.GetStatus().GetReason()
	if err := pause.wait(frontend.Context(), observation); err != nil {
		return err
	}
	if len(headers) != 0 {
		if err := frontend.SendHeader(headers); err != nil {
			slog.Warn("live test dependency failed", "err", err)
			return fmt.Errorf("forward backend publication headers: %w", err)
		}
	}
	if err := frontend.SendMsg(response); err != nil {
		slog.Warn("live test dependency failed", "err", err)
		return fmt.Errorf("forward backend publication response: %w", err)
	}
	return nil
}

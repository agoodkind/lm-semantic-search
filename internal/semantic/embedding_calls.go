package semantic

import (
	"context"
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/embedding"
	"google.golang.org/grpc/peer"
)

type embeddingCallCounterKey struct{}

type embeddingCallCounter struct {
	completed int32
}

func (service *Service) embedBatchWithProgress(ctx context.Context, texts []string) (embedding.BatchResult, error) {
	peerInfo, _ := peer.FromContext(ctx)
	result, err := service.embedder.EmbedBatch(ctx, texts)
	if err != nil {
		slog.ErrorContext(ctx, "embed provider batch failed", "peer", peerInfo.String(), "err", err)
		return result, fmt.Errorf("embed provider batch: %w", err)
	}
	if counter, found := ctx.Value(embeddingCallCounterKey{}).(*embeddingCallCounter); found {
		counter.completed++
	}
	return result, nil
}

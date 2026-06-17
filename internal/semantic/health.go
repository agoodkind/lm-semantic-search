package semantic

import (
	"context"
	"fmt"
	"strings"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/adapterr"
)

// ProbeHealth actively verifies that the vector store and the embedding endpoint
// both answer right now, so a caller can report whether search can serve a query
// instead of echoing the last job outcome. Both probes are cheap: a Milvus
// ListCollections (metadata only) and an embedder model-list (no embedding, so
// no model capacity consumed). The returned error is adapterr-classified so the
// daemon maps it to the right degraded banner; nil means search is usable now.
//
// A service with no Milvus address is not configured for semantic search rather
// than degraded, so it returns nil (no outage to surface). A configured service
// whose client is not connected returns a store-unavailable outage.
func (service *Service) ProbeHealth(ctx context.Context) error {
	if service == nil || strings.TrimSpace(service.cfg.MilvusAddress) == "" {
		return nil
	}
	if !service.Available() || service.milvus == nil || service.embedder == nil {
		return adapterr.NewMilvusUnavailable(ErrUnavailable)
	}
	if _, err := service.milvus.ListCollections(ctx, milvusclient.NewListCollectionOption()); err != nil {
		return adapterr.NewMilvusUnavailable(fmt.Errorf("probe Milvus: %w", err))
	}
	if err := service.embedder.Health(ctx); err != nil {
		return adapterr.NewEmbedderUnreachable(fmt.Errorf("probe embedder: %w", err))
	}
	return nil
}

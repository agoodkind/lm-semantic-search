package semantic

import (
	"context"
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/collection"
	"google.golang.org/grpc/peer"
)

// BackfillCollectionScalars writes backfill values into the declared columns
// that are null or an empty string on the rows of streamed items, through the
// collection store. It returns the rows that need the backfill: changed counts
// the rows of streamed items, and orphan counts the rest, which it leaves
// unchanged. A dry run counts and writes nothing. A missing collection returns
// ErrCollectionMissing.
func (service *Service) BackfillCollectionScalars(ctx context.Context, collectionName string, backfill collection.ScalarBackfill) (int, int, error) {
	peerInfo, _ := peer.FromContext(ctx)
	if !service.Available() {
		return 0, 0, ErrUnavailable
	}
	hasCollection, err := service.hasCollection(ctx, collectionName, "check Milvus collection "+collectionName)
	if err != nil {
		return 0, 0, err
	}
	if !hasCollection {
		return 0, 0, ErrCollectionMissing
	}
	if err := service.PrepareCollection(ctx, collectionName); err != nil {
		return 0, 0, err
	}
	lease, err := service.AcquireCollection(ctx, collectionName)
	if err != nil {
		return 0, 0, err
	}
	defer lease.Release()
	changed, orphan, err := service.collectionStore().BackfillScalars(ctx, collectionName, backfill)
	if err != nil {
		slog.ErrorContext(ctx, "scalar backfill failed", "collection", collectionName, "changed", changed, "orphan", orphan, "peer", peerInfo.String(), "err", err)
		return changed, orphan, fmt.Errorf("backfill scalars in %s: %w", collectionName, err)
	}
	return changed, orphan, nil
}

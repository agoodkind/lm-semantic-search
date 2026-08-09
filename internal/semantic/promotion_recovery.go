package semantic

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

func (service *Service) recoverInterruptedPromotions(ctx context.Context) error {
	collections, err := service.milvus.ListCollections(
		ctx,
		milvusclient.NewListCollectionOption(),
	)
	if err != nil {
		wrappedErr := fmt.Errorf("list collections for promotion recovery: %w", err)
		slog.ErrorContext(ctx, "list collections for promotion recovery failed", "error", wrappedErr)
		return wrappedErr
	}
	for _, recoveryName := range collections {
		if !isRecoveryCollection(recoveryName) {
			continue
		}
		liveName := strings.TrimSuffix(recoveryName, recoveryCollectionSuffix)
		stagingName := stagingCollectionName(liveName)
		maintenance, maintainErr := service.residency.Maintain(
			ctx,
			liveName,
			stagingName,
			recoveryName,
		)
		if maintainErr != nil {
			return maintainErr
		}
		recoverErr := service.recoverPromotionNames(ctx, liveName, recoveryName)
		maintenance.ReleaseContext(ctx)
		if recoverErr != nil {
			return recoverErr
		}
	}
	return nil
}

func (service *Service) recoverPromotionNames(
	ctx context.Context,
	liveName string,
	recoveryName string,
) error {
	hasLive, err := service.hasCollection(ctx, liveName, "check live collection "+liveName)
	if err != nil {
		return err
	}
	if hasLive {
		return service.dropIfExists(ctx, recoveryName)
	}
	return service.renameCollection(ctx, recoveryName, liveName)
}

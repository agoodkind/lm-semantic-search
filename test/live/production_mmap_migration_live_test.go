//go:build live && production

package live

import (
	"context"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func TestProductionMmapMigration(t *testing.T) {
	requireProductionOptIn(t)
	cfg, err := config.Default()
	if err != nil {
		t.Fatalf("resolve production configuration: %v", err)
	}
	cfg.MilvusDatabase = productionDatabaseName
	cfg.MilvusCollectionIdleTimeoutMS = 0
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	service, err := semantic.NewService(ctx, cfg)
	if err != nil {
		t.Fatalf("construct production semantic service: %v", err)
	}
	t.Cleanup(func() {
		closeContext, closeCancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer closeCancel()
		if err := service.Close(closeContext); err != nil {
			t.Fatalf("close production semantic service: %v", err)
		}
	})
	if !service.Available() {
		t.Fatal("production semantic service is unavailable")
	}
	service.EnsureMmapEnabledAllCollections(ctx)
}

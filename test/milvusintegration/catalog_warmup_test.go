//go:build milvusintegration

package milvusintegration

import (
	"context"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

const (
	catalogWarmupMemory   = "4g"
	catalogWarmupTimeout  = 4 * time.Minute
	catalogWarmupIndex    = "vector"
	catalogWarmupKey      = "warmup"
	catalogWarmupDisabled = "disable"
	catalogWarmupEnabled  = "sync"
	catalogWarmupContent  = "package example\nfunc CatalogWarmup() string { return \"stored vector\" }\n"
)

// TestReuseCatalogDenseIndexWarmupDisabled proves through the public service and a real
// throwaway Milvus that catalog creation and migration disable dense index warmup.
// It uses the real ONNX embedder and checks that migration preserves reuse vectors.
func TestReuseCatalogDenseIndexWarmupDisabled(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, catalogWarmupMemory)
	milvus := dialMilvus(t, stack)
	cfg := integrationConfig(t, stack, nil)
	ctx, cancel := context.WithTimeout(context.Background(), catalogWarmupTimeout)
	defer cancel()

	firstService := startCatalogWarmupService(t, cfg)
	codebasePath := filepath.Join(t.TempDir(), "catalog-warmup")
	chunks := []model.StoredChunk{{
		Content:      catalogWarmupContent,
		RelativePath: "warmup.go",
		StartLine:    1,
		EndLine:      2,
	}}
	if err := firstService.StageReindex(ctx, codebasePath, chunks, semantic.Removal{}, nil, nil, semantic.CodeColumns()); err != nil {
		t.Fatalf("write source chunks with the real embedder: %v", err)
	}
	catalogName := semantic.ReuseCatalogCollectionName(cfg)
	if got := catalogIndexWarmup(ctx, t, milvus, catalogName); got != catalogWarmupDisabled {
		t.Fatalf("created catalog index warmup = %q, want %q", got, catalogWarmupDisabled)
	}
	before, err := firstService.LoadReuseVectorsForContents(ctx, "", chunks)
	if err != nil {
		t.Fatalf("read catalog vectors after create: %v", err)
	}
	if len(before) != len(chunks) {
		t.Fatalf("catalog returned %d vectors after create, want %d", len(before), len(chunks))
	}

	if err := milvus.ReleaseCollection(ctx, milvusclient.NewReleaseCollectionOption(catalogName)); err != nil {
		t.Fatalf("release catalog to restore the earlier warmup policy: %v", err)
	}
	waitForLoadState(t, milvus, catalogName, entity.LoadStateNotLoad)
	if err := milvus.AlterIndexProperties(
		ctx,
		milvusclient.NewAlterIndexPropertiesOption(catalogName, catalogWarmupIndex).
			WithProperty(catalogWarmupKey, catalogWarmupEnabled),
	); err != nil {
		t.Fatalf("restore the earlier catalog warmup policy: %v", err)
	}
	if _, err := milvus.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(catalogName)); err != nil {
		t.Fatalf("reload catalog with the earlier warmup policy: %v", err)
	}
	waitForLoadState(t, milvus, catalogName, entity.LoadStateLoaded)
	if got := catalogIndexWarmup(ctx, t, milvus, catalogName); got != catalogWarmupEnabled {
		t.Fatalf("restored catalog index warmup = %q, want %q", got, catalogWarmupEnabled)
	}

	secondService := startCatalogWarmupService(t, cfg)
	secondService.EnsureMmapEnabledAllCollections(ctx)
	if got := catalogIndexWarmup(ctx, t, milvus, catalogName); got != catalogWarmupDisabled {
		t.Fatalf("migrated catalog index warmup = %q, want %q", got, catalogWarmupDisabled)
	}
	waitForLoadState(t, milvus, catalogName, entity.LoadStateLoaded)
	after, err := secondService.LoadReuseVectorsForContents(ctx, "", chunks)
	if err != nil {
		t.Fatalf("read catalog vectors after migration: %v", err)
	}
	if !reflect.DeepEqual(after, before) {
		t.Fatal("catalog vectors after migration differ from the vectors after create")
	}
}

func startCatalogWarmupService(t *testing.T, cfg config.Config) *semantic.Service {
	t.Helper()
	service, err := semantic.NewService(context.Background(), cfg)
	if err != nil {
		t.Fatalf("create semantic service: %v", err)
	}
	t.Cleanup(func() {
		if closeErr := service.Close(context.Background()); closeErr != nil {
			t.Errorf("close semantic service: %v", closeErr)
		}
	})
	return service
}

func catalogIndexWarmup(
	ctx context.Context,
	t *testing.T,
	milvus *milvusclient.Client,
	catalogName string,
) string {
	t.Helper()
	description, err := milvus.DescribeIndex(
		ctx,
		milvusclient.NewDescribeIndexOption(catalogName, catalogWarmupIndex),
	)
	if err != nil {
		t.Fatalf("describe catalog index %s: %v", catalogWarmupIndex, err)
	}
	return description.Params()[catalogWarmupKey]
}

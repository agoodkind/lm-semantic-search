//go:build milvusintegration

package milvusintegration

import (
	"context"
	"errors"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

const (
	catalogRecoveryMemory  = "4g"
	catalogRecoveryTimeout = 3 * time.Minute
	catalogRecoveryContent = "package example\nfunc CatalogRecovery() string { return \"stored vector\" }\n"
)

// TestCatalogReadRecoversAfterExternalRelease uses a real Milvus release to
// invalidate the service's cached residency, then retries the public reuse API.
func TestCatalogReadRecoversAfterExternalRelease(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, catalogRecoveryMemory)
	milvus := dialMilvus(t, stack)
	cfg := integrationConfig(t, stack, nil)
	service, err := semantic.NewService(context.Background(), cfg)
	if err != nil {
		t.Fatalf("create semantic service: %v", err)
	}
	t.Cleanup(func() {
		if closeErr := service.Close(context.Background()); closeErr != nil {
			t.Errorf("close semantic service: %v", closeErr)
		}
	})
	ctx, cancel := context.WithTimeout(context.Background(), catalogRecoveryTimeout)
	defer cancel()
	codebasePath := filepath.Join(t.TempDir(), "catalog-recovery")
	chunks := []model.StoredChunk{{
		Content:      catalogRecoveryContent,
		RelativePath: "catalog.go",
		StartLine:    1,
		EndLine:      2,
	}}
	if err := service.StageReindex(ctx, codebasePath, chunks, semantic.Removal{}, nil, nil, semantic.CodeColumns()); err != nil {
		t.Fatalf("write source chunks with the real embedder: %v", err)
	}
	before, err := service.LoadReuseVectorsForContents(ctx, "", chunks)
	if err != nil {
		t.Fatalf("read catalog before release: %v", err)
	}
	if len(before) != len(chunks) {
		t.Fatalf("catalog returned %d vectors before release, want %d", len(before), len(chunks))
	}
	catalogName := semantic.ReuseCatalogCollectionName(cfg)
	if err := milvus.ReleaseCollection(ctx, milvusclient.NewReleaseCollectionOption(catalogName)); err != nil {
		t.Fatalf("release isolated catalog: %v", err)
	}
	waitForLoadState(t, milvus, catalogName, entity.LoadStateNotLoad)
	started := time.Now()
	_, err = service.LoadReuseVectorsForContents(ctx, "", chunks)
	if !errors.Is(err, merr.ErrCollectionNotLoaded) {
		t.Fatalf("released catalog error = %v, want the real Milvus readiness response", err)
	}
	if !adapterr.IsInfraFailure(err) || !adapterr.IsTransient(err) {
		t.Fatalf("released catalog error = %v, want a resumable infrastructure failure", err)
	}
	t.Logf("released catalog returned ErrCollectionNotLoaded as %s after %s", adapterr.Code(err), time.Since(started).Round(time.Millisecond))
	started = time.Now()
	after, err := service.LoadReuseVectorsForContents(ctx, "", chunks)
	if err != nil {
		t.Fatalf("reuse retry did not reload the catalog: %v", err)
	}
	if !reflect.DeepEqual(after, before) {
		t.Fatalf("catalog vectors after recovery differ from the vectors before release")
	}
	t.Logf("the reuse retry reloaded the catalog and returned %d unchanged vectors in %s without restarting the service", len(after), time.Since(started).Round(time.Millisecond))
	_, schemaErr := milvus.Query(
		ctx,
		milvusclient.NewQueryOption(catalogName).
			WithFilter("missingCatalogField == 1").
			WithOutputFields("catalogKey"),
	)
	if schemaErr == nil {
		t.Fatal("the invalid field query did not return a schema error")
	}
	if errors.Is(schemaErr, merr.ErrCollectionNotLoaded) || errors.Is(schemaErr, merr.ErrCollectionOnRecovering) || adapterr.IsInfraFailure(schemaErr) {
		t.Fatalf("the invalid field query returned a readiness or infrastructure error: %v", schemaErr)
	}
	t.Logf("the invalid field query retained its schema error: %v", schemaErr)
}

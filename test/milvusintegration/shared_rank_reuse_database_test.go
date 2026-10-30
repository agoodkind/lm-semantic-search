//go:build milvusintegration

package milvusintegration

import (
	"context"
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/index"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"
)

const (
	databaseReuseOther = "rank_reuse_other"
	databaseReuseFlat  = "db_reuse_flat"
	databaseReuseMixed = "db_reuse_mixed"
	databaseReuseRows  = 200
	databaseReuseLimit = 10
)

// dialDatabase connects a test client to one database in the throwaway stack and closes the client during test cleanup.
func dialDatabase(t *testing.T, stack throwawayStack, database string) *milvusclient.Client {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), milvusCallTimeout)
	defer cancel()
	client, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		TelemetryConfig: &milvusclient.TelemetryConfig{Enabled: false},
		Address:         stack.milvusAddress(),
		DBName:          database,
	})
	if err != nil {
		t.Fatalf("dial database %s: %v", database, err)
	}
	t.Cleanup(func() {
		closeCtx, closeCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer closeCancel()
		_ = client.Close(closeCtx)
	})
	return client
}

// fillDatabaseCollection creates a collection pair with a new SharedStore on client, writes 200 rows with vectors from seed, and returns the store. When indexed is true, it flushes both collections and waits for the dense index.
func fillDatabaseCollection(t *testing.T, client *milvusclient.Client, options milvusstore.Options, name string, seed uint64, indexed bool) *milvusstore.SharedStore {
	t.Helper()
	ctx := context.Background()
	store, err := milvusstore.NewShared(client, options)
	if err != nil {
		t.Fatalf("new shared store for %s: %v", name, err)
	}
	if err := store.EnsureCollection(ctx, collection.EnsureRequest{Collection: name, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
		t.Fatalf("ensure %s: %v", name, err)
	}
	vectorsName := milvusstore.VectorCollectionName(name)
	loadSharedCollections(t, client, name, vectorsName)
	generator := rand.New(rand.NewPCG(seed, seed))
	rows := make([]collection.Row, 0, databaseReuseRows)
	for rowIndex := range databaseReuseRows {
		rows = append(rows, reuseRow(fmt.Sprintf("db-%d-%04d", seed, rowIndex), reuseSparseQuery, normalizedVector(generator)))
	}
	if err := store.Upsert(ctx, name, sharedDeclaration, rows); err != nil {
		t.Fatalf("upsert %s: %v", name, err)
	}
	time.Sleep(sharedVisibilityWait)
	if indexed {
		sealAndIndex(t, client, name, false)
		sealAndIndex(t, client, vectorsName, true)
	}
	return store
}

// TestSharedStoreRankReuseKeepsStoresInSeparateDatabasesApart compares scoped and unscoped Rank results for stores with matching collection names in two Milvus databases. The collections use different rows and FLAT or HNSW indexes.
func TestSharedStoreRankReuseKeepsStoresInSeparateDatabasesApart(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, "4g")
	defaultClient := dialMilvus(t, stack)
	ctx := context.Background()
	if err := defaultClient.CreateDatabase(ctx, milvusclient.NewCreateDatabaseOption(databaseReuseOther)); err != nil {
		t.Fatalf("create database %s: %v", databaseReuseOther, err)
	}
	otherClient := dialDatabase(t, stack, databaseReuseOther)
	flatOptions := milvusstore.Options{Hybrid: true, EmbeddingModel: "generated-model", DenseIndex: index.NewFlatIndex(entity.COSINE)}
	hnswOptions := milvusstore.Options{
		Hybrid:            true,
		EmbeddingModel:    "generated-model",
		DenseIndex:        index.NewHNSWIndex(entity.COSINE, 16, 64),
		DenseSearchParams: map[string]string{"ef": "64"},
	}
	defaultFlat := fillDatabaseCollection(t, defaultClient, flatOptions, databaseReuseFlat, sharedSeed+21, false)
	otherFlat := fillDatabaseCollection(t, otherClient, flatOptions, databaseReuseFlat, sharedSeed+22, false)
	defaultMixed := fillDatabaseCollection(t, defaultClient, flatOptions, databaseReuseMixed, sharedSeed+23, false)
	otherMixed := fillDatabaseCollection(t, otherClient, hnswOptions, databaseReuseMixed, sharedSeed+24, true)
	query := normalizedVector(rand.New(rand.NewPCG(sharedSeed+25, sharedSeed+25)))

	for _, pair := range []struct {
		name   string
		first  *milvusstore.SharedStore
		second *milvusstore.SharedStore
	}{{databaseReuseFlat, defaultFlat, otherFlat}, {databaseReuseMixed, defaultMixed, otherMixed}} {
		request := collection.SearchRequest{Collection: pair.name, Query: reuseSparseQuery, Vector: query, Declaration: sharedDeclaration, Limit: databaseReuseLimit}
		firstWant, err := pair.first.Rank(ctx, request)
		if err != nil {
			t.Fatalf("%s: unscoped rank in the default database: %v", pair.name, err)
		}
		secondWant, err := pair.second.Rank(ctx, request)
		if err != nil {
			t.Fatalf("%s: unscoped rank in %s: %v", pair.name, databaseReuseOther, err)
		}
		if slices.Equal(firstWant, secondWant) {
			t.Fatalf("%s: both databases returned the same candidates; the fixture needs different rows in each database", pair.name)
		}
		scope := milvusstore.WithRankReuse(ctx)
		firstGot, err := pair.first.Rank(scope, request)
		if err != nil {
			t.Fatalf("%s: scoped rank in the default database: %v", pair.name, err)
		}
		secondGot, err := pair.second.Rank(scope, request)
		if err != nil {
			t.Fatalf("%s: scoped rank in %s after the default database: %v", pair.name, databaseReuseOther, err)
		}
		if !slices.Equal(firstGot, firstWant) || !slices.Equal(secondGot, secondWant) {
			t.Fatalf("%s: one reuse scope returned results of one database for the other database", pair.name)
		}
	}
	t.Logf("one reuse scope kept FLAT and HNSW collections with the same names in two databases apart")
}

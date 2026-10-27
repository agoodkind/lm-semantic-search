//go:build live

package live

import (
	"context"
	"log/slog"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/semantic/milvusgrpc"
)

const poolProbeDimension = 4

// TestHarnessTeardownKeepsLeasedDatabaseAndDropsCollection checks that
// harness teardown keeps the leased database and drops the probe collection.
// It checks both outcomes through new clients connected to real Milvus.
func TestHarnessTeardownKeepsLeasedDatabaseAndDropsCollection(t *testing.T) {
	var databaseName string
	collectionName := "pool_probe_" + randomID()
	t.Run("harness", func(t *testing.T) {
		harness := newHarness(t)
		databaseName = harness.databaseName
		harness.trackTemporaryCollection(collectionName)
		createContext, cancelCreate := context.WithTimeout(harness.milvusContext, 30*time.Second)
		defer cancelCreate()
		if err := harness.milvus.CreateCollection(
			createContext,
			milvusclient.SimpleCreateCollectionOptions(collectionName, poolProbeDimension),
		); err != nil {
			t.Fatalf("create probe collection %s: %v", collectionName, err)
		}
	})
	if databaseName == "" {
		t.Skip("BLOCKED: the harness subtest did not lease a Milvus database")
	}

	defaultConfig, err := config.Default()
	if err != nil {
		t.Fatalf("resolve live configuration: %v", err)
	}
	operatorMilvus := connectPoolProbeClient(t, defaultConfig, "")
	databaseNames, err := listMilvusDatabases(operatorMilvus)
	if err != nil {
		t.Fatalf("list Milvus databases after teardown: %v", err)
	}
	if !slices.Contains(databaseNames, databaseName) {
		t.Fatalf("teardown dropped leased database %s; databases: %v", databaseName, databaseNames)
	}

	scopedMilvus := connectPoolProbeClient(t, defaultConfig, databaseName)
	listContext, cancelList := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancelList()
	collectionNames, err := scopedMilvus.ListCollections(listContext, milvusclient.NewListCollectionOption())
	if err != nil {
		t.Fatalf("list collections in %s after teardown: %v", databaseName, err)
	}
	if slices.Contains(collectionNames, collectionName) {
		t.Fatalf("teardown left probe collection %s in %s", collectionName, databaseName)
	}
}

// connectPoolProbeClient opens a Milvus client with telemetry off.
// It uses the requested database and closes the client at test cleanup.
func connectPoolProbeClient(t *testing.T, defaultConfig config.Config, databaseName string) *milvusclient.Client {
	t.Helper()
	probeContext := context.Background()
	dialContext, cancelDial := context.WithTimeout(probeContext, 5*time.Second)
	defer cancelDial()
	client, err := milvusclient.New(dialContext, &milvusclient.ClientConfig{
		TelemetryConfig: &milvusclient.TelemetryConfig{Enabled: false},
		Address:         strings.TrimSpace(defaultConfig.MilvusAddress),
		APIKey:          defaultConfig.MilvusToken,
		DBName:          databaseName,
		DialOptions:     milvusgrpc.DialOptions(probeContext, slog.Default(), milvusgrpc.DefaultCallTimeouts()),
	})
	if err != nil {
		t.Fatalf("connect Milvus database %q: %v", databaseName, err)
	}
	t.Cleanup(func() { closeMilvusClient(client) })
	return client
}

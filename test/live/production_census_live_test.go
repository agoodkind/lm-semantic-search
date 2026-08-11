//go:build live && production

package live

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/semantic/milvusgrpc"
)

type productionCensus struct {
	Collections             int            `json:"collections"`
	Loaded                  int            `json:"loaded"`
	Cold                    int            `json:"cold"`
	OtherLoadState          int            `json:"other_load_state"`
	Staging                 int            `json:"staging"`
	Recovery                int            `json:"recovery"`
	ConversationCollections int            `json:"conversation_collections"`
	ConversationDebt        []string       `json:"conversation_debt"`
	MmapMissing             int            `json:"mmap_missing"`
	ProductionLogicalRows   int64          `json:"production_logical_rows"`
	ProductionStatsRows     string         `json:"production_stats_rows"`
	LoadStates              map[string]int `json:"load_states"`
}

func TestProductionCensusAndConversationDebt(t *testing.T) {
	requireProductionOptIn(t)
	productionConversationCollection := requiredProductionEnvironment(
		t,
		"LMS_PRODUCTION_CONVERSATION_COLLECTION",
	)
	cfg, err := config.Default()
	if err != nil {
		t.Fatalf("resolve production configuration: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()
	client, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address:     cfg.MilvusAddress,
		APIKey:      cfg.MilvusToken,
		DBName:      productionDatabaseName,
		DialOptions: milvusgrpc.DialOptions(ctx, slog.Default(), milvusgrpc.DefaultCallTimeouts()),
	})
	if err != nil {
		t.Fatalf("connect production Milvus for census: %v", err)
	}
	t.Cleanup(func() { _ = client.Close(context.Background()) })

	names, err := client.ListCollections(ctx, milvusclient.NewListCollectionOption())
	if err != nil {
		t.Fatalf("list production collections: %v", err)
	}
	slices.Sort(names)
	census := productionCensus{
		Collections:           len(names),
		ConversationDebt:      make([]string, 0),
		LoadStates:            make(map[string]int, len(names)),
		ProductionLogicalRows: -1,
		ProductionStatsRows:   "unknown",
	}
	for _, name := range names {
		collectionCtx, collectionCancel := context.WithTimeout(ctx, 5*time.Minute)
		state, stateErr := client.GetLoadState(
			collectionCtx,
			milvusclient.NewGetLoadStateOption(name),
		)
		if stateErr != nil {
			collectionCancel()
			t.Fatalf("get production load state for %s: %v", name, stateErr)
		}
		census.LoadStates[name] = int(state.State)
		switch state.State {
		case entity.LoadStateLoaded:
			census.Loaded++
		case entity.LoadStateNotLoad:
			census.Cold++
		default:
			census.OtherLoadState++
		}
		if strings.HasSuffix(name, "_stg") {
			census.Staging++
		}
		if strings.Contains(name, "_swap_previous") {
			census.Recovery++
		}
		missingMmap := countMissingMmapTargets(t, collectionCtx, client, name)
		census.MmapMissing += missingMmap
		if missingMmap > 0 {
			t.Logf("MMAP_MISSING collection=%s count=%d", name, missingMmap)
		}
		if !strings.HasPrefix(name, "conv_chunks_") {
			collectionCancel()
			continue
		}
		census.ConversationCollections++
		if probeConversationDebt(t, collectionCtx, client, name, state.State) {
			census.ConversationDebt = append(census.ConversationDebt, name)
		}
		if name == productionConversationCollection {
			census.ProductionLogicalRows = strongRowCount(t, collectionCtx, client, name)
			stats, statsErr := client.GetCollectionStats(
				collectionCtx,
				milvusclient.NewGetCollectionStatsOption(name),
			)
			if statsErr != nil {
				t.Fatalf("read production conversation stats: %v", statsErr)
			}
			census.ProductionStatsRows = stats["row_count"]
		}
		collectionCancel()
	}
	encoded, err := json.Marshal(census)
	if err != nil {
		t.Fatalf("encode production census: %v", err)
	}
	if outputPath := os.Getenv("LMS_CENSUS_OUTPUT"); outputPath != "" {
		if err := os.WriteFile(outputPath, append(encoded, '\n'), 0o600); err != nil {
			t.Fatalf("write production census: %v", err)
		}
	}
	t.Logf("PRODUCTION_CENSUS_JSON=%s", encoded)
	if census.ProductionLogicalRows < 0 {
		t.Fatalf(
			"production conversation collection %q was not measured",
			productionConversationCollection,
		)
	}
	if len(census.ConversationDebt) != 0 {
		t.Fatalf("conversation scalar debt exists: %v", census.ConversationDebt)
	}
}

func probeConversationDebt(
	t *testing.T,
	ctx context.Context,
	client *milvusclient.Client,
	name string,
	originalState entity.LoadStateCode,
) bool {
	t.Helper()
	description, err := client.DescribeCollection(
		ctx,
		milvusclient.NewDescribeCollectionOption(name),
	)
	if err != nil {
		t.Fatalf("describe conversation collection %s: %v", name, err)
	}
	hasProvider := false
	for _, field := range description.Schema.Fields {
		if field.Name == "provider" {
			hasProvider = true
			break
		}
	}
	if !hasProvider {
		return true
	}
	if originalState == entity.LoadStateNotLoad {
		if _, err := client.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(name)); err != nil {
			t.Fatalf("load cold conversation collection %s: %v", name, err)
		}
		waitForClientLoadState(t, ctx, client, name, entity.LoadStateLoaded)
		defer func() {
			if err := client.ReleaseCollection(
				context.Background(),
				milvusclient.NewReleaseCollectionOption(name),
			); err != nil {
				t.Fatalf("restore cold conversation collection %s: %v", name, err)
			}
			waitForClientLoadState(
				t,
				context.Background(),
				client,
				name,
				entity.LoadStateNotLoad,
			)
		}()
	}
	iterator, err := client.QueryIterator(
		ctx,
		milvusclient.NewQueryIteratorOption(name).
			WithBatchSize(1).
			WithFilter("provider is null").
			WithOutputFields("id").
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		t.Fatalf("open null-provider probe for %s: %v", name, err)
	}
	result, err := iterator.Next(ctx)
	if errors.Is(err, io.EOF) {
		return false
	}
	if err != nil {
		t.Fatalf("query null provider in %s: %v", name, err)
	}
	return result.ResultCount > 0
}

func strongRowCount(
	t *testing.T,
	ctx context.Context,
	client *milvusclient.Client,
	name string,
) int64 {
	t.Helper()
	result, err := client.Query(
		ctx,
		milvusclient.NewQueryOption(name).
			WithOutputFields("count(*)").
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		t.Fatalf("strong count %s: %v", name, err)
	}
	countColumn := result.GetColumn("count(*)")
	if countColumn == nil {
		t.Fatalf("strong count %s returned no count column", name)
	}
	count, err := countColumn.GetAsInt64(0)
	if err != nil {
		t.Fatalf("read strong count %s: %v", name, err)
	}
	return count
}

func countMissingMmapTargets(
	t *testing.T,
	ctx context.Context,
	client *milvusclient.Client,
	name string,
) int {
	t.Helper()
	missing := 0
	for _, target := range snapshotMmapTargetsForClient(t, ctx, client, name) {
		if target.value != "true" {
			missing++
		}
	}
	return missing
}

func waitForClientLoadState(
	t *testing.T,
	ctx context.Context,
	client *milvusclient.Client,
	name string,
	want entity.LoadStateCode,
) {
	t.Helper()
	deadline := time.Now().Add(90 * time.Second)
	for time.Now().Before(deadline) {
		state, err := client.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(name))
		if err != nil {
			t.Fatalf("get load state for %s: %v", name, err)
		}
		if state.State == want {
			return
		}
		time.Sleep(250 * time.Millisecond)
	}
	t.Fatalf("collection %s did not reach load state %v", name, want)
}

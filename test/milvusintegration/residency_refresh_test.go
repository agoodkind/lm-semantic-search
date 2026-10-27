//go:build milvusintegration

package milvusintegration

import (
	"context"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/internal/metrics"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func TestImportedLoadingResidencyRefreshesWithoutDaemonLoad(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, loadCapMilvusMemory)
	milvus := dialMilvus(t, stack)
	const name = "lmstest_imported_loading"
	seedCollection(t, milvus, seedSpec{name: name, rows: loadCapRows, dimension: loadCapDimension, batchRows: loadCapBatchRows})
	cfg := integrationConfig(t, stack, map[string]string{
		"CLAUDE_CONTEXT_MILVUS_COLLECTION_IDLE_TIMEOUT_MS": "1000",
	})
	ctx, cancel := context.WithTimeout(context.Background(), loadStateWait)
	defer cancel()
	if _, err := milvus.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(name)); err != nil {
		t.Fatalf("start external collection load: %v", err)
	}
	if got := loadState(t, milvus, name).State; got != entity.LoadStateLoading {
		t.Fatalf("external load did not expose Loading before service startup: %v", got)
	}
	recorder := &milvusCallRecorder{}
	service := newService(t, cfg, recorder)
	deadline := time.Now().Add(30 * time.Second)
	for {
		entry, found := importedResidency(service.ResidencySnapshot(), name)
		if found && entry.State == "loading" {
			if entry.Loading || entry.Transitioning {
				t.Fatalf("imported state reports a daemon load: %+v", entry)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("service did not import the real Loading state: %+v", entry)
		}
		time.Sleep(loadStatePoll)
	}
	waitForLoadState(t, milvus, name, entity.LoadStateLoaded)
	for {
		entry, _ := importedResidency(service.ResidencySnapshot(), name)
		if entry.State == "ready" {
			if !entry.TimerArmed {
				t.Fatalf("observed ready collection omitted its idle timer: %+v", entry)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("Milvus completed the external load, but the controller still reports %+v", entry)
		}
		time.Sleep(loadStatePoll)
	}
	if calls := recorder.count("LoadCollection", ""); calls != 0 {
		t.Fatalf("state refresh sent %d LoadCollection requests, want 0", calls)
	}
	if snapshot := metrics.Read(); snapshot.MilvusCollectionsLoading != 0 || snapshot.MilvusCollectionsReady != 1 || snapshot.MilvusCollectionLoadInflight != 0 {
		t.Fatalf("metrics after remote load completion = %+v", snapshot)
	}
	waitForLoadState(t, milvus, name, entity.LoadStateNotLoad)
	for {
		entry, _ := importedResidency(service.ResidencySnapshot(), name)
		if entry.State == "idle" && !entry.Transitioning {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("controller did not finish the idle release: %+v", entry)
		}
		time.Sleep(loadStatePoll)
	}
	t.Log("The service imported Loading, observed Ready without issuing a load, updated the metrics, and released the collection after its idle timer.")
}

func importedResidency(snapshot semantic.ResidencySnapshot, name string) (semantic.ResidencyCollectionSnapshot, bool) {
	for _, entry := range snapshot.Collections {
		if entry.Collection == name {
			return entry, true
		}
	}
	return semantic.ResidencyCollectionSnapshot{}, false
}

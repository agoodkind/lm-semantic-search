//go:build milvusintegration

package milvusintegration

import (
	"context"
	"fmt"
	"strconv"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

const (
	loadCap             = 2
	loadCapCollections  = 6
	loadCapRows         = 20_000
	loadCapDimension    = 1024
	loadCapBatchRows    = 2_000
	loadCapMilvusMemory = "4g"
)

// After a Milvus restore the daemon asked for over a hundred collection loads
// within minutes and Milvus ran out of memory. This proves the cap against a
// real Milvus: six cold collections acquired at once under a cap of two never
// have more than two loading at the same instant, as Milvus itself reports
// through get_load_state sampled every 25ms, every caller still ends with a
// lease, and every collection ends loaded.
func TestConcurrentLoadsAreCappedInMilvus(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, loadCapMilvusMemory)
	milvus := dialMilvus(t, stack)

	names := make([]string, 0, loadCapCollections)
	for i := range loadCapCollections {
		name := fmt.Sprintf("lmstest_loadcap_%d", i)
		names = append(names, name)
		seedCollection(t, milvus, seedSpec{name: name, rows: loadCapRows, dimension: loadCapDimension, batchRows: loadCapBatchRows, flushEvery: 0})
	}

	recorder := &milvusCallRecorder{}
	cfg := integrationConfig(t, stack, map[string]string{
		"CLAUDE_CONTEXT_MILVUS_MAX_CONCURRENT_COLLECTION_LOADS": strconv.Itoa(loadCap),
	})
	if cfg.MilvusMaxConcurrentCollectionLoads != loadCap {
		t.Fatalf("resolved MilvusMaxConcurrentCollectionLoads = %d, want %d", cfg.MilvusMaxConcurrentCollectionLoads, loadCap)
	}
	service := newService(t, cfg, recorder)

	sampler := startLoadStateSampler(t, milvus, names)
	type outcome struct {
		name  string
		lease semantic.CollectionLease
		err   error
	}
	outcomes := make(chan outcome, len(names))
	started := time.Now()
	for _, name := range names {
		go func() {
			lease, err := service.AcquireCollection(context.Background(), name)
			outcomes <- outcome{name: name, lease: lease, err: err}
		}()
	}
	leases := make([]semantic.CollectionLease, 0, len(names))
	for range names {
		select {
		case result := <-outcomes:
			if result.err != nil {
				t.Fatalf("AcquireCollection(%s) returned error: %v", result.name, result.err)
			}
			leases = append(leases, result.lease)
		case <-time.After(loadStateWait):
			t.Fatal("an acquire did not finish within the wait")
		}
	}
	t.Logf("all %d acquires returned leases in %s", len(names), time.Since(started).Round(time.Millisecond))
	sampler.finish()

	peak, busy, trace := sampler.peakLoading()
	if busy == 0 {
		t.Fatal("the sampler never saw a collection loading, so it cannot vouch for the cap; slow the loads down or sample faster")
	}
	if peak > loadCap {
		t.Fatalf("Milvus reported %d collections loading at once, want at most %d:\n%s", peak, loadCap, trace)
	}
	t.Logf("Milvus never reported more than %d collections loading at once across %d samples (%d saw a load in flight):\n%s", peak, len(sampler.snapshot()), busy, trace)

	for _, name := range names {
		if state := loadState(t, milvus, name); state.State != entity.LoadStateLoaded {
			t.Fatalf("collection %s load state = %v after every acquire returned, want loaded", name, state.State)
		}
	}
	if calls := recorder.count("LoadCollection", ""); calls != len(names) {
		t.Fatalf("daemon sent %d LoadCollection requests, want one per collection (%d)", calls, len(names))
	}
	for _, lease := range leases {
		lease.Release()
	}
}

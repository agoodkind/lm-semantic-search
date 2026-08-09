package metrics

import (
	"expvar"
	"testing"
)

func TestRegisterPublishesMilvusCollectionMetrics(t *testing.T) {
	Register()
	wantNames := []string{
		"milvus_collection_loads_total",
		"milvus_collection_load_failures_total",
		"milvus_collection_load_wait_timeouts_total",
		"milvus_collection_load_inflight",
		"milvus_collection_load_latency_ms_sum",
		"milvus_collection_unloads_total",
		"milvus_collection_unload_failures_total",
		"milvus_collection_unload_skipped_in_use_total",
		"milvus_collection_unload_latency_ms_sum",
		"milvus_collection_leases_active",
		"milvus_collections_idle",
		"milvus_collections_loading",
		"milvus_collections_ready",
		"milvus_mmap_migrations_total",
		"milvus_mmap_migration_failures_total",
	}
	for _, name := range wantNames {
		if expvar.Get(expvarPrefix+name) == nil {
			t.Errorf("expvar missing %q", expvarPrefix+name)
		}
	}
}

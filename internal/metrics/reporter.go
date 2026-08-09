package metrics

import (
	"context"
	"fmt"
	"log/slog"
	"runtime"
	"time"
)

// reportMessage is the slog message that every periodic counter line
// carries; consumers select on it to scrape the aggregate counters.
const reportMessage = "daemon.perf_counters"

// StartReporter launches a panic-recovered goroutine that emits one
// [slog.InfoContext] line tagged reportMessage every interval, carrying
// every [Snapshot] field plus runtime gauges sampled from a single
// [runtime.ReadMemStats]. An interval of zero or less returns
// immediately without starting a goroutine, leaving the expvar surface
// as the only live view. The goroutine stops when ctx is cancelled.
func StartReporter(ctx context.Context, interval time.Duration) {
	if interval <= 0 {
		return
	}
	// The recover lives in the launching func literal rather than inside
	// runReporter so a panic on any path through the goroutine, including
	// the loop setup, is contained. This package cannot import
	// internal/daemon, so the recover is inlined here.
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.ErrorContext(ctx, "daemon.perf_counters.panic",
					"err", fmt.Errorf("perf reporter panic: %v", recovered))
			}
		}()
		runReporter(ctx, interval)
	}()
}

// runReporter drives the reporter ticker until ctx is cancelled. The
// caller in StartReporter installs the panic recover around this loop.
func runReporter(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			emit(ctx)
		}
	}
}

// emit logs one counter snapshot alongside runtime gauges. A single
// [runtime.ReadMemStats] feeds the heap and GC fields so the gauges are
// internally consistent.
func emit(ctx context.Context) {
	snapshot := Read()
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)

	slog.LogAttrs(ctx, slog.LevelInfo, reportMessage,
		slog.Int64("embed_batches_total", snapshot.EmbedBatchesTotal),
		slog.Int64("embed_batches_failed", snapshot.EmbedBatchesFailed),
		slog.Int64("embed_vectors_total", snapshot.EmbedVectorsTotal),
		slog.Int64("embed_latency_ms_sum", snapshot.EmbedLatencyMSSum),
		slog.Int64("embed_inflight", snapshot.EmbedInflight),
		slog.Int64("embed_chunks_reused_total", snapshot.EmbedChunksReusedTotal),
		slog.Int64("embed_inputs_refused_empty", snapshot.EmbedInputsRefusedEmpty),
		slog.Int64("converge_upsert_total", snapshot.ConvergeUpsertTotal),
		slog.Int64("converge_remove_total", snapshot.ConvergeRemoveTotal),
		slog.Int64("converge_copy_chunks_total", snapshot.ConvergeCopyChunksTotal),
		slog.Int64("sweep_runs_total", snapshot.SweepRunsTotal),
		slog.Int64("sweep_changed_total", snapshot.SweepChangedTotal),
		slog.Int64("sync_skipped_inflight_total", snapshot.SyncSkippedInflightTotal),
		slog.Int64("jobs_completed_total", snapshot.JobsCompletedTotal),
		slog.Int64("jobs_failed_total", snapshot.JobsFailedTotal),
		slog.Int64("jobs_cancelled_total", snapshot.JobsCancelledTotal),
		slog.Int64("boot_resumes_total", snapshot.BootResumesTotal),
		slog.Int64("jobs_active", snapshot.JobsActive),
		slog.Int64("milvus_collection_loads_total", snapshot.MilvusCollectionLoadsTotal),
		slog.Int64("milvus_collection_load_failures_total", snapshot.MilvusCollectionLoadFailuresTotal),
		slog.Int64("milvus_collection_load_wait_timeouts_total", snapshot.MilvusCollectionLoadWaitTimeoutsTotal),
		slog.Int64("milvus_collection_load_inflight", snapshot.MilvusCollectionLoadInflight),
		slog.Int64("milvus_collection_load_latency_ms_sum", snapshot.MilvusCollectionLoadLatencyMSSum),
		slog.Int64("milvus_collection_unloads_total", snapshot.MilvusCollectionUnloadsTotal),
		slog.Int64("milvus_collection_unload_failures_total", snapshot.MilvusCollectionUnloadFailuresTotal),
		slog.Int64("milvus_collection_unload_skipped_in_use_total", snapshot.MilvusCollectionUnloadSkippedInUseTotal),
		slog.Int64("milvus_collection_unload_latency_ms_sum", snapshot.MilvusCollectionUnloadLatencyMSSum),
		slog.Int64("milvus_collection_leases_active", snapshot.MilvusCollectionLeasesActive),
		slog.Int64("milvus_collections_idle", snapshot.MilvusCollectionsIdle),
		slog.Int64("milvus_collections_loading", snapshot.MilvusCollectionsLoading),
		slog.Int64("milvus_collections_ready", snapshot.MilvusCollectionsReady),
		slog.Int64("milvus_mmap_migrations_total", snapshot.MilvusMmapMigrationsTotal),
		slog.Int64("milvus_mmap_migration_failures_total", snapshot.MilvusMmapMigrationFailuresTotal),
		slog.Int("num_goroutine", runtime.NumGoroutine()),
		slog.Uint64("heap_alloc_bytes", mem.HeapAlloc),
		slog.Uint64("heap_inuse_bytes", mem.HeapInuse),
		slog.Uint64("num_gc", uint64(mem.NumGC)),
	)
}

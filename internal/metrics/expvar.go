package metrics

import (
	"expvar"
	"strconv"
	"sync"
	"sync/atomic"
)

// expvarPrefix namespaces every published variable so a process that
// hosts other expvar producers keeps the daemon counters grouped.
const expvarPrefix = "claude_contextd."

// registerOnce guards Register so repeated calls do not trip expvar's
// duplicate-publish panic.
var registerOnce sync.Once

// counterVar adapts a live atomic counter to the [expvar.Var] interface so
// each /debug/vars scrape reads the current value through String without
// routing the load through an adapter whose signature returns any.
type counterVar struct {
	value *atomic.Int64
}

// String renders the counter's current value as a base-10 integer. expvar
// calls it on every scrape, so the published value stays live without a
// copy through a Snapshot.
func (v counterVar) String() string {
	return strconv.FormatInt(v.value.Load(), 10)
}

// Register publishes every counter into expvar under the
// claude_contextd. prefix. Each variable is a counterVar over the live
// atomic, so reads always reflect the current value without copying
// through a Snapshot. It is safe to call multiple times; only the first
// call publishes. The publish is guarded by a [sync.Once].
func Register() {
	registerOnce.Do(publish)
}

// publish does the one-time expvar wiring behind the [sync.Once].
func publish() {
	expvar.Publish(expvarPrefix+"embed_batches_total", counterVar{value: &embedBatchesTotal})
	expvar.Publish(expvarPrefix+"embed_batches_failed", counterVar{value: &embedBatchesFailed})
	expvar.Publish(expvarPrefix+"embed_vectors_total", counterVar{value: &embedVectorsTotal})
	expvar.Publish(expvarPrefix+"embed_latency_ms_sum", counterVar{value: &embedLatencyMSSum})
	expvar.Publish(expvarPrefix+"embed_inflight", counterVar{value: &embedInflight})
	expvar.Publish(expvarPrefix+"converge_upsert_total", counterVar{value: &convergeUpsertTotal})
	expvar.Publish(expvarPrefix+"converge_remove_total", counterVar{value: &convergeRemoveTotal})
	expvar.Publish(expvarPrefix+"converge_copy_chunks_total", counterVar{value: &convergeCopyChunksTotal})
	expvar.Publish(expvarPrefix+"sweep_runs_total", counterVar{value: &sweepRunsTotal})
	expvar.Publish(expvarPrefix+"sweep_changed_total", counterVar{value: &sweepChangedTotal})
	expvar.Publish(expvarPrefix+"sync_skipped_inflight_total", counterVar{value: &syncSkippedInflightTotal})
	expvar.Publish(expvarPrefix+"jobs_completed_total", counterVar{value: &jobsCompletedTotal})
	expvar.Publish(expvarPrefix+"jobs_failed_total", counterVar{value: &jobsFailedTotal})
	expvar.Publish(expvarPrefix+"jobs_cancelled_total", counterVar{value: &jobsCancelledTotal})
	expvar.Publish(expvarPrefix+"boot_resumes_total", counterVar{value: &bootResumesTotal})
	expvar.Publish(expvarPrefix+"jobs_active", counterVar{value: &jobsActive})
}

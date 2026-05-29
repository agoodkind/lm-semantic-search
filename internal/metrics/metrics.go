// Package metrics holds process-wide aggregate performance counters for
// the daemon. Per-call timing belongs to [goodkind.io/claude-context-go/internal/spans];
// this package is strictly aggregate counters surfaced through expvar
// (see expvar.go) and a periodic slog line (see reporter.go). The
// counter identifiers double as expvar keys and slog field keys.
package metrics

import (
	"sync/atomic"
	"time"
)

// Counters live as package-level atomics so increment call sites stay
// lock-free and the expvar publishers can read the live values without
// copying. The grouping mirrors the daemon subsystems that own them.
var (
	embedBatchesTotal  atomic.Int64
	embedBatchesFailed atomic.Int64
	embedVectorsTotal  atomic.Int64
	embedLatencyMSSum  atomic.Int64
	embedInflight      atomic.Int64

	convergeUpsertTotal     atomic.Int64
	convergeRemoveTotal     atomic.Int64
	convergeCopyChunksTotal atomic.Int64

	sweepRunsTotal    atomic.Int64
	sweepChangedTotal atomic.Int64

	syncSkippedInflightTotal atomic.Int64

	jobsCompletedTotal atomic.Int64
	jobsFailedTotal    atomic.Int64
	jobsCancelledTotal atomic.Int64
	bootResumesTotal   atomic.Int64
	jobsActive         atomic.Int64
)

// Snapshot is a point-in-time copy of every counter. The field names map
// one-to-one onto the counter identifiers used as expvar keys and slog
// field keys. It carries no maps or interface values so callers can
// compare two snapshots field-by-field in deterministic tests.
type Snapshot struct {
	EmbedBatchesTotal  int64
	EmbedBatchesFailed int64
	EmbedVectorsTotal  int64
	EmbedLatencyMSSum  int64
	EmbedInflight      int64

	ConvergeUpsertTotal     int64
	ConvergeRemoveTotal     int64
	ConvergeCopyChunksTotal int64

	SweepRunsTotal    int64
	SweepChangedTotal int64

	SyncSkippedInflightTotal int64

	JobsCompletedTotal int64
	JobsFailedTotal    int64
	JobsCancelledTotal int64
	BootResumesTotal   int64
	JobsActive         int64
}

// EmbedBatchStarted records that an embedding batch entered flight,
// raising the embed_inflight gauge.
func EmbedBatchStarted() {
	embedInflight.Add(1)
}

// EmbedBatchDone records that an embedding batch left flight. It lowers
// the gauge, counts the batch and its vectors, accumulates latency in
// milliseconds, and counts the batch as failed when failed is true. The
// elapsed duration is passed in so this package depends on neither
// internal/daemon nor internal/clock.
func EmbedBatchDone(vectors int, elapsed time.Duration, failed bool) {
	embedInflight.Add(-1)
	embedBatchesTotal.Add(1)
	embedVectorsTotal.Add(int64(vectors))
	embedLatencyMSSum.Add(elapsed.Milliseconds())
	if failed {
		embedBatchesFailed.Add(1)
	}
}

// ConvergeUpsert counts one converge upsert operation.
func ConvergeUpsert() {
	convergeUpsertTotal.Add(1)
}

// ConvergeRemove counts one converge remove operation.
func ConvergeRemove() {
	convergeRemoveTotal.Add(1)
}

// ConvergeCopyChunks counts one converge copy-chunks fast-path operation.
func ConvergeCopyChunks() {
	convergeCopyChunksTotal.Add(1)
}

// SweepRan counts one reverse-reconcile sweep, also counting it as a
// changed sweep when changed is true.
func SweepRan(changed bool) {
	sweepRunsTotal.Add(1)
	if changed {
		sweepChangedTotal.Add(1)
	}
}

// SyncSkippedInflight counts one sync request skipped because a matching
// job was already in flight.
func SyncSkippedInflight() {
	syncSkippedInflightTotal.Add(1)
}

// JobStarted raises the jobs_active gauge as a job begins.
func JobStarted() {
	jobsActive.Add(1)
}

// JobFinished lowers the jobs_active gauge as a job ends, regardless of
// terminal state.
func JobFinished() {
	jobsActive.Add(-1)
}

// JobCompleted counts one job that reached the completed state.
func JobCompleted() {
	jobsCompletedTotal.Add(1)
}

// JobFailed counts one job that reached the failed state.
func JobFailed() {
	jobsFailedTotal.Add(1)
}

// JobCancelled counts one job that reached the cancelled state.
func JobCancelled() {
	jobsCancelledTotal.Add(1)
}

// JobResumed counts one job resumed from persisted state at daemon boot.
func JobResumed() {
	bootResumesTotal.Add(1)
}

// Read returns a consistent-enough snapshot of every counter. Each load
// is individually atomic; the snapshot is not a single transactional
// read, which is acceptable for monitoring counters.
func Read() Snapshot {
	return Snapshot{
		EmbedBatchesTotal:        embedBatchesTotal.Load(),
		EmbedBatchesFailed:       embedBatchesFailed.Load(),
		EmbedVectorsTotal:        embedVectorsTotal.Load(),
		EmbedLatencyMSSum:        embedLatencyMSSum.Load(),
		EmbedInflight:            embedInflight.Load(),
		ConvergeUpsertTotal:      convergeUpsertTotal.Load(),
		ConvergeRemoveTotal:      convergeRemoveTotal.Load(),
		ConvergeCopyChunksTotal:  convergeCopyChunksTotal.Load(),
		SweepRunsTotal:           sweepRunsTotal.Load(),
		SweepChangedTotal:        sweepChangedTotal.Load(),
		SyncSkippedInflightTotal: syncSkippedInflightTotal.Load(),
		JobsCompletedTotal:       jobsCompletedTotal.Load(),
		JobsFailedTotal:          jobsFailedTotal.Load(),
		JobsCancelledTotal:       jobsCancelledTotal.Load(),
		BootResumesTotal:         bootResumesTotal.Load(),
		JobsActive:               jobsActive.Load(),
	}
}

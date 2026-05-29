package metrics

import (
	"testing"
	"time"
)

// reset zeroes every counter so a test starts from a known baseline. It
// lives in the test file because counter zeroing is only meaningful for
// deterministic tests, not for the running daemon.
func reset() {
	embedBatchesTotal.Store(0)
	embedBatchesFailed.Store(0)
	embedVectorsTotal.Store(0)
	embedLatencyMSSum.Store(0)
	embedInflight.Store(0)
	convergeUpsertTotal.Store(0)
	convergeRemoveTotal.Store(0)
	convergeCopyChunksTotal.Store(0)
	sweepRunsTotal.Store(0)
	sweepChangedTotal.Store(0)
	syncSkippedInflightTotal.Store(0)
	jobsCompletedTotal.Store(0)
	jobsFailedTotal.Store(0)
	jobsCancelledTotal.Store(0)
	bootResumesTotal.Store(0)
	jobsActive.Store(0)
}

func TestEmbedBatchStartedAndDoneTrackInflight(t *testing.T) {
	reset()

	EmbedBatchStarted()
	if got := Read().EmbedInflight; got != 1 {
		t.Fatalf("EmbedInflight after start = %d, want 1", got)
	}

	EmbedBatchDone(7, 25*time.Millisecond, false)
	after := Read()
	if after.EmbedInflight != 0 {
		t.Fatalf("EmbedInflight after done = %d, want 0", after.EmbedInflight)
	}
	if after.EmbedBatchesTotal != 1 {
		t.Fatalf("EmbedBatchesTotal = %d, want 1", after.EmbedBatchesTotal)
	}
	if after.EmbedVectorsTotal != 7 {
		t.Fatalf("EmbedVectorsTotal = %d, want 7", after.EmbedVectorsTotal)
	}
	if after.EmbedLatencyMSSum != 25 {
		t.Fatalf("EmbedLatencyMSSum = %d, want 25", after.EmbedLatencyMSSum)
	}
	if after.EmbedBatchesFailed != 0 {
		t.Fatalf("EmbedBatchesFailed = %d, want 0", after.EmbedBatchesFailed)
	}
}

func TestEmbedBatchDoneFailedCounts(t *testing.T) {
	reset()

	EmbedBatchStarted()
	EmbedBatchDone(0, 0, true)
	after := Read()
	if after.EmbedBatchesFailed != 1 {
		t.Fatalf("EmbedBatchesFailed = %d, want 1", after.EmbedBatchesFailed)
	}
	if after.EmbedBatchesTotal != 1 {
		t.Fatalf("EmbedBatchesTotal = %d, want 1", after.EmbedBatchesTotal)
	}
}

// fieldDelta names a counter, the increment call that should move it,
// and the single Snapshot field it is allowed to touch.
type fieldDelta struct {
	name   string
	call   func()
	field  string
	expect int64
}

func TestEachIncrementMovesExactlyItsCounter(t *testing.T) {
	cases := []fieldDelta{
		{"ConvergeUpsert", ConvergeUpsert, "ConvergeUpsertTotal", 1},
		{"ConvergeRemove", ConvergeRemove, "ConvergeRemoveTotal", 1},
		{"ConvergeCopyChunks", ConvergeCopyChunks, "ConvergeCopyChunksTotal", 1},
		{"SyncSkippedInflight", SyncSkippedInflight, "SyncSkippedInflightTotal", 1},
		{"JobCompleted", JobCompleted, "JobsCompletedTotal", 1},
		{"JobFailed", JobFailed, "JobsFailedTotal", 1},
		{"JobCancelled", JobCancelled, "JobsCancelledTotal", 1},
		{"JobResumed", JobResumed, "BootResumesTotal", 1},
		{"JobStarted", JobStarted, "JobsActive", 1},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			reset()
			before := Read()
			tc.call()
			after := Read()
			assertOnlyDelta(t, before, after, tc.field, tc.expect)
		})
	}
}

func TestSweepRanChangedFlag(t *testing.T) {
	reset()

	SweepRan(false)
	mid := Read()
	if mid.SweepRunsTotal != 1 {
		t.Fatalf("SweepRunsTotal = %d, want 1", mid.SweepRunsTotal)
	}
	if mid.SweepChangedTotal != 0 {
		t.Fatalf("SweepChangedTotal = %d, want 0", mid.SweepChangedTotal)
	}

	SweepRan(true)
	after := Read()
	if after.SweepRunsTotal != 2 {
		t.Fatalf("SweepRunsTotal = %d, want 2", after.SweepRunsTotal)
	}
	if after.SweepChangedTotal != 1 {
		t.Fatalf("SweepChangedTotal = %d, want 1", after.SweepChangedTotal)
	}
}

func TestJobFinishedLowersGauge(t *testing.T) {
	reset()

	JobStarted()
	JobFinished()
	if got := Read().JobsActive; got != 0 {
		t.Fatalf("JobsActive after start+finish = %d, want 0", got)
	}
}

func TestResetZeroesEveryCounter(t *testing.T) {
	reset()

	EmbedBatchStarted()
	EmbedBatchDone(3, time.Millisecond, true)
	ConvergeUpsert()
	ConvergeRemove()
	ConvergeCopyChunks()
	SweepRan(true)
	SyncSkippedInflight()
	JobStarted()
	JobCompleted()
	JobFailed()
	JobCancelled()
	JobResumed()

	reset()
	if got := Read(); got != (Snapshot{}) {
		t.Fatalf("Read after Reset = %+v, want zero Snapshot", got)
	}
}

// snapshotFields flattens a Snapshot into name/value pairs so a test can
// assert on one field by name and hold every other field steady.
func snapshotFields(s Snapshot) map[string]int64 {
	return map[string]int64{
		"EmbedBatchesTotal":        s.EmbedBatchesTotal,
		"EmbedBatchesFailed":       s.EmbedBatchesFailed,
		"EmbedVectorsTotal":        s.EmbedVectorsTotal,
		"EmbedLatencyMSSum":        s.EmbedLatencyMSSum,
		"EmbedInflight":            s.EmbedInflight,
		"ConvergeUpsertTotal":      s.ConvergeUpsertTotal,
		"ConvergeRemoveTotal":      s.ConvergeRemoveTotal,
		"ConvergeCopyChunksTotal":  s.ConvergeCopyChunksTotal,
		"SweepRunsTotal":           s.SweepRunsTotal,
		"SweepChangedTotal":        s.SweepChangedTotal,
		"SyncSkippedInflightTotal": s.SyncSkippedInflightTotal,
		"JobsCompletedTotal":       s.JobsCompletedTotal,
		"JobsFailedTotal":          s.JobsFailedTotal,
		"JobsCancelledTotal":       s.JobsCancelledTotal,
		"BootResumesTotal":         s.BootResumesTotal,
		"JobsActive":               s.JobsActive,
	}
}

// assertOnlyDelta confirms that the named field moved by exactly expect
// and that every other counter held steady between before and after.
func assertOnlyDelta(t *testing.T, before, after Snapshot, field string, expect int64) {
	t.Helper()
	beforeFields := snapshotFields(before)
	afterFields := snapshotFields(after)

	if _, ok := beforeFields[field]; !ok {
		t.Fatalf("unknown field name %q", field)
	}

	for name, beforeValue := range beforeFields {
		delta := afterFields[name] - beforeValue
		if name == field {
			if delta != expect {
				t.Fatalf("target counter %s delta = %d, want %d", name, delta, expect)
			}
			continue
		}
		if delta != 0 {
			t.Errorf("unrelated counter %s moved: %d -> %d", name, beforeValue, afterFields[name])
		}
	}
}

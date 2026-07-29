package daemon

import (
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/metrics"
	"goodkind.io/lm-semantic-search/internal/model"
)

// metricByName finds one assembled metric so a test asserts on the name an
// operator reads rather than on a slice position.
func metricByName(list []*pb.Metric, name string) *pb.Metric {
	for _, metric := range list {
		if metric.GetName() == name {
			return metric
		}
	}
	return nil
}

// TestStatusMetricsCarryCounterNamesAndUnits proves the assembled list uses the
// counter identifiers verbatim, so a name on screen greps the daemon log, and
// that each carries the unit of what its increment call site counts.
func TestStatusMetricsCarryCounterNamesAndUnits(t *testing.T) {
	t.Parallel()

	snapshot := metrics.Snapshot{
		EmbedVectorsTotal:        3946,
		EmbedChunksReusedTotal:   4025,
		ConvergeUpsertTotal:      843,
		SyncSkippedInflightTotal: 1524,
	}
	list := buildStatusMetrics(nil, snapshot, time.Unix(1785156767, 0))

	cases := []struct {
		name  string
		value int64
		unit  string
	}{
		{"embed_vectors_total", 3946, unitVectors},
		{"embed_chunks_reused_total", 4025, unitChunks},
		{"converge_upsert_total", 843, unitPaths},
		{"sync_skipped_inflight_total", 1524, unitRequests},
	}
	for _, testCase := range cases {
		metric := metricByName(list, testCase.name)
		if metric == nil {
			t.Fatalf("%s absent from %d metrics", testCase.name, len(list))
		}
		if metric.GetIntValue() != testCase.value {
			t.Fatalf("%s = %d, want %d", testCase.name, metric.GetIntValue(), testCase.value)
		}
		if metric.GetUnit() != testCase.unit {
			t.Fatalf("%s unit = %q, want %q", testCase.name, metric.GetUnit(), testCase.unit)
		}
	}
}

// TestStatusMetricsSetZeroValuedOneofs proves a zero counter still carries a
// value. protojson omits a plain proto3 scalar at its zero value, so a metric
// whose oneof went unset would be indistinguishable from an absent measurement
// once it reached JSON.
func TestStatusMetricsSetZeroValuedOneofs(t *testing.T) {
	t.Parallel()

	list := buildStatusMetrics(nil, metrics.Snapshot{}, time.Unix(1785156767, 0))
	for _, name := range []string{"embed_batches_failed", "jobs_active", "converge_remove_total"} {
		metric := metricByName(list, name)
		if metric == nil {
			t.Fatalf("%s absent", name)
		}
		if metric.GetValue() == nil {
			t.Fatalf("%s carries no value; zero and absent must stay distinguishable", name)
		}
		if metric.GetIntValue() != 0 {
			t.Fatalf("%s = %d, want 0", name, metric.GetIntValue())
		}
	}
}

// TestStatusMetricsOmitManagerGroupsWithoutAManager proves the counter
// projection stands alone, so a caller with no manager reports no uptime and no
// slot count rather than defaulting to numbers nobody measured.
func TestStatusMetricsOmitManagerGroupsWithoutAManager(t *testing.T) {
	t.Parallel()

	list := buildStatusMetrics(nil, metrics.Snapshot{}, time.Unix(1785156767, 0))
	for _, name := range []string{"uptime_s", "index_slots_total", "codebases_total"} {
		if metricByName(list, name) != nil {
			t.Fatalf("%s present with no manager, want omitted", name)
		}
	}
}

// TestWatcherActivityRowCarriesAbsentJobID proves a file-change row states it
// has no job, rather than carrying an empty id a reader could mistake for one
// the job commands accept.
func TestWatcherActivityRowCarriesAbsentJobID(t *testing.T) {
	t.Parallel()

	row := watcherActivityRow(WatcherActivity{
		CodebaseID:   "cb-1",
		State:        WatcherStateQueued,
		PendingPaths: 12,
	}, "/Users/agoodkind/Sites/configs")

	jobID := metricByName(row.GetMetrics(), "job_id")
	if jobID == nil {
		t.Fatal("job_id absent from the row entirely; it must be present with no value")
	}
	if jobID.GetValue() != nil {
		t.Fatalf("job_id carries a value %+v, want none", jobID.GetValue())
	}

	source := metricByName(row.GetMetrics(), "source")
	if source.GetStringValue() != activitySourceWatcher {
		t.Fatalf("source = %q, want %q", source.GetStringValue(), activitySourceWatcher)
	}

	pending := metricByName(row.GetMetrics(), "pending_paths")
	if pending.GetIntValue() != 12 || pending.GetUnit() != unitPaths {
		t.Fatalf("pending_paths = %+v, want 12 paths", pending)
	}

	startedAt := metricByName(row.GetMetrics(), "started_at")
	if startedAt.GetValue() != nil {
		t.Fatalf("started_at carries a value for queued work that has not begun: %+v", startedAt.GetValue())
	}
}

// TestCodebaseMetricsEmitEveryStatusEvenAtZero proves a status with no
// codebases still reports a line. A line that appeared only when its count was
// positive would vanish at zero rather than report zero, which reads as the
// category never having existed, and the changing name set would shift the
// value column of every other line between two reads.
func TestCodebaseMetricsEmitEveryStatusEvenAtZero(t *testing.T) {
	t.Parallel()

	list := buildStatusMetrics(nil, metrics.Snapshot{}, time.Unix(1785156767, 0))
	if metricByName(list, "codebases.status=indexing") != nil {
		t.Fatal("codebase metrics present with no snapshot, want omitted")
	}

	// With no codebases tracked at all, every status still reports zero.
	withManager := codebaseMetrics(nil)
	for _, display := range everyDisplayStatus {
		name := "codebases.status=" + string(display)
		metric := metricByName(withManager, name)
		if metric == nil {
			t.Fatalf("%s absent; a status must report zero rather than vanish", name)
		}
		if metric.GetValue() == nil {
			t.Fatalf("%s carries no value, so zero and absent collapse", name)
		}
		if metric.GetIntValue() != 0 {
			t.Fatalf("%s = %d, want 0", name, metric.GetIntValue())
		}
	}
}

// TestPendingActivityRowNamesTheAbsentJob proves a coalesced request waiting on
// its codebase's active job reports as queued work with no job id, so it is
// visible rather than silently outstanding until it drains.
func TestPendingActivityRowNamesTheAbsentJob(t *testing.T) {
	t.Parallel()

	row := pendingActivityRow(PendingWork{
		CodebaseID:    "cb-1",
		CanonicalPath: "/Users/agoodkind/Sites/configs",
		Operation:     pendingOperationSync,
	})

	jobID := metricByName(row.GetMetrics(), "job_id")
	if jobID == nil || jobID.GetValue() != nil {
		t.Fatalf("job_id = %+v, want present with no value", jobID)
	}
	source := metricByName(row.GetMetrics(), "source")
	if source.GetStringValue() != activitySourcePending {
		t.Fatalf("source = %q, want %q", source.GetStringValue(), activitySourcePending)
	}
	state := metricByName(row.GetMetrics(), "state")
	if state.GetStringValue() != WatcherStateQueued {
		t.Fatalf("state = %q, want %q", state.GetStringValue(), WatcherStateQueued)
	}
}

// TestPendingWorkReportsBothSlotKinds proves both depth-1 slots reach a status
// read, since a code sync and a conversation upsert are stored in separate maps
// and reporting only one would hide the other.
func TestPendingWorkReportsBothSlotKinds(t *testing.T) {
	t.Parallel()

	manager := &Manager{
		codebases: map[string]model.Codebase{
			"cb-code": {ID: "cb-code", CanonicalPath: "/code"},
			"cb-conv": {ID: "cb-conv", CanonicalPath: "/conv"},
		},
		pendingCodeJobs:         map[string]pendingCodeRequest{"cb-code": {}},
		pendingConversationJobs: map[string]conversationJobPayload{"cb-conv": {}},
	}

	pending := manager.PendingWork()
	if len(pending) != 2 {
		t.Fatalf("pending work = %d rows, want 2: %+v", len(pending), pending)
	}
	byID := map[string]PendingWork{}
	for _, entry := range pending {
		byID[entry.CodebaseID] = entry
	}
	if byID["cb-code"].Operation != pendingOperationSync {
		t.Fatalf("code slot operation = %q, want %q", byID["cb-code"].Operation, pendingOperationSync)
	}
	if byID["cb-conv"].Operation != pendingOperationConversationIngest {
		t.Fatalf("conversation slot operation = %q, want %q",
			byID["cb-conv"].Operation, pendingOperationConversationIngest)
	}
	if byID["cb-code"].CanonicalPath != "/code" {
		t.Fatalf("code slot path = %q, want /code", byID["cb-code"].CanonicalPath)
	}
}

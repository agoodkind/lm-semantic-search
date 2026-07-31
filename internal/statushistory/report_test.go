package statushistory

import (
	"compress/gzip"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/model"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func TestBuildUsesOnlyDaemonStateAndReportsExclusiveTime(t *testing.T) {
	t.Parallel()

	stateRoot := t.TempDir()
	now := time.Date(2026, 7, 31, 14, 0, 0, 0, time.UTC)
	start := now.Add(-time.Hour)
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": start.Add(-time.Second), "msg": "daemon.perf_counters",
		"embed_vectors_total": float64(10), "embed_latency_ms_sum": float64(100),
		"embed_batches_total": float64(2), "jobs_active": float64(1),
	})
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": start.Add(30 * time.Minute), "msg": "daemon.perf_counters",
		"embed_vectors_total": float64(14), "embed_latency_ms_sum": float64(200),
		"embed_batches_total": float64(4), "jobs_active": float64(3),
	})
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": now.Add(-time.Minute), "msg": "daemon.span.completed", "span": "daemon.runDeltaSync",
		"span_id": "root", "job_id": "job-1", "duration_ms": float64(100),
	})
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": now.Add(-time.Minute), "msg": "daemon.span.completed", "span": "semantic.replace",
		"span_id": "child", "parent_span_id": "root", "job_id": "job-1", "duration_ms": float64(40),
	})
	completed := now.Add(-time.Minute)
	writeJob(t, filepath.Join(stateRoot, "jobs.jsonl"), model.JobEvent{
		Event: "job_completed", OccurredAt: completed,
		Job: model.Job{ID: "job-1", State: model.JobStateCompleted, StartedAt: completed.Add(-200 * time.Millisecond), CompletedAt: &completed},
	})

	report, err := Build(Input{
		StateRoot: stateRoot,
		Since:     time.Hour,
		Now:       now,
		Status: &pb.GetStatusResponse{
			ReadAt: timestamppb.New(now),
			Daemon: &pb.DaemonIdentity{Pid: 1, StartedAt: timestamppb.New(start.Add(-time.Hour))},
			Metrics: []*pb.Metric{
				{Name: "embed_vectors_total", Unit: "vectors", Value: &pb.Metric_IntValue{IntValue: 15}},
				{Name: "embed_latency_ms_sum", Unit: "ms", Value: &pb.Metric_IntValue{IntValue: 250}},
				{Name: "embed_batches_total", Unit: "batches", Value: &pb.Metric_IntValue{IntValue: 5}},
				{Name: "jobs_active", Unit: "jobs", Value: &pb.Metric_IntValue{IntValue: 2}},
			},
		},
	})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	vectors := metricByName(report, "embed_vectors_total")
	if vectors == nil || vectors.Delta == nil || *vectors.Delta != 5 {
		t.Fatalf("embed_vectors_total = %+v, want delta 5", vectors)
	}
	if vectors.Current == nil || *vectors.Current != 15 {
		t.Fatalf("embed_vectors_total current = %+v, want 15", vectors)
	}
	active := metricByName(report, "jobs_active")
	if active == nil || active.Min == nil || *active.Min != 2 || active.Max == nil || *active.Max != 3 {
		t.Fatalf("jobs_active = %+v, want observed gauge range 2..3", active)
	}
	root := durationByName(report, "daemon.runDeltaSync")
	if root == nil || root.TotalMS == nil || *root.TotalMS != 60 {
		t.Fatalf("root exclusive stage = %+v, want 60ms", root)
	}
	child := durationByName(report, "semantic.replace")
	if child == nil || child.TotalMS == nil || *child.TotalMS != 40 {
		t.Fatalf("child exclusive stage = %+v, want 40ms", child)
	}
	if report.UnattributedDurationMS == nil || *report.UnattributedDurationMS != 100 {
		t.Fatalf("unattributed = %+v, want 100ms", report.UnattributedDurationMS)
	}
	if durationByName(report, "embed_latency") != nil {
		t.Fatalf("aggregate embedding latency appeared in exclusive stages: %+v", report.TimeBreakdown)
	}
	latency := report.EmbeddingLatency
	if latency == nil || latency.TotalMS == nil || *latency.TotalMS != 150 || latency.Calls == nil || *latency.Calls != 3 {
		t.Fatalf("embed latency = %+v, want 150ms across 3 calls", latency)
	}
}

func TestBuildDoesNotTreatCurrentOnlyGaugeAsHistorical(t *testing.T) {
	stateRoot := t.TempDir()
	now := time.Date(2026, 7, 31, 14, 0, 0, 0, time.UTC)
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": now.Add(-time.Hour - time.Second), "msg": "daemon.perf_counters",
		"embed_vectors_total": float64(1), "uptime_s": float64(1),
	})

	report, err := Build(Input{
		StateRoot: stateRoot,
		Since:     time.Hour,
		Now:       now,
		Status: &pb.GetStatusResponse{
			ReadAt: timestamppb.New(now),
			Metrics: []*pb.Metric{
				{Name: "embed_vectors_total", Value: &pb.Metric_IntValue{IntValue: 2}},
				{Name: "uptime_s", Unit: "seconds", Value: &pb.Metric_IntValue{IntValue: 3600}},
			},
		},
	})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	uptime := metricByName(report, "uptime_s")
	if uptime == nil || uptime.Current == nil || *uptime.Current != 3600 {
		t.Fatalf("uptime = %+v, want current 3600", uptime)
	}
	if uptime.Min != nil || uptime.Mean != nil || uptime.Max != nil {
		t.Fatalf("uptime history = %+v, want unavailable summaries", uptime)
	}
}

func TestBuildMarksMalformedHistoryAsIncomplete(t *testing.T) {
	t.Parallel()

	stateRoot := t.TempDir()
	now := time.Date(2026, 7, 31, 14, 0, 0, 0, time.UTC)
	start := now.Add(-time.Hour)
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": start.Add(-time.Second), "msg": "daemon.perf_counters", "embed_vectors_total": float64(1),
	})
	file, err := os.OpenFile(filepath.Join(stateRoot, "logs", "daemon.jsonl"), os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatalf("OpenFile: %v", err)
	}
	if _, err := file.WriteString("not json\n"); err != nil {
		_ = file.Close()
		t.Fatalf("WriteString: %v", err)
	}
	if err := file.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	report, err := Build(Input{
		StateRoot: stateRoot, Since: time.Hour, Now: now,
		Status: &pb.GetStatusResponse{ReadAt: timestamppb.New(now), Metrics: []*pb.Metric{{Name: "embed_vectors_total", Value: &pb.Metric_IntValue{IntValue: 2}}}},
	})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if report.Coverage.Complete {
		t.Fatalf("coverage = %+v, want incomplete after malformed history", report.Coverage)
	}
}

func TestBuildReadsStructuredRecordsLargerThanScannerDefault(t *testing.T) {
	stateRoot := t.TempDir()
	now := time.Date(2026, 7, 31, 14, 0, 0, 0, time.UTC)
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": now.Add(-time.Hour - time.Second), "msg": "daemon.perf_counters",
		"embed_vectors_total": float64(1), "payload": strings.Repeat("x", 70<<10),
	})

	report, err := Build(Input{
		StateRoot: stateRoot,
		Since:     time.Hour,
		Now:       now,
		Status: &pb.GetStatusResponse{
			ReadAt:  timestamppb.New(now),
			Metrics: []*pb.Metric{{Name: "embed_vectors_total", Value: &pb.Metric_IntValue{IntValue: 2}}},
		},
	})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	vectors := metricByName(report, "embed_vectors_total")
	if vectors == nil || vectors.Delta == nil || *vectors.Delta != 1 {
		t.Fatalf("vectors = %+v, want delta 1", vectors)
	}
}

func TestBuildMarksOversizedStructuredRecordIncomplete(t *testing.T) {
	stateRoot := t.TempDir()
	now := time.Date(2026, 7, 31, 14, 0, 0, 0, time.UTC)
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": now.Add(-time.Hour - time.Second), "msg": "daemon.perf_counters",
		"embed_vectors_total": float64(1), "payload": strings.Repeat("x", (4<<20)+1),
	})

	report, err := Build(Input{
		StateRoot: stateRoot,
		Since:     time.Hour,
		Now:       now,
		Status: &pb.GetStatusResponse{
			ReadAt:  timestamppb.New(now),
			Metrics: []*pb.Metric{{Name: "embed_vectors_total", Value: &pb.Metric_IntValue{IntValue: 2}}},
		},
	})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if report.Coverage.Complete {
		t.Fatalf("coverage = %+v, want incomplete", report.Coverage)
	}
}

func TestBuildAnchorsAtStatusReadAtAndExcludesOldSpans(t *testing.T) {
	stateRoot := t.TempDir()
	readAt := time.Date(2026, 7, 31, 14, 0, 0, 0, time.UTC)
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": readAt.Add(-time.Hour - time.Second), "msg": "daemon.perf_counters", "embed_vectors_total": float64(1),
	})
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": readAt.Add(-2 * time.Hour), "msg": "daemon.span.completed", "span": "old", "span_id": "old", "duration_ms": float64(99),
	})
	report, err := Build(Input{StateRoot: stateRoot, Since: time.Hour, Status: &pb.GetStatusResponse{
		ReadAt: timestamppb.New(readAt), Metrics: []*pb.Metric{{Name: "embed_vectors_total", Value: &pb.Metric_IntValue{IntValue: 2}}},
	}})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if !report.Window.End.Equal(readAt) {
		t.Fatalf("window end = %s, want status read %s", report.Window.End, readAt)
	}
	if durationByName(report, "old") != nil {
		t.Fatalf("old span appeared in window: %+v", report.TimeBreakdown)
	}
}

func TestBuildMarksSamplingGapsAndCounterResetIncomplete(t *testing.T) {
	stateRoot := t.TempDir()
	now := time.Date(2026, 7, 31, 14, 0, 0, 0, time.UTC)
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": now.Add(-time.Hour - time.Second), "msg": "daemon.perf_counters", "embed_vectors_total": float64(100),
	})
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": now.Add(-40 * time.Minute), "msg": "daemon.perf_counters", "embed_vectors_total": float64(0),
	})
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": now.Add(-10 * time.Minute), "msg": "daemon.perf_counters", "embed_vectors_total": float64(120),
	})
	report, err := Build(Input{StateRoot: stateRoot, Since: time.Hour, Now: now, Status: &pb.GetStatusResponse{
		ReadAt: timestamppb.New(now), Metrics: []*pb.Metric{{Name: "embed_vectors_total", Value: &pb.Metric_IntValue{IntValue: 130}}},
	}})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if report.Coverage.Complete {
		t.Fatalf("coverage = %+v, want incomplete after gaps and reset", report.Coverage)
	}
	vectors := metricByName(report, "embed_vectors_total")
	if vectors == nil || vectors.Delta != nil || vectors.RatePerSecond != nil {
		t.Fatalf("reset counter = %+v, want no delta or rate", vectors)
	}
}

func TestBuildMarksJournalRetentionMismatchIncomplete(t *testing.T) {
	stateRoot := t.TempDir()
	now := time.Date(2026, 7, 31, 14, 0, 0, 0, time.UTC)
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": now.Add(-time.Hour - time.Second), "msg": "daemon.perf_counters", "jobs_completed_total": float64(1),
	})
	completed := now.Add(-time.Minute)
	writeJob(t, filepath.Join(stateRoot, "jobs.jsonl"), model.JobEvent{Event: "job_completed", OccurredAt: completed, Job: model.Job{ID: "one", State: model.JobStateCompleted, StartedAt: completed.Add(-time.Second), CompletedAt: &completed}})
	report, err := Build(Input{StateRoot: stateRoot, Since: time.Hour, Now: now, Status: &pb.GetStatusResponse{
		ReadAt: timestamppb.New(now), Metrics: []*pb.Metric{{Name: "jobs_completed_total", Value: &pb.Metric_IntValue{IntValue: 3}}},
	}})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if report.Coverage.Complete {
		t.Fatalf("coverage = %+v, want incomplete journal mismatch", report.Coverage)
	}
	completedMetric := metricByName(report, "jobs_completed")
	if completedMetric == nil || completedMetric.Current != nil || completedMetric.Delta == nil || completedMetric.RatePerSecond == nil {
		t.Fatalf("job metric = %+v, want delta and rate without current", completedMetric)
	}
}

func TestBuildIgnoresPoisonedClydeEnvironment(t *testing.T) {
	stateRoot := t.TempDir()
	clydeRoot := t.TempDir()
	t.Setenv("HOME", clydeRoot)
	t.Setenv("XDG_STATE_HOME", clydeRoot)
	now := time.Date(2026, 7, 31, 14, 0, 0, 0, time.UTC)
	writeRecord(t, filepath.Join(clydeRoot, ".clyde", "logs", "daemon.jsonl"), map[string]any{
		"time": now, "msg": "daemon.perf_counters", "clyde_sentinel_total": float64(999999),
	})
	writeRecord(t, filepath.Join(clydeRoot, ".config", "clyde", "logs", "daemon.jsonl"), map[string]any{
		"time": now, "msg": "daemon.span.completed", "span": "clyde-span", "duration_ms": float64(999999),
	})
	writeRecord(t, filepath.Join(clydeRoot, "clyde", "logs", "daemon.jsonl"), map[string]any{
		"time": now, "msg": "daemon.perf_counters", "clyde_sentinel_total": float64(999999),
	})
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": now.Add(-time.Hour - time.Second), "msg": "daemon.perf_counters", "embed_vectors_total": float64(1),
	})
	report, err := Build(Input{StateRoot: stateRoot, Since: time.Hour, Now: now, Status: &pb.GetStatusResponse{
		ReadAt: timestamppb.New(now), Metrics: []*pb.Metric{{Name: "embed_vectors_total", Value: &pb.Metric_IntValue{IntValue: 2}}},
	}})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if metricByName(report, "clyde_sentinel_total") != nil {
		t.Fatalf("poisoned Clyde metric entered report: %+v", report.Metrics)
	}
	if durationByName(report, "clyde-span") != nil {
		t.Fatalf("poisoned Clyde span entered report: %+v", report.TimeBreakdown)
	}
}

func TestBuildUsesRelevantResetsAndStatusUnits(t *testing.T) {
	stateRoot := t.TempDir()
	now := time.Date(2026, 7, 31, 14, 0, 0, 0, time.UTC)
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": now.Add(-2 * time.Hour), "msg": "daemon.perf_counters", "embed_vectors_total": float64(100),
	})
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": now.Add(-90 * time.Minute), "msg": "daemon.perf_counters", "embed_vectors_total": float64(1),
	})
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": now.Add(-time.Hour - time.Second), "msg": "daemon.perf_counters", "embed_vectors_total": float64(10), "num_gc": float64(7), "embed_chunks_reused_total": float64(5),
		"units": map[string]string{"embed_chunks_reused_total": "chunks"},
	})
	report, err := Build(Input{StateRoot: stateRoot, Since: time.Hour, Now: now, Status: &pb.GetStatusResponse{
		ReadAt: timestamppb.New(now), Metrics: []*pb.Metric{
			{Name: "embed_vectors_total", Unit: "vectors", Value: &pb.Metric_IntValue{IntValue: 20}},
			{Name: "num_gc", Unit: "cycles", Value: &pb.Metric_IntValue{IntValue: 9}},
			{Name: "embed_chunks_reused_total", Value: &pb.Metric_IntValue{IntValue: 9}},
		},
	}})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	vectors := metricByName(report, "embed_vectors_total")
	if vectors == nil || vectors.Delta == nil || *vectors.Delta != 10 || vectors.Unit != "vectors" {
		t.Fatalf("vectors = %+v, want delta 10 and vectors", vectors)
	}
	gc := metricByName(report, "num_gc")
	if gc == nil || gc.Kind != metricKindCounter || gc.Unit != "cycles" || gc.Delta == nil || *gc.Delta != 2 {
		t.Fatalf("num_gc = %+v, want counter in cycles", gc)
	}
	reused := metricByName(report, "embed_chunks_reused_total")
	if reused == nil || reused.Unit != "chunks" || reused.Delta == nil || *reused.Delta != 4 {
		t.Fatalf("embed_chunks_reused_total = %+v, want log unit and delta", reused)
	}
}

func TestBuildTreatsDaemonStartWithinWindowAsCounterReset(t *testing.T) {
	stateRoot := t.TempDir()
	now := time.Date(2026, 7, 31, 14, 0, 0, 0, time.UTC)
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": now.Add(-time.Hour - time.Second), "msg": "daemon.perf_counters", "embed_vectors_total": float64(100),
	})
	report, err := Build(Input{StateRoot: stateRoot, Since: time.Hour, Now: now, Status: &pb.GetStatusResponse{
		ReadAt:  timestamppb.New(now),
		Daemon:  &pb.DaemonIdentity{StartedAt: timestamppb.New(now.Add(-30 * time.Minute))},
		Metrics: []*pb.Metric{{Name: "embed_vectors_total", Unit: "vectors", Value: &pb.Metric_IntValue{IntValue: 20}}},
	}})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	vectors := metricByName(report, "embed_vectors_total")
	if vectors == nil || vectors.Delta != nil || vectors.RatePerSecond != nil {
		t.Fatalf("vectors = %+v, want daemon-start reset suppression", vectors)
	}
}

func TestBuildSuppressesRelevantResetAndEmbeddingLatency(t *testing.T) {
	stateRoot := t.TempDir()
	now := time.Date(2026, 7, 31, 14, 0, 0, 0, time.UTC)
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": now.Add(-time.Hour - time.Second), "msg": "daemon.perf_counters", "embed_vectors_total": float64(100), "embed_latency_ms_sum": float64(1000), "embed_batches_total": float64(10),
	})
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": now.Add(-30 * time.Minute), "msg": "daemon.perf_counters", "embed_vectors_total": float64(1), "embed_latency_ms_sum": float64(10), "embed_batches_total": float64(1),
	})
	report, err := Build(Input{StateRoot: stateRoot, Since: time.Hour, Now: now, Status: &pb.GetStatusResponse{
		ReadAt: timestamppb.New(now), Metrics: []*pb.Metric{
			{Name: "embed_vectors_total", Value: &pb.Metric_IntValue{IntValue: 120}},
			{Name: "embed_latency_ms_sum", Value: &pb.Metric_IntValue{IntValue: 1200}},
			{Name: "embed_batches_total", Value: &pb.Metric_IntValue{IntValue: 12}},
		},
	}})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	vectors := metricByName(report, "embed_vectors_total")
	if vectors == nil || vectors.Delta != nil || vectors.RatePerSecond != nil {
		t.Fatalf("vectors = %+v, want reset suppression", vectors)
	}
	if report.EmbeddingLatency != nil {
		t.Fatalf("embed latency survived reset: %+v", report.EmbeddingLatency)
	}
}

func TestBuildReadsCompressedRelevantRecordsAndMarksMalformedRelevantRecords(t *testing.T) {
	stateRoot := t.TempDir()
	now := time.Date(2026, 7, 31, 14, 0, 0, 0, time.UTC)
	writeGzipRecord(t, filepath.Join(stateRoot, "logs", "daemon-old.jsonl.gz"), map[string]any{
		"time": now.Add(-time.Hour - time.Second), "msg": "daemon.perf_counters", "embed_vectors_total": float64(1),
	})
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{"msg": "unrelated"})
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{"msg": "daemon.span.completed", "span": "broken"})
	report, err := Build(Input{StateRoot: stateRoot, Since: time.Hour, Now: now, Status: &pb.GetStatusResponse{
		ReadAt: timestamppb.New(now), Metrics: []*pb.Metric{{Name: "embed_vectors_total", Value: &pb.Metric_IntValue{IntValue: 2}}},
	}})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	vectors := metricByName(report, "embed_vectors_total")
	if vectors == nil || vectors.Delta == nil || *vectors.Delta != 1 {
		t.Fatalf("compressed baseline was not read: %+v", vectors)
	}
	if report.Coverage.Complete {
		t.Fatalf("coverage = %+v, want malformed relevant span to fail coverage", report.Coverage)
	}
	for _, warning := range report.Warnings {
		if warning == "structured counter record without a timestamp" {
			t.Fatalf("unrelated record produced warning: %v", report.Warnings)
		}
	}
}

func TestBuildRecoversFromCorruptCompressedUnrelatedHistory(t *testing.T) {
	stateRoot := t.TempDir()
	now := time.Date(2026, 7, 31, 14, 0, 0, 0, time.UTC)
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": now.Add(-time.Hour - time.Second), "msg": "daemon.perf_counters", "embed_vectors_total": float64(1),
	})
	corruptPath := filepath.Join(stateRoot, "logs", "clyde-unrelated.jsonl.gz")
	if err := os.WriteFile(corruptPath, []byte("not a gzip stream"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	report, err := Build(Input{StateRoot: stateRoot, Since: time.Hour, Now: now, Status: &pb.GetStatusResponse{
		ReadAt: timestamppb.New(now), Metrics: []*pb.Metric{{Name: "embed_vectors_total", Value: &pb.Metric_IntValue{IntValue: 2}}},
	}})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	vectors := metricByName(report, "embed_vectors_total")
	if vectors == nil || vectors.Delta == nil || *vectors.Delta != 1 {
		t.Fatalf("valid history was not retained: %+v", vectors)
	}
	if report.Coverage.Complete {
		t.Fatalf("coverage = %+v, want incomplete after corrupt compressed history", report.Coverage)
	}
	foundWarning := false
	for _, warning := range report.Warnings {
		if warning == "unreadable compressed daemon log" {
			foundWarning = true
		}
	}
	if !foundWarning {
		t.Fatalf("warnings = %v, want unreadable compressed-log warning", report.Warnings)
	}
}

func TestBuildRecoversFromTruncatedCompressedHistory(t *testing.T) {
	stateRoot := t.TempDir()
	now := time.Date(2026, 7, 31, 14, 0, 0, 0, time.UTC)
	writeRecord(t, filepath.Join(stateRoot, "logs", "daemon.jsonl"), map[string]any{
		"time": now.Add(-time.Hour - time.Second), "msg": "daemon.perf_counters", "embed_vectors_total": float64(1),
	})
	compressedPath := filepath.Join(stateRoot, "logs", "daemon-old.jsonl.gz")
	writeGzipRecord(t, compressedPath, map[string]any{"msg": "unrelated"})
	contents, err := os.ReadFile(compressedPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if err := os.WriteFile(compressedPath, contents[:len(contents)-4], 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	report, err := Build(Input{StateRoot: stateRoot, Since: time.Hour, Now: now, Status: &pb.GetStatusResponse{
		ReadAt: timestamppb.New(now), Metrics: []*pb.Metric{{Name: "embed_vectors_total", Value: &pb.Metric_IntValue{IntValue: 2}}},
	}})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if report.Coverage.Complete {
		t.Fatalf("coverage = %+v, want incomplete after truncated compressed history", report.Coverage)
	}
}

func writeRecord(t *testing.T, path string, record map[string]any) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	encoded, err := json.Marshal(record)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	file, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatalf("OpenFile: %v", err)
	}
	if _, err := file.Write(append(encoded, '\n')); err != nil {
		_ = file.Close()
		t.Fatalf("Write: %v", err)
	}
	if err := file.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func writeGzipRecord(t *testing.T, path string, record map[string]any) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	file, err := os.Create(path)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	writer := gzip.NewWriter(file)
	encoded, err := json.Marshal(record)
	if err != nil {
		_ = writer.Close()
		_ = file.Close()
		t.Fatalf("Marshal: %v", err)
	}
	if _, err := writer.Write(append(encoded, '\n')); err != nil {
		_ = writer.Close()
		_ = file.Close()
		t.Fatalf("Write: %v", err)
	}
	if err := writer.Close(); err != nil {
		_ = file.Close()
		t.Fatalf("Close gzip: %v", err)
	}
	if err := file.Close(); err != nil {
		t.Fatalf("Close file: %v", err)
	}
}

func metricByName(report Report, name string) *Metric {
	for index := range report.Metrics {
		if report.Metrics[index].Name == name {
			return &report.Metrics[index]
		}
	}
	return nil
}

func durationByName(report Report, name string) *Duration {
	for index := range report.TimeBreakdown {
		if report.TimeBreakdown[index].Name == name {
			return &report.TimeBreakdown[index]
		}
	}
	return nil
}

func writeJob(t *testing.T, path string, event model.JobEvent) {
	t.Helper()
	encoded, err := json.Marshal(event)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if err := os.WriteFile(path, append(encoded, '\n'), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
}

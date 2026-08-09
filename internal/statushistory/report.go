// Package statushistory builds bounded historical status reports from daemon-owned files.
package statushistory

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/model"
)

// Input names the daemon-owned state and current status used for a report.
type Input struct {
	StateRoot string
	Since     time.Duration
	Now       time.Time
	Status    *pb.GetStatusResponse
}

// Report is one bounded history report.
type Report struct {
	Window                 Window     `json:"window"`
	Coverage               Coverage   `json:"coverage"`
	Metrics                []Metric   `json:"metrics"`
	TimeBreakdown          []Duration `json:"time_breakdown"`
	EmbeddingLatency       *Duration  `json:"embedding_latency"`
	UnattributedDurationMS *int64     `json:"unattributed_duration_ms"`
	Warnings               []string   `json:"warnings"`
}

// Window bounds the report in UTC.
type Window struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

// Coverage describes which part of Window has reliable retained records.
type Coverage struct {
	Complete bool       `json:"complete"`
	Start    *time.Time `json:"start"`
	End      *time.Time `json:"end"`
}

// Metric summarizes one counter or gauge.
type Metric struct {
	Name          string     `json:"name"`
	Unit          string     `json:"unit"`
	Kind          MetricKind `json:"kind"`
	Current       *int64     `json:"current"`
	Delta         *int64     `json:"delta"`
	RatePerSecond *float64   `json:"rate_per_second"`
	Min           *int64     `json:"min"`
	Mean          *float64   `json:"mean"`
	Max           *int64     `json:"max"`
}

// MetricKind separates counter and gauge summaries.
type MetricKind string

const (
	metricKindCounter MetricKind = "counter"
	metricKindGauge   MetricKind = "gauge"
)

type logMessage string

const (
	logMessagePerfCounters logMessage = "daemon.perf_counters"
	logMessageSpanDone     logMessage = "daemon.span.completed"
)

// Duration summarizes a latency or exclusive stage duration.
type Duration struct {
	Name    string   `json:"name"`
	Unit    string   `json:"unit"`
	TotalMS *int64   `json:"total_ms"`
	Calls   *int64   `json:"calls"`
	MeanMS  *float64 `json:"mean_ms"`
	P50MS   *int64   `json:"p50_ms"`
	P95MS   *int64   `json:"p95_ms"`
	MaxMS   *int64   `json:"max_ms"`
}

type snapshot struct {
	at     time.Time
	values map[string]int64
	units  map[string]string
	live   bool
}

type span struct {
	at       time.Time
	name     string
	id       string
	parent   string
	jobID    string
	duration int64
}

const (
	maxSamplingGap          = 15 * time.Minute
	maxStructuredRecordSize = 4 << 20
)

type history struct {
	snapshots []snapshot
	spans     []span
	jobs      map[string]model.Job
	warnings  []string
	malformed bool
}

type metricDefinition struct {
	kind MetricKind
	unit string
}

var metricDefinitions = map[string]metricDefinition{
	"embed_batches_total":                           {kind: metricKindCounter, unit: "batches"},
	"embed_batches_failed":                          {kind: metricKindCounter, unit: "batches"},
	"embed_vectors_total":                           {kind: metricKindCounter, unit: "vectors"},
	"embed_latency_ms_sum":                          {kind: metricKindCounter, unit: "ms"},
	"embed_inflight":                                {kind: metricKindGauge, unit: "batches"},
	"embed_chunks_reused_total":                     {kind: metricKindCounter, unit: "chunks"},
	"embed_inputs_refused_empty":                    {kind: metricKindCounter, unit: "inputs"},
	"converge_upsert_total":                         {kind: metricKindCounter, unit: "paths"},
	"converge_remove_total":                         {kind: metricKindCounter, unit: "paths"},
	"converge_copy_chunks_total":                    {kind: metricKindCounter, unit: "paths"},
	"sweep_runs_total":                              {kind: metricKindCounter, unit: "runs"},
	"sweep_changed_total":                           {kind: metricKindCounter, unit: "runs"},
	"sync_skipped_inflight_total":                   {kind: metricKindCounter, unit: "requests"},
	"jobs_completed_total":                          {kind: metricKindCounter, unit: "jobs"},
	"jobs_failed_total":                             {kind: metricKindCounter, unit: "jobs"},
	"jobs_cancelled_total":                          {kind: metricKindCounter, unit: "jobs"},
	"boot_resumes_total":                            {kind: metricKindCounter, unit: "jobs"},
	"jobs_active":                                   {kind: metricKindGauge, unit: "jobs"},
	"milvus_collection_loads_total":                 {kind: metricKindCounter, unit: "loads"},
	"milvus_collection_load_failures_total":         {kind: metricKindCounter, unit: "loads"},
	"milvus_collection_load_wait_timeouts_total":    {kind: metricKindCounter, unit: "timeouts"},
	"milvus_collection_load_inflight":               {kind: metricKindGauge, unit: "loads"},
	"milvus_collection_load_latency_ms_sum":         {kind: metricKindCounter, unit: "ms"},
	"milvus_collection_unloads_total":               {kind: metricKindCounter, unit: "unloads"},
	"milvus_collection_unload_failures_total":       {kind: metricKindCounter, unit: "unloads"},
	"milvus_collection_unload_skipped_in_use_total": {kind: metricKindCounter, unit: "unloads"},
	"milvus_collection_unload_latency_ms_sum":       {kind: metricKindCounter, unit: "ms"},
	"milvus_collection_leases_active":               {kind: metricKindGauge, unit: "leases"},
	"milvus_collections_idle":                       {kind: metricKindGauge, unit: "collections"},
	"milvus_collections_loading":                    {kind: metricKindGauge, unit: "collections"},
	"milvus_collections_ready":                      {kind: metricKindGauge, unit: "collections"},
	"milvus_mmap_migrations_total":                  {kind: metricKindCounter, unit: "migrations"},
	"milvus_mmap_migration_failures_total":          {kind: metricKindCounter, unit: "migrations"},
	"num_goroutine":                                 {kind: metricKindGauge, unit: "goroutines"},
	"heap_alloc_bytes":                              {kind: metricKindGauge, unit: "bytes"},
	"heap_inuse_bytes":                              {kind: metricKindGauge, unit: "bytes"},
	"num_gc":                                        {kind: metricKindCounter, unit: "cycles"},
}

// Build reads daemon-owned retained history and combines it with status.
func Build(input Input) (Report, error) {
	if input.Since <= 0 {
		return Report{}, fmt.Errorf("since must be positive")
	}
	if input.Status == nil {
		return Report{}, fmt.Errorf("status response is required")
	}
	if input.Now.IsZero() {
		if input.Status.GetReadAt() != nil {
			input.Now = input.Status.GetReadAt().AsTime()
		} else {
			input.Now = clock.Now()
		}
	}
	window := Window{Start: input.Now.Add(-input.Since), End: input.Now}
	history, err := readHistory(input.StateRoot)
	if err != nil {
		return Report{}, err
	}
	current := statusSnapshot(input.Status, input.Now)
	history.snapshots = append(history.snapshots, current)
	slices.SortFunc(history.snapshots, func(left snapshot, right snapshot) int {
		return left.at.Compare(right.at)
	})

	var report Report
	report.Window = window
	report.Warnings = history.warnings
	report.Coverage = coverage(history.snapshots, window, input.Status)
	if history.malformed {
		report.Coverage.Complete = false
	}
	if samplingGap(history.snapshots, window) {
		report.Coverage.Complete = false
		report.Warnings = append(report.Warnings, "retained counter samples have a gap")
	}
	if !report.Coverage.Complete {
		report.Warnings = append(report.Warnings, "history does not cover the full requested window")
	}
	resets := counterResets(history.snapshots, window, input.Status)
	if len(resets) > 0 {
		report.Coverage.Complete = false
		report.Warnings = append(report.Warnings, "counter reset observed in the requested window")
	}
	report.Metrics = buildMetrics(history.snapshots, current, window, resets)
	report.Metrics = append(report.Metrics, jobMetrics(history.jobs, window)...)
	if journalMismatch(history.snapshots, current, history.jobs, window) {
		report.Coverage.Complete = false
		report.Warnings = append(report.Warnings, "job journal retention does not cover terminal job counters")
	}
	report.TimeBreakdown, report.UnattributedDurationMS = buildTiming(spansWithin(history.spans, window), history.jobs, window, &report.Warnings)
	report.EmbeddingLatency = embeddingLatency(history.snapshots, current, window, resets)
	return report, nil
}

func readHistory(stateRoot string) (history, error) {
	var result history
	result.jobs = map[string]model.Job{}
	logsRoot := filepath.Join(stateRoot, "logs")
	entries, err := os.ReadDir(logsRoot)
	if err != nil && !os.IsNotExist(err) {
		slog.Error("read daemon logs failed", "path", logsRoot, "err", err)
		return history{}, fmt.Errorf("read daemon logs: %w", err)
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if !strings.HasSuffix(entry.Name(), ".jsonl") && !strings.HasSuffix(entry.Name(), ".jsonl.gz") {
			continue
		}
		if err := readLog(filepath.Join(logsRoot, entry.Name()), &result); err != nil {
			if strings.HasSuffix(entry.Name(), ".jsonl.gz") {
				result.warnings = append(result.warnings, "unreadable compressed daemon log")
				result.malformed = true
				continue
			}
			return history{}, err
		}
	}
	if err := readJobs(filepath.Join(stateRoot, "jobs.jsonl"), &result); err != nil {
		return history{}, err
	}
	return result, nil
}

func readLog(path string, result *history) error {
	file, err := os.Open(path)
	if err != nil {
		slog.Error("open daemon log failed", "path", path, "err", err)
		return fmt.Errorf("open daemon log %s: %w", path, err)
	}
	defer func() { _ = file.Close() }()
	var reader io.Reader = file
	if strings.HasSuffix(path, ".gz") {
		gzipReader, err := gzip.NewReader(file)
		if err != nil {
			slog.Error("open compressed daemon log failed", "path", path, "err", err)
			return fmt.Errorf("open compressed daemon log %s: %w", path, err)
		}
		defer func() { _ = gzipReader.Close() }()
		reader = gzipReader
	}
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64*1024), maxStructuredRecordSize)
	for scanner.Scan() {
		var record map[string]json.RawMessage
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			result.warnings = append(result.warnings, "malformed structured log record")
			result.malformed = true
			continue
		}
		message := logMessage(stringValue(record["msg"]))
		switch message {
		case logMessagePerfCounters:
			at, ok := timeValue(record["time"])
			if !ok {
				result.warnings = append(result.warnings, "structured counter record without a timestamp")
				result.malformed = true
				continue
			}
			result.snapshots = append(result.snapshots, snapshot{at: at, values: numericFields(record), units: recordUnits(record), live: false})
		case logMessageSpanDone:
			at, ok := timeValue(record["time"])
			if !ok {
				result.warnings = append(result.warnings, "structured span record without a timestamp")
				result.malformed = true
				continue
			}
			duration, ok := intValue(record["duration_ms"])
			if !ok {
				result.warnings = append(result.warnings, "span record without duration")
				result.malformed = true
				continue
			}
			result.spans = append(result.spans, span{at: at, name: stringValue(record["span"]), id: stringValue(record["span_id"]), parent: stringValue(record["parent_span_id"]), jobID: stringValue(record["job_id"]), duration: duration})
		}
	}
	if err := scanner.Err(); err != nil {
		slog.Error("scan daemon log failed", "path", path, "err", err)
		result.warnings = append(result.warnings, "structured log scan stopped before end of file")
		result.malformed = true
	}
	return nil
}

func readJobs(path string, result *history) error {
	file, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		slog.Error("open daemon jobs journal failed", "path", path, "err", err)
		return fmt.Errorf("open daemon jobs journal: %w", err)
	}
	defer func() { _ = file.Close() }()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), maxStructuredRecordSize)
	for scanner.Scan() {
		var event model.JobEvent
		if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
			result.warnings = append(result.warnings, "malformed jobs journal record")
			result.malformed = true
			continue
		}
		if terminal(event.Job.State) {
			result.jobs[event.Job.ID] = event.Job
		}
	}
	if err := scanner.Err(); err != nil {
		slog.Error("scan daemon jobs journal failed", "path", path, "err", err)
		result.warnings = append(result.warnings, "job journal scan stopped before end of file")
		result.malformed = true
	}
	return nil
}

func statusSnapshot(status *pb.GetStatusResponse, now time.Time) snapshot {
	at := now
	if status.GetReadAt() != nil {
		at = status.GetReadAt().AsTime()
	}
	values := map[string]int64{}
	units := map[string]string{}
	for _, metric := range status.GetMetrics() {
		if value, ok := metric.GetValue().(*pb.Metric_IntValue); ok {
			values[metric.GetName()] = value.IntValue
			units[metric.GetName()] = metric.GetUnit()
		}
	}
	return snapshot{at: at, values: values, units: units, live: true}
}

func coverage(samples []snapshot, window Window, status *pb.GetStatusResponse) Coverage {
	var result Coverage
	for _, sample := range samples {
		if sample.at.After(window.Start) {
			break
		}
		at := sample.at
		result.Start = &at
	}
	if len(samples) > 0 {
		at := samples[len(samples)-1].at
		result.End = &at
	}
	result.Complete = result.Start != nil && !result.Start.After(window.Start) && result.End != nil && !result.End.Before(window.End)
	if daemon := status.GetDaemon(); daemon != nil && daemon.GetStartedAt() != nil && daemon.GetStartedAt().AsTime().After(window.Start) {
		result.Complete = false
	}
	return result
}

func samplingGap(samples []snapshot, window Window) bool {
	previous := window.Start
	for _, sample := range samples {
		if sample.at.Before(window.Start) {
			continue
		}
		if sample.at.After(window.End) {
			break
		}
		if sample.at.Sub(previous) > maxSamplingGap {
			return true
		}
		previous = sample.at
	}
	return window.End.Sub(previous) > maxSamplingGap
}

func journalMismatch(samples []snapshot, current snapshot, jobs map[string]model.Job, window Window) bool {
	expected := map[model.JobState]int64{
		model.JobStateCompleted: counterDelta(samples, current, "jobs_completed_total", window),
		model.JobStateFailed:    counterDelta(samples, current, "jobs_failed_total", window),
		model.JobStateCancelled: counterDelta(samples, current, "jobs_cancelled_total", window),
	}
	retained := map[model.JobState]int64{}
	for _, job := range jobs {
		if jobInWindow(job, window) {
			retained[job.State]++
		}
	}
	for state, count := range expected {
		if count > retained[state] {
			return true
		}
	}
	return false
}

func counterDelta(samples []snapshot, current snapshot, name string, window Window) int64 {
	baseline, found := baselineFor(samples, name, window.Start)
	if !found {
		return 0
	}
	value, found := current.values[name]
	if !found || value < baseline {
		return 0
	}
	return value - baseline
}

func spansWithin(spans []span, window Window) []span {
	filtered := make([]span, 0, len(spans))
	for _, current := range spans {
		if current.at.Before(window.Start) || current.at.After(window.End) {
			continue
		}
		filtered = append(filtered, current)
	}
	return filtered
}

func counterResets(samples []snapshot, window Window, status *pb.GetStatusResponse) map[string]bool {
	resets := map[string]bool{}
	last := map[string]int64{}
	seen := map[string]bool{}
	for _, sample := range samplesForResetDetection(samples, window) {
		for name, value := range sample.values {
			if isCounter(name) && seen[name] && value < last[name] {
				resets[name] = true
			}
			last[name] = value
			seen[name] = true
		}
	}
	if daemon := status.GetDaemon(); daemon != nil && daemon.GetStartedAt() != nil {
		startedAt := daemon.GetStartedAt().AsTime()
		if startedAt.After(window.Start) && !startedAt.After(window.End) {
			for name := range last {
				if isCounter(name) {
					resets[name] = true
				}
			}
		}
	}
	return resets
}

func samplesForResetDetection(samples []snapshot, window Window) []snapshot {
	selected := make([]snapshot, 0, len(samples))
	baselineIndex := -1
	for index, sample := range samples {
		if !sample.at.After(window.Start) {
			baselineIndex = index
			continue
		}
		break
	}
	if baselineIndex >= 0 {
		selected = append(selected, samples[baselineIndex])
	}
	for _, sample := range samples {
		if sample.at.Before(window.Start) || sample.at.After(window.End) {
			continue
		}
		selected = append(selected, sample)
	}
	return selected
}

func buildMetrics(samples []snapshot, current snapshot, window Window, resets map[string]bool) []Metric {
	names := map[string]bool{}
	for _, sample := range samples {
		for name := range sample.values {
			names[name] = true
		}
	}
	list := make([]Metric, 0, len(names))
	for name := range names {
		var metric Metric
		metric.Name = name
		metric.Unit = metricUnit(samples, current, name)
		if isCounter(name) {
			populateCounter(&metric, samples, current, window, resets[name])
		} else {
			metric.Kind = metricKindGauge
			values := valuesWithin(samples, name, window)
			if value, ok := current.values[name]; ok {
				metric.Current = &value
			}
			if hasRetainedValue(samples, name, window) {
				metric.Min, metric.Mean, metric.Max = summary(values)
			}
		}
		list = append(list, metric)
	}
	slices.SortFunc(list, func(left Metric, right Metric) int {
		return strings.Compare(left.Name, right.Name)
	})
	return list
}

func hasRetainedValue(samples []snapshot, name string, window Window) bool {
	for _, sample := range samples {
		if sample.live || sample.at.Before(window.Start) || sample.at.After(window.End) {
			continue
		}
		if _, ok := sample.values[name]; ok {
			return true
		}
	}
	return false
}

func populateCounter(metric *Metric, samples []snapshot, current snapshot, window Window, reset bool) {
	metric.Kind = metricKindCounter
	if value, ok := current.values[metric.Name]; ok {
		metric.Current = &value
	}
	baseline, found := baselineFor(samples, metric.Name, window.Start)
	if reset || !found || metric.Current == nil || *metric.Current < baseline {
		return
	}
	delta := *metric.Current - baseline
	metric.Delta = &delta
	rate := float64(delta) / window.End.Sub(window.Start).Seconds()
	metric.RatePerSecond = &rate
}

func jobMetrics(jobs map[string]model.Job, window Window) []Metric {
	values := map[string]int64{"jobs_completed": 0, "jobs_failed": 0, "jobs_cancelled": 0, "files_processed": 0, "files_embedded": 0, "chunks_processed": 0, "chunks_reused": 0, "chunks_embedded": 0, "embedding_batches": 0, "collection_rows_written": 0, "job_failures": 0}
	for _, job := range jobs {
		if !jobInWindow(job, window) {
			continue
		}
		values["jobs_"+string(job.State)]++
		values["files_processed"] += int64(job.Progress.FilesProcessed)
		values["files_embedded"] += int64(job.Progress.FilesEmbedded)
		values["chunks_processed"] += int64(job.Progress.ChunksProcessed)
		values["chunks_reused"] += int64(job.Progress.ChunksReused)
		values["chunks_embedded"] += int64(job.Progress.ChunksEmbedded)
		values["embedding_batches"] += int64(job.Progress.EmbeddingBatchesCompleted)
		values["collection_rows_written"] += int64(job.Progress.CollectionRowsWritten)
		if job.State == model.JobStateFailed {
			values["job_failures"]++
		}
	}
	list := make([]Metric, 0, len(values))
	for name, value := range values {
		var metric Metric
		metric.Name = name
		metric.Unit = journalMetricUnit(name)
		metric.Kind = metricKindCounter
		metric.Delta = &value
		rate := float64(value) / window.End.Sub(window.Start).Seconds()
		metric.RatePerSecond = &rate
		list = append(list, metric)
	}
	slices.SortFunc(list, func(left Metric, right Metric) int { return strings.Compare(left.Name, right.Name) })
	return list
}

func buildTiming(spans []span, jobs map[string]model.Job, window Window, warnings *[]string) ([]Duration, *int64) {
	byID := map[string]span{}
	children := map[string][]span{}
	for _, current := range spans {
		if current.id != "" {
			byID[current.id] = current
		}
		children[current.parent] = append(children[current.parent], current)
	}
	values := map[string][]int64{}
	rootDuration := int64(0)
	for _, current := range spans {
		exclusive := current.duration
		for _, child := range children[current.id] {
			exclusive -= child.duration
		}
		if exclusive < 0 {
			exclusive = 0
			*warnings = append(*warnings, "overlapping stage spans made exclusive time incomplete")
		}
		values[current.name] = append(values[current.name], exclusive)
		if _, parentKnown := byID[current.parent]; !parentKnown {
			rootDuration += current.duration
		}
	}
	list := make([]Duration, 0, len(values))
	for name, durations := range values {
		list = append(list, durationSummary(name, durations))
	}
	slices.SortFunc(list, func(left Duration, right Duration) int {
		return int(dereference(right.TotalMS) - dereference(left.TotalMS))
	})
	jobDuration := int64(0)
	for _, job := range jobs {
		if jobInWindow(job, window) && job.CompletedAt != nil && !job.StartedAt.IsZero() {
			jobDuration += job.CompletedAt.Sub(job.StartedAt).Milliseconds()
		}
	}
	if jobDuration == 0 {
		return list, nil
	}
	if rootDuration > jobDuration {
		*warnings = append(*warnings, "instrumented stage time exceeds measured job duration")
		rootDuration = jobDuration
	}
	unattributed := jobDuration - rootDuration
	return list, &unattributed
}

func durationSummary(name string, values []int64) Duration {
	sorted := append([]int64(nil), values...)
	slices.Sort(sorted)
	total := int64(0)
	for _, value := range sorted {
		total += value
	}
	calls := int64(len(sorted))
	mean := float64(total) / float64(calls)
	p50 := percentile(sorted, .5)
	p95 := percentile(sorted, .95)
	maximum := sorted[len(sorted)-1]
	var result Duration
	result.Name = name
	result.Unit = "ms"
	result.TotalMS = &total
	result.Calls = &calls
	result.MeanMS = &mean
	result.P50MS = &p50
	result.P95MS = &p95
	result.MaxMS = &maximum
	return result
}

func embeddingLatency(samples []snapshot, current snapshot, window Window, resets map[string]bool) *Duration {
	if resets["embed_latency_ms_sum"] || resets["embed_batches_total"] {
		return nil
	}
	totalBefore, hasTotal := baselineFor(samples, "embed_latency_ms_sum", window.Start)
	callsBefore, hasCalls := baselineFor(samples, "embed_batches_total", window.Start)
	if !hasTotal || !hasCalls {
		return nil
	}
	totalNow, hasTotal := current.values["embed_latency_ms_sum"]
	callsNow, hasCalls := current.values["embed_batches_total"]
	if !hasTotal || !hasCalls || totalNow < totalBefore || callsNow < callsBefore {
		return nil
	}
	total := totalNow - totalBefore
	calls := callsNow - callsBefore
	var result Duration
	result.Name = "embed_latency"
	result.Unit = "ms"
	result.TotalMS = &total
	result.Calls = &calls
	if calls > 0 {
		mean := float64(total) / float64(calls)
		result.MeanMS = &mean
	}
	return &result
}

func baselineFor(samples []snapshot, name string, at time.Time) (int64, bool) {
	var value int64
	found := false
	for _, sample := range samples {
		if sample.at.After(at) {
			break
		}
		if current, ok := sample.values[name]; ok {
			value, found = current, true
		}
	}
	return value, found
}

func valuesWithin(samples []snapshot, name string, window Window) []int64 {
	values := []int64{}
	for _, sample := range samples {
		if sample.at.Before(window.Start) || sample.at.After(window.End) {
			continue
		}
		if value, ok := sample.values[name]; ok {
			values = append(values, value)
		}
	}
	return values
}

func summary(values []int64) (*int64, *float64, *int64) {
	minimum, maximum, total := values[0], values[0], int64(0)
	for _, value := range values {
		minimum = minValue(minimum, value)
		maximum = maxValue(maximum, value)
		total += value
	}
	mean := float64(total) / float64(len(values))
	return &minimum, &mean, &maximum
}

func percentile(values []int64, fraction float64) int64 {
	return values[int(math.Ceil(float64(len(values))*fraction))-1]
}

func terminal(state model.JobState) bool {
	return state == model.JobStateCompleted || state == model.JobStateFailed || state == model.JobStateCancelled
}

func jobInWindow(job model.Job, window Window) bool {
	return job.CompletedAt != nil && !job.CompletedAt.Before(window.Start) && !job.CompletedAt.After(window.End)
}

func dereference(value *int64) int64 {
	if value == nil {
		return 0
	}
	return *value
}

func minValue(left int64, right int64) int64 {
	if left < right {
		return left
	}
	return right
}

func maxValue(left int64, right int64) int64 {
	if left > right {
		return left
	}
	return right
}

func stringValue(raw json.RawMessage) string {
	var value string
	_ = json.Unmarshal(raw, &value)
	return value
}

func timeValue(raw json.RawMessage) (time.Time, bool) {
	value := stringValue(raw)
	parsed, err := time.Parse(time.RFC3339Nano, value)
	return parsed, err == nil
}

func intValue(raw json.RawMessage) (int64, bool) {
	var value int64
	if err := json.Unmarshal(raw, &value); err == nil {
		return value, true
	}
	var decimal float64
	if err := json.Unmarshal(raw, &decimal); err != nil || decimal != math.Trunc(decimal) {
		return 0, false
	}
	return int64(decimal), true
}

func numericFields(record map[string]json.RawMessage) map[string]int64 {
	values := map[string]int64{}
	for name, raw := range record {
		if value, ok := intValue(raw); ok {
			values[name] = value
		}
	}
	return values
}

func recordUnits(record map[string]json.RawMessage) map[string]string {
	units := map[string]string{}
	raw, found := record["units"]
	if !found {
		return units
	}
	_ = json.Unmarshal(raw, &units)
	return units
}

func isCounter(name string) bool {
	definition, found := metricDefinitions[name]
	return found && definition.kind == metricKindCounter
}

func metricUnit(samples []snapshot, current snapshot, name string) string {
	if unit := current.units[name]; unit != "" {
		return unit
	}
	for _, sample := range slices.Backward(samples) {
		if unit := sample.units[name]; unit != "" {
			return unit
		}
	}
	definition, found := metricDefinitions[name]
	if found {
		return definition.unit
	}
	return ""
}

func journalMetricUnit(name string) string {
	units := map[string]string{
		"jobs_completed":          "jobs",
		"jobs_failed":             "jobs",
		"jobs_cancelled":          "jobs",
		"files_processed":         "files",
		"files_embedded":          "files",
		"chunks_processed":        "chunks",
		"chunks_reused":           "chunks",
		"chunks_embedded":         "chunks",
		"embedding_batches":       "batches",
		"collection_rows_written": "rows",
		"job_failures":            "jobs",
	}
	return units[name]
}

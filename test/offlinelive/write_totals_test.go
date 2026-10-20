//go:build offlinelive

package offlinelive

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/metrics"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/statushistory"
)

const writeTotalsFileCount = 2

func TestWriteTotalsCountEveryIndexedFile(t *testing.T) {
	harness := newHarness(t)
	directory := t.TempDir()
	writeTotalsSources(t, directory, "initial")
	jobID := harness.startIndexAt(directory)
	requireWriteTotalsActivity(t, harness, jobID)
	initial := harness.waitForJob(jobID)
	requireCompleted(t, initial)
	requirePersistedWriteTotals(t, harness.config.JobsPath, jobID, writeTotalsFileCount, writeTotalsFileCount)
	if got := initial.GetProgress().GetEmbeddingBatchesCompleted(); got != 1 {
		t.Fatalf("Current item completed batches = %d, want 1", got)
	}
	status := harness.indexStatusAt(directory)
	if got := status.GetCodebase().GetLastSuccessfulRun().GetTotalChunks(); got != writeTotalsFileCount {
		t.Fatalf("Stored chunks = %d, want %d", got, writeTotalsFileCount)
	}
	requireHistoricalWriteTotals(t, harness, writeTotalsFileCount, writeTotalsFileCount)

	writeTotalsSources(t, directory, "revised")
	sync, err := harness.client.SyncIndex(correlatedContext(), &pb.SyncIndexRequest{Path: directory, Client: harnessClientInfo()})
	if err != nil {
		t.Fatalf("SyncIndex: %v", err)
	}
	requireCompleted(t, harness.waitForJob(sync.GetJobId()))
	requirePersistedWriteTotals(t, harness.config.JobsPath, sync.GetJobId(), writeTotalsFileCount, writeTotalsFileCount)
	requireHistoricalWriteTotals(t, harness, 2*writeTotalsFileCount, 2*writeTotalsFileCount)

	legacyRoot := t.TempDir()
	journal := readWriteTotalsJournal(t, harness.config.JobsPath)
	for index := range journal {
		journal[index].Job.Progress.WriteTotals = nil
	}
	legacyPath := filepath.Join(legacyRoot, "jobs.jsonl")
	file, err := os.Create(legacyPath)
	if err != nil {
		t.Fatalf("Create legacy journal: %v", err)
	}
	encoder := json.NewEncoder(file)
	for _, event := range journal {
		if err := encoder.Encode(event); err != nil {
			_ = file.Close()
			t.Fatalf("Encode legacy journal: %v", err)
		}
	}
	if err := file.Close(); err != nil {
		t.Fatalf("Close legacy journal: %v", err)
	}
	report := writeTotalsReport(t, harness, legacyRoot)
	for _, metric := range report.Metrics {
		if metric.Name == "collection_rows_written" || metric.Name == "embedding_batches" {
			if metric.Delta != nil || metric.RatePerSecond != nil {
				t.Fatalf("Legacy %s must remain unknown: %+v", metric.Name, metric)
			}
		}
	}
	if !slices.Contains(report.Warnings, "retained jobs do not include complete write totals") {
		t.Fatalf("Legacy report omitted the write totals warning: %v", report.Warnings)
	}
}

func TestWriteTotalsCountWatcherConvergence(t *testing.T) {
	harness := newHarnessWith(t, harnessOptions{fileWatcher: true})
	directory := t.TempDir()
	writeTotalsSources(t, directory, "initial")
	initial := harness.waitForJob(harness.startIndexAt(directory))
	requireCompleted(t, initial)
	codebaseID := initial.GetCodebaseId()
	writeTotalsSources(t, directory, "revised")
	deadline := time.Now().Add(renameConvergeTimeout)
	for {
		var written, batches int64
		for _, event := range readWriteTotalsJournal(t, harness.config.JobsPath) {
			job := event.Job
			if event.Event != "job_completed" || job.CodebaseID != codebaseID || job.Operation != "converge" {
				continue
			}
			if job.Progress.WriteTotals == nil {
				t.Fatal("Watcher convergence omitted write totals.")
			}
			written += job.Progress.WriteTotals.CollectionRowsWritten
			batches += job.Progress.WriteTotals.EmbeddingBatchesCompleted
		}
		if written == writeTotalsFileCount && batches == writeTotalsFileCount {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("Watcher wrote %d rows in %d batches, want %d each", written, batches, writeTotalsFileCount)
		}
		time.Sleep(pollInterval)
	}
	requireHistoricalWriteTotals(t, harness, 2*writeTotalsFileCount, 2*writeTotalsFileCount)
	if err := os.Rename(filepath.Join(directory, "alpha.go"), filepath.Join(directory, "renamed.go")); err != nil {
		t.Fatalf("Rename source: %v", err)
	}
	harness.waitForRenameConverge(directory, "renamed.go")
	requireHistoricalWriteTotals(t, harness, 2*writeTotalsFileCount+1, 2*writeTotalsFileCount)
}

func TestWriteTotalsMeasureMultipleBatchesAndReuse(t *testing.T) {
	harness := newHarness(t)
	repository := newEmptyRepository(t)
	var source strings.Builder
	source.WriteString("package example\n\n")
	for index := range 180 {
		fmt.Fprintf(&source, "func Value%d() string { return %q }\n", index, strings.Repeat("value ", 30)+fmt.Sprint(index))
	}
	if err := os.WriteFile(filepath.Join(repository, "values.go"), []byte(source.String()), 0o644); err != nil {
		t.Fatalf("Write source: %v", err)
	}
	gitCommitAll(t, repository, "Add values")
	worktree := addNestedWorktree(t, repository)
	before := metrics.Read()
	initial := harness.waitForJob(harness.startIndexAt(repository))
	requireCompleted(t, initial)
	after := metrics.Read()
	batches := after.EmbedBatchesTotal - before.EmbedBatchesTotal
	if batches <= 1 {
		t.Fatalf("The real provider completed %d batches, want more than 1", batches)
	}
	rows := int64(initial.GetProgress().GetChunksProcessed())
	t.Logf("The ONNX provider completed %d batches for %d indexed chunks.", batches, rows)
	requirePersistedWriteTotals(t, harness.config.JobsPath, initial.GetId(), rows, batches)
	if got := int64(initial.GetProgress().GetEmbeddingBatchesCompleted()); got != batches {
		t.Fatalf("Current item reports %d batches, but the provider completed %d", got, batches)
	}
	before = metrics.Read()
	reused := harness.waitForJob(harness.startIndexWithSourceAt(worktree, repository))
	requireCompleted(t, reused)
	after = metrics.Read()
	if completed := after.EmbedBatchesTotal - before.EmbedBatchesTotal; completed != 0 {
		t.Fatalf("The reuse build called the provider %d times, want 0", completed)
	}
	if int64(reused.GetProgress().GetChunksReused()) != rows || reused.GetProgress().GetEmbeddingBatchesCompleted() != 0 {
		t.Fatalf("The reuse build must report %d reused chunks and 0 embedding batches: %s", rows, reused.GetProgress())
	}
	requirePersistedWriteTotals(t, harness.config.JobsPath, reused.GetId(), rows, 0)
	requireHistoricalWriteTotals(t, harness, 2*rows, batches)
}

func writeTotalsSources(t *testing.T, directory, revision string) {
	t.Helper()
	for _, name := range []string{"alpha", "beta"} {
		content := "package example\n\nfunc " + name + "() string { return \"" + name + " " + revision + "\" }\n"
		if err := os.WriteFile(filepath.Join(directory, name+".go"), []byte(content), 0o644); err != nil {
			t.Fatalf("Write source: %v", err)
		}
	}
}

func requireWriteTotalsActivity(t *testing.T, harness *harness, jobID string) {
	t.Helper()
	deadline := time.Now().Add(jobPollTimeout)
	for {
		response, err := harness.client.GetStatus(context.Background(), &pb.GetStatusRequest{})
		if err != nil {
			t.Fatalf("GetStatus: %v", err)
		}
		for _, row := range response.GetActivity() {
			matches := false
			measured := 0
			for _, metric := range row.GetMetrics() {
				if metric.GetName() == "job_id" && metric.GetStringValue() == jobID {
					matches = true
				}
				if metric.GetName() == "collection_rows_written_total" || metric.GetName() == "embedding_batches_completed_total" {
					if _, known := metric.GetValue().(*pb.Metric_IntValue); known {
						measured++
					}
				}
			}
			if matches && measured == 2 {
				return
			}
		}
		if time.Now().After(deadline) {
			t.Fatal("Current status did not report measured run write totals.")
		}
		time.Sleep(time.Millisecond)
	}
}

func readWriteTotalsJournal(t *testing.T, path string) []model.JobEvent {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("Open jobs journal: %v", err)
	}
	defer func() { _ = file.Close() }()
	scanner := bufio.NewScanner(file)
	var events []model.JobEvent
	for scanner.Scan() {
		var event model.JobEvent
		if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
			t.Fatalf("Decode job event: %v", err)
		}
		events = append(events, event)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("Read jobs journal: %v", err)
	}
	return events
}

func requirePersistedWriteTotals(t *testing.T, path, jobID string, expectedRows, expectedBatches int64) {
	t.Helper()
	for _, event := range readWriteTotalsJournal(t, path) {
		if event.Job.ID != jobID || event.Event != "job_completed" {
			continue
		}
		totals := event.Job.Progress.WriteTotals
		if totals == nil || totals.CollectionRowsWritten != expectedRows || totals.EmbeddingBatchesCompleted != expectedBatches {
			t.Fatalf("Job %s write totals = %+v, want %d rows and %d batches", jobID, totals, expectedRows, expectedBatches)
		}
		t.Logf("The journal records %d written rows and %d completed embedding batches for job %s.", totals.CollectionRowsWritten, totals.EmbeddingBatchesCompleted, jobID)
		return
	}
	t.Fatalf("Completed job %s is absent from the journal", jobID)
}

func writeTotalsReport(t *testing.T, harness *harness, stateRoot string) statushistory.Report {
	t.Helper()
	status, err := harness.client.GetStatus(context.Background(), &pb.GetStatusRequest{})
	if err != nil {
		t.Fatalf("GetStatus: %v", err)
	}
	report, err := statushistory.Build(statushistory.Input{StateRoot: stateRoot, Since: time.Hour, Status: status})
	if err != nil {
		t.Fatalf("Build historical status: %v", err)
	}
	return report
}

func requireHistoricalWriteTotals(t *testing.T, harness *harness, expectedRows, expectedBatches int64) {
	t.Helper()
	report := writeTotalsReport(t, harness, harness.config.StateRoot)
	checked := 0
	for _, metric := range report.Metrics {
		expected := expectedRows
		if metric.Name == "embedding_batches" {
			expected = expectedBatches
		}
		if metric.Name == "collection_rows_written" || metric.Name == "embedding_batches" {
			if metric.Delta == nil || *metric.Delta != expected {
				t.Fatalf("Historical %s = %+v, want %d", metric.Name, metric.Delta, expected)
			}
			checked++
		}
	}
	if checked != 2 {
		t.Fatal("Historical status omitted the write totals.")
	}
}

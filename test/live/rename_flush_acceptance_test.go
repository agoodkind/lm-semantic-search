//go:build live && acceptance

package live

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

const (
	renameReportVariable        = "LMS_RENAME_REPORT"
	renameSourceDirectory       = "before"
	renameSeedName              = "before/seed.go"
	renameSeedContent           = "package readings\n\nfunc Seed() int {\n\treturn 0\n}\n"
	renameJobTimeout            = 45 * time.Minute
	renameJobStartTimeout       = 2 * time.Minute
	renameRowsMoveGrace         = 30 * time.Second
	renamePollInterval          = 100 * time.Millisecond
	renameFlushWindow           = 11 * time.Second
	renameCopyMetric            = "converge_copy_chunks_total"
	renameUpsertMetric          = "converge_upsert_total"
	renameSearchLimit     int32 = 5
)

var renameCounts = []int{1, 10, 100}

type renameMeasurement struct {
	RenamedFiles          int    `json:"renamed_files"`
	JobID                 string `json:"job_id"`
	EmbeddingProvider     string `json:"embedding_provider"`
	EmbeddingModel        string `json:"embedding_model"`
	FlushRequests         int    `json:"flush_requests"`
	InsertRequests        int    `json:"insert_requests"`
	DeleteRequests        int    `json:"delete_requests"`
	CopyChunksDelta       int64  `json:"converge_copy_chunks_delta"`
	CopyPathUsed          bool   `json:"copy_path_used"`
	UpsertDelta           int64  `json:"converge_upsert_delta"`
	RowsBefore            int64  `json:"rows_before"`
	RowsAfter             int64  `json:"rows_after"`
	JobState              string `json:"job_state"`
	QueueMilliseconds     int64  `json:"queue_ms"`
	ActiveMilliseconds    int64  `json:"active_ms"`
	TotalMilliseconds     int64  `json:"rename_to_job_end_ms"`
	RowsMovedMilliseconds int64  `json:"rename_to_rows_moved_ms"`
	SearchReturnsNewPath  bool   `json:"search_returns_new_path"`
	SearchReturnsOldPath  bool   `json:"search_returns_old_path"`
}

func renameSourceName(index int) string {
	return fmt.Sprintf("%s/reading%04d.go", renameSourceDirectory, index)
}

func renameDestinationName(round int, index int) string {
	return fmt.Sprintf("moved%03d/reading%04d.go", round, index)
}

func renameSourceContent(index int) string {
	return fmt.Sprintf(
		"package readings\n\n// Reading%04d returns how many readings exceed the threshold %d.\nfunc Reading%04d(readings []int) int {\n\tcount := 0\n\tfor _, reading := range readings {\n\t\tif reading > %d {\n\t\t\tcount++\n\t\t}\n\t}\n\treturn count\n}\n",
		index, index*7, index, index*7,
	)
}

func renameSearchQuery(index int) string {
	return fmt.Sprintf("count how many readings exceed the threshold %d", index*7)
}

func newRenameRepository(t *testing.T) string {
	t.Helper()

	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatalf("Resolving symlinks of the temporary directory failed with error %v.", err)
	}
	repository := filepath.Join(root, "renames")
	if err := os.MkdirAll(filepath.Join(repository, renameSourceDirectory), 0o755); err != nil {
		t.Fatalf("Creating the before directory in repository %s failed with error %v.", repository, err)
	}
	writeRenameFile(t, filepath.Join(repository, renameSeedName), renameSeedContent)
	return repository
}

func writeRenameFile(t *testing.T, path string, content string) {
	t.Helper()

	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("Writing fixture file %s failed with error %v.", path, err)
	}
}

func writeRenameFixtures(t *testing.T, repository string, fileCount int) []string {
	t.Helper()

	paths := make([]string, 0, fileCount)
	for index := range fileCount {
		writeRenameFile(t, filepath.Join(repository, renameSourceName(index)), renameSourceContent(index))
		paths = append(paths, renameSourceName(index))
	}
	return paths
}

func renameAwaitFixtureRows(
	t *testing.T,
	liveHarness *harness,
	collectionName string,
	codebaseID string,
	paths []string,
) {
	t.Helper()

	startedAt := time.Now()
	filter := relativePathField + " in " + renameQuotedList(paths)
	rows := int64(0)
	deadline := startedAt.Add(renameJobTimeout)
	for time.Now().Before(deadline) {
		rows = renameCountRows(t, liveHarness, collectionName, filter)
		if rows >= int64(len(paths)) && !renameJobsActive(t, liveHarness, codebaseID) {
			t.Logf("The collection contained the required rows for %d fixture files and every converge job for the codebase had finished %s after the test wrote the fixture files.", len(paths), time.Since(startedAt))
			return
		}
		time.Sleep(renamePollInterval)
	}
	t.Fatalf("The wait for %d fixture files timed out with %d rows after %s. The collection did not contain enough rows for the fixture paths, or a converge job for the codebase remained queued or running.", len(paths), rows, renameJobTimeout)
}

func renameJobsActive(t *testing.T, liveHarness *harness, codebaseID string) bool {
	t.Helper()

	for _, job := range renameListJobs(t, liveHarness, codebaseID) {
		if !publicJobTerminal(job) {
			return true
		}
	}
	return false
}

func renameListJobs(t *testing.T, liveHarness *harness, codebaseID string) []*pb.Job {
	t.Helper()

	response, err := liveHarness.client.ListJobs(correlatedContext(), &pb.ListJobsRequest{CodebaseId: codebaseID})
	if err != nil {
		t.Fatalf("ListJobs for codebase %s returned error %v.", codebaseID, err)
	}
	return response.GetJobs()
}

func renameFiles(t *testing.T, repository string, round int, first int, count int) ([]string, []string) {
	t.Helper()

	destinationDirectory := filepath.Join(repository, filepath.Dir(renameDestinationName(round, first)))
	if err := os.MkdirAll(destinationDirectory, 0o755); err != nil {
		t.Fatalf("Creating destination directory %s failed with error %v.", destinationDirectory, err)
	}
	sources := make([]string, 0, count)
	destinations := make([]string, 0, count)
	for index := first; index < first+count; index++ {
		source := renameSourceName(index)
		destination := renameDestinationName(round, index)
		if err := os.Rename(filepath.Join(repository, source), filepath.Join(repository, destination)); err != nil {
			t.Fatalf("Renaming source path %s to destination path %s failed with error %v.", source, destination, err)
		}
		sources = append(sources, source)
		destinations = append(destinations, destination)
	}
	return sources, destinations
}

func renameQuotedList(paths []string) string {
	quoted := make([]string, 0, len(paths))
	for _, path := range paths {
		quoted = append(quoted, fmt.Sprintf("%q", path))
	}
	return "[" + strings.Join(quoted, ",") + "]"
}

func renameCountRows(t *testing.T, liveHarness *harness, collectionName string, filter string) int64 {
	t.Helper()

	queryContext, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	option := milvusclient.NewQueryOption(collectionName).
		WithFilter(filter).
		WithOutputFields(countOutputField).
		WithConsistencyLevel(entity.ClStrong)
	result, err := liveHarness.milvus.Query(queryContext, option)
	if err != nil {
		t.Fatalf("The Milvus count(*) Query for collection %s with filter %q failed with error %v.", collectionName, filter, err)
	}
	countColumn := result.GetColumn(countOutputField)
	if countColumn == nil {
		t.Fatalf("The Milvus count(*) Query for collection %s with filter %q did not return a count(*) column.", collectionName, filter)
	}
	count, err := countColumn.GetAsInt64(0)
	if err != nil {
		t.Fatalf("Reading the count(*) value as int64 for collection %s with filter %q failed with error %v.", collectionName, filter, err)
	}
	return count
}

func renameCountCalls(calls []milvusCall, method string, collectionName string) int {
	count := 0
	for _, call := range calls {
		if call.method == method && slices.Contains(call.collectionNames, collectionName) {
			count++
		}
	}
	return count
}

func renameWatcherJob(t *testing.T, liveHarness *harness, codebaseID string, after time.Time) *pb.Job {
	t.Helper()

	var newest *pb.Job
	for _, job := range renameListJobs(t, liveHarness, codebaseID) {
		if job.GetStartedAt().AsTime().Before(after) {
			continue
		}
		if newest == nil || job.GetStartedAt().AsTime().After(newest.GetStartedAt().AsTime()) {
			newest = job
		}
	}
	return newest
}

func measureRenameRound(
	t *testing.T,
	liveHarness *harness,
	repository string,
	collectionName string,
	codebaseID string,
	round int,
	first int,
	count int,
) renameMeasurement {
	t.Helper()

	rowsBefore := renameCountRows(t, liveHarness, collectionName, "")
	copyBefore := acceptanceMetric(t, liveHarness, renameCopyMetric)
	upsertBefore := acceptanceMetric(t, liveHarness, renameUpsertMetric)
	mark := acceptanceCallMark(liveHarness)

	renamedAt := time.Now()
	sources, destinations := renameFiles(t, repository, round, first, count)
	sourceFilter := relativePathField + " in " + renameQuotedList(sources)
	destinationFilter := relativePathField + " in " + renameQuotedList(destinations)

	var rowsMoved time.Duration
	var finished *pb.Job
	var firstTerminalAt time.Time
	deadline := renamedAt.Add(renameJobTimeout)
	for time.Now().Before(deadline) {
		if rowsMoved == 0 &&
			renameCountRows(t, liveHarness, collectionName, sourceFilter) == 0 &&
			renameCountRows(t, liveHarness, collectionName, destinationFilter) >= int64(count) {
			rowsMoved = time.Since(renamedAt)
		}
		job := renameWatcherJob(t, liveHarness, codebaseID, renamedAt)
		if job == nil && time.Since(renamedAt) > renameJobStartTimeout {
			t.Fatalf("The daemon did not report a job that started after the rename of %d files within %s.", count, renameJobStartTimeout)
		}
		if job != nil && publicJobTerminal(job) {
			finished = job
			if rowsMoved != 0 {
				break
			}
			if firstTerminalAt.IsZero() {
				firstTerminalAt = time.Now()
			}
			if time.Since(firstTerminalAt) >= renameRowsMoveGrace {
				break
			}
		}
		time.Sleep(renamePollInterval)
	}
	if finished == nil {
		t.Fatalf("A converge job that started after the rename of %d files did not finish as completed, failed, or cancelled within %s.", count, renameJobTimeout)
	}
	calls := acceptanceCallsSince(liveHarness, mark)
	copyDelta := acceptanceMetric(t, liveHarness, renameCopyMetric) - copyBefore

	sampleIndex := first
	results := renameSearch(t, liveHarness, repository, renameSearchQuery(sampleIndex))
	returnsNew := false
	returnsOld := false
	for _, result := range results {
		if result.GetRelativePath() == renameDestinationName(round, sampleIndex) {
			returnsNew = true
		}
		if result.GetRelativePath() == renameSourceName(sampleIndex) {
			returnsOld = true
		}
	}

	return renameMeasurement{
		RenamedFiles:          count,
		JobID:                 finished.GetId(),
		EmbeddingProvider:     finished.GetConfig().GetEmbeddingProvider(),
		EmbeddingModel:        finished.GetConfig().GetEmbeddingModel(),
		FlushRequests:         renameCountCalls(calls, "Flush", collectionName),
		InsertRequests:        renameCountCalls(calls, "Insert", collectionName),
		DeleteRequests:        renameCountCalls(calls, "Delete", collectionName),
		CopyChunksDelta:       copyDelta,
		CopyPathUsed:          copyDelta == int64(count),
		UpsertDelta:           acceptanceMetric(t, liveHarness, renameUpsertMetric) - upsertBefore,
		RowsBefore:            rowsBefore,
		RowsAfter:             renameCountRows(t, liveHarness, collectionName, ""),
		JobState:              finished.GetState(),
		QueueMilliseconds:     finished.GetStartedAt().AsTime().Sub(renamedAt).Milliseconds(),
		ActiveMilliseconds:    finished.GetCompletedAt().AsTime().Sub(finished.GetStartedAt().AsTime()).Milliseconds(),
		TotalMilliseconds:     finished.GetCompletedAt().AsTime().Sub(renamedAt).Milliseconds(),
		RowsMovedMilliseconds: rowsMoved.Milliseconds(),
		SearchReturnsNewPath:  returnsNew,
		SearchReturnsOldPath:  returnsOld,
	}
}

func renameIndex(t *testing.T, liveHarness *harness, path string) {
	t.Helper()

	requestedAt := time.Now()
	response := acceptanceStartIndex(t, liveHarness, path, "", "")
	if response.GetJobId() == "" {
		t.Fatalf("StartIndex for the first index build of repository %s did not return a job identifier.", path)
	}
	job := renameAwaitJob(t, liveHarness, response.GetJobId())
	t.Logf("The test observed the first index build of repository %s, job %s, finish %s after the StartIndex request.", path, job.GetId(), time.Since(requestedAt))
	if job.GetState() != "completed" {
		t.Fatalf("The first index build for repository %s finished with state %q instead of completed. The job reported error %s.", path, job.GetState(), job.GetDisplayError())
	}
}

func renameAwaitJob(t *testing.T, liveHarness *harness, jobID string) *pb.Job {
	t.Helper()

	deadline := time.Now().Add(renameJobTimeout)
	for time.Now().Before(deadline) {
		response, err := liveHarness.client.GetJob(correlatedContext(), &pb.GetJobRequest{JobId: jobID})
		if err != nil {
			t.Fatalf("GetJob for job %s returned error %v.", jobID, err)
		}
		job := response.GetJob()
		if job == nil {
			t.Fatalf("GetJob for job %s did not return a job.", jobID)
		}
		if publicJobTerminal(job) {
			return job
		}
		time.Sleep(jobPollInterval)
	}
	t.Fatalf("Job %s did not finish as completed, failed, or cancelled within %s.", jobID, renameJobTimeout)
	return nil
}

func renameSearch(t *testing.T, liveHarness *harness, path string, query string) []*pb.SearchResult {
	t.Helper()

	response, err := liveHarness.client.SearchCode(correlatedContext(), &pb.SearchCodeRequest{
		Path:   path,
		Query:  query,
		Limit:  renameSearchLimit,
		Client: acceptanceClientInfo(),
	})
	if err != nil {
		t.Fatalf("SearchCode for repository %s with query %q returned error %v.", path, query, err)
	}
	return response.GetResults()
}

func appendRenameReport(t *testing.T, measurement renameMeasurement) {
	t.Helper()

	encoded, err := json.Marshal(measurement)
	if err != nil {
		t.Fatalf("Encoding the rename measurement as JSON failed with error %v.", err)
	}
	t.Logf("The rename measurement is %s.", encoded)
	reportPath := os.Getenv(renameReportVariable)
	if reportPath == "" {
		return
	}
	report, err := os.OpenFile(reportPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatalf("Opening report file %s failed with error %v.", reportPath, err)
	}
	if _, err := report.Write(append(encoded, '\n')); err != nil {
		_ = report.Close()
		t.Fatalf("Writing the JSON line to report file %s failed with error %v.", reportPath, err)
	}
	if err := report.Close(); err != nil {
		t.Fatalf("Closing report file %s failed with error %v.", reportPath, err)
	}
}

func TestAcceptanceRenameConvergeMeasurement(t *testing.T) {
	totalRenamed := 0
	for _, count := range renameCounts {
		totalRenamed += count
	}
	liveHarness := newMeasurementHarness(t, true)
	t.Logf("This test run uses temporary Milvus database %s.", liveHarness.databaseName)
	repository := newRenameRepository(t)
	collectionName := liveHarness.trackCodebasePath(repository)
	renameIndex(t, liveHarness, repository)
	codebaseID := acceptanceCodebase(t, liveHarness, repository).GetId()
	fixturePaths := writeRenameFixtures(t, repository, totalRenamed)
	renameAwaitFixtureRows(t, liveHarness, collectionName, codebaseID, fixturePaths)

	first := 0
	for round, count := range renameCounts {
		time.Sleep(renameFlushWindow)
		measurement := measureRenameRound(t, liveHarness, repository, collectionName, codebaseID, round, first, count)
		appendRenameReport(t, measurement)
		if measurement.JobState != "completed" {
			t.Errorf("The converge job after the rename of %d files finished with state %q instead of completed.", count, measurement.JobState)
		}
		if !measurement.CopyPathUsed {
			t.Errorf("The test renamed %d files, but converge_copy_chunks_total increased by %d. The increase must equal the number of renamed files. A smaller increase means the converge job embedded some renamed files again instead of copying their rows.", count, measurement.CopyChunksDelta)
		}
		if measurement.RowsAfter != measurement.RowsBefore {
			t.Errorf("After the test renamed %d files, the collection row count changed from %d to %d.", count, measurement.RowsBefore, measurement.RowsAfter)
		}
		if !measurement.SearchReturnsNewPath || measurement.SearchReturnsOldPath {
			t.Errorf("The search after the rename of %d files reported %t for the presence of the new path and %t for the presence of the old path. The new path must appear, and the old path must not appear.", count, measurement.SearchReturnsNewPath, measurement.SearchReturnsOldPath)
		}
		first += count
	}
}

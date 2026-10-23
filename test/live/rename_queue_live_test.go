//go:build live

package live

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/metrics"
)

const (
	renameQueueReportVariable   = "LMS_RENAME_QUEUE_REPORT"
	renameQueueRenamedCount     = 10
	renameQueueUntouchedCount   = 40
	renameQueueFunctionsPerFile = 2
	renameQueueBlockerCount     = 6
	renameQueueSyncIntervalMS   = 10000
	renameQueueMarker           = "renamequeue"
	renameQueueBlockerMarker    = "blockerqueue"
	renameQueueSyncClient       = "daemon-sync"
	renameQueueWatcherClient    = "daemon-watcher"
	renameQueueStepTimeout      = 2 * time.Minute
	renameQueueSettleTime       = 6 * time.Second
	renameQueueSweepGrace       = time.Second
	renameQueueGatePoll         = 10 * time.Millisecond
)

type renameQueueMode string

const (
	renameQueueEmpty    renameQueueMode = "empty-queue"
	renameQueueBusy     renameQueueMode = "busy-queue"
	renameQueueSyncOnly renameQueueMode = "sync-only"
)

type renameQueueScenario struct {
	mode    renameQueueMode
	gitRepo bool
}

func (scenario renameQueueScenario) name() string {
	if scenario.gitRepo {
		return string(scenario.mode) + "-git"
	}
	return string(scenario.mode) + "-plain"
}

type renameQueueJob struct {
	ID             string `json:"id"`
	Client         string `json:"client"`
	Operation      string `json:"operation"`
	State          string `json:"state"`
	QueueSequence  uint64 `json:"queue_sequence"`
	FilesTotal     int32  `json:"files_total"`
	FilesProcessed int32  `json:"files_processed"`
	ChunksEmbedded int32  `json:"chunks_embedded"`
	ChunksReused   int32  `json:"chunks_reused"`
	StartedAt      string `json:"started_at"`
	CompletedAt    string `json:"completed_at"`
}

type renameQueueMeasurement struct {
	Scenario                string           `json:"scenario"`
	DatabaseName            string           `json:"database_name"`
	RenamedFiles            int              `json:"renamed_files"`
	RenamedChunks           int              `json:"renamed_chunks"`
	QueueAtRename           []renameQueueJob `json:"queue_at_rename"`
	QueueAtSweep            []renameQueueJob `json:"queue_at_sweep"`
	JobsAfterRename         []renameQueueJob `json:"jobs_after_rename"`
	EmbeddedRenamedInputs   int              `json:"embedded_renamed_inputs"`
	EmbeddedBlockerInputs   int              `json:"embedded_blocker_inputs"`
	ConvergeCopyChunksDelta int64            `json:"converge_copy_chunks_delta"`
	ConvergeUpsertDelta     int64            `json:"converge_upsert_delta"`
	SweepRunsDelta          int64            `json:"sweep_runs_delta"`
	SyncSkippedDelta        int64            `json:"sync_skipped_inflight_delta"`
	InsertCalls             int              `json:"insert_calls"`
	DeleteCalls             int              `json:"delete_calls"`
	DestinationRows         int              `json:"destination_rows"`
	SourceRows              int              `json:"source_rows"`
	DestinationsRecorded    int              `json:"destinations_recorded"`
}

// renameQueueGate wraps the live harness embedGate. Its goroutine answers each
// embed request immediately unless the atomic field holding is true. When
// holding is true, the goroutine sets held to 1 and waits until holding becomes
// false before answering the request.
type renameQueueGate struct {
	gate    *embedGate
	holding atomic.Bool
	held    atomic.Int64
}

func newRenameQueueGate() *renameQueueGate {
	controller := &renameQueueGate{gate: &embedGate{arrived: make(chan int), release: make(chan struct{})}}
	go controller.run()
	return controller
}

func (controller *renameQueueGate) run() {
	for range controller.gate.arrived {
		for controller.holding.Load() {
			controller.held.Store(1)
			time.Sleep(renameQueueGatePoll)
		}
		controller.held.Store(0)
		controller.gate.release <- struct{}{}
	}
}

func runRenameQueueScenario(t *testing.T, scenario renameQueueScenario) renameQueueMeasurement {
	t.Helper()
	controller := newRenameQueueGate()
	liveHarness := newHarnessWithDaemonOptions(t, controller.gate, 0, true, liveDaemonOptions{
		embedder:       harnessEmbedderFake,
		fileWatcher:    scenario.mode != renameQueueSyncOnly,
		backgroundSync: true,
		syncIntervalMS: renameQueueSyncIntervalMS,
	})
	t.Logf("Scenario %s uses temporary Milvus database %s.", scenario.name(), liveHarness.databaseName)

	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatalf("resolve temporary directory: %v", err)
	}
	repository := filepath.Join(base, "renames")
	blocker := filepath.Join(base, "blocker")
	writeRenameQueueRepository(t, repository, scenario.gitRepo)
	writeRenameQueueBlocker(t, blocker)
	collectionName := liveHarness.trackCodebasePath(repository)
	liveHarness.trackCodebasePath(blocker)

	requirePublicCompleted(t, startRenameQueueIndex(t, liveHarness, repository), "repository build")
	codebaseID := renameQueueCodebaseID(t, liveHarness, repository)
	measurement := renameQueueMeasurement{
		Scenario:      scenario.name(),
		DatabaseName:  liveHarness.databaseName,
		RenamedFiles:  renameQueueRenamedCount,
		RenamedChunks: renameQueueRowCount(t, liveHarness, collectionName, renameQueueSourcePaths()),
	}

	if scenario.mode != renameQueueBusy {
		requirePublicCompleted(t, startRenameQueueIndex(t, liveHarness, blocker), "blocker build")
	}
	waitForRenameQueueSweep(t)
	waitForRenameQueueIdle(t, liveHarness, codebaseID)

	knownJobs := publicWatcherJobIDs(t, liveHarness)
	embeddingMark := len(liveHarness.embeddingRecorder.snapshot())
	callMark := len(liveHarness.callRecorder.snapshot())
	before := metrics.Read()

	var blockerJobID string
	if scenario.mode == renameQueueBusy {
		controller.holding.Store(true)
		blockerJobID = startRenameQueueIndexAsync(t, liveHarness, blocker)
		waitForRenameQueueCondition(t, "the blocker build to wait on its first embed request", func() bool {
			return controller.held.Load() == 1
		})
	}

	waitForRenameQueueSweep(t)
	renameQueueMoveFiles(t, repository, scenario.gitRepo)
	measurement.QueueAtRename = renameQueueJobsSince(t, liveHarness, knownJobs)

	switch scenario.mode {
	case renameQueueBusy:
		waitForRenameQueueCondition(t, "a queued watcher converge for the renamed files", func() bool {
			return renameQueueHasJob(t, liveHarness, codebaseID, knownJobs, renameQueueWatcherClient, "queued")
		})
		waitForRenameQueueSweep(t)
		time.Sleep(renameQueueSweepGrace)
		measurement.QueueAtSweep = renameQueueJobsSince(t, liveHarness, knownJobs)
		controller.holding.Store(false)
		requirePublicCompleted(t, waitForRenameQueueJob(t, liveHarness, blockerJobID), "blocker build")
	case renameQueueSyncOnly:
		waitForRenameQueueCondition(t, "a scheduled sync for the renamed files", func() bool {
			return renameQueueHasJob(t, liveHarness, codebaseID, knownJobs, renameQueueSyncClient, "")
		})
	case renameQueueEmpty:
	}

	waitForRenameQueueDestinations(t, liveHarness, repository)
	time.Sleep(renameQueueSettleTime)
	waitForRenameQueueIdle(t, liveHarness, codebaseID)

	after := metrics.Read()
	measurement.JobsAfterRename = renameQueueJobsSince(t, liveHarness, knownJobs)
	for _, inputs := range liveHarness.embeddingRecorder.snapshot()[embeddingMark:] {
		for _, input := range inputs {
			if strings.Contains(input, renameQueueMarker) {
				measurement.EmbeddedRenamedInputs++
			}
			if strings.Contains(input, renameQueueBlockerMarker) {
				measurement.EmbeddedBlockerInputs++
			}
		}
	}
	measurement.ConvergeCopyChunksDelta = after.ConvergeCopyChunksTotal - before.ConvergeCopyChunksTotal
	measurement.ConvergeUpsertDelta = after.ConvergeUpsertTotal - before.ConvergeUpsertTotal
	measurement.SweepRunsDelta = after.SweepRunsTotal - before.SweepRunsTotal
	measurement.SyncSkippedDelta = after.SyncSkippedInflightTotal - before.SyncSkippedInflightTotal
	for _, call := range liveHarness.callRecorder.snapshot()[callMark:] {
		if !slices.Contains(call.collectionNames, collectionName) {
			continue
		}
		switch call.method {
		case "Insert":
			measurement.InsertCalls++
		case "Delete":
			measurement.DeleteCalls++
		}
	}
	measurement.DestinationRows = renameQueueRowCount(t, liveHarness, collectionName, renameQueueDestinationPaths())
	measurement.SourceRows = renameQueueRowCount(t, liveHarness, collectionName, renameQueueSourcePaths())
	snapshot := renameQueueSnapshot(t, liveHarness, repository)
	for _, destination := range renameQueueDestinationPaths() {
		if snapshot.HasFile(destination) {
			measurement.DestinationsRecorded++
		}
	}
	writeRenameQueueReport(t, measurement)
	return measurement
}

// requireRenameQueueReuse checks that the embedder did not receive inputs for
// renamed files. It checks that destination paths have the original total row
// count and source paths do not have rows. It also checks that the snapshot
// records every destination path.
func requireRenameQueueReuse(t *testing.T, measurement renameQueueMeasurement) {
	t.Helper()
	if measurement.EmbeddedRenamedInputs != 0 {
		t.Errorf("Scenario %s sent %d renamed-file inputs to the embedder. The test expected 0 inputs.", measurement.Scenario, measurement.EmbeddedRenamedInputs)
	}
	if measurement.DestinationRows != measurement.RenamedChunks {
		t.Errorf("Scenario %s stored %d rows for the destination paths. The test expected %d rows.", measurement.Scenario, measurement.DestinationRows, measurement.RenamedChunks)
	}
	if measurement.SourceRows != 0 {
		t.Errorf("Scenario %s kept %d rows for the source paths. The test expected 0 rows.", measurement.Scenario, measurement.SourceRows)
	}
	if measurement.DestinationsRecorded != measurement.RenamedFiles {
		t.Errorf("Scenario %s recorded %d destination paths in the snapshot. The test expected %d paths.", measurement.Scenario, measurement.DestinationsRecorded, measurement.RenamedFiles)
	}
}

func renameQueueSourceName(index int) string {
	return fmt.Sprintf("source/moved%02d.go", index)
}

func renameQueueDestinationName(index int) string {
	return fmt.Sprintf("destination/moved%02d.go", index)
}

func renameQueueSourcePaths() []string {
	paths := make([]string, 0, renameQueueRenamedCount)
	for index := range renameQueueRenamedCount {
		paths = append(paths, renameQueueSourceName(index))
	}
	return paths
}

func renameQueueDestinationPaths() []string {
	paths := make([]string, 0, renameQueueRenamedCount)
	for index := range renameQueueRenamedCount {
		paths = append(paths, renameQueueDestinationName(index))
	}
	return paths
}

func renameQueueFileContent(marker string, packageName string, fileIndex int) string {
	var builder strings.Builder
	fmt.Fprintf(&builder, "package %s\n\n", packageName)
	for functionIndex := range renameQueueFunctionsPerFile {
		name := fmt.Sprintf("Measure%s%02d%02d", marker, fileIndex, functionIndex)
		limit := fileIndex*100 + functionIndex
		fmt.Fprintf(&builder, "// %s counts readings above and at or below the limit %d, then returns the larger count minus the smaller count and two index numbers for marker %s.\n", name, limit, marker)
		fmt.Fprintf(&builder, "func %s(readings []int) (int, int) {\n\tabove := 0\n\tbelow := 0\n", name)
		fmt.Fprintf(&builder, "\tfor _, reading := range readings {\n\t\tif reading > %d {\n\t\t\tabove++\n\t\t\tcontinue\n\t\t}\n\t\tbelow++\n\t}\n", limit)
		fmt.Fprintf(&builder, "\tif above > below {\n\t\treturn above - below, %d\n\t}\n\treturn below - above, %d\n}\n\n", fileIndex, functionIndex)
	}
	return builder.String()
}

func writeRenameQueueFile(t *testing.T, path string, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("create directory for %s: %v", path, err)
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func writeRenameQueueRepository(t *testing.T, repository string, gitRepo bool) {
	t.Helper()
	for index := range renameQueueRenamedCount {
		writeRenameQueueFile(t, filepath.Join(repository, renameQueueSourceName(index)), renameQueueFileContent(renameQueueMarker, "source", index))
	}
	for index := range renameQueueUntouchedCount {
		writeRenameQueueFile(t, filepath.Join(repository, fmt.Sprintf("stable/stable%02d.go", index)), renameQueueFileContent("stablequeue", "stable", index))
	}
	if !gitRepo {
		return
	}
	renameQueueGit(t, repository, "init", "--quiet")
	renameQueueGit(t, repository, "add", "--all")
	renameQueueGit(t, repository, "-c", "commit.gpgsign=false", "commit", "--quiet", "--message", "fixture")
}

func writeRenameQueueBlocker(t *testing.T, blocker string) {
	t.Helper()
	for index := range renameQueueBlockerCount {
		writeRenameQueueFile(t, filepath.Join(blocker, fmt.Sprintf("blocker%02d.go", index)), renameQueueFileContent(renameQueueBlockerMarker, "blocker", index))
	}
}

func renameQueueGit(t *testing.T, directory string, arguments ...string) {
	t.Helper()
	command := exec.CommandContext(context.Background(), "git", append([]string{"-C", directory}, arguments...)...)
	command.Env = append(os.Environ(),
		"GIT_AUTHOR_NAME=rename-queue",
		"GIT_AUTHOR_EMAIL=rename-queue@example.invalid",
		"GIT_COMMITTER_NAME=rename-queue",
		"GIT_COMMITTER_EMAIL=rename-queue@example.invalid",
	)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s in %s: %v\n%s", strings.Join(arguments, " "), directory, err, output)
	}
}

func renameQueueMoveFiles(t *testing.T, repository string, gitRepo bool) {
	t.Helper()
	if err := os.MkdirAll(filepath.Join(repository, "destination"), 0o755); err != nil {
		t.Fatalf("create destination directory: %v", err)
	}
	for index := range renameQueueRenamedCount {
		source := renameQueueSourceName(index)
		destination := renameQueueDestinationName(index)
		if gitRepo {
			renameQueueGit(t, repository, "mv", source, destination)
			continue
		}
		if err := os.Rename(filepath.Join(repository, source), filepath.Join(repository, destination)); err != nil {
			t.Fatalf("rename %s to %s: %v", source, destination, err)
		}
	}
}

func startRenameQueueIndexAsync(t *testing.T, liveHarness *harness, path string) string {
	t.Helper()
	response, err := liveHarness.client.StartIndex(correlatedContext(), &pb.StartIndexRequest{
		Path:     path,
		Splitter: &pb.SplitterConfig{Type: "ast"},
		Client:   &pb.ClientInfo{Name: "rename-queue-live-harness"},
	})
	if err != nil {
		t.Fatalf("start index for %s: %v", path, err)
	}
	if response.GetJobId() == "" {
		t.Fatalf("start index for %s returned an empty job id", path)
	}
	return response.GetJobId()
}

func startRenameQueueIndex(t *testing.T, liveHarness *harness, path string) *pb.Job {
	t.Helper()
	return waitForRenameQueueJob(t, liveHarness, startRenameQueueIndexAsync(t, liveHarness, path))
}

func waitForRenameQueueJob(t *testing.T, liveHarness *harness, jobID string) *pb.Job {
	t.Helper()
	var finished *pb.Job
	waitForRenameQueueCondition(t, "job "+jobID+" to finish", func() bool {
		response, err := liveHarness.client.GetJob(correlatedContext(), &pb.GetJobRequest{JobId: jobID})
		if err != nil {
			t.Fatalf("get job %s: %v", jobID, err)
		}
		if response.GetJob() == nil || !publicJobTerminal(response.GetJob()) {
			return false
		}
		finished = response.GetJob()
		return true
	})
	return finished
}

func renameQueueCodebaseID(t *testing.T, liveHarness *harness, path string) string {
	t.Helper()
	response, err := liveHarness.client.GetIndex(correlatedContext(), &pb.GetIndexRequest{Path: path})
	if err != nil {
		t.Fatalf("get index for %s: %v", path, err)
	}
	codebaseID := response.GetCodebase().GetId()
	if codebaseID == "" {
		t.Fatalf("get index for %s returned an empty codebase id", path)
	}
	return codebaseID
}

func renameQueueSnapshot(t *testing.T, liveHarness *harness, path string) merkle.Snapshot {
	t.Helper()
	response, err := liveHarness.client.GetIndex(correlatedContext(), &pb.GetIndexRequest{Path: path})
	if err != nil {
		t.Fatalf("get index for %s: %v", path, err)
	}
	snapshot, err := merkle.ReadSnapshot(response.GetCodebase().GetMerkleSnapshotPath())
	if err != nil {
		t.Fatalf("read snapshot for %s: %v", path, err)
	}
	return snapshot
}

func renameQueueRowCount(t *testing.T, liveHarness *harness, collectionName string, paths []string) int {
	t.Helper()
	quoted := make([]string, 0, len(paths))
	for _, path := range paths {
		quoted = append(quoted, fmt.Sprintf("%q", path))
	}
	filter := relativePathField + " in [" + strings.Join(quoted, ",") + "]"
	return len(queryRowSnapshots(t, liveHarness, collectionName, filter))
}

func waitForRenameQueueCondition(t *testing.T, label string, condition func() bool) {
	t.Helper()
	deadline := time.Now().Add(renameQueueStepTimeout)
	for time.Now().Before(deadline) {
		if condition() {
			return
		}
		time.Sleep(jobPollInterval)
	}
	t.Fatalf("waited %s for %s", renameQueueStepTimeout, label)
}

func waitForRenameQueueSweep(t *testing.T) {
	t.Helper()
	start := metrics.Read().SweepRunsTotal
	waitForRenameQueueCondition(t, "a periodic sweep", func() bool {
		return metrics.Read().SweepRunsTotal > start
	})
}

func waitForRenameQueueIdle(t *testing.T, liveHarness *harness, codebaseID string) {
	t.Helper()
	waitForRenameQueueCondition(t, "every job of the renamed codebase to finish", func() bool {
		response, err := liveHarness.client.ListJobs(correlatedContext(), &pb.ListJobsRequest{CodebaseId: codebaseID})
		if err != nil {
			t.Fatalf("list jobs for %s: %v", codebaseID, err)
		}
		for _, job := range response.GetJobs() {
			if !publicJobTerminal(job) {
				return false
			}
		}
		return true
	})
}

func waitForRenameQueueDestinations(t *testing.T, liveHarness *harness, repository string) {
	t.Helper()
	waitForRenameQueueCondition(t, "the snapshot to record every destination path", func() bool {
		snapshot := renameQueueSnapshot(t, liveHarness, repository)
		for _, destination := range renameQueueDestinationPaths() {
			if !snapshot.HasFile(destination) {
				return false
			}
		}
		return true
	})
}

func renameQueueHasJob(t *testing.T, liveHarness *harness, codebaseID string, knownJobs map[string]struct{}, client string, state string) bool {
	t.Helper()
	response, err := liveHarness.client.ListJobs(correlatedContext(), &pb.ListJobsRequest{CodebaseId: codebaseID})
	if err != nil {
		t.Fatalf("list jobs for %s: %v", codebaseID, err)
	}
	for _, job := range response.GetJobs() {
		if _, known := knownJobs[job.GetId()]; known {
			continue
		}
		if job.GetClient().GetName() != client {
			continue
		}
		if state == "" || job.GetState() == state {
			return true
		}
	}
	return false
}

func renameQueueJobsSince(t *testing.T, liveHarness *harness, knownJobs map[string]struct{}) []renameQueueJob {
	t.Helper()
	response, err := liveHarness.client.ListJobs(correlatedContext(), &pb.ListJobsRequest{})
	if err != nil {
		t.Fatalf("list jobs: %v", err)
	}
	jobs := make([]renameQueueJob, 0, len(response.GetJobs()))
	for _, job := range response.GetJobs() {
		if _, known := knownJobs[job.GetId()]; known {
			continue
		}
		progress := job.GetProgress()
		recorded := renameQueueJob{
			ID:             job.GetId(),
			Client:         job.GetClient().GetName(),
			Operation:      job.GetOperation(),
			State:          job.GetState(),
			QueueSequence:  job.GetQueueSequence(),
			FilesTotal:     progress.GetFilesTotal(),
			FilesProcessed: progress.GetFilesProcessed(),
			ChunksEmbedded: progress.GetChunksEmbedded(),
			ChunksReused:   progress.GetChunksReused(),
		}
		if job.GetStartedAt() != nil {
			recorded.StartedAt = job.GetStartedAt().AsTime().Format(time.RFC3339Nano)
		}
		if job.GetCompletedAt() != nil {
			recorded.CompletedAt = job.GetCompletedAt().AsTime().Format(time.RFC3339Nano)
		}
		jobs = append(jobs, recorded)
	}
	slices.SortFunc(jobs, func(left renameQueueJob, right renameQueueJob) int {
		return int(left.QueueSequence) - int(right.QueueSequence)
	})
	return jobs
}

func writeRenameQueueReport(t *testing.T, measurement renameQueueMeasurement) {
	t.Helper()
	encoded, err := json.Marshal(measurement)
	if err != nil {
		t.Fatalf("encode measurement: %v", err)
	}
	t.Logf("measurement %s", encoded)
	reportPath := os.Getenv(renameQueueReportVariable)
	if reportPath == "" {
		return
	}
	report, err := os.OpenFile(reportPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatalf("open report %s: %v", reportPath, err)
	}
	if _, err := report.Write(append(encoded, '\n')); err != nil {
		_ = report.Close()
		t.Fatalf("write report %s: %v", reportPath, err)
	}
	if err := report.Close(); err != nil {
		t.Fatalf("close report %s: %v", reportPath, err)
	}
}

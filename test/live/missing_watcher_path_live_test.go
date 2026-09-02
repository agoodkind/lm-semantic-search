//go:build live

package live

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

const watcherJobTimeout = 30 * time.Second

type watcherIndexSnapshot struct {
	codebaseID      string
	collectionName  string
	checkpointPath  string
	checkpointBytes []byte
	checkpointInfo  os.FileInfo
	rowCount        int
	vectorChecksums []string
}

func TestWatcherRetainsRemovedPathAtPublicBoundary(t *testing.T) {
	harness := newWatcherHarness(t)
	root := t.TempDir()
	indexedPath := filepath.Join(root, "main.go")
	if err := os.WriteFile(indexedPath, []byte("package watcherfixture\n\nfunc Existing() string { return \"indexed\" }\n"), 0o600); err != nil {
		t.Fatalf("write indexed source: %v", err)
	}

	initial := startPublicCodebaseIndex(t, harness, root)
	requirePublicCompleted(t, initial, "initial codebase index")
	before := captureWatcherIndexSnapshot(t, harness, root)
	if before.rowCount == 0 {
		t.Fatal("initial codebase index wrote no rows")
	}
	watcherJobs := publicWatcherJobIDs(t, harness)

	for attempt := 1; attempt <= 2; attempt++ {
		job, duplicateNonFinalProcessed := triggerRemovedPathConverge(
			t,
			harness,
			root,
			watcherJobs,
			attempt,
		)
		requirePublicCompleted(t, job, fmt.Sprintf("watcher converge %d", attempt))
		assertWatcherConvergeResult(t, job)
		assertNoSemanticRemovalTrace(t, job)
		assertWatcherIndexSnapshot(t, harness, root, before)
		watcherJobs[job.GetId()] = struct{}{}
		t.Logf(
			"watcher converge %d job=%s duplicate_non_final_processed=%t",
			attempt,
			job.GetId(),
			duplicateNonFinalProcessed,
		)
	}
}

func startPublicCodebaseIndex(t *testing.T, harness *harness, root string) *pb.Job {
	t.Helper()
	response, err := harness.client.StartIndex(
		correlatedContext(),
		&pb.StartIndexRequest{
			Path: root,
			Splitter: &pb.SplitterConfig{
				Type: "ast",
			},
			Client: &pb.ClientInfo{Name: "missing-watcher-live-harness"},
		},
	)
	if err != nil {
		t.Fatalf("start codebase index: %v", err)
	}
	if response.GetJobId() == "" {
		t.Fatal("start codebase index returned an empty job id")
	}
	return waitForCodebasePublicJob(t, harness, response.GetJobId())
}

func triggerRemovedPathConverge(
	t *testing.T,
	harness *harness,
	root string,
	knownJobs map[string]struct{},
	attempt int,
) (*pb.Job, bool) {
	t.Helper()
	path := filepath.Join(root, fmt.Sprintf("removed-%d.go", attempt))
	startedAt := time.Now()
	if err := os.WriteFile(path, []byte("package watcherfixture\n"), 0o600); err != nil {
		t.Fatalf("write transient source %d: %v", attempt, err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatalf("remove transient source %d: %v", attempt, err)
	}
	if time.Since(startedAt) >= 2*time.Second {
		t.Fatalf("transient source %d outlived the watcher debounce window", attempt)
	}
	return waitForNewWatcherJob(t, harness, knownJobs)
}

func waitForCodebasePublicJob(t *testing.T, harness *harness, jobID string) *pb.Job {
	t.Helper()
	deadline := time.Now().Add(watcherJobTimeout)
	for time.Now().Before(deadline) {
		response, err := harness.client.GetJob(
			correlatedContext(),
			&pb.GetJobRequest{JobId: jobID},
		)
		if err != nil {
			t.Fatalf("get public job %s: %v", jobID, err)
		}
		job := response.GetJob()
		if job == nil {
			t.Fatalf("get public job %s returned no job", jobID)
		}
		if publicJobTerminal(job) {
			return job
		}
		time.Sleep(jobPollInterval)
	}
	t.Fatalf("public job %s did not finish within %s", jobID, watcherJobTimeout)
	return nil
}

func waitForNewWatcherJob(
	t *testing.T,
	harness *harness,
	knownJobs map[string]struct{},
) (*pb.Job, bool) {
	t.Helper()
	deadline := time.Now().Add(watcherJobTimeout)
	watcherJobID := ""
	var lastUpdatedAt time.Time
	var lastProcessed int32
	observedVersion := false
	duplicateNonFinalProcessed := false
	for time.Now().Before(deadline) {
		if watcherJobID == "" {
			response, err := harness.client.ListJobs(
				correlatedContext(),
				&pb.ListJobsRequest{},
			)
			if err != nil {
				t.Fatalf("list public watcher jobs: %v", err)
			}
			for _, job := range response.GetJobs() {
				if _, known := knownJobs[job.GetId()]; known {
					continue
				}
				if job.GetOperation() == "converge" &&
					job.GetClient().GetName() == "daemon-watcher" {
					watcherJobID = job.GetId()
					break
				}
			}
		}
		if watcherJobID != "" {
			response, err := harness.client.GetJob(
				correlatedContext(),
				&pb.GetJobRequest{JobId: watcherJobID},
			)
			if err != nil {
				t.Fatalf("get watcher job %s: %v", watcherJobID, err)
			}
			job := response.GetJob()
			if job == nil {
				t.Fatalf("get watcher job %s returned no job", watcherJobID)
			}
			updatedAt := job.GetUpdatedAt().AsTime()
			processed := job.GetProgress().GetFilesProcessed()
			if observedVersion && !updatedAt.Equal(lastUpdatedAt) &&
				processed == lastProcessed && processed > 0 &&
				processed < job.GetProgress().GetFilesTotal() {
				duplicateNonFinalProcessed = true
			}
			if !updatedAt.Equal(lastUpdatedAt) {
				lastUpdatedAt = updatedAt
				lastProcessed = processed
				observedVersion = true
			}
			if publicJobTerminal(job) {
				return job, duplicateNonFinalProcessed
			}
		}
		time.Sleep(jobPollInterval)
	}
	t.Fatal("watcher converge job did not finish within the public timeout")
	return nil, false
}

func publicWatcherJobIDs(t *testing.T, harness *harness) map[string]struct{} {
	t.Helper()
	response, err := harness.client.ListJobs(correlatedContext(), &pb.ListJobsRequest{})
	if err != nil {
		t.Fatalf("list public jobs before watcher event: %v", err)
	}
	jobIDs := make(map[string]struct{}, len(response.GetJobs()))
	for _, job := range response.GetJobs() {
		jobIDs[job.GetId()] = struct{}{}
	}
	return jobIDs
}

func publicJobTerminal(job *pb.Job) bool {
	switch job.GetState() {
	case "completed", "failed", "cancelled":
		return true
	default:
		return false
	}
}

func requirePublicCompleted(t *testing.T, job *pb.Job, label string) {
	t.Helper()
	if job.GetState() == "completed" {
		return
	}
	t.Fatalf("%s state = %q, want completed: %s", label, job.GetState(), job.GetDisplayError())
}

func assertWatcherConvergeResult(t *testing.T, job *pb.Job) {
	t.Helper()
	progress := job.GetProgress()
	if progress.GetUnit() != "path" {
		t.Fatalf("watcher progress unit = %q, want path", progress.GetUnit())
	}
	if progress.GetFilesTotal() != 1 || progress.GetFilesProcessed() != 1 {
		t.Fatalf(
			"watcher progress = %d of %d, want 1 of 1 path",
			progress.GetFilesProcessed(),
			progress.GetFilesTotal(),
		)
	}
	if progress.GetChunksEmbedded() != 0 {
		t.Fatalf("watcher embedded chunks = %d, want 0", progress.GetChunksEmbedded())
	}
	if !progress.GetHeartbeatAt().AsTime().After(job.GetStartedAt().AsTime()) {
		t.Fatalf(
			"watcher heartbeat = %s, want after start %s",
			progress.GetHeartbeatAt().AsTime(),
			job.GetStartedAt().AsTime(),
		)
	}
}

func assertNoSemanticRemovalTrace(t *testing.T, job *pb.Job) {
	t.Helper()
	breakdown := job.GetProgress().GetBreakdown()
	for _, rows := range [][]*pb.OutcomeRow{breakdown.GetFileRows(), breakdown.GetChunkRows()} {
		for _, row := range rows {
			if row.GetKind() == pb.OutcomeKind_OUTCOME_KIND_REMOVED {
				t.Fatalf("watcher job reported semantic removal: %+v", breakdown)
			}
		}
	}
}

func captureWatcherIndexSnapshot(
	t *testing.T,
	harness *harness,
	root string,
) watcherIndexSnapshot {
	t.Helper()
	response, err := harness.client.GetIndex(
		correlatedContext(),
		&pb.GetIndexRequest{Path: root},
	)
	if err != nil {
		t.Fatalf("get public codebase index: %v", err)
	}
	codebase := response.GetCodebase()
	if codebase == nil || codebase.GetCollectionName() == "" {
		t.Fatal("public codebase index omitted collection identity")
	}
	checkpointPath := codebase.GetMerkleSnapshotPath()
	if checkpointPath == "" {
		t.Fatal("public codebase index omitted checkpoint identity")
	}
	checkpointBytes, err := os.ReadFile(checkpointPath)
	if err != nil {
		t.Fatalf("read checkpoint %s: %v", checkpointPath, err)
	}
	checkpointInfo, err := os.Stat(checkpointPath)
	if err != nil {
		t.Fatalf("stat checkpoint %s: %v", checkpointPath, err)
	}
	rows := queryRowSnapshots(t, harness, codebase.GetCollectionName(), `id != ""`)
	checksums := make([]string, 0, len(rows))
	for _, row := range rows {
		checksums = append(checksums, row.vectorChecksum)
	}
	slices.Sort(checksums)
	harness.trackCollectionFamily(codebase.GetCollectionName())
	return watcherIndexSnapshot{
		codebaseID:      codebase.GetId(),
		collectionName:  codebase.GetCollectionName(),
		checkpointPath:  checkpointPath,
		checkpointBytes: checkpointBytes,
		checkpointInfo:  checkpointInfo,
		rowCount:        len(rows),
		vectorChecksums: checksums,
	}
}

func assertWatcherIndexSnapshot(
	t *testing.T,
	harness *harness,
	root string,
	before watcherIndexSnapshot,
) {
	t.Helper()
	after := captureWatcherIndexSnapshot(t, harness, root)
	if after.codebaseID != before.codebaseID || after.collectionName != before.collectionName {
		t.Fatalf("codebase identity changed: before=%+v after=%+v", before, after)
	}
	if after.checkpointPath != before.checkpointPath {
		t.Fatalf("checkpoint path changed: before=%s after=%s", before.checkpointPath, after.checkpointPath)
	}
	if !os.SameFile(before.checkpointInfo, after.checkpointInfo) {
		t.Fatal("watcher replaced the checkpoint file")
	}
	if !bytes.Equal(before.checkpointBytes, after.checkpointBytes) {
		t.Fatal("watcher changed checkpoint bytes")
	}
	if after.rowCount != before.rowCount {
		t.Fatalf("collection rows changed: before=%d after=%d", before.rowCount, after.rowCount)
	}
	if !slices.Equal(after.vectorChecksums, before.vectorChecksums) {
		t.Fatalf(
			"collection vector SHA-256 values changed: before=%v after=%v",
			before.vectorChecksums,
			after.vectorChecksums,
		)
	}
}

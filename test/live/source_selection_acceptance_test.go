//go:build live && acceptance

package live

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

const (
	selectionCasesVariable  = "LMS_SOURCE_SELECTION_CASES"
	selectionReportVariable = "LMS_SOURCE_SELECTION_REPORT"
	selectionMemoryInterval = 50 * time.Millisecond
	selectionJobTimeout     = 3 * time.Hour
	selectionVectorBytes    = 4

	selectionRuleAutomatic   = "automatic"
	selectionRuleMainFirst   = "main_worktree_first"
	selectionRuleLatestFirst = "latest_completed_run_first"
	selectionRuleNear        = "nearest_commit"
)

type selectionCase struct {
	Name         string `json:"name"`
	Repository   string `json:"repository"`
	MainCommit   string `json:"main_commit"`
	NearCommit   string `json:"near_commit"`
	StaleCommit  string `json:"stale_commit"`
	TargetCommit string `json:"target_commit"`
}

type selectionMeasurement struct {
	Case                   string `json:"case"`
	Rule                   string `json:"rule"`
	SourceRole             string `json:"source_role"`
	SourcePath             string `json:"source_path"`
	SelectionMilliseconds  int64  `json:"selection_ms,omitempty"`
	EmbeddingProvider      string `json:"embedding_provider"`
	EmbeddingModel         string `json:"embedding_model"`
	EmbeddingDimension     int32  `json:"embedding_dimension"`
	ChunksReused           int32  `json:"chunks_reused"`
	ChunksEmbedded         int32  `json:"chunks_embedded"`
	TotalChunks            int32  `json:"total_chunks"`
	ReuseVectorsLoaded     int32  `json:"reuse_vectors_loaded"`
	ReuseVectorBytesLoaded int64  `json:"reuse_vector_bytes_loaded"`
	SourceQueryCalls       int    `json:"source_query_calls"`
	FirstFileMilliseconds  int64  `json:"first_file_ms"`
	TotalMilliseconds      int64  `json:"total_ms"`
	HeapInuseBeforeBytes   uint64 `json:"heap_inuse_before_bytes"`
	HeapInusePeakBytes     uint64 `json:"heap_inuse_peak_bytes"`
	SourceBuildMS          int64  `json:"source_builds_ms,omitempty"`
}

type selectionWorktrees struct {
	main   string
	near   string
	stale  string
	target string
}

type heapSampler struct {
	mutex sync.Mutex
	peak  uint64
	stop  chan struct{}
	done  chan struct{}
}

func startHeapSampler() *heapSampler {
	sampler := &heapSampler{stop: make(chan struct{}), done: make(chan struct{})}
	go func() {
		defer close(sampler.done)
		ticker := time.NewTicker(selectionMemoryInterval)
		defer ticker.Stop()
		for {
			var stats runtime.MemStats
			runtime.ReadMemStats(&stats)
			sampler.mutex.Lock()
			sampler.peak = max(sampler.peak, stats.HeapInuse)
			sampler.mutex.Unlock()
			select {
			case <-sampler.stop:
				return
			case <-ticker.C:
			}
		}
	}()
	return sampler
}

func (sampler *heapSampler) finish() uint64 {
	close(sampler.stop)
	<-sampler.done
	sampler.mutex.Lock()
	defer sampler.mutex.Unlock()
	return sampler.peak
}

func currentHeapInuse() uint64 {
	runtime.GC()
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	return stats.HeapInuse
}

func readSelectionCases(t *testing.T) []selectionCase {
	t.Helper()

	casesPath := os.Getenv(selectionCasesVariable)
	if casesPath == "" {
		t.Skipf("%s is unset. Set it to the path of a JSON case file.", selectionCasesVariable)
	}
	encoded, err := os.ReadFile(casesPath)
	if err != nil {
		t.Fatalf("Reading case file %s failed with error %v.", casesPath, err)
	}
	var cases []selectionCase
	if err := json.Unmarshal(encoded, &cases); err != nil {
		t.Fatalf("Decoding the JSON case list from file %s failed with error %v.", casesPath, err)
	}
	return cases
}

func newSelectionWorktrees(t *testing.T, selection selectionCase) selectionWorktrees {
	t.Helper()

	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatalf("Resolving symlinks of the temporary directory failed with error %v.", err)
	}
	main := filepath.Join(root, "main")
	acceptanceGit(t, root, "clone", "--quiet", "--no-checkout", selection.Repository, main)
	acceptanceGit(t, main, "checkout", "--quiet", "--detach", selection.MainCommit)
	worktrees := selectionWorktrees{
		main:   main,
		near:   filepath.Join(root, "near"),
		stale:  filepath.Join(root, "stale"),
		target: filepath.Join(root, "target"),
	}
	acceptanceGit(t, main, "worktree", "add", "--quiet", "--detach", worktrees.near, selection.NearCommit)
	acceptanceGit(t, main, "worktree", "add", "--quiet", "--detach", worktrees.stale, selection.StaleCommit)
	return worktrees
}

func selectionStartIndex(t *testing.T, liveHarness *harness, path string, sourcePath string) *pb.Job {
	t.Helper()

	requestedAt := time.Now()
	response := acceptanceStartIndex(t, liveHarness, path, sourcePath, "")
	if response.GetJobId() == "" {
		t.Fatalf("StartIndex for path %s with source path %q returned no job identifier. The response text is %s.", path, sourcePath, response.GetDisplayText())
	}
	return selectionAwaitJob(t, liveHarness, response.GetJobId(), requestedAt).job
}

func selectionAwaitJob(t *testing.T, liveHarness *harness, jobID string, requestedAt time.Time) acceptanceJobTiming {
	t.Helper()

	var timing acceptanceJobTiming
	deadline := time.Now().Add(selectionJobTimeout)
	for time.Now().Before(deadline) {
		response, err := liveHarness.client.GetJob(correlatedContext(), &pb.GetJobRequest{JobId: jobID})
		if err != nil {
			t.Fatalf("GetJob for job %s returned error %v.", jobID, err)
		}
		job := response.GetJob()
		if !timing.firstFileSeen && job.GetProgress().GetFilesProcessed() > 0 {
			timing.firstFileSeen = true
			timing.firstFile = job.GetUpdatedAt().AsTime().Sub(job.GetStartedAt().AsTime())
		}
		if publicJobTerminal(job) {
			if job.GetState() != "completed" {
				t.Fatalf("Job %s finished with state %q instead of completed. The job display error is %s.", jobID, job.GetState(), job.GetDisplayError())
			}
			timing.job = job
			timing.totalObserved = time.Since(requestedAt)
			return timing
		}
		time.Sleep(acceptanceProgressPollInterval)
	}
	t.Fatalf("Job %s did not finish as completed, failed, or cancelled within %s.", jobID, selectionJobTimeout)
	return timing
}

func selectionAwaitAutomaticJob(t *testing.T, liveHarness *harness, codebaseID string, requestedAt time.Time) acceptanceJobTiming {
	t.Helper()

	deadline := time.Now().Add(selectionJobTimeout)
	for time.Now().Before(deadline) {
		response, err := liveHarness.client.ListJobs(correlatedContext(), &pb.ListJobsRequest{CodebaseId: codebaseID})
		if err != nil {
			t.Fatalf("ListJobs for discovered codebase %s returned error %v.", codebaseID, err)
		}
		if jobs := response.GetJobs(); len(jobs) > 0 {
			return selectionAwaitJob(t, liveHarness, jobs[0].GetId(), requestedAt)
		}
		time.Sleep(acceptanceProgressPollInterval)
	}
	t.Fatalf("No job for discovered codebase %s appeared within %s.", codebaseID, selectionJobTimeout)
	return acceptanceJobTiming{}
}

func selectionRecord(
	selection selectionCase,
	rule string,
	sourceRole string,
	sourcePath string,
	timing acceptanceJobTiming,
	sourceQueryCalls int,
	heapBefore uint64,
	heapPeak uint64,
) selectionMeasurement {
	job := timing.job
	progress := job.GetProgress()
	dimension := job.GetConfig().GetEmbeddingDimension()
	return selectionMeasurement{
		Case:                   selection.Name,
		Rule:                   rule,
		SourceRole:             sourceRole,
		SourcePath:             sourcePath,
		EmbeddingProvider:      job.GetConfig().GetEmbeddingProvider(),
		EmbeddingModel:         job.GetConfig().GetEmbeddingModel(),
		EmbeddingDimension:     dimension,
		ChunksReused:           progress.GetChunksReused(),
		ChunksEmbedded:         progress.GetChunksEmbedded(),
		TotalChunks:            progress.GetChunksReused() + progress.GetChunksEmbedded(),
		ReuseVectorsLoaded:     progress.GetReuseVectorsLoaded(),
		ReuseVectorBytesLoaded: int64(progress.GetReuseVectorsLoaded()) * int64(dimension) * selectionVectorBytes,
		SourceQueryCalls:       sourceQueryCalls,
		FirstFileMilliseconds:  timing.firstFile.Milliseconds(),
		TotalMilliseconds:      job.GetCompletedAt().AsTime().Sub(job.GetStartedAt().AsTime()).Milliseconds(),
		HeapInuseBeforeBytes:   heapBefore,
		HeapInusePeakBytes:     heapPeak,
	}
}

func appendSelectionReport(t *testing.T, measurement selectionMeasurement) {
	t.Helper()

	encoded, err := json.Marshal(measurement)
	if err != nil {
		t.Fatalf("Encoding the selection measurement as JSON failed with error %v.", err)
	}
	t.Logf("The selection measurement in JSON is %s.", encoded)
	reportPath := os.Getenv(selectionReportVariable)
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

func sourceRoleFor(worktrees selectionWorktrees, sourcePath string) string {
	switch sourcePath {
	case worktrees.main:
		return "main"
	case worktrees.near:
		return "near"
	case worktrees.stale:
		return "stale"
	case "":
		return "none"
	default:
		return "other"
	}
}

func measureExplicitSource(
	t *testing.T,
	liveHarness *harness,
	selection selectionCase,
	worktrees selectionWorktrees,
	rule string,
	sourcePath string,
) selectionMeasurement {
	t.Helper()

	clearResponse, err := liveHarness.client.ClearIndex(correlatedContext(), &pb.ClearIndexRequest{Path: worktrees.target, Client: acceptanceClientInfo()})
	if err != nil {
		t.Fatalf("ClearIndex for target path %s returned error %v. The response text is %s.", worktrees.target, err, clearResponse.GetDisplayText())
	}
	sourceCollection := liveHarness.trackCodebasePath(sourcePath)
	heapBefore := currentHeapInuse()
	sampler := startHeapSampler()
	mark := acceptanceCallMark(liveHarness)
	requestedAt := time.Now()
	response := acceptanceStartIndex(t, liveHarness, worktrees.target, sourcePath, "")
	if response.GetJobId() == "" {
		t.Fatalf("StartIndex for target path %s with explicit source path %s returned no job identifier. The response text is %s.", worktrees.target, sourcePath, response.GetDisplayText())
	}
	timing := selectionAwaitJob(t, liveHarness, response.GetJobId(), requestedAt)
	heapPeak := sampler.finish()
	queryCalls := renameCountCalls(acceptanceCallsSince(liveHarness, mark), "Query", sourceCollection)
	return selectionRecord(selection, rule, sourceRoleFor(worktrees, sourcePath), sourcePath, timing, queryCalls, heapBefore, heapPeak)
}

func TestAcceptanceSourceSelectionMeasurement(t *testing.T) {
	for _, selection := range readSelectionCases(t) {
		t.Run(selection.Name, func(t *testing.T) {
			liveHarness := newMeasurementHarness(t, false)
			t.Logf("This test run uses temporary Milvus database %s.", liveHarness.databaseName)
			worktrees := newSelectionWorktrees(t, selection)
			for _, path := range []string{worktrees.main, worktrees.near, worktrees.stale} {
				liveHarness.trackCodebasePath(path)
			}

			sourceBuildsStartedAt := time.Now()
			selectionStartIndex(t, liveHarness, worktrees.main, "")
			selectionStartIndex(t, liveHarness, worktrees.near, worktrees.main)
			selectionStartIndex(t, liveHarness, worktrees.stale, worktrees.main)
			sourceBuilds := time.Since(sourceBuildsStartedAt)
			acceptanceGit(t, worktrees.stale, "checkout", "--quiet", "--detach", selection.TargetCommit)
			acceptanceGit(t, worktrees.main, "worktree", "add", "--quiet", "--detach", worktrees.target, selection.TargetCommit)
			liveHarness.trackCodebasePath(worktrees.target)

			heapBefore := currentHeapInuse()
			sampler := startHeapSampler()
			mark := acceptanceCallMark(liveHarness)
			selectionStartedAt := time.Now()
			discovered, err := liveHarness.client.GetIndex(correlatedContext(), &pb.GetIndexRequest{Path: worktrees.target, Client: acceptanceClientInfo()})
			selectionDuration := time.Since(selectionStartedAt)
			if err != nil {
				t.Fatalf("GetIndex for target path %s returned error %v.", worktrees.target, err)
			}
			discoveredID := discovered.GetCodebase().GetId()
			if discoveredID == "" {
				t.Fatalf("GetIndex for target path %s returned no codebase identifier. The response text is %s.", worktrees.target, discovered.GetDisplayText())
			}
			automaticSource := discovered.GetCodebase().GetWorktreeSourcePath()
			timing := selectionAwaitAutomaticJob(t, liveHarness, discoveredID, selectionStartedAt)
			heapPeak := sampler.finish()
			queryCalls := 0
			if automaticSource != "" {
				automaticCollection := liveHarness.trackCodebasePath(automaticSource)
				queryCalls = renameCountCalls(acceptanceCallsSince(liveHarness, mark), "Query", automaticCollection)
			}
			automatic := selectionRecord(selection, selectionRuleAutomatic, sourceRoleFor(worktrees, automaticSource), automaticSource, timing, queryCalls, heapBefore, heapPeak)
			automatic.SelectionMilliseconds = selectionDuration.Milliseconds()
			automatic.SourceBuildMS = sourceBuilds.Milliseconds()
			appendSelectionReport(t, automatic)
			if automaticSource != worktrees.near {
				t.Errorf("GetIndex recorded automatic reuse source path %q instead of the expected near worktree path %q.", automaticSource, worktrees.near)
			}

			if os.Getenv("LMS_SOURCE_SELECTION_AUTOMATIC_ONLY") != "" {
				return
			}
			for _, rule := range []struct {
				name   string
				source string
			}{
				{name: selectionRuleMainFirst, source: worktrees.main},
				{name: selectionRuleLatestFirst, source: worktrees.stale},
				{name: selectionRuleNear, source: worktrees.near},
			} {
				appendSelectionReport(t, measureExplicitSource(t, liveHarness, selection, worktrees, rule.name, rule.source))
			}
		})
	}
}

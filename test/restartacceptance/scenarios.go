//go:build restartacceptance

package restartacceptance

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
	daemonclient "goodkind.io/lm-semantic-search/client"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	defaultScenarioFailureTimeout  = 15 * time.Second
	defaultScenarioRecoveryTimeout = 2 * time.Minute
	defaultScenarioReadyTimeout    = 30 * time.Second
	defaultScenarioPollInterval    = 100 * time.Millisecond
)

type scenarioTimeouts struct {
	Failure  time.Duration
	Recovery time.Duration
	Ready    time.Duration
	Poll     time.Duration
}

func (timeouts scenarioTimeouts) resolved() scenarioTimeouts {
	if timeouts.Failure <= 0 {
		timeouts.Failure = defaultScenarioFailureTimeout
	}
	if timeouts.Recovery <= 0 {
		timeouts.Recovery = defaultScenarioRecoveryTimeout
	}
	if timeouts.Ready <= 0 {
		timeouts.Ready = defaultScenarioReadyTimeout
	}
	if timeouts.Poll <= 0 {
		timeouts.Poll = defaultScenarioPollInterval
	}
	return timeouts
}

type rowSnapshotEntry struct {
	ID              string
	RelativePath    string
	StartLine       int
	EndLine         int
	SplitPosition   int
	EmbeddingModel  string
	DenseVectorHash string
}

type rowSnapshot struct {
	Entries map[string]rowSnapshotEntry
}

type checkpointSnapshot struct {
	FileHashes     map[string]string
	CompletedCount int
}

type rowSnapshotObserver func(context.Context) (rowSnapshot, error)
type checkpointSnapshotObserver func(context.Context) (checkpointSnapshot, error)
type embeddingCallObserver func() []string

type scenarioAInput struct {
	Client                pb.SemanticSearchDaemonServiceClient
	Path                  string
	Proxy                 *embeddingProxy
	ExpectedUnfinishedIDs []string
	ObserveRows           rowSnapshotObserver
	ObserveCheckpoint     checkpointSnapshotObserver
	ObserveEmbeddingCalls embeddingCallObserver
	Recorder              *evidenceRecorder
	Timeouts              scenarioTimeouts
}

type scenarioAResult struct {
	FailureCode           string
	Retryable             bool
	FailureElapsed        time.Duration
	Rows                  int
	UniqueRows            int
	CheckpointChunks      int32
	ResumedChunksEmbedded int32
	EmbeddedAfterFault    []string
}

func runScenarioA(ctx context.Context, input scenarioAInput) (scenarioAResult, error) {
	if input.Recorder == nil {
		return scenarioAResult{}, fmt.Errorf("scenario A requires an evidence recorder")
	}
	if input.Client == nil || input.Proxy == nil || input.ObserveRows == nil ||
		input.ObserveCheckpoint == nil || input.ObserveEmbeddingCalls == nil {
		return scenarioAResult{}, fmt.Errorf("scenario A requires daemon, proxy, row, checkpoint, and embedding observers")
	}
	timeouts := input.Timeouts.resolved()
	started, err := input.Client.StartIndex(ctx, &pb.StartIndexRequest{Path: input.Path, Client: scenarioClientInfo()})
	if err != nil {
		return scenarioAResult{}, fmt.Errorf("scenario A start ingest: %w", err)
	}
	checkpoint, err := waitForJob(ctx, input.Client, started.GetJobId(), timeouts.Recovery, timeouts.Poll, func(job *pb.Job) bool {
		return job.GetState() == "running" && job.GetProgress().GetChunksEmbedded() > 0
	})
	if err != nil {
		return scenarioAResult{}, fmt.Errorf("scenario A wait for mid-ingest checkpoint: %w", err)
	}
	beforeRows, beforeCheckpoint, err := captureRecoverySnapshots(ctx, input.ObserveRows, input.ObserveCheckpoint)
	if err != nil {
		return scenarioAResult{}, fmt.Errorf("scenario A capture pre-fault snapshots: %w", err)
	}
	preFaultJobs, err := captureJobSet(ctx, input.Client, started.GetCodebaseId())
	if err != nil {
		return scenarioAResult{}, fmt.Errorf("scenario A capture pre-fault jobs: %w", err)
	}
	preFaultEmbeddingCalls := input.ObserveEmbeddingCalls()
	faultActivatedAt := time.Now()
	failureStarted := faultActivatedAt
	input.Proxy.SetFailure(503, "acceptance embedding outage")
	defer input.Proxy.ClearFailure()
	failed, err := waitForJob(ctx, input.Client, started.GetJobId(), timeouts.Failure, timeouts.Poll, func(job *pb.Job) bool {
		return job.GetState() == "failed"
	})
	if err != nil {
		return scenarioAResult{}, fmt.Errorf("scenario A wait for bounded typed failure: %w", err)
	}
	if failed.GetError().GetCode() == "" || !failed.GetError().GetRetryable() {
		return scenarioAResult{}, fmt.Errorf("scenario A failure is not typed and retryable")
	}
	failureElapsed := time.Since(failureStarted)
	if failureElapsed > timeouts.Failure {
		return scenarioAResult{}, fmt.Errorf("scenario A failure took %s, exceeds %s", failureElapsed, timeouts.Failure)
	}
	input.Proxy.ClearFailure()
	resumed, err := waitForSuccessor(ctx, input.Client, started.GetCodebaseId(), preFaultJobs, faultActivatedAt, timeouts.Recovery, timeouts.Poll)
	if err != nil {
		return scenarioAResult{}, fmt.Errorf("scenario A wait for automatic resume: %w", err)
	}
	afterRows, afterCheckpoint, err := captureRecoverySnapshots(ctx, input.ObserveRows, input.ObserveCheckpoint)
	if err != nil {
		return scenarioAResult{}, fmt.Errorf("scenario A capture recovered snapshots: %w", err)
	}
	if err := validateRecoverySnapshots(beforeRows, beforeCheckpoint, afterRows, afterCheckpoint, input.ExpectedUnfinishedIDs); err != nil {
		return scenarioAResult{}, fmt.Errorf("scenario A validate recovery preservation: %w", err)
	}
	checkpointChunks := checkpoint.GetProgress().GetChunksEmbedded()
	resumedChunks := resumed.GetProgress().GetChunksEmbedded()
	if int(checkpointChunks) != beforeCheckpoint.CompletedCount || int(resumedChunks) != len(input.ExpectedUnfinishedIDs) {
		return scenarioAResult{}, fmt.Errorf("scenario A progress checkpoint=%d resumed=%d, want %d and %d", checkpointChunks, resumedChunks, beforeCheckpoint.CompletedCount, len(input.ExpectedUnfinishedIDs))
	}
	embeddingCalls := input.ObserveEmbeddingCalls()
	if len(embeddingCalls) < len(preFaultEmbeddingCalls) {
		return scenarioAResult{}, fmt.Errorf("scenario A embedding call history shrank")
	}
	embeddedAfterFault := embeddingCalls[len(preFaultEmbeddingCalls):]
	if !sameStringSetAndOrder(embeddedAfterFault, input.ExpectedUnfinishedIDs) {
		return scenarioAResult{}, fmt.Errorf("scenario A embedded identities after fault=%v, want %v", embeddedAfterFault, input.ExpectedUnfinishedIDs)
	}
	result := scenarioAResult{
		FailureCode:           failed.GetError().GetCode(),
		Retryable:             failed.GetError().GetRetryable(),
		FailureElapsed:        failureElapsed,
		Rows:                  len(afterRows.Entries),
		UniqueRows:            len(afterRows.Entries),
		CheckpointChunks:      checkpointChunks,
		ResumedChunksEmbedded: resumedChunks,
		EmbeddedAfterFault:    append([]string(nil), embeddedAfterFault...),
	}
	if err := recordScenario(input.Recorder, "a", map[string]string{
		"failure_code":            result.FailureCode,
		"failure_elapsed":         result.FailureElapsed.String(),
		"rows":                    strconv.Itoa(result.Rows),
		"unique_rows":             strconv.Itoa(result.UniqueRows),
		"checkpoint_chunks":       strconv.Itoa(int(result.CheckpointChunks)),
		"resumed_chunks_embedded": strconv.Itoa(int(result.ResumedChunksEmbedded)),
	}); err != nil {
		return scenarioAResult{}, err
	}
	return result, nil
}

type scenarioBInput struct {
	Client               pb.SemanticSearchDaemonServiceClient
	Path                 string
	Proxy                *milvusProxy
	ReleaseEmbeddingGate func() error
	EmbeddingGateReached <-chan struct{}
	ExpectedAddedHashes  map[string]string
	ObserveRows          rowSnapshotObserver
	ObserveCheckpoint    checkpointSnapshotObserver
	Recorder             *evidenceRecorder
	Timeouts             scenarioTimeouts
}

type scenarioBResult struct {
	SearchCode      codes.Code
	FailureCode     string
	FailureElapsed  time.Duration
	UnhealthyMode   string
	DaemonPIDBefore int32
	DaemonPIDAfter  int32
}

func releaseScenarioBEmbeddingGate(proxy *milvusProxy, clearGate func()) error {
	if !proxy.IsUnavailable() {
		return fmt.Errorf("Milvus fault is not active")
	}
	clearGate()
	return nil
}

func runScenarioB(ctx context.Context, input scenarioBInput) (scenarioBResult, error) {
	if input.Recorder == nil {
		return scenarioBResult{}, fmt.Errorf("scenario B requires an evidence recorder")
	}
	if input.Client == nil || input.Proxy == nil || input.EmbeddingGateReached == nil || input.ObserveRows == nil || input.ObserveCheckpoint == nil {
		return scenarioBResult{}, fmt.Errorf("scenario B requires daemon, proxy, embedding gate, row, and checkpoint observers")
	}
	timeouts := input.Timeouts.resolved()
	before, err := input.Client.GetStatus(ctx, &pb.GetStatusRequest{})
	if err != nil {
		return scenarioBResult{}, fmt.Errorf("scenario B read daemon identity: %w", err)
	}
	preflight, err := searchCodeWithin(ctx, input.Client, input.Path, timeouts.Ready)
	if err != nil {
		return scenarioBResult{}, fmt.Errorf("scenario B verify live search target: %w", err)
	}
	if len(preflight.GetResults()) == 0 {
		return scenarioBResult{}, fmt.Errorf("scenario B live search target returned no results")
	}
	beforeRows, beforeCheckpoint, err := captureRecoverySnapshots(ctx, input.ObserveRows, input.ObserveCheckpoint)
	if err != nil {
		return scenarioBResult{}, fmt.Errorf("scenario B capture pre-fault snapshots: %w", err)
	}
	started, err := input.Client.SyncIndex(ctx, &pb.SyncIndexRequest{Path: input.Path, Client: scenarioClientInfo()})
	if err != nil {
		return scenarioBResult{}, fmt.Errorf("scenario B start ingest: %w", err)
	}
	if err := waitForScenarioBActiveIngest(ctx, input.Client, started.GetJobId(), input.EmbeddingGateReached, timeouts); err != nil {
		return scenarioBResult{}, err
	}
	faultActivatedAt := time.Now()
	input.Proxy.SetUnavailable(codes.Unavailable, "acceptance Milvus outage")
	if err := releaseScenarioBGate(input.ReleaseEmbeddingGate); err != nil {
		return scenarioBResult{}, err
	}
	defer input.Proxy.ClearUnavailable()
	failureContext, cancelFailure := context.WithTimeout(ctx, timeouts.Failure)
	_, searchErr := input.Client.SearchCode(failureContext, &pb.SearchCodeRequest{Path: input.Path, Query: "acceptance", Limit: 1, Client: scenarioClientInfo()})
	if status.Code(searchErr) != codes.Unavailable {
		cancelFailure()
		return scenarioBResult{}, fmt.Errorf("scenario B search error code=%s, want Unavailable", status.Code(searchErr))
	}
	unhealthy, err := waitForIndexHealth(failureContext, input.Client, input.Path, "store_unavailable", timeouts.Failure, timeouts.Poll)
	if err != nil {
		cancelFailure()
		return scenarioBResult{}, fmt.Errorf("scenario B wait for unhealthy readiness: %w", err)
	}
	failed, err := waitForJob(failureContext, input.Client, started.GetJobId(), timeouts.Failure, timeouts.Poll, func(job *pb.Job) bool {
		return job.GetState() == "failed"
	})
	failureElapsed := time.Since(faultActivatedAt)
	cancelFailure()
	if err != nil {
		return scenarioBResult{}, fmt.Errorf("scenario B wait for failed ingest: %w", err)
	}
	if failureElapsed > timeouts.Failure {
		return scenarioBResult{}, fmt.Errorf("scenario B failure took %s, exceeds %s", failureElapsed, timeouts.Failure)
	}
	if failed.GetError().GetCode() != "milvus_unavailable" || !failed.GetError().GetRetryable() {
		return scenarioBResult{}, fmt.Errorf("scenario B ingest failure code=%q retryable=%v", failed.GetError().GetCode(), failed.GetError().GetRetryable())
	}
	input.Proxy.ClearUnavailable()
	recoveryContext, cancelRecovery := context.WithTimeout(ctx, timeouts.Recovery)
	defer cancelRecovery()
	recovered, err := input.Client.SyncIndex(recoveryContext, &pb.SyncIndexRequest{Path: input.Path, Client: scenarioClientInfo()})
	if err != nil {
		return scenarioBResult{}, fmt.Errorf("scenario B retry ingest after reconnect: %w", err)
	}
	if _, err := waitForJob(recoveryContext, input.Client, recovered.GetJobId(), timeouts.Recovery, timeouts.Poll, func(job *pb.Job) bool {
		return job.GetState() == "completed"
	}); err != nil {
		return scenarioBResult{}, fmt.Errorf("scenario B wait for recovered ingest: %w", err)
	}
	recoveredSearch, err := searchCodeWithin(recoveryContext, input.Client, input.Path, timeouts.Recovery)
	if err != nil {
		return scenarioBResult{}, fmt.Errorf("scenario B search after reconnect: %w", err)
	}
	if len(recoveredSearch.GetResults()) == 0 {
		return scenarioBResult{}, fmt.Errorf("scenario B search after reconnect returned no results")
	}
	if _, err := waitForIndexHealth(recoveryContext, input.Client, input.Path, "", timeouts.Recovery, timeouts.Poll); err != nil {
		return scenarioBResult{}, fmt.Errorf("scenario B wait for healthy readiness: %w", err)
	}
	afterRows, afterCheckpoint, err := captureRecoverySnapshots(recoveryContext, input.ObserveRows, input.ObserveCheckpoint)
	if err != nil {
		return scenarioBResult{}, fmt.Errorf("scenario B capture recovered snapshots: %w", err)
	}
	if err := validateRecoverySnapshotsWithHashes(beforeRows, beforeCheckpoint, afterRows, afterCheckpoint, input.ExpectedAddedHashes); err != nil {
		return scenarioBResult{}, fmt.Errorf("scenario B validate recovery preservation: %w", err)
	}
	after, err := input.Client.GetStatus(recoveryContext, &pb.GetStatusRequest{})
	if err != nil {
		return scenarioBResult{}, fmt.Errorf("scenario B read recovered daemon identity: %w", err)
	}
	if before.GetDaemon().GetPid() != after.GetDaemon().GetPid() {
		return scenarioBResult{}, fmt.Errorf("scenario B daemon PID changed from %d to %d", before.GetDaemon().GetPid(), after.GetDaemon().GetPid())
	}
	result := scenarioBResult{
		SearchCode:      status.Code(searchErr),
		FailureCode:     failed.GetError().GetCode(),
		FailureElapsed:  failureElapsed,
		UnhealthyMode:   unhealthy.GetDependencyHealth().GetMode(),
		DaemonPIDBefore: before.GetDaemon().GetPid(),
		DaemonPIDAfter:  after.GetDaemon().GetPid(),
	}
	if err := recordScenario(input.Recorder, "b", map[string]string{
		"search_code":     result.SearchCode.String(),
		"failure_code":    result.FailureCode,
		"failure_elapsed": result.FailureElapsed.String(),
		"unhealthy_mode":  result.UnhealthyMode,
		"daemon_pid":      strconv.Itoa(int(result.DaemonPIDBefore)),
	}); err != nil {
		return scenarioBResult{}, err
	}
	return result, nil
}

func releaseScenarioBGate(release func() error) error {
	if release == nil {
		return nil
	}
	if err := release(); err != nil {
		return fmt.Errorf("scenario B release embedding gate: %w", err)
	}
	return nil
}

func searchCodeWithin(
	ctx context.Context,
	client pb.SemanticSearchDaemonServiceClient,
	path string,
	timeout time.Duration,
) (*pb.SearchCodeResponse, error) {
	searchContext, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	return client.SearchCode(searchContext, &pb.SearchCodeRequest{Path: path, Query: "acceptance", Limit: 1, Client: scenarioClientInfo()})
}

type installedProcess struct {
	Path        string
	Args        []string
	Environment map[string]string
}

type scenarioCInput struct {
	Process               installedProcess
	SocketPath            string
	LockPath              string
	Path                  string
	ExpectedUnfinishedIDs []string
	ObserveRows           rowSnapshotObserver
	ObserveCheckpoint     checkpointSnapshotObserver
	Recorder              *evidenceRecorder
	Timeouts              scenarioTimeouts
}

type scenarioCResult struct {
	LockBusyBeforeKill     bool
	LockReclaimedAfterKill bool
	CheckpointChunks       int32
	ResumedChunksEmbedded  int32
	Rows                   int32
	FirstExecutable        string
	RestartExecutable      string
}

func runScenarioC(ctx context.Context, input scenarioCInput) (_ scenarioCResult, runErr error) {
	if err := validateScenarioCInput(input); err != nil {
		return scenarioCResult{}, err
	}
	timeouts := input.Timeouts.resolved()
	first, err := startInstalledProcess(input.Process)
	if err != nil {
		return scenarioCResult{}, err
	}
	firstStopped := false
	defer func() {
		if !firstStopped {
			runErr = errors.Join(runErr, stopInstalledProcess(first))
		}
	}()
	firstClient, firstConnection, err := waitForDaemon(ctx, input.SocketPath, timeouts.Ready, timeouts.Poll)
	if err != nil {
		return scenarioCResult{}, fmt.Errorf("scenario C wait for first daemon: %w", err)
	}
	started, err := firstClient.StartIndex(ctx, &pb.StartIndexRequest{Path: input.Path, Client: scenarioClientInfo()})
	if err != nil {
		_ = firstConnection.Close()
		return scenarioCResult{}, fmt.Errorf("scenario C start ingest: %w", err)
	}
	checkpoint, err := waitForJob(ctx, firstClient, started.GetJobId(), timeouts.Ready, timeouts.Poll, func(job *pb.Job) bool {
		return job.GetState() == "running" && job.GetProgress().GetChunksEmbedded() > 0
	})
	if err != nil {
		_ = firstConnection.Close()
		return scenarioCResult{}, fmt.Errorf("scenario C wait for held-lock checkpoint: %w", err)
	}
	beforeRows, beforeCheckpoint, err := captureRecoverySnapshots(ctx, input.ObserveRows, input.ObserveCheckpoint)
	if err != nil {
		_ = firstConnection.Close()
		return scenarioCResult{}, fmt.Errorf("scenario C capture pre-fault snapshots: %w", err)
	}
	preFaultJobs, err := captureJobSet(ctx, firstClient, started.GetCodebaseId())
	if err != nil {
		_ = firstConnection.Close()
		return scenarioCResult{}, fmt.Errorf("scenario C capture pre-fault jobs: %w", err)
	}
	lockAvailable, err := kernelLockAvailable(input.LockPath)
	if err != nil {
		_ = firstConnection.Close()
		return scenarioCResult{}, err
	}
	if lockAvailable {
		_ = firstConnection.Close()
		return scenarioCResult{}, fmt.Errorf("scenario C sync lock was not held by the running job")
	}
	faultActivatedAt := time.Now()
	if err := first.Process.Signal(os.Kill); err != nil {
		_ = firstConnection.Close()
		return scenarioCResult{}, fmt.Errorf("scenario C SIGKILL daemon: %w", err)
	}
	firstStopped = true
	_ = firstConnection.Close()
	processState, err := waitForSignaledProcess(first, os.Kill, 5*time.Second)
	if err != nil {
		return scenarioCResult{}, fmt.Errorf("scenario C wait for SIGKILL daemon: %w", err)
	}
	if processState.Success() {
		return scenarioCResult{}, fmt.Errorf("scenario C SIGKILL process exited successfully")
	}
	lockAvailable, err = kernelLockAvailable(input.LockPath)
	if err != nil {
		return scenarioCResult{}, err
	}
	if !lockAvailable {
		return scenarioCResult{}, fmt.Errorf("scenario C kernel did not reclaim sync lock after SIGKILL")
	}
	restarted, err := startInstalledProcess(input.Process)
	if err != nil {
		return scenarioCResult{}, err
	}
	defer func() {
		runErr = errors.Join(runErr, stopInstalledProcess(restarted))
	}()
	restartedClient, restartedConnection, err := waitForDaemon(ctx, input.SocketPath, timeouts.Ready, timeouts.Poll)
	if err != nil {
		return scenarioCResult{}, fmt.Errorf("scenario C wait for restarted daemon: %w", err)
	}
	defer func() { _ = restartedConnection.Close() }()
	resumed, err := waitForSuccessor(ctx, restartedClient, started.GetCodebaseId(), preFaultJobs, faultActivatedAt, timeouts.Recovery, timeouts.Poll)
	if err != nil {
		return scenarioCResult{}, fmt.Errorf("scenario C wait for checkpoint reconciliation: %w", err)
	}
	oldJob, err := restartedClient.GetJob(ctx, &pb.GetJobRequest{JobId: started.GetJobId()})
	if err != nil || oldJob.GetJob().GetState() != "cancelled" {
		return scenarioCResult{}, fmt.Errorf("scenario C orphan job was not reconciled to cancelled: %w", err)
	}
	afterRows, afterCheckpoint, err := captureRecoverySnapshots(ctx, input.ObserveRows, input.ObserveCheckpoint)
	if err != nil {
		return scenarioCResult{}, fmt.Errorf("scenario C capture recovered snapshots: %w", err)
	}
	if err := validateRecoverySnapshots(beforeRows, beforeCheckpoint, afterRows, afterCheckpoint, input.ExpectedUnfinishedIDs); err != nil {
		return scenarioCResult{}, fmt.Errorf("scenario C validate recovery preservation: %w", err)
	}
	result := scenarioCResult{
		LockBusyBeforeKill:     true,
		LockReclaimedAfterKill: true,
		CheckpointChunks:       checkpoint.GetProgress().GetChunksEmbedded(),
		ResumedChunksEmbedded:  resumed.GetProgress().GetChunksEmbedded(),
		Rows:                   int32(len(afterRows.Entries)),
		FirstExecutable:        first.Path,
		RestartExecutable:      restarted.Path,
	}
	if int(result.CheckpointChunks) != beforeCheckpoint.CompletedCount || int(result.ResumedChunksEmbedded) != len(input.ExpectedUnfinishedIDs) {
		return scenarioCResult{}, fmt.Errorf("scenario C progress checkpoint=%d resumed=%d, want %d and %d", result.CheckpointChunks, result.ResumedChunksEmbedded, beforeCheckpoint.CompletedCount, len(input.ExpectedUnfinishedIDs))
	}
	if result.FirstExecutable != result.RestartExecutable {
		return scenarioCResult{}, fmt.Errorf("scenario C restarted %q instead of %q", result.RestartExecutable, result.FirstExecutable)
	}
	if err := recordScenario(input.Recorder, "c", map[string]string{
		"checkpoint_chunks":       strconv.Itoa(int(result.CheckpointChunks)),
		"resumed_chunks_embedded": strconv.Itoa(int(result.ResumedChunksEmbedded)),
		"rows":                    strconv.Itoa(int(result.Rows)),
		"lock_reclaimed":          "true",
		"executable":              result.FirstExecutable,
	}); err != nil {
		return scenarioCResult{}, err
	}
	return result, nil
}

func validateScenarioCInput(input scenarioCInput) error {
	if input.Recorder == nil {
		return fmt.Errorf("scenario C requires an evidence recorder")
	}
	if input.Process.Path == "" || input.SocketPath == "" || input.LockPath == "" {
		return fmt.Errorf("scenario C requires an installed process, socket, and lock path")
	}
	return nil
}

type scenarioDInput struct {
	Process    installedProcess
	SocketPath string
	Paths      runPaths
	OwnerPID   int
	Path       string
	Recorder   *evidenceRecorder
	Timeouts   scenarioTimeouts
}

type scenarioDResult struct {
	CompletedWhileArtifactPresent bool
	UnrelatedProcessAlive         bool
}

func runScenarioD(ctx context.Context, input scenarioDInput) (_ scenarioDResult, runErr error) {
	if input.Recorder == nil {
		return scenarioDResult{}, fmt.Errorf("scenario D requires an evidence recorder")
	}
	if input.Process.Path == "" || input.OwnerPID <= 0 {
		return scenarioDResult{}, fmt.Errorf("scenario D requires an installed process, run paths, and owner PID")
	}
	legacyRoot, err := validateRetiredLockRoot(input.Paths)
	if err != nil {
		return scenarioDResult{}, err
	}
	if err := os.Mkdir(legacyRoot, 0o700); err != nil {
		return scenarioDResult{}, fmt.Errorf("scenario D create retired lock artifact: %w", err)
	}
	ownerPath := filepath.Join(legacyRoot, "owner.pid")
	if err := os.WriteFile(ownerPath, []byte(strconv.Itoa(input.OwnerPID)+"\n"), 0o600); err != nil {
		_ = os.Remove(legacyRoot)
		return scenarioDResult{}, fmt.Errorf("scenario D write retired owner PID: %w", err)
	}
	artifactCleaned := false
	defer func() {
		if !artifactCleaned {
			runErr = errors.Join(runErr, removeRetiredLockArtifact(ownerPath, legacyRoot))
		}
	}()
	process, err := startInstalledProcess(input.Process)
	if err != nil {
		return scenarioDResult{}, err
	}
	defer func() {
		runErr = errors.Join(runErr, stopInstalledProcess(process))
	}()
	timeouts := input.Timeouts.resolved()
	client, connection, err := waitForDaemon(ctx, input.SocketPath, timeouts.Ready, timeouts.Poll)
	if err != nil {
		return scenarioDResult{}, fmt.Errorf("scenario D wait for daemon: %w", err)
	}
	defer func() { _ = connection.Close() }()
	started, err := client.StartIndex(ctx, &pb.StartIndexRequest{Path: input.Path, Client: scenarioClientInfo()})
	if err != nil {
		return scenarioDResult{}, fmt.Errorf("scenario D start ingest: %w", err)
	}
	if _, err := waitForJob(ctx, client, started.GetJobId(), timeouts.Recovery, timeouts.Poll, func(job *pb.Job) bool {
		return job.GetState() == "completed"
	}); err != nil {
		return scenarioDResult{}, fmt.Errorf("scenario D wait for job completion: %w", err)
	}
	if _, err := os.Stat(ownerPath); err != nil {
		return scenarioDResult{}, fmt.Errorf("scenario D retired artifact disappeared before completion: %w", err)
	}
	ownerProcess, err := os.FindProcess(input.OwnerPID)
	if err != nil || ownerProcess.Signal(syscall.Signal(0)) != nil {
		return scenarioDResult{}, fmt.Errorf("scenario D unrelated owner process is not alive: %w", err)
	}
	result := scenarioDResult{CompletedWhileArtifactPresent: true, UnrelatedProcessAlive: true}
	if err := removeRetiredLockArtifact(ownerPath, legacyRoot); err != nil {
		return scenarioDResult{}, fmt.Errorf("scenario D clean retired artifact: %w", err)
	}
	artifactCleaned = true
	if err := ownerProcess.Signal(syscall.Signal(0)); err != nil {
		return scenarioDResult{}, fmt.Errorf("scenario D cleanup disturbed unrelated process: %w", err)
	}
	if err := recordScenario(input.Recorder, "d", map[string]string{
		"completed_while_artifact_present": "true",
		"unrelated_process_alive":          "true",
		"owner_pid":                        strconv.Itoa(input.OwnerPID),
	}); err != nil {
		return scenarioDResult{}, err
	}
	return result, nil
}

func scenarioClientInfo() *pb.ClientInfo {
	return &pb.ClientInfo{Name: "restart-acceptance", Pid: int32(os.Getpid())}
}

func waitForJob(ctx context.Context, client pb.SemanticSearchDaemonServiceClient, jobID string, timeout time.Duration, poll time.Duration, accepted func(*pb.Job) bool) (*pb.Job, error) {
	deadlineContext, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	for {
		response, err := client.GetJob(deadlineContext, &pb.GetJobRequest{JobId: jobID})
		if err == nil && response.GetJob() != nil && accepted(response.GetJob()) {
			return response.GetJob(), nil
		}
		select {
		case <-deadlineContext.Done():
			return nil, fmt.Errorf("wait for job %q: %w", jobID, context.Cause(deadlineContext))
		case <-time.After(poll):
		}
	}
}

func waitForScenarioBActiveIngest(
	ctx context.Context,
	client pb.SemanticSearchDaemonServiceClient,
	jobID string,
	gateReached <-chan struct{},
	timeouts scenarioTimeouts,
) error {
	if _, err := waitForJob(ctx, client, jobID, timeouts.Ready, timeouts.Poll, func(job *pb.Job) bool {
		return job.GetState() == "running"
	}); err != nil {
		return fmt.Errorf("scenario B wait for active ingest: %w", err)
	}
	gateContext, cancel := context.WithTimeout(ctx, timeouts.Ready)
	defer cancel()
	select {
	case <-gateReached:
		return nil
	case <-gateContext.Done():
		return fmt.Errorf("scenario B wait for blocked embedding request: %w", context.Cause(gateContext))
	}
}

func captureJobSet(ctx context.Context, client pb.SemanticSearchDaemonServiceClient, codebaseID string) (map[string]struct{}, error) {
	response, err := client.ListJobs(ctx, &pb.ListJobsRequest{CodebaseId: codebaseID})
	if err != nil {
		return nil, err
	}
	jobs := make(map[string]struct{}, len(response.GetJobs()))
	for _, job := range response.GetJobs() {
		if job.GetId() == "" {
			return nil, fmt.Errorf("pre-fault job has no ID")
		}
		jobs[job.GetId()] = struct{}{}
	}
	return jobs, nil
}

func waitForSuccessor(ctx context.Context, client pb.SemanticSearchDaemonServiceClient, codebaseID string, preFaultJobs map[string]struct{}, faultActivatedAt time.Time, timeout time.Duration, poll time.Duration) (*pb.Job, error) {
	deadlineContext, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	observedNonterminal := make(map[string]struct{})
	for {
		response, err := client.ListJobs(deadlineContext, &pb.ListJobsRequest{CodebaseId: codebaseID})
		if err == nil {
			for _, job := range response.GetJobs() {
				if _, existed := preFaultJobs[job.GetId()]; existed {
					continue
				}
				if job.GetId() == "" || job.GetStartedAt() == nil || job.GetUpdatedAt() == nil {
					continue
				}
				if !job.GetStartedAt().AsTime().After(faultActivatedAt) || !job.GetUpdatedAt().AsTime().After(faultActivatedAt) {
					continue
				}
				switch job.GetState() {
				case "queued", "running", "cancelling":
					observedNonterminal[job.GetId()] = struct{}{}
				case "completed":
					if _, observed := observedNonterminal[job.GetId()]; !observed {
						continue
					}
					return job, nil
				}
			}
		}
		select {
		case <-deadlineContext.Done():
			return nil, fmt.Errorf("wait for successor after %s: %w", faultActivatedAt.UTC().Format(time.RFC3339Nano), context.Cause(deadlineContext))
		case <-time.After(poll):
		}
	}
}

func captureRecoverySnapshots(ctx context.Context, observeRows rowSnapshotObserver, observeCheckpoint checkpointSnapshotObserver) (rowSnapshot, checkpointSnapshot, error) {
	if observeRows == nil || observeCheckpoint == nil {
		return rowSnapshot{}, checkpointSnapshot{}, fmt.Errorf("row and checkpoint observers are required")
	}
	rows, err := observeRows(ctx)
	if err != nil {
		return rowSnapshot{}, checkpointSnapshot{}, err
	}
	checkpoint, err := observeCheckpoint(ctx)
	if err != nil {
		return rowSnapshot{}, checkpointSnapshot{}, err
	}
	if len(rows.Entries) == 0 || checkpoint.CompletedCount != len(checkpoint.FileHashes) {
		return rowSnapshot{}, checkpointSnapshot{}, fmt.Errorf("snapshot counts are inconsistent")
	}
	if len(rows.Entries) != len(checkpoint.FileHashes) {
		return rowSnapshot{}, checkpointSnapshot{}, fmt.Errorf("snapshot row and checkpoint identities differ")
	}
	for id := range rows.Entries {
		if _, found := checkpoint.FileHashes[id]; !found {
			return rowSnapshot{}, checkpointSnapshot{}, fmt.Errorf("snapshot row %q has no checkpoint hash", id)
		}
	}
	return rows, checkpoint, nil
}

func validateRecoverySnapshots(beforeRows rowSnapshot, beforeCheckpoint checkpointSnapshot, afterRows rowSnapshot, afterCheckpoint checkpointSnapshot, addedIDs []string) error {
	addedHashes := make(map[string]string, len(addedIDs))
	for _, id := range addedIDs {
		hash, found := afterCheckpoint.FileHashes[id]
		if !found {
			return fmt.Errorf("recovered checkpoint does not track %q", id)
		}
		addedHashes[id] = hash
	}
	return validateRecoverySnapshotsWithHashes(beforeRows, beforeCheckpoint, afterRows, afterCheckpoint, addedHashes)
}

func validateRecoverySnapshotsWithHashes(beforeRows rowSnapshot, beforeCheckpoint checkpointSnapshot, afterRows rowSnapshot, afterCheckpoint checkpointSnapshot, addedHashes map[string]string) error {
	for id, beforeHash := range beforeCheckpoint.FileHashes {
		before, found := beforeRows.Entries[id]
		if !found {
			return fmt.Errorf("checkpoint ID %q has no row", id)
		}
		after, found := afterRows.Entries[id]
		if !found || after != before {
			return fmt.Errorf("completed row %q changed or disappeared", id)
		}
		if afterCheckpoint.FileHashes[id] != beforeHash {
			return fmt.Errorf("completed checkpoint hash for %q changed", id)
		}
	}
	expectedAfter := make(map[string]struct{}, len(beforeRows.Entries)+len(addedHashes))
	for id := range beforeRows.Entries {
		expectedAfter[id] = struct{}{}
	}
	for id := range addedHashes {
		expectedAfter[id] = struct{}{}
	}
	if len(afterRows.Entries) != len(expectedAfter) {
		return fmt.Errorf("recovered rows=%d, want %d", len(afterRows.Entries), len(expectedAfter))
	}
	for id := range afterRows.Entries {
		if _, expected := expectedAfter[id]; !expected {
			return fmt.Errorf("unexpected recovered row %q", id)
		}
	}
	if afterCheckpoint.CompletedCount != len(expectedAfter) {
		return fmt.Errorf("recovered checkpoint completed=%d, want %d", afterCheckpoint.CompletedCount, len(expectedAfter))
	}
	for id := range expectedAfter {
		if _, tracked := afterCheckpoint.FileHashes[id]; !tracked {
			return fmt.Errorf("recovered checkpoint does not track %q", id)
		}
	}
	for id, expectedHash := range addedHashes {
		if afterCheckpoint.FileHashes[id] != expectedHash {
			return fmt.Errorf("recovered checkpoint hash for %q does not match the added file", id)
		}
	}
	return nil
}

func sameStringSetAndOrder(actual []string, expected []string) bool {
	if len(actual) != len(expected) {
		return false
	}
	for index := range actual {
		if actual[index] != expected[index] {
			return false
		}
	}
	return true
}

func validateRetiredLockRoot(paths runPaths) (string, error) {
	expected := pathsForRun(paths.RunRoot)
	if paths.RunRoot == "" || paths.LMSContext != expected.LMSContext || !pathWithin(paths.RunRoot, paths.LMSContext) {
		return "", fmt.Errorf("scenario D requires the exact isolated context under validated RunRoot")
	}
	info, err := os.Lstat(paths.LMSContext)
	if err != nil {
		return "", fmt.Errorf("scenario D inspect isolated context: %w", err)
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return "", fmt.Errorf("scenario D isolated context is not a real directory or is a symlink")
	}
	resolvedRoot, err := filepath.EvalSymlinks(paths.RunRoot)
	if err != nil {
		return "", fmt.Errorf("scenario D resolve RunRoot: %w", err)
	}
	resolvedContext, err := filepath.EvalSymlinks(paths.LMSContext)
	if err != nil {
		return "", fmt.Errorf("scenario D resolve isolated context: %w", err)
	}
	if !pathWithin(resolvedRoot, resolvedContext) {
		return "", fmt.Errorf("scenario D isolated context escapes validated RunRoot")
	}
	return filepath.Join(paths.LMSContext, "mcp-sync.lock"), nil
}

func removeRetiredLockArtifact(ownerPath string, legacyRoot string) error {
	if err := os.Remove(ownerPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	entries, err := os.ReadDir(legacyRoot)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if len(entries) == 0 {
		return os.Remove(legacyRoot)
	}
	return nil
}

func waitForIndexHealth(ctx context.Context, client pb.SemanticSearchDaemonServiceClient, path string, mode string, timeout time.Duration, poll time.Duration) (*pb.GetIndexResponse, error) {
	deadlineContext, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	for {
		response, err := client.GetIndex(deadlineContext, &pb.GetIndexRequest{Path: path, Client: scenarioClientInfo()})
		if err == nil && response.GetDependencyHealth().GetMode() == mode {
			return response, nil
		}
		select {
		case <-deadlineContext.Done():
			return nil, fmt.Errorf("wait for dependency mode %q: %w", mode, context.Cause(deadlineContext))
		case <-time.After(poll):
		}
	}
}

func startInstalledProcess(process installedProcess) (*exec.Cmd, error) {
	command := exec.Command(process.Path, process.Args...)
	command.Stdout = os.Stderr
	command.Stderr = os.Stderr
	environment := make(map[string]string)
	for _, entry := range os.Environ() {
		name, value, found := stringsCut(entry, '=')
		if found {
			environment[name] = value
		}
	}
	for name, value := range process.Environment {
		environment[name] = value
	}
	names := make([]string, 0, len(environment))
	for name := range environment {
		names = append(names, name)
	}
	sort.Strings(names)
	command.Env = make([]string, 0, len(names))
	for _, name := range names {
		command.Env = append(command.Env, name+"="+environment[name])
	}
	if err := command.Start(); err != nil {
		return nil, fmt.Errorf("start installed binary %q: %w", process.Path, err)
	}
	return command, nil
}

func stringsCut(value string, separator byte) (string, string, bool) {
	for index := 0; index < len(value); index++ {
		if value[index] == separator {
			return value[:index], value[index+1:], true
		}
	}
	return value, "", false
}

func waitForDaemon(ctx context.Context, socketPath string, timeout time.Duration, poll time.Duration) (pb.SemanticSearchDaemonServiceClient, interface{ Close() error }, error) {
	deadlineContext, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	var lastErr error
	for {
		connection, client, err := daemonclient.DialDaemon(deadlineContext, socketPath)
		if err == nil {
			attemptTimeout := poll
			if attemptTimeout < 500*time.Millisecond {
				attemptTimeout = 500 * time.Millisecond
			}
			attemptContext, attemptCancel := context.WithTimeout(deadlineContext, attemptTimeout)
			_, statusErr := client.GetStatus(attemptContext, &pb.GetStatusRequest{})
			attemptCancel()
			if statusErr == nil {
				return client, connection, nil
			}
			lastErr = statusErr
			_ = connection.Close()
		} else {
			lastErr = err
		}
		select {
		case <-deadlineContext.Done():
			return nil, nil, fmt.Errorf("%w; last readiness error: %v", context.Cause(deadlineContext), lastErr)
		case <-time.After(poll):
		}
	}
}

func kernelLockAvailable(lockPath string) (bool, error) {
	file, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return false, fmt.Errorf("open sync lock probe: %w", err)
	}
	defer func() { _ = file.Close() }()
	err = unix.Flock(int(file.Fd()), unix.LOCK_EX|unix.LOCK_NB)
	if errors.Is(err, unix.EWOULDBLOCK) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("probe sync lock: %w", err)
	}
	if err := unix.Flock(int(file.Fd()), unix.LOCK_UN); err != nil {
		return false, fmt.Errorf("release sync lock probe: %w", err)
	}
	return true, nil
}

func recordScenario(recorder *evidenceRecorder, name string, details map[string]string) error {
	if recorder == nil {
		return fmt.Errorf("scenario %s requires an evidence recorder", name)
	}
	if err := recorder.Record("scenario_"+name, "passed", details); err != nil {
		return fmt.Errorf("record scenario %s evidence: %w", name, err)
	}
	return nil
}

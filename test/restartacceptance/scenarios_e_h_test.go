//go:build restartacceptance

package restartacceptance

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestRunScenarioEKeepsClydeAliveAndRecoversSearchAndFeeder(t *testing.T) {
	recorder, evidencePaths := scenarioTestRecorder(t)
	var mutex sync.Mutex
	lmsRunning := true
	lastSync := int64(10)
	manifest := 1
	embedded := 1
	searchCalls := 0
	syncCalls := 0

	result, err := runScenarioE(context.Background(), scenarioEInput{
		StopLMS: func(context.Context) error {
			mutex.Lock()
			lmsRunning = false
			mutex.Unlock()
			return nil
		},
		StartLMS: func(context.Context) error {
			mutex.Lock()
			lmsRunning = true
			mutex.Unlock()
			return nil
		},
		SearchClyde: func(context.Context) (semanticSearchObservation, error) {
			mutex.Lock()
			defer mutex.Unlock()
			searchCalls++
			if !lmsRunning {
				return semanticSearchObservation{Code: clydeSourceUnavailableCode}, nil
			}
			return semanticSearchObservation{Succeeded: true, Source: "semantic", Matches: 2, ResultIDs: []string{"conversation-1:0", "conversation-2:0"}}, nil
		},
		ClydeStatus: func(context.Context) (clydeStatusObservation, error) {
			mutex.Lock()
			defer mutex.Unlock()
			return clydeStatusObservation{PID: 4242, Responding: true, LastSyncUnix: lastSync, Manifest: manifest, Embedded: embedded}, nil
		},
		TriggerFeederSync: func(context.Context) error {
			mutex.Lock()
			syncCalls++
			lastSync++
			manifest++
			embedded++
			mutex.Unlock()
			return nil
		},
		Recorder: recorder,
		Timeouts: scenarioETimeouts{
			Failure:        100 * time.Millisecond,
			SearchRecovery: 200 * time.Millisecond,
			FeederRecovery: 200 * time.Millisecond,
			Poll:           time.Millisecond,
		},
	})
	if err != nil {
		t.Fatalf("runScenarioE returned error: %v", err)
	}
	if result.ClydePIDBefore != 4242 || result.ClydePIDAfter != 4242 {
		t.Fatalf("Clyde PIDs = %d/%d, want 4242/4242", result.ClydePIDBefore, result.ClydePIDAfter)
	}
	if result.OutageCode != clydeSourceUnavailableCode || result.RecoveredMatches != 2 {
		t.Fatalf("outage = %q recovered matches = %d", result.OutageCode, result.RecoveredMatches)
	}
	if searchCalls < 3 || syncCalls != 1 {
		t.Fatalf("search calls = %d sync calls = %d, want at least 3 and exactly 1", searchCalls, syncCalls)
	}
	assertSingleScenarioRecord(t, evidencePaths.EventsJSONL, "scenario_e")
}

func TestRunScenarioERejectsChangedRecoveredResultIdentities(t *testing.T) {
	recorder, _ := scenarioTestRecorder(t)
	lmsRunning := true
	searchCalls := 0
	_, err := runScenarioE(context.Background(), scenarioEInput{
		StopLMS:  func(context.Context) error { lmsRunning = false; return nil },
		StartLMS: func(context.Context) error { lmsRunning = true; return nil },
		SearchClyde: func(context.Context) (semanticSearchObservation, error) {
			searchCalls++
			if !lmsRunning {
				return semanticSearchObservation{Code: clydeSourceUnavailableCode}, nil
			}
			ids := []string{"baseline:0"}
			if searchCalls > 2 {
				ids = []string{"different:0"}
			}
			return semanticSearchObservation{Succeeded: true, Source: "semantic", Matches: 1, ResultIDs: ids}, nil
		},
		ClydeStatus: func(context.Context) (clydeStatusObservation, error) {
			return clydeStatusObservation{PID: 1, Responding: true, LastSyncUnix: 1, Manifest: 1, Embedded: 1}, nil
		},
		TriggerFeederSync: func(context.Context) error { return nil },
		Recorder:          recorder,
		Timeouts:          scenarioETimeouts{Failure: 20 * time.Millisecond, SearchRecovery: 20 * time.Millisecond, Poll: time.Millisecond},
	})
	if err == nil || !strings.Contains(err.Error(), "result identities changed") {
		t.Fatalf("runScenarioE error = %v, want changed identity rejection", err)
	}
}

func TestWaitForClydeFeederRejectsTimestampWithoutManifestConvergence(t *testing.T) {
	_, err := waitForClydeFeeder(
		context.Background(),
		func(context.Context) (clydeStatusObservation, error) {
			return clydeStatusObservation{PID: 7, Responding: true, LastSyncUnix: 11, Manifest: 2, Embedded: 1, Pending: 1}, nil
		},
		10,
		1,
		7,
		10*time.Millisecond,
		time.Millisecond,
	)
	if err == nil {
		t.Fatal("waitForClydeFeeder accepted timestamp-only progress without manifest convergence")
	}
}

func TestRunScenarioERejectsEmptySuccessDuringOutage(t *testing.T) {
	recorder, _ := scenarioTestRecorder(t)
	searchCalls := 0
	_, err := runScenarioE(context.Background(), scenarioEInput{
		StopLMS:  func(context.Context) error { return nil },
		StartLMS: func(context.Context) error { return nil },
		SearchClyde: func(context.Context) (semanticSearchObservation, error) {
			searchCalls++
			if searchCalls == 1 {
				return semanticSearchObservation{Succeeded: true, Source: "semantic", Matches: 1, ResultIDs: []string{"baseline:0"}}, nil
			}
			return semanticSearchObservation{Succeeded: true, Source: "keyword", Matches: 0}, nil
		},
		ClydeStatus: func(context.Context) (clydeStatusObservation, error) {
			return clydeStatusObservation{PID: 1, Responding: true, LastSyncUnix: 1}, nil
		},
		TriggerFeederSync: func(context.Context) error { return nil },
		Recorder:          recorder,
		Timeouts:          scenarioETimeouts{Failure: 20 * time.Millisecond, Poll: time.Millisecond},
	})
	if err == nil || !strings.Contains(err.Error(), "typed source unavailable") {
		t.Fatalf("runScenarioE error = %v, want typed outage rejection", err)
	}
}

func TestRunScenarioFRejectsEmptyEtcdThenRecoversFromUntouchedRestore(t *testing.T) {
	recorder, evidencePaths := scenarioTestRecorder(t)
	var mutex sync.Mutex
	events := make([]string, 0)
	startCount := 0
	metadataRestored := false
	sourceFingerprint := "immutable-source"
	configurationFingerprint := "same-config"

	result, err := runScenarioF(context.Background(), scenarioFInput{
		VerifyClonePayload: func(context.Context) error {
			mutex.Lock()
			events = append(events, "verify-objects-minio")
			mutex.Unlock()
			return nil
		},
		PrepareEmptyEtcd: func(context.Context) error {
			mutex.Lock()
			events = append(events, "empty-etcd")
			mutex.Unlock()
			return nil
		},
		StartLMS: func(context.Context) error {
			mutex.Lock()
			startCount++
			events = append(events, "start")
			mutex.Unlock()
			return nil
		},
		StopLMS: func(context.Context) error {
			mutex.Lock()
			events = append(events, "stop")
			mutex.Unlock()
			return nil
		},
		ObserveBoot: func(context.Context) (bootObservation, error) {
			return bootObservation{FailureCode: "collection_missing", SelfCheckExhausted: true}, nil
		},
		RestoreEtcd: func(context.Context) error {
			mutex.Lock()
			metadataRestored = true
			events = append(events, "restore-etcd")
			mutex.Unlock()
			return nil
		},
		Search: func(context.Context) (semanticSearchObservation, error) {
			mutex.Lock()
			defer mutex.Unlock()
			if !metadataRestored {
				return semanticSearchObservation{Code: "collection_missing"}, nil
			}
			return semanticSearchObservation{Succeeded: true, Source: "semantic", Matches: 1}, nil
		},
		EtcdFingerprint:          func() (string, error) { return sourceFingerprint, nil },
		ConfigurationFingerprint: func() (string, error) { return configurationFingerprint, nil },
		Recorder:                 recorder,
		Timeouts: scenarioFTimeouts{
			BootFailure: 200 * time.Millisecond,
			Recovery:    200 * time.Millisecond,
			Poll:        time.Millisecond,
		},
	})
	if err != nil {
		t.Fatalf("runScenarioF returned error: %v", err)
	}
	if result.BootFailureCode != "collection_missing" || result.RecoveredMatches != 1 {
		t.Fatalf("boot failure = %q recovered matches = %d", result.BootFailureCode, result.RecoveredMatches)
	}
	if startCount != 2 {
		t.Fatalf("LMS starts = %d, want 2", startCount)
	}
	wantEvents := []string{"verify-objects-minio", "empty-etcd", "start", "stop", "restore-etcd", "start"}
	if !slices.Equal(events, wantEvents) {
		t.Fatalf("events = %v, want %v", events, wantEvents)
	}
	assertSingleScenarioRecord(t, evidencePaths.EventsJSONL, "scenario_f")
}

func TestRunScenarioFFailsIfEmptyEtcdBootBecomesHealthy(t *testing.T) {
	recorder, _ := scenarioTestRecorder(t)
	_, err := runScenarioF(context.Background(), scenarioFInput{
		VerifyClonePayload:       func(context.Context) error { return nil },
		PrepareEmptyEtcd:         func(context.Context) error { return nil },
		StartLMS:                 func(context.Context) error { return nil },
		StopLMS:                  func(context.Context) error { return nil },
		ObserveBoot:              func(context.Context) (bootObservation, error) { return bootObservation{Healthy: true}, nil },
		RestoreEtcd:              func(context.Context) error { return nil },
		Search:                   func(context.Context) (semanticSearchObservation, error) { return semanticSearchObservation{}, nil },
		EtcdFingerprint:          func() (string, error) { return "etcd", nil },
		ConfigurationFingerprint: func() (string, error) { return "config", nil },
		Recorder:                 recorder,
		Timeouts:                 scenarioFTimeouts{BootFailure: 20 * time.Millisecond, Poll: time.Millisecond},
	})
	if err == nil || !strings.Contains(err.Error(), "became healthy") {
		t.Fatalf("runScenarioF error = %v, want unexpected health failure", err)
	}
}

func TestRunScenarioFRejectsChangedRestoredEtcdFingerprint(t *testing.T) {
	recorder, _ := scenarioTestRecorder(t)
	fingerprintCalls := 0
	_, err := runScenarioF(context.Background(), scenarioFInput{
		VerifyClonePayload: func(context.Context) error { return nil },
		PrepareEmptyEtcd:   func(context.Context) error { return nil },
		StartLMS:           func(context.Context) error { return nil },
		StopLMS:            func(context.Context) error { return nil },
		ObserveBoot: func(context.Context) (bootObservation, error) {
			return bootObservation{FailureCode: "collection_missing", SelfCheckExhausted: true}, nil
		},
		RestoreEtcd: func(context.Context) error { return nil },
		Search: func(context.Context) (semanticSearchObservation, error) {
			return semanticSearchObservation{Succeeded: true, Source: "semantic", Matches: 1}, nil
		},
		EtcdFingerprint: func() (string, error) {
			fingerprintCalls++
			if fingerprintCalls == 1 {
				return "preserved", nil
			}
			return "changed", nil
		},
		ConfigurationFingerprint: func() (string, error) { return "config", nil },
		Recorder:                 recorder,
		Timeouts:                 scenarioFTimeouts{BootFailure: 20 * time.Millisecond, Poll: time.Millisecond},
	})
	if err == nil || !strings.Contains(err.Error(), "restored etcd changed") {
		t.Fatalf("runScenarioF error = %v, want restored etcd fingerprint rejection", err)
	}
}

func TestRunScenarioGPermanentLoadingReleasesCapacityAndLoadsExactlyTwice(t *testing.T) {
	recorder, evidencePaths := scenarioTestRecorder(t)
	var mutex sync.Mutex
	loading := false
	loadCount := 0
	failurePolls := 0
	releasedSecond := false
	secondRunningAt := time.Now().Add(15 * time.Millisecond)

	result, err := runScenarioG(context.Background(), scenarioGInput{
		SetLoading:   func() { mutex.Lock(); loading = true; mutex.Unlock() },
		ClearLoading: func() { mutex.Lock(); loading = false; mutex.Unlock() },
		StartTargetJob: func(context.Context) (string, error) {
			mutex.Lock()
			loadCount++
			mutex.Unlock()
			return "target-job", nil
		},
		ObserveFailure: func(context.Context) (semanticSearchObservation, error) {
			mutex.Lock()
			defer mutex.Unlock()
			failurePolls++
			if failurePolls == 2 {
				loadCount++
			}
			if failurePolls >= 3 {
				return semanticSearchObservation{Code: collectionNotReadyCode}, nil
			}
			return semanticSearchObservation{}, nil
		},
		Status: func(context.Context) (daemonStatusObservation, error) {
			return daemonStatusObservation{Responding: true}, nil
		},
		StartSecondJob: func(context.Context) (jobObservation, error) {
			return jobObservation{ID: "second-job", State: "queued"}, nil
		},
		ObserveCapacityRelease: func(context.Context) (time.Duration, error) {
			return 5 * time.Millisecond, nil
		},
		ReleaseSecondJob: func() { releasedSecond = true },
		ObserveJob: func(_ context.Context, jobID string) (jobObservation, error) {
			if jobID == "second-job" && !releasedSecond {
				if time.Now().Before(secondRunningAt) {
					return jobObservation{ID: jobID, State: "queued"}, nil
				}
				return jobObservation{ID: jobID, State: "running"}, nil
			}
			return jobObservation{ID: jobID, State: "completed"}, nil
		},
		RestartTargetJob: func(context.Context) (string, error) { return "target-retry", nil },
		LoadCount:        func() int { mutex.Lock(); defer mutex.Unlock(); return loadCount },
		SearchEditedTarget: func(context.Context) (semanticSearchObservation, error) {
			mutex.Lock()
			defer mutex.Unlock()
			if loading {
				return semanticSearchObservation{Code: collectionNotReadyCode}, nil
			}
			return semanticSearchObservation{Succeeded: true, Source: "semantic", Matches: 1, ResultIDs: []string{"01.go:1:2"}}, nil
		},
		ExpectedEditedID: "01.go",
		Recorder:         recorder,
		Timeouts: scenarioGTimeouts{
			InitialLoad: 100 * time.Millisecond,
			Capacity:    100 * time.Millisecond,
			Observation: 5 * time.Millisecond,
			Failure:     200 * time.Millisecond,
			NoThirdLoad: 10 * time.Millisecond,
			Recovery:    100 * time.Millisecond,
			Poll:        time.Millisecond,
		},
	})
	if err != nil {
		t.Fatalf("runScenarioG returned error: %v", err)
	}
	if result.LoadCalls != 2 || result.FailureCode != collectionNotReadyCode {
		t.Fatalf("load calls = %d failure = %q", result.LoadCalls, result.FailureCode)
	}
	if result.SecondJobID != "second-job" || result.RecoveredMatches != 1 {
		t.Fatalf("second job = %q recovered matches = %d", result.SecondJobID, result.RecoveredMatches)
	}
	assertSingleScenarioRecord(t, evidencePaths.EventsJSONL, "scenario_g")
}

func TestReadCapacityReleaseEventUsesTheCurrentWatchdogEvent(t *testing.T) {
	logPath := filepath.Join(t.TempDir(), "daemon.jsonl")
	notBefore := time.Date(2026, time.August, 12, 12, 0, 0, 0, time.UTC)
	body := strings.Join([]string{
		`{"time":"2026-08-12T11:59:59Z","msg":"released indexing capacity for a stalled read","elapsed_ms":1}`,
		`{"time":"2026-08-12T12:00:05Z","msg":"released indexing capacity for a stalled read","elapsed_ms":5000}`,
	}, "\n")
	if err := os.WriteFile(logPath, []byte(body), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	grace, found, err := readCapacityReleaseEvent(logPath, notBefore)
	if err != nil {
		t.Fatalf("readCapacityReleaseEvent: %v", err)
	}
	if !found || grace != maximumJobCapacityRecovery {
		t.Fatalf("event found = %t grace = %s, want true and %s", found, grace, maximumJobCapacityRecovery)
	}
}

func TestStopInstalledProcessRejectsAPreexistingCrash(t *testing.T) {
	process := exec.Command("sh", "-c", "exit 7")
	if err := process.Start(); err != nil {
		t.Fatalf("start crashed process: %v", err)
	}
	time.Sleep(20 * time.Millisecond)
	if err := stopInstalledProcess(process); err == nil {
		t.Fatal("preexisting process crash was accepted as normal teardown")
	}
}

func TestRunScenarioGRejectsAThirdLoad(t *testing.T) {
	recorder, _ := scenarioTestRecorder(t)
	loadCount := 1
	releasedSecond := false
	_, err := runScenarioG(context.Background(), scenarioGInput{
		SetLoading:     func() {},
		ClearLoading:   func() {},
		StartTargetJob: func(context.Context) (string, error) { return "target", nil },
		ObserveFailure: func(context.Context) (semanticSearchObservation, error) {
			loadCount = 2
			return semanticSearchObservation{Code: collectionNotReadyCode}, nil
		},
		Status: func(context.Context) (daemonStatusObservation, error) {
			return daemonStatusObservation{Responding: true}, nil
		},
		StartSecondJob: func(context.Context) (jobObservation, error) {
			return jobObservation{ID: "second", State: "running"}, nil
		},
		ObserveCapacityRelease: func(context.Context) (time.Duration, error) {
			return time.Millisecond, nil
		},
		ReleaseSecondJob: func() { releasedSecond = true },
		ObserveJob: func(_ context.Context, jobID string) (jobObservation, error) {
			if jobID == "second" && !releasedSecond {
				return jobObservation{ID: jobID, State: "running"}, nil
			}
			return jobObservation{ID: jobID, State: "completed"}, nil
		},
		RestartTargetJob: func(context.Context) (string, error) { return "retry", nil },
		LoadCount: func() int {
			loadCount++
			return loadCount
		},
		SearchEditedTarget: func(context.Context) (semanticSearchObservation, error) {
			return semanticSearchObservation{Succeeded: true, Source: "semantic", Matches: 1, ResultIDs: []string{"01.go:1:2"}}, nil
		},
		ExpectedEditedID: "01.go",
		Recorder:         recorder,
		Timeouts:         scenarioGTimeouts{InitialLoad: 20 * time.Millisecond, Capacity: 20 * time.Millisecond, Failure: 20 * time.Millisecond, NoThirdLoad: time.Millisecond, Recovery: 20 * time.Millisecond, Poll: time.Millisecond},
	})
	if err == nil || !strings.Contains(err.Error(), "final load") {
		t.Fatalf("runScenarioG error = %v, want final load rejection", err)
	}
}

func TestRunScenarioGRejectsRecoveryWithoutEditedIdentity(t *testing.T) {
	recorder, _ := scenarioTestRecorder(t)
	loadCount := 1
	releasedSecond := false
	_, err := runScenarioG(context.Background(), scenarioGInput{
		SetLoading:     func() {},
		ClearLoading:   func() {},
		StartTargetJob: func(context.Context) (string, error) { return "target", nil },
		ObserveFailure: func(context.Context) (semanticSearchObservation, error) {
			loadCount = 2
			return semanticSearchObservation{Code: collectionNotReadyCode}, nil
		},
		Status: func(context.Context) (daemonStatusObservation, error) {
			return daemonStatusObservation{Responding: true}, nil
		},
		StartSecondJob: func(context.Context) (jobObservation, error) {
			return jobObservation{ID: "second", State: "running"}, nil
		},
		ObserveCapacityRelease: func(context.Context) (time.Duration, error) {
			return time.Millisecond, nil
		},
		ReleaseSecondJob: func() { releasedSecond = true },
		ObserveJob: func(_ context.Context, jobID string) (jobObservation, error) {
			if jobID == "second" && !releasedSecond {
				return jobObservation{ID: jobID, State: "running"}, nil
			}
			return jobObservation{ID: jobID, State: "completed"}, nil
		},
		RestartTargetJob: func(context.Context) (string, error) { return "retry", nil },
		LoadCount:        func() int { return loadCount },
		SearchEditedTarget: func(context.Context) (semanticSearchObservation, error) {
			return semanticSearchObservation{Succeeded: true, Source: "semantic", Matches: 1, ResultIDs: []string{"old.go:1:2"}}, nil
		},
		ExpectedEditedID: "01.go",
		Recorder:         recorder,
		Timeouts: scenarioGTimeouts{
			InitialLoad: 20 * time.Millisecond,
			Capacity:    20 * time.Millisecond,
			Failure:     20 * time.Millisecond,
			NoThirdLoad: time.Millisecond,
			Recovery:    20 * time.Millisecond,
			Poll:        time.Millisecond,
		},
	})
	if err == nil || !strings.Contains(err.Error(), "edited identity") {
		t.Fatalf("runScenarioG error = %v, want edited identity rejection", err)
	}
}

func TestRunScenarioGRejectsLoadDuringRecovery(t *testing.T) {
	recorder, _ := scenarioTestRecorder(t)
	loadCount := 1
	releasedSecond := false
	_, err := runScenarioG(context.Background(), scenarioGInput{
		SetLoading:     func() {},
		ClearLoading:   func() {},
		StartTargetJob: func(context.Context) (string, error) { return "target", nil },
		ObserveFailure: func(context.Context) (semanticSearchObservation, error) {
			loadCount = 2
			return semanticSearchObservation{Code: collectionNotReadyCode}, nil
		},
		Status: func(context.Context) (daemonStatusObservation, error) {
			return daemonStatusObservation{Responding: true}, nil
		},
		StartSecondJob: func(context.Context) (jobObservation, error) {
			return jobObservation{ID: "second", State: "running"}, nil
		},
		ObserveCapacityRelease: func(context.Context) (time.Duration, error) {
			return time.Millisecond, nil
		},
		ReleaseSecondJob: func() { releasedSecond = true },
		ObserveJob: func(_ context.Context, jobID string) (jobObservation, error) {
			if jobID == "second" && !releasedSecond {
				return jobObservation{ID: jobID, State: "running"}, nil
			}
			return jobObservation{ID: jobID, State: "completed"}, nil
		},
		RestartTargetJob: func(context.Context) (string, error) { return "retry", nil },
		LoadCount:        func() int { return loadCount },
		SearchEditedTarget: func(context.Context) (semanticSearchObservation, error) {
			loadCount++
			return semanticSearchObservation{Succeeded: true, Source: "semantic", Matches: 1, ResultIDs: []string{"01.go:1:2"}}, nil
		},
		ExpectedEditedID: "01.go",
		Recorder:         recorder,
		Timeouts: scenarioGTimeouts{
			InitialLoad: 20 * time.Millisecond,
			Capacity:    20 * time.Millisecond,
			Failure:     20 * time.Millisecond,
			NoThirdLoad: time.Millisecond,
			Recovery:    20 * time.Millisecond,
			Poll:        time.Millisecond,
		},
	})
	if err == nil || !strings.Contains(err.Error(), "final load") {
		t.Fatalf("runScenarioG error = %v, want recovery load rejection", err)
	}
}

func TestClassifySearchErrorUsesStructuredReason(t *testing.T) {
	base := status.New(codes.FailedPrecondition, "safe message without stable code text")
	withDetails, err := base.WithDetails(&errdetails.ErrorInfo{
		Reason: "collection_not_ready",
		Domain: "goodkind.io/lm-semantic-search",
	})
	if err != nil {
		t.Fatalf("WithDetails: %v", err)
	}
	if got := classifySearchError(withDetails.Err()); got != collectionNotReadyCode {
		t.Fatalf("classifySearchError = %q, want %q", got, collectionNotReadyCode)
	}
}

func TestClassifySearchErrorDoesNotParseMessageText(t *testing.T) {
	err := status.Error(codes.FailedPrecondition, "collection_not_ready appears only in prose")
	if got := classifySearchError(err); got != codes.FailedPrecondition.String() {
		t.Fatalf("classifySearchError = %q, want transport code", got)
	}
}

func TestRunScenarioHTracksActiveDependencyInBothOrdersWithoutDeletingState(t *testing.T) {
	recorder, evidencePaths := scenarioTestRecorder(t)
	controller := newOverlapController()

	result, err := runScenarioH(context.Background(), scenarioHInput{
		SetFault:      controller.setFault,
		ClearFault:    controller.clearFault,
		SearchLMS:     controller.search,
		SearchClyde:   controller.search,
		SnapshotState: controller.snapshot,
		LMSStatus: func(context.Context) (daemonStatusObservation, error) {
			return daemonStatusObservation{Responding: true}, nil
		},
		ClydeStatus: func(context.Context) (clydeStatusObservation, error) {
			return clydeStatusObservation{PID: 4242, Responding: true}, nil
		},
		Recorder: recorder,
		Timeouts: scenarioHTimeouts{Failure: 100 * time.Millisecond, Recovery: 100 * time.Millisecond, Poll: time.Millisecond},
	})
	if err != nil {
		t.Fatalf("runScenarioH returned error: %v", err)
	}
	if len(result.Orders) != 2 {
		t.Fatalf("orders = %d, want 2", len(result.Orders))
	}
	if result.Orders[0].FirstCode != embedderBusyCode || result.Orders[0].RemainingCode != milvusUnavailableCode {
		t.Fatalf("embedding-first codes = %+v", result.Orders[0])
	}
	if result.Orders[1].FirstCode != milvusUnavailableCode || result.Orders[1].RemainingCode != embedderBusyCode {
		t.Fatalf("Milvus-first codes = %+v", result.Orders[1])
	}
	assertSingleScenarioRecord(t, evidencePaths.EventsJSONL, "scenario_h")
}

type overlapController struct {
	mutex  sync.Mutex
	faults map[dependencyFault]bool
	state  stateObservation
}

func newOverlapController() *overlapController {
	return &overlapController{
		faults: make(map[dependencyFault]bool),
		state: stateObservation{
			Collections: []collectionStateObservation{
				{Name: "code", RowCount: 1, Rows: []rowStateObservation{{Identity: "01.go:1:2", DenseVectorHash: "code-vector"}}},
				{Name: "conversations", RowCount: 1, Rows: []rowStateObservation{{Identity: "conversation:0", DenseVectorHash: "conversation-vector"}}},
			},
			Checkpoints: []string{"checkpoint-1"},
		},
	}
}

func (controller *overlapController) setFault(fault dependencyFault) {
	controller.mutex.Lock()
	controller.faults[fault] = true
	controller.mutex.Unlock()
}

func (controller *overlapController) clearFault(fault dependencyFault) {
	controller.mutex.Lock()
	delete(controller.faults, fault)
	controller.mutex.Unlock()
}

func (controller *overlapController) search(context.Context) (semanticSearchObservation, error) {
	controller.mutex.Lock()
	defer controller.mutex.Unlock()
	if controller.faults[dependencyEmbedding] {
		return semanticSearchObservation{Code: embedderBusyCode}, nil
	}
	if controller.faults[dependencyMilvus] {
		return semanticSearchObservation{Code: milvusUnavailableCode}, nil
	}
	return semanticSearchObservation{Succeeded: true, Source: "semantic", Matches: 2}, nil
}

func (controller *overlapController) snapshot(context.Context) (stateObservation, error) {
	controller.mutex.Lock()
	defer controller.mutex.Unlock()
	return stateObservation{
		Collections: cloneCollectionStateObservations(controller.state.Collections),
		Checkpoints: slices.Clone(controller.state.Checkpoints),
	}, nil
}

func TestEqualStateObservationRejectsDenseVectorMutationWithinStableRows(t *testing.T) {
	before := stateObservation{
		Collections: []collectionStateObservation{{
			Name:     "code",
			RowCount: 1,
			Rows:     []rowStateObservation{{Identity: "01.go:1:2", DenseVectorHash: "before"}},
		}},
		Checkpoints: []string{"checkpoint=hash"},
	}
	after := stateObservation{
		Collections: []collectionStateObservation{{
			Name:     "code",
			RowCount: 1,
			Rows:     []rowStateObservation{{Identity: "01.go:1:2", DenseVectorHash: "after"}},
		}},
		Checkpoints: []string{"checkpoint=hash"},
	}
	if equalStateObservation(before, after) {
		t.Fatal("equalStateObservation accepted a dense-vector mutation")
	}
}

//go:build offlinelive

package offlinelive

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestOfflineProfileEndToEnd(t *testing.T) {
	harness := newHarness(t)

	job := harness.indexFixture()
	requireCompleted(t, job)

	status := harness.indexStatus()
	if !status.GetSearchable() {
		t.Fatalf("offline index is not searchable:\n%s", status.GetDisplayText())
	}
	if status.GetCodebase().GetStatus() != indexedStatus {
		t.Fatalf(
			"offline index status = %q, want %q",
			status.GetCodebase().GetStatus(),
			indexedStatus,
		)
	}
	totalChunks := status.GetCodebase().GetLastSuccessfulRun().GetTotalChunks()
	if totalChunks <= searchResultLimit {
		t.Fatalf(
			"offline fixture indexed %d chunks, want more than search limit %d",
			totalChunks,
			searchResultLimit,
		)
	}
	if status.GetDependencyHealth().GetDegraded() {
		t.Fatalf(
			"offline dependency health is degraded (%s):\n%s",
			status.GetDependencyHealth().GetMode(),
			status.GetDisplayText(),
		)
	}
	if status.GetCollectionReadiness() != collectionReady {
		t.Fatalf(
			"offline collection readiness = %q, want %q",
			status.GetCollectionReadiness(),
			collectionReady,
		)
	}
	harness.assertOfflineRuntime(job, status)
	harness.assertNoExternalDials()

	searchResponse := harness.search(fixtureQuery, searchResultLimit)
	if int32(len(searchResponse.GetResults())) != searchResultLimit {
		t.Fatalf(
			"offline search returned %d results, want truncated top %d",
			len(searchResponse.GetResults()),
			searchResultLimit,
		)
	}
	if !containsTargetResult(
		searchResponse.GetResults(),
		targetRelativePath,
		targetFunctionName,
	) {
		t.Fatalf(
			"offline search did not return %s from %q in the top %d results:\n%s",
			targetFunctionName,
			targetRelativePath,
			searchResultLimit,
			searchResponse.GetDisplayText(),
		)
	}

	queryP50 := harness.measureQueryP50(
		fixtureQuery,
		searchResultLimit,
		queryMeasurementCount,
	)
	t.Logf("offline search query p50: %s", queryP50)
	if queryP50 >= queryLatencyBound {
		t.Fatalf(
			"offline search query p50 = %s, want under %s",
			queryP50,
			queryLatencyBound,
		)
	}

	graphResponse := harness.waitForGraphTrace(targetFunctionName)
	if !strings.Contains(graphResponse.GetResultJson(), callerFunctionName) {
		t.Fatalf(
			"offline graph trace for %q did not include caller %q:\n%s",
			targetFunctionName,
			callerFunctionName,
			graphResponse.GetResultJson(),
		)
	}
	harness.assertNoExternalDials()
}

type stubCommandOutput struct {
	pgrepOutput []byte
	pgrepError  error
	psOutput    []byte
	psError     error
}

func (runner stubCommandOutput) run(
	_ context.Context,
	name string,
	_ ...string,
) ([]byte, error) {
	switch name {
	case "pgrep":
		return runner.pgrepOutput, runner.pgrepError
	case "ps":
		return runner.psOutput, runner.psError
	default:
		return nil, errors.New("unexpected command")
	}
}

type stubExitError struct {
	exitCode int
}

func (failure stubExitError) Error() string {
	return "stub command failed"
}

func (failure stubExitError) ExitCode() int {
	return failure.exitCode
}

func TestProductionDaemonSnapshotCommandFailures(t *testing.T) {
	t.Run("pgrep no match is empty", func(t *testing.T) {
		processIDs, err := snapshotProductionDaemonPids(
			context.Background(),
			stubCommandOutput{
				pgrepError: stubExitError{exitCode: 1},
			}.run,
		)
		if err != nil {
			t.Fatalf("snapshot returned error for pgrep no-match: %v", err)
		}
		if len(processIDs) != 0 {
			t.Fatalf("snapshot returned %d pids, want none", len(processIDs))
		}
	})

	t.Run("pgrep execution failure is an error", func(t *testing.T) {
		_, err := snapshotProductionDaemonPids(
			context.Background(),
			stubCommandOutput{
				pgrepError: errors.New("pgrep unavailable"),
			}.run,
		)
		if err == nil {
			t.Fatal("snapshot returned no error for pgrep execution failure")
		}
	})

	t.Run("ps execution failure is an error", func(t *testing.T) {
		_, err := snapshotProductionDaemonPids(
			context.Background(),
			stubCommandOutput{
				pgrepOutput: []byte("123\n"),
				psError:     errors.New("ps unavailable"),
			}.run,
		)
		if err == nil {
			t.Fatal("snapshot returned no error for ps execution failure")
		}
	})
}

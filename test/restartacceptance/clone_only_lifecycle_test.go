//go:build restartacceptance

package restartacceptance

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestExecuteRestartAcceptanceUsesOnlyCloneLifecycle(t *testing.T) {
	events := make([]string, 0, len(acceptanceScenarioNames)+4)
	run := acceptanceRun{
		ID:    "20260812T010203Z-abcdef01",
		Paths: pathsForRun("/tmp/20260812T010203Z-abcdef01"),
	}
	operations := acceptanceLifecycleOperations{
		Prepare: func(context.Context) (acceptanceRun, error) {
			events = append(events, "prepare")
			return run, nil
		},
		RunCase: func(_ context.Context, _ acceptanceRun, name string) error {
			events = append(events, "case-"+name)
			return nil
		},
		ConfirmClone: func(context.Context, acceptanceRun) error {
			events = append(events, "clone-confirmation")
			return nil
		},
		Cleanup: func(context.Context, acceptanceRun) error {
			events = append(events, "cleanup")
			return nil
		},
		Finish: func(_ acceptanceRun, _ acceptanceResult) error {
			events = append(events, "finish")
			return nil
		},
	}

	if err := executeRestartAcceptance(context.Background(), operations); err != nil {
		t.Fatalf("execute restart acceptance: %v", err)
	}
	want := []string{"prepare"}
	for _, name := range acceptanceScenarioNames {
		want = append(want, "case-"+name)
	}
	want = append(want, "clone-confirmation", "cleanup", "finish")
	if !slices.Equal(events, want) {
		t.Fatalf("lifecycle events = %v, want %v", events, want)
	}
}

func TestValidateRestartAcceptanceConfirmationRequiresOnlyCloneOptIn(t *testing.T) {
	if err := validateRestartAcceptanceConfirmation(restartAcceptanceConfirmation); err != nil {
		t.Fatalf("validate isolated-clone confirmation: %v", err)
	}
	if err := validateRestartAcceptanceConfirmation("default"); err == nil {
		t.Fatal("wrong isolated-clone confirmation was accepted")
	}
}

func TestHarnessChecksCloneOnlyAfterComposeStartup(t *testing.T) {
	paths := pathsForRun(filepath.Join(t.TempDir(), "lms-restart-acceptance", "20260812T010203Z-abcdef01"))
	for _, path := range []string{
		paths.SourceEtcd,
		paths.SourceMilvus,
		paths.SourceMinIO,
		paths.SourceMinIODefault,
		paths.Artifacts,
	} {
		if err := os.MkdirAll(path, 0o700); err != nil {
			t.Fatalf("create harness path: %v", err)
		}
	}
	runner := &recordingRunner{}
	harness := configuredTestHarness(t, paths, runner)
	scenarioRan := false
	censusCalls := 0
	harness.readiness = func(context.Context) error {
		if len(runner.calls) != 1 || runner.calls[0][len(runner.calls[0])-1] != "--wait" {
			return fmt.Errorf("clone readiness ran before compose startup: %v", runner.calls)
		}
		return nil
	}
	harness.census = func(context.Context) (cloneMilvusCensus, error) {
		if len(runner.calls) != 1 {
			return cloneMilvusCensus{}, fmt.Errorf("clone census ran outside active compose: %v", runner.calls)
		}
		censusCalls++
		if censusCalls == 1 && scenarioRan {
			return cloneMilvusCensus{}, fmt.Errorf("baseline clone census ran after scenario")
		}
		if censusCalls == 2 && !scenarioRan {
			return cloneMilvusCensus{}, fmt.Errorf("final clone census ran before scenario")
		}
		return cloneMilvusCensus{
			Databases: []string{"default"},
			Collections: collectionCensus{
				{Database: "default", Collection: "restored"}: "hash",
			},
		}, nil
	}
	err := harness.withCompose(context.Background(), "a-clone", func(context.Context) error {
		scenarioRan = true
		return nil
	})
	if err != nil {
		t.Fatalf("run clone-only scenario: %v", err)
	}
	if censusCalls != 2 {
		t.Fatalf("clone census calls = %d, want 2", censusCalls)
	}
}
